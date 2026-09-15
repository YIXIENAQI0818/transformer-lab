"""FlashAttention 的核心技巧（tiling 分块 + online softmax）从零实现。

「为什么换」：标准 attention 的显存是 O(T²)。看 model.py 里 naive 的写法——

    att = (q @ k.transpose(-2, -1)) * scale     # (B, n_head, T, T)  第一次物化 T×T
    att = softmax(att, dim=-1)                   # (B, n_head, T, T)  第二次物化 T×T
    y = att @ v                                 # (B, n_head, T, d)  结果只有 T×d

    中间那个 (B, n_head, T, T) 的 score 矩阵（softmax 后还要再存一份概率矩阵）被完整地
    写进显存，T 一大（如 2048）就是 4M 元素、GB 级显存。但注意：**最终输出 O 只有 T×d 大小**，
    我们根本不需要把中间那个 T×T 矩阵完整存下来——只是标准算法「顺路」把它物化了。

FlashAttention（Dao et al. 2022）的核心洞察分两层：

1. **数值层 —— online softmax**：softmax 的分母 sum(exp) 可以在不知道全局最大值的情况下
   增量维护。经典数值稳定 softmax 要两遍（先找全局 max 再算 sum），online softmax 维护
   running max + running sum，每来一块就按「新 max 尺度」修正旧的 sum，一遍就能算出
   正确的归一化因子。这让「分块算 softmax 再合并」在数学上可行。

2. **IO 层 —— tiling（分块）**：把 K/V 按块切，每块算局部 attention，用 online softmax 的
    rescale 技巧合并进累加器，算完立刻丢弃块内的 T×T 小矩阵。这样显存峰值从 O(T²) 降到
    O(T)（只同时存在一个 block 的小矩阵 + 一个 T×d 的累加器）。

注意一个关键事实：**FlashAttention 不是近似算法，它是精确的 attention**。结果和 naive 版
在浮点误差内完全一致——它只是改变了「计算顺序和中间张量的生命周期」，没有改变数学。所以
替换前后 loss 应该一模一样，收益全在「显存 / 速度」，而不是「精度」。

一个容易误会的点：本文件手写的 tiling 版在 PyTorch 里**反而更慢**（Python 双重 for 循环的
开销远超省下的显存）。真正的加速来自 CUDA 层面的 fused kernel（把整个 block 的 attention
在 SRAM 里算完，避免把 T×T 矩阵写回 HBM 再读回）。所以 model.py 里 flash 分支实际用
`F.scaled_dot_product_attention`（PyTorch 内置 fused 实现），本文件只用于看清内部算法——
与 rmsnorm.py 的关系一致：内置跑生产，手写看原理。
"""
import math

import torch
import torch.nn.functional as F


def online_softmax(x):
    """一维 online softmax：用 running max + running sum 一遍算出 softmax，无需先找全局 max。

    x: (..., T)，沿最后一维做 softmax。

    经典数值稳定 softmax 的两步：
        m = x.max();  e = exp(x - m);  softmax = e / e.sum()
    要先遍历一遍找 max，再遍历一遍算 sum。online 版把两步合成一遍：逐列扫过，维护
    (m, l) = (running max, running sum)，每遇到更大的值就把旧的 l 按 exp(m_old - m_new)
    「降级」到新 max 的尺度。扫完所有列后 (m, l) 恰好等于全局 max 和全局 sum(exp)。
    """
    m = torch.full(x.shape[:-1] + (1,), float("-inf"), device=x.device, dtype=x.dtype)
    l = torch.zeros_like(m)
    for j in range(x.shape[-1]):
        xj = x[..., j:j + 1]
        m_new = torch.maximum(m, xj)
        # 旧 sum 乘 exp(m_old - m_new) 降到新尺度，再加新项的 exp
        l = l * torch.exp(m - m_new) + torch.exp(xj - m_new)
        m = m_new
    return torch.exp(x - m) / l


def naive_attention(q, k, v, causal=False):
    """参照实现：物化完整 T×T score 矩阵的标准 attention（即 model.py 的 naive 分支）。"""
    B, H, T, d = q.shape
    scale = 1.0 / math.sqrt(d)
    att = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)
    if causal:
        mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))   # 上三角（未来）置 -inf
    att = F.softmax(att, dim=-1)
    return att @ v                                   # (B, H, T, d)


def flash_attention(q, k, v, block_size, causal=False):
    """tiling + online softmax 的 FlashAttention（教学实现，纯 PyTorch 循环）。

    q, k, v: (B, H, T, d)，返回 O: (B, H, T, d)，与 naive_attention 数值等价。

    算法（对每个 query 块，逐 key 块累加）：
        m_i, l_i, acc = -inf, 0, 0                 # running max / running sum / 加权和累加器
        for 每个 key 块 j（causal 时 j <= i）:
            S   = Q_i @ K_j^T / sqrt(d)            # 块内 score（只物化 block×block）
            m_ij = rowmax(S)
            m_new = max(m_i, m_ij)                 # 更新 running max
            P   = exp(S - m_new)                   # 块内概率（新 max 尺度）
            l_new = l_i * exp(m_i - m_new) + rowsum(P)   # 旧 sum 降级 + 新 sum
            acc  = acc * exp(m_i - m_new) + P @ V_j      # 旧加权和降级 + 新加权和
            m_i, l_i = m_new, l_new
        O_i = acc / l_i

    causal 的处理：query 块 i 只遍历 key 块 j <= i；对角块（i == j）内再套一个下三角 mask
    （query 位置只能看到 <= 自己的 key 位置）。
    """
    B, H, T, d = q.shape
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros_like(q)
    n_blocks = (T + block_size - 1) // block_size

    for i in range(n_blocks):
        q0, q1 = i * block_size, min((i + 1) * block_size, T)
        Qi = q[:, :, q0:q1]                                     # (B, H, Bq, d)
        m_i = torch.full((B, H, q1 - q0, 1), float("-inf"), device=q.device, dtype=q.dtype)
        l_i = torch.zeros(B, H, q1 - q0, 1, device=q.device, dtype=q.dtype)
        acc = torch.zeros_like(Qi)                              # 加权和累加器 (B, H, Bq, d)

        j_last = i + 1 if causal else n_blocks                  # causal：只看 j <= i
        for j in range(j_last):
            k0, k1 = j * block_size, min((j + 1) * block_size, T)
            Kj = k[:, :, k0:k1]
            Vj = v[:, :, k0:k1]
            S = Qi @ Kj.transpose(-2, -1) * scale               # (B, H, Bq, Bk)
            if causal and i == j:
                # 对角块内下三角 mask：query 位置 q_idx 只能看到 key 位置 <= q_idx
                q_idx = torch.arange(q0, q1, device=q.device).view(-1, 1)
                k_idx = torch.arange(k0, k1, device=q.device).view(1, -1)
                S = S.masked_fill(q_idx < k_idx, float("-inf"))

            m_ij = S.max(dim=-1, keepdim=True).values          # 块内 max
            m_new = torch.maximum(m_i, m_ij)                   # 更新 running max
            P = torch.exp(S - m_new)                            # 块内概率（新尺度）
            l_new = l_i * torch.exp(m_i - m_new) + P.sum(dim=-1, keepdim=True)
            acc = acc * torch.exp(m_i - m_new) + P @ Vj
            m_i, l_i = m_new, l_new

        O[:, :, q0:q1] = acc / l_i                              # 归一化写回
    return O


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, T, d = 1, 2, 32, 32   # 小规模，便于肉眼核对数值

    # 1) online softmax vs 标准 softmax：数值等价（running max/sum 合并的正确性）
    x = torch.randn(3, 16)
    ref = F.softmax(x, dim=-1)
    onl = online_softmax(x)
    print("1) online softmax 与标准 softmax 数值等价:")
    print("   最大绝对误差 = {:.2e}  （期望 ~1e-7）".format((onl - ref).abs().max().item()))
    print("   每行和 = {:.6f}（期望 1）".format(onl.sum(-1).mean().item()))

    # 2) flash_attention vs naive_attention：数值等价（tiling + online softmax 合并的正确性）
    #    不同 block_size、causal / 非 causal 都验证
    print("\n2) flash_attention 与 naive_attention 数值等价（不同 block_size）:")
    for causal in [False, True]:
        q = torch.randn(B, H, T, d)
        k = torch.randn(B, H, T, d)
        v = torch.randn(B, H, T, d)
        ref = naive_attention(q, k, v, causal=causal)
        tag = "causal" if causal else "非 causal"
        for bs in [8, 16, 32]:
            out = flash_attention(q, k, v, block_size=bs, causal=causal)
            err = (out - ref).abs().max().item()
            print("   [{:6s}] block_size={:2d}: 最大绝对误差 = {:.2e}".format(tag, bs, err))

    # 3) 显存元素数对比：naive 物化 O(T²)，flash 峰值 O(block² + T·d)
    T_big, bs, d_big = 1024, 128, 64
    naive_elems = 2 * T_big * T_big                    # score + 概率矩阵，各一个 T×T
    flash_elems = 2 * bs * bs + T_big * d_big          # 块内 S + P，加 T×d 累加器
    print("\n3) 显存元素数对比（T={}, block={}, d={}，单头单 batch）:".format(T_big, bs, d_big))
    print("   naive 峰值 = 2·T² = {} 元素".format(naive_elems))
    print("   flash 峰值 = 2·block² + T·d = {} 元素".format(flash_elems))
    print("   flash 峰值是 naive 的 {:.1%}（T 越大差距越悬殊，因为 naive 是 O(T²)、flash 是 O(T)）".format(
        flash_elems / naive_elems))

    # 4) 可微性：flash_attention 也能反向传播（真实训练需要）
    q = torch.randn(B, H, T, d, requires_grad=True)
    k = torch.randn(B, H, T, d, requires_grad=True)
    v = torch.randn(B, H, T, d, requires_grad=True)
    flash_attention(q, k, v, block_size=8, causal=True).sum().backward()
    print("\n4) 可微性: 反向传播 OK，q/k/v 梯度是否非空 = {}/{}/{}".format(
        q.grad is not None, k.grad is not None, v.grad is not None))
