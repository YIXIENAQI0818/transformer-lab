"""KV cache（键值缓存）的核心机制从零实现。

「为什么做」：自回归生成是一步吐一个 token。最笨的走法是每步都对「整条已生成序列」重算一遍
attention——第 t 步要算 t+1 个 query 对 t+1 个 key 的点积，共 (t+1)² 次，生成 T 个 token 总共
Σ(t+1)² ≈ T³/3 次（**三次方**增长）。但观察一下：前一步已经算过位置 0..t-1 的 K/V 了，下一步
唯一「新」的是第 t 个 token 的 Q/K/V，其余全是重复劳动。

KV cache 的思路：**把每层的 K/V 缓存起来，下一步只算新 token 的 Q/K/V，Q 只 attend 到
[缓存的过去 K/V + 自己的 K/V]**。这样第 t 步只算 (t+1) 次点积，总共 Σ(t+1) ≈ T²/2 次
（**二次方**增长，省掉一个 T 的维度）。KV cache 是「换计算方式」不是「换模型」：结果和
完整重算在浮点误差内完全一致，收益全在速度与显存（过去 K/V 不用重算、不用重存中间 score）。

一次完整的自回归生成分两个阶段：

1. **prefill（预填充）**：prompt 是「一段话」而非「一个词」，一次并行前向把整段 prompt 的
   K/V 都算出来并塞进 cache，同时拿到每个位置（含最后一个）的 logits。这一步本身就是 O(T²)
   的完整 attention——但只做一次，摊到后面每个 decode token 上就便宜了。
2. **decode（逐 token 生成）**：之后每次只喂一个新 token (B,1)，算它的 Q/K/V，K/V 追加进
   cache，Q attend 到全部 cache 取最后位置的 logits 采样下一个 token。每步 O(T) 而非 O(T²)。

GQA 下 cache 的形状是关键点：cache 按 **n_kv_head** 存（不是 n_head），shape 是
(B, n_kv_head, L, head_size)。广播到 n_head 在「读完 cache 之后」才做（见 gqa.py 的 repeat_kv）——
这正是 GQA 省 KV cache 显存的原因：缓存是省显存大户，按共享后的头数存就是直接省。

还有一个容易踩的坑（KV cache + RoPE）：decode 时新 token 的 Q/K 必须按**绝对位置**旋转
（当前 cache 长度 = 绝对位置），不能按「局部下标 0」旋转——否则每个新 token 都被当成位置 0，
相对位置信息全错。model.py 里 `pos_start = cache[0].size(2)` 就是为这个服务的，本文件第 2 个
验证会显式演示两种旋转方式的差别。
"""
import math

import torch
import torch.nn.functional as F

from gqa import repeat_kv
from rope import apply_rotary_emb, precompute_rope_cache


class KVCache:
    """一个 attention 层的 K/V 缓存，存两路张量 (B, n_kv_head, L, head_size)。

    按 n_kv_head 粒度存（不是 n_head）——GQA 省显存的核心。append 只做 cat，不复制。
    """

    def __init__(self):
        self.k = None  # (B, n_kv_head, L, head_size)
        self.v = None

    @property
    def length(self):
        return 0 if self.k is None else self.k.size(2)

    def append(self, k, v):
        """追加一个新 token 的 k/v（各 (B, n_kv_head, 1, head_size)），返回更新后的 cache。"""
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self


def decode_step(q, k, v, cache, scale):
    """增量 decode 的一步：把新 token 的 k/v 追加进 cache，q 只 attend 到 [过去 + 自己]。

    q: (B, n_head, 1, hs)       新 token 的查询（1 个位置）
    k: (B, n_kv_head, 1, hs)    新 token 的 key
    v: (B, n_kv_head, 1, hs)    新 token 的 value
    cache: (k_cache, v_cache)   (B, n_kv_head, L, hs)，首次调用传 (None, None)

    返回 (y, new_cache)：y 是 (B, n_head, 1, hs)，new_cache 是追加后的 (k, v)（n_kv_head 粒度）。

    两个关键点：
      - 无需因果 mask：q 是最新位置，attend 到全部 cache 天然满足因果（看不到未来，因为
        cache 里只有过去 + 自己）。
      - cache 在 n_kv_head 粒度追加、广播到 n_head 在 attention 前才做（省显存）。
    """
    k_cache, v_cache = cache
    k_full = k if k_cache is None else torch.cat([k_cache, k], dim=2)   # (B, n_kv_head, L+1, hs)
    v_full = v if v_cache is None else torch.cat([v_cache, v], dim=2)
    new_cache = (k_full, v_full)   # 存 n_kv_head 粒度

    n_rep = q.size(1) // k_full.size(1)
    k_full = repeat_kv(k_full, n_rep)   # 广播到 n_head
    v_full = repeat_kv(v_full, n_rep)

    att = (q @ k_full.transpose(-2, -1)) * scale   # (B, n_head, 1, L+1)
    att = F.softmax(att, dim=-1)
    y = att @ v_full                               # (B, n_head, 1, hs)
    return y, new_cache


def causal_attention(q, k, v, scale):
    """参照实现：完整序列的自注意力（下三角因果 mask），物化 (B,n_head,T,T) 的 score。"""
    B, H, T, hs = q.shape
    att = (q @ k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    att = att.masked_fill(mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v   # (B, n_head, T, hs)


if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) 缓存正确性：decode（逐 token 复用 cache）== full attention（一次算全序列）
    #    这是 KV cache 最核心的保证：缓存不改数学，结果应与完整重算逐 token 一致。
    B, n_head, n_kv_head, T, hs = 1, 4, 2, 16, 32
    scale = 1.0 / math.sqrt(hs)
    q = torch.randn(B, n_head, T, hs)
    k = torch.randn(B, n_kv_head, T, hs)
    v = torch.randn(B, n_kv_head, T, hs)

    n_rep = n_head // n_kv_head
    ref = causal_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), scale)  # (B,n_head,T,hs)

    cache = (None, None)
    out = []
    for t in range(T):
        yt, cache = decode_step(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1], cache, scale)
        out.append(yt)
    out = torch.cat(out, dim=2)   # (B, n_head, T, hs)
    err = (out - ref).abs().max().item()
    print("1) 缓存正确性: decode(复用 cache) == full attention(一次算全):")
    print("   逐 token 最大绝对误差 = {:.2e}  （期望 ~1e-7）".format(err))

    # 2) RoPE 绝对位置：decode 的新 token 必须按「绝对位置 t」旋转，误按「局部位置 0」会错。
    #    这是 KV cache + RoPE 最经典的 bug——缓存的是「按绝对位置旋转好的 K」，新 Q 若也按
    #    绝对位置旋转，q_t·k_j 才只依赖相对位置 (t-j)；若新 Q 恒按位置 0 旋转，相对位置全错。
    d, T2 = 16, 8
    cos, sin = precompute_rope_cache(d, T2)
    qb = torch.randn(T2, d)   # 未旋转的 q/k（等会儿按绝对位置旋转）
    kb = torch.randn(T2, d)
    vb = torch.randn(T2, d)   # V 不旋转（RoPE 只作用于 q/k）

    def rot(x, m):
        return apply_rotary_emb(x, cos[m], sin[m])

    Q = torch.stack([rot(qb[i], i) for i in range(T2)])[None, None]  # (1,1,T2,d)
    K = torch.stack([rot(kb[i], i) for i in range(T2)])[None, None]
    V = vb[None, None]
    ref2 = causal_attention(Q, K, V, 1.0 / math.sqrt(d))[0, 0]      # (T2, d)

    cache_k = None
    cache_v = None
    out_correct, out_bug = [], []
    sc = 1.0 / math.sqrt(d)
    for t in range(T2):
        qt_correct = rot(qb[t], t)[None, None, None]   # (1,1,1,d)：绝对位置 t
        qt_bug = rot(qb[t], 0)[None, None, None]       # 错误：局部位置 0
        kt = rot(kb[t], t)[None, None, None]           # 新 K 按绝对位置 t（会被正确缓存）
        vt = vb[t][None, None, None]
        cache_k = kt if cache_k is None else torch.cat([cache_k, kt], dim=2)
        cache_v = vt if cache_v is None else torch.cat([cache_v, vt], dim=2)
        out_correct.append((F.softmax((qt_correct @ cache_k.transpose(-2, -1)) * sc, dim=-1) @ cache_v)[0, 0, 0])
        out_bug.append((F.softmax((qt_bug @ cache_k.transpose(-2, -1)) * sc, dim=-1) @ cache_v)[0, 0, 0])
    err_correct = (torch.stack(out_correct) - ref2).abs().max().item()
    err_bug = (torch.stack(out_bug) - ref2).abs().max().item()
    print("\n2) RoPE 绝对位置: decode 的新 token 必须按绝对位置 t 旋转（经典 bug）:")
    print("   按绝对位置 t 旋转  ：误差 = {:.2e}  （期望 ~1e-7，与 full 一致）".format(err_correct))
    print("   误按局部位置 0 旋转：误差 = {:.2e}  （明显偏离，相对位置信息全错）".format(err_bug))

    # 3) GQA cache shape：cache 按 n_kv_head 存（不是 n_head），这是省显存的来源。
    #    承接 gqa.py 的「每 token K/V 元素数」伏笔，落到具体张量 shape。
    print("\n3) GQA cache shape（每层，L 个 token 的 K/V 缓存张量）:")
    for n_kv in [4, 2, 1]:
        kv_per_token = 2 * n_kv * hs
        print("   n_kv_head={}: cache 张量 (B, n_kv_head={}, L, head_size={})，每 token {} 元素（相对 MHA {:.0%}）".format(
            n_kv, n_kv, hs, kv_per_token, kv_per_token / (2 * n_head * hs)))

    # 4) 复杂度：naive 重算 O(T³) vs cache O(T²)——KV cache 省掉的正是那一个 T 的维度。
    print("\n4) 复杂度对比（自回归生成 T 个 token，单头，单位 = 点积次数）:")
    for T_gen in [16, 64, 256]:
        naive = sum((t + 1) ** 2 for t in range(T_gen))   # 第 t 步对 (t+1) 个 query × (t+1) 个 key
        cached = sum(t + 1 for t in range(T_gen))          # 第 t 步 1 个 query × (t+1) 个 key
        print("   T={:4d}: naive 重算 {:>10d} 次，cache {:>8d} 次（cache 是 naive 的 {:.1%}）".format(
            T_gen, naive, cached, cached / naive))
