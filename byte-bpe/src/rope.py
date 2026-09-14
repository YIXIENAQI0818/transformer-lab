"""RoPE（Rotary Position Embedding，旋转位置编码）从零实现。

「为什么换」：GPT-2 用 learned positional embedding（wpe），有两条硬伤——
  1. 要学，且参数锁死在训练长度：wpe 是 (block_size, n_embd) 的可学习表，
     训练时只见过 [0, block_size) 的位置，超长序列直接越界（长度外推差）。
  2. 编码的是「绝对位置」，但 attention 里真正起作用的是「相对位置」 q_i·k_j 的差 (i-j)。

RoPE（原论文《RoFormer: Enhanced Transformer with Rotational Position Embedding》）
的思路：不学位置，而是把每个 token 的 q / k 向量按「绝对位置 m」在 2D 子空间里
旋转一个与 m 成正比的角度 m·θ_i。因为旋转是正交变换，两个旋转后的向量点积只差
相对位置：

    q_m · k_n = (R(m) q) · (R(n) k) = qᵀ R(m)ᵀ R(n) k = qᵀ R(n-m) k

即 attention score 只依赖 (n-m)，而绝对位置信息（旋转角 m·θ）仍被保留在向量里。

数学细节：把 head_dim 拆成 head_dim/2 对维度 (2i, 2i+1)，第 i 对按角度 m·θ_i 做 2D 旋转，
θ_i = base^(-2i/head_dim)，base=10000（高频维度转得快，低频维度转得慢）。

    [x0']   [cos(mθ)  -sin(mθ)] [x0]
    [x1'] = [sin(mθ)   cos(mθ)] [x1]

代码用恒等式实现：x' = x·cos + rotate_pairs(x)·sin，其中 rotate_pairs(x)=(-x1, x0)。
"""
import torch


def precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device=None, dtype=None):
    """预计算 RoPE 的 cos / sin 表，shape 均为 (max_seq_len, head_dim)。

    - 第 i 对维度的频率 θ_i = base^(-2i/head_dim)（几何级数，从 1 衰减到 ~1/base）
    - 位置 m 的旋转角 = m·θ_i，构成 (max_seq_len, head_dim/2) 的角度表
    - repeat_interleave 把每个角度复制 2 次展开到 head_dim（对应 (2i, 2i+1) 两个分量）
    """
    i = torch.arange(0, head_dim // 2, device=device, dtype=dtype)
    theta = base ** (-2 * i / head_dim)          # (head_dim/2,)

    m = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(m, theta)               # (max_seq_len, head_dim/2)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)


def rotate_pairs(x):
    """把相邻分量 pair (x0, x1) 映射为 (-x1, x0)，即绕原点转 90°。

    有了它，任意角度 θ 的旋转就能写成线性组合 x·cosθ + rotate_pairs(x)·sinθ。
    """
    x0 = x[..., 0::2]                            # (..., d/2)
    x1 = x[..., 1::2]
    rotated = torch.stack((-x1, x0), dim=-1)     # (..., d/2, 2) = [-x1, x0]
    return rotated.flatten(-2)                   # (..., d)


def apply_rotary_emb(x, cos, sin):
    """对 x 施加旋转位置编码。x: (..., T, head_dim)，cos/sin: (T, head_dim)。

    2D 旋转矩阵展开后：x' = x·cos + rotate_pairs(x)·sin，即
    [x0, x1] -> [x0·cosθ - x1·sinθ, x0·sinθ + x1·cosθ]。
    """
    return x * cos + rotate_pairs(x) * sin


if __name__ == "__main__":
    torch.manual_seed(0)

    d = 8       # head_dim
    L = 16      # max_seq_len（验证用）

    cos, sin = precompute_rope_cache(d, L)

    def rot_at(x, m):
        """把向量 x 旋转到绝对位置 m。x: (d,)，返回 (d,)。"""
        return apply_rotary_emb(x, cos[m], sin[m])

    # 1) 旋转保范数：旋转是正交变换，不改变向量长度
    q = torch.randn(d)
    print("1) 旋转保范数: ||R(5)q|| = {:.6f}  vs  ||q|| = {:.6f}".format(
        rot_at(q, 5).norm().item(), q.norm().item()))

    # 2) 相对位置性质：q_m·k_n 只依赖 (n-m)，与绝对位置无关
    q = torch.randn(d)
    k = torch.randn(d)

    def dot(m, n):
        return (rot_at(q, m) * rot_at(k, n)).sum().item()

    print("\n2) 相对位置性质（点积只依赖 n-m）：")
    print("   dot(5,2) = {:.6f}   dot(10,7) = {:.6f}   (n-m 都是 -3)".format(dot(5, 2), dot(10, 7)))
    print("   dot(5,8) = {:.6f}   dot(10,13) = {:.6f}  (n-m 都是 +3)".format(dot(5, 8), dot(10, 13)))

    # 3) 可微性：旋转参与计算图，能回传梯度
    x = torch.randn(4, 2, 3, d, requires_grad=True)   # (B, n_head, T, d)
    y = apply_rotary_emb(x, cos[:3], sin[:3])
    y.sum().backward()
    print("\n3) 可微性: 反向传播 OK，x.grad 是否非空 = {}".format(x.grad is not None))
