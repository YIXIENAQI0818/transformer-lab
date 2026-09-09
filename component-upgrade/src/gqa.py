"""GQA（Grouped-Query Attention，分组查询注意力）的 K/V 头共享从零实现。

「为什么换」：MHA 里 Q/K/V 的头数一样多（各 n_head 个），一一配对。但两者角色不对称——
- Q 头是「问题」，每个头关注不同方面（语法/指代/局部…），**多样性靠 Q 投影提供，不能共享**；
- K/V 头是「资料」，都编码同一个 token 的内容，**头之间有冗余，可以共享**。

GQA 让多个 Q 头共享同一个 K/V 头（MQA 是共享到极致的特例：只剩 1 个 K/V 头）。动机是**省 KV cache**：
decode 时每个 token 要存它的 K 和 V 供后续查询，KV cache 大小正比于 K/V 头数。把 K/V 头从
n_head 砍到 n_kv_head，KV cache 缩小 n_head/n_kv_head 倍（省显存 + decode 更快）。GQA 论文
（Ainslie et al. 2023）验证：适度共享几乎不掉效果。

三者是同一个结构里 n_kv_head 取不同值：
    n_kv_head = n_head → MHA（不共享）
    n_kv_head < n_head → GQA（分组共享，如 8 头分 2 组）
    n_kv_head = 1      → MQA（全共享）

实现核心只有两处改动（见 model.py 的 CausalSelfAttention）：
    1. c_attn 的 K/V 投影输出维度从 n_embd 缩到 n_kv_head * head_size（Q 仍是全量 n_embd）；
    2. 把 K/V 从 n_kv_head 个头「广播」到 n_head 个头，再和 Q 做 attention。

本文件手写 repeat_kv（头广播）用于看清「共享」的本质：让相邻的 n_rep 个 Q 头指向同一个 K/V 头，
用 expand + reshape 实现——**不复制数据，只是让多个头共享同一块内存**（等价于 repeat_interleave，
但注意是 repeat_interleave 而非 repeat：GQA 要的是 [K0,K1]->[K0,K0,K1,K1] 的连续分组共享，
repeat 会产生错误的 [K0,K1,K0,K1]）。
"""
import torch


def repeat_kv(x, n_rep):
    """把 K/V 从 n_kv_head 个头广播到 n_head 个头（n_rep = n_head // n_kv_head）。

    x: (B, n_kv_head, T, head_size) -> (B, n_kv_head * n_rep, T, head_size)

    [K0, K1] --n_rep=2--> [K0, K0, K1, K1]（Q0/Q1 共享 K0，Q2/Q3 共享 K1）。
    expand 不复制数据，只是让相邻的 n_rep 个位置指向同一个头——这正是「共享」的本质。
    """
    B, n_kv_head, T, head_size = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :]                           # (B, n_kv_head, 1, T, head_size)
    x = x.expand(B, n_kv_head, n_rep, T, head_size)   # (B, n_kv_head, n_rep, T, head_size)
    return x.reshape(B, n_kv_head * n_rep, T, head_size)


if __name__ == "__main__":
    torch.manual_seed(0)

    n_head, head_size = 4, 32   # 对齐 model.py：n_head=4, head_size=32
    n_embd = n_head * head_size  # 128

    # 1) 头共享正确性：广播后，共享组内的 K 完全相同，不同组之间不同
    B, T, n_kv_head = 1, 4, 2
    k = torch.randn(B, n_kv_head, T, head_size)
    k_broad = repeat_kv(k, n_rep=2)  # (B, 4, T, head_size)
    same_01 = torch.allclose(k_broad[:, 0], k_broad[:, 1])
    same_23 = torch.allclose(k_broad[:, 2], k_broad[:, 3])
    diff_02 = not torch.allclose(k_broad[:, 0], k_broad[:, 2])
    print("1) 头共享正确性（n_kv_head=2 -> 广播成 4 头）:")
    print("   头0 与 头1 相同 = {}  （共享 K0）".format(same_01))
    print("   头2 与 头3 相同 = {}  （共享 K1）".format(same_23))
    print("   头0 与 头2 不同 = {}  （不同组）".format(diff_02))

    # 2) 参数量对比：c_attn 的 K/V 投影随 n_kv_head 缩小
    print("\n2) c_attn 参数量（n_embd={}）:".format(n_embd))
    for n_kv in [4, 2, 1]:
        out = n_embd + 2 * n_kv * head_size          # Q 全量 + K/V 各 n_kv 头
        n_params = n_embd * out + out                # weight + bias
        print("   n_kv_head={}: 输出 {} 维，c_attn {} 参数".format(n_kv, out, n_params))

    # 3) KV cache 大小：每 token 存 K + V，元素数 = 2 * n_kv_head * head_size
    print("\n3) KV cache 大小（每 token 的 K/V 元素数）:")
    for n_kv in [4, 2, 1]:
        kv_per_token = 2 * n_kv * head_size
        print("   n_kv_head={}: {} 元素/token  （相对 MHA 的 {:.0%}）".format(
            n_kv, kv_per_token, kv_per_token / (2 * n_head * head_size)))

    # 4) 退化等价：n_kv_head = n_head 时 n_rep=1，repeat_kv 是恒等（即 MHA）
    k = torch.randn(B, n_head, T, head_size)   # 已经是 n_head 个头
    out = repeat_kv(k, n_rep=1)
    print("\n4) 退化等价: n_rep=1 时 repeat_kv 是恒等 = {}  （n_kv_head=n_head 即 MHA）".format(
        torch.allclose(k, out)))
