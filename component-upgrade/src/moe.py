"""MoE（Mixture of Experts，混合专家）从零实现。

「为什么换」：前面 6 回合升级的都是「让一个组件更强」，但 FFN 有个根本矛盾——**表达力和计算量
绑死在一起**。一个 FFN 想更强，就得加宽（参数更多），而加宽后每个 token 都要过整个 FFN，计算量
同步涨。能不能「参数总量涨、但每个 token 只花一小部分算力」？

MoE 的思路：**把一个大 FFN 拆成 n 个小 FFN（专家），每个 token 只激活其中 top-k 个**。
- 总参数量 = n × 专家参数（容量放大 n 倍）；
- 每 token 激活 = top-k 个专家（计算量只放大 k 倍）。

二者解耦了：想涨容量就加专家数 n，计算量却只跟 k 挂钩。这是 Mixtral 8x7B、DeepSeek 等「小计算量
换大容量」的核心手段（47B 总参数，每 token 只激活 ~13B）。

结构三要素：
    1. Router（路由）：一个线性层 d -> n_expert，输出每个专家的「得分」；
    2. top-k 选择：对得分做 softmax，取最大的 k 个专家，权重重归一化（被选中的 k 个概率和为 1）；
    3. 专家（Expert）：每个专家是一个 FFN，现代模型（Mixtral/DeepSeek）都用 SwiGLU（复用 swiglu.py）。

MoE(x) = Σ_{e ∈ top-k} w_e(x) · Expert_e(x)，w_e 是重归一化后的路由权重。

关键难点 —— 负载均衡（load balancing）：
    router 是可学习的，训练时它会「偷懒」：发现某几个专家好用，就把所有 token 都路由过去，其余
    专家闲置。结果是「名义上 n 个专家，实际只用 1 个」，退化成稠密 FFN 还更慢。解法是加一个
    auxiliary loss 惩罚路由分布不均（Mixtral 式）：鼓励每个专家被选中的 token 数、被路由的概率
    都均匀。本文件 load_balancing_loss 就是它，compare_moe.py 会演示「无 aux 坍缩 vs 有 aux 均衡」。

注：本文件手写的是 MoE 的「逻辑」；真正工业实现还有 expert parallelism（专家分布到多卡）等，
不在此列。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from swiglu import SwiGLU


def load_balancing_loss(logits, topk_idx, n_expert):
    """Mixtral 式负载均衡损失：鼓励每个专家「被选中的 token 数」「被路由的概率」都均匀。

    logits:   (N, n_expert)  展平后的 router logits
    topk_idx: (N, k)         每个 token 被选中的 k 个专家编号
    返回标量，训练时乘一个系数加到主 loss 上。

    两个分量：
      f_e = 每个专家被选中的（token×slot）占比，Σ_e f_e = k（每个 token 有 k 个 slot）
      P_e = 每个专家的平均路由概率（softmax 后）
    loss = n_expert · Σ_e (f_e · P_e)。路由坍缩到单一专家时 f/P 都集中、loss 大；均匀时最小。
    乘 n_expert 是 Mixtral 惯例，把量级归一到 O(1)（均匀时 loss ≈ k，坍缩时 ≈ n_expert·k）。
    """
    N = logits.size(0)
    probs = F.softmax(logits, dim=-1)                # (N, n_expert)
    one_hot = F.one_hot(topk_idx, n_expert).float()  # (N, k, n_expert)
    f_e = one_hot.sum(dim=(0, 1)) / N                # (n_expert,) 每个专家被选中的占比
    P_e = probs.mean(dim=0)                          # (n_expert,) 每个专家平均路由概率
    return n_expert * (f_e * P_e).sum()


class MoE(nn.Module):
    """混合专家 FFN：router 选 top-k 个 SwiGLU 专家，加权求和。

    MoE(x) = Σ_{e ∈ top-k} w_e · SwiGLU_e(x)
      - router：d -> n_expert 的线性层，输出每个专家的得分（无 bias，路由不靠偏置兜底）
      - experts：n_expert 个 SwiGLU（复用 swiglu.py，hidden_dim 默认 8/3·d）
      - top_k  ：每个 token 激活的专家数（k=1 单专家；k=n_expert 退化为稠密加权平均）

    实现按「top-k 槽位 × 专家」两重循环：对每个槽位 i、每个专家 e，找出「第 i 槽选了专家 e」
    的那些 token，让专家 e 算它们的输出、按槽位权重累加。稀疏激活动作就藏在 `sel.any()` 里——
    没被任何 token 选中的专家根本不算（`self.experts[e](x[sel])` 不被调用），这是 MoE 省算力的来源。
    """

    def __init__(self, dim, n_expert=4, top_k=2, hidden_dim=None, bias=True):
        super().__init__()
        assert 1 <= top_k <= n_expert, f"top_k={top_k} 必须在 [1, n_expert={n_expert}] 内"
        if hidden_dim is None:
            hidden_dim = (8 * dim) // 3  # 与 SwiGLU 的 8/3·d 对齐
        self.n_expert = n_expert
        self.top_k = top_k
        self.router = nn.Linear(dim, n_expert, bias=False)  # d -> n_expert 个得分
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim, bias) for _ in range(n_expert)])
        # 上次 forward 的路由结果（展平），供 load_balancing_loss 使用（训练时读取）
        self._router_logits = None  # (N, n_expert)
        self._topk_idx = None       # (N, top_k)

    def forward(self, x):
        B, T, d = x.shape
        logits = self.router(x)                            # (B, T, n_expert)
        probs = F.softmax(logits, dim=-1)
        topk_weights, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # 各 (B, T, k)
        # 重归一化：被选中的 k 个概率之和为 1（softmax 后取 top-k 的和 < 1，须重新归一）
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 记录展平后的路由信息，供 load balancing 的 aux loss 用
        self._router_logits = logits.reshape(-1, self.n_expert)
        self._topk_idx = topk_idx.reshape(-1, self.top_k)

        y = torch.zeros_like(x)                            # (B, T, d)
        for i in range(self.top_k):
            e_idx = topk_idx[..., i]                       # (B, T) 第 i 槽选中的专家编号
            w = topk_weights[..., i].unsqueeze(-1)         # (B, T, 1) 该槽位的权重
            for e in range(self.n_expert):
                sel = e_idx == e                           # (B, T) 哪些 token 的第 i 槽选了专家 e
                if sel.any():
                    # 只对被选中的 token 算该专家（稀疏激活：没被选的专家不参与本次计算）
                    y[sel] = y[sel] + w[sel] * self.experts[e](x[sel])
        return y

    def aux_loss(self):
        """当前 batch 的负载均衡损失（需先 forward 一次）。"""
        assert self._router_logits is not None, "先 forward 才能取 aux loss"
        return load_balancing_loss(self._router_logits, self._topk_idx, self.n_expert)


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 128  # 对齐 model.py 的 n_embd

    # 1) top-k 路由语义：router 明显偏好某专家 -> token 被路由过去，权重和为 1（重归一化）
    #    手动构造 logits：4 个 token 各强烈偏一个专家，看 top-k 怎么选、权重怎么重归一。
    logits = torch.zeros(1, 4, 4)
    logits[:, 0, 0] = 10.0   # token 0 偏专家 0
    logits[:, 1, 1] = 10.0   # token 1 偏专家 1
    logits[:, 2, 2] = 10.0   # token 2 偏专家 2
    logits[:, 3, 3] = 10.0   # token 3 偏专家 3
    probs = F.softmax(logits, dim=-1)
    topk_weights, topk_idx = torch.topk(probs, 2, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    print("1) top-k 路由语义（每个位置各偏好一个专家）:")
    for t in range(4):
        print("   token {}: 选中专家 {}，权重 {}（和为 {:.3f}）".format(
            t, topk_idx[0, t].tolist(),
            [round(float(w), 3) for w in topk_weights[0, t]],
            topk_weights[0, t].sum().item()))

    # 2) 稀疏激活 + 参数量：总参数 = n × 专家参数，每 token 激活 = top_k × 专家参数
    #    这是 MoE 的核心：容量放大 n 倍、计算量只放大 k 倍，二者解耦。
    expert_one = SwiGLU(dim)
    n_expert_params = sum(p.numel() for p in expert_one.parameters())
    print("\n2) 参数量（dim={}, 每专家 SwiGLU {} 参数）:".format(dim, n_expert_params))
    for n in [1, 4, 8]:
        total = n * n_expert_params + dim * n   # + router（dim -> n，无 bias）
        print("   n_expert={}: 总 {} 参数（每 token 激活 top_k 个专家 = 计算量跟 k 走，不跟 n 走）".format(
            n, total))

    # 3) 退化等价：k=1 只用单专家；k=n_expert 退化成「所有专家 softmax 加权平均」= 稠密
    #    验证 k=n_expert 时 MoE 输出 == 直接对 n 个专家的 softmax 加权（稠密 FFN 的泛化）。
    x = torch.randn(8, 4, dim)
    moe_all = MoE(dim, n_expert=4, top_k=4)   # k = n_expert
    with torch.no_grad():
        y_moe = moe_all(x)
        logits_all = moe_all.router(x)
        probs_all = F.softmax(logits_all, dim=-1)           # (B,T,n_expert)
        y_ref = torch.zeros_like(x)
        for e in range(4):
            y_ref += probs_all[..., e].unsqueeze(-1) * moe_all.experts[e](x)
    print("\n3) 退化等价: top_k = n_expert 时 MoE == 全专家 softmax 加权平均 = {}".format(
        torch.allclose(y_moe, y_ref, atol=1e-6)))

    # 4) 负载均衡：无 aux 训练时 router 坍缩到单一专家（所有 token 都路由过去），有 aux 时均匀。
    #    用 load_balancing_loss 量化：坍缩的 loss 大、均匀的 loss 小。
    N = 64
    collapse_logits = torch.zeros(N, 4); collapse_logits[:, 0] = 5.0   # 全偏专家 0 -> 坍缩
    balanced_logits = torch.randn(N, 4)                                # 随机 -> 大致均匀
    _, collapse_idx = torch.topk(F.softmax(collapse_logits, -1), 2, -1)
    _, balanced_idx = torch.topk(F.softmax(balanced_logits, -1), 2, -1)
    print("\n4) 负载均衡（top_k=2, n_expert=4）:")
    print("   坍缩（全偏专家0）的 aux loss = {:.3f}  （期望较大）".format(
        load_balancing_loss(collapse_logits, collapse_idx, 4).item()))
    print("   均匀（随机得分）的 aux loss = {:.3f}  （期望较小）".format(
        load_balancing_loss(balanced_logits, balanced_idx, 4).item()))

    # 5) 可微性（含 top-k 的软路由权重可反传；专家编号不可微但不影响，路由概率可微）
    x = torch.randn(4, 8, dim, requires_grad=True)
    MoE(dim)(x).sum().backward()
    print("\n5) 可微性: 反向传播 OK，x.grad 是否非空 = {}".format(x.grad is not None))
