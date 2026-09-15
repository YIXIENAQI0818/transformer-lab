"""纯现代 decoder-only GPT（model-tokenizer 的模型骨架）。

从 model-upgrade 的「带 toggle 对比教学版」精简而来：去掉朴素组件分支，
固定为现代大模型配置 —— RoPE + RMSNorm + SwiGLU + GQA + FlashAttention，
仅保留 n_kv_head（GQA 头数）和 n_expert（MoE 专家数）两个模型超参。

本文件**自包含**：把 RoPE / SwiGLU / MoE 的实现内联进来（作为工具函数/类），
不再 import 外部模块（对应 model-upgrade/src/components/ 下的 rope / swiglu / moe）。

内联的部分：
  - RoPE  ：precompute_rope_cache / rotate_pairs / apply_rotary_emb
  - SwiGLU：swish / SwiGLU（被 MoE 的专家复用）
  - MoE   ：load_balancing_loss / MoE

保留的行为：KV cache（forward 参数 cache/use_cache）、RoPE 绝对位置 decode、GQA
repeat_interleave 广播、weight tying、GPT-2 残差缩放初始化、MoE aux_loss。

前向：idx (B,T) -> logits (B,T,vocab_size)；给 targets 则返回 (logits, loss)。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------- RoPE（内联实现） ----------------

def precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device=None, dtype=None):
    """预计算 RoPE 的 cos / sin 表，shape 均为 (max_seq_len, head_dim)。"""
    i = torch.arange(0, head_dim // 2, device=device, dtype=dtype)
    theta = base ** (-2 * i / head_dim)          # (head_dim/2,)
    m = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(m, theta)               # (max_seq_len, head_dim/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)


def rotate_pairs(x):
    """把相邻分量 pair (x0, x1) 映射为 (-x1, x0)，即绕原点转 90°。"""
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    return torch.stack((-x1, x0), dim=-1).flatten(-2)


def apply_rotary_emb(x, cos, sin):
    """x' = x·cos + rotate_pairs(x)·sin。"""
    return x * cos + rotate_pairs(x) * sin


# ---------------- SwiGLU（内联实现） ----------------

def swish(x):
    """Swish 激活，又名 SiLU：x · sigmoid(x)。"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """门控 FFN：SwiGLU(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))。

    hidden_dim 默认 8/3·dim（对齐标准 4·dim FFN 的参数量）。
    """

    def __init__(self, dim, hidden_dim=None, bias=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (8 * dim) // 3  # 8/3·d，整数运算避免浮点误差
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.down_proj(swish(self.gate_proj(x)) * self.up_proj(x))


# ---------------- MoE（内联实现） ----------------

def load_balancing_loss(logits, topk_idx, n_expert):
    """Mixtral 式负载均衡损失：鼓励每个专家被选中 / 被路由都均匀。

    loss = n_expert · Σ_e (f_e · P_e)。路由坍缩到单一专家时 loss 大，均匀时最小。
    """
    N = logits.size(0)
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(topk_idx, n_expert).float()
    f_e = one_hot.sum(dim=(0, 1)) / N
    P_e = probs.mean(dim=0)
    return n_expert * (f_e * P_e).sum()


class MoE(nn.Module):
    """混合专家 FFN：router 选 top-k 个 SwiGLU 专家，加权求和。

    MoE(x) = Σ_{e ∈ top-k} w_e · SwiGLU_e(x)。稀疏激活动作藏在 sel.any() 里——
    没被任何 token 选中的专家根本不算，这是 MoE 省算力的来源。
    """

    def __init__(self, dim, n_expert=4, top_k=2, hidden_dim=None, bias=True):
        super().__init__()
        assert 1 <= top_k <= n_expert, f"top_k={top_k} 必须在 [1, n_expert={n_expert}] 内"
        if hidden_dim is None:
            hidden_dim = (8 * dim) // 3
        self.n_expert = n_expert
        self.top_k = top_k
        self.router = nn.Linear(dim, n_expert, bias=False)  # d -> n_expert 个得分
        self.experts = nn.ModuleList([SwiGLU(dim, hidden_dim, bias) for _ in range(n_expert)])
        self._router_logits = None  # (N, n_expert)
        self._topk_idx = None       # (N, top_k)

    def forward(self, x):
        B, T, d = x.shape
        logits = self.router(x)                            # (B, T, n_expert)
        probs = F.softmax(logits, dim=-1)
        topk_weights, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        self._router_logits = logits.reshape(-1, self.n_expert)
        self._topk_idx = topk_idx.reshape(-1, self.top_k)

        y = torch.zeros_like(x)
        for i in range(self.top_k):
            e_idx = topk_idx[..., i]                       # (B, T) 第 i 槽选中的专家编号
            w = topk_weights[..., i].unsqueeze(-1)         # (B, T, 1)
            for e in range(self.n_expert):
                sel = e_idx == e
                if sel.any():
                    y[sel] = y[sel] + w[sel] * self.experts[e](x[sel])
        return y

    def aux_loss(self):
        """当前 batch 的负载均衡损失（需先 forward 一次）。"""
        assert self._router_logits is not None, "先 forward 才能取 aux loss"
        return load_balancing_loss(self._router_logits, self._topk_idx, self.n_expert)


@dataclass
class GPTConfig:
    """模型超参。默认值对应完整训练规模（6 层 x 384 x 6 头，block 256）。"""

    block_size: int = 256   # 上下文长度 T
    vocab_size: int = 256   # 词表大小，由调用方（tokenizer）注入；byte-level 最小 256
    n_layer: int = 6        # transformer block 数量
    n_head: int = 6         # attention 头数
    n_embd: int = 384       # 隐层维度 d_model
    dropout: float = 0.0
    bias: bool = True
    n_kv_head: int = 2      # GQA：K/V 头数（1=MQA，2=GQA，n_head=退化 MHA）
    n_expert: int = 0       # 0 = dense SwiGLU；n = MoE（n 个 SwiGLU 专家 + router）
    top_k: int = 2          # MoE 每个 token 激活的专家数（n_expert > 0 时生效）


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：GQA（K/V 头共享）+ RoPE（旋转位置）+ FlashAttention（sdpa）。

    n_kv_head 控制 K/V 头数（Q 始终是 n_head 个头）：n_kv_head < n_head 即 GQA，
    多 Q 头分组共享 K/V 头；n_kv_head == n_head 退化为 MHA。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0, (
            f"n_head={config.n_head} 必须能被 n_kv_head={config.n_kv_head} 整除（每组 Q 头数相等）"
        )
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        # c_attn 输出：Q 全量 n_embd + K/V 各 n_kv_head * head_size（GQA 时 K/V 投影更窄）
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd + 2 * self.n_kv_head * self.head_size, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, cache=None, use_cache=False):
        """前向。cache=None 是全序列自注意力（训练 / prefill）；cache 给定时是增量 decode。

        cache: (k_cache, v_cache)，各 (B, n_kv_head, L, head_size)。增量 decode 时 x 只有
        (B, 1, C)，把它的 k/v 追加进 cache 后，q 只 attend 到 [过去 L 个 + 自己]。
        返回 (y, new_cache) 当 cache 给定或 use_cache=True；否则只返回 y。
        """
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(
            [self.n_embd, self.n_kv_head * self.head_size, self.n_kv_head * self.head_size],
            dim=2,
        )
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        # RoPE：对 q/k 按绝对位置旋转。增量 decode 时新 token 的绝对位置是 cache 长度 pos_start，
        # 不能用局部下标 0，否则相对位置信息全错（KV cache + RoPE 的经典 bug）。
        pos_start = cache[0].size(2) if cache is not None else 0
        cos, sin = precompute_rope_cache(
            self.head_size, pos_start + T, device=x.device, dtype=x.dtype
        )
        cos = cos[pos_start:pos_start + T]
        sin = sin[pos_start:pos_start + T]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 先追加 cache 再广播：cache 存未广播的 n_kv_head 粒度（GQA 省显存的核心）。
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_cache = (k, v) if (cache is not None or use_cache) else None

        # GQA：K/V 从 n_kv_head 广播到 n_head。repeat_interleave 是「连续分组共享」
        # [K0,K1] -> [K0,K0,K1,K1]；repeat 会得到错误的交错共享 [K0,K1,K0,K1]。
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # FlashAttention：交给 F.scaled_dot_product_attention（底层 fused 实现）。
        # is_causal 只在 cache=None（完整自注意力）时需要；decode 时 q 是最新位置、
        # 应 attend 全部 key，无需因果 mask。
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=(cache is None),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = F.dropout(self.c_proj(y), p=self.dropout, training=self.training)

        if cache is not None or use_cache:
            return y, new_cache
        return y


class MLP(nn.Module):
    """前馈网络 FFN：dense SwiGLU（n_expert=0）或 MoE（n_expert=n）。

    - dense SwiGLU：gate/up/down 三个投影，中间维度 8/3·d（对齐 4·d FFN 参数量）。
    - MoE：n 个 SwiGLU 专家 + router + top-k 稀疏激活（见上方内联的 MoE 类）。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_expert = config.n_expert
        if config.n_expert > 0:
            self.moe = MoE(config.n_embd, n_expert=config.n_expert,
                           top_k=config.top_k, bias=config.bias)
        else:
            hidden = (8 * config.n_embd) // 3  # 整数运算，避免浮点误差（384 -> 1024）
            self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        if self.n_expert > 0:
            return F.dropout(self.moe(x), p=self.dropout, training=self.training)
        # SwiGLU(x) = down(SiLU(gate(x)) ⊙ up(x))
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return F.dropout(x, p=self.dropout, training=self.training)

    def aux_loss(self):
        """MoE 的负载均衡损失（dense FFN 无路由，返回 0，便于统一累加）。"""
        if self.n_expert > 0:
            return self.moe.aux_loss()
        return torch.zeros((), device=next(self.parameters()).device)


class Block(nn.Module):
    """pre-norm 的 attention + FFN，各带残差。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cache=None, use_cache=False):
        """cache 只沿 attention 进出（MLP 无状态、不缓存）。"""
        want_cache = cache is not None or use_cache
        attn_out = self.attn(self.ln_1(x), cache=cache, use_cache=use_cache)
        if want_cache:
            a, a_cache = attn_out
        else:
            a = attn_out
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        if want_cache:
            return x, a_cache
        return x


class GPT(nn.Module):
    """纯现代 decoder-only GPT：RoPE + RMSNorm + SwiGLU + GQA + FlashAttention。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying：lm_head 与 token embedding 共享权重
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 初始化：残差路径缩放（attention 的 c_proj、SwiGLU 的 down_proj 缩小初始化）
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, cache=None, use_cache=False):
        """前向。cache/use_cache 控制 KV cache（见 CausalSelfAttention.forward）。

          - cache=None, use_cache=False → 常规全序列前向（训练），返回 logits（或 (logits, loss)）。
          - cache=None, use_cache=True  → prefill：全前向并返回 (logits, cache)。
          - cache=<list>               → decode：idx 必须是 (B,1)，返回 (logits, new_cache)。

        cache 是「每层一个 (k_cache, v_cache)」的 list，形状各 (B, n_kv_head, L, head_size)。
        """
        B, T = idx.shape
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)

        want_cache = cache is not None or use_cache
        new_caches = []
        for i, block in enumerate(self.transformer.h):
            block_cache = cache[i] if cache is not None else None
            out = block(x, cache=block_cache, use_cache=want_cache)
            if want_cache:
                x, block_new_cache = out
                new_caches.append(block_new_cache)
            else:
                x = out

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if want_cache:
            return logits, new_caches

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成（KV cache 增量 decode：prefill 一次 + 每步只算新 token）。

        复用 forward 的 cache/use_cache 接口（回合 06 KV cache）：prefill 并行算整段 prompt
        并缓存 K/V，之后每步只算 1 个新 token、复用过去 K/V。RoPE 的绝对位置由 cache 长度给出。
        """
        logits, cache = self(idx, use_cache=True)          # prefill：并行算整段 prompt
        for _ in range(max_new_tokens):
            logit = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logit, min(top_k, logit.size(-1)))
                logit[logit < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            logits, cache = self(idx_next, cache=cache)  # decode：复用 cache 只算新 token
        return idx


if __name__ == "__main__":
    torch.manual_seed(1337)
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params/1e6:.3f}M（vocab=512, 4 层 x 128 x 4 头）")

    # 前向 + loss
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    y = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x, y)
    print(f"logits shape: {tuple(logits.shape)}   loss: {loss.item():.4f}")

    # 反向
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    print(f"反向传播 OK，收到梯度的参数: {n_grad}/{sum(1 for _ in model.parameters())}")

    # 生成
    out = model.generate(torch.randint(0, cfg.vocab_size, (1, 1)), max_new_tokens=8)
    print(f"生成 shape: {tuple(out.shape)}（1 x 9）")

    # KV cache：prefill + 逐个 decode 与一次性前向的 logits 一致
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    logits_full = model(x)
    logits_pre, cache = model(x[:, :8], use_cache=True)          # prefill 前 8 个
    logits_dec = []
    for i in range(8):
        logit, cache = model(x[:, 8 + i:9 + i], cache=cache)      # 逐个 token decode（每次 1 个）
        logits_dec.append(logit)
    logits_dec = torch.cat(logits_dec, dim=1)
    err = (torch.cat([logits_pre, logits_dec], dim=1) - logits_full).abs().max().item()
    print(f"KV cache 一致性: prefill+decode vs full 的 logits 最大误差 = {err:.2e}")

    # MoE smoke test：n_expert>0 时能前向 + aux_loss
    cfg_moe = GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=4, n_embd=128,
                        n_expert=4, top_k=2)
    moe_model = GPT(cfg_moe)
    x = torch.randint(0, cfg_moe.vocab_size, (2, 64))
    logits_moe, loss_moe = moe_model(x, y)
    aux = sum(b.mlp.aux_loss() for b in moe_model.transformer.h)
    print(f"MoE 前向 OK：loss={loss_moe.item():.4f}  aux_loss={aux.item():.4f}")
