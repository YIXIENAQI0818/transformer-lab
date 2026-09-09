"""可配置的 decoder-only GPT（component-upgrade 的共享骨架）。

从 gpt-from-scratch 的朴素实现演化而来：给每个待升级组件加一个开关，
7 个回合共用这一个 model.py，各回合模块只负责实现对应组件本身
（rope.py / rmsnorm.py / gqa.py ...）。

当前已引入的开关：
    pos_enc:    "learned"（朴素 wpe） | "rope"（旋转位置编码，见 rope.py）
    norm:       "layernorm"（朴素 LN） | "rmsnorm"（RMS 归一化，见 rmsnorm.py）
    activation: "gelu"（朴素 FFN） | "swiglu"（门控 FFN，见 swiglu.py）
    n_kv_head:  K/V 头数（0 = n_head 即 MHA；2 = GQA，见 gqa.py）
    attn_impl:  "naive"（手写 QKᵀ+softmax） | "flash"（F.scaled_dot_product_attention，见 flash_attn.py）

前向：idx (B,T) -> logits (B,T,vocab_size)；给 targets 则返回 (logits, loss)。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from rope import apply_rotary_emb, precompute_rope_cache


@dataclass
class GPTConfig:
    """模型超参。默认值对应 ~0.8M 参数的小模型，便于快速做替换前后对比。"""

    block_size: int = 64    # 上下文长度 T（故意设小，方便演示长度外推）
    vocab_size: int = 65    # 词表大小，由调用方（tokenizer）注入
    n_layer: int = 4        # transformer block 数量
    n_head: int = 4         # attention 头数
    n_embd: int = 128       # 隐层维度 d_model
    dropout: float = 0.0
    bias: bool = True
    pos_enc: str = "learned"  # "learned" | "rope"
    norm: str = "layernorm"   # "layernorm" | "rmsnorm"
    activation: str = "gelu"  # "gelu" | "swiglu"（swiglu 是门控结构，替换整个 FFN 而非仅激活函数）
    n_kv_head: int = 0  # K/V 头数：0 = 等于 n_head（MHA）；设 2 = GQA，1 = MQA（K/V 头共享，见 gqa.py）
    attn_impl: str = "naive"  # "naive" | "flash"（注意力实现，见 flash_attn.py）


def _build_norm(config: GPTConfig, dim: int):
    """按 config.norm 建归一化层。

    LayerNorm 有 gamma+beta 两个参数；RMSNorm 只有 gamma（省掉 beta 的 dim 个参数）。
    注意 RMSNorm 本身无 bias，config.bias 对它不生效（这是 RMSNorm 的设计，不是遗漏）。

    rmsnorm 分支直接用内置 nn.RMSNorm（PyTorch 的 fused 生产实现）；手写教学版在
    rmsnorm.py，用于看清 RMSNorm 内部逻辑（见该文件 docstring）。
    """
    if config.norm == "layernorm":
        return nn.LayerNorm(dim, bias=config.bias)
    if config.norm == "rmsnorm":
        return nn.RMSNorm(dim)
    raise ValueError(f"未知 norm: {config.norm}")


class CausalSelfAttention(nn.Module):
    """多头因果自注意力，支持 learned / rope 位置编码，以及 MHA / GQA / MQA。

    n_kv_head 控制 K/V 的头数（Q 始终是 n_head 个头）：
      - n_kv_head = n_head  → MHA（每个 Q 头配自己的 K/V 头）
      - n_kv_head < n_head  → GQA（多 Q 头分组共享 K/V 头）
      - n_kv_head = 1       → MQA（所有 Q 头共享 1 个 K/V 头）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head > 0 else config.n_head
        assert self.n_head % self.n_kv_head == 0, (
            f"n_head={self.n_head} 必须能被 n_kv_head={self.n_kv_head} 整除（每组 Q 头数相等）"
        )
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.pos_enc = config.pos_enc
        self.attn_impl = config.attn_impl
        self.dropout = config.dropout
        # c_attn 输出：Q 全量 n_embd + K/V 各 n_kv_head * head_size（GQA 时 K/V 投影更窄）
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd + 2 * self.n_kv_head * self.head_size, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # causal mask：下三角为 1（可看到），上三角为 0（看不到未来）。注册为 buffer。
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape

        # Q 全量，K/V 各 n_kv_head 个头（维度不是均分三段了）
        q, k, v = self.c_attn(x).split(
            [self.n_embd, self.n_kv_head * self.head_size, self.n_kv_head * self.head_size],
            dim=2,
        )

        # 切多头：Q 切 n_head 个头，K/V 各切 n_kv_head 个头
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        # RoPE：对 q/k 按绝对位置旋转（v 不旋转，位置信息只影响 attention score）。
        # 在广播前做，只对 n_kv_head 个 K 头旋转（先旋转再复制 = 先复制再旋转，但前者更省）。
        if self.pos_enc == "rope":
            cos, sin = precompute_rope_cache(self.head_size, T, device=x.device, dtype=x.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # GQA 关键：把 K/V 从 n_kv_head 个头广播到 n_head 个头。
        # 用 repeat_interleave 而非 repeat——GQA 是「连续分组共享」：
        #   [K0,K1] -> repeat_interleave(2) -> [K0,K0,K1,K1]（Q0/Q1 共享 K0，Q2/Q3 共享 K1）
        #   repeat 会得到错误的 [K0,K1,K0,K1]（交错共享，不是 GQA 语义）。
        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # 注意力计算，按 attn_impl 分支：
        #   "naive"：手写 QKᵀ + softmax，物化完整 (B,n_head,T,T) 矩阵（显存 O(T²)）
        #   "flash"：交给 F.scaled_dot_product_attention（PyTorch 内置 fused 实现，底层即
        #            FlashAttention / memory-efficient attention，不物化 T² 矩阵，见 flash_attn.py）
        if self.attn_impl == "naive":
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            if T <= self.block_size:
                mask = self.bias[:, :, :T, :T]
            else:
                # 长度外推时现算一个更大的下三角 mask（只有 rope 模式能走到这里）
                mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            y = att @ v  # (B,n_head,T,head_size)
        else:  # flash
            # sdpa 的 is_causal 用内置因果 mask（不占 block_size² 的 buffer，任意长度可用），
            # scale 默认 1/sqrt(head_size) 与 naive 一致；dropout 仅训练时生效。
            # k/v 已在前面 repeat_interleave 广播到 n_head（sdpa 本身也支持 GQA 广播，这里
            # 统一先广播、两分支共用同一份 k/v，保证 naive/flash 输入完全一致）。
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        # 拼回头并投影回 n_embd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return F.dropout(self.c_proj(y), p=self.dropout, training=self.training)


class MLP(nn.Module):
    """前馈网络 FFN，按 config.activation 分两种结构。

    - "gelu"  ：标准 FFN，Linear -> GELU -> Linear，中间维度 4 * n_embd。
    - "swiglu"：门控 FFN，SwiGLU(x) = down(SiLU(gate(x)) ⊙ up(x))，三个投影，
                 中间维度 8/3 * n_embd（对齐 4·d FFN 的参数量，见 swiglu.py）。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.activation = config.activation
        if config.activation == "gelu":
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        elif config.activation == "swiglu":
            # 中间维度 8/3·d：SwiGLU 有 3 个投影而 GELU 只有 2 个，取 8/3·d 使
            # 总参数量 3·(d·8d/3) = 8d² 与标准 4·d FFN 的 2·(d·4d) = 8d² 对齐。
            hidden = (8 * config.n_embd) // 3  # 整数运算，避免浮点误差（128 -> 341）
            self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)  # 门（过 SiLU）
            self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)    # 值（不过激活）
            self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)  # 输出
        else:
            raise ValueError(f"未知 activation: {config.activation}")
        self.dropout = config.dropout

    def forward(self, x):
        if self.activation == "gelu":
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
        else:  # swiglu：SwiGLU(x) = down(SiLU(gate(x)) ⊙ up(x))
            x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return F.dropout(x, p=self.dropout, training=self.training)


class Block(nn.Module):
    """pre-norm 的 attention + FFN，各带残差。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = _build_norm(config, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = _build_norm(config, config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """可配置 decoder-only GPT。pos_enc="rope" 时不建 wpe（省掉 block_size*n_embd 参数）。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=_build_norm(config, config.n_embd),
        ))
        # learned 模式才需要位置 embedding；rope 模式下位置信息由 attention 内的旋转给出
        if config.pos_enc == "learned":
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying：lm_head 与 token embedding 共享权重
        self.transformer.wte.weight = self.lm_head.weight

        # GPT-2 初始化：残差路径缩放
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            # 残差子层的输出投影（attention 的 c_proj、GELU FFN 的 c_proj、SwiGLU FFN 的
            # down_proj）都缩小初始化，保证残差流在训练初期稳定；SwiGLU 的输出投影叫
            # down_proj（LLaMA 惯例），需一并纳入，否则两种 FFN 初始化不一致、对比不公平。
            if pn.endswith("c_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # token embedding（+ learned 位置 embedding）
        x = self.transformer.wte(idx)              # (B,T,n_embd)
        if "wpe" in self.transformer:
            # learned 模式：wpe 只有 block_size 行，位置 >= block_size 无法编码。
            # 注意 nn.Embedding 默认不做越界检查（会静默读出垃圾值），必须显式断言，
            # 否则长度外推测试会"看似能跑"却得到错误结果。
            assert T <= self.config.block_size, (
                f"learned wpe 只能编码 block_size={self.config.block_size} 内的位置，"
                f"收到 T={T}（长度外推需要 RoPE）"
            )
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                   # (B,T,vocab_size)

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
