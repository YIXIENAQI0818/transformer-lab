"""decoder-only GPT 从零实现（阶段 1）。

目标：手写 GPT 的每个组件，不调用 nn.Transformer / nn.MultiheadAttention。
只依赖 torch 的基础张量算子与 nn.Linear / nn.Embedding / nn.LayerNorm，
attention 的 QKᵀ/√d、causal mask、softmax、加权求和全部手写。

结构（自上而下）：
    GPT
      ├─ wte  (token embedding)          vocab_size -> n_embd
      ├─ wpe  (positional embedding)     block_size -> n_embd
      ├─ drop
      ├─ h: [Block] * n_layer
      │     ├─ ln_1 -> attn (CausalSelfAttention) -> 残差
      │     └─ ln_2 -> mlp  (MLP)                 -> 残差
      └─ ln_f -> lm_head (与 wte 共享权重)

前向：idx (B,T) -> logits (B,T,vocab_size)；若给 targets，返回 (logits, cross_entropy_loss)。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """模型超参。默认值对应 ~10M 参数的小模型（6 层 x 384 x 6 头）。"""

    block_size: int = 256   # 上下文长度 T
    vocab_size: int = 65    # 词表大小（阶段 1 用 toy 值，阶段 2 换成真实 tokenizer）
    n_layer: int = 6        # transformer block 数量
    n_head: int = 6         # attention 头数
    n_embd: int = 384       # 隐层维度 d_model
    dropout: float = 0.0    # 阶段 1 关闭，训练时可开
    bias: bool = True       # Linear / LayerNorm 是否带 bias


class CausalSelfAttention(nn.Module):
    """多头因果自注意力。

    流程：Q/K/V 线性变换 -> 切分成 n_head 个头 -> 每头做
    softmax(QKᵀ / √d_head + causal_mask) · V -> 拼回 -> 输出投影。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 一个 Linear 同时产出 Q/K/V（3 * n_embd），nanoGPT 做法
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask：下三角为 1（可看到），上三角为 0（看不到未来）
        # 注册为 buffer：随模型移动 device，但不参与梯度
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, 时间步, 通道(n_embd)

        # 1. Q/K/V：一个线性层拆成三份
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. 切成多头：(B,T,n_embd) -> (B,n_head,T,head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 3. 缩放点积注意力（手写）
        #    att = softmax(q @ k^T / sqrt(d_head) + mask) @ v
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # mask：把上三角（未来位置）置为 -inf，softmax 后概率为 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v  # (B,n_head,T,head_size)

        # 4. 拼回头并投影回 n_embd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        return y


class MLP(nn.Module):
    """前馈网络 FFN：Linear -> GELU -> Linear，中间维度 4 * n_embd。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Block(nn.Module):
    """一个 transformer block：pre-norm 的 attention + FFN，各带残差。

    pre-LayerNorm（GPT-2 / nanoGPT 做法）：norm 放在子层之前，残差相加在之后。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # attention + 残差
        x = x + self.mlp(self.ln_2(x))    # FFN + 残差
        return x


class GPT(nn.Module):
    """完整 decoder-only GPT。前向 token ids -> logits，可选返回 loss。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying：lm_head 与 token embedding 共享权重（GPT-2 做法，省参且效果更好）
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化：残差路径缩放，保证训练稳定（nanoGPT 的 GPT-2 初始化）
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
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
        assert T <= self.config.block_size, \
            f"序列长度 {T} 超过 block_size {self.config.block_size}"

        # token embedding + positional embedding
        tok_emb = self.transformer.wte(idx)            # (B,T,n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)            # (T,n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 堆叠 block
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                       # (B,T,vocab_size)

        if targets is None:
            return logits
        # cross-entropy loss：把 (B,T,vocab_size) 摊平成 (B*T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成（阶段 2 采样会用到，这里先放最小版本）。"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits = self(idx_cond)  # 无 targets，返回 logits
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(1337)
    cfg = GPTConfig()
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数总量: {n_params/1e6:.2f}M")

    # 随机 toy 输入：batch=2, 长度=32
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    # 独立随机 targets（不取自 x），这样初始 loss 应接近 ln(vocab_size)
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss = model(x, y)
    print("logits shape:", tuple(logits.shape))
    print(f"loss: {loss.item():.4f}")
    print(f"期望 loss ≈ ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}")

    # 反向 + 梯度检查：确认计算图能回传
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    print(f"反向传播 OK，收到梯度的参数: {n_grad}")
