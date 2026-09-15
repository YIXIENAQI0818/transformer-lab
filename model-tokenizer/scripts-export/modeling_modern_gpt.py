"""自定义 HF 模型类：ModernGPTForCausalLM（纯现代骨架的 PreTrainedModel 版）。

背景：char 时代我们的模型是标准 GPT-2，能直接复用 transformers 的 GPT2LMHeadModel；
但 model-tokenizer 的纯现代骨架（RoPE / RMSNorm / SwiGLU / GQA / FlashAttention）不是标准架构，
transformers 里没有现成的类，所以要自己写一个 PreTrainedModel。

关键设计：**参数名与 model-tokenizer/src/model.py 完全一致**（transformer.wte.weight、
transformer.h.{i}.attn.c_attn.weight、transformer.h.{i}.mlp.gate_proj.weight ...），
这样权重迁移变成「直接 load_state_dict」，无需 char 时代的 Conv1D 转置 / attn.bias 剔除等坑。

结构（照搬 model-tokenizer/src/model.py，带 KV cache，去掉朴素 toggle）：
    ModernGPTForCausalLM
      └─ transformer (nn.ModuleDict)
           ├─ wte  (token embedding)
           ├─ drop
           ├─ h: [Block] * n_layer    （RMSNorm + GQA/RoPE/Flash attn + SwiGLU）
           └─ ln_f (RMSNorm)
      └─ lm_head （与 wte 共享权重）
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


# ---------------- RoPE（内联，照搬 model-tokenizer/src/model.py） ----------------

def precompute_rope_cache(head_dim, max_seq_len, base=10000.0, device=None, dtype=None):
    """预计算 RoPE 的 cos/sin 表，shape (max_seq_len, head_dim)。"""
    i = torch.arange(0, head_dim // 2, device=device, dtype=dtype)
    theta = base ** (-2 * i / head_dim)
    m = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(m, theta)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)


def rotate_pairs(x):
    """把相邻分量 pair (x0, x1) 映射为 (-x1, x0)，绕原点转 90°。"""
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    return torch.stack((-x1, x0), dim=-1).flatten(-2)


def apply_rotary_emb(x, cos, sin):
    """x' = x·cos + rotate_pairs(x)·sin。"""
    return x * cos + rotate_pairs(x) * sin


# ---------------- 配置 ----------------

class ModernGPTConfig(PretrainedConfig):
    """纯现代骨架的 HF 配置类，字段对齐 model-tokenizer 的 GPTConfig。"""

    model_type = "modern_gpt"
    # 让 config.save_pretrained 自动写 auto_map["AutoConfig"]，AutoConfig.from_pretrained
    # (trust_remote_code=True) 才能识别这个自定义配置类。
    _auto_class = "AutoConfig"

    def __init__(self, vocab_size=256, block_size=256, n_layer=6, n_head=6, n_embd=384,
                 dropout=0.0, bias=True, n_kv_head=2, n_expert=0, num_experts_per_tok=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.n_kv_head = n_kv_head
        self.n_expert = n_expert
        # 注意：MoE 的「每 token 专家数」在 model-tokenizer 里叫 top_k，但 top_k 是 transformers
        # generation 的保留字段名（采样的 top-k），撞名会导致 save_pretrained 校验失败，
        # 故这里改名为 num_experts_per_tok（Mixtral 的标准命名）。
        self.num_experts_per_tok = num_experts_per_tok
        # transformers 标准字段别名（generate / DynamicCache 等内部逻辑需要）
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.max_position_embeddings = block_size
        self.head_dim = n_embd // n_head


# ---------------- 组件 ----------------

class CausalSelfAttention(nn.Module):
    """GQA + RoPE + FlashAttention（照搬 model-tokenizer/src/model.py，去掉 KV cache）。"""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.c_attn = nn.Linear(
            config.n_embd, config.n_embd + 2 * self.n_kv_head * self.head_size, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, position_ids=None, use_causal=True, past_key_values=None, layer_idx=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(
            [self.n_embd, self.n_kv_head * self.head_size, self.n_kv_head * self.head_size], dim=2
        )
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        # RoPE 绝对位置：position_ids 由外层 forward 在进入层循环前一次性算好
        # （prefill 从头 0..T-1、decode 从 past_len 起），避免多层 prefill 时每层现调
        # get_seq_length() 读到前面层已 update 的长度（KV cache + RoPE 的经典 bug）。
        pos = position_ids if position_ids is not None else torch.arange(T, device=x.device)
        cos, sin = precompute_rope_cache(
            self.head_size, int(pos.max().item()) + 1, device=x.device, dtype=x.dtype
        )
        cos, sin = cos[pos], sin[pos]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV cache：把新 token 的 K/V 追加进 cache，得到完整 K/V（过去 + 新）。
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx)

        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # 因果 mask：prefill 时 use_causal=True、decode 时 False（由外层 forward 传入）。
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return F.dropout(self.c_proj(y), p=self.dropout, training=self.training)


class MLP(nn.Module):
    """SwiGLU FFN（n_expert=0 的 dense 分支；MoE 暂不实现，ckpt 为 dense）。"""

    def __init__(self, config):
        super().__init__()
        self.n_expert = config.n_expert
        if config.n_expert > 0:
            raise NotImplementedError("本导出暂只支持 dense SwiGLU（n_expert=0）")
        hidden = (8 * config.n_embd) // 3
        self.gate_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.down_proj = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return F.dropout(x, p=self.dropout, training=self.training)


class Block(nn.Module):
    """pre-norm 的 attention + FFN，各带残差（RMSNorm）。"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, position_ids=None, use_causal=True, past_key_values=None, layer_idx=None):
        x = x + self.attn(
            self.ln_1(x), position_ids=position_ids, use_causal=use_causal,
            past_key_values=past_key_values, layer_idx=layer_idx
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class ModernGPTForCausalLM(PreTrainedModel, GenerationMixin):
    """纯现代 decoder-only GPT 的 HF 版（参数名与 model-tokenizer/src/model.py 对齐）。"""

    config_class = ModernGPTConfig
    base_model_prefix = "transformer"
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    # _auto_class 标记本模型对应哪个 Auto 类：save_pretrained 时据此触发 custom_object_save，
    # 自动把 modeling_modern_gpt.py 复制进产物目录 + 在 config 里写 auto_map（别人 trust_remote_code 加载）。
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # post_init 会调用 _init_weights（初始化）+ tie_weights（weight tying）+
        # 设置 all_tied_weights_keys（save_pretrained 识别 shared tensor 需要）
        self.post_init()

        # 残差缩放（GPT-2 初始化：c_proj/down_proj 缩小，保证残差流训练初期稳定）
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

    def get_input_embeddings(self):
        return self.transformer.wte

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, labels=None, attention_mask=None, past_key_values=None,
                position_ids=None, use_cache=False, **kwargs):
        B, T = input_ids.shape
        x = self.transformer.wte(input_ids)
        x = self.transformer.drop(x)

        # use_cache=True 且首次调用（无 cache）时初始化 DynamicCache。
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        # 进入层循环前，一次性算好绝对位置和因果标志：
        # - prefill（cache 空，past_len=0）：position_ids=[0..T-1]，use_causal=True
        # - decode（cache 有内容，past_len>0）：position_ids=[past_len]，use_causal=False
        # 不能在各层现算 get_seq_length()，因为前面层 update 后长度已变（多层 prefill 的坑）。
        if position_ids is None:
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(past_len, past_len + T, device=input_ids.device)
        use_causal = (position_ids.min().item() == 0)

        for i, block in enumerate(self.transformer.h):
            x = block(x, position_ids=position_ids, use_causal=use_causal,
                      past_key_values=past_key_values, layer_idx=i)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values,
                                      hidden_states=None, attentions=None)
