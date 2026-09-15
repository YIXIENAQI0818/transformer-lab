# 把自建纯现代 GPT 导出到 HF 标准格式

回答一个问题：**model-tokenizer 训出的纯现代 GPT（RoPE/RMSNorm/SwiGLU/GQA/Flash + BPE），怎么变成 `from_pretrained` 能加载的标准 HF 模型？**

> 本项目是 hf-export 的第二次改造：第一次导出的是 char-level 标准 GPT-2（复用 GPT2LMHeadModel）；这次导出的是 model-tokenizer 的纯现代骨架（需自定义建模类）。

## 核心理解：数据 vs 代码（不变）

模型 = 「数据」+「代码」：

- **数据**：权重（float 张量）+ 词表（BPE 的 merges/vocab）
- **代码**：模型结构（怎么算 attention）+ tokenizer 算法（怎么 encode/decode）

迁移的本质：把「数据」按 HF 标准存好、把「代码」对应到 HF 的实现上。

## 和 char 时代的关键差异

| | char 时代（旧） | model-tokenizer 时代（现在） |
|---|---|---|
| 模型结构 | 标准 GPT-2 | 纯现代（RoPE/RMSNorm/SwiGLU/GQA/Flash） |
| 结构代码 | 复用 GPT2LMHeadModel | **自己写 modeling_modern_gpt.py** |
| 权重迁移 | Conv1D 转置 + attn.bias 剔除等 4 个坑 | **零转换**（参数名一致 + nn.Linear） |
| tokenizer | char（自定义 PreTrainedTokenizer） | BPE（HF 原生，直接包装） |

**核心权衡**：char 时代结构恰好是标准 GPT-2，所以「代码」复用现成的，但要处理 4 个权重坑；这次结构是纯现代（非标准），所以要「自己写整个建模类」，但换来权重迁移零转换、tokenizer 更简单。

## 流程概览

```
model-tokenizer/out/ckpt.pt + lib_tokenizer.json   （数据）
        │
        ├── ① 盘点 ── ckpt 里 model/config（无 meta，tokenizer 单独存）
        │
        ├── ② 权重迁移 ── ModernGPTForCausalLM 直接 load_state_dict（零转换）
        │        └─→ out/hf/model.safetensors + config.json
        │
        ├── ③ tokenizer 迁移 ── PreTrainedTokenizerFast 包装 → tokenizer.json
        │        └─→ out/hf/tokenizer.json + tokenizer_config.json
        │
        └── ④ 验证 ── logits 与原模型逐元素一致（0.00e+00）
```

## 关键：建模类 `modeling_modern_gpt.py`

纯现代骨架（RoPE/RMSNorm/SwiGLU/GQA/Flash）在 transformers 里没有现成类，所以自己写：

```python
class ModernGPTConfig(PretrainedConfig):
    model_type = "modern_gpt"
    # 字段对齐 model-tokenizer 的 GPTConfig（vocab_size / n_layer / n_kv_head / n_expert ...）

class ModernGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ModernGPTConfig
    base_model_prefix = "transformer"
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    # 内部结构照搬 model-tokenizer/src/model.py，参数名完全一致
```

**关键设计：参数名与 model-tokenizer/src/model.py 完全一致**（`transformer.wte.weight`、`transformer.h.{i}.attn.c_attn.weight`、`transformer.h.{i}.mlp.gate_proj.weight`…），所以 ② 的权重迁移是 `load_state_dict` 直接拷，零转换。

## transformers 5.x 踩的坑（这次新增）

1. **`_tied_weights_keys` 是 dict 不是 list**：格式 `{tied_key: target_key}`，如 `{"lm_head.weight": "transformer.wte.weight"}`（对比 GPT2 的 `{"lm_head.weight": "transformer.wte.weight"}`）。
2. **`all_tied_weights_keys` 在 `post_init()` 里设置**：必须调 `self.post_init()`（它会 init_weights + tie_weights + 设置 tied keys），不能只手动 `tie_weights()`。
3. **`generate` 要显式继承 `GenerationMixin`**：`PreTrainedModel` 本身不带 generate。
4. **config 要加标准字段别名**：`num_hidden_layers` / `hidden_size` / `num_attention_heads` / `head_dim` 等，generate 内部的 DynamicCache 会读它们（我们的字段叫 `n_layer` / `n_embd` / `n_head`）。

## 步骤④ 验证结果

喂同一段输入，比较原模型（model-tokenizer 的 GPT）和 HF 版（ModernGPTForCausalLM）的 logits：

```
logits 最大绝对误差: 0.00e+00
✅ logits 逐元素一致 —— 权重迁移 + 结构对齐都正确
```

误差 0 是因为权重零转换（参数名一致、无转置），比 char 时代的 7.63e-06（有 Conv1D 转置的浮点误差）还干净。最后 `hf_model.generate(...)` 生成一段，确认标准格式真的能跑。

## 之后怎么部署（vLLM）

HF 格式落地后，部署走 vLLM（内部用 transformers 加载，直接吃 HF 格式）：

```bash
vllm serve out/hf
```

（注：vLLM 对自定义架构需要额外的模型注册，本子项目只做到 HF 格式为止。）
