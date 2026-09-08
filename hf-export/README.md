# 把自建 GPT 导出到 HF 标准格式 —— 逐步详解

这个项目回答一个问题：**我们手写训练的 `ckpt.pt`，怎么变成 `AutoModel.from_pretrained` 能加载的标准 HF 模型？**

> 部署侧（vLLM / Ollama 等）后续在 llm-lab 里做，本子项目**只做到 HF 格式为止**——因为 HF 格式是「训练/微调 → 部署」整条链的枢纽：训练产 HF、微调产 HF、vLLM 也吃 HF。

## 核心理解：数据 vs 代码

之前几轮讨论澄清了一个贯穿始终的概念：

> 模型 = 「数据」 + 「代码」
> - **数据**：权重（一堆 float 数字）+ 词表（char↔id 映射）
> - **代码**：模型结构（怎么算 attention）+ tokenizer 算法（怎么 encode/decode）

我们的 `ckpt.pt`（数据）+ `model.py`/`tokenizer.py`（代码）都在自己仓库里。transformers 库把「代码」内置得很全、把「数据」格式定得很标准。所以「迁移」的本质，就是把我们的「数据」按 HF 标准存好、把「代码」对应到 transformers 库里已有的实现上。

我们的模型结构恰好就是**标准 GPT-2**（阶段 1 照 nanoGPT/GPT-2 写的），所以「结构代码」可以直接复用 transformers 的 `GPT2LMHeadModel`——这就是为什么迁移比想象的简单。

## 流程概览

```
pretraining/out/ckpt.pt          （数据：state_dict + config + meta）
        │
        ├── ① 盘点 ── 看清 ckpt 里有什么
        │
        ├── ② 权重迁移 ── 转成 GPT2LMHeadModel + 处理 Conv1D 转置等 4 个坑
        │        └─→ out/hf/model.safetensors + config.json
        │
        ├── ③ tokenizer 迁移 ── stoi/itos → vocab.json + 自定义 PreTrainedTokenizer
        │        └─→ out/hf/vocab.json + tokenizer_config.json
        │
        └── ④ 验证 ── AutoModel 加载，logits 与原模型逐元素一致
                 └─→ 标准 HF 格式，可直接被 transformers / vLLM 使用
```

---

## 步骤① `01_inspect_ckpt.py` —— 盘点 ckpt

`torch.save` 存的是一个 dict，含 4 个 key：

| key | 是什么 | 角色 |
|-----|--------|------|
| `model` | 所有权重（state_dict，float 张量） | 数据的主体 |
| `config` | 结构超参（vocab_size/n_layer/n_embd…） | 数据（是「量」不是「逻辑」） |
| `meta` | tokenizer 词表（stoi/itos） | tokenizer 的数据 |
| `iter` | 训练步数 | 元信息 |

跑 `python src/01_inspect_ckpt.py`，能看到 `transformer.wte.weight (65,384)`、`transformer.h.0.attn.c_attn.weight (1152,384)` 等 85 个 key。**注意 `c_attn.weight` 是 `(1152,384)`——这是 `nn.Linear` 的 `(out,in)` 布局，后面步骤② 的关键坑就来自这里。**

---

## 步骤② `02_convert_weights.py` —— 权重迁移（最核心）

核心结论：**我们的 key 名和 `GPT2LMHeadModel` 几乎完全一致**（因为 nanoGPT 照 GPT-2 写），所以迁移是「逐 key 拷贝 + 处理 4 个坑」：

**坑 1：Conv1D vs Linear 转置**
- 我们的 `nn.Linear(in,out)` 权重存成 `(out,in)`，如 `c_attn.weight (1152,384)`；
- transformers 的 GPT-2 用 `Conv1D(in,out)`，权重反着存成 `(in,out)`，即 `(384,1152)`。
- 所以 `c_attn` / `c_proj`(attn) / `c_fc` / `c_proj`(mlp) 共 **24 个权重**要 `.t()` 转置（bias 不用）。

**坑 2：剔除 `attn.bias`** —— 这是 `register_buffer` 存的 causal mask（下三角），GPT-2 不存它，前向时用 `attention_mask` 动态生成。要用 `k.endswith(".attn.bias")` 精确剔除（别误伤 `c_attn.bias`，那是 QKV 投影的 bias）。

**坑 3：token id 越界** —— `GPT2Config` 默认 `bos/eos_token_id=50256`，在 vocab=65 下越界，置 `None` 消除警告。

**坑 4：激活函数对齐** —— nanoGPT 用 `nn.GELU()`（精确 erf 版），GPT-2 默认 `gelu_new`（tanh 近似）。不设 `activation_function="gelu"` 会导致步骤④ 的 logits 对不上。

最后 `save_pretrained(safe_serialization=True)` 存 `model.safetensors` + `config.json`，并用 `strict=False` 检查 `missing/unexpected` 都为空（key 完全对齐）。

---

## 步骤③ `03_convert_tokenizer.py` —— tokenizer 迁移

tokenizer 也分「数据 + 代码」：
- **数据**（stoi/itos 那张 65 条映射）→ 存成 `vocab.json`（JSON 正确转义换行符）；
- **代码**（encode/decode 逻辑）→ 写成 `HFCharTokenizer`（继承 `PreTrainedTokenizer`，见 `hf_char_tokenizer.py`），实现 `_tokenize`（逐字符）、`_convert_token_to_id`、`convert_tokens_to_string`（直接拼接）。

两个 HF 约定（踩过的坑）：
1. **`vocab_files_names = {"vocab_file": "vocab.json"}`** + `__init__` 接受 `vocab_file`（文件**路径**）而不是 `vocab`（dict）——`AutoTokenizer` 加载时会把 vocab.json 的路径作为 `vocab_file` 参数传入。
2. **`auto_map` 格式**是 `["module.ClassName", null]`（`module` 不含 `.py`），让 `AutoTokenizer.from_pretrained(dir, trust_remote_code=True)` 能从我们复制进去的 `hf_char_tokenizer.py` 找到类。

---

## 步骤④ `04_verify_hf.py` —— 验证迁移正确

严谨的验证：喂同一段输入，比较**原模型 GPT** 和 **HF 版 GPT2LMHeadModel** 的 logits。

结果：**最大绝对误差 7.63e-06**（< 1e-5），逐元素一致。这证明：权重没丢、Conv1D 转置对、激活函数对齐对——「数据 + 代码」在 HF 侧正确组合了。脚本最后用 `hf_model.generate(...)` 生成一段莎士比亚对白，确认标准格式真的能跑。

---

## 之后怎么部署（vLLM）

HF 格式落地后，部署直接走 vLLM（它内部用 transformers 加载，**直接吃 HF 格式**，无需转换）：

```bash
pip install vllm
vllm serve out/hf        # 起一个 OpenAI 兼容服务，和 Ollama 一样提供 API
```

这条「transformers 训练 → HF 格式 → vLLM 部署」的链路，后续在 llm-lab 里对大模型微调后同样适用（微调产出也是 HF 格式）。

---

## 附：push 到 HuggingFace Hub（可选）

本地已生成标准 HF 格式，如需真实上传：

```bash
pip install huggingface_hub
huggingface-cli login
cd out/hf
huggingface-cli upload <你的用户名>/my-gpt-shakespeare . .   # 上传整个目录
```

（本项目按约定不真实 push，仅本地生成 + 验证。）
