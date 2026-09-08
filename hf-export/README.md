# 把自建 GPT 导出到 HF / Ollama —— 逐步详解

这个项目回答一个问题：**我们手写训练的 `ckpt.pt`，怎么变成 `AutoModel.from_pretrained` 和 `ollama run` 都能用的标准模型？**

## 核心理解：数据 vs 代码

之前几轮讨论澄清了一个贯穿始终的概念：

> 模型 = 「数据」 + 「代码」
> - **数据**：权重（一堆 float 数字）+ 词表（char↔id 映射）
> - **代码**：模型结构（怎么算 attention）+ tokenizer 算法（怎么 encode/decode）

我们的 `ckpt.pt`（数据）+ `model.py`/`tokenizer.py`（代码）都在自己仓库里。HF 和 Ollama 的区别只是：**它们把「代码」内置得更全、把「数据」格式更标准**。所以「迁移」的本质，就是把我们的「数据」按标准格式存好、把「代码」对应到标准库已有的实现上。

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
        ├── ④ 验证 ── AutoModel 加载，logits 与原模型逐元素一致
        │
        └── ⑤ 导入 Ollama ── ollama create --experimental 读 safetensors
                 └─→ my-gpt（受 Ollama 0.33.2 Linux MLX 限制，见文末）
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

## 步骤⑤ `05_import_ollama.sh` —— 导入 Ollama

写一个 `Modelfile`（`FROM ./out/hf`），然后：

```bash
ollama create --experimental my-gpt -f Modelfile   # 直接读 safetensors，自动转 GGUF
ollama list                                        # 出现 my-gpt（43 MB）
ollama run my-gpt "ROMEO:"                         # 生成
```

**结论**：`ollama create` 成功读懂了我们的 `model.safetensors` + `config.json` + `vocab.json`，把 76 个 tensor 导入成 `my-gpt`（出现在 `ollama list`）——这证明了「迁移到 HF → Ollama 使用」的链路是通的。

**⚠️ 已知障碍（环境限制，非迁移问题）**：Ollama 0.33.2 的 `--experimental` safetensors 导入在 **Linux 上依赖 MLX**（Apple 的机器学习框架，macOS 专用），运行/量化时报 `MLX not available`。两条绕过方向：

1. **手动写 GGUF**（用 `pip install gguf` 的 `GGUFWriter`，绕开 Ollama 的 MLX 量化器）——标准 GGUF 走 llama.cpp runner，本机 `qwen2.5:3b` 就是 GGUF 能正常跑。这是最干净的下一步。
2. **降级/换 Ollama 版本**，或换用 llama.cpp 直接跑。

另外一个小坑：`ollama create` 后若 `ollama list` 看不到新模型，是**旧 ollama server 没刷新**，重启 server 即可（`ollama stop` 在新版是停模型，需 kill 进程或用新端口 `OLLAMA_HOST=127.0.0.1:11435 ollama serve`）。

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
