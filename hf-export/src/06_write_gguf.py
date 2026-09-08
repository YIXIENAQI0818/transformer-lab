"""步骤 6：把 ckpt 直接写成 GGUF（单文件、自描述、Ollama 可读）。

为什么绕开 Ollama 的 --experimental：
  Ollama 0.33.2 直接读 safetensors 在 Linux 上走 MLX（macOS 专用）报错；
  而标准 GGUF 走 llama.cpp runner（本机 qwen2.5:3b 就是 GGUF，能跑）。

为什么从 ckpt 直接写，而不是从 HF 写：
  GGUF 的 tensor 布局和我们的原始 nn.Linear 一致（都是 (out,in)），
  而 HF 的 Conv1D 是转置的 (in,out)。所以直接读 ckpt（Linear 布局）写 GGUF，不用转置。

对照「数据 vs 代码」：
  - 数据：权重 -> GGUF tensor；词表 -> tokenizer.ggml.tokens
  - 结构：config -> 元数据字段（gpt2.block_count 等）
  - 代码：靠 general.architecture="gpt2" 让 llama.cpp 查表（结构代码在 llama.cpp 里）

运行：python src/06_write_gguf.py
"""
import os

import numpy as np
import torch
from gguf import GGUFWriter
from gguf.constants import GGUFValueType

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # hf-export/src
PROJ_DIR = os.path.dirname(SRC_DIR)                           # hf-export
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(LAB_DIR, "pretraining", "out", "ckpt.pt")
OUT_GGUF = os.path.join(PROJ_DIR, "out", "gguf")
GGUF_PATH = os.path.join(OUT_GGUF, "model.gguf")


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model"]          # 我们的 state_dict（Linear 布局）
    cfg = ckpt["config"]        # 超参
    meta = ckpt["meta"]         # 词表 stoi/itos

    n_layer = cfg["n_layer"]
    n_embd = cfg["n_embd"]
    vocab = cfg["vocab_size"]

    os.makedirs(OUT_GGUF, exist_ok=True)
    # GGUFWriter(path, "gpt2") 会在 __init__ 里自动写 general.architecture="gpt2"
    writer = GGUFWriter(GGUF_PATH, "gpt2")

    # ---- 结构元数据（config -> GGUF 字段）----
    writer.add_context_length(cfg["block_size"])          # gpt2.context_length = 256
    writer.add_embedding_length(n_embd)                   # gpt2.embedding_length = 384
    writer.add_block_count(n_layer)                       # gpt2.block_count = 6
    writer.add_head_count(cfg["n_head"])                  # gpt2.attention.head_count = 6
    writer.add_head_count_kv(cfg["n_head"])               # 无 GQA，KV 头数 = 头数
    writer.add_layer_norm_eps(1e-5)                       # LayerNorm eps（GPT-2 默认）
    writer.add_feed_forward_length(4 * n_embd)            # gpt2.feed_forward_length = 1536

    # ---- tokenizer（词表 -> tokenizer.ggml.*）----
    writer.add_tokenizer_model("gpt2")                    # char-level 用 gpt2 框架（merges 为空）
    # tokens 按 id 顺序排列（itos[i] 是 id=i 的字符）
    itos = {int(i): ch for i, ch in meta["itos"].items()}
    tokens = [itos[i] for i in range(vocab)]
    writer.add_token_list(tokens)
    # ⚠️ 已知障碍：char-level 分词「没有 BPE 合并规则」，merges 天然为空；
    #    但 GGUF 的 ARRAY 类型不支持空数组（gguf 库 raise ValueError），
    #    而不写 merges 字段，llama.cpp 加载时报 "cannot find tokenizer merges"。
    #    这是 llama.cpp 生态对 char-level 的根本限制（它只为 BPE/sentencepiece 设计）。
    #    要真正跑 Ollama，需在 pretraining 进阶阶段换成 BPE tokenizer（届时 merges 非空）。
    # 下面这行一旦执行会抛 ValueError，故注释掉；权重与结构部分（上面）是正确的。
    # writer.add_key_value("tokenizer.ggml.merges", [], GGUFValueType.ARRAY, GGUFValueType.STRING)

    # ---- 权重（state_dict -> GGUF tensor，Linear 布局直接对应，不转置）----
    def t(name, key):
        writer.add_tensor(name, sd[key].float().numpy())

    t("token_embd.weight", "transformer.wte.weight")
    t("position_embd.weight", "transformer.wpe.weight")
    t("output.weight", "lm_head.weight")                   # 与 wte 共享（weight tying）
    t("output_norm.weight", "transformer.ln_f.weight")
    t("output_norm.bias", "transformer.ln_f.bias")
    for i in range(n_layer):
        p = f"transformer.h.{i}."
        b = f"blk.{i}."
        t(f"{b}attn_norm.weight", f"{p}ln_1.weight")
        t(f"{b}attn_norm.bias", f"{p}ln_1.bias")
        t(f"{b}attn_qkv.weight", f"{p}attn.c_attn.weight")    # (1152,384) 直接对应
        t(f"{b}attn_qkv.bias", f"{p}attn.c_attn.bias")
        t(f"{b}attn_output.weight", f"{p}attn.c_proj.weight")  # (384,384)
        t(f"{b}attn_output.bias", f"{p}attn.c_proj.bias")
        t(f"{b}ffn_norm.weight", f"{p}ln_2.weight")
        t(f"{b}ffn_norm.bias", f"{p}ln_2.bias")
        t(f"{b}ffn_up.weight", f"{p}mlp.c_fc.weight")          # (1536,384)
        t(f"{b}ffn_up.bias", f"{p}mlp.c_fc.bias")
        t(f"{b}ffn_down.weight", f"{p}mlp.c_proj.weight")      # (384,1536)
        t(f"{b}ffn_down.bias", f"{p}mlp.c_proj.bias")

    # ---- 写文件：header -> kv -> tensor（write_tensors_to_file 内部会写 ti data）----
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()

    size_mb = os.path.getsize(GGUF_PATH) / 1e6
    print(f"已生成 GGUF -> {GGUF_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
