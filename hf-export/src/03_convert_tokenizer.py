"""步骤 3：把 byte-bpe 的 BPE tokenizer 迁成 HF 标准格式。

关键差异 vs char 时代：char tokenizer 不是标准，要写自定义 PreTrainedTokenizer + vocab.json；
而 byte-bpe 用的是 tokenizers 库的 ByteLevel BPE（HF 原生格式），tokenizer 已经存成
lib_tokenizer.json，这里只需用 PreTrainedTokenizerFast 包装后 save_pretrained，
产出 tokenizer.json + tokenizer_config.json。这是「工业库 tokenizer」比「自写 char」省事的地方。

运行：python src/03_convert_tokenizer.py
"""
import os

from transformers import PreTrainedTokenizerFast

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(SRC_DIR)
LAB_DIR = os.path.dirname(PROJ_DIR)
LIB_TOK_PATH = os.path.join(LAB_DIR, "byte-bpe", "out", "lib_tokenizer.json")
OUT_HF = os.path.join(PROJ_DIR, "out", "hf")


def main():
    # 用 tokenizer_file 直接加载 byte-bpe 训好的库版 BPE tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=LIB_TOK_PATH)
    print(f"① 加载 byte-bpe 的 BPE tokenizer：vocab_size = {tok.vocab_size}")

    # save_pretrained 产出 tokenizer.json + tokenizer_config.json + special_tokens_map.json
    tok.save_pretrained(OUT_HF)
    print(f"② 已保存 tokenizer 文件 -> {OUT_HF}")

    # 自检：encode/decode 往返
    s = "First Citizen:\nBefore we proceed any further."
    ids = tok.encode(s)
    dec = tok.decode(ids)
    print(f"\n③ 自检 encode/decode 往返:")
    print(f"   {s[:40]!r}... -> {ids[:12]}... -> {dec[:40]!r}...")
    assert dec == s, "decode(encode(s)) 应还原 s"
    print("   ✔ 往返一致")


if __name__ == "__main__":
    main()
