"""步骤 3：把 char-level tokenizer 迁移成 HF 标准 tokenizer。

回顾核心概念：tokenizer 也分「数据」和「代码」两半。
  - 数据：stoi / itos（char <-> id 的映射表，65 条）
  - 代码：encode / decode 逻辑（怎么查表、怎么拼字符串）

迁移到 HF，就是把这两半分别安放：
  - 数据 → vocab.json                 （由 HFCharTokenizer.save_vocabulary 写入，JSON 正确转义换行符）
  - 代码 → HFCharTokenizer 类          （继承 PreTrainedTokenizer，见 hf_char_tokenizer.py）

产出的文件：
  vocab.json               —— 数据：{char: id}
  tokenizer_config.json    —— 配置：tokenizer_class、auto_map 等
  special_tokens_map.json  —— 特殊 token（char-level 无 bos/eos/unk）
  hf_char_tokenizer.py     —— 代码：类定义（复制过来，供 AutoTokenizer trust_remote_code 加载）

运行：python src/03_convert_tokenizer.py
"""
import json
import os
import shutil

import torch

from hf_char_tokenizer import HFCharTokenizer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # hf-export/src
PROJ_DIR = os.path.dirname(SRC_DIR)                           # hf-export
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(LAB_DIR, "pretraining", "out", "ckpt.pt")
OUT_HF = os.path.join(PROJ_DIR, "out", "hf")


def main():
    # 1. 读 ckpt 里的 meta（tokenizer 的「数据」）
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    stoi = ckpt["meta"]["stoi"]
    print(f"① 从 ckpt meta 读到词表：{len(stoi)} 个 char -> id 映射")

    # 2. 用「数据」(stoi) + 「代码」(HFCharTokenizer 类) 构建 HF tokenizer
    tok = HFCharTokenizer(vocab=stoi)
    print(f"② 构建 HFCharTokenizer，vocab_size = {tok.vocab_size}")

    # 3. 保存：vocab.json + tokenizer_config.json + special_tokens_map.json
    tok.save_pretrained(OUT_HF)
    print(f"③ 已保存 vocab.json / tokenizer_config.json -> {OUT_HF}")

    # 4. 把「代码」（类定义文件）复制进模型目录，并设置 auto_map，
    #    这样 AutoTokenizer.from_pretrained(dir, trust_remote_code=True) 才能找到类
    shutil.copy(os.path.join(SRC_DIR, "hf_char_tokenizer.py"), OUT_HF)
    cfg_path = os.path.join(OUT_HF, "tokenizer_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    # auto_map 格式：[slow_tokenizer_ref, fast_tokenizer_ref]；ref 是 "module.ClassName"（不含 .py）
    cfg["auto_map"] = {"AutoTokenizer": ["hf_char_tokenizer.HFCharTokenizer", None]}
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("④ 已设置 auto_map（AutoTokenizer -> hf_char_tokenizer.py:HFCharTokenizer）")

    # 5. 自检 encode/decode 往返
    s = "ROMEO:\n"
    ids = tok.encode(s)
    dec = tok.decode(ids)
    print(f"\n⑤ 自检 encode/decode 往返:")
    print(f"   {s!r} -> {ids} -> {dec!r}")
    assert dec == s, "decode(encode(s)) 应还原 s"

    print(f"\n最终模型目录文件: {sorted(os.listdir(OUT_HF))}")


if __name__ == "__main__":
    main()
