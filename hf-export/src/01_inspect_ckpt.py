"""步骤 1：看清 byte-bpe ckpt 里有什么。

ckpt 里存「数据」：model（state_dict）+ config（超参）+ iter。tokenizer 不在 ckpt 里，
单独存 byte-bpe/out/lib_tokenizer.json（tokenizers 库格式）。

「代码」不在 ckpt 里：模型结构在 byte-bpe/src/model.py（纯现代骨架），tokenizer 算法在
tokenizers 库。这正是迁移的起点——把数据按 HF 标准存好、代码对应到 HF 的实现。

运行：python src/01_inspect_ckpt.py
"""
import os

import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # hf-export/src
PROJ_DIR = os.path.dirname(SRC_DIR)                           # hf-export
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(LAB_DIR, "byte-bpe", "out", "ckpt.pt")


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    print("=" * 66)
    print("① ckpt 顶层 key")
    print("=" * 66)
    print("  ", list(ckpt.keys()))
    print("   （注意：没有 meta——tokenizer 单独存 byte-bpe/out/lib_tokenizer.json）")
    print()

    print("=" * 66)
    print("② config —— 纯现代骨架超参")
    print("=" * 66)
    for k, v in ckpt["config"].items():
        print(f"    {k:12s} = {v}")
    print()

    print("=" * 66)
    print("③ model —— state_dict（无 wpe / 无 attn.bias / RMSNorm 无 bias / SwiGLU 三投影）")
    print("=" * 66)
    sd = ckpt["model"]
    total = 0
    for k, v in sd.items():
        total += v.numel()
        # 只打印首层 + 尾部，避免 75 行太长
        if "h.0." in k or k in ("transformer.wte.weight", "transformer.ln_f.weight", "lm_head.weight"):
            print(f"    {k:45s} shape={str(tuple(v.shape)):20s} numel={v.numel():>9,}")
    print(f"    ...（共 {len(sd)} 个 key，每层 12 个 × 6 层 + wte + ln_f + lm_head）")
    print(f"    合计 {total:,} 参数，fp32 下 {total * 4 / 1e6:.1f} MB")
    print()

    print(f"训练步数 iter = {ckpt['iter']}")


if __name__ == "__main__":
    main()
