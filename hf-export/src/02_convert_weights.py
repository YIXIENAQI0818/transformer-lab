"""步骤 2：把 byte-bpe ckpt 的权重迁移到 ModernGPTForCausalLM，存成 HF 标准格式。

核心结论：modeling_modern_gpt.py 是照 byte-bpe/src/model.py 写的（参数名完全一致），
且两边都用 nn.Linear（无 Conv1D）、无 attn.bias、RMSNorm 无 bias，所以权重迁移是
「直接 load_state_dict」，零转换。char 时代的 4 个坑（Conv1D 转置 / attn.bias 剔除 /
token 越界 / 激活对齐）都不存在了——代价是模型类要自己写（见 modeling_modern_gpt.py）。

运行：python src/02_convert_weights.py
"""
import os

import torch

from modeling_modern_gpt import ModernGPTConfig, ModernGPTForCausalLM

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # hf-export/src
PROJ_DIR = os.path.dirname(SRC_DIR)                           # hf-export
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(LAB_DIR, "byte-bpe", "out", "ckpt.pt")
OUT_HF = os.path.join(PROJ_DIR, "out", "hf")


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model"]          # 我们的 state_dict（75 个 key）
    cfg = ckpt["config"]        # 我们的超参（字段名与 ModernGPTConfig 完全一致）

    # ---- 构建自定义配置 + 模型（字段直接对齐，无需 char 时代的字段名映射）----
    config = ModernGPTConfig(**cfg)
    model = ModernGPTForCausalLM(config)

    # ---- 直接加载：参数名完全一致，零转换 ----
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing（模型缺但没提供）  :", missing)
    print("unexpected（我们多出模型没有）:", unexpected)
    assert not missing and not unexpected, "key 没对齐，检查 modeling_modern_gpt.py 的参数名"

    # ---- 保存为标准 HF 文件：config.json + model.safetensors ----
    os.makedirs(OUT_HF, exist_ok=True)
    model.save_pretrained(OUT_HF, safe_serialization=True)
    print(f"\n已保存 -> {OUT_HF}")
    print("  文件:", sorted(os.listdir(OUT_HF)))


if __name__ == "__main__":
    main()
