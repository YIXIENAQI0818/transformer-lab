"""步骤 4：验证迁移正确——HF 版模型 logits 与原始模型逐元素一致。

严谨验证：喂同一段输入，比较两个模型的 logits：
  - 原始模型：model-tokenizer/src/model.py 的 GPT（我们手写的纯现代骨架）
  - HF 版模型：ModernGPTForCausalLM（自定义 PreTrainedModel）

若 logits 一致，说明权重迁移没丢、结构代码两边等价（RoPE/RMSNorm/SwiGLU/GQA/Flash 全部对齐）。

运行：python src/04_verify_hf.py
"""
import os
import sys

import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # model-tokenizer/scripts-export
PROJ_DIR = os.path.dirname(SRC_DIR)                           # model-tokenizer
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(PROJ_DIR, "out", "train", "ckpt.pt")
OUT_HF = os.path.join(PROJ_DIR, "out", "export", "hf")

# import 原始模型（model-tokenizer/src/model.py）
sys.path.insert(0, os.path.join(PROJ_DIR, "src"))
from model import GPT, GPTConfig                              # noqa: E402  原始模型

from modeling_modern_gpt import ModernGPTForCausalLM          # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. 原始模型：model-tokenizer 的 GPT + ckpt 权重 ----
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    orig = GPT(GPTConfig(**ckpt["config"]))
    orig.load_state_dict(ckpt["model"])
    orig.eval().to(device)

    # ---- 2. HF 版模型：ModernGPTForCausalLM.from_pretrained ----
    hf_model = ModernGPTForCausalLM.from_pretrained(OUT_HF).to(device)
    hf_model.eval()

    # ---- 3. tokenizer：BPE 是标准，AutoTokenizer 直接加载（无需 trust_remote_code）----
    tok = AutoTokenizer.from_pretrained(OUT_HF)

    # ---- 4. 同一段输入，两个模型各前向一次 ----
    text = "First Citizen:\nBefore we proceed any further, hear me speak."
    ids = tok.encode(text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"输入文本: {text[:45]!r}...")
    print(f"token ids: {ids[:12]}...（共 {len(ids)} token）")
    print()

    with torch.no_grad():
        orig_logits = orig(input_ids)                 # 原始 GPT 直接返回 logits
        hf_logits = hf_model(input_ids).logits        # 取 CausalLMOutput 的 .logits

    print(f"原始模型 logits shape: {tuple(orig_logits.shape)}")
    print(f"HF 模型  logits shape: {tuple(hf_logits.shape)}")
    assert orig_logits.shape == hf_logits.shape

    diff = (orig_logits - hf_logits).abs()
    print(f"\nlogits 最大绝对误差: {diff.max().item():.2e}")
    print(f"logits 平均绝对误差: {diff.mean().item():.2e}")
    assert diff.max().item() < 1e-5, "logits 不一致！迁移有错"
    print("✅ logits 逐元素一致 —— 权重迁移 + 结构对齐都正确")

    # ---- 5. 用 HF 模型生成一段，确认标准格式真的能跑 ----
    print("\n" + "=" * 60)
    print("HF 模型（ModernGPTForCausalLM）生成示例：")
    print("=" * 60)
    torch.manual_seed(42)
    gen_ids = hf_model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.8)
    print(tok.decode(gen_ids[0].tolist()))


if __name__ == "__main__":
    main()
