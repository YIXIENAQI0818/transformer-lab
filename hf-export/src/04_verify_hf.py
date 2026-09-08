"""步骤 4：验证迁移正确——HF 版模型 logits 与原始模型逐元素一致。

迁移最关键的问题是：数据（权重）搬过去之后，结构（代码）套对了没有？
严谨的验证方法：喂同一段输入，比较两个模型的 logits，应该逐元素一致（误差 < 1e-5）。

  - 原始模型：pretraining/src/model.py 的 GPT（我们手写的结构代码）
  - HF 版模型：AutoModelForCausalLM.from_pretrained(out/hf) 的 GPT2LMHeadModel（transformers 的结构代码）

如果两者 logits 一致，说明：
  ① 权重迁移没丢、没转置错（Conv1D 转置正确）
  ② 结构代码两边等价（激活函数 gelu 对齐、attention 数学一致）
  ③ 「数据 + 代码」在 HF 侧正确组合了

运行：python src/04_verify_hf.py
"""
import os
import sys

import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # hf-export/src
PROJ_DIR = os.path.dirname(SRC_DIR)                           # hf-export
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(LAB_DIR, "pretraining", "out", "ckpt.pt")
OUT_HF = os.path.join(PROJ_DIR, "out", "hf")

# 让本脚本能 import pretraining 的原始模型（它不在包结构里）
sys.path.insert(0, os.path.join(LAB_DIR, "pretraining", "src"))
from model import GPT, GPTConfig                              # noqa: E402  原始模型

from transformers import AutoModelForCausalLM, AutoTokenizer   # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1. 原始模型：我们的 GPT + ckpt 权重 ----
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    orig = GPT(GPTConfig(**ckpt["config"]))
    orig.load_state_dict(ckpt["model"])     # 严格加载（含 attn.bias buffer）
    orig.eval().to(device)

    # ---- 2. HF 版模型：AutoModelForCausalLM 加载标准 GPT2LMHeadModel ----
    hf_model = AutoModelForCausalLM.from_pretrained(OUT_HF).to(device)
    hf_model.eval()

    # ---- 3. tokenizer：AutoTokenizer + trust_remote_code（因为 char-level 是自定义类）----
    tok = AutoTokenizer.from_pretrained(OUT_HF, trust_remote_code=True)

    # ---- 4. 同一段输入，两个模型各前向一次 ----
    text = "ROMEO:\nWherefore art thou Romeo?"
    ids = tok.encode(text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    print(f"输入文本: {text!r}")
    print(f"token ids: {ids}")
    print()

    with torch.no_grad():
        orig_logits = orig(input_ids)                # 原始 GPT 直接返回 logits
        hf_logits = hf_model(input_ids).logits       # GPT2LMHeadModel 返回 CausalLMOutput，取 .logits

    print(f"原始模型 logits shape: {tuple(orig_logits.shape)}")
    print(f"HF 模型  logits shape: {tuple(hf_logits.shape)}")
    assert orig_logits.shape == hf_logits.shape

    # ---- 5. 逐元素对比 ----
    diff = (orig_logits - hf_logits).abs()
    print(f"\nlogits 最大绝对误差: {diff.max().item():.2e}")
    print(f"logits 平均绝对误差: {diff.mean().item():.2e}")
    assert diff.max().item() < 1e-5, "logits 不一致！迁移有错（转置/激活函数/权重丢失）"
    print("✅ logits 逐元素一致 —— 权重迁移 + 结构对齐都正确")

    # ---- 6. 用 HF 模型生成一段，确认「标准格式」真的能跑 ----
    print("\n" + "=" * 60)
    print("HF 模型（GPT2LMHeadModel）生成示例：")
    print("=" * 60)
    torch.manual_seed(42)
    gen_ids = hf_model.generate(input_ids, max_new_tokens=120, do_sample=True, temperature=0.8)
    print(tok.decode(gen_ids[0].tolist()))


if __name__ == "__main__":
    main()
