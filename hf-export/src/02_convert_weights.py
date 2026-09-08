"""步骤 2：把 ckpt 的权重迁移到 GPT2LMHeadModel，存成 HF 标准格式。

这是整个迁移里最核心、最容易出错的一步。先给结论：

    我们的模型结构 == 标准 GPT-2（因为阶段 1 就是照 nanoGPT/GPT-2 写的），
    所以「结构代码」可以直接复用 transformers 的 GPT2LMHeadModel，不用自己写 modeling 脚本。
    这就是上一问「配套」的落地：千问的结构代码在 transformers 里（modeling_qwen2.py），
    GPT-2 的也在（modeling_gpt2.py），而我们的结构恰好就是 GPT-2。

但「结构一样」不等于「权重能直接拷」。有 4 个必须处理的坑：

  ① Conv1D vs Linear 转置
     我们的 nn.Linear(in, out) 权重存成 (out, in)；
     transformers 的 GPT-2 用 Conv1D(in, out)，权重反着存成 (in, out)。
     → c_attn / c_proj(attn) / c_fc / c_proj(mlp) 的 weight 要 .t() 转置（共 6 层 × 4 = 24 个），bias 不用。

  ② attn.bias 剔除
     这是 register_buffer 存的 causal mask（下三角），GPT-2 不存它，而是在前向时用
     attention_mask 动态生成，所以 state_dict 里没有这个 key，要剔除。

  ③ token id 越界
     GPT2Config 默认 bos_token_id / eos_token_id = 50256（GPT-2 的 50257 词表），
     但我们 vocab=65，50256 越界会报警告。置 None 消除。

  ④ 激活函数对齐
     nanoGPT 的 MLP 用 nn.GELU()（精确 erf 版）；transformers GPT-2 默认 activation_function
     是 "gelu_new"（tanh 近似版，GPT-2 原论文用的）。两者数值不同，不改成 "gelu" 会导致
     验证时 logits 对不上。这一步是把「结构细节」也严格对齐。

运行：python src/02_convert_weights.py
"""
import os

import torch
from transformers import GPT2Config, GPT2LMHeadModel

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # hf-export/src
PROJ_DIR = os.path.dirname(SRC_DIR)                           # hf-export
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(LAB_DIR, "pretraining", "out", "ckpt.pt")
OUT_HF = os.path.join(PROJ_DIR, "out", "hf")

# 需要转置的权重后缀：这些层在我们这边是 nn.Linear，GPT-2 那边是 Conv1D
TRANSPOSE_SUFFIX = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model"]          # 我们的 state_dict
    cfg = ckpt["config"]        # 我们的超参

    # ---- 构建 GPT-2 配置：字段名映射（block_size -> n_positions）+ 4 个坑的处理 ----
    gpt2_cfg = GPT2Config(
        vocab_size=cfg["vocab_size"],       # 65
        n_positions=cfg["block_size"],      # 256（我们的 block_size = GPT-2 的 n_positions）
        n_embd=cfg["n_embd"],               # 384
        n_layer=cfg["n_layer"],             # 6
        n_head=cfg["n_head"],               # 6
        activation_function="gelu",         # 坑④：对齐 nanoGPT 的精确 GELU（默认是 gelu_new）
        bos_token_id=None,                  # 坑③：避免 50256 越界
        eos_token_id=None,
    )
    gpt2 = GPT2LMHeadModel(gpt2_cfg)

    # ---- 逐 key 迁移：转置 + 剔除 ----
    new_sd = {}
    skipped = []
    print("逐 key 迁移（转置 / 剔除）：")
    for k, v in sd.items():
        if k.endswith(".attn.bias"):                     # 坑②：causal mask buffer（精确匹配，避免误伤 c_attn.bias）
            skipped.append(k)
            continue
        if k.endswith(TRANSPOSE_SUFFIX):                 # 坑①：Linear -> Conv1D 转置
            v = v.t().contiguous()
            print(f"    转置 {k:35s} -> {tuple(v.shape)}")
        new_sd[k] = v
    for k in skipped:
        print(f"    剔除 {k}  （causal mask buffer，GPT-2 用 attention_mask）")

    # ---- strict=False 拿 missing/unexpected，再断言必须都为空 ----
    missing, unexpected = gpt2.load_state_dict(new_sd, strict=False)
    print()
    print(f"GPT-2 缺但没提供 (missing)   : {missing}")
    print(f"我们多出 GPT-2 没有 (unexpected): {unexpected}")
    assert not missing and not unexpected, "key 没有完全对齐，检查转置/剔除逻辑"

    # ---- 数值抽检：转置后 [i,j] 应等于原 [j,i] ----
    k = "transformer.h.0.attn.c_attn.weight"
    orig = sd[k]
    moved = gpt2.state_dict()[k]
    print(f"\n数值抽检 {k}:")
    print(f"  原 Linear   权重[0,0]={orig[0, 0].item():.6f}  权重[10,5]={orig[10, 5].item():.6f}")
    print(f"  转置后 Conv1D 权重[0,0]={moved[0, 0].item():.6f}  权重[5,10]={moved[5, 10].item():.6f}")
    print(f"  （转置后 [i,j] == 原 [j,i]，故 [5,10] 应等于原 [10,5]）")

    # ---- 保存为标准 HF 文件：config.json + model.safetensors ----
    os.makedirs(OUT_HF, exist_ok=True)
    gpt2.save_pretrained(OUT_HF, safe_serialization=True)
    print(f"\n已保存 -> {OUT_HF}")
    print("  文件:", sorted(os.listdir(OUT_HF)))


if __name__ == "__main__":
    main()
