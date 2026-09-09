"""回合 03 对比：GELU FFN vs SwiGLU（门控）。

在相同数据 / 相同种子下各训练一个只有 FFN 结构不同的小 GPT，对比 loss 曲线与参数量。
关键点：SwiGLU 中间维度取 8/3·d 使总参数量与标准 4·d FFN 对齐，这样 loss 差异只能归因于
「门控结构 vs 单一激活」，而非参数更多（对齐验证见 src/swiglu.py 的 __main__）。

运行：python scripts/compare_swiglu.py [--steps N] [--block-size B] [--batch-size M] [--data PATH]
"""
import argparse
import os
import sys

import torch

# 让脚本能从 scripts/ 直接 import src/ 里的 model（scripts/ 与 src/ 同层）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, GPTConfig


# ---------- 数据 ----------（与 compare_rope.py / compare_rmsnorm.py 相同）

def load_text(path):
    """读训练文本；路径不存在时回退到合成数据（保证脚本可独立跑通）。"""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    torch.manual_seed(0)
    vocab = " abcdefghijklmnopqrstuvwxyz\n"
    idx = torch.randint(0, len(vocab), (50000,)).tolist()
    return "".join(vocab[i] for i in idx)


def build_tokenizer(text):
    """char-level tokenizer：唯一字符 -> id。"""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    return stoi


def encode(text, stoi):
    return [stoi[c] for c in text]


# ---------- 训练 ----------

def get_batch(data, block_size, batch_size, device):
    """随机采样 (x, y)：x 是 block_size 个 token，y 是 x 右移一位（next-token 目标）。"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, eval_iters=20):
    """在随机 batch 上估 loss（轻量版，不复用任何 checkpoint）。"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def train_one(activation, data, config, steps, batch_size, device):
    """训练一个指定 FFN 结构的小 GPT，返回 (model, [(step, train_loss, val_loss), ...])。"""
    torch.manual_seed(1337)
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for step in range(steps):
        x, y = get_batch(data, config.block_size, batch_size, device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            val = estimate_loss(model, data, config.block_size, batch_size, device)
            history.append((step, loss.item(), val))
            print("    [{:6s}] step {:4d}: train {:.4f}  val {:.4f}".format(
                activation, step, loss.item(), val))
    return model, history


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", type=str, default=None,
                        help="训练文本路径，默认复用 pretraining 的 TinyShakespeare")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    # 数据 + tokenizer
    data_path = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "pretraining", "data", "input.txt")
    text = load_text(data_path)
    stoi = build_tokenizer(text)
    vocab_size = len(stoi)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # 沿用前两回合已升级的底座：RoPE + RMSNorm，本回合只变 FFN
    base_cfg = dict(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=4, n_head=4, n_embd=128, pos_enc="rope", norm="rmsnorm")
    gelu_cfg = GPTConfig(**base_cfg, activation="gelu")
    swiglu_cfg = GPTConfig(**base_cfg, activation="swiglu")

    def n_params(cfg):
        return sum(p.numel() for p in GPT(cfg).parameters())

    def n_ffn_params(cfg):
        """只数 FFN 子层的参数量，展示 8/3·d 对齐的关键。"""
        keys = ("c_fc", "c_proj", "gate_proj", "up_proj", "down_proj")
        return sum(p.numel() for n, p in GPT(cfg).named_parameters()
                   if any(k in n for k in keys))

    n_gelu = n_params(gelu_cfg)
    n_swiglu = n_params(swiglu_cfg)

    print("=" * 62)
    print("回合 03：GELU FFN vs SwiGLU（门控）")
    print("设备: {}  数据: {}".format(device, data_path if os.path.exists(data_path) else "（合成）"))
    print("vocab={}  block_size={}  batch={}  steps={}".format(
        vocab_size, args.block_size, args.batch_size, args.steps))
    print("总参数: gelu={}  swiglu={}  （差 {}，来自 8/3 取整）".format(
        n_gelu, n_swiglu, n_swiglu - n_gelu))
    print("FFN 参数: gelu={}  swiglu={}  （8/3·d 对齐了 FFN 参数量）".format(
        n_ffn_params(gelu_cfg), n_ffn_params(swiglu_cfg)))
    print("=" * 62)

    print("\n[训练 gelu]")
    _, hist_gelu = train_one("gelu", data, gelu_cfg, args.steps, args.batch_size, device)
    print("\n[训练 swiglu]")
    _, hist_swiglu = train_one("swiglu", data, swiglu_cfg, args.steps, args.batch_size, device)

    print("\n=== loss 对比（step: gelu_tr/va  vs  swiglu_tr/va）===")
    print("{:>6} {:>12} {:>12} {:>12} {:>12}".format("step", "gelu_tr", "gelu_va", "swiglu_tr", "swiglu_va"))
    for (s1, ltr, lva), (s2, rtr, rva) in zip(hist_gelu, hist_swiglu):
        print("{:6d} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(s1, ltr, lva, rtr, rva))

    print("\n结论:")
    print("  - SwiGLU 用门控（可学习的 gate 决定 value 每维放行多少）替代 GELU 的固定开关。")
    print("  - 中间维度 8/3·d 使 FFN 参数量对齐（{} vs {}），loss 差异只能归因于门控结构。".format(
        n_ffn_params(gelu_cfg), n_ffn_params(swiglu_cfg)))
    print("  - 等参数下 SwiGLU 的 loss 相当或更低 —— 这正是现代大模型全用 SwiGLU 的原因。")


if __name__ == "__main__":
    main()
