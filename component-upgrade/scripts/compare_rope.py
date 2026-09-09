"""回合 01 对比：learned wpe vs RoPE。

在相同数据 / 相同种子下各训练一个只有位置编码不同的小 GPT，对比 loss 曲线，
并演示 RoPE 的长度外推能力（训练 block_size=64，测试 T=128）。

运行：python scripts/compare_rope.py [--steps N] [--block-size B] [--batch-size M] [--data PATH]
"""
import argparse
import os
import sys

import torch

# 让脚本能从 scripts/ 直接 import src/ 里的 model（scripts/ 与 src/ 同层）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, GPTConfig


# ---------- 数据 ----------

def load_text(path):
    """读训练文本；路径不存在时回退到合成数据（保证脚本可独立跑通）。

    注意：合成数据是 i.i.d. 随机字符，没有语言结构，loss 会停在 ~ln(vocab) 附近；
    只有真实语料（如 TinyShakespeare）才能体现「学到规律 → loss 下降」的对比。
    """
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


def train_one(pos_enc, data, config, steps, batch_size, device):
    """训练一个指定位置编码的小 GPT，返回 (model, [(step, train_loss, val_loss), ...])。"""
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
            print("    [{:7s}] step {:4d}: train {:.4f}  val {:.4f}".format(
                pos_enc, step, loss.item(), val))
    return model, history


# ---------- 长度外推 ----------

def demo_extrapolation(learned, rope, vocab_size, block_size, device):
    """训练长度 block_size，用 T = 2*block_size 的输入测试两个模型。"""
    print("\n=== 长度外推演示（训练 block_size={}，测试 T={}）===".format(block_size, 2 * block_size))
    T = 2 * block_size
    x = torch.randint(0, vocab_size, (1, T), device=device)

    rope.eval()
    with torch.no_grad():
        out = rope(x)
    print("  ✅ RoPE   : 前向 OK，logits shape = {}（位置由旋转给出，不依赖 block_size）".format(tuple(out.shape)))

    learned.eval()
    try:
        with torch.no_grad():
            learned(x)
        print("  ⚠️ learned: 居然没报错（不应发生，说明越界检查失效）")
    except (AssertionError, IndexError):
        print("  ❌ learned: wpe 越界（wpe 只有 {} 行，编码不了位置 >= {}）".format(block_size, block_size))


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

    base_cfg = dict(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=4, n_head=4, n_embd=128)
    learned_cfg = GPTConfig(**base_cfg, pos_enc="learned")
    rope_cfg = GPTConfig(**base_cfg, pos_enc="rope")

    def n_params(cfg):
        return sum(p.numel() for p in GPT(cfg).parameters())

    n_learned = n_params(learned_cfg)
    n_rope = n_params(rope_cfg)

    print("=" * 62)
    print("回合 01：learned wpe vs RoPE")
    print("设备: {}  数据: {}".format(device, data_path if os.path.exists(data_path) else "（合成）"))
    print("vocab={}  block_size={}  batch={}  steps={}".format(
        vocab_size, args.block_size, args.batch_size, args.steps))
    print("参数量: learned={}  rope={}  （RoPE 省 {} = wpe）".format(n_learned, n_rope, n_learned - n_rope))
    print("=" * 62)

    print("\n[训练 learned wpe]")
    learned, hist_learned = train_one("learned", data, learned_cfg, args.steps, args.batch_size, device)
    print("\n[训练 rope]")
    rope, hist_rope = train_one("rope", data, rope_cfg, args.steps, args.batch_size, device)

    print("\n=== loss 对比（step: learned_train/val  vs  rope_train/val）===")
    print("{:>6} {:>12} {:>12} {:>12} {:>12}".format("step", "learned_tr", "learned_va", "rope_tr", "rope_va"))
    for (s1, ltr, lva), (s2, rtr, rva) in zip(hist_learned, hist_rope):
        print("{:6d} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(s1, ltr, lva, rtr, rva))

    demo_extrapolation(learned, rope, vocab_size, args.block_size, device)

    print("\n结论:")
    print("  - RoPE 用更少参数（省掉 wpe 的 block_size*n_embd 个参数）达到与 learned 相当的 loss。")
    print("  - RoPE 天然支持任意长度，learned wpe 被训练长度锁死（长度外推）。")


if __name__ == "__main__":
    main()
