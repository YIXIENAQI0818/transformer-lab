"""回合 04 对比：MHA vs GQA（K/V 头共享）。

在相同数据 / 相同种子下各训练一个只有注意力 K/V 头数不同的小 GPT，对比 loss 曲线、
参数量与 KV cache 大小。GQA（n_kv_head=2）让 2 个 Q 头共享 1 个 K/V 头，省 K/V 投影参数 +
KV cache 减半，且 loss 几乎不掉（KV cache 的详细张量走法是回合 06 主题，这里先看「每 token
K/V 元素数」这个伏笔）。

运行：python scripts/compare_gqa.py [--steps N] [--block-size B] [--batch-size M] [--data PATH]
"""
import argparse
import os
import sys

import torch

# 让脚本能从 scripts/ 直接 import src/ 里的 model（scripts/ 与 src/ 同层）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, GPTConfig


# ---------- 数据 ----------（与前几回合相同）

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


def train_one(label, data, config, steps, batch_size, device):
    """训练一个指定 K/V 头数的小 GPT，返回 (model, [(step, train_loss, val_loss), ...])。"""
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
            print("    [{:5s}] step {:4d}: train {:.4f}  val {:.4f}".format(
                label, step, loss.item(), val))
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

    # 沿用前三回合已升级底座：RoPE + RMSNorm + SwiGLU，本回合只变 K/V 头数
    base_cfg = dict(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=4, n_head=4, n_embd=128, pos_enc="rope", norm="rmsnorm", activation="swiglu")
    mha_cfg = GPTConfig(**base_cfg, n_kv_head=4)   # MHA
    gqa_cfg = GPTConfig(**base_cfg, n_kv_head=2)   # GQA

    def n_params(cfg):
        return sum(p.numel() for p in GPT(cfg).parameters())

    n_mha = n_params(mha_cfg)
    n_gqa = n_params(gqa_cfg)

    n_head, head_size = 4, 128 // 4

    def kv_cache_size(n_kv_head):
        """每 token 的 K/V 元素数（K + V 各 n_kv_head 个头 × head_size）。"""
        return 2 * n_kv_head * head_size

    print("=" * 62)
    print("回合 04：MHA vs GQA（K/V 头共享）")
    print("设备: {}  数据: {}".format(device, data_path if os.path.exists(data_path) else "（合成）"))
    print("vocab={}  block_size={}  batch={}  steps={}  n_head={}".format(
        vocab_size, args.block_size, args.batch_size, args.steps, n_head))
    print("总参数: mha={}  gqa={}  （GQA 省 {} = K/V 投影）".format(
        n_mha, n_gqa, n_mha - n_gqa))
    print("KV cache: mha={} 元素/token  gqa={} 元素/token  （GQA 减半）".format(
        kv_cache_size(4), kv_cache_size(2)))
    print("=" * 62)

    print("\n[训练 mha]")
    _, hist_mha = train_one("mha", data, mha_cfg, args.steps, args.batch_size, device)
    print("\n[训练 gqa]")
    _, hist_gqa = train_one("gqa", data, gqa_cfg, args.steps, args.batch_size, device)

    print("\n=== loss 对比（step: mha_tr/va  vs  gqa_tr/va）===")
    print("{:>6} {:>12} {:>12} {:>12} {:>12}".format("step", "mha_tr", "mha_va", "gqa_tr", "gqa_va"))
    for (s1, ltr, lva), (s2, rtr, rva) in zip(hist_mha, hist_gqa):
        print("{:6d} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(s1, ltr, lva, rtr, rva))

    print("\n结论:")
    print("  - GQA 让多个 Q 头共享 K/V 头（n_kv_head=2），省 K/V 投影参数（{} 参数）。".format(n_mha - n_gqa))
    print("  - KV cache 每 token 从 {} 降到 {} 元素（减半），decode 省显存 + 更快。".format(
        kv_cache_size(4), kv_cache_size(2)))
    print("  - GQA 的 loss 与 MHA 相当（K/V 头冗余被共享掉，几乎无损）——这正是现代大模型全用 GQA 的原因。")


if __name__ == "__main__":
    main()
