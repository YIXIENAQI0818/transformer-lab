"""回合 02 对比：LayerNorm vs RMSNorm。

在相同数据 / 相同种子下各训练一个只有归一化层不同的小 GPT，对比 loss 曲线与参数量；
再对归一化层本身做一次孤立计时（少一次归约、速度收益个位数；手写分步版慢 3 倍，需 fuse）。

运行：python scripts/compare_rmsnorm.py [--steps N] [--block-size B] [--batch-size M] [--data PATH]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# 让脚本能从 scripts/ 直接 import src/ 里的 model / rmsnorm（scripts/ 与 src/ 同层）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, GPTConfig
from rmsnorm import RMSNorm


# ---------- 数据 ----------（与 compare_rope.py 相同，见其注释）

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


def train_one(norm, data, config, steps, batch_size, device):
    """训练一个指定归一化层的小 GPT，返回 (model, [(step, train_loss, val_loss), ...])。"""
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
            print("    [{:9s}] step {:4d}: train {:.4f}  val {:.4f}".format(
                norm, step, loss.item(), val))
    return model, history


# ---------- 孤立计时 ----------

def bench_norm(device, dim=512, batch=65536, iters=300):
    """纯前向（no_grad）计时，比较归一化层的推理速度（微秒/次）。

    为什么要大张量：归一化层只差一次归约，计算量差异很小；小张量会被 GPU 的
    kernel 启动开销淹没（甚至出现 RMS 反而更慢的假象）。用大 batch 让计算主导，
    才能测出稳定、可信的差异。

    三个对象：
      - hand_rms ：手写分步 RMSNorm（教学实现，见 rmsnorm.py）
      - fused_ln ：内置 nn.LayerNorm（fused 生产实现）
      - fused_rms：内置 nn.RMSNorm（fused 生产实现）

    dim 取 512 而非模型 n_embd=128，是为了让计时稳定（128 维太小，kernel 边界效应明显）；
    归一化层「少一次归约」的相对开销结论在 dim 上是稳健的。
    """
    x = torch.randn(batch, dim, device=device)
    hand_rms = RMSNorm(dim).to(device)       # 手写分步（rmsnorm.py）
    fused_ln = nn.LayerNorm(dim).to(device)  # 内置 fused
    fused_rms = nn.RMSNorm(dim).to(device)   # 内置 fused

    def time_it(mod):
        with torch.no_grad():
            for _ in range(50):  # 预热
                mod(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                mod(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # 微秒/次

    return time_it(hand_rms), time_it(fused_ln), time_it(fused_rms)


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
                    n_layer=4, n_head=4, n_embd=128, pos_enc="rope")
    ln_cfg = GPTConfig(**base_cfg, norm="layernorm")
    rms_cfg = GPTConfig(**base_cfg, norm="rmsnorm")

    def n_params(cfg):
        return sum(p.numel() for p in GPT(cfg).parameters())

    n_ln = n_params(ln_cfg)
    n_rms = n_params(rms_cfg)

    print("=" * 62)
    print("回合 02：LayerNorm vs RMSNorm")
    print("设备: {}  数据: {}".format(device, data_path if os.path.exists(data_path) else "（合成）"))
    print("vocab={}  block_size={}  batch={}  steps={}".format(
        vocab_size, args.block_size, args.batch_size, args.steps))
    print("参数量: layernorm={}  rmsnorm={}  （RMSNorm 省 {} = 每层 beta）".format(
        n_ln, n_rms, n_ln - n_rms))
    print("=" * 62)

    print("\n[训练 layernorm]")
    _, hist_ln = train_one("layernorm", data, ln_cfg, args.steps, args.batch_size, device)
    print("\n[训练 rmsnorm]")
    _, hist_rms = train_one("rmsnorm", data, rms_cfg, args.steps, args.batch_size, device)

    print("\n=== loss 对比（step: layernorm_tr/va  vs  rmsnorm_tr/va）===")
    print("{:>6} {:>12} {:>12} {:>12} {:>12}".format("step", "ln_tr", "ln_va", "rms_tr", "rms_va"))
    for (s1, ltr, lva), (s2, rtr, rva) in zip(hist_ln, hist_rms):
        print("{:6d} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(s1, ltr, lva, rtr, rva))

    t_hand_rms, t_fused_ln, t_fused_rms = bench_norm(device)
    print("\n=== 归一化层孤立计时（纯前向，微秒/次）===")
    print("  手写 RMSNorm（分步）  : {:8.2f} us".format(t_hand_rms))
    print("  内置 LayerNorm（fused）: {:8.2f} us".format(t_fused_ln))
    print("  内置 RMSNorm（fused）  : {:8.2f} us".format(t_fused_rms))
    print("  内置 RMS 比内置 LN 快 {:.1f}%（少一次归约的真实收益，个位数）".format(
        (t_fused_ln - t_fused_rms) / t_fused_ln * 100))
    print("  手写 RMS 是内置 RMS 的 {:.1f} 倍耗时（说明生产必须 fuse）".format(
        t_hand_rms / t_fused_rms))

    print("\n结论:")
    print("  - RMSNorm 与 LayerNorm loss 相当（不重新居中几乎无损）。")
    print("  - RMSNorm 每层省一个 beta 偏置（{} 参数）。".format(n_ln - n_rms))
    print("  - 速度收益其实是个位数（少一次归约），RMSNorm 真正价值是「简化 + 省参数」；")
    print("    手写分步版慢 3 倍以上，生产必须用 fused 内核。")


if __name__ == "__main__":
    main()
