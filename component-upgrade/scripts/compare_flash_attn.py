"""回合 05 对比：naive attention vs FlashAttention。

在相同数据 / 相同种子下各训练一个只有注意力实现不同的小 GPT，对比 loss（**应该精确一致**，
因为 FlashAttention 是精确算法、不是近似）；再对注意力计算本身做孤立计时（手写 QKᵀ / 内置
fused sdpa / 手写 tiling 三者）和显存元素数对比，看清收益在哪。

关键教学点：FlashAttention 换的是「实现」不是「数学」，所以 loss 不变，收益全在显存（O(T²)→O(T)）
和速度（GPU 上 fused kernel 省掉 HBM 往返）。手写 Python 版 tiling 反而最慢——加速来自
CUDA fused kernel，而非「分块」这个想法本身。

运行：python scripts/compare_flash_attn.py [--steps N] [--block-size B] [--batch-size M] [--data PATH] [--bench-t T]
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

# 让脚本能从 scripts/ 直接 import src/ 里的 model / flash_attn（scripts/ 与 src/ 同层）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, GPTConfig
from flash_attn import flash_attention


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
    """训练一个指定注意力实现的小 GPT，返回 (model, [(step, train_loss, val_loss), ...])。"""
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


# ---------- 孤立计时 ----------

def bench_attn(device, B, H, T, d, iters=30):
    """纯前向（no_grad）计时，比较三种注意力实现的推理速度（毫秒/次）。

    三个对象（对应「为什么手写 tiling 不加速」这个教学点）：
      - naive    ：手写 QKᵀ + softmax，物化完整 T×T 矩阵
      - sdpa     ：F.scaled_dot_product_attention（内置 fused，底层即 FlashAttention）
      - hand_flash：flash_attn.py 手写的 tiling + online softmax（Python 双重循环）

    T 取大（默认 256）而非模型 block_size=64，才能让差异显现——小 T 下 kernel 启动开销
    淹没计算差异，会测出假结论。hand_flash 的 Python 循环在任意 T 下都最慢，这正好说明：
    「分块」这个想法本身不加速，加速来自 CUDA fused kernel 在 SRAM 内算完整个 block。
    """
    q = torch.randn(B, H, T, d, device=device)
    k = torch.randn(B, H, T, d, device=device)
    v = torch.randn(B, H, T, d, device=device)
    scale = 1.0 / math.sqrt(d)
    causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def naive(q, k, v):
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        return att @ v

    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def hand_flash(q, k, v):
        return flash_attention(q, k, v, block_size=64, causal=True)

    def time_it(fn):
        with torch.no_grad():
            for _ in range(10):  # 预热
                fn(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3  # 毫秒/次

    return time_it(naive), time_it(sdpa), time_it(hand_flash)


def bench_memory(device, B, H, T, d):
    """测量前向峰值显存（仅 CUDA；CPU 上返回 None，改用元素数估算）。"""
    if device.type != "cuda":
        return None, None
    q = torch.randn(B, H, T, d, device=device)
    k = torch.randn(B, H, T, d, device=device)
    v = torch.randn(B, H, T, d, device=device)
    scale = 1.0 / math.sqrt(d)
    causal_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    torch.cuda.reset_peak_memory_stats()
    att = (q @ k.transpose(-2, -1)) * scale
    att = att.masked_fill(causal_mask, float("-inf"))
    att = F.softmax(att, dim=-1)
    out = att @ v
    naive_peak = torch.cuda.max_memory_allocated()
    del att, out

    torch.cuda.reset_peak_memory_stats()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    flash_peak = torch.cuda.max_memory_allocated()
    return naive_peak, flash_peak


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data", type=str, default=None,
                        help="训练文本路径，默认复用 pretraining 的 TinyShakespeare")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bench-t", type=int, default=256,
                        help="孤立计时用的序列长度（默认 256，远大于模型 64 以显现差异）")
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

    # 沿用前四回合已升级底座：RoPE + RMSNorm + SwiGLU + GQA，本回合只变注意力实现
    base_cfg = dict(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=4, n_head=4, n_embd=128, pos_enc="rope", norm="rmsnorm",
                    activation="swiglu", n_kv_head=2)
    naive_cfg = GPTConfig(**base_cfg, attn_impl="naive")
    flash_cfg = GPTConfig(**base_cfg, attn_impl="flash")

    def n_params(cfg):
        return sum(p.numel() for p in GPT(cfg).parameters())

    n_naive = n_params(naive_cfg)
    n_flash = n_params(flash_cfg)

    print("=" * 62)
    print("回合 05：naive attention vs FlashAttention")
    print("设备: {}  数据: {}".format(device, data_path if os.path.exists(data_path) else "（合成）"))
    print("vocab={}  block_size={}  batch={}  steps={}".format(
        vocab_size, args.block_size, args.batch_size, args.steps))
    print("参数量: naive={}  flash={}  （应一致，FlashAttention 不改变参数）".format(n_naive, n_flash))
    print("=" * 62)

    print("\n[训练 naive]")
    _, hist_naive = train_one("naive", data, naive_cfg, args.steps, args.batch_size, device)
    print("\n[训练 flash]")
    _, hist_flash = train_one("flash", data, flash_cfg, args.steps, args.batch_size, device)

    print("\n=== loss 对比（step: naive_tr/va  vs  flash_tr/va）===")
    print("{:>6} {:>12} {:>12} {:>12} {:>12}".format("step", "naive_tr", "naive_va", "flash_tr", "flash_va"))
    for (s1, ltr, lva), (s2, rtr, rva) in zip(hist_naive, hist_flash):
        print("{:6d} {:12.4f} {:12.4f} {:12.4f} {:12.4f}".format(s1, ltr, lva, rtr, rva))
    final_naive, final_flash = hist_naive[-1][2], hist_flash[-1][2]
    print("最终 val loss 差 = {:.4f}（FlashAttention 是精确算法，期望 ~0）".format(
        final_flash - final_naive))

    # 孤立计时
    B, H, T, d = 1, 4, args.bench_t, 32
    t_naive, t_sdpa, t_hand = bench_attn(device, B, H, T, d)
    print("\n=== 注意力孤立计时（纯前向，毫秒/次，T={}, d={}）===".format(T, d))
    print("  手写 naive（物化 T×T）      : {:8.3f} ms".format(t_naive))
    print("  内置 sdpa（fused FlashAttn） : {:8.3f} ms".format(t_sdpa))
    print("  手写 tiling（Python 循环）   : {:8.3f} ms".format(t_hand))
    if t_naive > 0 and t_sdpa > 0:
        print("  sdpa 比 naive 快 {:.1f}%（GPU 上差距更大，CPU 上 sdpa 无专门 kernel）".format(
            (t_naive - t_sdpa) / t_naive * 100))
    print("  手写 tiling 是 sdpa 的 {:.1f} 倍耗时（说明加速来自 fused kernel，而非分块想法本身）".format(
        t_hand / t_sdpa if t_sdpa > 0 else float("inf")))

    # 显存：用大 T 测，让 T² 主导、体现 FlashAttention 的真实收益（T 小时 q/k/v 本身占大头，看不出差别）
    T_mem, d_mem = 2048, 64
    naive_peak, flash_peak = bench_memory(device, B, H, T_mem, d_mem)
    block_mem = 64
    naive_elems = 2 * T_mem * T_mem
    flash_elems = 2 * block_mem * block_mem + T_mem * d_mem  # 块内 S + P（block=64），加 T×d 累加器
    print("\n=== 显存对比（T={}, block=64, d={}）===".format(T_mem, d_mem))
    if naive_peak is not None:
        print("  真实峰值显存: naive={:.1f} MB  flash={:.1f} MB  （flash 省 {:.0f}%）".format(
            naive_peak / 1e6, flash_peak / 1e6, (1 - flash_peak / naive_peak) * 100))
    else:
        print("  （CPU 无峰值显存统计，改用下面的元素数估算）")
    print("  中间矩阵元素数: naive=2·T²={}  flash≈2·block²+T·d={}  （flash 是 naive 的 {:.1%}）".format(
        naive_elems, flash_elems, flash_elems / naive_elems))

    print("\n结论:")
    print("  - FlashAttention 是精确算法不是近似：loss 与 naive 完全一致（val 差 ~0）。")
    print("  - 收益全在显存（不物化 T×T 矩阵，O(T²)→O(T)）和速度（GPU 上 fused kernel 省 HBM 往返）。")
    print("  - 手写 Python tiling 反而最慢——「分块」不加速，加速来自 CUDA fused kernel。")
    print("  - 本回合只埋「不物化 T²」这个伏笔，decode 时 KV cache 的张量走法是回合 06 主题。")


if __name__ == "__main__":
    main()
