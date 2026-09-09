"""回合 06 对比：naive 重算 vs KV cache（prefill / decode 张量走法 + GQA cache shape）。

在**同一个模型**上做自回归生成，对比两种推理走法：

  - naive  ：每吐一个 token 都对「整条已生成序列」重算一遍 attention（O(T²)/步，总 O(T³)）。
  - cached ：prefill 一次性把 prompt 的 K/V 塞进 cache，之后每步只算新 token 的 Q/K/V、
             Q 只 attend 到 cache（O(T)/步，总 O(T²)）。

关键教学点：KV cache 换的是「推理时的计算方式」不是「模型」，所以**生成结果必须与 naive 完全一致**
（这是精确优化、不是近似）；收益全在速度（decode 每步 O(T) 而非 O(T²)）和显存（过去 K/V 只存不算）。
本脚本与前面回合的「mini 训练对比」不同——KV cache 不影响训练、不改变 loss，所以不做训练，
直接用一个随机初始化的模型验证「两条走法 logits 逐 token 一致 + 计时差异 + 张量走法」。

运行：python scripts/compare_kv_cache.py [--block-size B] [--new-tokens N] [--prompt-len P] [--data PATH]
"""
import argparse
import math
import os
import sys
import time

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


# ---------- 两种生成走法 ----------

@torch.no_grad()
def generate_naive(model, prompt, n_new):
    """naive：每步对 [prompt + 已生成] 整条序列重算一遍 attention，取末位 logits。

    返回 (tokens, logits)：tokens (1, P+n_new)，logits (1, n_new, vocab_size)。
    logits[:, t] 是「位置 P-1+t 的 logit」，即用来预测第 t 个新 token（位置 P+t）的 logit。
    """
    tokens = prompt.clone()
    logits_list = []
    for _ in range(n_new):
        logits = model(tokens)             # (1, T, V)，T 每步 +1，重算全部 T²
        next_logit = logits[:, -1, :]      # 末位（位置 P-1+t）的 logits
        logits_list.append(next_logit)
        tokens = torch.cat([tokens, next_logit.argmax(-1, keepdim=True)], dim=1)
    return tokens, torch.stack(logits_list, dim=1)   # (1, n_new, V)


@torch.no_grad()
def generate_cached(model, prompt, n_new):
    """cached：prefill 一次算完 prompt 的 K/V 进 cache，之后每步只算新 token。

    返回 (tokens, logits)：tokens (1, P+n_new)，logits (1, n_new, vocab_size)。
    与 generate_naive 对齐：logits[:, t] 同样是「位置 P-1+t 的 logit」——第一个来自 prefill
    的末位，之后的来自每次 decode 的末位。
    """
    tokens = prompt.clone()
    logits, cache = model(tokens, use_cache=True)   # prefill：并行算整段 prompt，缓存 K/V
    logits_list = []
    for _ in range(n_new):
        logits_list.append(logits[:, -1, :])        # 位置 P-1+t 的 logit（第 0 次是 prefill 末位）
        next_token = logits[:, -1, :].argmax(-1, keepdim=True)   # (1,1)
        tokens = torch.cat([tokens, next_token], dim=1)
        logits, cache = model(next_token, cache=cache)   # decode：喂新 token，复用 cache
    return tokens, torch.stack(logits_list, dim=1)   # (1, n_new, V)


# ---------- 主流程 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32,
                        help="自回归生成的新 token 数")
    parser.add_argument("--prompt-len", type=int, default=16,
                        help="prompt 长度（prefill 阶段的输入长度）")
    parser.add_argument("--data", type=str, default=None,
                        help="训练文本路径，默认复用 pretraining 的 TinyShakespeare")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    # 数据 + tokenizer（只用来拿词表和一段真实 prompt）
    data_path = args.data or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "pretraining", "data", "input.txt")
    text = load_text(data_path)
    stoi = build_tokenizer(text)
    vocab_size = len(stoi)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    # 沿用前五回合已升级底座：RoPE + RMSNorm + SwiGLU + GQA + FlashAttention
    # KV cache 不改模型结构，用一个随机初始化模型即可（验证的是「两条走法是否一致 + 快慢」）
    base_cfg = dict(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=4, n_head=4, n_embd=128, pos_enc="rope", norm="rmsnorm",
                    activation="swiglu", n_kv_head=2, attn_impl="flash")
    torch.manual_seed(1337)
    model = GPT(GPTConfig(**base_cfg)).to(device).eval()

    # prompt：取数据里一段（长度 prompt_len）
    prompt = data[:args.prompt_len].unsqueeze(0).to(device)   # (1, P)

    n_head, n_kv_head, head_size = 4, 2, 128 // 4
    n_layer = 4

    print("=" * 64)
    print("回合 06：naive 重算 vs KV cache")
    print("设备: {}  数据: {}".format(device, data_path if os.path.exists(data_path) else "（合成）"))
    print("vocab={}  block_size={}  n_head={}  n_kv_head={}  head_size={}  n_layer={}".format(
        vocab_size, args.block_size, n_head, n_kv_head, head_size, n_layer))
    print("prompt 长度 P={}  生成新 token N={}".format(args.prompt_len, args.new_tokens))
    print("=" * 64)

    # ---- 1) prefill / decode 张量走法 ----
    with torch.no_grad():
        logits_pre, cache_pre = model(prompt, use_cache=True)   # prefill
    print("\n[1] prefill 阶段（一次并行前向算整段 prompt）:")
    print("    输入 idx          : {}".format(tuple(prompt.shape)))
    print("    输出 logits       : {}".format(tuple(logits_pre.shape)))
    print("    cache（{} 层，每层 k/v 各一个）:".format(len(cache_pre)))
    print("      k 张量 shape     : {}  （n_kv_head 粒度，不是 n_head）".format(tuple(cache_pre[0][0].shape)))
    with torch.no_grad():
        logits_d1, cache_d1 = model(logits_pre[:, -1, :].argmax(-1, keepdim=True), cache=cache_pre)
    print("\n[2] decode 阶段（每步只喂一个新 token）:")
    print("    输入 idx          : {}".format((1, 1)))
    print("    输出 logits       : {}".format(tuple(logits_d1.shape)))
    print("    cache k 长度       : {} -> {}（追加 1 个 token）".format(
        cache_pre[0][0].size(2), cache_d1[0][0].size(2)))

    # ---- 2) 正确性：naive 与 cached 的 logits 逐 token 一致 ----
    print("\n[3] 正确性：两条走法生成 N={} 个 token，对比 logits".format(args.new_tokens))
    t0 = time.perf_counter()
    tokens_naive, logits_naive = generate_naive(model, prompt, args.new_tokens)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    tokens_cached, logits_cached = generate_cached(model, prompt, args.new_tokens)
    t_cached = time.perf_counter() - t0

    logits_err = (logits_naive - logits_cached).abs().max().item()
    tokens_match = torch.equal(tokens_naive, tokens_cached)
    print("    生成 token 序列一致      : {}".format(tokens_match))
    print("    logits 最大绝对误差      : {:.2e}  （期望 ~1e-6，KV cache 是精确优化）".format(logits_err))

    # ---- 3) 计时：cached 比 naive 快多少，及复杂度对比 ----
    print("\n[4] 计时（生成 N={} 个 token，含 python 循环开销）:".format(args.new_tokens))
    print("    naive 重算 : {:8.3f} ms".format(t_naive * 1e3))
    print("    KV cache   : {:8.3f} ms".format(t_cached * 1e3))
    if t_naive > 0:
        print("    cached 快 {:.1f}%（N=32 时序列还短、python 循环开销占大头，差距被掩盖；".format(
            (t_naive - t_cached) / t_naive * 100))
        print("              N 越大差距越明显：naive 每步 O(T²)、cache 每步 O(T)，见下方理论点积数）")
    # 理论复杂度：naive 每步重算 (P+t)² 个点积，cache 每步只算 (P+t) 个
    P = args.prompt_len
    naive_ops = sum((P + t) ** 2 for t in range(args.new_tokens))
    cached_ops = sum(P + t for t in range(args.new_tokens))
    print("    理论点积次数: naive={}  cache={}  （cache 是 naive 的 {:.1%}）".format(
        naive_ops, cached_ops, cached_ops / naive_ops))

    # ---- 4) GQA cache shape：cache 按 n_kv_head 存，省显存 ----
    print("\n[5] GQA 下 cache shape（每层，L 个 token 的 K/V 元素数）:")
    for n_kv in [4, 2, 1]:
        kv_per_token = 2 * n_kv * head_size
        print("    n_kv_head={}: cache 张量 (B, {}, L, {})，每 token {} 元素（相对 MHA {:.0%}）".format(
            n_kv, n_kv, head_size, kv_per_token, kv_per_token / (2 * n_head * head_size)))

    print("\n结论:")
    print("  - KV cache 换「计算方式」不换「模型」：naive 与 cached 生成结果逐 token 一致（误差 ~1e-6）。")
    print("  - 收益在速度与显存：decode 每步 O(T) 而非 O(T²)（总 O(T²) 而非 O(T³)），K/V 只存不算。")
    print("  - prefill 一次算整段 prompt 的 K/V，decode 逐 token 复用；cache 按 n_kv_head 存（GQA 省显存）。")
    print("  - 一个坑：decode 的新 token 必须按绝对位置做 RoPE（见 src/kv_cache.py 验证 2）。")


if __name__ == "__main__":
    main()
