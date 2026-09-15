"""训练「组件升级后」的模型（现代组件全开）。

用 char-level tokenizer 在 TinyShakespeare 上训练可配置骨架的现代配置
（RoPE + RMSNorm + SwiGLU + GQA + FlashAttention），得到升级后的最终模型。
与 model-core 的朴素 GPT-2 同数据同规模，只把模型换成现代组件。

运行（从 model-upgrade/ 目录）：python scripts-training/train.py
"""
import math
import os
import sys
import time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import GPT, GPTConfig
from tokenizer import CharTokenizer

# ---------------- 超参（对齐 model-core 的完整训练规模） ----------------
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
N_KV_HEAD = 2          # 回合04 GQA：K/V 头共享
DROPOUT = 0.2

LEARNING_RATE = 1e-3
WARMUP_ITERS = 100
MIN_LR = 1e-4
DECAY_ITERS = 5000        # cosine 降到 min_lr 的总步数（= max_iters）
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 200
LOG_INTERVAL = 10
GEN_TOKENS = 200          # 边训边生成时采样长度

TRAIN_SPLIT = 0.9
SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "input.txt")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "train")
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")


def prepare_data():
    """读语料 -> char-level encode -> 按 TRAIN_SPLIT 切 train/val。"""
    text = open(DATA_PATH, encoding="utf-8").read()
    tok = CharTokenizer(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * TRAIN_SPLIT)
    return tok, ids[:n], ids[n:]


def get_batch(split, train_ids, val_ids):
    """随机采 BATCH_SIZE 个 BLOCK_SIZE 窗口；x=前 T 个 token，y=后移一位的 next-token。"""
    ids = train_ids if split == "train" else val_ids
    ix = torch.randint(len(ids) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([ids[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids):
    """在 train/val 上各估 EVAL_ITERS 个 batch 的平均 loss（关 dropout，不反传）。"""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split, train_ids, val_ids)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    """linear warmup -> cosine decay -> min_lr 平台。"""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it > DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def configure_optimizers(model):
    """AdamW：2D 及以上权重（Linear/Embedding）加 weight_decay，bias/LayerNorm 不加。"""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=LEARNING_RATE, betas=(0.9, 0.95))


@torch.no_grad()
def sample(model, tok, max_new_tokens=GEN_TOKENS):
    """从换行符起始生成一段文本，观察训练过程中输出从乱码变通顺。"""
    model.eval()
    idx = torch.tensor([tok.encode("\n")], dtype=torch.long, device=DEVICE)
    gen = _generate(model, idx, max_new_tokens)
    model.train()
    return tok.decode(gen[0].tolist())


@torch.no_grad()
def _generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """KV cache 自回归采样（prefill 一次 + 逐步 decode），返回完整 token 序列 (1, T)。"""
    logits, cache = model(idx, use_cache=True)          # prefill：并行算整段 prompt
    for _ in range(max_new_tokens):
        logit = logits[:, -1, :] / temperature         # 末位 logits
        if top_k is not None:
            v, _ = torch.topk(logit, min(top_k, logit.size(-1)))
            logit[logit < v[:, [-1]]] = -float("Inf")
        probs = torch.softmax(logit, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)
        idx = torch.cat([idx, next_token], dim=1)
        logits, cache = model(next_token, cache=cache)  # decode：复用 cache 只算新 token
    return idx


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    tok, train_ids, val_ids = prepare_data()
    print(f"语料 {len(train_ids) + len(val_ids)} tokens, vocab_size={tok.vocab_size}, "
          f"device={DEVICE}")

    config = GPTConfig(
        vocab_size=tok.vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
        pos_enc="rope", norm="rmsnorm", activation="swiglu",
        n_kv_head=N_KV_HEAD, attn_impl="flash",
    )
    model = GPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.2f}M（现代组件：RoPE/RMSNorm/SwiGLU/GQA/FlashAttention）")

    optimizer = configure_optimizers(model)
    t0 = time.time()

    for it in range(MAX_ITERS):
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch("train", train_ids, val_ids)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % LOG_INTERVAL == 0:
            dt = time.time() - t0
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")

        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            est = estimate_loss(model, train_ids, val_ids)
            print(f"== step {it:5d} | train loss {est['train']:.4f} | val loss {est['val']:.4f}")
            print("---- 采样 ----")
            print(sample(model, tok))
            print("--------------")

    ckpt = {"model": model.state_dict(), "config": asdict(config), "meta": tok.meta, "iter": MAX_ITERS}
    torch.save(ckpt, CKPT_PATH)
    print(f"已保存 ckpt -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
