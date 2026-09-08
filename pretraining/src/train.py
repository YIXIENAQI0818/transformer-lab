"""预训练训练循环（阶段 2）。

用 char-level tokenizer 在 TinyShakespeare 上训练 decoder-only GPT，
跑通「数据 -> tokenizer -> 模型 -> 训练 -> 生成」全流程。

超参对齐 nanoGPT config/train_shakespeare_char.py（备选见文件底部注释）。

运行（从 pretraining/ 目录）：python src/train.py
"""
import math
import os
import time
from dataclasses import asdict

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

# ---------------- 超参（nanoGPT shakespeare_char，备选见底部） ----------------
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
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

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
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
    gen = model.generate(idx, max_new_tokens=max_new_tokens)
    model.train()
    return tok.decode(gen[0].tolist())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    tok, train_ids, val_ids = prepare_data()
    print(f"语料 {len(train_ids) + len(val_ids)} tokens, vocab_size={tok.vocab_size}, "
          f"device={DEVICE}")

    config = GPTConfig(
        vocab_size=tok.vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=DROPOUT,
    )
    model = GPT(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params / 1e6:.2f}M")

    optimizer = configure_optimizers(model)
    t0 = time.time()

    for it in range(MAX_ITERS):
        # 按调度器更新 lr
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

# ---- 备选超参 ----
# 更大的模型（nanoGPT gpt2-small）：n_layer=12, n_head=12, n_embd=768, lr=6e-4,
#   max_iters=600000, batch_size=12, block_size=1024（显存/时间开销大幅增加）。
# 想更快验证：把 MAX_ITERS 调到 500~1000，EVAL_INTERVAL 同步缩小。
