"""用 byte-level BPE 训练小模型（步骤 05）。

复用 pretraining 的完整训练循环（AdamW + cosine 调度 + 梯度裁剪 + eval + 采样生成），
tokenizer 用自写 BpeTokenizer（byte-level BPE），模型用纯现代骨架（src/model.py，
RoPE + RMSNorm + SwiGLU + GQA + FlashAttention）。

时序：先在语料上训好 BPE tokenizer（学出 merge 规则，之后冻结，只做 encode/decode），
再训模型。ckpt 里同时存 model 权重 + tokenizer meta，生成阶段用 from_meta 重建 tokenizer。

运行：
  python scripts/train_model.py --max-iters 100   # 小步数验证脚本能跑通
  python scripts/train_model.py                   # 完整训练（GPU ~35min）
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from model import GPT, GPTConfig
from bpe import BpeTokenizer

# ---------------- 超参（对齐 pretraining 的完整训练规模） ----------------
BATCH_SIZE = 64
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
N_KV_HEAD = 2          # GQA：K/V 头共享
DROPOUT = 0.2

LEARNING_RATE = 1e-3
WARMUP_ITERS = 100
MIN_LR = 1e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

EVAL_INTERVAL = 250
EVAL_ITERS = 200
LOG_INTERVAL = 10
GEN_TOKENS = 200

TRAIN_SPLIT = 0.9
SEED = 1337


def load_text(path):
    """读 TinyShakespeare；不存在则回退合成文本（保证脚本独立可跑）。"""
    if path and os.path.exists(path):
        return open(path, encoding="utf-8").read()
    torch.manual_seed(0)
    vocab = " abcdefghijklmnopqrstuvwxyz\n"
    idx = torch.randint(0, len(vocab), (50000,)).tolist()
    return "".join(vocab[i] for i in idx)


def prepare_data(text, tok):
    """encode 全文本 -> tensor，按 TRAIN_SPLIT 切 train/val。"""
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * TRAIN_SPLIT)
    return ids[:n], ids[n:]


def get_batch(ids, device):
    """随机采 BATCH_SIZE 个 BLOCK_SIZE 窗口；x=前 T 个 token，y=后移一位的 next-token。"""
    ix = torch.randint(len(ids) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([ids[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, device):
    """在 train/val 上各估 EVAL_ITERS 个 batch 的平均 loss。"""
    model.eval()
    out = {}
    for split, ids in (("train", train_ids), ("val", val_ids)):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(ids, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, max_iters):
    """linear warmup -> cosine decay -> min_lr 平台。"""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if it > max_iters:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (max_iters - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def configure_optimizers(model):
    """AdamW：2D 及以上权重加 weight_decay，bias/norm 不加。"""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=LEARNING_RATE, betas=(0.9, 0.95),
    )


@torch.no_grad()
def sample(model, tok, device, max_new_tokens=GEN_TOKENS):
    """从换行符起始生成一段文本，观察输出从乱码变通顺。"""
    model.eval()
    idx = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
    gen = model.generate(idx, max_new_tokens=max_new_tokens)
    model.train()
    return tok.decode(gen[0].tolist())


def make_config(vocab_size):
    """纯现代骨架配置：RoPE/RMSNorm/SwiGLU/GQA 固定，vocab_size 由 BPE tokenizer 注入。"""
    return GPTConfig(
        vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        n_kv_head=N_KV_HEAD, dropout=DROPOUT,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=5000,
                        help="训练步数（默认 5000；先小步数验证可传 100）")
    parser.add_argument("--vocab-size", type=int, default=512, help="BPE 目标词表")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
    os.makedirs(out_dir, exist_ok=True)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "pretraining", "data", "input.txt")
    text = load_text(data_path)

    print("=" * 66)
    print("用 byte-level BPE 训练小模型（步骤 05）")
    print("设备: {}  数据: {}（{} 字符）".format(
        device, "TinyShakespeare" if os.path.exists(data_path) else "合成", len(text)))
    print("=" * 66)

    # ---- BPE tokenizer：只训一次，之后从 out/tokenizer_meta.json 加载（不重训）----
    tok_meta_path = os.path.join(out_dir, "tokenizer_meta.json")
    if os.path.exists(tok_meta_path):
        with open(tok_meta_path) as f:
            tok = BpeTokenizer.from_meta(json.load(f))
        print("从 {} 加载已训练的 BPE tokenizer（不重训）".format(tok_meta_path))
    else:
        print("\n[BPE 训练中...] 语料 {} 字符，目标 vocab {}".format(len(text), args.vocab_size))
        t0 = time.time()
        tok = BpeTokenizer().train(text, args.vocab_size)
        with open(tok_meta_path, "w") as f:
            json.dump(tok.meta, f)
        print("BPE 训练完成并保存：vocab={}（{} merges），耗时 {:.1f}s -> {}".format(
            tok.vocab_size, len(tok.merges), time.time() - t0, tok_meta_path))

    n_tokens = len(tok.encode(text))
    print("压缩比：{} 字符 -> {} token（{:.2f}x）".format(
        len(text), n_tokens, len(text) / n_tokens))

    # byte-level 永不 OOV：训练集外字符也能 encode->decode 还原
    for s in ["中文", "emoji 🙂"]:
        assert tok.decode(tok.encode(s)) == s, s
    print("OOV 兜底：训练集外字符（中文/emoji）encode->decode 还原 ✔")

    # ---- 准备数据 + 建模型 ----
    train_ids, val_ids = prepare_data(text, tok)
    torch.manual_seed(SEED)
    config = make_config(tok.vocab_size)
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("模型：vocab={}  参数量 {:.2f}M（RoPE/RMSNorm/SwiGLU/GQA/Flash）".format(
        tok.vocab_size, n_params / 1e6))

    # ---- 训练 ----
    optimizer = configure_optimizers(model)
    t0 = time.time()
    for it in range(args.max_iters):
        lr = get_lr(it, args.max_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_ids, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if it % LOG_INTERVAL == 0:
            dt = time.time() - t0
            print("iter {:5d} | loss {:.4f} | lr {:.2e} | {:.1f}s".format(
                it, loss.item(), lr, dt))

        if it % EVAL_INTERVAL == 0 or it == args.max_iters - 1:
            est = estimate_loss(model, train_ids, val_ids, device)
            print("== step {:5d} | train loss {:.4f} | val loss {:.4f} | ppl {:.2f}".format(
                it, est["train"], est["val"], math.exp(est["val"])))
            print("---- 采样 ----")
            print(sample(model, tok, device))
            print("--------------")

    # ---- 保存 ckpt（model 权重 + tokenizer meta，生成阶段 from_meta 重建）----
    ckpt = {
        "model": model.state_dict(),
        "config": asdict(config),
        "tokenizer_meta": tok.meta,
        "iter": args.max_iters,
    }
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    torch.save(ckpt, ckpt_path)
    print("已保存 ckpt -> {}".format(ckpt_path))


if __name__ == "__main__":
    main()
