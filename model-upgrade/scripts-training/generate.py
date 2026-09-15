"""从 ckpt 加载模型采样生成（升级后模型）。

用法（从 model-upgrade/ 目录）：
    python scripts-training/generate.py --ckpt out/train/ckpt.pt --prompt "ROMEO:" --max_new_tokens 500
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from model import GPT, GPTConfig
from tokenizer import CharTokenizer

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "train", "ckpt.pt")


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k):
    """KV cache 自回归采样：prefill 一次 + 逐步 decode，返回完整 token 序列 (1, T)。"""
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
    parser = argparse.ArgumentParser(description="从训练好的 ckpt 采样生成文本")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--prompt", type=str, default="\n", help='起始文本，如 "ROMEO:"')
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.from_meta(ckpt["meta"])

    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    gen = generate(model, idx, args.max_new_tokens, args.temperature, args.top_k)
    print(tok.decode(gen[0].tolist()))


if __name__ == "__main__":
    main()
