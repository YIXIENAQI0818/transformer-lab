"""从 ckpt 加载模型采样生成（阶段 2）。

用法（从 pretraining/ 目录）：
    python src/generate.py --ckpt out/ckpt.pt --prompt "ROMEO:" --max_new_tokens 500
"""
import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import CharTokenizer

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "ckpt.pt")


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
    gen = model.generate(
        idx, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k,
    )
    print(tok.decode(gen[0].tolist()))


if __name__ == "__main__":
    main()
