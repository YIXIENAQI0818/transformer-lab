"""步骤 1：看清 ckpt 里到底有什么。

迁移的第一步是「盘点」——搞清楚我们要迁移的这个 ckpt 文件里存了哪些东西，
它们各自扮演什么角色。这正好印证之前聊的核心概念：

    模型 = 「数据」 + 「代码」

而 ckpt 里只存「数据」那一半。torch.save 存的是一个 dict，包含 4 个 key：

    model  : 所有权重（state_dict，一堆 float 张量）——这是「数据」的主体
    config : 结构超参（vocab_size / n_layer / n_embd ...）——是「量」，不是「结构逻辑」
    meta   : tokenizer 的词表映射（stoi / itos，char <-> id）——tokenizer 的「数据」
    iter   : 训练步数

「代码」（模型结构 model.py、tokenizer 算法 tokenizer.py）不在这里，在我们的仓库里。
这正是「为什么千问下载下来能直接被 Ollama 跑，而我们要带 model.py」的根源：
数据在这里，代码不在标准工具里。

运行：python src/01_inspect_ckpt.py
"""
import os

import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # hf-export/src
PROJ_DIR = os.path.dirname(SRC_DIR)                           # hf-export
LAB_DIR = os.path.dirname(PROJ_DIR)                           # transformer-lab
CKPT_PATH = os.path.join(LAB_DIR, "pretraining", "out", "ckpt.pt")


def main():
    # weights_only=False：ckpt 里有 stoi/itos 等普通 dict，不全是 tensor
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

    print("=" * 60)
    print("① ckpt 顶层有哪些 key")
    print("=" * 60)
    print("  ", list(ckpt.keys()))
    print()

    print("=" * 60)
    print("② config —— 结构超参（只描述「量」，不描述「怎么算」）")
    print("=" * 60)
    for k, v in ckpt["config"].items():
        print(f"    {k:14s} = {v}")
    print()

    print("=" * 60)
    print("③ meta —— tokenizer 的词表映射（stoi/itos，这是「数据」不是「参数」）")
    print("=" * 60)
    meta = ckpt["meta"]
    print(f"    vocab_size = {len(meta['stoi'])}")
    items = list(meta["stoi"].items())
    print(f"    前 15 条 char->id 映射: {items[:15]}")
    print()

    print("=" * 60)
    print("④ model —— state_dict（所有权重，每层结构相同）")
    print("=" * 60)
    sd = ckpt["model"]
    total = 0
    for k, v in sd.items():
        n = v.numel()
        total += n
        # 完整打印：wte/wpe/ln_f/lm_head + 每层 8 个参数。6 层共 ~85 个 key。
        print(f"    {k:35s} shape={str(tuple(v.shape)):22s} numel={n:>9,}")
    print(f"    {'— 合计 —':35s} {total:>33,} 参数")
    print(f"    fp32 下 = {total * 4 / 1e6:.1f} MB")
    print()

    print("=" * 60)
    print(f"训练步数 iter = {ckpt['iter']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
