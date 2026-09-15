"""步骤 04：手写 byte-level BPE vs HuggingFace tokenizers 库对照。

目的：用工业标准库（tokenizers，Rust 实现）在同一语料、同一 vocab_size 下训练一个
byte-level BPE，和手写版对照，验证手写实现的语义正确。

对照项（软对照）：
  1. 词表大小 / 合并次数一致
  2. 高频 merge 结果吻合（手写「字节对」 vs 库「byte->unicode 字符对」反映射回字节）
  3. 同一文本编码长度相近
  4. round-trip 都正确
  5. 都消除 OOV（中文 / emoji 都能编码）

为什么是「软对照」而非逐 token 逐 id 一致：
  手写版直接在「原始字节 id（0~255）」上做 BPE；tokenizers 的 ByteLevel 先在字节上套一层
  GPT-2 的 byte->unicode 映射（把不可打印字节换成可打印 unicode 字符），再在这些字符上做
  BPE。两者数学语义一致、token 表示不同，故对照时把库的 token 反映射回字节再比。

运行：python scripts/compare_tokenizers.py [--max-chars N] [--vocab-size V]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from bpe import BpeTokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def bytes_to_unicode():
    """GPT-2 的 byte -> unicode 映射（tokenizers ByteLevel 内部就是这个）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


UNICODE_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}


def lib_token_to_text(token_str):
    """把库的一个 token 字符串（byte->unicode 映射后的）反映射回字节并解码成可读文本。"""
    bs = bytes(UNICODE_TO_BYTE[c] for c in token_str)
    return bs.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chars", type=int, default=100000,
                        help="取语料前 N 字符（控制手写版训练时间）")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "data", "input.txt")
    if os.path.exists(data_path):
        text = open(data_path, encoding="utf-8").read()[:args.max_chars]
        src = data_path
    else:
        text = ("First Citizen: Before we proceed any further, hear me speak.\n"
                "All: Speak, speak.\n" * 500)
        src = "内嵌文本"

    print("=" * 72)
    print("手写 byte-level BPE vs tokenizers 库对照（步骤 04）")
    print("语料: {}（{} 字符）  vocab_size 目标: {}".format(src, len(text), args.vocab_size))
    print("=" * 72)

    # ---- 手写版 ----
    hand = BpeTokenizer().train(text, args.vocab_size)
    n_hand_merges = hand.vocab_size - 256  # 合并次数 = 词表 - 256 基础字节

    # ---- 库版 ----
    lib = Tokenizer(models.BPE(unk_token=None))
    lib.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    lib.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 字节字符兜底，对齐手写版
    )
    lib.train_from_iterator([text], trainer=trainer)
    lib_vocab = lib.get_vocab()            # {token_str: id}
    lib_id_to_token = {v: k for k, v in lib_vocab.items()}
    n_lib_merges = lib.get_vocab_size() - 256

    # ① 词表大小
    print("\n① 词表大小:")
    print("   手写: vocab={}（256 基础字节 + {} 合并）".format(hand.vocab_size, n_hand_merges))
    print("   库  : vocab={}（256 基础字符 + {} 合并）".format(lib.get_vocab_size(), n_lib_merges))

    # ② merge 产物对照：BPE 的 tie-breaking（最高频对并列时取谁）在不同实现里不同——
    #    手写版 Python 的 max 取「序列中先遇到」的对，库版 Rust 按自己的排序取。
    #    这会让两者的 merge 顺序（rank）发散，但「学出的高频子词集合」应一致。
    #    所以对照「前 20 个 merge 产物的集合」，而不是逐 rank 比（逐 rank 会因顺序错位）。
    n_show = 20
    hand_prod = [hand.vocab[256 + i].decode("utf-8", errors="replace")
                 for i in range(n_show) if 256 + i in hand.vocab]
    lib_prod = [lib_token_to_text(lib_id_to_token[256 + i])
                for i in range(n_show) if 256 + i in lib_id_to_token]
    common = sorted(set(hand_prod) & set(lib_prod))
    print("\n② 前 {} 个 merge 产物对照（顺序会因 tie-breaking 发散，故比「集合」）:".format(n_show))
    print("   手写: {}".format(hand_prod))
    print("   库  : {}".format(lib_prod))
    print("   共同学出的子词（{} 个）: {}".format(len(common), common))

    # ③ 编码长度对照
    sample = text[:5000]
    hand_ids = hand.encode(sample)
    lib_ids = lib.encode(sample).ids
    print("\n③ 编码长度对照（同一段 {} 字符文本）:".format(len(sample)))
    print("   手写: {} token   库: {} token".format(len(hand_ids), len(lib_ids)))

    # ④ round-trip
    print("\n④ round-trip:")
    for s in ["First Citizen: hear me speak.", "Hello world!"]:
        h_ok = hand.decode(hand.encode(s)) == s
        l_ok = lib.decode(lib.encode(s).ids) == s
        print("   {:<30} 手写 {}  库 {}".format(
            repr(s), "✔" if h_ok else "✘", "✔" if l_ok else "✘"))

    # ⑤ OOV 对照
    print("\n⑤ OOV 对照（训练语料是纯 ASCII，中文/emoji 是「训练集外」字符）:")
    for s in ["中文", "emoji 🙂"]:
        h_ok = hand.decode(hand.encode(s)) == s
        l_ok = lib.decode(lib.encode(s).ids) == s
        print("   {:<12} 手写 encode->decode {}  库 encode->decode {}".format(
            repr(s), "✔" if h_ok else "✘", "✔" if l_ok else "✘"))


if __name__ == "__main__":
    main()
