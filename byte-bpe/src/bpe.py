"""byte-level BPE 分词器从零实现（步骤 02 + 03）。

在「字节（0~255）」层面做 BPE（Byte Pair Encoding，字节对编码），三步：

  1. 训练 train()：从 UTF-8 字节序列出发，反复统计「最高频的相邻字节对」并把它合并成一个
     新 token（子词），记录 merge rule，直到达到目标 vocab_size。
  2. 编码 encode()：用学到的 merge rules 把文本贪心合并成 token id 序列。
  3. 解码 decode()：把每个 token id 展开回字节序列，再 UTF-8 解码成文本。

「为什么在字节上做 BPE」：
    byte-level（见 byte_encoding.py）保证任何文本都能变成 0~255 的字节序列、永不 OOV；
    但纯字节序列「太长」——中文 3 字节/字符、emoji 4 字节/字符，序列长度 = 字符数 × 平均字节数。
    BPE 在字节序列上反复合并高频相邻对，合并出「子词」（如 "th"、"he"、"the"），
    压缩序列长度。两者结合 = 既保留 byte-level 的「永不 OOV」，又拿到 BPE 的「压缩」。

注：工业 byte-level BPE（GPT-2 / tokenizers 库）会在字节上再套一层「字节 → 可打印 unicode 字符」
映射（bytes_to_unicode），本文件为了教学清晰直接在原始字节 id 上做，两者语义一致、token id 不同，
差异详见 scripts/compare_tokenizers.py。

运行：python src/bpe.py
"""
from byte_encoding import text_to_bytes


def _get_stats(ids):
    """统计相邻对 (ids[i], ids[i+1]) 的出现次数，返回 {pair: count}。

    这是 BPE 训练和 encode 的核心动作：找出「哪两个相邻 token 最常一起出现」。
    手写 dict 统计（不依赖 collections.Counter），让「数频率」这一步完全透明。
    """
    stats = {}
    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    return stats


def _merge(ids, pair, new_id):
    """把序列中所有相邻的 pair=(a, b) 替换成单个 new_id，返回新序列。

    一次只合并「这一个对」，其他对保持原样——BPE 每次迭代只合并最高频的那一个对。
    """
    a, b = pair
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            new_ids.append(new_id)   # 命中 a|b -> 合并成一个 new_id
            i += 2                    # 跳过 b，继续看后面
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BpeTokenizer:
    """byte-level BPE 分词器：在字节序列上训练 merge rules，再据此 encode / decode。

    - merges: {(byte_a, byte_b) -> new_id}，按合并先后顺序插入（Python dict 保序），
      new_id 从 256 开始递增，越小表示越早合并（rank 越小、越优先）。
    - vocab : {token_id -> bytes}，token 到字节序列的展开（decode 用），训练后构建。
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    # ---------- 训练 ----------

    def train(self, text, vocab_size):
        """在 text 上训练，直到词表达到 vocab_size（>= 256，含 256 个基础字节）。

        每次迭代：统计最高频相邻对 → 合并成一个新 token → 记录 merge rule。
        若语料太小、没有可合并的对（所有对都只出现 1 次且无更高频），提前 break。
        """
        assert vocab_size >= 256, f"vocab_size={vocab_size} 必须 >= 256（基础字节数）"
        ids = text_to_bytes(text)                  # 起点：纯字节序列（复用 byte_encoding 的 UTF-8 编码）
        self.merges = {}
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = _get_stats(ids)
            if not stats:
                break                              # 序列已无法再合并（极短语料）
            pair = max(stats, key=stats.get)       # 最高频对；并列时取先遇到的（tie-breaking）
            new_id = 256 + i
            ids = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
        self._build_vocab()
        return self

    def _build_vocab(self):
        """把每个 token id 展开成字节序列：256 个基础字节 + 按合并顺序递归拼接。

        依赖 dict 保序：遍历 merges 时 a、b 的展开一定先于 new_id 构建
        （a、b 要么是 <256 的基础字节，要么是更早合并出来的小 id）。
        """
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self.merges.items():
            vocab[idx] = vocab[a] + vocab[b]
        self.vocab = vocab

    # ---------- 编码 / 解码 ----------

    def encode(self, text):
        """text -> token id 列表（GPT-2 风格：反复找「rank 最小」的可合并对贪心合并）。

        rank = 合并顺序（new_id 越小越早合并、越优先）。每次找当前序列里 rank 最小的相邻对，
        合并之；直到没有任何可合并的对（所有相邻对都不在 merges 里）。
        这样保证同一个文本用同一个 tokenizer 编码结果唯一、且优先用「更早学到的」子词。
        """
        ids = text_to_bytes(text)
        while len(ids) >= 2:
            stats = _get_stats(ids)
            # rank 最小 = merges 里 value 最小；不在 merges 里的对 rank 记为 +inf（最后考虑）
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                              # 最小的都不在 merges 里 -> 无对可合并
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        """token id 列表 -> 文本：每个 id 展开成字节拼起来，再 UTF-8 解码。

        errors="replace" 兜底：非法字节序列用 U+FFFD 替换而非抛异常（decode 永远有定义）。
        """
        b = b"".join(self.vocab[i] for i in ids)
        # 直接对 bytes 对象 .decode（join 出来就是 bytes），不必走 bytes_to_text——
        # 那个接口面向 list[int]，会多一次 bytes -> list -> bytes 的往返。
        return b.decode("utf-8", errors="replace")

    # ---------- 元信息（供 ckpt 保存 / 重建，对齐 CharTokenizer.meta） ----------

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    @property
    def meta(self):
        """merges 序列化为可 JSON 保存的 [[a, b, new_id], ...]（tuple key 不可直接 JSON）。"""
        return {"merges": [[a, b, idx] for (a, b), idx in self.merges.items()]}

    @classmethod
    def from_meta(cls, meta):
        """从 meta 重建 tokenizer（无需原始语料，见第 5 步 ckpt 加载）。"""
        tok = cls.__new__(cls)
        tok.merges = {(a, b): idx for a, b, idx in meta["merges"]}
        tok._build_vocab()
        return tok


if __name__ == "__main__":
    import os
    import time

    # 语料：优先读 TinyShakespeare（真实英语，能合并出 th/he/the 等真实子词），
    # 读不到则用内嵌文本。取前 20000 字节控制训练时间（秒级）。
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "pretraining", "data", "input.txt")
    if os.path.exists(data_path):
        text = open(data_path, encoding="utf-8").read()[:20000]
        src = data_path
    else:
        text = ("First Citizen: Before we proceed any further, hear me speak.\n"
                "All: Speak, speak.\n" * 50)
        src = "内嵌文本"

    VOCAB_SIZE = 512
    tok = BpeTokenizer()
    t0 = time.time()
    tok.train(text, VOCAB_SIZE)
    dt = time.time() - t0

    print("=" * 66)
    print("手写 byte-level BPE（步骤 02 + 03）")
    print("语料: {}（前 {} 字节）  vocab_size 目标: {}".format(
        src, len(text.encode("utf-8")), VOCAB_SIZE))
    print("训练 {} 次合并耗时 {:.2f}s，最终 vocab_size = {}".format(
        len(tok.merges), dt, tok.vocab_size))
    print("=" * 66)

    # ① 前 10 个 merge：展示 BPE 真的合并出了高频子词（字节对 -> 展开 -> 可读字符串）
    print("\n① 前 10 个 merge（byte pair -> 展开字节 -> 可读子词）:")
    for (a, b), idx in list(tok.merges.items())[:10]:
        merged = tok.vocab[idx]
        print("   ({:>3},{:>3}) -> 字节 {} -> '{}'".format(
            a, b, list(merged), merged.decode("utf-8", errors="replace")))

    # ② round-trip：任意文本（含中文/emoji）encode -> decode 恒还原
    print("\n② round-trip（encode -> decode 恒等）:")
    for s in ["First Citizen: hear me speak.", "Hello world!", "中文也永不 OOV", "emoji 🙂🎉"]:
        ids = tok.encode(s)
        back = tok.decode(ids)
        ok = "✔" if back == s else "✘ 不一致！"
        print("   {:<28} -> {} token -> '{}' {}".format(
            repr(s), len(ids), back, ok))

    # ③ 压缩效果：同一段文本，字符数 vs token 数（BPE 压缩序列长度）
    sample = text[:2000]
    n_chars = len(sample)
    n_tokens = len(tok.encode(sample))
    print("\n③ 压缩效果: 2000 字符 -> {} token（压缩比 {:.2f}x）".format(
        n_tokens, n_chars / n_tokens))
    print("   纯字节基线: 2000 字符 -> {} 字节（BPE 比纯字节再短 {:.0f}%）".format(
        len(sample.encode("utf-8")),
        100 * (1 - n_tokens / len(sample.encode("utf-8")))))

    # ④ meta 序列化 round-trip：from_meta 重建的 tokenizer 编码结果与原始一致
    tok2 = BpeTokenizer.from_meta(tok.meta)
    assert tok2.encode("First Citizen") == tok.encode("First Citizen")
    print("\n④ meta 序列化: from_meta 重建的 tokenizer 编码结果一致 ✔")
