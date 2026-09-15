"""char-level tokenizer（model-upgrade 训练用）。

把每个字符映射为一个 token id。TinyShakespeare 的 65 个唯一字符正好构成词表：
换行 + 空格 + 标点 + 大小写字母。byte-level BPE 升级在 model-tokenizer 阶段。
"""


class CharTokenizer:
    """字符级分词器：chars <-> ids 双向映射，可随 ckpt 保存/重建。"""

    def __init__(self, text: str):
        # sorted(set()) 保证词表顺序确定、可复现（不依赖 dict 遍历顺序）
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char -> id
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # id -> char

    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    @property
    def meta(self) -> dict:
        """随 ckpt 一起保存，generate 阶段据此重建（无需原始语料）。"""
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_meta(cls, meta: dict) -> "CharTokenizer":
        """从保存的 meta 重建 tokenizer。"""
        tok = cls.__new__(cls)
        tok.stoi = meta["stoi"]
        tok.itos = {i: ch for ch, i in tok.stoi.items()}
        tok.chars = sorted(tok.stoi.keys())
        tok.vocab_size = len(tok.chars)
        return tok
