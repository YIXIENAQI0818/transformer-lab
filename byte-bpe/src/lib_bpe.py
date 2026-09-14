"""用 HuggingFace tokenizers 库训练 byte-level BPE，包装成与自写 BpeTokenizer 相同的接口。

这一步验证「tokenizer 可替换」：工业库的 tokenizer 只要实现 encode / decode / vocab_size
三个接口，就能和自写版一样接入模型训练，训练脚本无需改任何一行。

和自写 BpeTokenizer 的差异：
  - 自写：直接在原始字节 id（0~255）上做 BPE，encode 返回 list，vocab_size 是 property。
  - 库版：ByteLevel 在 byte->unicode 映射后的字符上做 BPE，encode 返回 Encoding 对象（要 .ids），
    vocab_size 是方法（get_vocab_size）。本文件的 LibBpeTokenizer 把这些统一成自写版的接口。
"""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_lib_bpe(text, vocab_size):
    """用 tokenizers 库训练 byte-level BPE，返回训练好的 Tokenizer。

    initial_alphabet 显式设为全部 256 个字节字符，与自写版的「256 字节兜底」对齐。
    """
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([text], trainer=trainer)
    return tok


class LibBpeTokenizer:
    """包装 tokenizers 库的 ByteLevel BPE，暴露与 BpeTokenizer 相同的接口。

    encode(text) -> list[int]、decode(ids) -> str、vocab_size（property）。
    这样 train_model.py 里 hand / lib 两种 tokenizer 可无缝替换。
    """

    def __init__(self, tok):
        self._tok = tok

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)
