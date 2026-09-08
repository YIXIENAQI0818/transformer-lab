"""HF 版的 char-level tokenizer。

把 pretraining 里 CharTokenizer 的「数据 + 代码」翻译成 transformers 的 PreTrainedTokenizer 接口。

回顾之前的核心概念：tokenizer 也分「数据」和「代码」两半：
  - 数据：char <-> id 的映射表（stoi / itos，65 条）→ 存成 vocab.json
  - 代码：encode / decode 怎么查表、怎么拼字符串 → 就是这个类的几个方法

为什么要继承 PreTrainedTokenizer：
  HF 生态加载 tokenizer 用 AutoTokenizer.from_pretrained，它要求 tokenizer 是
  PreTrainedTokenizer 的子类（内置的 GPT2Tokenizer / BERTTokenizer 都是）。
  我们 char-level 不是标准类型，所以自定义一个子类，实现几个核心方法即可。

这就是「tokenizer 的代码部分」从我们的 tokenizer.py 变成这个类；
而「数据部分」（stoi/itos）通过 vocab.json 保存（JSON 能正确转义换行符等特殊字符）。
"""
import json
import os

from transformers import PreTrainedTokenizer


class HFCharTokenizer(PreTrainedTokenizer):
    """字符级 tokenizer：每个字符一个 token，encode 逐字符查表，decode 直接拼接。"""

    # 告诉 transformers：词表存成 vocab.json，加载时以 vocab_file 参数传入 __init__
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(self, vocab_file=None, vocab=None, **kwargs):
        # 两种构造方式：
        #   - 加载：AutoTokenizer.from_pretrained 传 vocab_file（vocab.json 的路径），自己读文件
        #   - 保存：转换脚本直接传 vocab（{char: id} 字典）
        if vocab_file is not None:
            with open(vocab_file, encoding="utf-8") as f:
                vocab = json.load(f)
        self.stoi = dict(vocab) if vocab else {}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def get_vocab(self):
        # HF 约定：返回 {token_str: id}
        return dict(self.stoi)

    # ---- 以下四个方法是「代码」部分：把 encode/decode 逻辑写在这里 ----

    def _tokenize(self, text):
        # char-level：每个字符就是一个 token
        return list(text)

    def _convert_token_to_id(self, token):
        return self.stoi.get(token)

    def _convert_id_to_token(self, index):
        return self.itos.get(index)

    def convert_tokens_to_string(self, tokens):
        # decode：直接拼接（char-level 没有分隔符）
        return "".join(tokens)

    # ---- 保存 / 加载「数据」部分 ----

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """把 stoi 存成 vocab.json（用 JSON 而非 vocab.txt，正确处理换行符等特殊字符）。"""
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False)
        return (vocab_file,)
