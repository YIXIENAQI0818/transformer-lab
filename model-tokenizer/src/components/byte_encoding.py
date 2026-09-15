"""byte-level 编码：文本 ↔ UTF-8 字节序列（步骤 01）。

byte-level BPE 的「byte-level」层：在字节（0~255 闭集）层面操作。
任何 Unicode 字符都是 UTF-8 的 1~4 个字节，字节永远只有 256 种，
因此任何输入都能编码——从原理上消灭 OOV。

对比 char-level 动态词表（见 model-core/scripts-training/tokenizer.py 的 CharTokenizer）的致命缺陷：
词表 = 训练数据里出现过的字符，推理时遇到训练集外的字符（中文 / emoji / 罕见符号）
直接 KeyError（OOV）。byte-level 用「256 个字节兜底」解决这个问题。

运行：python -m src.components.byte_encoding
"""


def text_to_bytes(text: str) -> list[int]:
    """str -> UTF-8 字节序列（0~255 的 int 列表）。

    每个 Unicode 字符被编码成 1~4 个字节：ASCII 1 字节、多数欧洲 / 中东文字 2 字节、
    中文 3 字节、emoji 4 字节。字节值永远落在 0~255 闭集内。
    """
    return list(text.encode("utf-8"))


def bytes_to_text(ids) -> str:
    """0~255 的字节序列 -> str（UTF-8 解码）。

    errors="replace" 兜底：非法字节序列（如截断的多字节字符）用 U+FFFD 替换而非抛异常，
    保证 decode 永远有定义——这是「永不 OOV」在 decode 侧的对应（编码侧是 256 闭集）。
    """
    return bytes(ids).decode("utf-8", errors="replace")


def roundtrip(text: str) -> str:
    """text -> bytes -> text，验证编码可逆（应恒等于原文本）。"""
    return bytes_to_text(text_to_bytes(text))


if __name__ == "__main__":
    # 1) 三类字符的字节表示：ASCII 1 字节、中文 3 字节、emoji 4 字节
    samples = [
        ("ASCII", "A"),           # 1 字节
        ("中文", "中"),            # 3 字节
        ("emoji", "🙂"),          # 4 字节
        ("混合", "Hi 中 🙂"),      # 1 + 3 + 4 混合
    ]
    print("1) 不同字符的 UTF-8 字节表示（字节值都落在 0~255 闭集内）:")
    for label, ch in samples:
        b = text_to_bytes(ch)
        print("   {:<6} {:>12} -> 字节 {}（{} 字节 / {} 字符）".format(
            label, repr(ch), b, len(b), len(ch)))

    # 2) 字节闭集：任何字符序列都能编码成 0~255，编码侧永不失败
    all_kinds = "ASCII 字母 + 中文汉字 + emoji 🙂🎉 + 罕见符号 ꙮ𠀀"
    b_all = text_to_bytes(all_kinds)
    assert all(0 <= x <= 255 for x in b_all), "字节值必须都在 0~255"
    print("\n2) 任意文本都能编码成 0~255 字节：{} 字符 -> {} 字节，全部落在 [0,255]".format(
        len(all_kinds), len(b_all)))

    # 3) round-trip：编码再解码 = 原文本（编码可逆）
    for text in ["A", "中", "🙂", "Hi 中 🙂", all_kinds]:
        assert roundtrip(text) == text, text
    print("\n3) round-trip: text -> bytes -> text 恒等（编码可逆）✔")

    # 4) 对比 char-level 的 OOV：用一段纯 ASCII 语料建 char 词表（同 CharTokenizer 逻辑），
    #    再对训练集外字符（中文 / emoji）做 char encode -> KeyError；byte-level -> 成功。
    ascii_corpus = "First Citizen: Before we proceed any further, hear me speak."
    char_vocab = sorted(set(ascii_corpus))  # 训练集里出现过的字符（同 CharTokenizer）
    stoi = {ch: i for i, ch in enumerate(char_vocab)}

    def char_encode(s):
        return [stoi[ch] for ch in s]  # 遇到词表外字符直接 KeyError

    print("\n4) char-level vs byte-level 的 OOV 对比:")
    print("   char 词表（来自纯 ASCII 语料）: {} 个字符".format(len(char_vocab)))
    for ch in ["F", "中", "🙂"]:  # "F" 在语料词表内（"First"），"中"/"🙂" 在词表外
        try:
            char_encode(ch)
            char_result = "成功"
        except KeyError:
            char_result = "KeyError（OOV）"
        byte_result = "{} 字节".format(len(text_to_bytes(ch)))
        print("   字符 {:>4}: char-level {:<18} byte-level {}".format(
            repr(ch), char_result, byte_result))
