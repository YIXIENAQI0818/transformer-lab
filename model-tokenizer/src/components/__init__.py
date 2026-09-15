"""手写 BPE 参考实现：byte-level 编码 + 手写 BPE 训练 / encode / decode。

训练实际用 lib_bpe（官方 tokenizers 库的包装），这里的手写版用于理解原理：
  - byte_encoding：闭集 256 字节如何从原理上消除 OOV
  - bpe：BPE 合并规则怎么学、byte→子词→id 怎么走
compare_tokenizers.py 用它与库版做「手写 vs 库」对照。
"""
