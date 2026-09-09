"""SwiGLU（Swish-Gated Linear Unit）门控 FFN 从零实现。

「为什么换」：标准 FFN 用 GELU 做非线性，但 GELU 是一个「写死的开关」——它对每个值套
同一个固定函数，只认数值大小（大就放行、负就关掉），所有维度、所有输入一视同仁。

SwiGLU 引入「门控」（gating）：把「信息多大」和「该不该放行」解耦成两条独立的路径，
其中「放行多少」由另一套可学习权重决定，模型能在训练中学到「什么情况下、该把哪个维度
的信息关掉或放大」。表达式力更强，Shazeer 2020《GLU Variants Improve Transformer》验证
loss 更低，LLaMA / Qwen / Mistral 等现代模型全部使用。

数学形式（⊙ 为逐元素乘）：
    Swish(x)    = x · σ(x)                      # 也叫 SiLU，σ 是 sigmoid
    SwiGLU(x)   = Swish(xW_gate) ⊙ (xW_up)      # 门控：gate 决定 value 每维放行多少
                → 再过输出投影 W_down

关键细节 —— 中间维度为什么是 8/3·d：
    SwiGLU 有 3 个投影（gate / up / down），标准 GELU FFN 只有 2 个（升维 / 降维）。
    为了让两者参数量对齐（这样对比 loss 才公平，优势只能归因于门控结构而非参数更多）：
        GELU  FFN：2 个矩阵，参数量 2 · (d · 4d) = 8d²
        SwiGLU   ：3 个矩阵，参数量 3 · (d · m)
        令 3dm = 8d²  →  m = 8d/3
    这就是 LLaMA 的 SwiGLU 中间维度取 8/3·d（约 2.67 倍）而非 4 倍的原因。

注：PyTorch 核心库没有内置 nn.SwiGLU（只有相近的 nn.GLU，那是 sigmoid 门控、非 Swish）。
Swish/SiLU 本身有内置（nn.SiLU / F.silu）。本文件手写 SwiGLU 是三个 nn.Linear + 一次
F.silu + 一次逐元素乘的组合，与 model.py 中 MLP 的 swiglu 分支逻辑一致，仅供学习与验证。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    """Swish 激活，又名 SiLU：x · sigmoid(x)。"""
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """门控 FFN：SwiGLU(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))。

    - gate_proj：门投影，过 SiLU 得到「每个维度放行多少」
    - up_proj  ：值投影，承载「要传递的信息」，不过激活
    - down_proj：输出投影，把门控结果降回 dim 维

    hidden_dim 默认 8/3·dim（对齐标准 4·dim FFN 的参数量，见模块 docstring）。
    """

    def __init__(self, dim, hidden_dim=None, bias=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (8 * dim) // 3  # 8/3·d，整数运算避免浮点误差
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.down_proj(swish(self.gate_proj(x)) * self.up_proj(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    dim = 128  # 对齐 model.py 的 n_embd，参数量数字有实际意义

    # 1) Swish 形状：平滑、有负值区、非单调（与 GELU 形似，但负值更深）
    print("1) Swish 激活形状（关键点）:")
    for v in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        print("   swish({:4.1f}) = {:8.4f}".format(v, swish(torch.tensor(v)).item()))
    print("   （x→-∞ 趋近 0 但保留负值；x=0 处为 0；x→+∞ 趋近 x）")

    # 2) 门控语义：gate 全关 -> 输出全 0（门控能完全掐掉信息，GELU 做不到「想关就关」）
    #    用 bias=False 隔离出门控本身：否则 down_proj 的 bias 会在 gate 关死时仍输出非零。
    sg = SwiGLU(dim, bias=False)
    with torch.no_grad():
        sg.gate_proj.weight.zero_()
    x = torch.randn(4, 8, dim)
    y_off = sg(x)
    print("\n2) 门控语义: gate 全 0（关死）时输出范数 = {:.2e}  （期望 ≈ 0）".format(
        y_off.norm().item()))

    # 3) 参数量对齐：SwiGLU(8/3·d) ≈ GELU FFN(4·d)
    gelu_ffn = nn.Sequential(
        nn.Linear(dim, 4 * dim),
        nn.GELU(),
        nn.Linear(4 * dim, dim),
    )
    swiglu_ffn = SwiGLU(dim)  # hidden = (8*128)//3 = 341
    n_gelu = sum(p.numel() for p in gelu_ffn.parameters())
    n_swiglu = sum(p.numel() for p in swiglu_ffn.parameters())
    print("\n3) 参数量对齐:")
    print("   GELU  FFN(4·d)    = {} 参数".format(n_gelu))
    print("   SwiGLU(8/3·d)    = {} 参数".format(n_swiglu))
    print("   差 = {} （8/3 取整的微小差异，占总参数 ~0.1%）".format(n_swiglu - n_gelu))

    # 4) 可微性
    x = torch.randn(4, 8, dim, requires_grad=True)
    SwiGLU(dim)(x).sum().backward()
    print("\n4) 可微性: 反向传播 OK，x.grad 是否非空 = {}".format(x.grad is not None))
