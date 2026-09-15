"""RMSNorm（Root Mean Square Layer Normalization）从零实现。

「为什么换」：LayerNorm 做两步——① 减均值 re-center ② 除以标准差 re-scale，再加可学习
仿射 y = gamma·x̂ + beta。Zhang & Sennrich 2019（《Root Mean Square Layer Normalization》）
发现真正起作用的是「除以尺度」这一步，减均值这一步是冗余的：在 pre-norm + 残差结构下，
进入归一化层的信号天然近似零均值，把均值减掉对结果几乎没影响，却多算一次 mean 归约、
多存一个 beta 偏置。

RMSNorm 只保留 re-scale：
    y = (x / RMS(x)) · gamma，  RMS(x) = sqrt(mean(x²) + eps)
    - 不做减均值（省一次归约 + 省 beta 偏置的 dim 个参数）
    - 数值上用 x · rsqrt(mean(x²) + eps) 实现，rsqrt 单算子更稳（LLaMA/Mistral 同款写法）

对比 LayerNorm：LayerNorm 对 x 的「尺度 + 平移」都不变；RMSNorm 只对尺度不变（平移会被 x²
体现出来）。代价是丢掉平移不变性——实践中几乎无损，换来更少的算力和参数。现代模型
（LLaMA / Mistral / Qwen / Gemma）全部用 RMSNorm。

注：新版 PyTorch 已内置 `torch.nn.RMSNorm`（fused 生产实现，与 nn.LayerNorm 同级）。
本项目 model.py 实际使用内置版；本文件手写一份是为了看清 RMSNorm 内部逻辑
（与 nn.LayerNorm 逐行对照），仅供学习，不参与 model.py 的前向计算。
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS 归一化：y = x / RMS(x) · gamma，只 re-scale 不 re-center。

    底层参数（不暴露为构造参数，内部硬编码，备选见下）：
        eps = 1e-6  —— 加到 mean(x²) 上防除零。LLaMA/Mistral 用 1e-6；
                       LayerNorm 默认 1e-5（两者的 eps 语义不同，见模块 docstring）。
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        # gamma：只做缩放，没有 beta 偏置（这正是 RMSNorm 省参数的地方）
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (..., dim)，沿最后一维归一化
        # 等价于 x / sqrt(mean(x²)+eps)；写成 rsqrt(mean(x²)+eps) 是单算子、数值更稳
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


if __name__ == "__main__":
    torch.manual_seed(0)

    dim = 16
    rms = RMSNorm(dim)
    x = torch.randn(4, 8, dim)

    # 1) RMS 归一化：归一化后（不带 gamma）RMS = 1
    y_no_gamma = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms.eps)
    print("1) RMS 归一化: RMS(y) = {:.6f}  （期望 ≈ 1）".format(
        y_no_gamma.pow(2).mean(-1).sqrt().mean().item()))

    # 2) 尺度不变性：RMSNorm(αx) ≈ RMSNorm(x)（归一化抹掉幅值；eps 引入 ~1e-6 的极小误差）
    alpha = 3.7
    y1 = rms(x)
    y2 = rms(x * alpha)
    print("\n2) 尺度不变性: ||RMSNorm(αx) - RMSNorm(x)|| = {:.2e}  （期望 ≈ 0）".format(
        (y1 - y2).norm().item()))

    # 3) 不重新居中（与 LayerNorm 的关键区别）：
    #    LayerNorm 输出恒 mean=0；RMSNorm 不再强制，证明省掉了「减均值」这一步
    ln = nn.LayerNorm(dim)
    print("\n3) 是否重新居中：")
    print("   mean(LayerNorm(x)) = {:.2e}  （恒为 0）".format(ln(x).mean().item()))
    print("   mean(RMSNorm(x))  = {:.2e}  （不再强制为 0）".format(rms(x).mean().item()))

    # 4) 可微性
    x = torch.randn(4, 8, dim, requires_grad=True)
    rms(x).sum().backward()
    print("\n4) 可微性: 反向传播 OK，x.grad 是否非空 = {}".format(x.grad is not None))
