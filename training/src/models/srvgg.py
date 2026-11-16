"""Real-ESRGAN 轻量骨干 SRVGGNetCompact 的改造版（同尺寸复原，scale=1）。

与同目录 srvgg_official.py 的三处结构差异，全部由「×4 超分 → scale=1 复原」引起：

    conv 层数     官方 1 + 32 + 1 = 34    →  本版 1 + (32-1) + 1 = 33
    输出层        64 → 48（3×4×4）        →  64 → 3
    上采样        PixelShuffle(4)         →  Identity
    残差          + nearest(x, 4)         →  + x
    参数量        1,213,296               →  1,150,339

参数量并不相等（差 62,957，来自少一层 64→64 36,928 + 输出层 27,696→1,731 25,965
+ 少一个 PReLU 64），只算「同级」。

权重兼容性：`body.0` ~ `body.62` 共 32 层 conv 及对应 PReLU 形状完全一致，可以从
官方 .pth 直接载入；官方第 33 层（body.64，64→64）在本版位置被输出层占用，载入时
按形状过滤丢弃。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _act(kind: str, num_feat: int) -> nn.Module:
    if kind == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"未知激活 {kind}")


class SRVGGNetCompact(nn.Module):
    """ECCV 2022 Real-ESRGAN 的 compact 生成器，scale=1 版本。

    scale=1 时不接 PixelShuffle，残差直接加输入，输出与输入同尺寸。
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 1,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)]
        body.append(_act(act_type, num_feat))
        for _ in range(num_conv - 1):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(_act(act_type, num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()

        self._init()

    def _init(self) -> None:
        convs = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        for m in convs:
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # 最后一层置零：让初始输出严格等于输入（恒等映射），
        # 训练起点就是"什么都不做"，不会一上来就把图弄坏。
        # 注意：新增模块必须在 _init() 之后挂载，否则 convs[-1] 会指向新模块，
        # 输出卷积保留 kaiming 初始化，恒等起点被破坏。
        if convs:
            nn.init.zeros_(convs[-1].weight)
            if convs[-1].bias is not None:
                nn.init.zeros_(convs[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x)
        out = self.upsampler(out)
        return out + x  # 残差学习：模型只学"要补回什么"

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(scale: int = 1, num_feat: int = 64, num_conv: int = 32) -> SRVGGNetCompact:
    return SRVGGNetCompact(upscale=scale, num_feat=num_feat, num_conv=num_conv)


__all__ = ["SRVGGNetCompact", "build_model"]
