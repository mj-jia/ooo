"""Real-ESRGAN 官方 SRVGGNetCompact 的等价实现（不依赖 basicsr）。

上游：xinntao/Real-ESRGAN · realesrgan/archs/srvgg_arch.py
许可：BSD-3-Clause, Copyright (c) 2021, Xintao Wang
      https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE

本文件与上游**逐层等价**，只做三处工程化包装：
  1. 去掉 `@ARCH_REGISTRY.register()` 与 basicsr 导入（本项目不装 basicsr）
  2. 激活函数抽成 `_make_activation()`，与上游三处 if/elif 完全等价
  3. 补 `build_official_model()` / `load_official_weights()` / `dni_interpolate()`

与同目录 srvgg.py 的关系
--------------------------
    srvgg.py            本项目改造版：scale=1 同尺寸复原，33 层 conv，1,150,339 参数
    srvgg_official.py   官方原版：×2 / ×4 超分，34 层 conv，1,213,296 参数

两者 state_dict 的 `body.0` ~ `body.62` 形状相同、可互相加载，
但输出层通道数与残差形式不同，**不能整体互换**。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(act_type: str, num_feat: int) -> nn.Module:
    """与上游三处 if/elif 完全等价的激活函数工厂。"""
    if act_type == "relu":
        return nn.ReLU(inplace=True)
    if act_type == "prelu":
        return nn.PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"未知激活 {act_type}")


class SRVGGNetCompact(nn.Module):
    """Real-ESRGAN 的轻量 VGG 式骨干，上游原版。

    结构要点（与上游一致，改动会破坏权重对应关系）：
      - body 用 ModuleList 顺序展开：首层 conv + act，然后 `num_conv` 组 conv + act，最后输出层
        所以 conv 总数 = 1 + num_conv + 1 = num_conv + 2
      - 输出层通道数 = num_out_ch * upscale^2，紧接 PixelShuffle(upscale) 完成上采样
      - 残差是 **最近邻放大 upscale 倍的输入**，不是原输入
        （只有 upscale=1 时才退化为 `+x`）

    Args:
        num_in_ch: 输入通道，3
        num_out_ch: 输出通道，3
        num_feat: 中间特征通道，64
        num_conv: 中间 conv 组数，16（XS）/ 32（S）
        upscale: 上采样倍数，2 或 4
        act_type: prelu / relu / leakyrelu
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        body: list[nn.Module] = []
        # 首层
        body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        body.append(_make_activation(act_type, num_feat))
        # 中间层
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(_make_activation(act_type, num_feat))
        # 输出层
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))

        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # 加上最近邻放大的输入，让网络学残差
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------
# 官方模型预设（参数取自官方 inference_realesrgan.py，不是猜的）
# --------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    # S 尺寸，通用场景，官方默认带 DNI 去噪
    "realesr-general-x4v3": dict(num_feat=64, num_conv=32, upscale=4),
    # XS 尺寸，动漫视频
    "realesr-animevideov3": dict(num_feat=64, num_conv=16, upscale=4),
}

# 参数量断言表：结构一旦写错（层数 off-by-one、输出通道算错）会直接失败，
# 这是"与官方等价"最硬的证据，不需要 GPU 也能验证。
EXPECTED_PARAMS: dict[str, int] = {
    "realesr-general-x4v3": 1_213_296,
    "realesr-animevideov3": 621_424,
}


def build_official_model(name: str = "realesr-general-x4v3", **overrides) -> SRVGGNetCompact:
    """按官方预设构造模型。

    >>> m = build_official_model("realesr-general-x4v3")
    >>> m.num_params()
    1213296
    """
    if name not in PRESETS:
        raise KeyError(f"未知预设 {name}，可选：{list(PRESETS)}")
    cfg = dict(PRESETS[name])
    cfg.update(overrides)
    return SRVGGNetCompact(**cfg)


# --------------------------------------------------------------------------
# 权重加载
# --------------------------------------------------------------------------
def unwrap_state_dict(raw):
    """官方 .pth 有两种包装：`params_ema`（EMA）和 `params`，也可能直接是裸 dict。"""
    if isinstance(raw, dict):
        for key in ("params_ema", "params"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
    return raw


def dni_interpolate(sd_a: dict, sd_b: dict, weight_a: float, weight_b: float) -> dict:
    """Deep Network Interpolation：两组权重线性插值。

    官方 x4v3 默认 `denoise_strength=0.5`，即把
    `realesr-general-x4v3` 与 `realesr-general-wdn-x4v3` 按 [0.5, 0.5] 混合后再推理。
    只加载 x4v3 一个文件，得到的结果**不等于**官方默认输出。
    """
    merged: dict = {}
    for key, value in sd_a.items():
        other = sd_b.get(key)
        if other is not None and value.is_floating_point() and other.shape == value.shape:
            merged[key] = value * weight_a + other * weight_b
        else:
            merged[key] = value
    return merged


def load_official_weights(
    model: SRVGGNetCompact,
    ckpt,
    dni_weight: float | None = None,
    strict: bool = True,
) -> dict:
    """加载官方权重。

    Args:
        model: 由 build_official_model() 构造的模型
        ckpt: 单个路径，或 [x4v3, wdn_x4v3] 两个路径
        dni_weight: 仅当 ckpt 是两元素列表时生效，对应官方 `--denoise_strength`。
                    None 表示不插值（直接用第一个）。
        strict: True 时要求 key 完全对应，False 时只报告缺失项

    Returns:
        dict: 加载统计 `{"hit": n, "miss": [...], "extra": [...], "total": n}`
    """
    paths = [ckpt] if isinstance(ckpt, (str, bytes)) or hasattr(ckpt, "__fspath__") else list(ckpt)
    state = unwrap_state_dict(torch.load(str(paths[0]), map_location="cpu"))

    if len(paths) == 2 and dni_weight is not None:
        second = unwrap_state_dict(torch.load(str(paths[1]), map_location="cpu"))
        state = dni_interpolate(state, second, float(dni_weight), 1.0 - float(dni_weight))

    mine = model.state_dict()
    miss = [k for k in mine if k not in state or state[k].shape != mine[k].shape]
    extra = [k for k in state if k not in mine]

    if miss and strict:
        raise RuntimeError(
            f"有 {len(miss)} 个参数对不上，结构可能与权重不匹配。"
            f"前几个：{miss[:5]}。确认用的是 build_official_model() 而不是 build_model()。"
        )

    model.load_state_dict(state, strict=False)
    return {"hit": len(mine) - len(miss), "miss": miss, "extra": extra, "total": len(mine)}


def verify_arch(name: str = "realesr-general-x4v3") -> int:
    """构造模型并断言参数量与官方一致，返回实际参数量。"""
    model = build_official_model(name)
    actual = model.num_params()
    expected = EXPECTED_PARAMS[name]
    if actual != expected:
        raise AssertionError(f"{name} 参数量 {actual:,} != 官方 {expected:,}，结构不等价")
    return actual


__all__ = [
    "PRESETS",
    "EXPECTED_PARAMS",
    "SRVGGNetCompact",
    "build_official_model",
    "dni_interpolate",
    "load_official_weights",
    "unwrap_state_dict",
    "verify_arch",
]
