"""模型定义：官方原版 + 本项目改造版。

    srvgg_official.SRVGGNetCompact   ×2 / ×4 超分，34 层，1,213,296 参数
    srvgg.SRVGGNetCompact            scale=1 同尺寸复原，33 层，1,150,339 参数

两者同名类，导出时用别名区分，避免混用：

    from src.models import build_official_model, build_model
"""

from . import srvgg, srvgg_official
from .srvgg import SRVGGNetCompact as SRVGGNetCompactModified, build_model
from .srvgg_official import (
    EXPECTED_PARAMS,
    PRESETS,
    SRVGGNetCompact as SRVGGNetCompactOfficial,
    build_official_model,
    load_official_weights,
    verify_arch,
)

__all__ = [
    "srvgg",
    "srvgg_official",
    "SRVGGNetCompactModified",
    "SRVGGNetCompactOfficial",
    "build_model",
    "build_official_model",
    "load_official_weights",
    "verify_arch",
    "PRESETS",
    "EXPECTED_PARAMS",
]
