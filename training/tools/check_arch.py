"""结构自检：不需要 GPU、不需要数据，纯 CPU 几秒钟跑完。

检查四件事：
  1. 官方结构与官方参数量一致（1,213,296）
  2. 改造版参数量符合预期（1,150,339）
  3. 改造版在初始态是**精确恒等映射**（max|out - in| == 0.0，不是"接近 0"）
  4. 官方权重能按 key + 形状过滤后载入改造版，并列出被丢弃的那一层

用法（在仓库根目录执行）：
    python training/tools/check_arch.py
    python training/tools/check_arch.py --weights training/weights/realesr-general-x4v3.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # training/
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    print("[FAIL] 没装 torch。这是唯一必须的依赖：pip install torch")
    raise SystemExit(1)

from src.models import (  # noqa: E402
    EXPECTED_PARAMS,
    build_model,
    build_official_model,
    unwrap_state_dict,
)

MODIFIED_PARAMS = 1_150_339


def check_official() -> bool:
    ok = True
    for name, expected in EXPECTED_PARAMS.items():
        model = build_official_model(name)
        actual = model.num_params()
        flag = "ok  " if actual == expected else "FAIL"
        print(f"[{flag}] 官方 {name:24s} 参数量 {actual:>9,}  期望 {expected:>9,}")
        ok &= actual == expected
    return ok


def check_modified() -> bool:
    model = build_model(scale=1)
    actual = model.num_params()
    ok = actual == MODIFIED_PARAMS
    print(f"[{'ok  ' if ok else 'FAIL'}] 改造版 scale=1              参数量 {actual:>9,}  期望 {MODIFIED_PARAMS:>9,}")

    # 初始态必须是精确恒等：末层置零 → body(x)=0 → out = 0 + x = x
    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, 32, 32)
        diff = float((model(x) - x).abs().max())
    identity_ok = diff == 0.0
    print(f"[{'ok  ' if identity_ok else 'FAIL'}] 初始态恒等映射            max|out-in| = {diff}")
    return ok and identity_ok


def check_cross_load(weights: Path) -> bool:
    if not weights.exists():
        print(f"[skip] 权重不存在：{weights}（先跑 fetch_official_weights.py）")
        return True

    modified = build_model(scale=1)
    state = unwrap_state_dict(torch.load(str(weights), map_location="cpu"))
    mine = modified.state_dict()

    hit = {k: v for k, v in state.items() if k in mine and v.shape == mine[k].shape}
    dropped = [k for k in state if k not in hit]

    modified.load_state_dict({**mine, **hit})
    print(f"[ok  ] 官方权重载入改造版  命中 {len(hit)}/{len(mine)} 个参数")
    if dropped:
        print(f"       丢弃 {len(dropped)} 个：{dropped}")
        for key in dropped[:3]:
            print(f"         {key:20s} 官方 {tuple(state[key].shape)} → 本版无对应位置")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="模型结构自检")
    ap.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "weights" / "realesr-general-x4v3.pth",
        help="官方权重的路径，用于交叉载入检查",
    )
    args = ap.parse_args()

    print("=" * 68)
    ok = check_official()
    ok &= check_modified()
    ok &= check_cross_load(args.weights)
    print("=" * 68)
    print("结论：" + ("全部通过" if ok else "存在不一致，见上面 FAIL 行"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
