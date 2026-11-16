"""下载 Real-ESRGAN 官方权重到 training/weights/ 目录。

用法（在仓库根目录执行）：
    python training/tools/fetch_official_weights.py                  # 下 x4v3 + wdn-x4v3（官方默认组合）
    python training/tools/fetch_official_weights.py --all            # 下全部
    python training/tools/fetch_official_weights.py realesr-animevideov3
    python training/tools/fetch_official_weights.py --verify         # 下完顺带校验结构

为什么要下两个：官方 `realesr-general-x4v3` 默认 `--denoise_strength 0.5`，
推理时是把 x4v3 与 wdn-x4v3 两组权重按 [0.5, 0.5] 线性混合后跑一份网络（DNI）。
只下前一个，得到的结果不等于官方默认输出。

权重不进版本库（.gitignore 已排除），原因：x4v3 单个约 64 MB，仓库会迅速膨胀，
且官方 Releases 本身就是稳定分发的地址，重复提交没有意义。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # training/
WEIGHTS = ROOT / "weights"
BASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"

FILES: dict[str, str] = {
    # S 尺寸通用模型，主力
    "realesr-general-x4v3": f"{BASE}/v0.2.5.0/realesr-general-x4v3.pth",
    # 与上一个配套的"强去噪"权重，官方默认按 0.5 混进去
    "realesr-general-wdn-x4v3": f"{BASE}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    # XS 尺寸动漫视频模型
    "realesr-animevideov3": f"{BASE}/v0.2.5.0/realesr-animevideov3.pth",
    # RRDB 大模型，架构不同，仅备用
    "RealESRGAN_x2plus": f"{BASE}/v0.2.1/RealESRGAN_x2plus.pth",
}

DEFAULT_SET = ["realesr-general-x4v3", "realesr-general-wdn-x4v3"]


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {blocks * block_size / 1e6:.1f} MB")
        return
    done = min(blocks * block_size, total)
    pct = done / total * 100
    bar = "#" * int(pct / 4)
    sys.stdout.write(f"\r  [{bar:<25}] {pct:5.1f}%  {done / 1e6:.1f}/{total / 1e6:.1f} MB")
    sys.stdout.flush()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, force: bool = False) -> Path:
    if name not in FILES:
        raise KeyError(f"未知权重 {name}，可选：{list(FILES)}")
    dst = WEIGHTS / f"{name}.pth"
    if dst.exists() and not force:
        print(f"[skip] {dst.name} 已存在（{dst.stat().st_size / 1e6:.2f} MB），用 --force 重下")
        return dst

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    url = FILES[name]
    print(f"[get ] {name}")
    print(f"       {url}")
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    tmp.replace(dst)
    print(f"[ok  ] {dst}  {dst.stat().st_size / 1e6:.2f} MB")
    print(f"       sha256 {sha256_of(dst)}")
    return dst


def verify() -> int:
    """把已下载的权重灌进官方结构，确认 key 与形状全部对得上。"""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[warn] 没装 torch，跳过结构校验")
        return 0

    sys.path.insert(0, str(ROOT))
    from src.models.srvgg_official import PRESETS, build_official_model, load_official_weights

    failed = 0
    for name in PRESETS:
        path = WEIGHTS / f"{name}.pth"
        if not path.exists():
            continue
        model = build_official_model(name)
        try:
            info = load_official_weights(model, path, strict=True)
            print(f"[ok  ] {name}: {info['hit']}/{info['total']} 参数命中，参数量 {model.num_params():,}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name}: {exc}")
            failed += 1
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 Real-ESRGAN 官方权重")
    ap.add_argument("names", nargs="*", help=f"要下的权重名，默认 {DEFAULT_SET}")
    ap.add_argument("--all", action="store_true", help="下载全部")
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    ap.add_argument("--verify", action="store_true", help="下完校验结构与权重是否匹配")
    args = ap.parse_args()

    names = list(FILES) if args.all else (args.names or DEFAULT_SET)
    for name in names:
        download(name, force=args.force)
    return verify() if args.verify else 0


if __name__ == "__main__":
    raise SystemExit(main())
