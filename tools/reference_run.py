#!/usr/bin/env python
"""Python 参考实现：用 onnxruntime 跑与网页端完全相同的分块逻辑。

用途：
  1. 浏览器跑不动时，用它确认模型本身没问题
  2. 与网页输出逐像素对拍，隔离「是模型糊还是拼接错」
  3. 给出 CPU 上的耗时基准，作为浏览器侧的性能下限参考

用法：
    python tools/reference_run.py                          # 内置测试图 + 默认模型
    python tools/reference_run.py photo.jpg                # 指定输入
    python tools/reference_run.py photo.jpg -o out.png
    python tools/reference_run.py photo.jpg -m models/realesr-general-x4v3.onnx
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tools" / "out"

SCALE = 4
TILE = 256
MARGIN = 24


def reflect_index(idx: np.ndarray, n: int) -> np.ndarray:
    """越界索引折回图内（与 src/app.js 的 reflect() 等价）。"""
    if n <= 1:
        return np.zeros_like(idx)
    period = 2 * n - 2
    j = np.mod(idx, period)
    return np.where(j < n, j, period - j)


def upscale(arr: np.ndarray, sess, tile: int = TILE, margin: int = MARGIN) -> np.ndarray:
    """arr: HxWx3 uint8 -> (H*SCALE) x (W*SCALE) x 3 uint8"""
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    h, w = arr.shape[:2]
    out = np.zeros((h * SCALE, w * SCALE, 3), dtype=np.uint8)
    tiles_y = (h + tile - 1) // tile
    tiles_x = (w + tile - 1) // tile
    total = tiles_y * tiles_x
    done = 0
    t0 = time.time()

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            x0, y0 = tx * tile, ty * tile
            tw, th = min(tile, w - x0), min(tile, h - y0)
            pw, ph = tw + 2 * margin, th + 2 * margin

            ys = reflect_index(np.arange(y0 - margin, y0 - margin + ph), h)
            xs = reflect_index(np.arange(x0 - margin, x0 - margin + pw), w)
            patch = arr[ys][:, xs].astype(np.float32) / 255.0
            tensor = np.transpose(patch, (2, 0, 1))[None]

            res = sess.run([out_name], {in_name: tensor})[0]
            res = np.clip(res[0], 0.0, 1.0)

            om = margin * SCALE
            ow, oh = pw * SCALE, ph * SCALE
            core = res[:, om:oh - om, om:ow - om]
            out[y0 * SCALE:(y0 + th) * SCALE, x0 * SCALE:(x0 + tw) * SCALE] = (
                np.transpose(core, (1, 2, 0)) * 255.0 + 0.5
            ).astype(np.uint8)

            done += 1
            print(f"  tile {done}/{total}  {time.time() - t0:.1f}s", flush=True)

    print(f"  总耗时 {time.time() - t0:.1f}s")
    return out


def build_test_image() -> Image.Image:
    """没有输入图时生成一张可复现的低清测试图。"""
    for candidate in ("C:/Windows/Web/Wallpaper/Windows/img0.jpg", "/usr/share/pixmaps/debian-logo.png"):
        path = Path(candidate)
        if path.exists():
            return Image.open(path).convert("RGB").resize((256, 256), Image.LANCZOS)
    return Image.effect_mandelbrot((512, 512), (-2, -1.2, 1, 1.2), 60).convert("RGB").resize((256, 256), Image.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-ESRGAN 分块推理参考实现")
    parser.add_argument("input", nargs="?", help="输入图片路径，省略则用内置测试图")
    parser.add_argument("-o", "--output", default="output.png", help="输出文件名（写入 tools/out/）")
    parser.add_argument("-m", "--model", default="models/upscaler-x4.onnx", help="模型路径，默认高质量 x4plus")
    parser.add_argument("--tile", type=int, default=TILE, help=f"分块边长，默认 {TILE}")
    parser.add_argument("--margin", type=int, default=MARGIN, help=f"上下文边距，默认 {MARGIN}")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        print(f"模型不存在：{model_path}")
        return 1

    src = Image.open(args.input).convert("RGB") if args.input else build_test_image()
    src.save(OUT_DIR / "input.png")
    print(f"输入 {src.size[0]}x{src.size[1]}")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print(f"模型 {model_path.name}　输入 {sess.get_inputs()[0].name}，输出 {sess.get_outputs()[0].name}")

    t0 = time.time()
    out = upscale(np.array(src), sess, tile=args.tile, margin=args.margin)
    print(f"整体 {time.time() - t0:.1f}s，输出 {out.shape[1]}x{out.shape[0]}")

    dst = OUT_DIR / args.output
    Image.fromarray(out).save(dst)
    print(f"已保存 {dst}")

    bicubic = np.array(src.resize((src.size[0] * SCALE, src.size[1] * SCALE), Image.BICUBIC))
    Image.fromarray(bicubic).save(OUT_DIR / "bicubic.png")
    print(f"对照图 {OUT_DIR / 'bicubic.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
