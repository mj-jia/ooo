"""参考实现：用 Python + onnxruntime 跑与网页端完全相同的 tiling 逻辑，
用来验证模型可用、分块拼接无缝，并给出 CPU 上的耗时基准。

用法：
    python tools/reference_run.py [输入图] [输出图]
不传参数时用系统壁纸生成一张 256x256 的低清测试图。
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "upscaler-x4.onnx"
OUT_DIR = ROOT / "tools" / "out"

SCALE = 4
TILE = 256
MARGIN = 24


def reflect_index(idx: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros_like(idx)
    period = 2 * n - 2
    j = np.mod(idx, period)
    return np.where(j < n, j, period - j)


def upscale(arr: np.ndarray, sess, tile=TILE, margin=MARGIN) -> np.ndarray:
    """arr: HxWx3 uint8 -> (H*SCALE)x(W*SCALE)x3 uint8"""
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
            patch = arr[ys][:, xs].astype(np.float32) / 255.0       # ph x pw x 3
            tensor = np.transpose(patch, (2, 0, 1))[None]            # 1 x 3 x ph x pw

            res = sess.run(["rgb_out"], {"rgb": tensor})[0]          # 1 x 3 x 4ph x 4pw
            res = np.clip(res[0], 0.0, 1.0)

            om = margin * SCALE
            ow, oh = pw * SCALE, ph * SCALE
            core = res[:, om:oh - om, om:ow - om]                    # 3 x 4th x 4tw
            out[y0 * SCALE:(y0 + th) * SCALE, x0 * SCALE:(x0 + tw) * SCALE] = (
                np.transpose(core, (1, 2, 0)) * 255.0 + 0.5
            ).astype(np.uint8)

            done += 1
            print(f"  tile {done}/{total}  {time.time() - t0:.1f}s", flush=True)

    print(f"  总耗时 {time.time() - t0:.1f}s")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        src = Image.open(sys.argv[1]).convert("RGB")
    else:
        wp = Path("C:/Windows/Web/Wallpaper/Windows/img0.jpg")
        if wp.exists():
            src = Image.open(wp).convert("RGB")
        else:
            src = Image.effect_mandelbrot((512, 512), (-2, -1.2, 1, 1.2), 60).convert("RGB")
        src = src.resize((256, 256), Image.LANCZOS)

    src_path = OUT_DIR / "input.png"
    src.save(src_path)
    print(f"输入 {src.size[0]}x{src.size[1]} -> {src_path}")

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"模型输入 {inp.name} {inp.shape}，输出 {sess.get_outputs()[0].shape}")

    t0 = time.time()
    out = upscale(np.array(src), sess)
    print(f"整体 {time.time() - t0:.1f}s，输出 {out.shape[1]}x{out.shape[0]}")

    dst = OUT_DIR / (sys.argv[2] if len(sys.argv) > 2 else "output.png")
    Image.fromarray(out).save(dst)
    print(f"已保存 {dst}")

    # 与双三次放大做对照，粗略量化提升
    bicubic = np.array(src.resize((src.size[0] * SCALE, src.size[1] * SCALE), Image.BICUBIC))
    Image.fromarray(bicubic).save(OUT_DIR / "bicubic.png")


if __name__ == "__main__":
    main()
