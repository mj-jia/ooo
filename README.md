# 图片变清晰 · Real-ESRGAN

把模糊、压缩过、分辨率低的图片放大 4 倍并补出细节。全部计算在浏览器里完成——图片不上传、不用装软件、不用注册。

拖一张图进去，点一下按钮，下载 PNG。

## 在线使用

部署在 GitHub Pages：`https://mj-jia.github.io/ooo/`

## 两个模型怎么选

| 选项 | 模型 | 体积 | 适用 |
| --- | --- | --- | --- |
| 均衡（默认） | realesr-general-x4v3 | 4.7 MB | 日常用，几秒出图，手机也能跑 |
| 高质量 | Real-ESRGAN x4plus | 66 MB | 追求细节，慢数倍，建议有独立显卡 |

模型只在第一次使用时下载，之后缓存在浏览器 Cache Storage 里，再打开不用重复下载。

## 它是怎么工作的

1. 图片按 EXIF 方向解码，长边超过模型上限（均衡 1280 / 高质量 640）会等比缩小
2. 按 256×256 切块，每块向四周多读 24 像素上下文，边缘用镜像反射补齐
3. 每块以 NCHW 浮点张量送入 ONNX 模型，得到 4 倍输出，裁掉那 24 像素（放大后 96 像素）的边距
4. 拼回整图、量化到 8 bit；原图有透明通道时按同比例放大保留 alpha
5. 结果交给 Canvas 显示、对比、导出

分块是为了让大图也跑得动——整图一次推理会直接把内存打爆。

推理后端优先 **WebGPU**（Chrome / Edge 113+），不可用时回退 **WebAssembly**。

## 本地运行

页面用了 ES module 和 `fetch`，必须经 http 打开，双击 html 文件不行：

```bash
cd ooo
python -m http.server 8123 --bind 127.0.0.1
# 浏览器打开 http://127.0.0.1:8123/
```

## 目录

```
index.html          页面结构
style.css           样式
app.js              模型管理、分块推理、交互
models/
  realesr-general-x4v3.onnx   均衡模型，4.7 MB
  upscaler-x4.onnx            高质量模型，66 MB
vendor/             ONNX Runtime Web 1.31.1（WebGPU + WASM）
tools/
  reference_run.py  Python 参考实现，复用同一套分块逻辑，用于对拍
  e2e.mjs           无头浏览器冒烟测试
  inspect_model.py  打印模型输入输出与算子构成
```

`tools/` 只用于开发验证，跑页面不需要。

## 实测数据

**Python + onnxruntime CPU（多线程），同一套分块逻辑：**

| 模型 | 输入 | 分块 | 耗时 |
| --- | --- | --- | --- |
| x4v3 | 256 × 256 | 1 | 0.8–2.1 s |
| x4plus | 256 × 256 | 1 | 10.2 s |
| x4plus | 520 × 380 | 6 | 36.6 s |

**无头 Edge（无 GPU，WebGPU 走软件后端）实测：**

| 模型 | 输入 | 输出 | 端到端耗时 |
| --- | --- | --- | --- |
| x4v3 | 128 × 128 | 512 × 512 | 42.3 s |
| x4plus | 48 × 48 | 192 × 192 | 178.9 s |

这两个数字是**没有显卡**的最坏情况。真机开 WebGPU 会快一个量级。

分块接缝已用数值方式检查：拼接处的相邻列梯度与周边差异在图像自身纹理波动范围内，无可见断层。

## 模型与许可

- 模型权重：Real-ESRGAN，作者 Xintao Wang 等，BSD-3-Clause
  - `upscaler-x4.onnx` 来自 [smile-forge/real-esrgan-gfpgan-onnx](https://huggingface.co/smile-forge/real-esrgan-gfpgan-onnx)（fp32，动态尺寸，带内建归一化）
  - `realesr-general-x4v3.onnx` 来自 [CoderViking/realesr-general-x4v3-onnx](https://huggingface.co/CoderViking/realesr-general-x4v3-onnx)
- 推理运行时：ONNX Runtime Web 1.31.1，MIT
- 本项目代码：MIT，见 `LICENSE`

## 已知限制

- 输入长边超过上限会先被缩小，不是「原图直接放大」
- 不做人脸专门修复（GFPGAN 那类模型 341 MB，不适合放进纯前端页面）
- 只处理静态图，不支持动图 GIF / HEIC
- 结果是模型生成的内容，原图里没有的文字、纹理可能被「编」出来，不适合作为证据类图片使用
- GitHub Pages 无法设置 COOP/COEP 响应头，WASM 只能单线程，这也是优先 WebGPU 的原因
