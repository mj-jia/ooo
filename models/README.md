# 模型

两个模型都是 **Real-ESRGAN** 的 ONNX 导出，原生 4 倍放大，动态输入尺寸（长宽可任意，无需是 32 的倍数）。
清单、校验和与结构信息见 [`manifest.json`](manifest.json)，用 `npm run verify` 可以核对文件是否被改动。

| 文件 | 体积 | 结构 | 输入 → 输出 | 来源 |
| --- | --- | --- | --- | --- |
| `realesr-general-x4v3.onnx` | 4.6 MB | SRVGGNetCompact（1.21 M 参数，34 卷积） | `input` → `output`，×4 | [CoderViking/realesr-general-x4v3-onnx](https://huggingface.co/CoderViking/realesr-general-x4v3-onnx) |
| `upscaler-x4.onnx` | 66.0 MB | RRDB（16.7 M 参数，1094 节点） | `rgb` → `rgb_out`，×4 | [smile-forge/real-esrgan-gfpgan-onnx](https://huggingface.co/smile-forge/real-esrgan-gfpgan-onnx) |

两个文件都是 fp32、opset 17 / 20，只用到 Conv / PRelu / LeakyRelu / Concat / Add / Mul / Resize / DepthToSpace / Clip，这些算子 ONNX Runtime Web 的 WebGPU 与 WebAssembly 后端都支持。

## 许可

- 权重来自 [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（Xintao Wang 等），**BSD-3-Clause**
- 上面的 ONNX 导出由第三方完成，转换脚本未随权重发布，此处按其 HuggingFace 仓库的声明沿用 BSD-3-Clause
- 本项目代码为 MIT（见仓库根目录 `LICENSE`）

商用前请自行核对上游许可原文。BSD-3-Clause 允许商用，但要求保留版权声明与免责声明。

## 为什么权重直接进 Git，而不是 Git LFS

GitHub Pages 不会把 LFS 指针解析成真实文件，页面 `fetch('./models/*.onnx')` 会拿到一段文本而不是模型。
所以权重必须是普通文件。66 MB 单文件在 GitHub 的 100 MB 硬上限内，代价是仓库体积大、首次 clone 慢。

如果你 fork 后只想改前端、不想拉 71 MB 权重，可以：

```bash
git clone --filter=blob:none --no-checkout <repo>
git sparse-checkout set --no-cone '/*' '!/models/*.onnx'
git checkout
```

然后在本地放自己的模型，或改 `src/app.js` 里的 `MODELS` 指向 CDN 地址。

## 换模型

1. 把新的 `.onnx` 放进本目录，确认它是 4 倍（或你想要的倍数）放大，且输入形状为 `[1,3,H,W]`
2. 用 `python tools/inspect_model.py models/你的模型.onnx` 查出输入输出节点名与尺寸
3. 在 `src/app.js` 的 `MODELS` 表里加一项（`url`、`input`、`output`、`label`、`bytes`、`maxInput`）
4. 在 `index.html` 的 `<select id="model">` 里加一个 `<option>`，`value` 与上一步的 key 一致
5. 把文件信息补进 `manifest.json`，跑 `npm run verify`

模型如果是 2 倍或 8 倍，除了改 `SCALE`，还要注意 `MARGIN * SCALE`（裁边像素数）会跟着变。
