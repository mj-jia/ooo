# ONNX Runtime Web 运行时

这里放的是 **onnxruntime-web 1.31.1** 的预构建产物（MIT 许可），页面直接以 ES module 引用，不经打包器。

| 文件 | 作用 |
| --- | --- |
| `ort.all.min.mjs` | ORT Web 的 JS 入口，包含所有后端（webgpu / wasm / webgl）的绑定 |
| `ort-wasm-simd-threaded.jsep.wasm` | WebAssembly 引擎本体，`jsep` 表示带 WebGPU 支持（JSEP = JavaScript Execution Provider） |
| `ort-wasm-simd-threaded.jsep.mjs` | 上面那个 wasm 的加载胶水代码，由 `ort.all.min.mjs` 动态 import |

`src/app.js` 里两处配合：

```js
import * as ort from '../vendor/ort.all.min.mjs';
ort.env.wasm.wasmPaths = new URL('../vendor/', import.meta.url).href;
```

`wasmPaths` 必须设，否则 ORT 会去 CDN 拉 wasm，离线或网络受限时直接失败。

## 更新版本

```bash
npm install onnxruntime-web@<版本> --no-save
cp node_modules/onnxruntime-web/dist/ort.all.min.mjs \
   node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.mjs \
   node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm \
   vendor/
npm run verify   # 会提示 sha256 不匹配，按新值更新 models/manifest.json
```

## 关于体积

`ort-wasm-simd-threaded.jsep.wasm` 是 29 MB。同版本还有个纯 CPU 版 `ort-wasm-simd-threaded.wasm`（约 12 MB），但没有 WebGPU 支持。
GitHub Pages 会对它做 gzip，实际传输约 7 MB。换纯 CPU 版能省到约 4 MB，代价是彻底失去 GPU 加速——在本项目的实测里不划算，所以没换。

分线程需要 `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` 响应头，GitHub Pages 无法设置，因此 WASM 后端是**单线程**。这也是页面优先尝试 WebGPU 的原因。
