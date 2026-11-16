# 本地开发

页面本身没有构建步骤、没有运行时依赖，改完刷新即可。下面的工具都是可选的。

## 起本地服务

必须经 HTTP 打开，双击 `index.html` 会失败（ES module 与 `fetch` 在 `file://` 下被浏览器拦住）：

```bash
npm run serve                    # = python serve.py 8123
python serve.py 8000             # 换端口
# 浏览器打开 http://127.0.0.1:8123/
```

`serve.py` 会带上 `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` 两个响应头。有了它们浏览器才启用 `SharedArrayBuffer`，onnxruntime-web 才能走多线程 WASM，本地调试速度能快 3~4 倍。

代价是**本地比线上快**：GitHub Pages 设不了这两个头，线上 WASM 后端固定单线程。拿本地数据跟线上比会误判。

## 校验仓库自洽性

```bash
npm run verify
```

检查三件事：模型与运行时文件的 sha256 是否与 `models/manifest.json` 一致、`index.html` / `app.js` 引用的资源是否真实存在、`app.js` 里写死的输入节点名是否与 manifest 对得上。
不依赖浏览器，纯 Node，几十毫秒跑完，适合挂到 CI 或 pre-push 钩子。

## 端到端测试

用本机 Edge 无头跑一次真实推理（需要先起本地服务）：

```bash
npm install                 # 只装 puppeteer-core，不下载 Chromium
npm test                    # 默认 http://127.0.0.1:8123/ + tools/out/small128.png

node tools/e2e.mjs tools/out/small48.png high                  # 指定图片与模型
SITE_URL=https://mj-jia.github.io/ooo/ npm test                # 直接测线上站点
```

参数：`node tools/e2e.mjs [图片] [fast|high]`。超时上限 7 分钟，无 GPU 的机器跑 `high` 会慢。

输出写到 `tools/out/`：`browser_output.png`（推理结果）、`ui_result.png`（整页截图）、控制台日志。

脚本里的 Edge 路径是写死的 Windows 路径，换机器或换 Chromium 需要改第 7 行的 `EDGE`。

## Python 参考实现

```bash
pip install -r requirements.txt

python tools/inspect_model.py                                    # 打印模型输入输出与算子构成
python tools/reference_run.py                                    # 内置测试图 + x4plus
python tools/reference_run.py photo.jpg -m models/realesr-general-x4v3.onnx -o ref.png
```

`reference_run.py` 复刻了 `app.js` 里同一套切块、反射补边、裁边距逻辑，用来对拍。

**对拍结果**（同一张 128×128 输入、同一个 x4v3 模型，浏览器输出 vs Python 输出）：

| 指标 | 数值 |
| --- | --- |
| 平均绝对误差 | 0.0 |
| 最大单通道差值 | 1 |
| 逐像素完全一致占比 | 99.99 % |
| PSNR | 92.94 dB |

剩下 0.01% 的像素差 1 级，来自两端的取整方式，属正常范围。这条对拍能证明：**网页侧的分块拼装没有引入误差**，如果哪天出现接缝或错位，这个数字会立刻掉下来。

## 可选：接 CI

`tools/verify.mjs` 不依赖浏览器，几十毫秒跑完，适合每次 push 都跑一遍。新建 `.github/workflows/verify.yml`：

```yaml
name: verify
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - name: 语法检查
        run: |
          node --check app.js
          node --check tools/verify.mjs
          node --check tools/e2e.mjs
      - name: 校验模型与资源
        run: node tools/verify.mjs
```

注意：用 OAuth token（`gh auth login`）推送含 workflow 文件的提交时，token 必须带 `workflow` 权限，否则 GitHub 会拒收整个推送并报 `refusing to allow an OAuth App to create or update workflow`。补权限用 `gh auth refresh -s workflow`，或在网页上直接新建该文件。

## 部署

推 `main` 即发布，GitHub Pages 直接从仓库根目录取文件（`Settings → Pages → Source: Deploy from a branch, main / (root)`）。

仓库里的 `models/*.onnx` 与 `vendor/*` 都是静态资源，与页面同源，不需要额外配置 CORS。

Pages 无法设置 `COOP` / `COEP` 响应头，因此 WASM 后端固定单线程；WebGPU 不受影响。

## 常见问题

**页面白屏、控制台报 `Failed to load module script`**
用 `file://` 打开了。起本地服务。

**控制台报 `no available backend found` 或 wasm 404**
`ort.env.wasm.wasmPaths` 没指对。`app.js` 里是 `new URL('./vendor/', import.meta.url)`，移动 `app.js` 位置时要一起改。

**推理结果周围有一圈暗边**
镜像反射补边失效了（`reflect()` 被改成补 0）。块边缘会看到暗框。

**换了模型但页面还是旧结果**
模型走 Cache Storage，缓存名是 `upsr-model-v2`。改 `app.js` 里的缓存名版本号，或在 DevTools → Application → Storage 里清掉。

**线上第一次打开要等很久**
模型和 WASM 运行时从 `github.io` 拉，国内通常一百多 KB/s。第二次打开命中缓存后正常。
