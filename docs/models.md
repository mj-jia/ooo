# 模型侧：官方结构 + 本项目改造版

页面跑的是 `models/*.onnx`——第三方从 HuggingFace 转好的黑盒：没有结构定义、没有权重来源、
改一行都没法复现。本目录补上「模型本身」：结构代码、官方权重下载、结构自检。

```
src/models/
  srvgg_official.py   官方 SRVGGNetCompact 原版（逐层等价，去掉 basicsr）
  srvgg.py            本项目改造版（scale=1 同尺寸复原）
  __init__.py         两个类同名，用别名导出避免混用
tools/
  fetch_official_weights.py   从官方 Releases 下 .pth
  check_arch.py               结构自检，CPU 几秒跑完
weights/              下载的 .pth 落这里（不进版本库）
```

`src/` 里只有 Python。页面侧的 `app.js` / `style.css` 在仓库根目录，与 `index.html` 同级，
两边不互相 import——只通过 `models/*.onnx` 交接。

## 两个模型的关系

两者同名类 `SRVGGNetCompact`，**结构差异全部由「×4 超分 → scale=1 复原」引起**，
没有一处是随手加的设计：

| | `srvgg_official.py` | `srvgg.py` |
|---|---|---|
| 任务 | ×2 / ×4 超分 | scale=1 同尺寸复原 |
| conv 层数 | `1 + num_conv + 1` = **34** | `1 + (num_conv−1) + 1` = **33** |
| 输出层 | 64 → 48（3×4×4） | 64 → 3 |
| 上采样 | `PixelShuffle(4)` | `Identity` |
| 残差 | `+ nearest(x, 4)` | `+ x` |
| 输出尺寸 | 3×4H×4W | 3×H×W |
| 参数量 | **1,213,296** | **1,150,339** |

同一个 `num_conv=32` 在两边含义不同（循环次数差 1），所以参数量不等，只算「同级」。
差 62,957 = 少一层 64→64（36,928）+ 输出层 27,696→1,731（25,965）+ 少一个 PReLU（64）。

**权重兼容性**：`body.0` ~ `body.62` 共 32 层 conv 及对应 PReLU 形状完全一致，
可以从官方 `.pth` 直接载入（96/98 个参数命中）；官方第 33 层（`body.64`，64→64）在改造版
同一位置被输出层（64→3）占用，形状不符，按 key + shape 过滤时连同它后面的 PReLU、
输出层一起丢弃。`tools/check_arch.py` 会把丢弃项打印出来。

## 快速开始

```bash
# 1) 装 torch（只做自检用 CPU 版就够，约 200 MB）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) 下官方权重（x4v3 + wdn-x4v3 两个，官方默认是 DNI 0.5 混合）
python tools/fetch_official_weights.py

# 3) 结构自检：参数量、恒等起点、权重交叉载入
python tools/check_arch.py
```

第 3 步正常输出：

```
====================================================================
[ok  ] 官方 realesr-general-x4v3     参数量 1,213,296  期望 1,213,296
[ok  ] 官方 realesr-animevideov3     参数量   621,424  期望   621,424
[ok  ] 改造版 scale=1                参数量 1,150,339  期望 1,150,339
[ok  ] 初始态恒等映射                max|out-in| = 0.0
[ok  ] 官方权重载入改造版  命中 96/98 个参数
       丢弃 5 个：['body.64.weight', 'body.64.bias', 'body.65.weight', 'body.66.weight', 'body.66.bias']
         body.64.weight       官方 (64, 64, 3, 3) → 本版无对应位置
         body.64.bias         官方 (64,) → 本版无对应位置
         body.65.weight       官方 (64,) → 本版无对应位置
====================================================================
结论：全部通过
```

丢弃的 5 个条目里，只有 `body.64.*` 是「形状冲突」（官方的第 33 层 64→64 撞上本版的
输出层 64→3）；`body.65.weight`、`body.66.*` 是官方多出来的第 33 层 conv 及其 PReLU，
本版根本没有这一层。真正被载入的是 32 层 conv 的 96 个参数。

**`max|out-in| = 0.0` 这一行是关键**：改造版 `_init()` 把最后一层权重置零，
所以 step 0 时 `body(x) = 0`，整体输出精确等于输入。它不是「接近恒等」，是浮点意义上
的精确恒等——这是加载官方预训练主干时不会破坏已收敛特征的原因，也是微调能从
「什么都不做」起步的原因。

> 注意：新增模块必须挂在 `_init()` **之后**。若挂在之前，`_init()` 里的
> `convs[-1]` 会指向新模块的最后一层，原输出卷积保留 kaiming 初始化，恒等起点就没了。

## 权重为什么不进仓库

`realesr-general-x4v3.pth` 约 64 MB，`wdn-x4v3` 同量级。提交进 git 会让仓库迅速膨胀，
而官方 Releases 本身就是稳定分发地址，重复存储没有意义。`.gitignore` 已排除
`weights/`、`*.pth`、`*.part`，`fetch_official_weights.py` 负责下载并打印 sha256。

若确实要在仓库里留一份（比如评审需要），用 GitHub Release 而不是 commit：

```bash
gh release create weights-v1 weights/*.pth -t "官方权重镜像" -n "Real-ESRGAN v0.2.5.0"
```

## 与 `models/*.onnx` 的关系

现在页面跑的两个 ONNX 都是**第三方从 HuggingFace 转的 ×4 模型**，与 `src/models/` 里的代码
没有血缘关系：不知道对应哪份权重、不知道前后处理怎么做的。`models/manifest.json` 里记录了
每个文件的 `bytes` 和 `sha256`，`npm run verify` 会校验。

要让「自己训的模型 → 页面」这条路打通，还差一步 ONNX 导出，并且必须注意两件事：

1. **前后处理要一致**。`upscaler-x4.onnx` 在 manifest 里标注「模型中已包含预处理与后处理」，
   自己导出时若把归一化放进图里（或不放），必须和 `app.js` 的假设对上，否则输出偏色。
2. **换文件必须重算 manifest**。`bytes` 和 `sha256` 变了却不更新，`npm run verify` 直接失败。

导出脚本目前不在仓库里。

## 许可

`srvgg_official.py` 是上游
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesrgan/archs/srvgg_arch.py`
的等价实现，遵循 **BSD-3-Clause**，Copyright (c) 2021, Xintao Wang。原许可全文见上游仓库
`LICENSE`。主要改动是去掉 `@ARCH_REGISTRY.register()` 与 basicsr 依赖、抽出激活函数工厂、
补充权重加载与 DNI 插值工具——结构层面逐层等价，参数量与官方一致可由断言验证。
