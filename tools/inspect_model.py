#!/usr/bin/env python
"""打印 ONNX 模型的输入输出、算子构成与参数量。

用法：
    python tools/inspect_model.py                     # 检查 models/ 下所有模型
    python tools/inspect_model.py models/xxx.onnx     # 检查指定文件
"""

import sys
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent.parent
DTYPE = {1: "float32", 2: "uint8", 3: "int8", 10: "float16", 11: "float64"}


def shape_of(value):
    dims = []
    for d in value.type.tensor_type.shape.dim:
        dims.append(d.dim_param or d.dim_value or "?")
    return dims


def inspect(path):
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph

    print(f"== {path.name}  ({path.stat().st_size / 1048576:.1f} MB)")
    print("   opset   :", ", ".join(f"{o.domain or 'ai.onnx'} {o.version}" for o in model.opset_import))
    print("   producer:", model.producer_name or "-", model.producer_version or "")

    for value in graph.input:
        print(f"   IN      : {value.name} {DTYPE.get(value.type.tensor_type.elem_type, '?')} {shape_of(value)}")
    for value in graph.output:
        print(f"   OUT     : {value.name} {DTYPE.get(value.type.tensor_type.elem_type, '?')} {shape_of(value)}")

    ops = {}
    for node in graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    params = sum(int(onnx.numpy_helper.to_array(t).size) for t in graph.initializer)

    print(f"   nodes   : {len(graph.node)}（{len(ops)} 种算子）")
    top = sorted(ops.items(), key=lambda kv: -kv[1])[:8]
    print("   算子    :", ", ".join(f"{k}×{v}" for k, v in top))
    print(f"   参数量  : {params / 1e6:.2f} M")
    print()


def main():
    args = sys.argv[1:]
    if args:
        targets = [Path(a) for a in args]
    else:
        targets = sorted((ROOT / "models").glob("*.onnx"))

    if not targets:
        print("没有找到 .onnx 文件")
        return 1

    for path in targets:
        inspect(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
