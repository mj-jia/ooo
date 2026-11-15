import onnx

m = onnx.load(r'D:/workbuddy.task/2026-11-15-22-33-43/ooo/models/upscaler-x4.onnx', load_external_data=False)
g = m.graph
print('opset:', [(o.domain, o.version) for o in m.opset_import])


def shp(v):
    return [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]


for i in g.input:
    print('IN ', i.name, i.type.tensor_type.elem_type, shp(i))
for o in g.output:
    print('OUT', o.name, o.type.tensor_type.elem_type, shp(o))

ops = {}
for n in g.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print('nodes:', len(g.node), 'optypes:', len(ops))
print(sorted(ops.items(), key=lambda x: -x[1])[:14])
print('initializers MB:', round(sum(len(t.raw_data) for t in g.initializer) / 1e6, 1))
