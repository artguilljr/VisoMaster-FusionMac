"""Build an FP32, reflection-pad-free Inswapper copy for CoreML evaluation.

This is a derived model: converting existing FP16 weights to FP32 does not
recover precision already lost when the original weights were quantized.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def approved_candidate(source: str):
    """Accept only the source/candidate pair checked by the Mac benchmark."""
    original = Path(source)
    cache = original.parent / "coreml_mac"
    manifest = cache / "approved.json"
    candidate = cache / "inswapper_fp32_reflect.onnx"
    if not manifest.is_file() or not candidate.is_file():
        return None
    data = json.loads(manifest.read_text())
    if data.get("version") != 1 or data.get("units") not in ("ALL", "CPUAndGPU"):
        return None
    for path, expected in ((original, data.get("source_sha256")),
                           (candidate, data.get("candidate_sha256"))):
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError("Inswapper files changed after benchmark; run prepare_inswapper_mac.py again")
    return str(candidate), data["units"]


def build_candidate(source: Path, destination: Path) -> dict:
    model = onnx.load(str(source))
    graph = model.graph
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for n in graph.node for a in n.attribute):
        raise ValueError("Nested graphs are not supported by this Inswapper conversion")

    promoted = 0
    for tensor in graph.initializer:
        if tensor.data_type == TensorProto.FLOAT16:
            tensor.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(tensor).astype(np.float32), name=tensor.name))
            promoted += 1
    for value in (*graph.input, *graph.output, *graph.value_info):
        if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    for node in graph.node:
        for attribute in node.attribute:
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i == TensorProto.FLOAT16:
                attribute.i = TensorProto.FLOAT
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == TensorProto.FLOAT16:
                tensor = attribute.t
                tensor.CopyFrom(numpy_helper.from_array(
                    numpy_helper.to_array(tensor).astype(np.float32), name=tensor.name))

    constants = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    nodes = []
    count = 0
    used = {name for node in graph.node for name in (*node.input, *node.output)}

    def name():
        nonlocal count
        while True:
            count += 1
            candidate = f"vm_coreml_reflect_{count}"
            if candidate not in used:
                used.add(candidate)
                return candidate

    def constant(values):
        key = name()
        graph.initializer.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), key))
        return key

    def slice_one(value, axis, start, end):
        output = name()
        nodes.append(helper.make_node("Slice", [value, constant([start]), constant([end]),
                                               constant([axis]), constant([1])], [output], name=name()))
        return output

    replaced = 0
    for node in graph.node:
        mode = next((helper.get_attribute_value(a) for a in node.attribute if a.name == "mode"), b"constant")
        if node.op_type != "Pad" or mode != b"reflect":
            nodes.append(node)
            continue
        if len(node.input) < 2 or node.input[1] not in constants:
            raise ValueError(f"Nonconstant reflection padding in {node.name}")
        pads = [int(x) for x in constants[node.input[1]].reshape(-1)]
        if len(pads) != 8 or any(p < 0 or p > 3 for p in pads) or any(pads[i] for i in (0, 1, 4, 5)):
            raise ValueError(f"Unexpected reflection padding {pads} in {node.name}")
        current = node.input[0]
        for axis in (2, 3):
            before, after = pads[axis], pads[axis + 4]
            if not (before or after):
                continue
            pieces = [slice_one(current, axis, i, i + 1) for i in range(before, 0, -1)]
            pieces.append(current)
            pieces.extend(slice_one(current, axis, -(i + 1), -i) for i in range(1, after + 1))
            output = name()
            nodes.append(helper.make_node("Concat", pieces, [output], axis=axis, name=name()))
            current = output
        nodes.append(helper.make_node("Identity", [current], list(node.output), name=name()))
        replaced += 1
    if replaced != 14:
        raise ValueError(f"Expected 14 Inswapper reflection pads, found {replaced}")
    del graph.node[:]
    graph.node.extend(nodes)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return {"fp16_weights_promoted": promoted, "reflection_pads_replaced": replaced}
