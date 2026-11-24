"""Optional AlphaFace copy with fixed reflection padding for CoreML evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

KNOWN_SOURCE_SHA256 = "5514d967ab6cc27e1b0edc092e05ee97d235adccb4da68574a9b1a1e221a4c6a"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def approved_candidate(source):
    original = Path(source)
    cache = original.parent.parent / "coreml_mac" / "alphaface_profile"
    candidate = cache / "alphaface_reflect.onnx"
    manifest = cache / "approved.json"
    if not (candidate.is_file() and manifest.is_file()):
        return None
    data = json.loads(manifest.read_text())
    if data.get("version") != 1 or data.get("units") not in ("ALL", "CPUAndGPU"):
        return None
    if digest(original) != data.get("source_sha256") or digest(candidate) != data.get("candidate_sha256"):
        raise ValueError("AlphaFace model or candidate changed; rerun tools/profile_alphaface_mac.py")
    return str(candidate), data["units"]


def build_candidate(source, destination):
    """Replace only the 14 known reflect pads, leaving weights and originals intact.

    AlphaFace 256 has fixed spatial inputs; its E/G pads are 3 pixels and its
    12 fusion pads are 1 pixel. Reject other graphs instead of guessing.
    """
    if digest(source) != KNOWN_SOURCE_SHA256:
        raise ValueError("Unknown AlphaFace model; this conversion requires the official checked release")
    model = onnx.load(str(source))
    graph = model.graph
    pads = [n for n in graph.node if n.op_type == "Pad"]
    if len(pads) != 14 or any(
        helper.get_attribute_value(next(a for a in n.attribute if a.name == "mode")) != b"reflect"
        for n in pads
    ):
        raise ValueError("Unexpected AlphaFace padding graph")
    expected = {"/E/Pad", "/G/Pad"} | {
        f"/E/fusion_layer_{i}/OP{j}/Pad" for i in range(6) for j in (1, 2)
    }
    if {n.name for n in pads} != expected:
        raise ValueError("Unexpected AlphaFace pad names")
    if [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim] != [1, 3, 256, 256]:
        raise ValueError("AlphaFace target shape changed")
    used = {x for n in graph.node for x in (*n.input, *n.output)}
    suffix = 0

    def fresh():
        nonlocal suffix
        while True:
            suffix += 1
            name = f"vm_alpha_reflect_{suffix}"
            if name not in used:
                used.add(name)
                return name

    def slice_axis(value, axis, start, end, output_nodes):
        args = [value]
        for arr in ([start], [end], [axis], [1]):
            name = fresh()
            graph.initializer.append(numpy_helper.from_array(np.array(arr, dtype=np.int64), name))
            args.append(name)
        result = fresh()
        output_nodes.append(helper.make_node("Slice", args, [result], name=fresh()))
        return result

    nodes = []
    for node in graph.node:
        if node.op_type != "Pad":
            nodes.append(node)
            continue
        width = 3 if node.name in ("/E/Pad", "/G/Pad") else 1
        current = node.input[0]
        for axis in (2, 3):
            parts = [slice_axis(current, axis, i, i + 1, nodes) for i in range(width, 0, -1)]
            parts.append(current)
            parts.extend(slice_axis(current, axis, -(i + 1), -i, nodes)
                         for i in range(1, width + 1))
            result = fresh()
            nodes.append(helper.make_node("Concat", parts, [result], axis=axis, name=fresh()))
            current = result
        nodes.append(helper.make_node("Identity", [current], list(node.output), name=fresh()))
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return {"reflection_pads_replaced": len(pads)}
