"""A benchmark-gated, fixed-shape CoreML copy of the texture VGG model."""
import hashlib
import json
from pathlib import Path

import onnx

SOURCE_SHA256 = "1068ee41e3c67dcfbbeccbc93e539eb06f89bba08618951bb33e9be2c1fbc986"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def build_candidate(source, destination):
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Unknown VGG model; this fixed-shape conversion requires the checked release")
    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise ValueError("Unexpected VGG graph inputs or outputs")
    if graph.input[0].name != "input" or graph.output[0].name != "features":
        # Keep model input names intact; reject an unexpected export.
        raise ValueError("Unexpected VGG input or output names")
    for value, shape in ((graph.input[0], (1, 3, 512, 512)),
                         (graph.output[0], (1, 512, 128, 128))):
        dims = value.type.tensor_type.shape.dim
        if len(dims) != len(shape):
            raise ValueError("Unexpected VGG tensor rank")
        for dimension, size in zip(dims, shape):
            if dimension.HasField("dim_value") and dimension.dim_value not in (0, size):
                raise ValueError("Unexpected VGG tensor dimension")
            dimension.ClearField("dim_param")
            dimension.dim_value = size
    onnx.checker.check_model(model)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return {"input_shape": [1, 3, 512, 512], "output_shape": [1, 512, 128, 128]}


def approved_candidate(source):
    original = Path(source)
    cache = original.parent / "coreml_mac" / "vgg_profile"
    candidate, manifest = cache / "vgg_static.onnx", cache / "approved.json"
    if not (candidate.is_file() and manifest.is_file()):
        return None
    data = json.loads(manifest.read_text())
    if data.get("version") != 1 or data.get("units") not in ("ALL", "CPUAndGPU"):
        return None
    if digest(original) != data.get("source_sha256") or digest(candidate) != data.get("candidate_sha256"):
        raise ValueError("VGG model or approved copy changed; rerun tools/prepare_vgg_mac.py")
    return str(candidate), data["units"]
