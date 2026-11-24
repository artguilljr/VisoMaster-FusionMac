"""Optional fixed-shape copies for CoreML; originals remain untouched."""

import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


SIZES = {"RetinaFace": 512, "Inswapper128ArcFace": 112}


def static_copy(name: str, source: str) -> str:
    size = SIZES[name]
    source_path = Path(source)
    stat = source_path.stat()
    key = hashlib.sha256(
        f"static-v1:{source_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{size}".encode()
    ).hexdigest()[:16]
    cache = Path.home() / ".cache" / "visomaster-fusion" / "coreml"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}-{size}-{key}.onnx"
    if target.is_file():
        return str(target)

    model = onnx.load(str(source_path))
    for value in model.graph.input:
        dims = value.type.tensor_type.shape.dim
        for axis, dimension in enumerate(dims):
            if dimension.HasField("dim_value") and dimension.dim_value > 0:
                continue
            if axis == 0:
                value_at_axis = 1
            elif len(dims) == 4 and axis in (2, 3):
                value_at_axis = size
            elif len(dims) == 4 and axis == 1:
                value_at_axis = 3
            else:
                raise ValueError(f"Unexpected dynamic dimension in {name}: {value.name}")
            dimension.ClearField("dim_param")
            dimension.dim_value = value_at_axis

    if name == "RetinaFace":
        # The original graph declares 640px output sizes even for 512px inputs.
        # Replace stale output metadata with shapes observed from a 512px CPU run.
        cpu = ort.InferenceSession(str(source_path), providers=["CPUExecutionProvider"])
        inputs = cpu.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Unexpected RetinaFace input layout")
        sample = np.zeros((1, 3, size, size), dtype=np.float32)
        outputs = cpu.run(None, {inputs[0].name: sample})
        if len(outputs) != len(model.graph.output):
            raise ValueError("RetinaFace output count changed")
        for value, output in zip(model.graph.output, outputs):
            dims = value.type.tensor_type.shape.dim
            if len(dims) != output.ndim:
                raise ValueError(f"Unexpected output rank: {value.name}")
            for dimension, length in zip(dims, output.shape):
                dimension.ClearField("dim_param")
                dimension.dim_value = length

    onnx.checker.check_model(model)
    temporary = target.with_suffix(".tmp.onnx")
    onnx.save(model, str(temporary))
    temporary.replace(target)
    return str(target)
