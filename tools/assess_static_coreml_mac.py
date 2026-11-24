"""Test fixed-shape CoreML variants without changing the app's model files.

Run from the checkout root: python tools/assess_static_coreml_mac.py
Reports graph placement and numerical difference from CPU for each candidate.
"""

from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from profile_coreml_mac import MODELS, OPTIONS, ROOT, input_array


def fixed_model(source: Path, output: Path, size: int) -> None:
    model = onnx.load(str(source))
    for value in model.graph.input:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dims = tensor.shape.dim
        for axis, dim in enumerate(dims):
            if dim.HasField("dim_value") and dim.dim_value > 0:
                continue
            if axis == 0:
                value_at_axis = 1
            elif len(dims) == 4 and axis in (2, 3):
                value_at_axis = size
            elif len(dims) == 4 and axis == 1:
                value_at_axis = 3
            elif len(dims) == 2 and axis == 1:
                value_at_axis = 512
            else:
                raise ValueError(f"Unknown dimension in {value.name} at axis {axis}")
            dim.ClearField("dim_param")
            dim.dim_value = value_at_axis
    onnx.checker.check_model(model)
    onnx.save(model, str(output))


def profile(path: Path, feeds: dict, providers):
    settings = ort.SessionOptions()
    settings.enable_profiling = True
    session = ort.InferenceSession(str(path), sess_options=settings, providers=providers)
    try:
        output = session.run(None, feeds)
    finally:
        data = json.loads(Path(session.end_profiling()).read_text())
    counts = Counter(e.get("args", {}).get("provider", "unknown")
                     for e in data if e.get("cat") == "Node")
    return output, counts


def main():
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("CoreML provider unavailable")
    with tempfile.TemporaryDirectory(prefix="visomaster-coreml-test-") as directory:
        for name, filename, default_size in MODELS:
            source = ROOT / filename
            if not source.exists():
                print(name, "missing model", source)
                continue
            sizes = (512, 640) if name == "RetinaFace" else (default_size,)
            for size in sizes:
                candidate = Path(directory) / f"{size}_{filename}"
                try:
                    fixed_model(source, candidate, size)
                    baseline_session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
                    rng = np.random.default_rng(1234)
                    feeds = {}
                    for meta in baseline_session.get_inputs():
                        sample = input_array(meta, size)
                        if np.issubdtype(sample.dtype, np.floating):
                            sample = rng.random(sample.shape).astype(sample.dtype)
                        feeds[meta.name] = sample
                    baseline = baseline_session.run(None, feeds)
                    output, counts = profile(candidate, feeds,
                        [("CoreMLExecutionProvider", OPTIONS), "CPUExecutionProvider"])
                    if len(baseline) != len(output) or any(a.shape != b.shape for a, b in zip(baseline, output)):
                        raise ValueError("CPU and CoreML output shapes differ")
                    diffs = [float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
                             for a, b in zip(baseline, output) if a.size and b.size]
                    print(f"{name} at {size}: providers={dict(counts)}, max_abs_diff={max(diffs, default=0):.6g}")
                except Exception as exc:
                    print(f"{name} at {size}: candidate failed: {type(exc).__name__}: {exc}")
    print("Original model files were not modified.")


if __name__ == "__main__":
    main()
