"""Report real ONNX node placement on a Mac, using synthetic model inputs.

Run from the repository root: python tools/profile_coreml_mac.py
CoreML placement does not distinguish GPU from Neural Engine; inspect the
Activity Monitor GPU History and CPU usage during an actual render as well.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent.parent / "model_assets"
MODELS = (
    ("RetinaFace", "det_10g.onnx", 640),
    ("Inswapper128ArcFace", "w600k_r50.onnx", 112),
    ("Inswapper128", "inswapper_128.fp16.onnx", 128),
)
OPTIONS = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "ALL",
    "RequireStaticInputShapes": "1",
    "AllowLowPrecisionAccumulationOnGPU": "1",
}


def input_array(meta, size):
    shape = []
    for axis, dimension in enumerate(meta.shape):
        if isinstance(dimension, int) and dimension > 0:
            shape.append(dimension)
        elif axis == 0:
            shape.append(1)
        elif len(meta.shape) == 4 and axis in (2, 3):
            shape.append(size)
        elif len(meta.shape) == 4 and axis == 1:
            shape.append(3)
        elif len(meta.shape) == 2 and axis == 1:
            shape.append(512)
        else:
            raise ValueError(f"Unknown dynamic dimension for {meta.name}: {meta.shape}")
    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
             "tensor(int64)": np.int64, "tensor(int32)": np.int32}.get(meta.type)
    if dtype is None:
        raise ValueError(f"Unsupported input type {meta.type} for {meta.name}")
    return np.zeros(shape, dtype=dtype)


def main():
    print("Available providers:", ort.get_available_providers())
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("CoreMLExecutionProvider unavailable in this Python environment")
    for name, filename, size in MODELS:
        path = ROOT / filename
        if not path.is_file():
            print(f"{name}: missing {path}")
            continue
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(ROOT / f"profile_{name}")
        session = ort.InferenceSession(
            str(path), sess_options=options,
            providers=[("CoreMLExecutionProvider", OPTIONS), "CPUExecutionProvider"],
        )
        try:
            feeds = {meta.name: input_array(meta, size) for meta in session.get_inputs()}
            session.run(None, feeds)
        except Exception as exc:
            print(f"{name}: sample inference failed: {exc}")
        finally:
            profile = Path(session.end_profiling())
        events = json.loads(profile.read_text())
        counts = Counter(
            event.get("args", {}).get("provider", "unknown")
            for event in events if event.get("cat") == "Node"
        )
        print(f"{name}: node execution counts {dict(counts)}; profile: {profile}")
        if not counts:
            print("  No nodes executed; check the sample inference error above.")


if __name__ == "__main__":
    main()
