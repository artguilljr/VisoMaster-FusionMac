"""Measure GFPGANv1.4 provider placement and latency on Apple Silicon.

Run from checkout: python tools/profile_gfpgan_mac.py
This only reads the installed ONNX model and writes a profiling report.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "model_assets" / "GFPGANv1.4.onnx"
OUT = ROOT / "model_assets" / "coreml_mac" / "gfpgan_profile"
MODES = {"cpu": None, "gpu": "CPUAndGPU", "all": "ALL"}


def worker(mode):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(OUT / mode)
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = ["CPUExecutionProvider"]
    if MODES[mode]:
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime has no CoreML provider")
        providers.insert(0, ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": MODES[mode],
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "1",
        }))
    session = ort.InferenceSession(str(SOURCE), sess_options=options, providers=providers)
    rng = np.random.default_rng(2026)
    sample = rng.uniform(-1, 1, (1, 3, 512, 512)).astype(np.float32)
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 512, 512]:
        raise ValueError("Unexpected GFPGAN input")
    feed = {inputs[0].name: sample}
    session.run(None, feed)  # Exclude first inference and compilation.
    milliseconds = []
    count = 1 if mode == "cpu" else 4
    for _ in range(count):
        start = time.perf_counter()
        outputs = session.run(None, feed)
        milliseconds.append((time.perf_counter() - start) * 1000)
    if len(outputs) != 1 or outputs[0].shape != sample.shape or not np.isfinite(outputs[0]).all():
        raise ValueError("Unexpected GFPGAN output")
    np.save(OUT / f"{mode}.npy", outputs[0])
    events = json.loads(Path(session.end_profiling()).read_text())
    counts = Counter(event.get("args", {}).get("provider") for event in events
                     if event.get("cat") == "Node" and event.get("args", {}).get("provider"))
    (OUT / f"{mode}.json").write_text(json.dumps({
        "median_ms": statistics.median(milliseconds), "samples_ms": milliseconds,
        "providers": dict(counts), "session_providers": session.get_providers(),
    }, indent=2))


def main():
    import numpy as np

    if not SOURCE.is_file():
        raise SystemExit(f"GFPGAN model missing: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for mode in MODES:
        print(f"Testing GFPGAN {mode}; compilation may take a few minutes...", flush=True)
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode],
                           check=True, timeout=360)
            report[mode] = json.loads((OUT / f"{mode}.json").read_text())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"{mode} failed: {exc}", flush=True)
    if "cpu" in report:
        reference = np.load(OUT / "cpu.npy", mmap_mode="r")
        for mode, details in report.items():
            output = np.load(OUT / f"{mode}.npy", mmap_mode="r")
            details["max_diff"] = float(np.max(np.abs(output - reference)))
            details["mean_diff"] = float(np.mean(np.abs(output - reference)))
    for mode, details in report.items():
        print(f"{mode}: {details['median_ms']:.1f} ms, providers={details['providers']}, "
              f"max_diff={details.get('max_diff', 'no CPU reference')}", flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Report: {OUT / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODES)
    args = parser.parse_args()
    worker(args.worker) if args.worker else main()
