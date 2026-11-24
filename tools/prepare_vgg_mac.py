"""Benchmark and optionally approve the static VGG texture model for CoreML.

Run: python tools/prepare_vgg_mac.py
The original model is never changed. An approved copy still requires an opt-in.
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
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "model_assets" / "vgg_combo_relu3_3_relu3_1.onnx"
OUT = ROOT / "model_assets" / "coreml_mac" / "vgg_profile"
MODES = {"cpu": None, "original_gpu": "CPUAndGPU", "original_all": "ALL",
         "candidate_cpu": None, "candidate_gpu": "CPUAndGPU", "candidate_all": "ALL"}


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
            raise RuntimeError("This ONNX Runtime build has no CoreML provider")
        providers.insert(0, ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": MODES[mode],
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "0" if mode.startswith("candidate") else "1",
        }))
    path = OUT / "vgg_static.onnx" if mode.startswith("candidate") else SOURCE
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    rng = np.random.default_rng(2468)
    # Normalized RGB-like values, including texture-rich input and smooth input.
    samples = [rng.random((1, 3, 512, 512), dtype=np.float32) * 3.0 - 1.5,
               np.broadcast_to(np.linspace(-1.0, 1.0, 512, dtype=np.float32),
                               (1, 3, 512, 512)).copy()]
    session.run(None, {"input": samples[0]})  # exclude compilation
    times = []
    for sample in (samples if not MODES[mode] else samples * 3):
        start = time.perf_counter()
        result = session.run(None, {"input": sample})
        times.append((time.perf_counter() - start) * 1000)
        if len(result) != 1 or result[0].shape != (1, 512, 128, 128):
            raise ValueError("Unexpected VGG features shape")
    outputs = {}
    for index, sample in enumerate(samples):
        output = session.run(None, {"input": sample})[0]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite VGG features")
        outputs[str(index)] = output
    np.savez_compressed(OUT / f"{mode}.npz", **outputs)
    events = json.loads(Path(session.end_profiling()).read_text())
    counts = Counter(event.get("args", {}).get("provider") for event in events
                     if event.get("cat") == "Node" and event.get("args", {}).get("provider"))
    (OUT / f"{mode}.json").write_text(json.dumps({
        "median_ms": statistics.median(times), "samples_ms": times,
        "providers": dict(counts), "session_providers": session.get_providers(),
    }, indent=2))


def main():
    import numpy as np
    from app.processors.utils.coreml_vgg import build_candidate, digest

    if not SOURCE.is_file():
        raise SystemExit(f"Missing VGG model: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "approved.json").unlink(missing_ok=True)
    print("Building separate fixed-shape VGG copy...", flush=True)
    print(build_candidate(SOURCE, OUT / "vgg_static.onnx"), flush=True)
    reports = {}
    for mode in MODES:
        print(f"Profiling {mode}...", flush=True)
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode],
                           check=True, timeout=240)
            reports[mode] = json.loads((OUT / f"{mode}.json").read_text())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            print(f"{mode} failed: {exc}; not approved", flush=True)
    if "cpu" not in reports or "candidate_cpu" not in reports:
        raise SystemExit("CPU validation failed. No candidate approved.")
    with np.load(OUT / "cpu.npz") as reference:
        for mode, report in reports.items():
            valid, max_diff, mean_diff = True, 0.0, 0.0
            with np.load(OUT / f"{mode}.npz") as actual:
                for key in reference.files:
                    a, b = reference[key], actual[key]
                    if a.shape != b.shape or not np.isfinite(b).all():
                        valid = False
                        continue
                    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
                    max_diff = max(max_diff, float(d.max()))
                    mean_diff = max(mean_diff, float(d.mean()))
                    valid &= bool(np.allclose(a, b, rtol=0.01, atol=0.001))
            report.update(valid=valid, max_diff=max_diff, mean_diff=mean_diff)
            print(f"{mode}: {report['median_ms']:.1f} ms; {report['providers']}; "
                  f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, valid={valid}", flush=True)
    (OUT / "report.json").write_text(json.dumps(reports, indent=2))
    baseline = min((r["median_ms"] for name, r in reports.items()
                    if name in ("original_gpu", "original_all")), default=float("inf"))
    eligible = [(name, r) for name, r in reports.items()
                if name in ("candidate_gpu", "candidate_all") and r["valid"]
                and r["providers"].get("CoreMLExecutionProvider", 0) > 0
                and r["median_ms"] < baseline * 0.9]
    if reports["candidate_cpu"]["valid"] and eligible:
        name, best = min(eligible, key=lambda item: item[1]["median_ms"])
        (OUT / "approved.json").write_text(json.dumps({
            "version": 1, "source_sha256": digest(SOURCE),
            "candidate_sha256": digest(OUT / "vgg_static.onnx"),
            "units": MODES[name], "median_ms": best["median_ms"],
            "baseline_ms": baseline,
        }, indent=2))
        print(f"Approved {name} for a short real-video test. "
              "Enable with VISOMASTER_VGG_OPTIMIZED=1.", flush=True)
    else:
        print("No VGG candidate passed accuracy and speed checks; app remains unchanged.", flush=True)
    print(f"Report: {OUT / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODES)
    args = parser.parse_args()
    worker(args.worker) if args.worker else main()
