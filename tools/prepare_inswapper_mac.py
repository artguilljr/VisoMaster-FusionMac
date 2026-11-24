"""Build, compare and benchmark Inswapper variants in isolated processes.

Run from the checkout: python tools/prepare_inswapper_mac.py
Only a faster candidate that passes sample-output comparisons is approved.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "model_assets" / "coreml_mac"
SOURCE = ROOT / "model_assets" / "inswapper_128.fp16.onnx"
CANDIDATE = CACHE / "inswapper_fp32_reflect.onnx"
MODES = {"reference_cpu": "CPU", "original_gpu": "CPUAndGPU",
         "candidate_gpu": "CPUAndGPU", "candidate_all": "ALL"}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def worker(label):
    import numpy as np
    import onnxruntime as ort

    units = MODES[label]
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = 2
    settings.inter_op_num_threads = 1
    settings.add_session_config_entry("session.intra_op.allow_spinning", "0")
    settings.enable_profiling = True
    settings.profile_file_prefix = str(CACHE / label)
    providers = ["CPUExecutionProvider"]
    if units != "CPU":
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("This Python environment has no CoreML execution provider")
        providers.insert(0, ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": units,
            "RequireStaticInputShapes": "1", "AllowLowPrecisionAccumulationOnGPU": "0",
        }))
    path = CANDIDATE if label.startswith("candidate") else SOURCE
    session = ort.InferenceSession(str(path), sess_options=settings, providers=providers)
    samples = []
    # A source latent must have unit L2 norm, like the application's latent.
    # The earlier diagnostic used unnormalised random latents.
    for seed in (7, 19, 31):
        rng = np.random.default_rng(seed)
        latent = rng.standard_normal((1, 512)).astype(np.float32)
        latent /= np.linalg.norm(latent)
        samples.append({"target": rng.random((1, 3, 128, 128)).astype(np.float32), "source": latent})
    session.run(None, samples[0])  # warm up; exclude first-use compilation
    outputs, milliseconds = {}, []
    for index, feed in enumerate(samples):
        start = time.perf_counter()
        results = session.run(None, feed)
        milliseconds.append((time.perf_counter() - start) * 1000)
        if len(results) != 1:
            raise ValueError("Unexpected Inswapper output count")
        outputs[f"sample_{index}"] = results[0]
    profile = Path(session.end_profiling())
    events = json.loads(profile.read_text())
    counts = Counter(e.get("args", {}).get("provider") for e in events
                     if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
    np.savez(CACHE / f"{label}.npz", **outputs)
    (CACHE / f"{label}.json").write_text(json.dumps({
        "label": label, "units": units, "median_ms": statistics.median(milliseconds),
        "sample_ms": milliseconds, "providers": dict(counts),
    }, indent=2))


def main():
    import numpy as np
    from app.processors.utils.coreml_inswapper import build_candidate

    if not SOURCE.is_file():
        raise SystemExit(f"Missing model: {SOURCE}")
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = CACHE / "approved.json"
    manifest.unlink(missing_ok=True)
    print("Building an FP32 copy with equivalent Slice/Concat reflection padding...", flush=True)
    print(build_candidate(SOURCE, CANDIDATE), flush=True)
    reports = {}
    for label in MODES:
        print(f"Testing {label} (first CoreML compilation can take time)...", flush=True)
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", label],
                                    timeout=300)
        except subprocess.TimeoutExpired:
            print(f"{label}: timed out; rejected", flush=True)
            continue
        if result.returncode:
            print(f"{label}: failed with exit code {result.returncode}; rejected", flush=True)
            continue
        reports[label] = json.loads((CACHE / f"{label}.json").read_text())
    if "reference_cpu" not in reports:
        raise SystemExit("CPU reference failed. No model approved.")
    with np.load(CACHE / "reference_cpu.npz") as reference:
        for label, report in reports.items():
            max_error, mean_error, valid = 0.0, 0.0, True
            with np.load(CACHE / f"{label}.npz") as result:
                for key in reference.files:
                    a, b = reference[key], result[key]
                    if a.shape != b.shape or not np.isfinite(b).all():
                        valid = False
                        continue
                    difference = np.abs(a.astype(np.float32) - b.astype(np.float32))
                    max_error = max(max_error, float(difference.max()))
                    mean_error = max(mean_error, float(difference.mean()))
            # Numerical screening only; a real face render is still required.
            valid = valid and max_error <= 0.02 and mean_error <= 0.002
            report.update(valid=valid, max_error=max_error, mean_error=mean_error)
            print(f"{label}: {report['median_ms']:.1f} ms, providers={report['providers']}, "
                  f"max_diff={max_error:.6f}, mean_diff={mean_error:.6f}, valid={valid}", flush=True)
    baseline = min((r["median_ms"] for k, r in reports.items()
                    if not k.startswith("candidate") and r["valid"]), default=float("inf"))
    eligible = [r for k, r in reports.items() if k.startswith("candidate") and r["valid"]
                and r["providers"].get("CoreMLExecutionProvider", 0) > 0
                and r["median_ms"] < baseline * 0.9]
    if eligible:
        best = min(eligible, key=lambda r: r["median_ms"])
        manifest.write_text(json.dumps({"version": 1, "source_sha256": digest(SOURCE),
            "candidate_sha256": digest(CANDIDATE), "units": best["units"],
            "median_ms": best["median_ms"], "baseline_ms": baseline}, indent=2))
        print(f"Approved {best['label']} for a short real-video test. Enable with VISOMASTER_INSWAPPER_OPTIMIZED=1.")
    else:
        print("No candidate met both the accuracy and speed checks. App model selection is unchanged.")
    (CACHE / "report.json").write_text(json.dumps(reports, indent=2))
    print(f"Report: {CACHE / 'report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODES)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        main()
