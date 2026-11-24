"""Measure AlphaFace 256 on the Mac without changing the installed model.

Run: python tools/profile_alphaface_mac.py
Each provider runs in its own process so CoreML compilation and profiling are isolated.
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
SOURCE = ROOT / "model_assets/alphaface/alphaface_swapper_fused_norm.onnx"
OUT = ROOT / "model_assets/coreml_mac/alphaface_profile"
MODES = {"cpu": None, "coreml_gpu": "CPUAndGPU", "coreml_all": "ALL",
         "candidate_cpu": None, "candidate_gpu": "CPUAndGPU", "candidate_all": "ALL"}


def worker(mode, tag=""):
    import numpy as np
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_profiling = True
    label = f"{mode}_{tag}" if tag else mode
    options.profile_file_prefix = str(OUT / label)
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = ["CPUExecutionProvider"]
    if MODES[mode]:
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Install an ONNX Runtime build with CoreMLExecutionProvider")
        providers.insert(0, ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": MODES[mode],
            "RequireStaticInputShapes": "1",
            "AllowLowPrecisionAccumulationOnGPU": "0",
        }))
    model_path = OUT / "alphaface_reflect.onnx" if mode.startswith("candidate") else SOURCE
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    rng = np.random.default_rng(4711)
    latent = rng.standard_normal((1, 512)).astype(np.float32)
    latent /= np.linalg.norm(latent)
    feeds = [{"target": rng.random((1, 3, 256, 256)).astype(np.float32),
              "source_embedding": latent} for _ in range(12 if MODES[mode] else 4)]
    session.run(None, feeds[0])  # Exclude first inference and compilation.
    times = []
    for feed in feeds:
        start = time.perf_counter()
        result = session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000)
        if len(result) != 1 or result[0].shape != (1, 3, 256, 256):
            raise ValueError("Unexpected AlphaFace output")
        if not np.isfinite(result[0]).all():
            raise ValueError("Nonfinite AlphaFace output")
    np.savez_compressed(OUT / f"{label}.npz", **{str(i): session.run(None, f)[0]
                                                    for i, f in enumerate(feeds[:2])})
    events = json.loads(Path(session.end_profiling()).read_text())
    counts = Counter(e.get("args", {}).get("provider") for e in events
                     if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
    (OUT / f"{label}.json").write_text(json.dumps({
        "median_ms": statistics.median(times), "samples_ms": times,
        "providers": dict(counts), "session_providers": session.get_providers(),
    }, indent=2))


def main():
    import numpy as np
    from app.processors.utils.coreml_alphaface import build_candidate, digest

    if not SOURCE.is_file():
        raise SystemExit(f"Missing model: {SOURCE}; run python download_models.py")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "approved.json").unlink(missing_ok=True)
    print("Building a separate AlphaFace copy with equivalent reflection padding...", flush=True)
    print(build_candidate(SOURCE, OUT / "alphaface_reflect.onnx"), flush=True)
    reports = {}
    for mode in MODES:
        print(f"Profiling {mode}; first CoreML compilation may take a few minutes...", flush=True)
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", mode],
                           check=True, timeout=300)
            reports[mode] = json.loads((OUT / f"{mode}.json").read_text())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            print(f"{mode} failed: {exc}", flush=True)
    if "cpu" not in reports:
        raise SystemExit("CPU reference failed")
    with np.load(OUT / "cpu.npz") as reference:
        for mode, report in reports.items():
            valid = True
            max_diff, mean_diff = 0.0, 0.0
            with np.load(OUT / f"{mode}.npz") as actual:
                for key in reference.files:
                    a, b = reference[key], actual[key]
                    if a.shape != b.shape or not np.isfinite(b).all():
                        valid = False
                        continue
                    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
                    max_diff = max(max_diff, float(diff.max()))
                    mean_diff = max(mean_diff, float(diff.mean()))
            valid = valid and max_diff <= 0.02 and mean_diff <= 0.002
            report.update(valid=valid, max_diff=max_diff, mean_diff=mean_diff)
            print(f"{mode}: {report['median_ms']:.1f} ms / 256px inference, "
                  f"providers={report['providers']}, max_diff={max_diff:.6f}, "
                  f"mean_diff={mean_diff:.6f}, valid={valid}", flush=True)
    (OUT / "report.json").write_text(json.dumps(reports, indent=2))
    baseline = min((r["median_ms"] for k, r in reports.items()
                    if k in ("coreml_gpu", "coreml_all") and r["valid"]), default=float("inf"))
    candidates = [(k, r) for k, r in reports.items()
                  if k in ("candidate_gpu", "candidate_all") and r["valid"]
                  and r["providers"].get("CoreMLExecutionProvider", 0) > 0
                  and r["median_ms"] < baseline * 0.95]
    cpu_ok = reports.get("candidate_cpu", {}).get("valid", False)
    if cpu_ok and candidates:
        best_name, best = min(candidates, key=lambda pair: pair[1]["median_ms"])
        baseline_mode = "coreml_all" if best_name == "candidate_all" else "coreml_gpu"
        # A gain of a few percent needs an independent, paired remeasurement.
        print(f"Confirming {baseline_mode} versus {best_name}...", flush=True)
        repeat = {}
        for mode in (baseline_mode, best_name):
            try:
                subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                "--worker", mode, "--tag", "confirm"], check=True, timeout=300)
                repeat[mode] = json.loads((OUT / f"{mode}_confirm.json").read_text())
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                print(f"Confirmation failed for {mode}: {exc}", flush=True)
        if len(repeat) == 2:
            reference_ms = repeat[baseline_mode]["median_ms"]
            candidate_ms = repeat[best_name]["median_ms"]
            print(f"Confirmation: original {reference_ms:.1f} ms, "
                  f"candidate {candidate_ms:.1f} ms", flush=True)
            if candidate_ms < reference_ms * 0.95:
                (OUT / "approved.json").write_text(json.dumps({
                    "version": 1, "source_sha256": digest(SOURCE),
                    "candidate_sha256": digest(OUT / "alphaface_reflect.onnx"),
                    "units": MODES[best_name], "median_ms": candidate_ms,
                    "baseline_ms": reference_ms,
                }, indent=2))
                print(f"Approved {best_name} for a short real-video test: "
                      "VISOMASTER_ALPHAFACE_OPTIMIZED=1 python main.py", flush=True)
    if not (OUT / "approved.json").exists():
        print("No candidate passed accuracy and repeatable CoreML speed checks; "
              "AlphaFace selection remains unchanged.", flush=True)
    print(f"Report: {OUT / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    worker(args.worker, args.tag) if args.worker else main()
