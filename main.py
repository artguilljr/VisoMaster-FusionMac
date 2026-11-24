import os
import sys
import argparse
import traceback
from datetime import datetime
from pathlib import Path


def _configure_bundled_ffmpeg() -> None:
    """Prefer runnable, native pip-installed media tools on macOS."""
    if sys.platform != "darwin":
        return
    import shutil
    import subprocess

    def working(name: str) -> bool:
        executable = shutil.which(name)
        if not executable:
            return False
        try:
            return subprocess.run(
                [executable, "-version"], capture_output=True, timeout=5
            ).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    # A working `ffmpeg -version` does not imply this binary supports HEVC.
    # Conda can put a minimal FFmpeg ahead of the pip binary in PATH.
    original_path = os.environ.get("PATH", "")
    try:
        from portable_ffmpeg import get_ffmpeg

        ffmpeg_path, ffprobe_path = map(Path, get_ffmpeg())
        if ffmpeg_path.parent != ffprobe_path.parent:
            raise RuntimeError("FFmpeg and FFprobe are in different directories")
        directory = str(ffmpeg_path.parent)
        os.environ["PATH"] = directory + os.pathsep + original_path
        if not (working("ffmpeg") and working("ffprobe")):
            raise RuntimeError("downloaded FFmpeg or FFprobe cannot run")
        print(f"[INFO] Using pip FFmpeg tools: {directory}")
    except (ImportError, OSError, RuntimeError) as exc:
        os.environ["PATH"] = original_path
        if working("ffmpeg") and working("ffprobe"):
            print(f"[WARN] Pip FFmpeg tools unavailable ({exc}); using FFmpeg from PATH")
        else:
            print(f"[WARN] Pip FFmpeg tools unavailable: {exc}")


_configure_bundled_ffmpeg()

# --- PyTorch VRAM Optimization ---
# MUST be set BEFORE importing torch (which occurs during _run_app).
# Instructs the caching allocator to release unused memory segments back to the CUDA driver.
# Dropped 'max_split_size_mb:128' to prevent heavy cudaMalloc/cudaFree churn and implicit
# device synchronizations when allocating large frame tensors (e.g., 4K, VR180).
# Left overridable so other values can be A/B'd against peak VRAM without a
# rebuild, e.g. PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")

import torch

torch.set_grad_enabled(False)


def _write_crash_log(exc: BaseException) -> Path:
    """Persist a full traceback to disk so the diagnostic survives even if the
    console window closes before the user can copy it.

    Returns the path of the written log so the caller can print it.
    """
    log_dir = Path(__file__).resolve().parent / "crash_logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"crash_{stamp}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"VisoMaster crash report — {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n")
        try:
            import platform

            f.write(f"Python:   {sys.version}\n")
            f.write(f"Platform: {platform.platform()}\n")
        except Exception:
            pass
        f.write("=" * 70 + "\n\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    return log_path


def _run_app() -> None:
    """Boot the Qt app. Imports are inside the function so any startup error is
    captured by the outer try/except (otherwise a top-level import error would
    bypass the crash-log writer)."""
    from app.ui import main_ui
    from PySide6 import QtWidgets

    import qdarktheme
    from app.ui.core.proxy_style import ProxyStyle

    parser = argparse.ArgumentParser(description="VisoMaster")
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA GPU device ID to use (default: 0)",
    )
    args, remaining = parser.parse_known_args()

    app = QtWidgets.QApplication(remaining)
    app.setStyle(ProxyStyle())
    with open("app/ui/styles/true_dark_styles.qss", "r") as f:
        _style = f.read()
        _style = (
            qdarktheme.load_stylesheet(
                theme="dark", custom_colors={"primary": "#4090a3"}
            )
            + "\n"
            + _style
        )
        app.setStyleSheet(_style)
    window = main_ui.MainWindow(gpu_id=args.gpu_id)
    window.show()
    app.exec()


if __name__ == "__main__":
    try:
        _run_app()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_path = _write_crash_log(e)
        print("\n" + "=" * 70)
        print("[FATAL] VisoMaster crashed.")
        print("  Crash log written to:")
        print(f"    {log_path}")
        print(f"  Error: {e}")
        print("=" * 70)
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
