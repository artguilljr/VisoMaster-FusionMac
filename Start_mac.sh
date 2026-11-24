#!/usr/bin/env bash
# VisoMaster Fusion launcher for macOS.
#
# The Windows entry points (Start.bat / Start_Portable.bat) assume a portable
# NVIDIA runtime that does not exist here. This script just activates the local
# venv and runs the app from the repo root, which the Qt stylesheet loading in
# main.py depends on.
#
# First-time setup:
#   uv venv --python 3.12 .venv
#   uv pip install --python .venv/bin/python -r requirements_mac_arm64.txt  # Apple Silicon
#   uv pip install --python .venv/bin/python -r requirements_mac.txt        # Intel
#   .venv/bin/python download_models.py
#   Both macOS requirements files install FFmpeg and FFprobe through pip.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -x .venv/bin/python ]]; then
    echo "[FATAL] .venv not found. Run the first-time setup in the header of this script." >&2
    exit 1
fi

exec .venv/bin/python main.py "$@"
