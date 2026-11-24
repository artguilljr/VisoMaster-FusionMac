#!/bin/bash
# Terminal version of the same launcher, useful when you want to see live logs.
APP_DIR=/Applications/visomaster-fusion
PYTHON_BIN=/opt/miniconda3/envs/visomaster-mac/bin/python
if [[ ! -f "$APP_DIR/main.py" || ! -x "$PYTHON_BIN" ]]; then
    echo "VisoMaster or visomaster-mac Python is missing; expected:" >&2
    echo "$APP_DIR/main.py" >&2
    echo "$PYTHON_BIN" >&2
    read -r -p "Press Return to close..." _ || true
    exit 1
fi
export PATH="${PYTHON_BIN%/*}:$PATH"
export VISOMASTER_COREML_STATIC=1
export VISOMASTER_COREML_GPU_ONLY=1
export VISOMASTER_INSWAPPER_OPTIMIZED=1
export VISOMASTER_ALPHAFACE_OPTIMIZED=1
cd "$APP_DIR" || exit 1
"$PYTHON_BIN" main.py
status=$?
if (( status != 0 )); then
    echo "VisoMaster exited with status $status." >&2
    read -r -p "Press Return to close..." _ || true
fi
exit "$status"
