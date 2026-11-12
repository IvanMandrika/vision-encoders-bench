#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
SEED="${SEED:-0}"

if [ ! -x "$PY" ]; then
    python3.12 -m venv .venv
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install torch==2.7.1 torchvision==0.22.1
    "$PY" -m pip install -e ".[dev]" -c requirements.lock
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if [ "${ALLOW_CPU:-0}" != 1 ]; then
    "$PY" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "CUDA is not available; set ALLOW_CPU=1 to embed on CPU (hours)"; exit 1; }
fi

"$PY" -m pytest -q
"$PY" -m veb.embed --seed "$SEED"
CUDA_VISIBLE_DEVICES= "$PY" -m veb.exp_shift --seed "$SEED"
CUDA_VISIBLE_DEVICES= "$PY" -m veb.exp_coreset --seed "$SEED"
"$PY" -m veb.exp_svg_metrics --seed "$SEED"
