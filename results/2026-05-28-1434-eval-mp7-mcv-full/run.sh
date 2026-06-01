#!/bin/bash
# Full FSC-147 val (1/3, seed=42): OCCAM-M + M6 area + P7 + FINCH + MCV.
# Baseline: MP7_no_filter + pred=max (MAE 30.71, 201+ MAE 312.96).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="$REPO_ROOT/codes/eval/eval_fsc147_full.py"
PYTHON="${PYTHON:-$REPO_ROOT/../FreeCounting/venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,expandable_segments:True}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "$REPO_ROOT/codes"
exec "$PYTHON" "$EVAL_SCRIPT" \
  --mode multi \
  --splits val \
  --fraction 0.333 \
  --seed 42 \
  --min-mask-area 0.0005 \
  --max-mask-area 0.10 \
  --mask-policy p7 \
  --cluster-method finch \
  --pred-strategy mode_cluster_vote \
  --output-dir "$SCRIPT_DIR" \
  --data-dir "$REPO_ROOT/datasets/FSC147" \
  --gpu-temp-limit 78 \
  --gpu-cooldown-sec 30 \
  --gpu-hysteresis 5 \
  --gpu-check-every 5 \
  2>&1 | tee "$SCRIPT_DIR/run.log"
