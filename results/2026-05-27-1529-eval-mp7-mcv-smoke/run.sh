#!/bin/bash
# SMOKE test: MP7 + MCV on FSC-147 val, --limit 20.
# Verifies pipeline + MCV prediction head end-to-end on real FSC-147 images.
# Same upstream as MP7 (multi mode + min=5e-4/max=0.10 + p7 + finch),
# only changes pred_strategy to mode_cluster_vote.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="$REPO_ROOT/codes/eval/eval_fsc147_full.py"
PYTHON="/home/czp/ws_yiyang/FreeCounting/venv/bin/python3"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "$REPO_ROOT"

"$PYTHON" "$EVAL_SCRIPT" \
    --mode multi \
    --splits val \
    --fraction 0.333 \
    --seed 42 \
    --limit 20 \
    --min-mask-area 0.0005 \
    --max-mask-area 0.10 \
    --mask-policy p7 \
    --cluster-method finch \
    --pred-strategy mode_cluster_vote \
    --output-dir "$SCRIPT_DIR" \
    --data-dir "$REPO_ROOT/datasets/FSC147" \
    --gpu-temp-limit 74 \
    --gpu-cooldown-sec 30 \
    --gpu-hysteresis 5 \
    --gpu-check-every 5 \
    2>&1 | tee "$SCRIPT_DIR/run.log"
