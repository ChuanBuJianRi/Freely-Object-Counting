#!/bin/bash
set -uo pipefail
RUN_DIR=/home/czp/ws_yiyang/GOC-Freely-Object-Counting/results/2026-05-29-1250-eval-mp7-mcv-test
REPO_ROOT=/home/czp/ws_yiyang/GOC-Freely-Object-Counting
PYTHON=/home/czp/ws_yiyang/FreeCounting/venv/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "$REPO_ROOT"
"$PYTHON" "$REPO_ROOT/codes/eval/eval_fsc147_full.py" \
    --mode multi --splits test --fraction 0.333 --seed 42 \
    --min-mask-area 0.0005 --max-mask-area 0.10 \
    --mask-policy p7 --cluster-method finch \
    --pred-strategy mode_cluster_vote \
    --output-dir "$RUN_DIR" \
    --data-dir "$REPO_ROOT/datasets/FSC147" \
    --gpu-temp-limit 78 --gpu-cooldown-sec 30 --gpu-hysteresis 5 --gpu-check-every 5
