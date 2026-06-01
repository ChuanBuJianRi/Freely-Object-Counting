#!/bin/bash
# OCCAM-S paper reproduction (SMOKE, 8 images) on FSC-147 test 1/3.
# Paper config (Spanakis et al., Table 1/8): single-class, bbox 224,
# dense seed-grid predictor (spacing=10), IoU=0.1 dedup, 3x3 multiscale ON,
# area-window mask processing (P0), custom FINCH thresholds 12/9/7.75,
# pred=total. Target (full test, <=300 objects): MAE 11.29 / RMSE 25.17.
set -uo pipefail
SMK=/home/czp/ws_yiyang/GOC-Freely-Object-Counting/results/2026-05-29-1730-repro-occamS-smoke
REPO=/home/czp/ws_yiyang/GOC-Freely-Object-Counting
PY=/home/czp/ws_yiyang/FreeCounting/venv/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "$REPO"
"$PY" "$REPO/codes/eval/eval_fsc147_full.py" \
    --mode single --splits test --fraction 0.333 --seed 42 --limit 8 \
    --mask-backend predictor --seed-spacing 10 --duplicate-iou 0.1 \
    --enable-multiscale \
    --min-mask-area 0.0005 --max-mask-area 0.5 \
    --mask-policy p0 --cluster-method finch --pred-strategy total \
    --output-dir "$SMK" --data-dir "$REPO/datasets/FSC147" \
    --gpu-temp-limit 78 --gpu-cooldown-sec 30 --gpu-hysteresis 5 --gpu-check-every 5
