#!/bin/bash
# OCCAM-S repro, PATH C: paper config but tighten max_mask_area 0.5->0.10
# AND switch pred head total->max. Hypothesis: the paper's mask processing
# is more aggressive at discarding large background masks (max ~0.1-0.2);
# our 0.5 kept giant background clusters that pred=total then over-counts.
# Offline trace replay of the 0.5/total run predicted pred=max -> ~18 MAE
# (<=300); this run additionally tightens the area window at mask stage.
# Target: approach paper OCCAM-S FSC-147 test <=300 MAE 11.29 / RMSE 25.17.
set -uo pipefail
RUN=/home/czp/ws_yiyang/GOC-Freely-Object-Counting/results/2026-05-29-1928-repro-occamS-area010-max
REPO=/home/czp/ws_yiyang/GOC-Freely-Object-Counting
PY=/home/czp/ws_yiyang/FreeCounting/venv/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "$REPO"
"$PY" "$REPO/codes/eval/eval_fsc147_full.py" \
    --mode single --splits test --fraction 0.333 --seed 42 \
    --mask-backend predictor --seed-spacing 10 --duplicate-iou 0.1 \
    --enable-multiscale \
    --min-mask-area 0.0005 --max-mask-area 0.10 \
    --mask-policy p0 --cluster-method finch --pred-strategy max \
    --output-dir "$RUN" --data-dir "$REPO/datasets/FSC147" \
    --gpu-temp-limit 78 --gpu-cooldown-sec 30 --gpu-hysteresis 5 --gpu-check-every 5
