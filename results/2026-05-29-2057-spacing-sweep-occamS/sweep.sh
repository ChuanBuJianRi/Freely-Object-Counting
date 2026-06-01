#!/bin/bash
# seed-spacing sweep on OCCAM-S paper config, FSC-147 test, fraction=0.15 (fast).
# Fixed: single mode, predictor backend, IoU=0.1, multiscale ON, max_area=0.10,
#        P0, FINCH 12/9/7.75, pred=max.  Varying: --seed-spacing in {10,15,20}.
# spacing=10 included as a same-fraction control (the 0.333 run is not comparable).
# Hypothesis: spacing=10 over-segments mid-density (51-200) images on 384px
# inputs; relaxing to 15/20 may yield cleaner clusters and lower 51-200 MAE.
# Resumable: a sub-run with metrics.json already present is skipped.
# GPU thermal guard is handled inside the evaluator (--gpu-* flags).
set -uo pipefail
RUN=/home/czp/ws_yiyang/GOC-Freely-Object-Counting/results/2026-05-29-2057-spacing-sweep-occamS
REPO=/home/czp/ws_yiyang/GOC-Freely-Object-Counting
PY=/home/czp/ws_yiyang/FreeCounting/venv/bin/python3
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "$REPO"

for SP in 10 15 20; do
    OUT="$RUN/spacing_${SP}"
    if [ -f "$OUT/metrics.json" ]; then
        echo "[SKIP] spacing=$SP already done"
        continue
    fi
    mkdir -p "$OUT"
    echo "==== spacing=$SP  $(date) ===="
    "$PY" "$REPO/codes/eval/eval_fsc147_full.py" \
        --mode single --splits test --fraction 0.15 --seed 42 \
        --mask-backend predictor --seed-spacing "$SP" --duplicate-iou 0.1 \
        --enable-multiscale \
        --min-mask-area 0.0005 --max-mask-area 0.10 \
        --mask-policy p0 --cluster-method finch --pred-strategy max \
        --output-dir "$OUT" --data-dir "$REPO/datasets/FSC147" \
        --gpu-temp-limit 78 --gpu-cooldown-sec 30 --gpu-hysteresis 5 --gpu-check-every 5 \
        > "$OUT/run.out" 2>&1
    echo "==== spacing=$SP done  $(date) ===="
done
echo "ALL SPACING SWEEP DONE $(date)"
