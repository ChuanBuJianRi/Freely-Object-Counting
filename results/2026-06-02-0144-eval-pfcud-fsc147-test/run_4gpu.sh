#!/usr/bin/env bash
# Launch PF-CUD FSC-147 test evaluation sharded across GPUs 4/5/6/7.
# Each shard = images[offset::4]; shard k runs on physical GPU (4+k) with the
# thermal guard watching logical index 0 (== that physical card under
# CUDA_VISIBLE_DEVICES). Results merged afterwards by merge_shards.py.
set -euo pipefail

RUN_DIR="/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/results/2026-06-02-0144-eval-pfcud-fsc147-test"
PROJECT="/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach"
DATASET="/home/gaoyiyang/ws_yiyang/datasets/FSC147"
VENV="/home/gaoyiyang/venvs/fsc/bin/activate"

GPUS=(4 5 6 7)

source "$VENV"
cd "$PROJECT"

for k in 0 1 2 3; do
  gpu="${GPUS[$k]}"
  CUDA_VISIBLE_DEVICES="$gpu" python -m pf_cud.eval.eval_fsc147 \
    --split test \
    --dataset_root "$DATASET" \
    --stride 4 --offset "$k" \
    --out_json "$RUN_DIR/per_image_test_shard${k}.json" \
    --gpu-index 0 \
    > "$RUN_DIR/run_shard${k}.log" 2>&1 &
  echo "shard $k -> GPU $gpu (pid $!)"
done

wait
echo "all shards finished"
