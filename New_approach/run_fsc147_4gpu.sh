#!/usr/bin/env bash
# Launch PF-CUD FSC147 full-test evaluation sharded across GPUs 4,5,6,7.
#
# Sharding: stride=4, offset=0..3 -> each GPU processes ~298 of the 1190 test
# images. Each shard is pinned to one physical GPU via CUDA_VISIBLE_DEVICES,
# so inside the process the visible GPU is index 0 (thermal guard --gpu-index 0).
#
# Config: full test split, DINOv2 visual features ON, edge candidates ON.
set -euo pipefail

PY=/home/gaoyiyang/venvs/fsc/bin/python
ROOT=/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach
OUT=$ROOT/outputs/fsc147_full_plateau
mkdir -p "$OUT"

cd "$ROOT"

GPUS=(4 5 6 7)
for i in "${!GPUS[@]}"; do
  gpu=${GPUS[$i]}
  offset=$i
  CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -m pf_cud.eval.eval_fsc147 \
    --split test \
    --stride 4 \
    --offset "$offset" \
    --use_edge \
    --gpu-index 0 \
    --out_json "$OUT/shard_offset${offset}.json" \
    > "$OUT/shard_offset${offset}.log" 2>&1 &
  echo "launched shard offset=$offset on physical GPU $gpu (pid $!)"
done

wait
echo "all shards finished"
