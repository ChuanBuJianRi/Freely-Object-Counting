#!/bin/bash
# FSC-147 TEST split (fraction=1/3, seed=42, n=396): same upstream as MP7
# (OCCAM-M + min=5e-4/max=0.10 + p7 + finch). Prediction head = MCV v1 (no
# guard), trace saved per image so max / MCV-v1 / MCV+guard A=30 numbers
# can all be reconstructed offline (see scripts/reduce_predict_trace.py /
# the in-line analyser used at the end of this run).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="$REPO_ROOT/codes/eval/eval_fsc147_full.py"
PYTHON="${PYTHON:-/home/czp/ws_yiyang/FreeCounting/venv/bin/python3}"

# ---- single-instance guard ----
SELF_PID=$$
EXISTING=$(pgrep -af "eval_fsc147_full.py" | awk -v me="$SELF_PID" '$1 != me {print $1}' | head -3)
if [ -n "$EXISTING" ]; then
    echo "ERROR: another eval_fsc147_full.py is already running (PIDs: $EXISTING)." >&2
    exit 1
fi
GPU_USED=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
if [ "$GPU_USED" -gt 2000 ]; then
    echo "ERROR: GPU already has ${GPU_USED} MiB used (>2GiB). Refuse to start." >&2
    /usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2 || true
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "$REPO_ROOT"
exec "$PYTHON" "$EVAL_SCRIPT" \
    --mode multi \
    --splits test \
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
