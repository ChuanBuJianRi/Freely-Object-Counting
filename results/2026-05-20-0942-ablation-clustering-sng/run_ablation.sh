#!/bin/bash
# Clustering-method ablation on FSC-147 val.
# Compares FINCH vs SNG (epsilon=10, delta=3) under both A6 (single, min=5e-4 max=0.10)
# and M6 (multi, min=5e-4 max=0.10) configurations. PRED uses the LARGEST cluster's
# mask count (--pred-strategy max), reporting MAE/MSE/RMSE/NAE on (max-cluster, GT).

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/../origin_simulation/eval_fsc147_full.py"
RESULTS_ROOT="$SCRIPT_DIR/results"
PYTHON="/home/czp/ws_yiyang/FreeCounting/venv/bin/python3"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

TEMP_LIMIT=74
COOLDOWN_SEC=60
INTER_EXP_SLEEP=30

check_gpu_temp() {
    local temp
    temp=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")
    echo "$temp"
}

wait_for_cooldown() {
    while true; do
        local temp
        temp=$(check_gpu_temp)
        if [ "$temp" -lt "$TEMP_LIMIT" ]; then
            break
        fi
        echo "  [thermal] GPU temp=${temp}°C >= ${TEMP_LIMIT}°C, cooling down ${COOLDOWN_SEC}s..."
        sleep "$COOLDOWN_SEC"
    done
}

# EXP_ID  MODE  MIN  MAX  CLUSTER  EPS  DELTA
declare -a EXPERIMENTS=(
    "A6_FINCH_max  single  0.0005  0.10  finch  10  3"
    "A6_SNG_max    single  0.0005  0.10  sng    10  3"
    "M6_FINCH_max  multi   0.0005  0.10  finch  10  3"
    "M6_SNG_max    multi   0.0005  0.10  sng    10  3"
)

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "  Clustering Ablation (FINCH vs SNG, PRED=max-cluster)"
echo "  FSC-147 val (fraction=0.333, seed=42)"
echo "  $(date)"
echo "============================================================"

for exp_line in "${EXPERIMENTS[@]}"; do
    read -r EXP_ID MODE MIN_MASK MAX_MASK CLUSTER EPS DELTA <<< "$exp_line"
    OUT_DIR="$RESULTS_ROOT/$EXP_ID"

    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[SKIP] $EXP_ID — already completed"
        continue
    fi

    wait_for_cooldown

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment: $EXP_ID"
    echo "  mode=$MODE  cluster=$CLUSTER  eps=$EPS  delta=$DELTA"
    echo "  min_mask=$MIN_MASK  max_mask=$MAX_MASK  PRED=max-cluster"
    echo "  Started: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$OUT_DIR"

    $PYTHON "$EVAL_SCRIPT" \
        --mode "$MODE" \
        --splits val \
        --fraction 0.333 \
        --seed 42 \
        --min-mask-area "$MIN_MASK" \
        --max-mask-area "$MAX_MASK" \
        --cluster-method "$CLUSTER" \
        --sng-epsilon "$EPS" \
        --sng-delta "$DELTA" \
        --pred-strategy max \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/run.log"

    echo "  Finished: $(date)"
    echo "  GPU temp: $(check_gpu_temp)°C"
    echo "  [inter-exp] Cooling ${INTER_EXP_SLEEP}s between experiments..."
    sleep "$INTER_EXP_SLEEP"
    wait_for_cooldown
done

echo ""
echo "============================================================"
echo "  Clustering ablation finished at $(date)"
echo "============================================================"
