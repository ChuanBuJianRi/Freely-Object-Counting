#!/bin/bash
# Mask area ratio ablation experiments on FSC-147 val set.
# Uses fraction=0.333 (same as origin_simulation baseline) for fast screening.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/../origin_simulation/eval_fsc147_full.py"
RESULTS_ROOT="$SCRIPT_DIR/results"
PYTHON="/home/czp/ws_yiyang/FreeCounting/venv/bin/python3"

# GPU memory limit: restrict to ~16GB (of 24GB) to avoid OOM / thermal issues
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export PYTHONUNBUFFERED=1

# Temperature monitoring: pause if GPU temp exceeds this threshold (Celsius)
TEMP_LIMIT=80
COOLDOWN_SEC=60

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

# Experiment configurations: EXP_ID  MIN_MASK  MAX_MASK
declare -a EXPERIMENTS=(
    "A0_baseline   0.0005  0.5"
    "A1_min0001    0.0001  0.5"
    "A2_min001     0.001   0.5"
    "A3_min005     0.005   0.5"
    "A4_min01      0.01    0.5"
    "A5_max025     0.0005  0.25"
    "A6_max010     0.0005  0.10"
    "A7_tight      0.001   0.25"
)

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "  Mask Area Ratio Ablation — FSC-147 val (fraction=0.333)"
echo "  $(date)"
echo "============================================================"
echo ""

for exp_line in "${EXPERIMENTS[@]}"; do
    read -r EXP_ID MIN_MASK MAX_MASK <<< "$exp_line"
    OUT_DIR="$RESULTS_ROOT/$EXP_ID"

    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[SKIP] $EXP_ID — already completed"
        continue
    fi

    wait_for_cooldown

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment: $EXP_ID"
    echo "  min_mask_area=$MIN_MASK  max_mask_area=$MAX_MASK"
    echo "  Started: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$OUT_DIR"

    $PYTHON "$EVAL_SCRIPT" \
        --mode single \
        --splits val \
        --fraction 0.333 \
        --seed 42 \
        --min-mask-area "$MIN_MASK" \
        --max-mask-area "$MAX_MASK" \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/run.log"

    echo "  Finished: $(date)"
    echo "  GPU temp: $(check_gpu_temp)°C"
done

echo ""
echo "============================================================"
echo "  All experiments done at $(date)"
echo "============================================================"

# Summary table
echo ""
echo "| Experiment | min_ratio | max_ratio | MAE | RMSE | NAE |"
echo "|------------|-----------|-----------|-----|------|-----|"
for exp_line in "${EXPERIMENTS[@]}"; do
    read -r EXP_ID MIN_MASK MAX_MASK <<< "$exp_line"
    MF="$RESULTS_ROOT/$EXP_ID/metrics.json"
    if [ -f "$MF" ]; then
        MAE=$($PYTHON -c "import json; d=json.load(open('$MF')); print(d['val']['mae'])")
        RMSE=$($PYTHON -c "import json; d=json.load(open('$MF')); print(d['val']['rmse'])")
        NAE=$($PYTHON -c "import json; d=json.load(open('$MF')); print(d['val']['nae'])")
        echo "| $EXP_ID | $MIN_MASK | $MAX_MASK | $MAE | $RMSE | $NAE |"
    else
        echo "| $EXP_ID | $MIN_MASK | $MAX_MASK | MISSING | MISSING | MISSING |"
    fi
done
