#!/bin/bash
# Mask area ratio ablation experiments on FSC-147 val set — OCCAM-M (multi) mode.
# Uses fraction=0.333 (same as baseline) for fast screening.

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

declare -a EXPERIMENTS=(
    "M0_baseline   0.0005  0.5"
    "M1_min0001    0.0001  0.5"
    "M2_min001     0.001   0.5"
    "M3_min005     0.005   0.5"
    "M4_min01      0.01    0.5"
    "M5_max025     0.0005  0.25"
    "M6_max010     0.0005  0.10"
    "M7_tight      0.001   0.25"
)

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "  Mask Area Ratio Ablation (OCCAM-M) — FSC-147 val (fraction=0.333)"
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
    echo "  Experiment: $EXP_ID (OCCAM-M multi)"
    echo "  min_mask_area=$MIN_MASK  max_mask_area=$MAX_MASK"
    echo "  Started: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$OUT_DIR"

    $PYTHON "$EVAL_SCRIPT" \
        --mode multi \
        --splits val \
        --fraction 0.333 \
        --seed 42 \
        --min-mask-area "$MIN_MASK" \
        --max-mask-area "$MAX_MASK" \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/run.log"

    echo "  Finished: $(date)"
    echo "  GPU temp: $(check_gpu_temp)°C"

    # Cool down between experiments
    echo "  [inter-exp] Cooling ${INTER_EXP_SLEEP}s between experiments..."
    sleep "$INTER_EXP_SLEEP"
    wait_for_cooldown
done

echo ""
echo "============================================================"
echo "  All OCCAM-M experiments done at $(date)"
echo "============================================================"
