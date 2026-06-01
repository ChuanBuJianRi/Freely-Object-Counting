#!/bin/bash
# SNG (epsilon, delta) parameter sweep on FSC-147 val.
# Single mode, mask filter min=5e-4 max=0.10 (= A6 best), PRED=max-cluster.
# Existing (eps=10, delta=3) reuses A6_SNG_max — skipped automatically.

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

# EXP_ID                EPS  DELTA
declare -a EXPERIMENTS=(
    "A6_SNG_e5_d2     5  2"
    "A6_SNG_e5_d3     5  3"
    "A6_SNG_e5_d5     5  5"
    "A6_SNG_e10_d2   10  2"
    "A6_SNG_e10_d5   10  5"
    "A6_SNG_e20_d2   20  2"
    "A6_SNG_e20_d3   20  3"
    "A6_SNG_e20_d5   20  5"
)

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "  SNG (eps, delta) Sweep — single mode, A6 mask filter"
echo "  PRED = max-cluster size, FSC-147 val (fraction=0.333, seed=42)"
echo "  Reference: A6_FINCH_max MAE=32.10  A6_SNG_max(10,3) MAE=41.67"
echo "  $(date)"
echo "============================================================"

for exp_line in "${EXPERIMENTS[@]}"; do
    read -r EXP_ID EPS DELTA <<< "$exp_line"
    OUT_DIR="$RESULTS_ROOT/$EXP_ID"

    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[SKIP] $EXP_ID — already completed"
        continue
    fi

    wait_for_cooldown

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment: $EXP_ID  (eps=$EPS, delta=$DELTA)"
    echo "  Started: $(date)  GPU: $(check_gpu_temp)°C"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$OUT_DIR"

    $PYTHON "$EVAL_SCRIPT" \
        --mode single \
        --splits val \
        --fraction 0.333 \
        --seed 42 \
        --min-mask-area 0.0005 \
        --max-mask-area 0.10 \
        --cluster-method sng \
        --sng-epsilon "$EPS" \
        --sng-delta "$DELTA" \
        --pred-strategy max \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/run.log"

    echo "  Finished: $(date)  GPU: $(check_gpu_temp)°C"
    echo "  [inter-exp] Cooling ${INTER_EXP_SLEEP}s..."
    sleep "$INTER_EXP_SLEEP"
    wait_for_cooldown
done

echo ""
echo "============================================================"
echo "  SNG sweep finished at $(date)"
echo "============================================================"
echo ""
echo "Summary:"
for d in "$RESULTS_ROOT"/A6_SNG_*; do
    if [ -f "$d/metrics.json" ]; then
        mae=$($PYTHON -c "import json; d=json.load(open('$d/metrics.json')); v=d.get('val',d); print(f\"{v['mae']:.2f}\")" 2>/dev/null)
        echo "  $(basename $d) — MAE=$mae"
    fi
done
