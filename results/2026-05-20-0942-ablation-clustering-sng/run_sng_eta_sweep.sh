#!/bin/bash
# Theory-driven SNG sweep: pick (eps, delta) by parameter health indicator η.
# η = (δ - ε²/n) / (ε - 1 - ε²/n), n≈150 on FSC-147 val.
# Sweet spot η ≈ 0.35–0.65 (sec 6.4 of Model_innovation.md).

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

# Theory-driven candidates (η computed at n=150)
# EXP_ID                    EPS  DELTA   # η     reason
declare -a EXPERIMENTS=(
    "A6_SNG_e8_d3_eta039      8  3"   # 0.39 sweet
    "A6_SNG_e10_d4_eta040    10  4"   # 0.40 sweet — fills ε=10 row
    "A6_SNG_e12_d5_eta040    12  5"   # 0.40 sweet — between ε=10 and ε=20
    "A6_SNG_e15_d6_eta036    15  6"   # 0.36 sweet — boundary case
    "A6_SNG_e10_d6_eta064    10  6"   # 0.64 above sweet — δ increase test
)

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "  SNG η-driven Sweep (theory test, sec 6.4)"
echo "  Mask filter: A6 best (min=5e-4, max=0.10)  PRED: max"
echo "  FSC-147 val (fraction=0.333, seed=42)"
echo "  Reference: A6_SNG_e10_d5 (η=0.52) MAE=39.63"
echo "             A6_FINCH_max         MAE=32.10"
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
echo "  η sweep finished at $(date)"
echo "============================================================"
echo ""
echo "Summary:"
for d in "$RESULTS_ROOT"/A6_SNG_e*_eta*; do
    if [ -f "$d/metrics.json" ]; then
        mae=$($PYTHON -c "import json; v=json.load(open('$d/metrics.json'))['val']; print(f\"{v['mae']:.2f}\")" 2>/dev/null)
        echo "  $(basename $d) — MAE=$mae"
    fi
done
