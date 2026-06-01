#!/bin/bash
# Mask filtering POLICY ablation on FSC-147 val (OCCAM-MULTI mode).
# Upstream fixed at A6/M6 best mask area window; downstream uses FINCH (current
# strongest in multi mode); we only vary the OCCAM-side mask filtering policy.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/../origin_simulation/eval_fsc147_full.py"
RESULTS_ROOT="$SCRIPT_DIR/results"
PYTHON="/home/czp/ws_yiyang/FreeCounting/venv/bin/python3"

# ---- Single-instance guard (refuse to run if another eval Python is alive) ----
SELF_PID=$$
EXISTING=$(pgrep -af "eval_fsc147_full.py" | awk -v me="$SELF_PID" '$1 != me {print $1}' | head -3)
if [ -n "$EXISTING" ]; then
    echo "ERROR: another eval_fsc147_full.py is already running (PIDs: $(echo $EXISTING | tr '\n' ' '))." >&2
    echo "       Refuse to start to avoid GPU contention. Kill it first." >&2
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

# Fixed shared settings (best so far).
MIN_AREA=0.0005
MAX_AREA=0.10
MODE=multi
CLUSTER=finch
PRED=max
FRAC=0.333
SEED=42

# 8 representative policies (one config per policy family).
# Format:  EXP_ID  POLICY  EXTRA_FLAGS
declare -a EXPERIMENTS=(
    "MP0_baseline      p0  "
    "MP1_score090      p1  --mask-score-thresh 0.90"
    "MP2_topk100       p2  --mask-topk 100"
    "MP3_iqr15         p3  --mask-iqr-k 1.5"
    "MP4_otsu          p4  "
    "MP5_score090_area p5  --mask-score-thresh 0.90"
    "MP6_score_nms     p6  "
    "MP7_no_filter     p7  "
)

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "  Mask Filtering POLICY Ablation (OCCAM-M, FINCH, max)"
echo "  fraction=$FRAC seed=$SEED  area=[$MIN_AREA,$MAX_AREA]"
echo "  $(date)"
echo "============================================================"
echo ""

for exp_line in "${EXPERIMENTS[@]}"; do
    # Parse via bash array so multi-space separators don't confuse cut/awk.
    # shellcheck disable=SC2206
    tokens=($exp_line)
    EXP_ID="${tokens[0]}"
    POLICY="${tokens[1]}"
    EXTRA="${tokens[*]:2}"
    OUT_DIR="$RESULTS_ROOT/$EXP_ID"

    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[SKIP] $EXP_ID — already completed"
        continue
    fi

    wait_for_cooldown

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment: $EXP_ID"
    echo "  Policy: $POLICY    Extra: $EXTRA"
    echo "  Started: $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$OUT_DIR"

    # shellcheck disable=SC2086
    $PYTHON "$EVAL_SCRIPT" \
        --mode "$MODE" \
        --splits val \
        --fraction "$FRAC" \
        --seed "$SEED" \
        --min-mask-area "$MIN_AREA" \
        --max-mask-area "$MAX_AREA" \
        --cluster-method "$CLUSTER" \
        --pred-strategy "$PRED" \
        --mask-policy "$POLICY" \
        $EXTRA \
        --output-dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/run.log"

    echo "  Finished: $(date)  GPU: $(check_gpu_temp)°C"
    echo "  [inter-exp] Cooling ${INTER_EXP_SLEEP}s..."
    sleep "$INTER_EXP_SLEEP"
    wait_for_cooldown
done

echo ""
echo "============================================================"
echo "  Mask-policy sweep finished at $(date)"
echo "============================================================"
echo "Summary (val MAE):"
for d in "$RESULTS_ROOT"/*/; do
    name=$(basename "$d")
    s="$d/summary.txt"
    if [ -f "$s" ]; then
        mae=$(grep -m1 "MAE   =" "$s" | awk '{print $3}')
        printf "  %-22s MAE=%s\n" "$name" "$mae"
    fi
done
