#!/bin/bash
# OCCAM-M (multi mode) sweep using the best 5 (ε, δ) found in single mode.
#
# Logic:
#   1. Scan all A6_SNG_* runs in results/ that have metrics.json
#   2. Parse (epsilon, delta) from each run.log invocation line
#   3. Sort by val MAE ascending, take top 5
#   4. Run each on OCCAM-M (multi), with M6 best mask filter (min=5e-4, max=0.10)
#      and pred-strategy=max (consistent with single-mode SNG sweep)
#
# Output: results_multi/A6M_SNG_e{eps}_d{delta}/metrics.json

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/../origin_simulation/eval_fsc147_full.py"
RESULTS_SINGLE="$SCRIPT_DIR/results"
RESULTS_MULTI="$SCRIPT_DIR/results_multi"
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

# ---- Pick top-5 (eps, delta) from completed single-mode SNG runs ----
# Output format: "eps delta single_mae"  one per line, sorted by MAE asc, head 5
TOP5=$($PYTHON - <<'PY'
import json, re, glob, os
runs = []
for d in sorted(glob.glob("/home/czp/ws_yiyang/FreeCounting/ws_yiyang/OCCAM_experiments_series/ablation_clustering/results/A6_SNG_*")):
    mj = os.path.join(d, "metrics.json")
    rl = os.path.join(d, "run.log")
    if not os.path.isfile(mj):
        continue
    try:
        mae = json.load(open(mj))["val"]["mae"]
    except Exception:
        continue
    name = os.path.basename(d)
    # Try parse (eps, delta) from log first, then from dir name
    eps = delta = None
    if os.path.isfile(rl):
        head = open(rl).read(4000)
        m = re.search(r"eps=(\d+),\s*delta=(\d+)", head)
        if m:
            eps, delta = int(m.group(1)), int(m.group(2))
    if eps is None:
        m = re.search(r"e(\d+)_d(\d+)", name)
        if m:
            eps, delta = int(m.group(1)), int(m.group(2))
    if eps is None or delta is None:
        continue
    runs.append((mae, eps, delta, name))
runs.sort()
seen = set()
out = []
for mae, eps, delta, name in runs:
    key = (eps, delta)
    if key in seen:
        continue
    seen.add(key)
    out.append(f"{eps} {delta} {mae:.4f} {name}")
    if len(out) == 5:
        break
print("\n".join(out))
PY
)

mkdir -p "$RESULTS_MULTI"

echo "============================================================"
echo "  OCCAM-M (multi) sweep — FINCH baseline + top-5 SNG"
echo "  Mask filter: min=5e-4, max=0.10 (= M6 best)"
echo "  PRED: max  |  FSC-147 val (fraction=0.333, seed=42)"
echo "  $(date)"
echo "------------------------------------------------------------"
echo "  Selected SNG configs (eps  delta  single-MAE  source-name):"
echo "$TOP5" | sed 's/^/    /'
echo "============================================================"

# ---------- Step 1: FINCH + max baseline on multi mode ----------
FINCH_DIR="$RESULTS_MULTI/A6M_FINCH_max"
if [ -f "$FINCH_DIR/metrics.json" ]; then
    echo "[SKIP] A6M_FINCH_max — already completed"
else
    wait_for_cooldown
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment: A6M_FINCH_max  (multi-mode FINCH baseline, pred=max)"
    echo "  Started: $(date)  GPU: $(check_gpu_temp)°C"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    mkdir -p "$FINCH_DIR"
    $PYTHON "$EVAL_SCRIPT" \
        --mode multi \
        --splits val \
        --fraction 0.333 \
        --seed 42 \
        --min-mask-area 0.0005 \
        --max-mask-area 0.10 \
        --cluster-method finch \
        --pred-strategy max \
        --output-dir "$FINCH_DIR" \
        2>&1 | tee "$FINCH_DIR/run.log"
    echo "  Finished: $(date)  GPU: $(check_gpu_temp)°C"
    echo "  [inter-exp] Cooling ${INTER_EXP_SLEEP}s..."
    sleep "$INTER_EXP_SLEEP"
    wait_for_cooldown
fi

# ---------- Step 2: Top-5 SNG configs on multi mode ----------

if [ -z "$TOP5" ]; then
    echo "ERROR: no completed single-mode SNG runs found. Aborting."
    exit 1
fi

# Iterate
while IFS= read -r line; do
    [ -z "$line" ] && continue
    EPS=$(echo "$line" | awk '{print $1}')
    DELTA=$(echo "$line" | awk '{print $2}')
    SRC_MAE=$(echo "$line" | awk '{print $3}')

    EXP_ID="A6M_SNG_e${EPS}_d${DELTA}"
    OUT_DIR="$RESULTS_MULTI/$EXP_ID"

    if [ -f "$OUT_DIR/metrics.json" ]; then
        echo "[SKIP] $EXP_ID — already completed"
        continue
    fi

    wait_for_cooldown

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Experiment: $EXP_ID  (eps=$EPS, delta=$DELTA, single-MAE=$SRC_MAE)"
    echo "  Started: $(date)  GPU: $(check_gpu_temp)°C"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    mkdir -p "$OUT_DIR"

    $PYTHON "$EVAL_SCRIPT" \
        --mode multi \
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
done <<< "$TOP5"

echo ""
echo "============================================================"
echo "  OCCAM-M top-5 sweep finished at $(date)"
echo "============================================================"
echo "Summary (single-MAE → multi-MAE):"
if [ -f "$RESULTS_MULTI/A6M_FINCH_max/metrics.json" ]; then
    fmae=$($PYTHON -c "import json; print(f\"{json.load(open('$RESULTS_MULTI/A6M_FINCH_max/metrics.json'))['val']['mae']:.2f}\")")
    echo "  FINCH (baseline)            multi=$fmae"
fi
while IFS= read -r line; do
    [ -z "$line" ] && continue
    EPS=$(echo "$line" | awk '{print $1}')
    DELTA=$(echo "$line" | awk '{print $2}')
    SRC_MAE=$(echo "$line" | awk '{print $3}')
    MULTI_MJ="$RESULTS_MULTI/A6M_SNG_e${EPS}_d${DELTA}/metrics.json"
    if [ -f "$MULTI_MJ" ]; then
        multi_mae=$($PYTHON -c "import json; print(f\"{json.load(open('$MULTI_MJ'))['val']['mae']:.2f}\")")
        echo "  SNG ε=$EPS δ=$DELTA  single=$SRC_MAE  multi=$multi_mae"
    fi
done <<< "$TOP5"
