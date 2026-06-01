#!/bin/bash
# Smoke test: --limit 3 across all 8 policies to ensure no code path is broken.
# Includes the same GPU-temperature guard as the full sweep.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="$SCRIPT_DIR/../origin_simulation/eval_fsc147_full.py"
PYTHON="/home/czp/ws_yiyang/FreeCounting/venv/bin/python3"
TMP="$SCRIPT_DIR/_smoke"
mkdir -p "$TMP"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export PYTHONUNBUFFERED=1

TEMP_LIMIT=74
COOLDOWN_SEC=45
INTER_EXP_SLEEP=15

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
        echo "  [thermal] GPU=${temp}°C >= ${TEMP_LIMIT}°C, sleeping ${COOLDOWN_SEC}s..."
        sleep "$COOLDOWN_SEC"
    done
}

for line in \
    "P0 p0" \
    "P1 p1 --mask-score-thresh 0.90" \
    "P2 p2 --mask-topk 100" \
    "P3 p3 --mask-iqr-k 1.5" \
    "P4 p4" \
    "P5 p5 --mask-score-thresh 0.90" \
    "P6 p6" \
    "P7 p7" ; do
    name=$(echo "$line" | awk '{print $1}')
    pol=$(echo "$line" | awk '{print $2}')
    extra=$(echo "$line" | cut -d' ' -f3-)
    [ "$extra" = "$pol" ] && extra=""

    wait_for_cooldown
    echo "===== $name ($pol) $extra   GPU=$(check_gpu_temp)°C ====="
    out="$TMP/$name"
    rm -rf "$out"
    mkdir -p "$out"
    # shellcheck disable=SC2086
    $PYTHON "$EVAL_SCRIPT" \
        --mode multi --splits val --fraction 0.01 --seed 42 --limit 3 \
        --min-mask-area 0.0005 --max-mask-area 0.10 \
        --cluster-method finch --pred-strategy max \
        --mask-policy "$pol" $extra \
        --output-dir "$out" 2>&1 | tail -25
    echo "  done at $(date +%H:%M:%S)  GPU=$(check_gpu_temp)°C"
    sleep "$INTER_EXP_SLEEP"
    echo
done
echo "ALL POLICIES SMOKE-TESTED OK"
