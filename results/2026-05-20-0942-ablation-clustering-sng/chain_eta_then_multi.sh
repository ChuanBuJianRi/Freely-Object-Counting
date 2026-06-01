#!/bin/bash
# Wait for the η-sweep to finish, then launch top-5 OCCAM-M multi sweep.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[chain] Waiting for η-sweep to complete..."

# Poll: η-sweep is done when no eval_fsc147 process is running
# AND all 5 expected eta dirs have metrics.json
EXPECTED=(
    "A6_SNG_e8_d3_eta039"
    "A6_SNG_e10_d4_eta040"
    "A6_SNG_e12_d5_eta040"
    "A6_SNG_e15_d6_eta036"
    "A6_SNG_e10_d6_eta064"
)

while true; do
    all_done=1
    for name in "${EXPECTED[@]}"; do
        if [ ! -f "$SCRIPT_DIR/results/$name/metrics.json" ]; then
            all_done=0
            break
        fi
    done
    running=$(pgrep -af "eval_fsc147" | grep -v pgrep | wc -l)
    if [ "$all_done" = "1" ] && [ "$running" = "0" ]; then
        break
    fi
    sleep 60
done

echo "[chain] η-sweep done at $(date). Launching OCCAM-M top-5 sweep..."
sleep 30
exec bash "$SCRIPT_DIR/run_sng_multi_top5.sh"
