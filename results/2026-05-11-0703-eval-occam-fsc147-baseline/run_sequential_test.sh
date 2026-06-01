#!/bin/bash
# 等 val 进程结束后，在 GPU 0 跑 test
echo "[$(date)] 等待 val 进程 (PID 997766) 完成..."
while kill -0 997766 2>/dev/null; do
    sleep 30
done
echo "[$(date)] val 完成，开始跑 test (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/gyy/.conda/envs/occam/bin/python3 -u eval_fsc147_full.py \
  --mode single --device cuda --splits test \
  >> run_test.log 2>&1
echo "[$(date)] test 完成"
