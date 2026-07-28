#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs/mogeflow-control"
RUN="hy273_raw_flow_hml3d_ddp8_h1024_d6s12_b16_20260708_124003"
LOG="${ROOT}/logs/${RUN}.log"
RUN_DIR="${ROOT}/checkpoints/t2m/${RUN}"
MODEL_DIR="${RUN_DIR}/model"
HEALTH_LOG="${ROOT}/logs/${RUN}.health.jsonl"

cd "${ROOT}"

while true; do
  date -Is
  /root/miniconda3/envs/mogo/bin/python tools/monitor_hy273_training.py \
    --log "${LOG}" \
    --run-dir "${RUN_DIR}" \
    --health-log "${HEALTH_LOG}"
  /root/miniconda3/envs/mogo/bin/python tools/prune_hy273_checkpoints.py \
    --model_dir "${MODEL_DIR}" \
    --keep_recent 8 \
    --keep_every 50000
  sleep 1800
done
