#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29761}"
STEPS="${STEPS:-50000}"
[[ "${STEPS}" == "50000" ]] || {
  echo "The physical-temporal comparison requires STEPS=50000" >&2
  exit 2
}

RUN_NAME="${RUN_NAME:-hy273_r14_physical_temporal_positive_ddp4_400k_to450k_${STAMP}}"
LOG_PATH="${ROOT_DIR}/run_logs/${RUN_NAME}.log"
SESSION="${SESSION:-r14_physical_temporal_${STAMP}}"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT_DIR}' && env \
    TREATMENT=physical_temporal_positive_only \
    CONFIG=configs/hy273_multitask_r13_stage_c1_decomposed_cfg_edit.yaml \
    RUN_NAME='${RUN_NAME}' \
    GPU_IDS='${GPU_IDS}' \
    NPROC_PER_NODE=4 \
    BATCH_SIZE_PER_RANK=32 \
    SMOKE_STEPS='${STEPS}' \
    SAVE_SMOKE=1 \
    FRESH_FORK=1 \
    RESEARCH_NO_UPDATE=0 \
    MASTER_PORT='${MASTER_PORT}' \
    bash scripts/launch/train_hy273_r13_edit_objective_pilot_ddp8.sh \
    2>&1 | tee '${LOG_PATH}'"

echo "launched treatment=physical_temporal_positive_only"
echo "gpus=${GPU_IDS}"
echo "tmux=${SESSION}"
echo "log=${LOG_PATH}"
