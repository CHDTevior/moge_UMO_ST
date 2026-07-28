#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29771}"
STEPS="${STEPS:-5000}"
SAVE_SMOKE="${SAVE_SMOKE:-1}"
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] && (( STEPS <= 50000 )) || {
  echo "STEPS must be in [1,50000]" >&2
  exit 2
}
[[ "${SAVE_SMOKE}" == "0" || "${SAVE_SMOKE}" == "1" ]] || {
  echo "SAVE_SMOKE must be 0 or 1" >&2
  exit 2
}
END_STEP=$((400000 + STEPS))

RUN_NAME="${RUN_NAME:-hy273_r15_source_token_block_positive_ddp8_400k_to${END_STEP}_${STAMP}}"
LOG_PATH="${ROOT_DIR}/run_logs/${RUN_NAME}.log"
SESSION="${SESSION:-r15_source_token_${STAMP}}"

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT_DIR}' && env \
    TREATMENT=source_token_block_positive_only \
    CONFIG=configs/hy273_multitask_r13_stage_c1_decomposed_cfg_edit.yaml \
    RUN_NAME='${RUN_NAME}' \
    GPU_IDS='${GPU_IDS}' \
    NPROC_PER_NODE=8 \
    BATCH_SIZE_PER_RANK=16 \
    SMOKE_STEPS='${STEPS}' \
    SAVE_SMOKE='${SAVE_SMOKE}' \
    FRESH_FORK=1 \
    RESEARCH_NO_UPDATE=0 \
    MASTER_PORT='${MASTER_PORT}' \
    bash scripts/launch/train_hy273_r13_edit_objective_pilot_ddp8.sh \
    2>&1 | tee '${LOG_PATH}'"

echo "launched treatment=source_token_block_positive_only"
echo "gpus=${GPU_IDS}"
echo "tmux=${SESSION}"
echo "log=${LOG_PATH}"
