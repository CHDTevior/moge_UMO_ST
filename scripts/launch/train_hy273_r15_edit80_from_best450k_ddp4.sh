#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29785}"
STEPS="${STEPS:-50000}"
[[ "${STEPS}" == "50000" ]] || {
  echo "The registered Edit80 continuation requires STEPS=50000" >&2
  exit 2
}

PARENT="${PARENT:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r13_decompcfg_no_rank_positive_only_ddp4_400k_to450k_20260723_motionfix_decompcfg/model/step_00450000.pt}"
RUN_NAME="${RUN_NAME:-hy273_r15_edit80_from_positive450k_ddp4_${STAMP}}"
LOG_PATH="${ROOT_DIR}/run_logs/${RUN_NAME}.log"
SESSION="${SESSION:-r15_edit80_${STAMP}}"

[[ -f "${PARENT}" ]] || { echo "Missing checkpoint: ${PARENT}" >&2; exit 2; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT_DIR}' && env \
    TREATMENT=no_rank_positive_only \
    CONFIG=configs/hy273_multitask_r15_stage_c2_edit80_from_positive450k.yaml \
    RESUME='${PARENT}' \
    RUN_NAME='${RUN_NAME}' \
    GPU_IDS='${GPU_IDS}' \
    NPROC_PER_NODE=4 \
    BATCH_SIZE_PER_RANK=32 \
    MATERIALIZE_WORKERS=4 \
    SMOKE_STEPS='${STEPS}' \
    SAVE_SMOKE=1 \
    FRESH_FORK=1 \
    RESEARCH_NO_UPDATE=0 \
    MASTER_PORT='${MASTER_PORT}' \
    bash scripts/launch/train_hy273_r13_edit_objective_pilot_ddp8.sh \
    2>&1 | tee '${LOG_PATH}'"

echo "launched task_mix=t2m10_control10_edit80"
echo "edit_inner=source_text75_identity10_text_only5_unconditional5_source_text_control5"
echo "parent=${PARENT}"
echo "gpus=${GPU_IDS}"
echo "tmux=${SESSION}"
echo "log=${LOG_PATH}"
