#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-5000}"
[[ "${STEPS}" == "5000" ]] || {
  echo "The registered R15 matched A/B requires STEPS=5000" >&2
  exit 2
}

CONFIG="${CONFIG:-configs/hy273_multitask_r13_stage_c1_decomposed_cfg_edit.yaml}"
LOG_DIR="${ROOT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

launch_arm() {
  local label="$1"
  local treatment="$2"
  local gpu_ids="$3"
  local port="$4"
  local run_name="hy273_r15_matched_${label}_ddp4_400k_to405k_${STAMP}"
  local log_path="${LOG_DIR}/${run_name}.log"
  local session="r15_${label}_${STAMP}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 2
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && env \
      TREATMENT='${treatment}' \
      CONFIG='${CONFIG}' \
      RUN_NAME='${run_name}' \
      GPU_IDS='${gpu_ids}' \
      NPROC_PER_NODE=4 \
      BATCH_SIZE_PER_RANK=32 \
      SMOKE_STEPS=5000 \
      SAVE_SMOKE=1 \
      FRESH_FORK=1 \
      RESEARCH_NO_UPDATE=0 \
      MASTER_PORT='${port}' \
      bash scripts/launch/train_hy273_r13_edit_objective_pilot_ddp8.sh \
      2>&1 | tee '${log_path}'"
  echo "launched label=${label} treatment=${treatment} gpus=${gpu_ids}"
  echo "tmux=${session}"
  echo "log=${log_path}"
}

launch_arm additive no_rank_positive_only 0,1,2,3 "${PORT_A:-29781}"
launch_arm token_block source_token_block_positive_only 4,5,6,7 "${PORT_B:-29782}"
