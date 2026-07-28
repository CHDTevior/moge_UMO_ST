#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-50000}"
[[ "${STEPS}" == "50000" ]] || {
  echo "The registered decomposed-CFG A/B requires STEPS=50000" >&2
  exit 2
}

CONFIG="${CONFIG:-configs/hy273_multitask_r13_stage_c1_decomposed_cfg_edit.yaml}"
LOG_DIR="${ROOT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

launch_arm() {
  local treatment="$1"
  local gpu_ids="$2"
  local port="$3"
  local run_name="hy273_r13_decompcfg_${treatment}_ddp4_400k_to450k_${STAMP}"
  local log_path="${LOG_DIR}/${run_name}.log"
  local session="r13_decompcfg_${treatment}_${STAMP}"

  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && env \
      TREATMENT='${treatment}' \
      CONFIG='${CONFIG}' \
      RUN_NAME='${run_name}' \
      GPU_IDS='${gpu_ids}' \
      NPROC_PER_NODE=4 \
      BATCH_SIZE_PER_RANK=32 \
      SMOKE_STEPS='${STEPS}' \
      SAVE_SMOKE=1 \
      FRESH_FORK=1 \
      RESEARCH_NO_UPDATE=0 \
      MASTER_PORT='${port}' \
      bash scripts/launch/train_hy273_r13_edit_objective_pilot_ddp8.sh \
      2>&1 | tee '${log_path}'"
  echo "launched treatment=${treatment} gpus=${gpu_ids} tmux=${session} log=${log_path}"
}

launch_arm no_rank_positive_only 0,1,2,3 "${PORT_A:-29751}"
launch_arm same_source_changed_positive_only 4,5,6,7 "${PORT_B:-29752}"
