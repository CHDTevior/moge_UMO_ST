#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the Unified Reaction run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/hy273_unified_reaction}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
TRAIN_SESSION="${TRAIN_SESSION:-hy273_unified_reaction_continue250k}"
POLL_SECONDS="${POLL_SECONDS:-300}"
FINAL_CHECKPOINT="${RUN_ROOT}/model/step_00250000.pt"
LOG_PATH="${RUN_ROOT}/logs/eval_stage_b_250k_auto.log"

while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep "${POLL_SECONDS}"
done

if [[ ! -f "${FINAL_CHECKPOINT}" ]]; then
  echo "Training ended without ${FINAL_CHECKPOINT}; evaluation not started" >&2
  exit 3
fi

RUN_NAME="${RUN_NAME}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash scripts/launch/eval_hy273_unified_reaction_stage_b_250k.sh \
  2>&1 | tee "${LOG_PATH}"
