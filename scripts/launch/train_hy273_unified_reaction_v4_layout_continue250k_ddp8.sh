#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v4_layout_20260804_151750}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v4_layout}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model/latest.pt}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BASE_CONFIG="${BASE_CONFIG:-configs/hy273_unified_fulltext_reaction_v4_layout.yaml}"
EXTENDED_CONFIG="${EXTENDED_CONFIG:-configs/hy273_unified_fulltext_reaction_v4_layout_continue250k.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Reaction-v4 continuation checkpoint: ${CHECKPOINT}" >&2
  exit 2
}
[[ -d "${RUN_DIR}" ]] || {
  echo "Missing Reaction-v4 run directory: ${RUN_DIR}" >&2
  exit 2
}
[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Reaction-v4 continuation requires 8-card DDP" >&2
  exit 2
}
[[ -z "${MAX_UPDATES:-}" ]] || {
  echo "Reaction-v4 formal continuation does not accept MAX_UPDATES" >&2
  exit 2
}

checkpoint_step() {
  "${PYTHON_BIN}" - "$1" "${RUN_NAME}" <<'PY'
import sys
import torch

from train_hy273_unified_actor import CHECKPOINT_FORMAT

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Continuation requires a unified-actor checkpoint")
if checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("Continuation checkpoint belongs to a different run")
print(int(checkpoint.get("next_global_step", -1)))
PY
}

CURRENT_CHECKPOINT="${CHECKPOINT}"
CURRENT_STEP="$(checkpoint_step "${CURRENT_CHECKPOINT}")"
if (( CURRENT_STEP < 150000 || CURRENT_STEP > 250000 )); then
  echo "Expected a Reaction-v4 checkpoint in [150000,250000], got ${CURRENT_STEP}" >&2
  exit 2
fi

if (( CURRENT_STEP < 200000 )); then
  CONFIG="${BASE_CONFIG}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=200000 GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" MASTER_PORT=29816 \
    bash scripts/launch/train_hy273_unified_reaction_v4_layout_stage_b_ddp8.sh
  CURRENT_CHECKPOINT="${RUN_DIR}/model/step_00200000.pt"
  CURRENT_STEP=200000
fi

if (( CURRENT_STEP < 250000 )); then
  CONFIG="${EXTENDED_CONFIG}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=250000 GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" MASTER_PORT=29817 \
    bash scripts/launch/train_hy273_unified_reaction_stage_b_continue250k_ddp8.sh
fi

FINAL_CHECKPOINT="${RUN_DIR}/model/step_00250000.pt"
FINAL_STEP="$(checkpoint_step "${FINAL_CHECKPOINT}")"
if (( FINAL_STEP != 250000 )); then
  echo "Reaction-v4 continuation ended without an exact 250K archive" >&2
  exit 3
fi
echo "Reaction-v4 continuation reached 250K: ${RUN_DIR}/model/step_00250000.pt"
