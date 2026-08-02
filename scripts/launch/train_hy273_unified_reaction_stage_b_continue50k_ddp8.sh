#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/hy273_unified_fulltext_reaction_v1_continue350k.yaml}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the existing Unified Reaction run}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_unified_reaction}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29794}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
STOP_STEP="${STOP_STEP:-300000}"

[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Unified Reaction continuation requires 8-card DDP" >&2
  exit 2
}
[[ "${STOP_STEP}" == "300000" || "${STOP_STEP}" == "350000" ]] || {
  echo "STOP_STEP must be 300000 or 350000" >&2
  exit 2
}

START_BOUNDARY=$((STOP_STEP - 50000))
printf -v DEFAULT_CHECKPOINT "${OUTPUT_DIR}/${RUN_NAME}/model/step_%08d.pt" "${START_BOUNDARY}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Unified Reaction continuation checkpoint: ${CHECKPOINT}" >&2
  exit 2
}
[[ -d "${OUTPUT_DIR}/${RUN_NAME}" ]] || {
  echo "Missing existing Unified Reaction run directory" >&2
  exit 2
}

NEXT_STEP=$("${PYTHON_BIN}" - "${CHECKPOINT}" "${RUN_NAME}" <<'PY'
import sys
import torch

from train_hy273_unified_actor import CHECKPOINT_FORMAT

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Continuation requires a unified-actor checkpoint")
if checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("Continuation checkpoint belongs to a different run")
config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError("Continuation checkpoint has no resolved config")
if config.get("data", {}).get("paired_task") != "reaction":
    raise RuntimeError("Continuation checkpoint is not the single-target Reaction model")
if config.get("model", {}).get("text_token_sequence") != "sentence_plus_context":
    raise RuntimeError("Continuation checkpoint does not use the full text stream")
print(int(checkpoint.get("next_global_step", -1)))
PY
)
if (( NEXT_STEP < START_BOUNDARY || NEXT_STEP >= STOP_STEP )); then
  echo "Continuation resume step must be in [${START_BOUNDARY},${STOP_STEP}), got ${NEXT_STEP}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "stage=B-continue tasks=t2m,edit,reaction mix=30,35,35 run=${RUN_NAME} interval=${NEXT_STEP}:${STOP_STEP}"
EXTRA_ARGS=()
[[ -z "${MAX_UPDATES:-}" ]] || EXTRA_ARGS+=(--max_updates "${MAX_UPDATES}")
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" train_hy273_unified_actor.py \
  --config "${CONFIG}" --name "${RUN_NAME}" --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT}" --stop_step "${STOP_STEP}" \
  --phase_contract fulltext_stage_b_continue "${EXTRA_ARGS[@]}"
