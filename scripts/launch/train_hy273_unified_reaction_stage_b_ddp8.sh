#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/hy273_unified_fulltext_reaction_v1.yaml}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the completed Reaction Stage-A run}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_unified_reaction}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00100000.pt}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29792}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
STOP_STEP="${STOP_STEP:-200000}"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "Reaction Stage B requires 8-card DDP" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing Stage-A checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ "${STOP_STEP}" == "200000" ]] || { echo "Reaction Stage B stops at 200000" >&2; exit 2; }
[[ -d "${OUTPUT_DIR}/${RUN_NAME}" ]] || { echo "Missing Stage-A run directory" >&2; exit 2; }

NEXT_STEP=$("${PYTHON_BIN}" - "${CHECKPOINT}" "${RUN_NAME}" <<'PY'
import sys
import torch
from train_hy273_unified_actor import CHECKPOINT_FORMAT, validate_config
checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Stage B requires a unified checkpoint")
if checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("Stage B run name differs from its parent checkpoint")
config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError("Checkpoint has no resolved config")
validate_config(config)
if config["data"].get("paired_task") != "reaction":
    raise RuntimeError("Stage B parent is not the single-target Reaction architecture")
if config["model"].get("text_token_sequence") != "sentence_plus_context":
    raise RuntimeError("Stage B parent does not use the full main text stream")
print(int(checkpoint.get("next_global_step", -1)))
PY
)
if (( NEXT_STEP < 100000 || NEXT_STEP >= 200000 )); then
  echo "Stage B resume step must be in [100000,200000), got ${NEXT_STEP}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "stage=B tasks=t2m,edit,reaction run=${RUN_NAME} interval=${NEXT_STEP}:${STOP_STEP}"
EXTRA_ARGS=()
[[ -z "${MAX_UPDATES:-}" ]] || EXTRA_ARGS+=(--max_updates "${MAX_UPDATES}")
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" train_hy273_unified_actor.py \
  --config "${CONFIG}" --name "${RUN_NAME}" --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT}" --stop_step "${STOP_STEP}" \
  --phase_contract fulltext_stage_b "${EXTRA_ARGS[@]}"
