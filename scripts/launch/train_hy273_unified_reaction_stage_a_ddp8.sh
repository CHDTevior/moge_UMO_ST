#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/hy273_unified_fulltext_reaction_v1.yaml}"
RUN_NAME="${RUN_NAME:-hy273_unified_fulltext_reaction_v1_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_unified_reaction}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29791}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
STOP_STEP="${STOP_STEP:-100000}"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "Reaction Stage A requires 8-card DDP" >&2; exit 2; }
[[ "${STOP_STEP}" == "100000" ]] || { echo "Reaction Stage A stops at 100000" >&2; exit 2; }
if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || { echo "Missing Stage-A resume checkpoint: ${RESUME}" >&2; exit 2; }
  [[ -d "${OUTPUT_DIR}/${RUN_NAME}" ]] || { echo "Missing Stage-A run: ${OUTPUT_DIR}/${RUN_NAME}" >&2; exit 2; }
else
  [[ ! -e "${OUTPUT_DIR}/${RUN_NAME}" ]] || { echo "Run already exists: ${OUTPUT_DIR}/${RUN_NAME}" >&2; exit 2; }
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ARGS=(
  --config "${CONFIG}"
  --name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --stop_step "${STOP_STEP}"
  --phase_contract fulltext_stage_a
)
[[ -z "${RESUME:-}" ]] || ARGS+=(--resume "${RESUME}")
[[ -z "${MAX_UPDATES:-}" ]] || ARGS+=(--max_updates "${MAX_UPDATES}")
[[ -z "${MATERIALIZE_WORKERS:-}" ]] || ARGS+=(--materialize_workers "${MATERIALIZE_WORKERS}")

echo "stage=A task=t2m scratch=$([[ -z "${RESUME:-}" ]] && echo yes || echo no) run=${RUN_NAME} stop=${STOP_STEP} gpus=${GPU_IDS}"
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" train_hy273_unified_actor.py "${ARGS[@]}"
