#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TEXT_FUSION_MODE="${TEXT_FUSION_MODE:?Set f00, f10, f01, or f11}"
RUN_NAME="${RUN_NAME:?Set a unique Stage-A run name}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_text_fusion}"
CONFIG="configs/hy273_multitask_r13_stage_a_t2m.yaml"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE=4
GLOBAL_BATCH_SIZE=128
BATCH_SIZE_PER_RANK=32
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29710}"
TEXT_GLOBAL_CONDITIONING="pooled_adaln"

case "${TEXT_FUSION_MODE}" in
  f00|f10|f01|f11) ;;
  *)
    echo "Unsupported TEXT_FUSION_MODE=${TEXT_FUSION_MODE}" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
[[ "${#GPU_ARRAY[@]}" == "${NPROC_PER_NODE}" ]] || {
  echo "GPU_IDS count must equal NPROC_PER_NODE" >&2
  exit 2
}
[[ $((NPROC_PER_NODE * BATCH_SIZE_PER_RANK)) == "${GLOBAL_BATCH_SIZE}" ]] || {
  echo "NPROC_PER_NODE * BATCH_SIZE_PER_RANK must equal GLOBAL_BATCH_SIZE" >&2
  exit 2
}
[[ -x "${PYTHON_BIN}" && -x "${TORCHRUN_BIN}" ]] || {
  echo "Missing Python or torchrun executable" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ARGS=(
  --config "${CONFIG}"
  --name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}"
  --materialize_workers "${MATERIALIZE_WORKERS}"
  --text_global_conditioning "${TEXT_GLOBAL_CONDITIONING}"
  --text_fusion_mode "${TEXT_FUSION_MODE}"
)

echo "protocol=HY273_text_fusion_stage_a mode=${TEXT_FUSION_MODE} run=${RUN_NAME} gpu_ids=${GPU_IDS} world_size=${NPROC_PER_NODE} batch_per_rank=${BATCH_SIZE_PER_RANK} global_batch=${GLOBAL_BATCH_SIZE}"
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_hy273_multitask.py "${ARGS[@]}"
