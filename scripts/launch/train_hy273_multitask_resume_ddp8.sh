#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAGE="${STAGE:?Set STAGE to b1, b2, or c}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the existing Stage-A run name}"
RESUME="${RESUME:?Set RESUME to the exact previous-stage checkpoint}"

case "${STAGE}" in
  b1)
    CONFIG="configs/hy273_multitask_stage_b1_control_bootstrap.yaml"
    ;;
  b2)
    CONFIG="configs/hy273_multitask_stage_b2_joint_adapt.yaml"
    ;;
  c)
    CONFIG="configs/hy273_multitask_stage_c_consolidate.yaml"
    ;;
  *)
    echo "Unsupported STAGE=${STAGE}; expected b1, b2, or c" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_multitask}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
  echo "Missing Python or torchrun: ${PYTHON_BIN} ${TORCHRUN_BIN}" >&2
  exit 2
fi
if [[ ! -f "${RESUME}" ]]; then
  echo "Resume checkpoint is missing: ${RESUME}" >&2
  exit 2
fi
if [[ ! -f "${OUTPUT_DIR}/${RUN_NAME}/run_identity.json" ]]; then
  echo "Existing run identity is missing: ${OUTPUT_DIR}/${RUN_NAME}" >&2
  exit 2
fi

RESUME_SHA256="${RESUME_SHA256:-$(sha256sum "${RESUME}" | awk '{print $1}')}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "stage=${STAGE} config=${CONFIG}"
echo "run_name=${RUN_NAME}"
echo "resume=${RESUME}"
echo "resume_sha256=${RESUME_SHA256}"
echo "gpus=${GPU_IDS} global_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_RANK))"

exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_hy273_multitask.py \
  --config "${CONFIG}" \
  --name "${RUN_NAME}" \
  --resume "${RESUME}" \
  --resume_sha256 "${RESUME_SHA256}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --materialize_workers "${MATERIALIZE_WORKERS}"
