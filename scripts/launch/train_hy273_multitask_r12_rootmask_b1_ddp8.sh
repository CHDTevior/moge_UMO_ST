#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
CONFIG="${CONFIG:-configs/hy273_multitask_r12_stage_b1_control_bootstrap.yaml}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_multitask}"
RUN_NAME="${RUN_NAME:-hy273_multitask_r12_rootmask_b1_ddp8_$(date +%Y%m%d_%H%M%S)}"
PARENT_CHECKPOINT="/mnt/afs/mogeflow-control/outputs/hy273_multitask/hy273_multitask_r11_stage_a_t2m_ddp8_20260715_1510/model/step_00200000.pt"
PARENT_SHA256="e06b397df60e9b68e628fa68bede687c97ecb9bb25e556f3d96a311423e1744e"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "R12 B1 requires NPROC_PER_NODE=8" >&2; exit 2; }
[[ "${BATCH_SIZE_PER_RANK}" == "16" ]] || { echo "R12 B1 requires batch/rank=16" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" && -x "${TORCHRUN_BIN}" ]] || { echo "Missing Python/torchrun" >&2; exit 2; }
[[ -f "${PARENT_CHECKPOINT}" ]] || { echo "Missing R11 200K parent: ${PARENT_CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${OUTPUT_DIR}/${RUN_NAME}" ]] || { echo "R12 run already exists: ${OUTPUT_DIR}/${RUN_NAME}" >&2; exit 2; }

actual_sha="$(sha256sum "${PARENT_CHECKPOINT}" | awk '{print $1}')"
[[ "${actual_sha}" == "${PARENT_SHA256}" ]] || { echo "R11 200K parent SHA mismatch" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "protocol=R12_rootmask_v1 stage=B1"
echo "run_name=${RUN_NAME}"
echo "parent=${PARENT_CHECKPOINT}"
echo "parent_sha256=${PARENT_SHA256}"
echo "gpus=${GPU_IDS} global_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_RANK))"

exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_hy273_multitask.py \
  --config "${CONFIG}" \
  --name "${RUN_NAME}" \
  --resume "${PARENT_CHECKPOINT}" \
  --resume_sha256 "${PARENT_SHA256}" \
  --fork_from_r11_stage_a \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --materialize_workers "${MATERIALIZE_WORKERS}"
