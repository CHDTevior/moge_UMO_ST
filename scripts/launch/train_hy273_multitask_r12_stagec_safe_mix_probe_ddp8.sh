#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set a new RUN_NAME for the Stage-C safety fork}"
RESUME="${RESUME:?Set RESUME to the exact R12 400K checkpoint}"
CONFIG="${CONFIG:-configs/hy273_multitask_r12_stage_c_safe_mix_probe.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_multitask}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"

[[ "${NPROC_PER_NODE}" == "8" && "${BATCH_SIZE_PER_RANK}" == "16" ]] || {
  echo "Stage-C safety probe requires DDP8 with batch/rank=16" >&2
  exit 2
}
[[ -x "${TORCHRUN_BIN}" && -f "${RESUME}" && -f "${CONFIG}" ]] || {
  echo "Missing torchrun, resume checkpoint, or config" >&2
  exit 2
}

RESUME_SHA256="${RESUME_SHA256:-$(sha256sum "${RESUME}" | awk '{print $1}')}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_hy273_multitask.py \
  --config "${CONFIG}" \
  --name "${RUN_NAME}" \
  --resume "${RESUME}" \
  --resume_sha256 "${RESUME_SHA256}" \
  --fork_stage_c_schedule \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --materialize_workers "${MATERIALIZE_WORKERS}"
