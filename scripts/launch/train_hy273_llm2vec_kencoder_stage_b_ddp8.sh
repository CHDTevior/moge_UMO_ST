#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the completed K-Encoder Stage-A run}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_text_fusion}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00200000.pt}"
LLM2VEC_CACHE_DIR="${LLM2VEC_CACHE_DIR:-/mnt/afs/mogo_base/datasets/HY273_multitask_v1/llm2vec_llama3_8b_profile_v1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29742}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "K-Encoder Stage-B uses eight-card DDP" >&2
  exit 2
}
[[ "${BATCH_SIZE_PER_RANK}" == "16" ]] || {
  echo "Stage-B must preserve the Stage-A global batch with 8x16 samples" >&2
  exit 2
}
[[ "${PREFLIGHT_ONLY}" == "0" || "${PREFLIGHT_ONLY}" == "1" ]] || {
  echo "PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 2
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing exact 200K Stage-A checkpoint: ${CHECKPOINT}" >&2
  exit 2
}
[[ -d "${LLM2VEC_CACHE_DIR}" ]] || {
  echo "Missing LLM2Vec cache: ${LLM2VEC_CACHE_DIR}" >&2
  exit 2
}

mapfile -t CHECKPOINT_META < <(
  "${PYTHON_BIN}" - "${CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1],
    map_location="cpu",
    mmap=True,
    weights_only=False,
)
batcher = checkpoint.get("batcher") or {}
print(int(checkpoint.get("next_global_step", -1)))
print(int(batcher.get("world_size", -1)))
print(int(batcher.get("batch_size_per_rank", -1)))
PY
)
CHECKPOINT_STEP="${CHECKPOINT_META[0]:--1}"
CHECKPOINT_WORLD_SIZE="${CHECKPOINT_META[1]:--1}"
CHECKPOINT_BATCH_SIZE="${CHECKPOINT_META[2]:--1}"
RESHARD_MODE=""
if [[ "${CHECKPOINT_STEP}" == "200000" ]]; then
  [[ "${CHECKPOINT_WORLD_SIZE}" == "4" && "${CHECKPOINT_BATCH_SIZE}" == "32" ]] || {
    echo "The formal 200K Stage-B parent must use 4x32, got ${CHECKPOINT_WORLD_SIZE}x${CHECKPOINT_BATCH_SIZE}" >&2
    exit 2
  }
  RESHARD_MODE="formal_4x32_to_8x16"
elif (( CHECKPOINT_STEP > 200000 && CHECKPOINT_STEP < 400000 )); then
  [[ "${CHECKPOINT_WORLD_SIZE}" == "8" && "${CHECKPOINT_BATCH_SIZE}" == "16" ]] || {
    echo "A Stage-B continuation must already use 8x16, got ${CHECKPOINT_WORLD_SIZE}x${CHECKPOINT_BATCH_SIZE}" >&2
    exit 2
  }
  RESHARD_MODE="ordinary_8x16_resume"
else
  echo "Stage-B checkpoint step must be 200K or inside (200K,400K), got ${CHECKPOINT_STEP}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ARGS=(
  --config configs/hy273_multitask_r16_stage_b_fixed_control.yaml
  --name "${RUN_NAME}"
  --resume "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}"
  --materialize_workers "${MATERIALIZE_WORKERS}"
  --conditioning_architecture llm2vec_flux
  --llm2vec_cache_dir "${LLM2VEC_CACHE_DIR}"
  --text_global_conditioning llm2vec_tokens_only
  --text_fusion_mode f00
  --base_representation_loss_space velocity_mse
  --base_contact_loss_space velocity_mse
)
if [[ "${RESHARD_MODE}" == "formal_4x32_to_8x16" ]]; then
  ARGS+=(--research_reshard_same_global_batch)
fi
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  ARGS+=(--smoke_steps "${SMOKE_STEPS}")
fi
if [[ "${SAVE_SMOKE:-1}" == "0" ]]; then
  ARGS+=(--no-save_smoke)
fi

echo "protocol=KEncoder_R16_fixed_control mode=${RESHARD_MODE} step=${CHECKPOINT_STEP} run=${RUN_NAME} resume=${CHECKPOINT} gpus=${GPU_IDS} global_batch=128"
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  exit 0
fi
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_hy273_multitask.py "${ARGS[@]}"
