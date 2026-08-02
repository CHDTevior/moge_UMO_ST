#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARENT_RUN="${PARENT_RUN:-hy273_kencoder_stageBE_t2m60_edit40_ddp8x16_20260728_131901}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_text_fusion}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${PARENT_RUN}/model/step_00250000.pt}"
RUN_NAME="${RUN_NAME:-hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_$(date +%Y%m%d_%H%M%S)}"
LLM2VEC_CACHE_DIR="${LLM2VEC_CACHE_DIR:-/mnt/afs/mogo_base/datasets/HY273_multitask_v1/llm2vec_llama3_8b_profile_v1}"
EASE_STATS_DIR="${EASE_STATS_DIR:-/mnt/afs/mogo_base/datasets/HY273_multitask_v1/derived_stats/hy273_ease_stats_v1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29772}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "K-Encoder Stage-BC uses eight-card DDP" >&2
  exit 2
}
[[ "${BATCH_SIZE_PER_RANK}" == "16" ]] || {
  echo "Stage-BC preserves global batch 128 with 8x16" >&2
  exit 2
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Stage-BC checkpoint: ${CHECKPOINT}" >&2
  exit 2
}
[[ -d "${LLM2VEC_CACHE_DIR}" ]] || {
  echo "Missing LLM2Vec cache: ${LLM2VEC_CACHE_DIR}" >&2
  exit 2
}
[[ -f "${EASE_STATS_DIR}/metadata.json" ]] || {
  echo "Missing Ease stats: ${EASE_STATS_DIR}" >&2
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
print(str(checkpoint.get("high_level_schedule_version", "")))
PY
)
CHECKPOINT_STEP="${CHECKPOINT_META[0]:--1}"
CHECKPOINT_WORLD_SIZE="${CHECKPOINT_META[1]:--1}"
CHECKPOINT_BATCH_SIZE="${CHECKPOINT_META[2]:--1}"
CHECKPOINT_SCHEDULE="${CHECKPOINT_META[3]:-}"
MODE=""

if [[ "${CHECKPOINT_STEP}" == "250000" ]]; then
  [[ "${CHECKPOINT_SCHEDULE}" == "hy273_kencoder_stage_be_fixed_60_0_40_v1" ]] || {
    echo "The 250K parent must be the completed Stage-BE run" >&2
    exit 2
  }
  MODE="fork_250k"
elif (( CHECKPOINT_STEP > 250000 && CHECKPOINT_STEP < 400000 )); then
  [[ "${CHECKPOINT_SCHEDULE}" == "hy273_kencoder_stage_bc_fixed_10_70_20_ease_v1" ]] || {
    echo "Stage-BC continuation has the wrong schedule" >&2
    exit 2
  }
  MODE="resume"
else
  echo "Stage-BC checkpoint must be 250K or inside (250K,400K), got ${CHECKPOINT_STEP}" >&2
  exit 2
fi
[[ "${CHECKPOINT_WORLD_SIZE}" == "8" && "${CHECKPOINT_BATCH_SIZE}" == "16" ]] || {
  echo "Stage-BC requires an 8x16 checkpoint, got ${CHECKPOINT_WORLD_SIZE}x${CHECKPOINT_BATCH_SIZE}" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ARGS=(
  --config configs/hy273_multitask_kencoder_stage_bc_ease_control.yaml
  --name "${RUN_NAME}"
  --resume "${CHECKPOINT}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}"
  --materialize_workers "${MATERIALIZE_WORKERS}"
  --conditioning_architecture llm2vec_flux
  --llm2vec_cache_dir "${LLM2VEC_CACHE_DIR}"
  --ease_stats_dir "${EASE_STATS_DIR}"
  --text_global_conditioning llm2vec_tokens_only
  --text_fusion_mode f00
  --base_representation_loss_space velocity_mse
  --base_contact_loss_space velocity_mse
)
if [[ "${MODE}" == "fork_250k" ]]; then
  ARGS+=(--fork_kencoder_stage_bc_ease_control)
fi
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  ARGS+=(--smoke_steps "${SMOKE_STEPS}")
fi
if [[ "${SAVE_SMOKE:-1}" == "0" ]]; then
  ARGS+=(--no-save_smoke)
fi

echo "protocol=KEncoder_StageBC_Ease_Control mode=${MODE} step=${CHECKPOINT_STEP} run=${RUN_NAME} resume=${CHECKPOINT} gpus=${GPU_IDS} global_batch=128 mix=10/70/20 ease=t2m25%/control50%/edit0%"
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  exit 0
fi
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_hy273_multitask.py "${ARGS[@]}"
