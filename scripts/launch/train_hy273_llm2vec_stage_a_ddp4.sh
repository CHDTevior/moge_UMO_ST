#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TREATMENT="${TREATMENT:?Set TREATMENT to kencoder or kfull}"
case "${TREATMENT}" in
  kencoder)
    CONDITIONING_ARCHITECTURE="llm2vec_flux"
    DEFAULT_BASE_REPRESENTATION_LOSS_SPACE="velocity_mse"
    DEFAULT_BASE_CONTACT_LOSS_SPACE="velocity_mse"
    DEFAULT_G0_LR_OVERRIDE=""
    ;;
  kfull)
    CONDITIONING_ARCHITECTURE="llm2vec_kimodo_prefix"
    DEFAULT_BASE_REPRESENTATION_LOSS_SPACE="clean_x0_smooth_l1"
    DEFAULT_BASE_CONTACT_LOSS_SPACE="clean_x0_smooth_l1"
    DEFAULT_G0_LR_OVERRIDE="2e-5"
    ;;
  *)
    echo "Unsupported TREATMENT=${TREATMENT}; expected kencoder or kfull" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-hy273_${TREATMENT}_stageA_ddp4x32_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_text_fusion}"
LLM2VEC_CACHE_DIR="${LLM2VEC_CACHE_DIR:-/mnt/afs/mogo_base/datasets/HY273_multitask_v1/llm2vec_llama3_8b_profile_v1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-32}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29720}"
RESUME="${RESUME:-}"
BASE_REPRESENTATION_LOSS_SPACE="${BASE_REPRESENTATION_LOSS_SPACE:-${DEFAULT_BASE_REPRESENTATION_LOSS_SPACE}}"
BASE_CONTACT_LOSS_SPACE="${BASE_CONTACT_LOSS_SPACE:-${DEFAULT_BASE_CONTACT_LOSS_SPACE}}"
G0_LR_OVERRIDE="${G0_LR_OVERRIDE:-${DEFAULT_G0_LR_OVERRIDE}}"

[[ "${NPROC_PER_NODE}" == "4" ]] || {
  echo "This controlled treatment requires NPROC_PER_NODE=4" >&2
  exit 2
}
[[ -d "${LLM2VEC_CACHE_DIR}" ]] || {
  echo "Missing LLM2Vec cache: ${LLM2VEC_CACHE_DIR}" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ARGS=(
  --config configs/hy273_multitask_r13_stage_a_t2m.yaml
  --name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}"
  --materialize_workers "${MATERIALIZE_WORKERS}"
  --conditioning_architecture "${CONDITIONING_ARCHITECTURE}"
  --llm2vec_cache_dir "${LLM2VEC_CACHE_DIR}"
  --text_global_conditioning llm2vec_tokens_only
  --text_fusion_mode f00
  --base_representation_loss_space "${BASE_REPRESENTATION_LOSS_SPACE}"
  --base_contact_loss_space "${BASE_CONTACT_LOSS_SPACE}"
)
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  ARGS+=(--smoke_steps "${SMOKE_STEPS}")
fi
if [[ -n "${G0_LR_OVERRIDE}" ]]; then
  ARGS+=(--g0_lr_override "${G0_LR_OVERRIDE}")
fi
if [[ "${SAVE_SMOKE:-1}" == "0" ]]; then
  ARGS+=(--no-save_smoke)
fi

echo "treatment=${TREATMENT} architecture=${CONDITIONING_ARCHITECTURE} run=${RUN_NAME} gpus=${GPU_IDS} representation_loss=${BASE_REPRESENTATION_LOSS_SPACE} contact_loss=${BASE_CONTACT_LOSS_SPACE} g0_lr_override=${G0_LR_OVERRIDE:-none}"
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_hy273_multitask.py "${ARGS[@]}"
