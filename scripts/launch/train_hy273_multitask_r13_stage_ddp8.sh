#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAGE="${STAGE:?Set STAGE to a, b1, b2, c1, or c2}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME for the shared R13 run}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_multitask}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29673}"
TEXT_GLOBAL_CONDITIONING="${TEXT_GLOBAL_CONDITIONING:-pooled_adaln}"
TEXT_FUSION_MODE="${TEXT_FUSION_MODE:-f00}"

case "${STAGE}" in
  a)
    CONFIG="configs/hy273_multitask_r13_stage_a_t2m.yaml"
    RESUME=""
    ;;
  b1)
    CONFIG="configs/hy273_multitask_r13_stage_b1_control_bootstrap.yaml"
    RESUME="${RESUME:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00200000.pt}"
    ;;
  b2)
    CONFIG="configs/hy273_multitask_r13_stage_b2_joint_adapt.yaml"
    RESUME="${RESUME:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00250000.pt}"
    ;;
  c1)
    CONFIG="configs/hy273_multitask_r13_stage_c1_unified_edit.yaml"
    RESUME="${RESUME:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00400000.pt}"
    ;;
  c2)
    CONFIG="configs/hy273_multitask_r13_stage_c2_edit40.yaml"
    RESUME="${RESUME:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00450000.pt}"
    ;;
  *)
    echo "Unsupported STAGE=${STAGE}; expected a, b1, b2, c1, or c2" >&2
    exit 2
    ;;
esac

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "R13 formal training uses DDP8" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" && -x "${TORCHRUN_BIN}" ]] || { echo "Missing Python/torchrun" >&2; exit 2; }
if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
  echo "Missing stage checkpoint: ${RESUME}" >&2
  exit 2
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
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}"
  --materialize_workers "${MATERIALIZE_WORKERS}"
  --text_global_conditioning "${TEXT_GLOBAL_CONDITIONING}"
  --text_fusion_mode "${TEXT_FUSION_MODE}"
)
if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  ARGS+=(--smoke_steps "${SMOKE_STEPS}")
fi
if [[ "${SAVE_SMOKE:-1}" == "0" ]]; then
  ARGS+=(--no-save_smoke)
fi

echo "protocol=R13_unified273 stage=${STAGE} run=${RUN_NAME} config=${CONFIG} resume=${RESUME:-none} text_global_conditioning=${TEXT_GLOBAL_CONDITIONING} text_fusion_mode=${TEXT_FUSION_MODE}"
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_hy273_multitask.py "${ARGS[@]}"
