#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TREATMENT="${TREATMENT:?Set an Edit research treatment}"
CONFIG="${CONFIG:-configs/hy273_multitask_r13_stage_c1_unified_edit.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_multitask}"
PARENT_400K="${PARENT_400K:-${OUTPUT_DIR}/hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/model/step_00400000.pt}"
RUN_NAME="${RUN_NAME:-hy273_r13_edit_${TREATMENT}_pilot_ddp8_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-${PARENT_400K}}"
FRESH_FORK="${FRESH_FORK:-1}"
SMOKE_STEPS="${SMOKE_STEPS:-10000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-$((128 / NPROC_PER_NODE))}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
EDIT_SAME_SOURCE_GROUPS="${EDIT_SAME_SOURCE_GROUPS:-${ROOT_DIR}/outputs/hy273_multitask/diagnostics/r13_edit_objective_pilot_405k_20260722/tiny_overfit_candidate_groups.json}"
EDIT_SAME_SOURCE_MIN_TARGET_MSE="${EDIT_SAME_SOURCE_MIN_TARGET_MSE:-0.10}"
SAVE_SMOKE="${SAVE_SMOKE:-0}"
RESEARCH_NO_UPDATE="${RESEARCH_NO_UPDATE:-0}"
MASTER_PORT="${MASTER_PORT:-29731}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"

case "${TREATMENT}" in
  baseline|anchored_identity|anchored_identity_low_t|same_source_contrast|same_source_hinge_only|same_source_softplus_only|source_target_discrepancy_x0|no_rank_positive_only|same_source_changed_positive_only|physical_temporal_positive_only|source_token_block_positive_only) ;;
  *) echo "Unknown TREATMENT=${TREATMENT}" >&2; exit 2 ;;
esac
[[ "${NPROC_PER_NODE}" == "4" || "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Research comparison requires DDP4 or DDP8" >&2
  exit 2
}
[[ "${FRESH_FORK}" == "0" || "${FRESH_FORK}" == "1" ]] || {
  echo "FRESH_FORK must be 0 or 1" >&2
  exit 2
}
[[ "${SMOKE_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SMOKE_STEPS must be a positive integer" >&2
  exit 2
}
[[ "${SAVE_SMOKE}" == "0" || "${SAVE_SMOKE}" == "1" ]] || {
  echo "SAVE_SMOKE must be 0 or 1" >&2
  exit 2
}
[[ "${RESEARCH_NO_UPDATE}" == "0" || "${RESEARCH_NO_UPDATE}" == "1" ]] || {
  echo "RESEARCH_NO_UPDATE must be 0 or 1" >&2
  exit 2
}
[[ -f "${RESUME}" ]] || { echo "Missing checkpoint: ${RESUME}" >&2; exit 2; }

ARGS=(
  --config "${CONFIG}"
  --name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --resume "${RESUME}"
  --research_treatment "${TREATMENT}"
  --smoke_steps "${SMOKE_STEPS}"
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}"
  --materialize_workers "${MATERIALIZE_WORKERS}"
  --edit_same_source_groups "${EDIT_SAME_SOURCE_GROUPS}"
  --edit_same_source_min_target_mse "${EDIT_SAME_SOURCE_MIN_TARGET_MSE}"
)
if [[ "${FRESH_FORK}" == "1" ]]; then
  ARGS+=(--research_fork)
fi
if [[ "${NPROC_PER_NODE}" == "4" && "${FRESH_FORK}" == "1" ]]; then
  ARGS+=(--research_reshard_same_global_batch)
fi
if [[ "${RESEARCH_NO_UPDATE}" == "1" ]]; then
  ARGS+=(--research_no_update)
fi
if [[ "${SAVE_SMOKE}" == "0" ]]; then
  ARGS+=(--no-save_smoke)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "protocol=R13_edit_objective_pilot treatment=${TREATMENT} run=${RUN_NAME} resume=${RESUME} fresh=${FRESH_FORK} steps=${SMOKE_STEPS}"
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  train_hy273_multitask.py "${ARGS[@]}"
