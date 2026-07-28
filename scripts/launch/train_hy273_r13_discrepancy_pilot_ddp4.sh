#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TREATMENT="${TREATMENT:?Set baseline or source_target_discrepancy_x0}"
case "${TREATMENT}" in
  baseline|source_target_discrepancy_x0) ;;
  *) echo "Unsupported discrepancy pilot treatment: ${TREATMENT}" >&2; exit 2 ;;
esac

CONFIG="${CONFIG:-configs/hy273_multitask_r13_stage_c1_unified_edit.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_multitask}"
PARENT_400K="${PARENT_400K:-${OUTPUT_DIR}/hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/model/step_00400000.pt}"
RUN_NAME="${RUN_NAME:-hy273_r13_edit_${TREATMENT}_pilot_ddp4_$(date +%Y%m%d_%H%M%S)}"
SMOKE_STEPS="${SMOKE_STEPS:-5000}"
GPU_IDS="${GPU_IDS:?Set one four-GPU group, for example 0,1,2,3}"
MASTER_PORT="${MASTER_PORT:?Set a unique port for this four-GPU group}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
SAVE_SMOKE="${SAVE_SMOKE:-1}"
RESEARCH_NO_UPDATE="${RESEARCH_NO_UPDATE:-0}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"

[[ "${GPU_IDS}" =~ ^[0-7],[0-7],[0-7],[0-7]$ ]] || {
  echo "GPU_IDS must list exactly four visible GPU indices" >&2
  exit 2
}
[[ "${SMOKE_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "SMOKE_STEPS must be a positive integer" >&2
  exit 2
}
[[ "${SAVE_SMOKE}" == "0" || "${SAVE_SMOKE}" == "1" ]] || exit 2
[[ "${RESEARCH_NO_UPDATE}" == "0" || "${RESEARCH_NO_UPDATE}" == "1" ]] || exit 2
[[ -f "${PARENT_400K}" ]] || { echo "Missing checkpoint: ${PARENT_400K}" >&2; exit 2; }
if [[ "${RESEARCH_NO_UPDATE}" == "1" && "${SAVE_SMOKE}" != "0" ]]; then
  echo "No-update calibration requires SAVE_SMOKE=0" >&2
  exit 2
fi

ARGS=(
  --config "${CONFIG}"
  --name "${RUN_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --resume "${PARENT_400K}"
  --research_fork
  --research_reshard_same_global_batch
  --research_treatment "${TREATMENT}"
  --smoke_steps "${SMOKE_STEPS}"
  --batch_size_per_rank 32
  --materialize_workers "${MATERIALIZE_WORKERS}"
)
if [[ "${SAVE_SMOKE}" == "0" ]]; then
  ARGS+=(--no-save_smoke)
fi
if [[ "${RESEARCH_NO_UPDATE}" == "1" ]]; then
  ARGS+=(--research_no_update)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "protocol=R13_source_target_discrepancy_x0 treatment=${TREATMENT} run=${RUN_NAME} gpus=${GPU_IDS} steps=${SMOKE_STEPS} no_update=${RESEARCH_NO_UPDATE}"
exec "${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node=4 \
  --master_port="${MASTER_PORT}" \
  train_hy273_multitask.py "${ARGS[@]}"
