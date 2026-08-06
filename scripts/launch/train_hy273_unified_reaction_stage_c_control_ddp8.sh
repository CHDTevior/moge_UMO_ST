#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/hy273_unified_fulltext_reaction_v1_stage_c_control.yaml}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the completed 350K Unified Reaction run}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29795}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
STOP_STEP="${STOP_STEP:-400000}"

[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Unified Reaction Control Stage C requires 8-card DDP" >&2
  exit 2
}
[[ "${STOP_STEP}" == "400000" || "${STOP_STEP}" == "450000" || "${STOP_STEP}" == "500000" ]] || {
  echo "STOP_STEP must be 400000, 450000, or 500000" >&2
  exit 2
}

START_BOUNDARY=$((STOP_STEP - 50000))
printf -v DEFAULT_CHECKPOINT "%s/%s/model/step_%08d.pt" \
  "${OUTPUT_DIR}" "${RUN_NAME}" "${START_BOUNDARY}"
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Stage C resume checkpoint: ${CHECKPOINT}" >&2
  exit 2
}

NEXT_STEP=$("${PYTHON_BIN}" - "${CHECKPOINT}" "${RUN_NAME}" <<'PY'
import sys
import torch

from train_hy273_unified_actor import CHECKPOINT_FORMAT

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Stage C requires a unified-actor checkpoint")
if checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("Stage C checkpoint belongs to a different run")
config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError("Stage C checkpoint has no resolved config")
if config.get("data", {}).get("paired_task") != "reaction":
    raise RuntimeError("Stage C requires the single-target Reaction architecture")
if config.get("model", {}).get("text_token_sequence") != "sentence_plus_context":
    raise RuntimeError("Stage C requires the full contextual text stream")
if any("ease" in name.lower() for name in checkpoint.get("model", {})):
    raise RuntimeError("Stage C parent unexpectedly contains Ease parameters")
print(int(checkpoint.get("next_global_step", -1)))
PY
)
if (( NEXT_STEP < START_BOUNDARY || NEXT_STEP >= STOP_STEP )); then
  echo "Stage C resume step must be in [${START_BOUNDARY},${STOP_STEP}), got ${NEXT_STEP}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "stage=C-control tasks=30,35,35 control=90% no_control=10% ease=off interval=${NEXT_STEP}:${STOP_STEP}"
EXTRA_ARGS=()
[[ -z "${MAX_UPDATES:-}" ]] || EXTRA_ARGS+=(--max_updates "${MAX_UPDATES}")
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" train_hy273_unified_actor.py \
  --config "${CONFIG}" --name "${RUN_NAME}" --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT}" --stop_step "${STOP_STEP}" \
  --phase_contract fulltext_stage_c_control "${EXTRA_ARGS[@]}"
