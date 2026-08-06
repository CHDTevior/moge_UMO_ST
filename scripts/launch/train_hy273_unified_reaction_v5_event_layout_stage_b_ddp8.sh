#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/hy273_unified_fulltext_reaction_v5_event_layout.yaml}"
PARENT_OUTPUT_DIR="${PARENT_OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction}"
PARENT_RUN_NAME="${PARENT_RUN_NAME:-hy273_unified_fulltext_reaction_v1_20260801_0315}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-${PARENT_OUTPUT_DIR}/${PARENT_RUN_NAME}/model/step_00100000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_event_layout}"
RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v5_event_layout_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-${PARENT_CHECKPOINT}}"
STOP_STEP="${STOP_STEP:-200000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29815}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Reaction-v5 event-layout Stage B requires 8-card DDP" >&2
  exit 2
}
[[ "${STOP_STEP}" == "200000" ]] || {
  echo "Reaction-v5 event-layout is pre-registered to stop at 200000" >&2
  exit 2
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Reaction-v5 parent checkpoint: ${CHECKPOINT}" >&2
  exit 2
}

NEXT_STEP=$("${PYTHON_BIN}" - "${CHECKPOINT}" "${RUN_NAME}" <<'PY'
import sys
import torch

from train_hy273_unified_actor import CHECKPOINT_FORMAT

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Reaction-v5 requires a unified-actor checkpoint")
step = int(checkpoint.get("next_global_step", -1))
if step < 100_000:
    raise RuntimeError(f"Reaction-v5 requires step >=100K, got {step}")
if step > 100_000 and checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("A Reaction-v5 continuation checkpoint belongs to another run")
print(step)
PY
)
if (( NEXT_STEP >= STOP_STEP )); then
  echo "Checkpoint step ${NEXT_STEP} has already reached STOP_STEP=${STOP_STEP}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "stage=reaction-v5-event-layout tasks=t2m,edit,reaction mix=30,35,35 low_t=30%@0:0.15 run=${RUN_NAME} interval=${NEXT_STEP}:${STOP_STEP}"
EXTRA_ARGS=()
[[ -z "${MAX_UPDATES:-}" ]] || EXTRA_ARGS+=(--max_updates "${MAX_UPDATES}")
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" train_hy273_unified_actor.py \
  --config "${CONFIG}" --name "${RUN_NAME}" --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT}" --stop_step "${STOP_STEP}" \
  --phase_contract fulltext_reaction_v2_stage_b "${EXTRA_ARGS[@]}"
