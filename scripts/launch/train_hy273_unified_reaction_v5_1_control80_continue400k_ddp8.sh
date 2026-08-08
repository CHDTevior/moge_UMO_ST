#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/hy273_unified_fulltext_reaction_v5_1_control80_continue400k.yaml}"
RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v5_1_full_contact_20260806_1750}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact}"
STOP_STEP="${STOP_STEP:-400000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29831}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
MODEL_DIR="${OUTPUT_DIR}/${RUN_NAME}/model"

[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Reaction-v5.1 Control80 requires 8-card DDP" >&2
  exit 2
}
[[ "${STOP_STEP}" == "400000" ]] || {
  echo "Reaction-v5.1 Control80 must stop at 400000" >&2
  exit 2
}
[[ -z "${MAX_UPDATES:-}" ]] || {
  echo "The formal 300K-400K launcher rejects MAX_UPDATES; run smoke commands directly" >&2
  exit 2
}

if [[ -z "${CHECKPOINT:-}" ]]; then
  CHECKPOINT=$("${PYTHON_BIN}" - "${MODEL_DIR}" "${RUN_NAME}" <<'PY'
import sys
from pathlib import Path

import torch

from train_hy273_unified_actor import CHECKPOINT_FORMAT

model_dir = Path(sys.argv[1])
run_name = sys.argv[2]
candidates = [model_dir / "latest.pt", *sorted(model_dir.glob("step_*.pt"))]
valid = []
completed = []
seen = set()
for path in candidates:
    if not path.is_file():
        continue
    identity = (path.stat().st_dev, path.stat().st_ino)
    if identity in seen:
        continue
    seen.add(identity)
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    step = int(checkpoint.get("next_global_step", -1))
    if (
        checkpoint.get("format") == CHECKPOINT_FORMAT
        and checkpoint.get("run_name") == run_name
    ):
        if step >= 400_000:
            completed.append((step, path))
        elif step >= 300_000:
            valid.append((step, path))
if completed:
    step, path = max(completed, key=lambda item: item[0])
    raise RuntimeError(
        f"Control80 stage is already complete at step {step}: {path}; refusing to replay"
    )
if not valid:
    raise RuntimeError(f"No valid Control80 resume checkpoint in {model_dir}")
print(max(valid, key=lambda item: item[0])[1])
PY
  )
fi

[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Reaction-v5.1 Control80 resume checkpoint: ${CHECKPOINT}" >&2
  exit 2
}

NEXT_STEP=$("${PYTHON_BIN}" - "${CHECKPOINT}" "${RUN_NAME}" <<'PY'
import sys
import torch

from train_hy273_unified_actor import CHECKPOINT_FORMAT

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Reaction-v5.1 Control80 requires a unified-actor checkpoint")
if checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("Reaction-v5.1 Control80 checkpoint belongs to another run")
config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError("Reaction-v5.1 Control80 checkpoint has no resolved config")
if config.get("data", {}).get("paired_task") != "reaction":
    raise RuntimeError("Reaction-v5.1 Control80 requires the Reaction architecture")
if config.get("model", {}).get("text_token_sequence") != "sentence_plus_context":
    raise RuntimeError("Reaction-v5.1 Control80 requires the full contextual text stream")
if any("ease" in name.lower() for name in checkpoint.get("model", {})):
    raise RuntimeError("Reaction-v5.1 Control80 parent unexpectedly contains Ease parameters")
step = int(checkpoint.get("next_global_step", -1))
if not 300_000 <= step < 400_000:
    raise RuntimeError(f"Control80 resume step must be in [300000,400000), got {step}")
print(step)
PY
)

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "stage=reaction-v5.1-control80 tasks=30,35,35 control=80% no_control=20% ease=off interval=${NEXT_STEP}:${STOP_STEP}"
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" train_hy273_unified_actor.py \
  --config "${CONFIG}" --name "${RUN_NAME}" --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT}" --stop_step "${STOP_STEP}" \
  --phase_contract fulltext_reaction_v5_1_control
