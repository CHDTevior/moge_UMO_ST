#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

STAGE="${STAGE:?Set STAGE to b1, b2, or c}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME to the existing R12 run name}"
RESUME="${RESUME:?Set RESUME to the exact previous-stage R12 checkpoint}"

case "${STAGE}" in
  b1) CONFIG="configs/hy273_multitask_r12_stage_b1_control_bootstrap.yaml" ;;
  b2) CONFIG="configs/hy273_multitask_r12_stage_b2_joint_adapt.yaml" ;;
  c) CONFIG="configs/hy273_multitask_r12_stage_c_consolidate.yaml" ;;
  *) echo "Unsupported R12 STAGE=${STAGE}; expected b1, b2, or c" >&2; exit 2 ;;
esac

TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_multitask}"

[[ "${NPROC_PER_NODE}" == "8" && "${BATCH_SIZE_PER_RANK}" == "16" ]] || { echo "R12 requires DDP8 batch/rank=16" >&2; exit 2; }
[[ -x "${TORCHRUN_BIN}" && -x "${PYTHON_BIN}" && -f "${RESUME}" ]] || { echo "Missing Python/torchrun or resume checkpoint" >&2; exit 2; }
[[ -f "${OUTPUT_DIR}/${RUN_NAME}/run_identity.json" ]] || { echo "Missing R12 run identity" >&2; exit 2; }
RESUME_SHA256="${RESUME_SHA256:-$(sha256sum "${RESUME}" | awk '{print $1}')}"

if [[ "${STAGE}" == "b2" ]]; then
  R12_GATE_ARTIFACT="${R12_GATE_ARTIFACT:?Set R12_GATE_ARTIFACT to the passed R12 250K gate}"
  "${PYTHON_BIN}" tools/gate_hy273_r12_step250k.py validate-resume \
    --gate_artifact "${R12_GATE_ARTIFACT}" \
    --checkpoint "${RESUME}" \
    --checkpoint_sha256 "${RESUME_SHA256}"
  "${PYTHON_BIN}" - "${R12_GATE_ARTIFACT}" "${OUTPUT_DIR}/${RUN_NAME}/r12_b2_admission.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
payload = {
    "format": "hy273_r12_b2_admission_v1",
    "gate_artifact": str(source.resolve()),
    "gate_artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
fi

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
  --output_dir "${OUTPUT_DIR}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --materialize_workers "${MATERIALIZE_WORKERS}"
