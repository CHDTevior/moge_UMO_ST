#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to an immutable step_*.pt archive}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the evidence-v2 evaluation}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PROFILE="${PROFILE:-research}"
WEIGHT_SOURCE="${WEIGHT_SOURCE:-model}"
NUM_STEPS="${NUM_STEPS:-32}"
CFG_SCALE="${CFG_SCALE:-2.0}"
CONTROL_CFG_SCALE="${CONTROL_CFG_SCALE:-2.0}"
SEED="${SEED:-3407}"
CASES_PER_SUBTYPE="${CASES_PER_SUBTYPE:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint is missing: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ "${PROFILE}" == "production" && "$(basename "${CHECKPOINT}")" != step_*.pt ]]; then
  echo "Production evaluation requires a step_*.pt archive: ${CHECKPOINT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -ne 8 ]]; then
  echo "The full scientific benchmark requires exactly 8 visible GPUs" >&2
  exit 2
fi

CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-$(sha256sum "${CHECKPOINT}" | awk '{print $1}')}"
PREFLIGHT="${OUTPUT_DIR}/preflight_manifest.json"
mkdir -p "${OUTPUT_DIR}/logs"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${PREFLIGHT}" ]]; then
  "${PYTHON_BIN}" tools/eval_hy273_kimodo_v5_contact.py \
    --profile "${PROFILE}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint_sha256 "${CHECKPOINT_SHA256}" \
    --output_dir "${OUTPUT_DIR}" \
    --weight_source "${WEIGHT_SOURCE}" \
    --num_steps "${NUM_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --control_cfg_scale "${CONTROL_CFG_SCALE}" \
    --seed "${SEED}" \
    --cases_per_subtype "${CASES_PER_SUBTYPE}" \
    --num_shards 8 \
    --preflight_only
fi
PREFLIGHT_SHA256="$(sha256sum "${PREFLIGHT}" | awk '{print $1}')"

echo "checkpoint=${CHECKPOINT}"
echo "checkpoint_sha256=${CHECKPOINT_SHA256}"
echo "preflight=${PREFLIGHT}"
echo "preflight_sha256=${PREFLIGHT_SHA256}"
echo "output_dir=${OUTPUT_DIR}"

pids=()
for shard_id in 0 1 2 3 4 5 6 7; do
  "${PYTHON_BIN}" tools/eval_hy273_kimodo_v5_contact.py \
    --profile "${PROFILE}" \
    --checkpoint "${CHECKPOINT}" \
    --checkpoint_sha256 "${CHECKPOINT_SHA256}" \
    --output_dir "${OUTPUT_DIR}" \
    --preflight_manifest "${PREFLIGHT}" \
    --preflight_sha256 "${PREFLIGHT_SHA256}" \
    --device "cuda:${shard_id}" \
    --weight_source "${WEIGHT_SOURCE}" \
    --num_steps "${NUM_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --control_cfg_scale "${CONTROL_CFG_SCALE}" \
    --seed "${SEED}" \
    --cases_per_subtype "${CASES_PER_SUBTYPE}" \
    --shard_id "${shard_id}" \
    --num_shards 8 \
    >"${OUTPUT_DIR}/logs/shard_${shard_id}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for shard_id in 0 1 2 3 4 5 6 7; do
  if ! wait "${pids[${shard_id}]}"; then
    echo "Shard ${shard_id} failed; inspect ${OUTPUT_DIR}/logs/shard_${shard_id}.log" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" tools/eval_hy273_kimodo_v5_contact.py \
  --profile "${PROFILE}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_sha256 "${CHECKPOINT_SHA256}" \
  --output_dir "${OUTPUT_DIR}" \
  --preflight_manifest "${PREFLIGHT}" \
  --preflight_sha256 "${PREFLIGHT_SHA256}" \
  --weight_source "${WEIGHT_SOURCE}" \
  --num_steps "${NUM_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --control_cfg_scale "${CONTROL_CFG_SCALE}" \
  --seed "${SEED}" \
  --cases_per_subtype "${CASES_PER_SUBTYPE}" \
  --num_shards 8 \
  --aggregate

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path(${OUTPUT_DIR@Q})
summary = json.loads((root / "summary.json").read_text())
index = json.loads((root / "artifact_index.json").read_text())
if summary.get("status") != "validated" or index.get("status") != "validated":
    raise SystemExit("Control evidence did not validate")
print(json.dumps({
    "status": "complete",
    "protocol": summary["protocol"]["protocol_version"],
    "case_count": summary["case_count"],
    "artifact_index": str(root / "artifact_index.json"),
}, sort_keys=True))
PY
