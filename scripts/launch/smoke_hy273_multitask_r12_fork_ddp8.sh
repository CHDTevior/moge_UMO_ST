#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARENT_CHECKPOINT="${ROOT_DIR}/outputs/hy273_multitask/hy273_multitask_r11_stage_a_t2m_ddp8_20260715_1510/model/step_00200000.pt"
PARENT_SHA256="e06b397df60e9b68e628fa68bede687c97ecb9bb25e556f3d96a311423e1744e"
PARENT_RUN_UUID="8805e8ff-6c53-4d0e-9d68-562e471babe8"
SCRATCH_TAG="${SCRATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-hy273_multitask_r12_fork_smoke_${SCRATCH_TAG}}"
SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-${ROOT_DIR}/outputs/hy273_multitask_smoke/r12_fork_ddp8_${SCRATCH_TAG}}"
SCRATCH_RUN_DIR="${SCRATCH_OUTPUT_ROOT}/${RUN_NAME}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
MAX_PRELAUNCH_USED_MIB="${MAX_PRELAUNCH_USED_MIB:-1024}"
MAX_ALLOCATED_GIB="${MAX_ALLOCATED_GIB:-70.0}"
LOG_PATH="${SCRATCH_OUTPUT_ROOT}/torchrun.log"
POSTCHECK_JSON="${SCRATCH_OUTPUT_ROOT}/postcheck.json"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "R12 fork smoke requires DDP8" >&2; exit 2; }
[[ "${BATCH_SIZE_PER_RANK}" == "16" ]] || { echo "R12 fork smoke requires batch/rank=16" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" && -x "${TORCHRUN_BIN}" ]] || { echo "Missing Python/torchrun" >&2; exit 2; }
[[ -f "${PARENT_CHECKPOINT}" ]] || { echo "Missing immutable R11 200K parent" >&2; exit 2; }
[[ ! -e "${SCRATCH_OUTPUT_ROOT}" ]] || { echo "Scratch output already exists: ${SCRATCH_OUTPUT_ROOT}" >&2; exit 2; }

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
[[ "${#GPU_ARRAY[@]}" == "8" ]] || { echo "GPU_IDS must contain exactly eight GPUs" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: ${gpu@Q}" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || { echo "Duplicate GPU id: ${gpu}" >&2; exit 2; }
  SEEN_GPUS["${gpu}"]=1
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  (( used <= MAX_PRELAUNCH_USED_MIB )) || {
    echo "GPU ${gpu} is not idle: ${used} MiB" >&2
    exit 2
  }
  compute_processes="$(
    nvidia-smi --id="${gpu}" --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )"
  [[ -z "${compute_processes}" ]] || {
    echo "GPU ${gpu} has active compute processes: ${compute_processes}" >&2
    exit 2
  }
done

PARENT_SHA_BEFORE="$(sha256sum "${PARENT_CHECKPOINT}" | awk '{print $1}')"
[[ "${PARENT_SHA_BEFORE}" == "${PARENT_SHA256}" ]] || { echo "R11 parent SHA mismatch" >&2; exit 2; }
mkdir -p "${SCRATCH_OUTPUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

set +e
"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_hy273_multitask.py \
  --config configs/hy273_multitask_r12_stage_b1_control_bootstrap.yaml \
  --name "${RUN_NAME}" \
  --resume "${PARENT_CHECKPOINT}" \
  --resume_sha256 "${PARENT_SHA256}" \
  --fork_from_r11_stage_a \
  --output_dir "${SCRATCH_OUTPUT_ROOT}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --materialize_workers "${MATERIALIZE_WORKERS}" \
  --smoke_steps 1 \
  --no-save_smoke 2>&1 | tee "${LOG_PATH}"
TORCHRUN_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${TORCHRUN_STATUS}" == "0" ]] || exit "${TORCHRUN_STATUS}"

PARENT_SHA_AFTER="$(sha256sum "${PARENT_CHECKPOINT}" | awk '{print $1}')"
"${PYTHON_BIN}" - \
  "${SCRATCH_RUN_DIR}" "${LOG_PATH}" "${PARENT_CHECKPOINT}" \
  "${PARENT_SHA_BEFORE}" "${PARENT_SHA_AFTER}" "${PARENT_RUN_UUID}" \
  "${MAX_ALLOCATED_GIB}" "${POSTCHECK_JSON}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

run_dir, log_path, parent_path, sha_before, sha_after, parent_uuid, max_memory, output = sys.argv[1:]
run_dir = Path(run_dir)
log_path = Path(log_path)
parent_path = str(Path(parent_path).resolve())
output = Path(output)
max_memory = float(max_memory)

def require(value, message):
    if not value:
        raise RuntimeError(message)

def require_zero(metrics, key):
    require(abs(float(metrics[key])) <= 1e-12, f"{key} is not exact zero: {metrics[key]}")

records = [
    json.loads(line)
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
require(len(records) == 1 and records[0]["step"] == 200_001, "one-step metric sequence mismatch")
metrics = records[0]["metrics"]
for key, value in metrics.items():
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(float(value)), f"non-finite metric {key}")

for key in (
    "batch/source_present",
    "batch/stream_edit",
    "grad/context_preclip",
    "update/context/sampled_norm",
    "train/context_update_count",
):
    require_zero(metrics, key)
require(metrics["grad/base_preclip"] > 0.0, "base gradient is zero")
require(metrics["update/base/sampled_norm"] > 0.0, "base update is zero")
require(metrics["batch/root_position_only_per_sample"] > 0.0, "no XZ-only root mask realized")
require(metrics["batch/root_position_rotation_per_sample"] > 0.0, "no XZ+heading root mask realized")
require(metrics["train/next_global_step"] == 200_001.0, "next step mismatch")
require(metrics["train/ema_update_count"] == 20_001.0, "EMA count mismatch")
require(metrics["memory/max_allocated_gib"] <= max_memory, "GPU memory ceiling exceeded")

identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
runtime = json.loads((run_dir / "runtime_identity.json").read_text(encoding="utf-8"))
require(identity["run_uuid"] != parent_uuid, "fork reused the R11 UUID")
require(identity["origin_parent"] == runtime["origin_parent"], "origin lineage mismatch")
origin = identity["origin_parent"]
require(origin["checkpoint"] == parent_path, "origin path mismatch")
require(origin["checkpoint_sha256"] == sha_before == sha_after, "parent SHA changed")
require(runtime["immediate_resume_parent"]["checkpoint_sha256"] == sha_before, "direct parent SHA mismatch")
require(runtime["world_size"] == 8 and runtime["batch_size_per_rank"] == 16, "DDP topology mismatch")
require(runtime["production"] is False, "fork smoke must be non-production")

log = log_path.read_text(encoding="utf-8")
starts = [json.loads(line) for line in log.splitlines() if line.startswith('{') and '"event": "training_start"' in line]
ends = [json.loads(line) for line in log.splitlines() if line.startswith('{') and '"event": "stage_complete"' in line]
require(len(starts) == 1 and starts[0]["start_step"] == 200_000 and starts[0]["stop_step"] == 200_001, "training_start mismatch")
require(len(ends) == 1 and ends[0]["next_global_step"] == 200_001, "stage_complete mismatch")
require(not list((run_dir / "model").glob("*.pt")) if (run_dir / "model").exists() else True, "smoke checkpoint must not exist")
nccl_warnings = [line for line in log.splitlines() if re.search(r"NCCL.*(?:WARN|ERROR)", line, re.I)]
require(not nccl_warnings, f"NCCL warning/error detected: {nccl_warnings[:3]}")

payload = {
    "format": "hy273_multitask_r12_fork_smoke_postcheck_v1",
    "status": "passed",
    "start_step": 200_000,
    "final_step": 200_001,
    "run_name": identity["run_name"],
    "run_uuid": identity["run_uuid"],
    "origin_parent": origin,
    "parent_sha256_before_after": sha_before,
    "root_position_only_fraction": metrics["batch/root_position_only_per_sample"],
    "root_position_rotation_fraction": metrics["batch/root_position_rotation_per_sample"],
    "base_update_norm": metrics["update/base/sampled_norm"],
    "context_update_count": metrics["train/context_update_count"],
    "ema_update_count": metrics["train/ema_update_count"],
    "max_allocated_gib": metrics["memory/max_allocated_gib"],
    "scratch_checkpoints": [],
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

echo "R12 fork smoke passed: ${POSTCHECK_JSON}"
