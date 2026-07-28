#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_multitask_r11_stage_a_t2m_ddp8_20260715_1510}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${ROOT_DIR}/outputs/hy273_multitask}"
FORMAL_RUN_DIR="${FORMAL_OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT:-${FORMAL_RUN_DIR}/model/step_00250000.pt}"
SCRATCH_TAG="${SCRATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
SCRATCH_OUTPUT_ROOT="${SCRATCH_OUTPUT_ROOT:-${ROOT_DIR}/outputs/hy273_multitask_smoke/b2_ddp8_${SCRATCH_TAG}}"
SCRATCH_RUN_DIR="${SCRATCH_OUTPUT_ROOT}/${RUN_NAME}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-16}"
MATERIALIZE_WORKERS="${MATERIALIZE_WORKERS:-4}"
SMOKE_STEPS="${SMOKE_STEPS:-3937}"
MAX_PRELAUNCH_USED_MIB="${MAX_PRELAUNCH_USED_MIB:-1024}"
MAX_ALLOCATED_GIB="${MAX_ALLOCATED_GIB:-70.0}"

[[ "${NPROC_PER_NODE}" == "8" ]] || { echo "B2 smoke requires NPROC_PER_NODE=8" >&2; exit 2; }
[[ "${BATCH_SIZE_PER_RANK}" == "16" ]] || { echo "B2 smoke requires batch/rank=16" >&2; exit 2; }
[[ "${SMOKE_STEPS}" == "3937" ]] || { echo "B2 smoke is frozen at 3937 steps" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" && -x "${TORCHRUN_BIN}" ]] || { echo "Missing Python/torchrun" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing formal 250K checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${FORMAL_RUN_DIR}/run_identity.json" ]] || { echo "Missing formal run identity" >&2; exit 2; }
[[ ! -e "${SCRATCH_OUTPUT_ROOT}" ]] || { echo "Scratch output already exists: ${SCRATCH_OUTPUT_ROOT}" >&2; exit 2; }

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
[[ "${#GPU_ARRAY[@]}" == "8" ]] || { echo "GPU_IDS must contain exactly eight GPUs" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU id: ${gpu@Q}" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[${gpu}]:-}" ]] || { echo "Duplicate GPU id: ${gpu}" >&2; exit 2; }
  SEEN_GPUS["${gpu}"]=1
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( used > MAX_PRELAUNCH_USED_MIB )); then
    echo "GPU ${gpu} is not idle: ${used} MiB used (limit ${MAX_PRELAUNCH_USED_MIB})" >&2
    exit 2
  fi
  compute_processes="$(
    nvidia-smi --id="${gpu}" \
      --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
  )"
  if [[ -n "${compute_processes}" ]]; then
    echo "GPU ${gpu} has active compute processes: ${compute_processes}" >&2
    exit 2
  fi
done

mkdir -p "${SCRATCH_RUN_DIR}"
cp "${FORMAL_RUN_DIR}/run_identity.json" "${SCRATCH_RUN_DIR}/run_identity.json"
CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
PRECHECK_JSON="${SCRATCH_OUTPUT_ROOT}/precheck.json"
POSTCHECK_JSON="${SCRATCH_OUTPUT_ROOT}/postcheck.json"
LOG_PATH="${SCRATCH_OUTPUT_ROOT}/torchrun.log"

"${PYTHON_BIN}" - \
  "${CHECKPOINT}" \
  "${CHECKPOINT_SHA256}" \
  "${SCRATCH_RUN_DIR}/run_identity.json" \
  "${PRECHECK_JSON}" <<'PY'
import json
import sys
from pathlib import Path

import torch

checkpoint_path, checkpoint_sha, identity_path, output_path = map(Path, sys.argv[1:])
checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
identity = json.loads(identity_path.read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

require(checkpoint.get("format") == "hy273_multitask_checkpoint_v2", "checkpoint format mismatch")
require(checkpoint.get("next_global_step") == 250_000, "checkpoint is not exactly step 250K")
require(checkpoint.get("phase_id") == 2, "checkpoint phase_id is not Stage B2")
require(checkpoint.get("context_update_count") == 0, "context was updated before Stage B2")
require(identity.get("format") == "hy273_multitask_run_identity_v1", "run identity format mismatch")
require(isinstance(identity.get("run_name"), str) and identity["run_name"], "run identity name is empty")
require(isinstance(identity.get("run_uuid"), str) and identity["run_uuid"], "run identity UUID is empty")
require(checkpoint.get("run_name") == identity.get("run_name"), "run name mismatch")
require(checkpoint.get("run_uuid") == identity.get("run_uuid"), "run UUID mismatch")

batcher = checkpoint["batcher"]
scheduler = batcher["scheduler"]
require(batcher.get("manifest_sha256") == "9daa5685cd6cb265abe89f746764006908a72941212d582623c0c6bd8a5dbe45", "manifest SHA mismatch")
require(batcher.get("world_size") == 8, "checkpoint world_size mismatch")
require(batcher.get("batch_size_per_rank") == 16, "checkpoint batch/rank mismatch")
require(batcher.get("global_batch_size") == 128, "checkpoint global batch mismatch")
require(batcher.get("next_global_sample_ordinal") == 32_000_000, "sample ordinal mismatch")
for key, expected in {
    "next_step": 250_000,
    "debt_hml": 0,
    "debt_edit": 0,
    "realized_hml": 250_000,
    "realized_edit": 0,
}.items():
    require(scheduler.get(key) == expected, f"scheduler {key} mismatch")

edit_cursor = batcher["cursors"]["1"]
require(edit_cursor.get("cycle") == 0, "edit cursor cycle is not zero")
require(edit_cursor.get("offset") == 0, "edit cursor offset is not zero")
require(edit_cursor.get("pending_batches") == [], "edit cursor already has pending batches")

optimizer = checkpoint["optimizer"]
groups = optimizer["param_groups"]
require([group.get("group_name") for group in groups] == ["G0_existing", "G1_context_weight", "G2_context_bias"], "optimizer groups mismatch")
context_ids = groups[1]["params"] + groups[2]["params"]
require(not any(parameter_id in optimizer["state"] for parameter_id in context_ids), "context optimizer state exists before B2")

payload = {
    "format": "hy273_multitask_b2_smoke_precheck_v1",
    "checkpoint": str(checkpoint_path.resolve()),
    "checkpoint_sha256": str(checkpoint_sha),
    "next_global_step": checkpoint["next_global_step"],
    "phase_id": checkpoint["phase_id"],
    "run_name": checkpoint["run_name"],
    "run_uuid": checkpoint["run_uuid"],
    "context_update_count": checkpoint["context_update_count"],
    "manifest_sha256": batcher["manifest_sha256"],
    "next_global_sample_ordinal": batcher["next_global_sample_ordinal"],
    "scheduler": scheduler,
    "edit_cursor": edit_cursor,
    "smoke_start_step": 250_000,
    "smoke_stop_step": 253_937,
    "expected_edit_updates": 16,
    "expected_max_target_frames": 150,
    "expected_max_source_frames": 151,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "checkpoint=${CHECKPOINT}"
echo "checkpoint_sha256=${CHECKPOINT_SHA256}"
echo "scratch_output_root=${SCRATCH_OUTPUT_ROOT}"
echo "smoke_steps=${SMOKE_STEPS} stop_step=253937"

set +e
"${TORCHRUN_BIN}" \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_hy273_multitask.py \
  --config configs/hy273_multitask_stage_b2_joint_adapt.yaml \
  --name "${RUN_NAME}" \
  --resume "${CHECKPOINT}" \
  --resume_sha256 "${CHECKPOINT_SHA256}" \
  --output_dir "${SCRATCH_OUTPUT_ROOT}" \
  --batch_size_per_rank "${BATCH_SIZE_PER_RANK}" \
  --materialize_workers "${MATERIALIZE_WORKERS}" \
  --smoke_steps "${SMOKE_STEPS}" \
  --no-save_smoke 2>&1 | tee "${LOG_PATH}"
TORCHRUN_STATUS="${PIPESTATUS[0]}"
set -e
[[ "${TORCHRUN_STATUS}" == "0" ]] || { echo "B2 smoke failed with status ${TORCHRUN_STATUS}" >&2; exit "${TORCHRUN_STATUS}"; }

"${PYTHON_BIN}" - \
  "${SCRATCH_RUN_DIR}/metrics.jsonl" \
  "${LOG_PATH}" \
  "${CHECKPOINT}" \
  "${CHECKPOINT_SHA256}" \
  "${SCRATCH_RUN_DIR}" \
  "${MAX_ALLOCATED_GIB}" \
  "${RUN_NAME}" \
  "${POSTCHECK_JSON}" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

metrics_path, log_path, checkpoint_path, expected_sha, scratch_run, max_memory, expected_run_name, output_path = sys.argv[1:]
metrics_path = Path(metrics_path)
log_path = Path(log_path)
checkpoint_path = Path(checkpoint_path)
scratch_run = Path(scratch_run)
output_path = Path(output_path)
max_memory = float(max_memory)

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def require_close(actual, expected, name, atol=1e-12):
    require(abs(float(actual) - float(expected)) <= atol, f"{name}: {actual} != {expected}")

def walk_numeric(value, path="metrics"):
    if isinstance(value, dict):
        require(bool(value), f"empty metric dictionary at {path}")
        for key, item in value.items():
            walk_numeric(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            walk_numeric(item, f"{path}[{index}]")
    else:
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"nonnumeric metric leaf at {path}: {value!r}",
        )
        require(math.isfinite(float(value)), f"non-finite value at {path}")

records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expected_steps = list(range(250_020, 253_921, 20)) + [253_937]
require([record["step"] for record in records] == expected_steps, "metrics step sequence mismatch")
require(len(records) == 197, "expected exactly 197 metric records")
for record in records:
    walk_numeric(record["metrics"])

for record in records:
    if record["step"] > 250_700:
        break
    metrics = record["metrics"]
    for key in (
        "batch/source_present",
        "batch/stream_edit",
        "grad/context_preclip",
        "update/context/sampled_norm",
        "train/context_update_count",
    ):
        require_close(metrics[key], 0.0, f"pre-edit {record['step']} {key}")

first_edit_window = next(record["metrics"] for record in records if record["step"] == 250_720)
for key, expected in {
    "batch/source_present": 0.05,
    "batch/stream_edit": 0.05,
    "batch/stream_hml": 0.95,
    "train/context_update_count": 1.0,
    "loss/overall/rank_steps": 160.0,
    "loss/overall/samples": 2560.0,
    "loss/stream/MOTION_EDIT/rank_steps": 8.0,
    "loss/stream/MOTION_EDIT/samples": 128.0,
    "loss/capability/MOTION_EDIT/rank_steps": 8.0,
    "loss/capability/MOTION_EDIT/samples": 103.0,
    "loss/capability/MOTION_EDIT_CONTROL/rank_steps": 8.0,
    "loss/capability/MOTION_EDIT_CONTROL/samples": 25.0,
}.items():
    require_close(first_edit_window[key], expected, f"first edit window {key}")
require(first_edit_window["grad/context_preclip"] > 0.0, "first edit context gradient is zero")
require(first_edit_window["update/context/sampled_norm"] > 0.0, "first edit context update is zero")
require(first_edit_window["grad/base_preclip"] > 0.0, "first edit base gradient is zero")
require(first_edit_window["update/base/sampled_norm"] > 0.0, "first edit base update is zero")

final = records[-1]["metrics"]
for key, expected in {
    "batch/source_present": 1.0 / 17.0,
    "batch/stream_edit": 1.0 / 17.0,
    "batch/stream_hml": 16.0 / 17.0,
    "train/context_update_count": 16.0,
    "loss/overall/rank_steps": 136.0,
    "loss/overall/samples": 2176.0,
    "loss/stream/MOTION_EDIT/rank_steps": 8.0,
    "loss/stream/MOTION_EDIT/samples": 128.0,
    "loss/capability/MOTION_EDIT/rank_steps": 8.0,
    "loss/capability/MOTION_EDIT/samples": 96.0,
    "loss/capability/MOTION_EDIT_CONTROL/rank_steps": 8.0,
    "loss/capability/MOTION_EDIT_CONTROL/samples": 32.0,
}.items():
    require_close(final[key], expected, f"longest edit window {key}")
require(final["grad/context_preclip"] > 0.0, "longest edit context gradient is zero")
require(final["update/context/sampled_norm"] > 0.0, "longest edit context update is zero")
require(final["grad/base_preclip"] > 0.0, "longest edit base gradient is zero")
require(final["update/base/sampled_norm"] > 0.0, "longest edit base update is zero")
require(final["throughput/samples_per_second"] > 0.0, "throughput is not positive")
require(0.0 < final["memory/max_allocated_gib"] <= max_memory, "memory headroom gate failed")
expected_final_trace = "b80682722023418edd1ce1cf5ae1c0beafc5656a2374ecccf2ccb7c3c222fb92"
require(records[-1].get("plan_trace") == expected_final_trace, "longest edit plan trace mismatch")

log_text = log_path.read_text(encoding="utf-8", errors="replace")
json_events = []
for line in log_text.splitlines():
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(event, dict) and event.get("event") == "stage_complete":
        json_events.append(event)
require(len(json_events) == 1, f"expected one stage_complete event, got {len(json_events)}")
stage_complete = json_events[0]
require(stage_complete.get("run") == expected_run_name, "stage_complete run mismatch")
require(stage_complete.get("next_global_step") == 253_937, "stage_complete step mismatch")
require(stage_complete.get("context_update_count") == 16, "stage_complete context count mismatch")
bad_patterns = (
    r"Traceback \(most recent call last\)",
    r"CUDA out of memory",
    r"ChildFailedError",
    r"(?:NCCL|ProcessGroupNCCL)[^\n]*(?:abort(?:ed)?|timeout|watchdog|unhandled CUDA error|system error|internal error|remote process exited|duplicate GPU detected)",
    r"DDP ranks selected different",
    r"non-finite",
)
for pattern in bad_patterns:
    require(re.search(pattern, log_text, flags=re.IGNORECASE) is None, f"error pattern in log: {pattern}")

checkpoints = sorted(str(path) for path in scratch_run.rglob("*.pt"))
require(not checkpoints, f"scratch smoke wrote checkpoints: {checkpoints}")
actual_sha = sha256_file(checkpoint_path)
require(actual_sha == expected_sha, "formal checkpoint SHA changed during smoke")

payload = {
    "format": "hy273_multitask_b2_smoke_postcheck_v1",
    "status": "passed",
    "checkpoint": str(checkpoint_path.resolve()),
    "checkpoint_sha256_before_after": actual_sha,
    "record_count": len(records),
    "first_edit_window_step": 250_720,
    "first_edit_context_update_count": first_edit_window["train/context_update_count"],
    "final_step": records[-1]["step"],
    "final_plan_trace": records[-1]["plan_trace"],
    "final_context_update_count": final["train/context_update_count"],
    "max_allocated_gib": final["memory/max_allocated_gib"],
    "throughput_samples_per_second_final_window": final["throughput/samples_per_second"],
    "nccl_warning_lines": [
        line for line in log_text.splitlines() if "NCCL" in line and "WARN" in line.upper()
    ],
    "scratch_checkpoints": checkpoints,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

echo "B2 DDP8 smoke passed: ${POSTCHECK_JSON}"
