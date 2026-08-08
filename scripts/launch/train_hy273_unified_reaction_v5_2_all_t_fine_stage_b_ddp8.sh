#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/hy273_unified_fulltext_reaction_v5_2_all_t_fine.yaml}"
REFERENCE_CONFIG="${REFERENCE_CONFIG:-configs/hy273_unified_fulltext_reaction_v5_1_full_contact.yaml}"
PARENT_OUTPUT_DIR="${PARENT_OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction}"
PARENT_RUN_NAME="${PARENT_RUN_NAME:-hy273_unified_fulltext_reaction_v1_20260801_0315}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-${PARENT_OUTPUT_DIR}/${PARENT_RUN_NAME}/model/step_00100000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_2_all_t_fine}"
RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v5_2_all_t_fine_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT="${CHECKPOINT:-${PARENT_CHECKPOINT}}"
STOP_STEP="${STOP_STEP:-200000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29832}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Reaction-v5.2 requires 8-card DDP" >&2
  exit 2
}
[[ "${STOP_STEP}" == "200000" ]] || {
  echo "Reaction-v5.2 comparison must stop at 200000" >&2
  exit 2
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Reaction-v5.2 checkpoint: ${CHECKPOINT}" >&2
  exit 2
}

NEXT_STEP=$("${PYTHON_BIN}" - \
  "${CHECKPOINT}" "${RUN_NAME}" "${PARENT_RUN_NAME}" \
  "${CONFIG}" "${REFERENCE_CONFIG}" <<'PY'
import sys
import torch

from tools.compare_hy273_reaction_matched import _config_differences
from train_hy273_unified_actor import CHECKPOINT_FORMAT, load_config


def format_differences(differences):
    return [
        {"path": ".".join(path), "baseline": old, "candidate": new}
        for path, old, new in differences
    ]

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Reaction-v5.2 requires a unified-actor checkpoint")
candidate_config, _ = load_config(sys.argv[4])
reference_config, _ = load_config(sys.argv[5])
config_differences = _config_differences(reference_config, candidate_config)
expected_differences = [
    (("reaction_loss", "fine_min_flow_t"), 0.2, 0.0),
    (("reaction_loss", "min_flow_t"), 0.2, 0.0),
]
if config_differences != expected_differences:
    raise RuntimeError(
        "Reaction-v5.2 config must differ from v5.1 only in the all-timestep "
        f"gates: {format_differences(config_differences)}"
    )
step = int(checkpoint.get("next_global_step", -1))
embedded_config = checkpoint.get("config")
if not isinstance(embedded_config, dict):
    raise RuntimeError("Reaction-v5.2 checkpoint has no embedded training config")
if step == 100_000:
    if checkpoint.get("run_name") != sys.argv[3]:
        raise RuntimeError("Reaction-v5.2 must start from the designated common parent run")
    parent_non_reaction = {
        key: value for key, value in embedded_config.items() if key != "reaction_loss"
    }
    reference_non_reaction = {
        key: value for key, value in reference_config.items() if key != "reaction_loss"
    }
    parent_differences = _config_differences(
        parent_non_reaction, reference_non_reaction
    )
    if parent_differences:
        raise RuntimeError(
            "Reaction-v5.2 parent differs from the v5.1 common training contract: "
            f"{format_differences(parent_differences)}"
        )
elif 100_000 < step < 200_000:
    if checkpoint.get("run_name") != sys.argv[2]:
        raise RuntimeError("Continuation checkpoint belongs to another run")
    continuation_differences = _config_differences(
        embedded_config, candidate_config
    )
    if continuation_differences:
        raise RuntimeError(
            "Reaction-v5.2 continuation checkpoint uses another experiment config: "
            f"{format_differences(continuation_differences)}"
        )
else:
    raise RuntimeError(f"Reaction-v5.2 requires a checkpoint in [100K,200K), got {step}")
print(step)
PY
)
if (( NEXT_STEP >= STOP_STEP )); then
  echo "Checkpoint step ${NEXT_STEP} has reached STOP_STEP=${STOP_STEP}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "stage=reaction-v5.2-all-t-fine tasks=t2m,edit,reaction mix=30,35,35 run=${RUN_NAME} interval=${NEXT_STEP}:${STOP_STEP}"
echo "checkpoint=${CHECKPOINT} parent_run=${PARENT_RUN_NAME} config=${CONFIG} reference_config=${REFERENCE_CONFIG}"
EXTRA_ARGS=()
[[ -z "${MAX_UPDATES:-}" ]] || EXTRA_ARGS+=(--max_updates "${MAX_UPDATES}")
exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" train_hy273_unified_actor.py \
  --config "${CONFIG}" --name "${RUN_NAME}" --output_dir "${OUTPUT_DIR}" \
  --resume "${CHECKPOINT}" --stop_step "${STOP_STEP}" \
  --phase_contract fulltext_reaction_v2_stage_b "${EXTRA_ARGS[@]}"
