#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v5_2_all_t_fine_20260808_144750}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_2_all_t_fine}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
BASE_CONFIG="${BASE_CONFIG:-configs/hy273_unified_fulltext_reaction_v5_2_all_t_fine.yaml}"
EXTENDED_CONFIG="${EXTENDED_CONFIG:-configs/hy273_unified_fulltext_reaction_v5_2_all_t_fine_continue300k.yaml}"
CHECKPOINT="${CHECKPOINT:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

[[ -d "${RUN_DIR}" ]] || {
  echo "Missing Reaction-v5.2 run directory: ${RUN_DIR}" >&2
  exit 2
}
[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Reaction-v5.2 continuation requires 8-card DDP" >&2
  exit 2
}
[[ -z "${MAX_UPDATES:-}" ]] || {
  echo "Formal Reaction-v5.2 continuation does not accept MAX_UPDATES" >&2
  exit 2
}

checkpoint_step() {
  "${PYTHON_BIN}" - "$1" "${RUN_NAME}" "${BASE_CONFIG}" "${EXTENDED_CONFIG}" <<'PY'
import sys
import torch

from train_hy273_unified_actor import (
    CHECKPOINT_FORMAT,
    RNG_CONTRACT,
    load_config,
    validate_resume_config,
)

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Continuation requires a unified-actor checkpoint")
if checkpoint.get("rng_contract") != RNG_CONTRACT:
    raise RuntimeError("Continuation checkpoint uses a different RNG contract")
if checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("Continuation checkpoint belongs to a different run")
embedded = checkpoint.get("config")
if not isinstance(embedded, dict):
    raise RuntimeError("Continuation checkpoint has no resolved config")
base_config, _ = load_config(sys.argv[3])
extended_config, _ = load_config(sys.argv[4])
step = int(checkpoint.get("next_global_step", -1))
if step == 200_000:
    if embedded != base_config:
        raise RuntimeError("The 200K checkpoint is not the audited v5.2 training arm")
    validate_resume_config(
        embedded,
        extended_config,
        allow_same_mix_extension_at_step=step,
    )
elif 200_000 < step <= 300_000:
    if embedded != extended_config:
        raise RuntimeError("Post-200K checkpoint does not use the frozen v5.2 extension")
else:
    raise RuntimeError(f"Expected a v5.2 checkpoint in [200K,300K], got {step}")
print(step)
PY
}

# Resume from the furthest valid state so an interrupted continuation keeps the
# exact deterministic task/data cursors instead of replaying an earlier span.
if [[ -n "${CHECKPOINT}" ]]; then
  [[ -f "${CHECKPOINT}" ]] || {
    echo "Missing Reaction-v5.2 continuation checkpoint: ${CHECKPOINT}" >&2
    exit 2
  }
  CURRENT_CHECKPOINT="${CHECKPOINT}"
  CURRENT_STEP="$(checkpoint_step "${CURRENT_CHECKPOINT}")"
else
  CURRENT_CHECKPOINT=""
  CURRENT_STEP=-1
  for candidate in \
    "${RUN_DIR}/model/step_00300000.pt" \
    "${RUN_DIR}/model/latest.pt" \
    "${RUN_DIR}/model/step_00250000.pt" \
    "${RUN_DIR}/model/step_00200000.pt"; do
    [[ -f "${candidate}" ]] || continue
    candidate_step="$(checkpoint_step "${candidate}")"
    if (( candidate_step > CURRENT_STEP )); then
      CURRENT_CHECKPOINT="${candidate}"
      CURRENT_STEP="${candidate_step}"
    fi
  done
  [[ -n "${CURRENT_CHECKPOINT}" ]] || {
    echo "No valid Reaction-v5.2 continuation checkpoint found under ${RUN_DIR}/model" >&2
    exit 2
  }
fi

echo "Reaction-v5.2 continuation source: ${CURRENT_CHECKPOINT} (next step ${CURRENT_STEP})"

if (( CURRENT_STEP < 250000 )); then
  CONFIG="${EXTENDED_CONFIG}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=250000 GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" MASTER_PORT=29833 \
    bash scripts/launch/train_hy273_unified_reaction_stage_b_continue250k_ddp8.sh
  CURRENT_CHECKPOINT="${RUN_DIR}/model/step_00250000.pt"
  CURRENT_STEP="$(checkpoint_step "${CURRENT_CHECKPOINT}")"
  if (( CURRENT_STEP != 250000 )); then
    echo "Reaction-v5.2 continuation ended without an exact 250K archive" >&2
    exit 3
  fi
fi

if (( CURRENT_STEP < 300000 )); then
  CONFIG="${EXTENDED_CONFIG}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=300000 GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" MASTER_PORT=29834 \
    bash scripts/launch/train_hy273_unified_reaction_stage_b_continue50k_ddp8.sh
fi

FINAL_CHECKPOINT="${RUN_DIR}/model/step_00300000.pt"
FINAL_STEP="$(checkpoint_step "${FINAL_CHECKPOINT}")"
if (( FINAL_STEP != 300000 )); then
  echo "Reaction-v5.2 continuation ended without an exact 300K archive" >&2
  exit 3
fi
echo "Reaction-v5.2 continuation reached 300K: ${FINAL_CHECKPOINT}"
