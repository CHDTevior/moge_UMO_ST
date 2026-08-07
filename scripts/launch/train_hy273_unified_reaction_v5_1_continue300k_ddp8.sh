#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v5_1_full_contact_20260806_1750}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact}"
RUN_DIR="${OUTPUT_DIR}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EXTENDED_CONFIG="${EXTENDED_CONFIG:-configs/hy273_unified_fulltext_reaction_v5_1_full_contact_continue300k.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

[[ -d "${RUN_DIR}" ]] || {
  echo "Missing Reaction-v5.1 run directory: ${RUN_DIR}" >&2
  exit 2
}
[[ "${NPROC_PER_NODE}" == "8" ]] || {
  echo "Reaction-v5.1 continuation requires 8-card DDP" >&2
  exit 2
}
[[ -z "${MAX_UPDATES:-}" ]] || {
  echo "Reaction-v5.1 formal continuation does not accept MAX_UPDATES" >&2
  exit 2
}

checkpoint_step() {
  "${PYTHON_BIN}" - "$1" "${RUN_NAME}" "${EXTENDED_CONFIG}" <<'PY'
import sys
import torch
import yaml

from train_hy273_unified_actor import (
    CHECKPOINT_FORMAT,
    RNG_CONTRACT,
    validate_resume_config,
)

checkpoint = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Continuation requires a unified-actor checkpoint")
if checkpoint.get("rng_contract") != RNG_CONTRACT:
    raise RuntimeError("Continuation checkpoint uses a different RNG contract")
if checkpoint.get("run_name") != sys.argv[2]:
    raise RuntimeError("Continuation checkpoint belongs to a different run")
config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError("Continuation checkpoint has no resolved config")
with open(sys.argv[3], "r", encoding="utf-8") as handle:
    extended_config = yaml.safe_load(handle)
step = int(checkpoint.get("next_global_step", -1))
validate_resume_config(
    config,
    extended_config,
    allow_same_mix_extension_at_step=step,
)
if config.get("data", {}).get("paired_task") != "reaction":
    raise RuntimeError("Continuation checkpoint is not the single-target Reaction model")
if config.get("model", {}).get("text_token_sequence") != "sentence_plus_context":
    raise RuntimeError("Continuation checkpoint does not use the full text stream")
reaction_loss = config.get("reaction_loss", {})
expected = {
    "fk_contact_map_positive": 0.001,
    "fk_contact_map_negative": 0.005,
    "fk_contact_vector": 0.002,
    "fk_contact_transition": 0.003,
}
for key, value in expected.items():
    if float(reaction_loss.get(key, -1.0)) != value:
        raise RuntimeError(f"Checkpoint is not Reaction-v5.1: {key}")
print(step)
PY
}

# A restarted research run resumes from the highest valid saved state. An
# explicit CHECKPOINT still takes precedence for controlled replay/debugging.
if [[ -n "${CHECKPOINT}" ]]; then
  [[ -f "${CHECKPOINT}" ]] || {
    echo "Missing Reaction-v5.1 continuation checkpoint: ${CHECKPOINT}" >&2
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
    echo "No valid Reaction-v5.1 continuation checkpoint found under ${RUN_DIR}/model" >&2
    exit 2
  }
fi

echo "Reaction-v5.1 continuation source: ${CURRENT_CHECKPOINT} (next step ${CURRENT_STEP})"
if (( CURRENT_STEP < 200000 || CURRENT_STEP > 300000 )); then
  echo "Expected a Reaction-v5.1 continuation checkpoint in [200000,300000], got ${CURRENT_STEP}" >&2
  exit 2
fi

if (( CURRENT_STEP < 250000 )); then
  CONFIG="${EXTENDED_CONFIG}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=250000 GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" MASTER_PORT=29820 \
    bash scripts/launch/train_hy273_unified_reaction_stage_b_continue250k_ddp8.sh
  CURRENT_CHECKPOINT="${RUN_DIR}/model/step_00250000.pt"
  CURRENT_STEP="$(checkpoint_step "${CURRENT_CHECKPOINT}")"
  if (( CURRENT_STEP != 250000 )); then
    echo "Reaction-v5.1 continuation ended without an exact 250K archive" >&2
    exit 3
  fi
fi

ARCHIVE_250K="${RUN_DIR}/model/step_00250000.pt"
ARCHIVE_250K_STEP="$(checkpoint_step "${ARCHIVE_250K}")"
if (( ARCHIVE_250K_STEP != 250000 )); then
  echo "Reaction-v5.1 continuation requires an exact retained 250K archive" >&2
  exit 3
fi

if (( CURRENT_STEP < 300000 )); then
  CONFIG="${EXTENDED_CONFIG}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
    CHECKPOINT="${CURRENT_CHECKPOINT}" STOP_STEP=300000 GPU_IDS="${GPU_IDS}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" MASTER_PORT=29821 \
    bash scripts/launch/train_hy273_unified_reaction_stage_b_continue50k_ddp8.sh
fi

FINAL_CHECKPOINT="${RUN_DIR}/model/step_00300000.pt"
FINAL_STEP="$(checkpoint_step "${FINAL_CHECKPOINT}")"
if (( FINAL_STEP != 300000 )); then
  echo "Reaction-v5.1 continuation ended without an exact 300K archive" >&2
  exit 3
fi
echo "Reaction-v5.1 continuation reached 300K: ${FINAL_CHECKPOINT}"
