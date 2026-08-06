#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v4_layout_20260804_151750}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v4_layout}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/model/step_00150000.pt}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_v4_150k_fixed}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
GPU_ID="${GPU_ID:-0}"

V3_ROOT="${V3_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v3_adaptive/hy273_unified_reaction_v3_adaptive_20260804_0628_smoke/eval_v3_150k_screen/v3_150k/reaction/val}"
V3_REPORT="${V3_REPORT:-${V3_ROOT}/reaction_val.json}"
V3_PREDICTIONS="${V3_PREDICTIONS:-${V3_ROOT}/predictions}"

VAL_ROOT="${EVAL_ROOT}/reaction/val"
V4_REPORT="${VAL_ROOT}/reaction_val.json"
V4_PREDICTIONS="${VAL_ROOT}/predictions"
COMPARISON="${EVAL_ROOT}/matched_v3_vs_v4_150k.json"
PROTOCOL_LOCK="${EVAL_ROOT}/reaction/final_protocol_lock.json"
FOCUS_UIDS="${FOCUS_UIDS:-G030T004A009R010,G041T007A020R002,G043T002A011R010,G023T006A031R006,G054T000A003R023,G021T001A006R008,G002T009A039R009}"

for path in "${CHECKPOINT}" "${V3_REPORT}"; do
  [[ -f "${path}" ]] || {
    echo "Missing required file: ${path}" >&2
    exit 2
  }
done
[[ -d "${V3_PREDICTIONS}" ]] || {
  echo "Missing v3 prediction directory: ${V3_PREDICTIONS}" >&2
  exit 2
}

mkdir -p "${VAL_ROOT}" "${EVAL_ROOT}/reaction" "${EVAL_ROOT}/logs"

"${PYTHON_BIN}" - "${CHECKPOINT}" "${PROTOCOL_LOCK}" <<'PY'
import json
from pathlib import Path
import sys

import torch

from train_hy273_unified_actor import CHECKPOINT_FORMAT, validate_config

checkpoint_path = Path(sys.argv[1]).expanduser().resolve()
lock_path = Path(sys.argv[2]).expanduser().resolve()
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    mmap=True,
    weights_only=False,
)
if checkpoint.get("format") != CHECKPOINT_FORMAT:
    raise RuntimeError("Reaction-v4 evaluation requires a unified-actor checkpoint")
if int(checkpoint.get("next_global_step", -1)) != 150_000:
    raise RuntimeError("Reaction-v4 evaluation requires the exact 150K checkpoint")
config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError("Checkpoint has no resolved config")
validate_config(config)
if config["data"].get("paired_task") != "reaction":
    raise RuntimeError("Checkpoint is not a single-target Reaction model")

expected_layout = {
    "relative_root": 0.0195,
    "relative_heading": 0.0217,
    "heading_beta": 0.10,
    "layout_initial_frames": 15,
    "layout_initial_multiplier": 3.0,
    "layout_precontact_multiplier": 2.0,
    "layout_contact_threshold_m": 0.20,
}
reaction_loss = config.get("reaction_loss", {})
for key, expected in expected_layout.items():
    if reaction_loss.get(key) != expected:
        raise RuntimeError(
            f"Reaction-v4 layout contract mismatch for {key}: "
            f"{reaction_loss.get(key)!r} != {expected!r}"
        )

ema_every = int(config["training"]["ema_every"])
if int(checkpoint.get("ema_update_count", -1)) != 150_000 // ema_every:
    raise RuntimeError("EMA update count is inconsistent with step 150K")
batcher = checkpoint.get("batcher", {})
scheduler_state = batcher.get("scheduler", {}).get("state", {})
expected_counts = {
    "next_step": 150_000,
    "realized_hml": 115_000,
    "realized_edit": 17_500,
    "realized_interaction": 17_500,
}
for key, expected in expected_counts.items():
    if int(scheduler_state.get(key, -1)) != expected:
        raise RuntimeError(
            f"150K task scheduler mismatch for {key}: "
            f"{scheduler_state.get(key)!r} != {expected}"
        )

payload = {
    "format": "hy273_reaction_eval_cfg_lock_v1",
    "checkpoint": str(checkpoint_path),
    "checkpoint_next_global_step": 150_000,
    "weight_source": "ema",
    "num_steps": 32,
    "source_cfg_scale": 2.0,
    "text_cfg_scale": 2.0,
    "caption_policy": "uid_balanced",
    "seed": 20260801,
    "selection_policy": "preregistered_fixed_before_val_and_test",
    "splits": ["val", "test"],
}
lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"checkpoint": str(checkpoint_path), "protocol_lock": str(lock_path)}))
PY

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/eval_hy273_reaction.py \
  --checkpoint "${CHECKPOINT}" \
  --split val \
  --weight_source ema \
  --device cuda:0 \
  --batch_size 8 \
  --num_steps 32 \
  --source_cfg_scale 2.0 \
  --text_cfg_scale 2.0 \
  --seed 20260801 \
  --caption_policy uid_balanced \
  --bootstrap_resamples 10000 \
  --save_predictions \
  --final_protocol_lock "${PROTOCOL_LOCK}" \
  --require_final_protocol \
  --output_json "${V4_REPORT}" \
  >"${EVAL_ROOT}/logs/reaction_val.log" 2>&1

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/compare_hy273_reaction_matched.py \
  --baseline "${V3_REPORT}" \
  --candidate "${V4_REPORT}" \
  --baseline_label reaction_v3_150k \
  --candidate_label reaction_v4_layout_150k \
  --baseline_predictions "${V3_PREDICTIONS}" \
  --candidate_predictions "${V4_PREDICTIONS}" \
  --training_contract reaction_v4_layout \
  --bootstrap_resamples 10000 \
  --seed 20260804 \
  --output "${COMPARISON}" \
  >"${EVAL_ROOT}/logs/matched_v3_vs_v4.log" 2>&1

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
  --report_json "${V4_REPORT}" \
  --prediction_dir "${V4_PREDICTIONS}" \
  --output_dir "${EVAL_ROOT}/gifs_action_balanced" \
  --max_videos 12 \
  --joint_source position \
  --fps 30 \
  --stride 3 \
  >"${EVAL_ROOT}/logs/render_action_balanced.log" 2>&1

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
  --report_json "${V4_REPORT}" \
  --prediction_dir "${V4_PREDICTIONS}" \
  --output_dir "${EVAL_ROOT}/gifs_focus_cases" \
  --max_videos 7 \
  --uids "${FOCUS_UIDS}" \
  --joint_source position \
  --fps 30 \
  --stride 3 \
  >"${EVAL_ROOT}/logs/render_focus_cases.log" 2>&1

echo "Reaction-v4 150K evaluation complete: ${EVAL_ROOT}"
