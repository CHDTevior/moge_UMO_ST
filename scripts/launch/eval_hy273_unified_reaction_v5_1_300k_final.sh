#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_unified_reaction_v5_1_full_contact_20260806_1750}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction/hy273_unified_fulltext_reaction_v1_20260801_0315/model/step_00100000.pt}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${RUN_ROOT}/model/step_00200000.pt}"
CANDIDATE_CHECKPOINT="${CANDIDATE_CHECKPOINT:-${RUN_ROOT}/model/step_00300000.pt}"
BASELINE_EVAL_ROOT="${BASELINE_EVAL_ROOT:-${RUN_ROOT}/eval_v5_1_200k_final}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_v5_1_300k_final}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

for path in \
  "${PARENT_CHECKPOINT}" \
  "${BASELINE_CHECKPOINT}" \
  "${CANDIDATE_CHECKPOINT}" \
  "${BASELINE_EVAL_ROOT}/reaction/test/reaction_test.json"; do
  [[ -f "${path}" ]] || {
    echo "Missing required v5.1 dose-evaluation file: ${path}" >&2
    exit 2
  }
done
[[ -d "${BASELINE_EVAL_ROOT}/reaction/test/predictions" ]] || {
  echo "Missing v5.1 200K Reaction predictions" >&2
  exit 2
}

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" - \
  "${BASELINE_CHECKPOINT}" "${CANDIDATE_CHECKPOINT}" <<'PY'
import sys

from tools.compare_hy273_reaction_matched import _validate_training_contract

_validate_training_contract(
    {
        "checkpoint": sys.argv[1],
        "checkpoint_next_global_step": 200_000,
    },
    {
        "checkpoint": sys.argv[2],
        "checkpoint_next_global_step": 300_000,
    },
    mode="same_run_dose_extension",
    baseline_checkpoint_step=200_000,
    candidate_checkpoint_step=300_000,
)
print("Reaction-v5.1 200K->300K training-dose contract verified")
PY

RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
STAGE_A_CHECKPOINT="${PARENT_CHECKPOINT}" \
STAGE_B_CHECKPOINT="${CANDIDATE_CHECKPOINT}" \
STAGE_B_STEP=300000 \
EVAL_ROOT="${EVAL_ROOT}" \
GPU_IDS="${GPU_IDS}" \
PYTHON_BIN="${PYTHON_BIN}" \
ALLOW_REACTION_LOSS_ABLATION=1 \
ALLOW_SAME_MIX_EXTENSION_AT_STEP=200000 \
EVAL_PHASE=all \
  bash scripts/launch/eval_hy273_unified_reaction_stage_b_200k.sh

BASELINE_REPORT="${BASELINE_EVAL_ROOT}/reaction/test/reaction_test.json"
BASELINE_PREDICTIONS="${BASELINE_EVAL_ROOT}/reaction/test/predictions"
CANDIDATE_REPORT="${EVAL_ROOT}/reaction/test/reaction_test.json"
CANDIDATE_PREDICTIONS="${EVAL_ROOT}/reaction/test/predictions"

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/compare_hy273_reaction_matched.py \
  --baseline "${BASELINE_REPORT}" \
  --candidate "${CANDIDATE_REPORT}" \
  --baseline_label reaction_v5_1_200000 \
  --candidate_label reaction_v5_1_300000 \
  --baseline_predictions "${BASELINE_PREDICTIONS}" \
  --candidate_predictions "${CANDIDATE_PREDICTIONS}" \
  --training_contract same_run_dose_extension \
  --baseline_checkpoint_step 200000 \
  --candidate_checkpoint_step 300000 \
  --expected_split test \
  --bootstrap_resamples 10000 \
  --seed 20260807 \
  --output "${EVAL_ROOT}/reaction/test/matched_v5_1_200k_vs_300k.json" \
  >"${EVAL_ROOT}/logs/matched_v5_1_200k_vs_300k_test.log" 2>&1

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
  --report_json "${CANDIDATE_REPORT}" \
  --prediction_dir "${CANDIDATE_PREDICTIONS}" \
  --output_dir "${EVAL_ROOT}/reaction/test/gifs_action_balanced_fk" \
  --max_videos 16 \
  --joint_source fk \
  --fps 30 \
  --stride 3 \
  >"${EVAL_ROOT}/logs/render_reaction_test_fk.log" 2>&1

echo "Reaction-v5.1 300K dose evaluation complete: ${EVAL_ROOT}"
