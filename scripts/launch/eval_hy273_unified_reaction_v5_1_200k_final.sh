#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the completed Reaction-v5.1 run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/model/step_00200000.pt}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction/hy273_unified_fulltext_reaction_v1_20260801_0315/model/step_00100000.pt}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_v5_1_200k_final}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

BASELINE_ROOT="${BASELINE_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_event_layout/hy273_unified_reaction_v5_event_layout_20260805_1345}"
BASELINE_REPORT="${BASELINE_REPORT:-${BASELINE_ROOT}/eval_v5_200k_final/reaction/test/reaction_test.json}"
BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS:-${BASELINE_ROOT}/eval_v5_200k_final/reaction/test/predictions}"

for path in "${CHECKPOINT}" "${PARENT_CHECKPOINT}" "${BASELINE_REPORT}"; do
  [[ -f "${path}" ]] || {
    echo "Missing required file: ${path}" >&2
    exit 2
  }
done
[[ -d "${BASELINE_PREDICTIONS}" ]] || {
  echo "Missing Reaction-v5 prediction directory: ${BASELINE_PREDICTIONS}" >&2
  exit 2
}

RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
STAGE_A_CHECKPOINT="${PARENT_CHECKPOINT}" \
STAGE_B_CHECKPOINT="${CHECKPOINT}" \
STAGE_B_STEP=200000 \
EVAL_ROOT="${EVAL_ROOT}" \
GPU_IDS="${GPU_IDS}" \
PYTHON_BIN="${PYTHON_BIN}" \
ALLOW_REACTION_LOSS_ABLATION=1 \
EVAL_PHASE=all \
  bash scripts/launch/eval_hy273_unified_reaction_stage_b_200k.sh

CANDIDATE_REPORT="${EVAL_ROOT}/reaction/test/reaction_test.json"
CANDIDATE_PREDICTIONS="${EVAL_ROOT}/reaction/test/predictions"
CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/compare_hy273_reaction_matched.py \
  --baseline "${BASELINE_REPORT}" \
  --candidate "${CANDIDATE_REPORT}" \
  --baseline_label reaction_v5_200000 \
  --candidate_label reaction_v5_1_200000 \
  --baseline_predictions "${BASELINE_PREDICTIONS}" \
  --candidate_predictions "${CANDIDATE_PREDICTIONS}" \
  --training_contract reaction_v5_1_full_contact \
  --expected_checkpoint_step 200000 \
  --expected_split test \
  --bootstrap_resamples 10000 \
  --seed 20260806 \
  --output "${EVAL_ROOT}/reaction/test/matched_v5_vs_v5_1_200000.json" \
  >"${EVAL_ROOT}/logs/matched_v5_vs_v5_1_test.log" 2>&1

CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
  --report_json "${CANDIDATE_REPORT}" \
  --prediction_dir "${CANDIDATE_PREDICTIONS}" \
  --output_dir "${EVAL_ROOT}/reaction/test/gifs_action_balanced_fk" \
  --max_videos 16 \
  --joint_source fk \
  --fps 30 \
  --stride 3 \
  >"${EVAL_ROOT}/logs/render_reaction_test_fk.log" 2>&1

echo "Reaction-v5.1 200K full evaluation complete: ${EVAL_ROOT}"
