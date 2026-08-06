#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to a Reaction-v5.1 run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact}"
STEP="${STEP:-150000}"
SPLITS="${SPLITS:-val}"
GPU_ID="${GPU_ID:-0}"
BASELINE_GPU_ID="${BASELINE_GPU_ID:-1}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/model/$(printf 'step_%08d.pt' "${STEP}")}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_v5_1_${STEP}}"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_event_layout/hy273_unified_reaction_v5_event_layout_20260805_1345}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${BASELINE_RUN_ROOT}/model/$(printf 'step_%08d.pt' "${STEP}")}"
FOCUS_UIDS="${FOCUS_UIDS:-G030T004A009R010,G041T007A020R002,G043T002A011R010,G023T006A031R006,G054T000A003R023,G021T001A006R008,G002T009A039R009}"

[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Reaction-v5.1 checkpoint: ${CHECKPOINT}" >&2
  exit 2
}
[[ -f "${BASELINE_CHECKPOINT}" ]] || {
  echo "Missing same-step Reaction-v5 checkpoint: ${BASELINE_CHECKPOINT}" >&2
  exit 2
}
mkdir -p "${EVAL_ROOT}/logs"

run_reaction_eval() {
  local checkpoint="$1"
  local split="$2"
  local report="$3"
  local log="$4"
  local gpu_id="$5"
  mkdir -p "$(dirname "${report}")"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" tools/eval_hy273_reaction.py \
    --checkpoint "${checkpoint}" \
    --split "${split}" \
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
    --output_json "${report}" \
    >"${log}" 2>&1
}

baseline_existing_root() {
  local split="$1"
  if [[ "${STEP}" == "150000" ]]; then
    printf '%s\n' "${BASELINE_RUN_ROOT}/eval_v5_150k_gate/reaction/${split}"
  elif [[ "${STEP}" == "200000" ]]; then
    printf '%s\n' "${BASELINE_RUN_ROOT}/eval_v5_200k_final/reaction/${split}"
  else
    printf '%s\n' ""
  fi
}

IFS=',' read -r -a REQUESTED_SPLITS <<< "${SPLITS}"
for split in "${REQUESTED_SPLITS[@]}"; do
  [[ "${split}" == "val" || "${split}" == "test" ]] || {
    echo "SPLITS accepts val,test" >&2
    exit 2
  }
  split_root="${EVAL_ROOT}/reaction/${split}"
  report="${split_root}/reaction_${split}.json"
  mkdir -p "${split_root}"
  existing_baseline_root="$(baseline_existing_root "${split}")"
  if [[ -n "${existing_baseline_root}" \
        && -f "${existing_baseline_root}/reaction_${split}.json" \
        && -d "${existing_baseline_root}/predictions" ]]; then
    baseline_report="${existing_baseline_root}/reaction_${split}.json"
    baseline_predictions="${existing_baseline_root}/predictions"
    run_reaction_eval \
      "${CHECKPOINT}" \
      "${split}" \
      "${report}" \
      "${EVAL_ROOT}/logs/reaction_${split}.log" \
      "${GPU_ID}"
  else
    baseline_root="${EVAL_ROOT}/baseline_v5/reaction/${split}"
    baseline_report="${baseline_root}/reaction_${split}.json"
    baseline_predictions="${baseline_root}/predictions"
    run_reaction_eval \
      "${CHECKPOINT}" \
      "${split}" \
      "${report}" \
      "${EVAL_ROOT}/logs/reaction_${split}.log" \
      "${GPU_ID}" &
    candidate_pid="$!"
    run_reaction_eval \
      "${BASELINE_CHECKPOINT}" \
      "${split}" \
      "${baseline_report}" \
      "${EVAL_ROOT}/logs/baseline_v5_reaction_${split}.log" \
      "${BASELINE_GPU_ID}" &
    baseline_pid="$!"
    status=0
    wait "${candidate_pid}" || status=1
    wait "${baseline_pid}" || status=1
    if [[ "${status}" -ne 0 ]]; then
      echo "Candidate or baseline Reaction evaluation failed for ${split}" >&2
      exit 1
    fi
  fi

  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/compare_hy273_reaction_matched.py \
    --baseline "${baseline_report}" \
    --candidate "${report}" \
    --baseline_label "reaction_v5_${STEP}" \
    --candidate_label "reaction_v5_1_${STEP}" \
    --baseline_predictions "${baseline_predictions}" \
    --candidate_predictions "${split_root}/predictions" \
    --training_contract reaction_v5_1_full_contact \
    --expected_checkpoint_step "${STEP}" \
    --expected_split "${split}" \
    --bootstrap_resamples 10000 \
    --seed 20260806 \
    --output "${split_root}/matched_v5_vs_v5_1_${STEP}.json" \
    >"${EVAL_ROOT}/logs/matched_v5_vs_v5_1_${split}.log" 2>&1

  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
    --report_json "${report}" \
    --prediction_dir "${split_root}/predictions" \
    --output_dir "${split_root}/gifs_action_balanced" \
    --max_videos 16 \
    --joint_source fk \
    --fps 30 \
    --stride 3 \
    >"${EVAL_ROOT}/logs/render_${split}.log" 2>&1

  if [[ "${split}" == "val" ]]; then
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
      --report_json "${report}" \
      --prediction_dir "${split_root}/predictions" \
      --output_dir "${split_root}/gifs_focus_cases" \
      --max_videos 7 \
      --uids "${FOCUS_UIDS}" \
      --joint_source fk \
      --fps 30 \
      --stride 3 \
      >"${EVAL_ROOT}/logs/render_focus_${split}.log" 2>&1

    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
      --report_json "${baseline_report}" \
      --prediction_dir "${baseline_predictions}" \
      --output_dir "${EVAL_ROOT}/baseline_v5/reaction/${split}/gifs_focus_cases" \
      --max_videos 7 \
      --uids "${FOCUS_UIDS}" \
      --joint_source fk \
      --fps 30 \
      --stride 3 \
      >"${EVAL_ROOT}/logs/render_baseline_focus_${split}.log" 2>&1
  fi
done

echo "Reaction-v5.1 evaluation complete: ${EVAL_ROOT}"
