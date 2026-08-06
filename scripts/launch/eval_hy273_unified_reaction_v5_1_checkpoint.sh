#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to a Reaction-v5.1 run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v5_1_full_contact}"
STEP="${STEP:-150000}"
SPLITS="${SPLITS:-val}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/model/$(printf 'step_%08d.pt' "${STEP}")}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_v5_1_${STEP}}"

[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing Reaction-v5.1 checkpoint: ${CHECKPOINT}" >&2
  exit 2
}
mkdir -p "${EVAL_ROOT}/logs"

IFS=',' read -r -a REQUESTED_SPLITS <<< "${SPLITS}"
for split in "${REQUESTED_SPLITS[@]}"; do
  [[ "${split}" == "val" || "${split}" == "test" ]] || {
    echo "SPLITS accepts val,test" >&2
    exit 2
  }
  split_root="${EVAL_ROOT}/reaction/${split}"
  report="${split_root}/reaction_${split}.json"
  mkdir -p "${split_root}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" tools/eval_hy273_reaction.py \
    --checkpoint "${CHECKPOINT}" \
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
    >"${EVAL_ROOT}/logs/reaction_${split}.log" 2>&1

  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
    --report_json "${report}" \
    --prediction_dir "${split_root}/predictions" \
    --output_dir "${split_root}/gifs_action_balanced" \
    --max_videos 16 \
    --joint_source fk \
    --fps 30 \
    --stride 3 \
    >"${EVAL_ROOT}/logs/render_${split}.log" 2>&1
done

echo "Reaction-v5.1 evaluation complete: ${EVAL_ROOT}"
