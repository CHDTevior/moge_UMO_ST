#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a unified Reaction checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for the fixed control gate}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TASKS="${TASKS:-t2m,edit,reaction}"
CASES_PER_SUBTYPE="${CASES_PER_SUBTYPE:-8}"
NUM_STEPS="${NUM_STEPS:-32}"
WEIGHT_SOURCE="${WEIGHT_SOURCE:-ema}"
TEXT_CFG="${TEXT_CFG:-2.0}"
SOURCE_CFG="${SOURCE_CFG:-2.0}"
EDIT_CFG="${EDIT_CFG:-2.0}"
CONTROL_CFG="${CONTROL_CFG:-2.0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"
if (( NUM_SHARDS < 1 )); then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs"
for task in "${TASK_ARRAY[@]}"; do
  pids=()
  for shard_id in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[${shard_id}]}"
    log="${OUTPUT_DIR}/logs/${task}_shard_${shard_id}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
      tools/eval_hy273_unified_task_control.py \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${OUTPUT_DIR}" \
      --task "${task}" \
      --device cuda:0 \
      --weight_source "${WEIGHT_SOURCE}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_id "${shard_id}" \
      --cases_per_subtype "${CASES_PER_SUBTYPE}" \
      --num_steps "${NUM_STEPS}" \
      --text_cfg_scale "${TEXT_CFG}" \
      --source_cfg_scale "${SOURCE_CFG}" \
      --edit_cfg_scale "${EDIT_CFG}" \
      --control_cfg_scale "${CONTROL_CFG}" \
      >"${log}" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if (( failed )); then
    echo "Control gate failed for task=${task}; inspect ${OUTPUT_DIR}/logs" >&2
    exit 1
  fi
  "${PYTHON_BIN}" tools/eval_hy273_unified_task_control.py \
    --output_dir "${OUTPUT_DIR}" --task "${task}" --aggregate
done

echo "Control gate complete: ${OUTPUT_DIR}"
