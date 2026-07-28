#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
PARENT="${PARENT:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/model/step_00400000.pt}"
POSITIVE="${POSITIVE:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r13_decompcfg_no_rank_positive_only_ddp4_400k_to450k_20260723_motionfix_decompcfg/model/step_00450000.pt}"
TEMPORAL="${TEMPORAL:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r14_physical_temporal_positive_ddp4_400k_to450k_20260723_150740/model/step_00450000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_multitask/diagnostics/r14_dynamic_edit_450k_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-4}"

mkdir -p "${OUTPUT_DIR}/logs"
"${PYTHON_BIN}" tools/eval_hy273_dynamic_edits.py \
  --mode prepare \
  --output_dir "${OUTPUT_DIR}" \
  --limit_per_category 16 \
  --seed 20260724

pids=()
launch_worker() {
  local gpu="$1"
  local label="$2"
  local checkpoint="$3"
  local shard_id="$4"
  local num_shards="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" tools/eval_hy273_dynamic_edits.py \
    --mode run \
    --checkpoint "${label}=${checkpoint}" \
    --device cuda:0 \
    --shard_id "${shard_id}" \
    --num_shards "${num_shards}" \
    --batch_size "${BATCH_SIZE}" \
    --weight_source model \
    --ode_steps 32 \
    --source_cfg_scale 2.0 \
    --edit_cfg_scale 2.0 \
    --seed 20260724 \
    --overwrite \
    --output_dir "${OUTPUT_DIR}" \
    >"${OUTPUT_DIR}/logs/${label}_shard${shard_id}.log" 2>&1 &
  pids+=("$!")
}

launch_worker 0 parent400k "${PARENT}" 0 3
launch_worker 1 parent400k "${PARENT}" 1 3
launch_worker 2 parent400k "${PARENT}" 2 3
launch_worker 3 positive450k "${POSITIVE}" 0 2
launch_worker 4 positive450k "${POSITIVE}" 1 2
launch_worker 5 temporal450k "${TEMPORAL}" 0 3
launch_worker 6 temporal450k "${TEMPORAL}" 1 3
launch_worker 7 temporal450k "${TEMPORAL}" 2 3

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "One or more dynamic Edit workers failed; inspect ${OUTPUT_DIR}/logs" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" tools/eval_hy273_dynamic_edits.py \
  --mode aggregate \
  --output_dir "${OUTPUT_DIR}" \
  --labels parent400k,positive450k,temporal450k \
  --render_per_category 3 \
  --render_stride 3

echo "dynamic Edit evaluation complete: ${OUTPUT_DIR}"
