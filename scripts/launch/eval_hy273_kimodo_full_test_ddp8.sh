#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/mnt/afs/mogeflow-control}"
PYTHON="${PYTHON:-/root/miniconda3/envs/mogo/bin/python}"
CHECKPOINT="${CHECKPOINT:-/mnt/afs/mogeflow-control/checkpoints/t2m/hy273_redenoise_kimodo_complete_stage2_control_ddp8_20260713_0547/model/step_00400000.pt}"
RUN_NAME="${RUN_NAME:-hy273_step400k_hml3d_kimodo_full_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/eval_runs/${RUN_NAME}}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_STEPS="${NUM_STEPS:-32}"
CFG_SCALE="${CFG_SCALE:-2.0}"
CONTROL_CFG_SCALE="${CONTROL_CFG_SCALE:-2.0}"
ASSIGNMENT="${ASSIGNMENT:-balanced_partition}"
CASES_PER_SUBTYPE="${CASES_PER_SUBTYPE:-0}"
MAX_SPARSE_KEYFRAMES="${MAX_SPARSE_KEYFRAMES:-20}"
MIN_FRAMES="${MIN_FRAMES:-2}"
SEED="${SEED:-3407}"
CAPTION_POLICY="${CAPTION_POLICY:-first_full_motion}"
EXPECTED_DATASET_SIZE="${EXPECTED_DATASET_SIZE:-4042}"
if [[ -z "${EXPECTED_CASE_COUNT:-}" ]]; then
  if (( CASES_PER_SUBTYPE == 0 )); then
    EXPECTED_CASE_COUNT=8084
  else
    EXPECTED_CASE_COUNT=$((CASES_PER_SUBTYPE * 13 * 2))
  fi
fi
ASSET_VERIFICATION_CACHE="${ASSET_VERIFICATION_CACHE:-/dev/shm/hy273_asset_verification_ff8da22b41f440931c35a9c1.json}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-499}"
MAX_GPU_UTILIZATION_PERCENT="${MAX_GPU_UTILIZATION_PERCENT:-5}"
ATTEMPT_ID="${ATTEMPT_ID:-$(date +%Y%m%d_%H%M%S)_pid$$}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
NUM_SHARDS="${#GPUS[@]}"
if (( NUM_SHARDS < 1 )); then
  echo "No GPUs specified" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/launch_attempts"
ATTEMPT_DIR="${OUTPUT_DIR}/launch_attempts/${ATTEMPT_ID}"
if [[ -e "${ATTEMPT_DIR}" ]]; then
  echo "Launch attempt already exists: ${ATTEMPT_DIR}" >&2
  exit 1
fi
mkdir -p "${ATTEMPT_DIR}/logs" "${ATTEMPT_DIR}/pids"
printf '%s\n' "${ATTEMPT_DIR}" > "${OUTPUT_DIR}/latest_attempt.txt"
GPU_INVENTORY_BEFORE_PREFLIGHT="${ATTEMPT_DIR}/gpu_inventory_before_preflight.json"
GPU_INVENTORY_BEFORE_LAUNCH="${ATTEMPT_DIR}/gpu_inventory_before_launch.json"
"${PYTHON}" "${ROOT_DIR}/tools/validate_gpu_inventory.py" \
  --gpu-list "${GPU_LIST}" \
  --phase before_preflight \
  --output "${GPU_INVENTORY_BEFORE_PREFLIGHT}" \
  --max-memory-used-mib "${MAX_GPU_MEMORY_USED_MIB}" \
  --max-utilization-percent "${MAX_GPU_UTILIZATION_PERCENT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf '%s\n' \
  "output_dir=${OUTPUT_DIR}" \
  "attempt_id=${ATTEMPT_ID}" \
  "attempt_dir=${ATTEMPT_DIR}" \
  "checkpoint=${CHECKPOINT}" \
  "gpu_list=${GPU_LIST}" \
  "num_shards=${NUM_SHARDS}" \
  "num_steps=${NUM_STEPS}" \
  "cfg_scale=${CFG_SCALE}" \
  "control_cfg_scale=${CONTROL_CFG_SCALE}" \
  "assignment=${ASSIGNMENT}" \
  "cases_per_subtype=${CASES_PER_SUBTYPE}" \
  "min_frames=${MIN_FRAMES}" \
  "seed=${SEED}" \
  "caption_policy=${CAPTION_POLICY}" \
  "expected_dataset_size=${EXPECTED_DATASET_SIZE}" \
  "expected_case_count=${EXPECTED_CASE_COUNT}" \
  "asset_verification_cache=${ASSET_VERIFICATION_CACHE}" \
  "gpu_inventory_before_preflight=${GPU_INVENTORY_BEFORE_PREFLIGHT}" \
  "gpu_inventory_before_launch=${GPU_INVENTORY_BEFORE_LAUNCH}" \
  "max_gpu_memory_used_mib=${MAX_GPU_MEMORY_USED_MIB}" \
  "max_gpu_utilization_percent=${MAX_GPU_UTILIZATION_PERCENT}" \
  > "${ATTEMPT_DIR}/launch_config.txt"

PREFLIGHT_MANIFEST="${OUTPUT_DIR}/preflight_manifest.json"
"${PYTHON}" -u "${ROOT_DIR}/eval_hy273_kimodo_full_test.py" \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --assignment "${ASSIGNMENT}" \
  --cases_per_subtype "${CASES_PER_SUBTYPE}" \
  --min_frames "${MIN_FRAMES}" \
  --max_sparse_keyframes "${MAX_SPARSE_KEYFRAMES}" \
  --seed "${SEED}" \
  --caption_policy "${CAPTION_POLICY}" \
  --expected_dataset_size "${EXPECTED_DATASET_SIZE}" \
  --expected_case_count "${EXPECTED_CASE_COUNT}" \
  --preflight_only \
  | tee "${ATTEMPT_DIR}/logs/preflight.log"

# The content-addressed preflight can take over a minute. Re-resolve selectors to
# canonical UUIDs and certify idleness again immediately before worker launch.
"${PYTHON}" "${ROOT_DIR}/tools/validate_gpu_inventory.py" \
  --gpu-list "${GPU_LIST}" \
  --phase before_launch \
  --output "${GPU_INVENTORY_BEFORE_LAUNCH}" \
  --max-memory-used-mib "${MAX_GPU_MEMORY_USED_MIB}" \
  --max-utilization-percent "${MAX_GPU_UTILIZATION_PERCENT}"

pids=()
for shard_id in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_id]}"
  log_path="${ATTEMPT_DIR}/logs/shard_$(printf '%02d' "${shard_id}").log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT_DIR}/eval_hy273_kimodo_full_test.py" \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --shard_id "${shard_id}" \
    --num_shards "${NUM_SHARDS}" \
    --num_steps "${NUM_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --control_cfg_scale "${CONTROL_CFG_SCALE}" \
    --assignment "${ASSIGNMENT}" \
    --cases_per_subtype "${CASES_PER_SUBTYPE}" \
    --min_frames "${MIN_FRAMES}" \
    --max_sparse_keyframes "${MAX_SPARSE_KEYFRAMES}" \
    --seed "${SEED}" \
    --caption_policy "${CAPTION_POLICY}" \
    --expected_dataset_size "${EXPECTED_DATASET_SIZE}" \
    --expected_case_count "${EXPECTED_CASE_COUNT}" \
    --preflight_manifest "${PREFLIGHT_MANIFEST}" \
    --asset_verification_cache "${ASSET_VERIFICATION_CACHE}" \
    --gpu_inventory_manifest "${GPU_INVENTORY_BEFORE_LAUNCH}" \
    --batch_size 1 \
    --weight_source ema \
    > "${log_path}" 2>&1 &
  pid=$!
  pids+=("${pid}")
  printf '%s\n' "${pid}" > "${ATTEMPT_DIR}/pids/shard_$(printf '%02d' "${shard_id}").pid"
  echo "launched shard=${shard_id} gpu=${gpu} pid=${pid} log=${log_path}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status == 0 )); then
  "${PYTHON}" "${ROOT_DIR}/eval_hy273_kimodo_full_test.py" \
    --output_dir "${OUTPUT_DIR}" \
    --aggregate_only
else
  "${PYTHON}" "${ROOT_DIR}/eval_hy273_kimodo_full_test.py" \
    --output_dir "${OUTPUT_DIR}" \
    --aggregate_only \
    --allow_incomplete || true
  echo "One or more evaluation shards failed; inspect ${ATTEMPT_DIR}/logs" >&2
  exit 1
fi

echo "Kimodo full-test evaluation complete: ${OUTPUT_DIR}"
