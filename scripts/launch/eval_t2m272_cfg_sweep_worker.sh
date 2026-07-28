#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <gpu_id> <sweep_root> <cfg1> [cfg2 ...]" >&2
  exit 2
fi

GPU_ID="$1"
SWEEP_ROOT="$2"
shift 2

PYTHON="${PYTHON:-/mnt/afs/conda_path/envs/codeflow/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/afs/mogeflow-umo}"
RUN_NAME="${RUN_NAME:-codeflow_t2m_272_rvq1024_bestfid_4gpu_bz16_eval25_20260627_180755}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/checkpoints/t2m/${RUN_NAME}}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model/best_fid.pt}"

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-96}"
SEED="${SEED:-42}"
METRIC_SET="${METRIC_SET:-fid_top3}"
DECODE_MODE="${DECODE_MODE:-nearest}"

mkdir -p "${SWEEP_ROOT}/logs"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

for CFG in "$@"; do
  CFG_FMT="$(printf '%.1f' "${CFG}")"
  CFG_SLUG="${CFG_FMT/./p}"
  EVAL_DIR="${SWEEP_ROOT}/cfg_${CFG_SLUG}"
  LOG_FILE="${SWEEP_ROOT}/logs/gpu${GPU_ID}_cfg_${CFG_SLUG}.log"
  mkdir -p "${EVAL_DIR}"

  {
    echo "CFG_SWEEP_START gpu=${GPU_ID} cfg=${CFG_FMT} checkpoint=${CHECKPOINT} eval_dir=${EVAL_DIR} time=$(date -Is)"
    "${PYTHON}" eval_codeflow_t2m_motionstreamer272.py \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${RUN_DIR}" \
      --eval_dir "${EVAL_DIR}" \
      --gpu_id 0 \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --steps "${STEPS}" \
      --cond_scale "${CFG_FMT}" \
      --repeat_times 1 \
      --seed "${SEED}" \
      --metric_set "${METRIC_SET}" \
      --decode_mode "${DECODE_MODE}" \
      --best_checkpoint_limit 0
    echo "CFG_SWEEP_DONE gpu=${GPU_ID} cfg=${CFG_FMT} time=$(date -Is)"
  } 2>&1 | tee "${LOG_FILE}"
done
