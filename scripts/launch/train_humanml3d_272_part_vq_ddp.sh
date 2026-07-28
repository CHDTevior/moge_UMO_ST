#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/run_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_humanml3d_272_part_vq_ddp_${TIMESTAMP}.log}"

if [[ "${DETACH:-0}" == "1" ]]; then
  DETACH=0 LOG_FILE="${LOG_FILE}" LOG_DIR="${LOG_DIR}" setsid "$0" "$@" > "${LOG_FILE}" 2>&1 < /dev/null &
  PID="$!"
  echo "$PID" > "${LOG_FILE%.log}.pid"
  echo "PID=$PID"
  echo "LOG=$LOG_FILE"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/conda_path/envs/codeflow/bin/python}"
GPUS="${GPUS:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("${PYTHON_BIN}" - <<PY
print(len([x for x in "${GPUS}".split(",") if x.strip()]))
PY
)}"
MASTER_PORT="${MASTER_PORT:-29631}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset/HumanML3D_272}"
MOTION_DIR="${MOTION_DIR:-${DATA_ROOT}/new_joint_vecs}"
STATS_DIR="${STATS_DIR:-${DATA_ROOT}}"
SPLIT_DIR="${SPLIT_DIR:-${DATA_ROOT}}"
PARTITION_FILE="${PARTITION_FILE:-${REPO_ROOT}/configs/humanml3d_272_skeleton_partition_pscf_nooverlap.json}"

SEED="${SEED:-3407}"
MAX_EPOCH="${MAX_EPOCH:-200}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-32}"
WINDOW_SIZE="${WINDOW_SIZE:-96}"
MIN_MOTION_LENGTH="${MIN_MOTION_LENGTH:-60}"
MAX_MOTION_LENGTH="${MAX_MOTION_LENGTH:-300}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_LATEST="${SAVE_LATEST:-1000}"
PRINT_ITER="${PRINT_ITER:-500}"
LR="${LR:-2e-4}"
WARM_UP_ITER="${WARM_UP_ITER:-1000}"
MILESTONES="${MILESTONES:-200000 1000000 1800000}"
SCALE_MILESTONES="${SCALE_MILESTONES:-1}"
MOTIONMILLION_REFERENCE_TOTAL_ITER="${MOTIONMILLION_REFERENCE_TOTAL_ITER:-2385000}"
GAMMA="${GAMMA:-0.2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
MU="${MU:-0.99}"
COMMIT="${COMMIT:-0.02}"
LOSS_VEL="${LOSS_VEL:-0.5}"
DDP_CODEBOOK_SYNC="${DDP_CODEBOOK_SYNC:-sum}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/vq/humanml3d_272_pscf_nooverlap_w${WINDOW_SIZE}_n1024_d128_ddp${NPROC_PER_NODE}_b${BATCH_SIZE_PER_GPU}_ep${MAX_EPOCH}_seed${SEED}_${TIMESTAMP}}"

read -r -a MILESTONE_VALUES <<< "${MILESTONES}"
MILESTONE_ARGS=(--milestones "${MILESTONE_VALUES[@]}")
if [[ "${SCALE_MILESTONES}" == "1" ]]; then
  MILESTONE_ARGS+=(--scale_motionmillion_milestones)
  MILESTONE_ARGS+=(--motionmillion_reference_total_iter "${MOTIONMILLION_REFERENCE_TOTAL_ITER}")
fi

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
if [[ "${FREEZE_CODEBOOK_UPDATES:-0}" == "1" ]]; then
  RESUME_ARGS+=(--freeze_codebook_updates)
fi

echo "[launch] repo=${REPO_ROOT}"
echo "[launch] gpus=${GPUS} nproc=${NPROC_PER_NODE} master_port=${MASTER_PORT}"
echo "[launch] data_root=${DATA_ROOT}"
echo "[launch] partition=${PARTITION_FILE}"
echo "[launch] output=${OUTPUT_DIR}"
echo "[launch] log=${LOG_FILE}"
echo "[launch] seed=${SEED} max_epoch=${MAX_EPOCH} batch_size_per_gpu=${BATCH_SIZE_PER_GPU}"
echo "[launch] lr=${LR} warm_up_iter=${WARM_UP_ITER} milestones=${MILESTONES} scale_milestones=${SCALE_MILESTONES} gamma=${GAMMA}"
echo "[launch] codebooks=6 nb_code=1024 code_dim=128 ddp_codebook_sync=${DDP_CODEBOOK_SYNC}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  tools/train_humanml3d_272_part_vq_ddp.py \
  --dataset_name humanml3d_272 \
  --data_root "${DATA_ROOT}" \
  --motion_dir "${MOTION_DIR}" \
  --stats_dir "${STATS_DIR}" \
  --split_dir "${SPLIT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --partition_file "${PARTITION_FILE}" \
  --batch_size "${BATCH_SIZE_PER_GPU}" \
  --window_size "${WINDOW_SIZE}" \
  --sampling_mode random_crop \
  --min_motion_length "${MIN_MOTION_LENGTH}" \
  --max_motion_length "${MAX_MOTION_LENGTH}" \
  --num_workers "${NUM_WORKERS}" \
  --max_epoch "${MAX_EPOCH}" \
  --lr "${LR}" \
  --warm_up_iter "${WARM_UP_ITER}" \
  "${MILESTONE_ARGS[@]}" \
  --gamma "${GAMMA}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --mu "${MU}" \
  --commit "${COMMIT}" \
  --loss_vel "${LOSS_VEL}" \
  --print_iter "${PRINT_ITER}" \
  --eval_every_epoch \
  --save_latest "${SAVE_LATEST}" \
  --nb_code 1024 \
  --code_dim 128 \
  --output_emb_width 128 \
  --down_t 2 \
  --stride_t 2 \
  --width 512 \
  --depth 3 \
  --vq_act relu \
  --quantizer ema_reset \
  --ddp_codebook_sync "${DDP_CODEBOOK_SYNC}" \
  --seed "${SEED}" \
  --resume_budget_mode "${RESUME_BUDGET_MODE:-auto}" \
  --resume_ema_code_count "${RESUME_EMA_CODE_COUNT:-1024}" \
  "${RESUME_ARGS[@]}" \
  "$@" 2>&1 | tee "${LOG_FILE}"
