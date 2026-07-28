#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs/mogeflow-control"
RUN_DIR="${ROOT}/checkpoints/t2m/kv_control_adapter_encoder_target_baseonly_ddp4_20260705"
MODEL_DIR="${RUN_DIR}/model"
LOG_DIR="${RUN_DIR}/logs"
PY="/mnt/afs/conda_path/envs/codeflow/bin/python"

MEM_LIMIT_MB="${MEM_LIMIT_MB:-500}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-120}"
B_MAX_SAMPLES="${B_MAX_SAMPLES:-64}"
C_MAX_SAMPLES="${C_MAX_SAMPLES:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
HOST_TAG="$(hostname -s 2>/dev/null || hostname)"
REQUEST_ID="${REQUEST_ID:-20260706_control_eval_encoder_latest}"
QUEUE_LOG="${LOG_DIR}/control_eval_wait_empty_encoder_latest_${HOST_TAG}_${STAMP}.log"
LOCK_DIR="${LOG_DIR}/.control_eval_encoder_latest_${REQUEST_ID}.lock"
DONE_MARKER="${LOG_DIR}/.control_eval_encoder_latest_${REQUEST_ID}.done"

choose_empty_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="${MEM_LIMIT_MB}" '
        {
          idx=$1; mem=$2;
          gsub(/ /, "", idx); gsub(/ /, "", mem);
          if ((mem + 0) < limit) { print idx; exit }
        }'
}

{
  echo "[wait] root=${ROOT}"
  echo "[wait] host=${HOST_TAG} request_id=${REQUEST_ID}"
  echo "[wait] mem_limit_mb=${MEM_LIMIT_MB} check_interval_sec=${CHECK_INTERVAL_SEC}"
  echo "[wait] b_max_samples=${B_MAX_SAMPLES} c_max_samples=${C_MAX_SAMPLES}"

  while true; do
    if [[ -f "${DONE_MARKER}" ]]; then
      echo "[done] marker already exists: ${DONE_MARKER}"
      exit 0
    fi
    GPU_ID="$(choose_empty_gpu || true)"
    if [[ -n "${GPU_ID}" ]]; then
      if mkdir "${LOCK_DIR}" 2>/dev/null; then
        trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
        echo "[wait] found empty gpu=${GPU_ID} and acquired lock at $(date -Is)"
        break
      fi
      echo "[wait] found gpu=${GPU_ID}, but another watcher holds lock: ${LOCK_DIR}"
    fi
    echo "[wait] no empty gpu at $(date -Is)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep "${CHECK_INTERVAL_SEC}"
  done

  META="${MODEL_DIR}/latest.pt.meta.json"
  EPOCH="$(sed -n 's/.*"epoch": \([0-9][0-9]*\).*/\1/p' "${META}" | head -1)"
  STEP="$(sed -n 's/.*"step": \([0-9][0-9]*\).*/\1/p' "${META}" | head -1)"
  SNAP="${MODEL_DIR}/eval_snapshot_epoch$(printf "%04d" "${EPOCH}")_step$(printf "%08d" "${STEP}")_${STAMP}.pt"

  echo "[snapshot] latest epoch=${EPOCH} step=${STEP} -> ${SNAP}"
  if ! ln "${MODEL_DIR}/latest.pt" "${SNAP}"; then
    cp --reflink=auto "${MODEL_DIR}/latest.pt" "${SNAP}"
  fi
  cp "${META}" "${SNAP}.meta.json"

  COMMON_ARGS=(
    eval_codeflow_kv_control.py
    --checkpoint "${SNAP}"
    --eval_dir "${LOG_DIR}"
    --gpu_id 0
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --steps 32
    --cond_scale 3.0
    --repeat_times 1
    --seed 3407
    --decode_mode continuous
    --min_keyframes 1
    --max_keyframes 5
    --min_joints 1
    --max_joints 6
    --control_dropout_prob 0.0
  )

  echo "[eval] B-only start at $(date -Is)"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" "${COMMON_ARGS[@]}" \
    --max_samples "${B_MAX_SAMPLES}" \
    --guidance_mode none \
    --save_json_name "control_eval_encoder_latest_s32_Bonly_${B_MAX_SAMPLES}_${STAMP}.json"

  echo "[eval] B+C clean-guidance start at $(date -Is)"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" "${COMMON_ARGS[@]}" \
    --max_samples "${C_MAX_SAMPLES}" \
    --guidance_mode gradient \
    --guidance_variable clean \
    --guidance_optimizer adamw \
    --guidance_eta 0.08 \
    --guidance_total_iters 1000 \
    --guidance_iter_schedule linear_increase \
    --guidance_eta_schedule constant \
    --guidance_start 0.0 \
    --guidance_end 1.0 \
    --guidance_loss l2 \
    --guidance_grad_clip 0.0 \
    --save_json_name "control_eval_encoder_latest_s32_BplusC_clean_eta008_total1000_increase_l2_${C_MAX_SAMPLES}_${STAMP}.json"

  echo "[done] finished at $(date -Is)"
  touch "${DONE_MARKER}"
} 2>&1 | tee -a "${QUEUE_LOG}"
