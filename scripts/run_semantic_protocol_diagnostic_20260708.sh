#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <physical_gpu> <control_profile>" >&2
  exit 2
fi

GPU="$1"
PROFILE="$2"

cd /mnt/afs/mogeflow-control

RUN_DIR="checkpoints/t2m/kv_control_adapter_encoder_target_baseonly_ddp4_20260705"
CHECKPOINT="${RUN_DIR}/model/stopped_failed_random_joint_protocol_epoch2392_step00484401_20260708.pt"
OUT_DIR="${RUN_DIR}/logs/semantic_protocol_diag_${PROFILE}_epoch2392_20260708"
LOG="${RUN_DIR}/logs/semantic_protocol_diag_${PROFILE}_epoch2392_20260708.log"

mkdir -p "${RUN_DIR}/logs"

env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  /mnt/afs/conda_path/envs/codeflow/bin/python \
  tools/sweep_kv_control_strategies.py \
  --checkpoint "${CHECKPOINT}" \
  --gpu_id 0 \
  --num_samples 16 \
  --sample_seed 3407 \
  --control_seed 93407 \
  --noise_seed 13407 \
  --control_profile "${PROFILE}" \
  --control_keyframe_strategy uniform \
  --min_keyframes 5 \
  --max_keyframes 5 \
  --preset broad \
  --strategies B_only,clean_eta006_total1000_late_l2,clean_eta006_total1000_inc_l2_strong_footsafe \
  --out_dir "${OUT_DIR}" \
  --steps 32 \
  --save_npz \
  2>&1 | tee "${LOG}"
