#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 <physical_gpu> <group_name> <comma_separated_strategies>" >&2
  exit 2
fi

GPU="$1"
GROUP="$2"
STRATEGIES="$3"

cd /mnt/afs/mogeflow-control

RUN_DIR="checkpoints/t2m/kv_control_adapter_encoder_target_baseonly_ddp4_20260705"
CHECKPOINT="${RUN_DIR}/model/strategy_sweep_snapshot_epoch2370_step00480375_20260708.pt"
OUT_DIR="${RUN_DIR}/logs/strategy_sweep_epoch2370_${GROUP}_20260708"
LOG="${RUN_DIR}/logs/strategy_sweep_epoch2370_${GROUP}_20260708.log"

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
  --preset broad \
  --strategies "${STRATEGIES}" \
  --out_dir "${OUT_DIR}" \
  --steps 32 \
  --save_npz \
  2>&1 | tee "${LOG}"
