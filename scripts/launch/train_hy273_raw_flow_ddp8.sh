#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29673}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

DATA_ROOT="${DATA_ROOT:-/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22}"
TEXT_ROOT="${TEXT_ROOT:-/mnt/afs/mogo_base/datasets/HumanML3D/texts}"
CLIP_PATH="${CLIP_PATH:-${REPO_ROOT}/checkpoints/clip/ViT-B-32.pt}"
RUN_NAME="${RUN_NAME:-hy273_raw_flow_hml3d_ddp8_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/t2m}"
CONFIG="${CONFIG:-configs/raw_flow_hy273.yaml}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  train_hy273_raw_flow.py \
  --config "${CONFIG}" \
  --name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --data_root "${DATA_ROOT}" \
  --text_root "${TEXT_ROOT}" \
  --clip_path "${CLIP_PATH}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --max_epochs "${MAX_EPOCHS:-4000}" \
  --lr "${LR:-0.0001}" \
  --amp \
  "$@"
