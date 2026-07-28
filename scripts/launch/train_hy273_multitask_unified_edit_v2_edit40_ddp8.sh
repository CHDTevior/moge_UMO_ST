#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/root/miniconda3/envs/mogo/bin/python}
CONFIG=${CONFIG:-$ROOT/configs/hy273_multitask_r12_stage_c_unified_edit_v2_edit40.yaml}
PARENT=${PARENT:-$ROOT/outputs/hy273_multitask/hy273_multitask_unified_edit_v2_from400k_ddp8_20260719_045221/model/step_00450000.pt}
PARENT_SHA256=${PARENT_SHA256:-$(sha256sum "$PARENT" | awk '{print $1}')}
NAME=${NAME:-hy273_multitask_unified_edit_v2_edit40_from450k_ddp8_$(date +%Y%m%d_%H%M%S)}
MASTER_PORT=${MASTER_PORT:-29651}

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  train_hy273_multitask.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --resume "$PARENT" \
  --resume_sha256 "$PARENT_SHA256" \
  --fork_stage_c_unified_edit40 \
  "$@"
