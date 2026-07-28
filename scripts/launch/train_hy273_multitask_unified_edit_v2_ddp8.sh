#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-/root/miniconda3/envs/mogo/bin/python}
CONFIG=${CONFIG:-$ROOT/configs/hy273_multitask_r12_stage_c_unified_edit_v2.yaml}
PARENT=${PARENT:-$ROOT/outputs/hy273_multitask/hy273_multitask_r12_rootmask_b1_ddp8_20260716_200754/model/step_00400000.pt}
NAME=${NAME:-hy273_multitask_unified_edit_v2_from400k_ddp8_$(date +%Y%m%d_%H%M%S)}
MASTER_PORT=${MASTER_PORT:-29651}

cd "$ROOT"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  train_hy273_multitask.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --resume "$PARENT" \
  --fork_stage_c_unified_edit \
  "$@"
