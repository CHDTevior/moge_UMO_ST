#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/afs/mogeflow-control
PYTHON=${PYTHON:-/root/miniconda3/envs/mogo/bin/python}
CONFIG=${CONFIG:-$ROOT/configs/hy273_context_only_e1.yaml}
NAME=${NAME:-hy273_context_only_edit_identity_from500k_ddp8_$(date +%Y%m%d_%H%M%S)}
STOP_UPDATE=${STOP_UPDATE:-10000}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29631}

cd "$ROOT"
exec "$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  train_hy273_context_only.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --stop_update "$STOP_UPDATE" \
  "$@"
