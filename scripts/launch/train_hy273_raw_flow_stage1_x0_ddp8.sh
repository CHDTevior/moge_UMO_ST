#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export RUN_NAME="${RUN_NAME:-hy273_raw_flow_hml3d_stage1_x0_t2m_sem_ddp8_$(date +%Y%m%d_%H%M%S)}"
export MASTER_PORT="${MASTER_PORT:-29691}"

"${SCRIPT_DIR}/train_hy273_raw_flow_ddp8.sh" \
  --prediction_type x0 \
  --control_modes none \
  --flow_loss_weight "${FLOW_LOSS_WEIGHT:-1.0}" \
  --contact_loss_weight "${CONTACT_LOSS_WEIGHT:-0.1}" \
  --control_cont_loss_weight 0.0 \
  --control_contact_loss_weight 0.0 \
  --clean_cont_loss_weight 0.0 \
  --clean_root_vel_loss_weight "${CLEAN_ROOT_VEL_LOSS_WEIGHT:-0.01}" \
  --clean_joint_vel_loss_weight "${CLEAN_JOINT_VEL_LOSS_WEIGHT:-0.01}" \
  --foot_lock_loss_weight "${FOOT_LOCK_LOSS_WEIGHT:-0.01}" \
  --semantic_loss_fps "${SEMANTIC_LOSS_FPS:-30.0}" \
  --foot_lock_contact_threshold "${FOOT_LOCK_CONTACT_THRESHOLD:-0.5}" \
  --ema \
  --ema_decay "${EMA_DECAY:-0.995}" \
  --ema_every "${EMA_EVERY:-10}" \
  --max_steps "${MAX_STEPS:-500000}" \
  --save_every "${SAVE_EVERY:-50000}" \
  "$@"
