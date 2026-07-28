#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${RESUME:-}" ]]; then
  echo "RESUME must point to a stage1 checkpoint, e.g. RESUME=/path/to/latest.pt" >&2
  exit 2
fi

if [[ ! -f "${RESUME}" ]]; then
  echo "Stage1 checkpoint not found: ${RESUME}" >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-hy273_raw_flow_hml3d_stage2_x0_control_sem_ddp8_$(date +%Y%m%d_%H%M%S)}"
export MASTER_PORT="${MASTER_PORT:-29692}"
export HYTEXT_CACHE_DIR="${HYTEXT_CACHE_DIR:-/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128}"
export CONFIG="${CONFIG:-configs/raw_flow_hy273_hytext.yaml}"
export MAX_EPOCHS="${MAX_EPOCHS:-10000}"

if [[ ! -f "${HYTEXT_CACHE_DIR}/manifest.json" ]]; then
  echo "HYText cache manifest not found: ${HYTEXT_CACHE_DIR}/manifest.json" >&2
  exit 2
fi

"${SCRIPT_DIR}/train_hy273_raw_flow_ddp8.sh" \
  --resume "${RESUME}" \
  --prediction_type x0 \
  --text_encoder hy_cache \
  --max_text_tokens "${MAX_TEXT_TOKENS:-128}" \
  --hytext_cache_dir "${HYTEXT_CACHE_DIR}" \
  --control_modes "${CONTROL_MODES:-none,root,endpoints,fullpose,mixed}" \
  --endpoint_preset "${ENDPOINT_PRESET:-kimodo_ee}" \
  --endpoint_subset_mode "${ENDPOINT_SUBSET_MODE:-random_nonempty}" \
  --endpoint_root_ref_mode "${ENDPOINT_ROOT_REF_MODE:-kimodo_hidden_root}" \
  --flow_loss_weight "${FLOW_LOSS_WEIGHT:-1.0}" \
  --contact_loss_weight "${CONTACT_LOSS_WEIGHT:-0.1}" \
  --control_cont_loss_weight "${CONTROL_CONT_LOSS_WEIGHT:-0.25}" \
  --control_contact_loss_weight "${CONTROL_CONTACT_LOSS_WEIGHT:-0.05}" \
  --clean_cont_loss_weight 0.0 \
  --clean_root_vel_loss_weight "${CLEAN_ROOT_VEL_LOSS_WEIGHT:-0.01}" \
  --clean_joint_vel_loss_weight "${CLEAN_JOINT_VEL_LOSS_WEIGHT:-0.01}" \
  --foot_lock_loss_weight "${FOOT_LOCK_LOSS_WEIGHT:-0.01}" \
  --semantic_loss_fps "${SEMANTIC_LOSS_FPS:-30.0}" \
  --foot_lock_contact_threshold "${FOOT_LOCK_CONTACT_THRESHOLD:-0.5}" \
  --ema \
  --ema_decay "${EMA_DECAY:-0.995}" \
  --ema_every "${EMA_EVERY:-10}" \
  --max_steps "${MAX_STEPS:-1500000}" \
  --save_every "${SAVE_EVERY:-50000}" \
  "$@"
