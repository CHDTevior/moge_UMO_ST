#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the active Reaction run}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_unified_reaction}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00100000.pt}"
POLL_SECONDS="${POLL_SECONDS:-600}"
STAGE_B_SMOKE_UPDATES="${STAGE_B_SMOKE_UPDATES:-500}"

echo "[advance] waiting for exact Stage-A checkpoint: ${CHECKPOINT}"
while [[ ! -f "${CHECKPOINT}" ]]; do
  if ! pgrep -f "train_hy273_unified_actor.py .*--name ${RUN_NAME} .*--phase_contract fulltext_stage_a" >/dev/null; then
    echo "[advance] Stage A exited before producing ${CHECKPOINT}" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

while pgrep -f "train_hy273_unified_actor.py .*--name ${RUN_NAME} .*--phase_contract fulltext_stage_a" >/dev/null; do
  sleep 10
done

echo "[advance] exact 100K checkpoint ready; launching ${STAGE_B_SMOKE_UPDATES}-update Stage-B smoke"
env \
  RUN_NAME="${RUN_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  CHECKPOINT="${CHECKPOINT}" \
  GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
  MASTER_PORT="${MASTER_PORT:-29795}" \
  MAX_UPDATES="${STAGE_B_SMOKE_UPDATES}" \
  bash scripts/launch/train_hy273_unified_reaction_stage_b_ddp8.sh

SMOKE_CHECKPOINT="${OUTPUT_DIR}/${RUN_NAME}/model/latest.pt"
EXPECTED_STEP=$((100000 + STAGE_B_SMOKE_UPDATES))
ACTUAL_STEP="$(/root/miniconda3/envs/mogo/bin/python - "${SMOKE_CHECKPOINT}" <<'PY'
import sys
import torch

checkpoint = torch.load(
    sys.argv[1], map_location="cpu", mmap=True, weights_only=False
)
print(int(checkpoint.get("next_global_step", -1)))
PY
)"
[[ "${ACTUAL_STEP}" == "${EXPECTED_STEP}" ]] || {
  echo "[advance] Stage-B smoke checkpoint step mismatch: expected ${EXPECTED_STEP}, got ${ACTUAL_STEP}" >&2
  exit 1
}
echo "[advance] Stage-B smoke complete at ${ACTUAL_STEP}; inspect Reaction gradient ratios before continuation"
