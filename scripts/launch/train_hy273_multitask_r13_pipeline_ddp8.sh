#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_multitask_r13_unified273_ddp8_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_multitask}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

if [[ -e "${OUTPUT_DIR}/${RUN_NAME}" ]]; then
  echo "Fresh R13 pipeline refuses existing run directory: ${OUTPUT_DIR}/${RUN_NAME}" >&2
  exit 2
fi

echo "R13 pipeline run=${RUN_NAME} stages=A:200K B1:50K B2:150K C1:50K C2:50K"
for stage in a b1 b2 c1 c2; do
  echo "[pipeline] starting stage=${stage} at $(date --iso-8601=seconds)"
  STAGE="${stage}" RUN_NAME="${RUN_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" GPU_IDS="${GPU_IDS}" \
    bash scripts/launch/train_hy273_multitask_r13_stage_ddp8.sh
  echo "[pipeline] completed stage=${stage} at $(date --iso-8601=seconds)"
done

echo "[pipeline] complete run=${RUN_NAME} checkpoint=${OUTPUT_DIR}/${RUN_NAME}/model/step_00500000.pt"
