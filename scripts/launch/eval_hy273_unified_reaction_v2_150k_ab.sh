#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

V1_OUTPUT_ROOT="${V1_OUTPUT_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction}"
V1_RUN_NAME="${V1_RUN_NAME:-hy273_unified_fulltext_reaction_v1_20260801_0315}"
V2_OUTPUT_ROOT="${V2_OUTPUT_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction_v2}"
V2_RUN_NAME="${V2_RUN_NAME:-hy273_unified_reaction_v2_20260803_082413}"
V1_RUN_ROOT="${V1_OUTPUT_ROOT}/${V1_RUN_NAME}"
V2_RUN_ROOT="${V2_OUTPUT_ROOT}/${V2_RUN_NAME}"
V1_STAGE_A="${V1_STAGE_A:-${V1_RUN_ROOT}/model/step_00100000.pt}"
V1_STAGE_B="${V1_STAGE_B:-${V1_RUN_ROOT}/model/step_00150000.pt}"
V2_STAGE_B="${V2_STAGE_B:-${V2_RUN_ROOT}/model/step_00150000.pt}"
AB_ROOT="${AB_ROOT:-${V2_RUN_ROOT}/eval_reaction_v2_150k_ab}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
EVAL_TARGET="${EVAL_TARGET:-both}"
EVAL_PHASE="${EVAL_PHASE:-all}"

for checkpoint in "${V1_STAGE_A}" "${V1_STAGE_B}" "${V2_STAGE_B}"; do
  [[ -f "${checkpoint}" ]] || {
    echo "Missing 150K Reaction-v2 A/B checkpoint: ${checkpoint}" >&2
    exit 2
  }
done
[[ "${EVAL_TARGET}" == "baseline" || "${EVAL_TARGET}" == "candidate" || "${EVAL_TARGET}" == "both" ]] || {
  echo "EVAL_TARGET must be baseline, candidate, or both" >&2
  exit 2
}

run_baseline() {
  RUN_NAME="${V1_RUN_NAME}" \
  OUTPUT_ROOT="${V1_OUTPUT_ROOT}" \
  STAGE_B_STEP=150000 \
  STAGE_A_CHECKPOINT="${V1_STAGE_A}" \
  STAGE_B_CHECKPOINT="${V1_STAGE_B}" \
  EVAL_ROOT="${AB_ROOT}/v1_150k" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_PHASE="${EVAL_PHASE}" \
  ALLOW_REACTION_LOSS_ABLATION=0 \
    bash scripts/launch/eval_hy273_unified_reaction_stage_b_200k.sh
}

run_candidate() {
  RUN_NAME="${V2_RUN_NAME}" \
  OUTPUT_ROOT="${V2_OUTPUT_ROOT}" \
  STAGE_B_STEP=150000 \
  STAGE_A_CHECKPOINT="${V1_STAGE_A}" \
  STAGE_B_CHECKPOINT="${V2_STAGE_B}" \
  EVAL_ROOT="${AB_ROOT}/v2_150k" \
  GPU_IDS="${GPU_IDS}" \
  EVAL_PHASE="${EVAL_PHASE}" \
  ALLOW_REACTION_LOSS_ABLATION=1 \
    bash scripts/launch/eval_hy273_unified_reaction_stage_b_200k.sh
}

if [[ "${EVAL_TARGET}" == "baseline" || "${EVAL_TARGET}" == "both" ]]; then
  run_baseline
fi
if [[ "${EVAL_TARGET}" == "candidate" || "${EVAL_TARGET}" == "both" ]]; then
  run_candidate
fi

echo "Reaction-v2 150K A/B evaluation complete: ${AB_ROOT}"
