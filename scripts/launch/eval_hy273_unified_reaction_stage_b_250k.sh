#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export STAGE_B_STEP="${STAGE_B_STEP:-250000}"

if [[ "${RUN_NATIVE_TMR_COMPARE:-1}" == "1" ]]; then
  bash "${ROOT_DIR}/scripts/launch/eval_hy273_unified_reaction_native_tmr_200k_vs_250k.sh"
fi

exec bash "${ROOT_DIR}/scripts/launch/eval_hy273_unified_reaction_stage_b_200k.sh"
