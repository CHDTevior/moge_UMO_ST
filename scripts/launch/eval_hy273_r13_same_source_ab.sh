#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARENT="${PARENT:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/model/step_00400000.pt}"
HINGE="${HINGE:?Set HINGE to the 405K hinge checkpoint}"
SOFTPLUS="${SOFTPLUS:?Set SOFTPLUS to the 405K softplus checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_multitask/diagnostics/r13_same_source_ab_eval}"
DEVICE="${DEVICE:-cuda:0}"

for path in "${PARENT}" "${HINGE}" "${SOFTPLUS}"; do
  [[ -f "${path}" ]] || { echo "Missing checkpoint: ${path}" >&2; exit 2; }
done
mkdir -p "${OUTPUT_DIR}"

run_fixed_t() {
  local weight_source="$1"
  local ode_steps="$2"
  /root/miniconda3/envs/mogo/bin/python \
    tools/eval_hy273_edit_same_source_fixed_t.py \
    --checkpoint "parent400k=${PARENT}" \
    --checkpoint "hinge405k=${HINGE}" \
    --checkpoint "softplus405k=${SOFTPLUS}" \
    --system_expectation "parent400k=400000,none" \
    --system_expectation "hinge405k=405000,same_source_hinge_only" \
    --system_expectation "softplus405k=405000,same_source_softplus_only" \
    --weight_source "${weight_source}" \
    --timesteps 0,0.05,0.1 \
    --ode_steps "${ode_steps}" \
    --ode_groups_per_batch "${ODE_GROUPS_PER_BATCH:-1}" \
    --source_cfg_scale 1.0 \
    --edit_cfg_scale 1.0 \
    --direct_comparison "hinge405k,softplus405k" \
    --device "${DEVICE}" \
    --output "${OUTPUT_DIR}/fixed_t_${weight_source}.json"
}

# The short pilot is selected on raw weights. ODE32 is a raw-primary endpoint;
# EMA is reported only as a fixed-t support result because the pilot is short.
run_fixed_t model 32
run_fixed_t ema 0

if [[ "${RUN_GUARDRAILS:-1}" == "1" ]]; then
  PARENT="${PARENT}" \
  HINGE="${HINGE}" \
  SOFTPLUS="${SOFTPLUS}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  T2M_DEVICE="${DEVICE}" \
  GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  CONTROL_CASES_PER_SUBTYPE="${CONTROL_CASES_PER_SUBTYPE:-32}" \
    bash scripts/launch/eval_hy273_r13_same_source_ab_guardrails.sh
fi

echo "evaluation complete: ${OUTPUT_DIR}"
