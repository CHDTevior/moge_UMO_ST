#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARENT="${PARENT:?Set PARENT to the 400K checkpoint}"
HINGE="${HINGE:?Set HINGE to the 405K hinge checkpoint}"
SOFTPLUS="${SOFTPLUS:?Set SOFTPLUS to the 405K softplus checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the same-source A/B evaluation directory}"
FIXED_T_RAW="${FIXED_T_RAW:-${OUTPUT_DIR}/fixed_t_model.json}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
T2M_DEVICE="${T2M_DEVICE:-cuda:0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CONTROL_CASES_PER_SUBTYPE="${CONTROL_CASES_PER_SUBTYPE:-32}"

for path in "${PARENT}" "${HINGE}" "${SOFTPLUS}" "${FIXED_T_RAW}"; do
  [[ -f "${path}" ]] || { echo "Missing evaluation input: ${path}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_DIR}/t2m_physical" "${OUTPUT_DIR}/control"

run_t2m() {
  local label="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" tools/eval_hy273_multitask_t2m_visual16.py \
    --checkpoint "${checkpoint}" \
    --output_dir "${OUTPUT_DIR}/t2m_physical/${label}" \
    --device "${T2M_DEVICE}" \
    --weight_source model \
    --num_steps 32 \
    --cfg_scale 2.0 \
    --seed 3407 \
    --max_samples 16
}

run_control() {
  local label="$1"
  local checkpoint="$2"
  CHECKPOINT="${checkpoint}" \
  OUTPUT_DIR="${OUTPUT_DIR}/control/${label}" \
  GPU_IDS="${GPU_IDS}" \
  PROFILE=research \
  WEIGHT_SOURCE=model \
  NUM_STEPS=32 \
  CFG_SCALE=2.0 \
  CONTROL_CFG_SCALE=2.0 \
  SEED=3407 \
  CASES_PER_SUBTYPE="${CONTROL_CASES_PER_SUBTYPE}" \
    bash scripts/launch/eval_hy273_kimodo_v5_contact_8gpu.sh
}

run_t2m parent400k "${PARENT}"
run_t2m hinge405k "${HINGE}"
run_t2m softplus405k "${SOFTPLUS}"

run_control parent400k "${PARENT}"
run_control hinge405k "${HINGE}"
run_control softplus405k "${SOFTPLUS}"

"${PYTHON_BIN}" tools/eval_hy273_r13_same_source_ab_decision.py \
  --fixed_t_raw "${FIXED_T_RAW}" \
  --t2m "parent400k=${OUTPUT_DIR}/t2m_physical/parent400k/quality.json" \
  --t2m "hinge405k=${OUTPUT_DIR}/t2m_physical/hinge405k/quality.json" \
  --t2m "softplus405k=${OUTPUT_DIR}/t2m_physical/softplus405k/quality.json" \
  --control "parent400k=${OUTPUT_DIR}/control/parent400k/summary.json" \
  --control "hinge405k=${OUTPUT_DIR}/control/hinge405k/summary.json" \
  --control "softplus405k=${OUTPUT_DIR}/control/softplus405k/summary.json" \
  --output "${OUTPUT_DIR}/decision.json"

echo "guardrails complete: ${OUTPUT_DIR}/decision.json"
