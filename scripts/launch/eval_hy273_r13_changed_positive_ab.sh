#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PARENT="${PARENT:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/model/step_00400000.pt}"
POSITIVE="${POSITIVE:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r13_decompcfg_no_rank_positive_only_ddp4_400k_to450k_20260723_motionfix_decompcfg/model/step_00450000.pt}"
CHANGED="${CHANGED:-${ROOT_DIR}/outputs/hy273_multitask/hy273_r13_decompcfg_same_source_changed_positive_only_ddp4_400k_to450k_20260723_motionfix_decompcfg/model/step_00450000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_multitask/diagnostics/r13_changed_positive_ab_450k_20260723}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CONTROL_CASES_PER_SUBTYPE="${CONTROL_CASES_PER_SUBTYPE:-0}"

for path in "${PARENT}" "${POSITIVE}" "${CHANGED}"; do
  [[ -f "${path}" ]] || { echo "Missing checkpoint: ${path}" >&2; exit 2; }
done
mkdir -p "${OUTPUT_DIR}"/{visual_samples,visuals,t2m_physical,control}

run_fixed_t() {
  local weight_source="$1"
  local ode_steps="$2"
  "${PYTHON_BIN}" tools/eval_hy273_edit_same_source_fixed_t.py \
    --checkpoint "parent400k=${PARENT}" \
    --checkpoint "positive450k=${POSITIVE}" \
    --checkpoint "changed450k=${CHANGED}" \
    --system_expectation "parent400k=400000,none" \
    --system_expectation "positive450k=450000,no_rank_positive_only" \
    --system_expectation "changed450k=450000,same_source_changed_positive_only" \
    --weight_source "${weight_source}" \
    --timesteps 0,0.05,0.1 \
    --ode_steps "${ode_steps}" \
    --ode_groups_per_batch 1 \
    --source_cfg_scale 2.0 \
    --edit_cfg_scale 2.0 \
    --direct_comparison positive450k,changed450k \
    --device "${DEVICE}" \
    --output "${OUTPUT_DIR}/fixed_t_${weight_source}.json"
}

run_edit_visual_samples() {
  local label="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" tools/sample_hy273_edit_same_source_visuals.py \
    --checkpoint "${checkpoint}" \
    --label "${label}" \
    --weight_source model \
    --ode_steps 32 \
    --source_cfg_scale 2.0 \
    --edit_cfg_scale 2.0 \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}/visual_samples/${label}"
}

run_t2m() {
  local label="$1"
  local checkpoint="$2"
  "${PYTHON_BIN}" tools/eval_hy273_multitask_t2m_visual16.py \
    --checkpoint "${checkpoint}" \
    --output_dir "${OUTPUT_DIR}/t2m_physical/${label}" \
    --device "${DEVICE}" \
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

run_fixed_t model 32
run_fixed_t ema 0

run_edit_visual_samples parent400k "${PARENT}"
run_edit_visual_samples positive450k "${POSITIVE}"
run_edit_visual_samples changed450k "${CHANGED}"
"${PYTHON_BIN}" tools/render_hy273_edit_same_source_comparison.py \
  --system "parent400k=${OUTPUT_DIR}/visual_samples/parent400k" \
  --system "positive450k=${OUTPUT_DIR}/visual_samples/positive450k" \
  --system "changed450k=${OUTPUT_DIR}/visual_samples/changed450k" \
  --branch_system changed450k \
  --output_dir "${OUTPUT_DIR}/visuals"

run_t2m parent400k "${PARENT}"
run_t2m positive450k "${POSITIVE}"
run_t2m changed450k "${CHANGED}"

run_control parent400k "${PARENT}"
run_control positive450k "${POSITIVE}"
run_control changed450k "${CHANGED}"

"${PYTHON_BIN}" tools/eval_hy273_r13_same_source_ab_decision.py \
  --fixed_t_raw "${OUTPUT_DIR}/fixed_t_model.json" \
  --t2m "parent400k=${OUTPUT_DIR}/t2m_physical/parent400k/quality.json" \
  --t2m "positive450k=${OUTPUT_DIR}/t2m_physical/positive450k/quality.json" \
  --t2m "changed450k=${OUTPUT_DIR}/t2m_physical/changed450k/quality.json" \
  --control "parent400k=${OUTPUT_DIR}/control/parent400k/summary.json" \
  --control "positive450k=${OUTPUT_DIR}/control/positive450k/summary.json" \
  --control "changed450k=${OUTPUT_DIR}/control/changed450k/summary.json" \
  --parent_label parent400k \
  --candidate_label positive450k \
  --candidate_label changed450k \
  --direct_comparison positive450k,changed450k \
  --output "${OUTPUT_DIR}/decision.json"

echo "evaluation complete: ${OUTPUT_DIR}"
