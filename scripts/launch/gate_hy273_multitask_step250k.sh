#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_multitask_r11_stage_a_t2m_ddp8_20260715_1510}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/outputs/hy273_multitask/${RUN_NAME}/model/step_00250000.pt}"
OUTPUT_TAG="${OUTPUT_TAG:-20260716}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

BASELINE_EVAL="${BASELINE_EVAL:-${ROOT_DIR}/outputs/hy273_multitask/gates/control_step200k_20260716_r3}"
BASELINE_COMPARATOR="${BASELINE_COMPARATOR:-${ROOT_DIR}/outputs/hy273_multitask/gates/control_step200k_comparator_20260716}"
FROZEN_GATE="${BASELINE_COMPARATOR}/gate_matrix.json"
FROZEN_GATE_SHA256="${FROZEN_GATE_SHA256:-a722704c94650d4eafd299b2f93a77faed357ba01c324c699ec6d14cf789ec41}"

CONTROL_OUTPUT="${CONTROL_OUTPUT:-${ROOT_DIR}/outputs/hy273_multitask/gates/control_step250k_${OUTPUT_TAG}}"
COMPARATOR_OUTPUT="${COMPARATOR_OUTPUT:-${ROOT_DIR}/outputs/hy273_multitask/gates/control_step250k_comparator_${OUTPUT_TAG}}"
VISUAL_OUTPUT="${VISUAL_OUTPUT:-${ROOT_DIR}/generation/hy273_multitask_r11_step250k_t2m_visual16_${OUTPUT_TAG}}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing 250K checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ -f "${BASELINE_EVAL}/artifact_index.json" ]] || { echo "Missing baseline eval" >&2; exit 2; }
[[ -f "${FROZEN_GATE}" ]] || { echo "Missing frozen gate matrix" >&2; exit 2; }

CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-$(sha256sum "${CHECKPOINT}" | awk '{print $1}')}"
ACTUAL_GATE_SHA256="$(sha256sum "${FROZEN_GATE}" | awk '{print $1}')"
[[ "${ACTUAL_GATE_SHA256}" == "${FROZEN_GATE_SHA256}" ]] || {
  echo "Frozen gate SHA mismatch: ${ACTUAL_GATE_SHA256}" >&2
  exit 2
}

"${PYTHON_BIN}" - "${CHECKPOINT}" <<'PY'
import sys
import torch

path = sys.argv[1]
checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
step = int(checkpoint.get("next_global_step", -1))
if step != 250_000:
    raise SystemExit(f"Expected next_global_step=250000, got {step}")
if checkpoint.get("format") != "hy273_multitask_checkpoint_v2":
    raise SystemExit(f"Unexpected checkpoint format: {checkpoint.get('format')!r}")
print(f"checkpoint_step={step} checkpoint_format={checkpoint['format']}")
PY

echo "checkpoint=${CHECKPOINT}"
echo "checkpoint_sha256=${CHECKPOINT_SHA256}"
echo "control_output=${CONTROL_OUTPUT}"
echo "comparator_output=${COMPARATOR_OUTPUT}"
echo "visual_output=${VISUAL_OUTPUT}"

CHECKPOINT="${CHECKPOINT}" \
CHECKPOINT_SHA256="${CHECKPOINT_SHA256}" \
OUTPUT_DIR="${CONTROL_OUTPUT}" \
GPU_IDS="${GPU_IDS}" \
bash scripts/launch/eval_hy273_kimodo_v5_contact_8gpu.sh

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" tools/eval_hy273_multitask_t2m_visual16.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "${VISUAL_OUTPUT}" \
  --device cuda:0 \
  --weight_source ema \
  --num_steps 32 \
  --cfg_scale 2.0 \
  --seed 3407 \
  --max_samples 16

"${PYTHON_BIN}" tools/render_hy273_samples.py \
  --sample_dir "${VISUAL_OUTPUT}" \
  --output_dir "${VISUAL_OUTPUT}/render_world" \
  --max_videos 16 \
  --format gif

"${PYTHON_BIN}" tools/render_hy273_samples.py \
  --sample_dir "${VISUAL_OUTPUT}" \
  --output_dir "${VISUAL_OUTPUT}/render_centered" \
  --max_videos 16 \
  --format gif \
  --center_root

"${PYTHON_BIN}" tools/compare_hy273_nonregression.py \
  --baseline_eval_dir "${BASELINE_EVAL}" \
  --candidate_eval_dir "${CONTROL_OUTPUT}" \
  --output_dir "${COMPARATOR_OUTPUT}" \
  --profile production \
  --frozen_gate_matrix "${FROZEN_GATE}" \
  --frozen_gate_matrix_sha256 "${FROZEN_GATE_SHA256}" \
  --bootstrap_resamples 10000 \
  --confidence 0.95 \
  --relative_tolerance 0.05 \
  --seed 3407

"${PYTHON_BIN}" - "${BASELINE_EVAL}" "${CONTROL_OUTPUT}" "${COMPARATOR_OUTPUT}" "${VISUAL_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

baseline, candidate, comparator, visual = map(Path, sys.argv[1:])
candidate_index = json.loads((candidate / "artifact_index.json").read_text())
comparison_index = json.loads((comparator / "artifact_index.json").read_text())
visual_metadata = json.loads((visual / "metadata.json").read_text())
payload = {
    "status": "validated",
    "baseline_eval": str(baseline.resolve()),
    "candidate_eval": str(candidate.resolve()),
    "candidate_cases": candidate_index["case_count"],
    "comparison_kind": comparison_index["comparison_kind"],
    "nonregression_decision": comparison_index["nonregression_decision"],
    "visual_dir": str(visual.resolve()),
    "visual_checkpoint_step": visual_metadata["checkpoint_next_global_step"],
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
PY
