#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the formal R12 run}"
MODE="${MODE:-run}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
RUN_DIR="${ROOT_DIR}/outputs/hy273_multitask/${RUN_NAME}"
CHECKPOINT="${RUN_DIR}/model/step_00250000.pt"
RUN_IDENTITY="${RUN_DIR}/run_identity.json"
TAG="${OUTPUT_TAG:-${RUN_NAME}}"

BASELINE_EVAL="${ROOT_DIR}/outputs/hy273_multitask/gates/control_step200k_20260716_r3"
R11_250K_EVAL="${ROOT_DIR}/outputs/hy273_multitask/gates/control_step250k_20260716"
BASELINE_VISUAL="${ROOT_DIR}/generation/hy273_multitask_r11_stage_a_step200k_t2m_visual16_cfg2_20260717"
CONTROL_OUTPUT="${ROOT_DIR}/outputs/hy273_multitask/gates/r12_control_step250k_${TAG}"
VISUAL_OUTPUT="${ROOT_DIR}/generation/hy273_multitask_r12_step250k_t2m_visual16_${TAG}"
HUMAN_VERDICT="${HUMAN_VERDICT:-${VISUAL_OUTPUT}/human_verdict.json}"
HUMAN_TEMPLATE="${VISUAL_OUTPUT}/human_verdict.template.json"
GATE_ARTIFACT="${GATE_ARTIFACT:-${ROOT_DIR}/outputs/hy273_multitask/gates/r12_step250k_gate_${TAG}.json}"

[[ "${MODE}" == "run" || "${MODE}" == "finalize" ]] || { echo "MODE must be run or finalize" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" && -f "${CHECKPOINT}" && -f "${RUN_IDENTITY}" ]] || { echo "Missing Python/R12 checkpoint/run identity" >&2; exit 2; }
[[ -f "${BASELINE_EVAL}/artifact_index.json" && -f "${R11_250K_EVAL}/artifact_index.json" ]] || { echo "Missing frozen control evidence" >&2; exit 2; }
CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"

"${PYTHON_BIN}" - "${CHECKPOINT}" "${CHECKPOINT_SHA256}" "${RUN_IDENTITY}" <<'PY'
import json
import sys
import torch

checkpoint_path, checkpoint_sha, identity_path = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
identity = json.load(open(identity_path, encoding="utf-8"))
if checkpoint.get("format") != "hy273_multitask_checkpoint_v2": raise SystemExit("checkpoint format mismatch")
if checkpoint.get("train_contract") != "hy273_multitask_train_contract_v3_rootmask": raise SystemExit("not an R12 checkpoint")
if checkpoint.get("next_global_step") != 250_000: raise SystemExit("not the exact 250K checkpoint")
if checkpoint.get("run_uuid") != identity.get("run_uuid"): raise SystemExit("run UUID mismatch")
origin = checkpoint.get("runtime_identity", {}).get("origin_parent")
if origin != identity.get("origin_parent"): raise SystemExit("origin lineage mismatch")
if origin.get("checkpoint_sha256") != "e06b397df60e9b68e628fa68bede687c97ecb9bb25e556f3d96a311423e1744e": raise SystemExit("bad R12 origin")
print(json.dumps({"checkpoint_sha256": checkpoint_sha, "run_uuid": checkpoint["run_uuid"], "origin": origin}, sort_keys=True))
PY

if [[ "${MODE}" == "run" ]]; then
  [[ ! -e "${CONTROL_OUTPUT}" && ! -e "${VISUAL_OUTPUT}" ]] || {
    echo "Gate output already exists; use MODE=finalize after supplying the human verdict" >&2
    exit 2
  }

  CHECKPOINT="${CHECKPOINT}" CHECKPOINT_SHA256="${CHECKPOINT_SHA256}" \
  OUTPUT_DIR="${CONTROL_OUTPUT}" GPU_IDS="${GPU_IDS}" \
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

  "${PYTHON_BIN}" tools/render_hy273_samples.py --sample_dir "${VISUAL_OUTPUT}" --output_dir "${VISUAL_OUTPUT}/render_world" --max_videos 16 --format gif
  "${PYTHON_BIN}" tools/render_hy273_samples.py --sample_dir "${VISUAL_OUTPUT}" --output_dir "${VISUAL_OUTPUT}/render_centered" --max_videos 16 --format gif --center_root

  "${PYTHON_BIN}" - "${RUN_IDENTITY}" "${CHECKPOINT_SHA256}" "${HUMAN_TEMPLATE}" <<'PY'
import json
import sys
from pathlib import Path

identity_path, checkpoint_sha, output = sys.argv[1:]
identity = json.load(open(identity_path, encoding="utf-8"))
payload = {
    "format": "hy273_r12_fixed16_human_verdict_v1",
    "status": "replace_with_passed_or_failed",
    "checkpoint_sha256": checkpoint_sha,
    "run_uuid": identity["run_uuid"],
    "review": "Describe semantic match, motion naturalness, transitions, and foot sliding across all 16 samples.",
}
Path(output).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
PY
fi

[[ -f "${CONTROL_OUTPUT}/artifact_index.json" && -f "${VISUAL_OUTPUT}/quality.json" ]] || { echo "R12 gate evidence is incomplete" >&2; exit 2; }
if [[ ! -f "${HUMAN_VERDICT}" ]]; then
  echo "Control/visual evaluation finished. Review ${VISUAL_OUTPUT}/render_world and render_centered, then write ${HUMAN_VERDICT} from ${HUMAN_TEMPLATE} and rerun with MODE=finalize." >&2
  exit 3
fi

"${PYTHON_BIN}" tools/gate_hy273_r12_step250k.py evaluate \
  --baseline_eval "${BASELINE_EVAL}" \
  --r11_eval "${R11_250K_EVAL}" \
  --candidate_eval "${CONTROL_OUTPUT}" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_sha256 "${CHECKPOINT_SHA256}" \
  --run_identity "${RUN_IDENTITY}" \
  --baseline_visual "${BASELINE_VISUAL}" \
  --candidate_visual "${VISUAL_OUTPUT}" \
  --human_verdict "${HUMAN_VERDICT}" \
  --output "${GATE_ARTIFACT}" \
  --bootstrap_resamples 10000 \
  --confidence 0.95 \
  --relative_tolerance 0.05 \
  --seed 3407

echo "R12 250K gate passed: ${GATE_ARTIFACT}"
