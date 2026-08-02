#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the Unified Reaction run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/hy273_unified_reaction}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
CHECKPOINT_200K="${CHECKPOINT_200K:-${RUN_ROOT}/model/step_00200000.pt}"
CHECKPOINT_250K="${CHECKPOINT_250K:-${RUN_ROOT}/model/step_00250000.pt}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_stage_b_250k/t2m/native_tmr_ema_ode32_cfg3p5}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/mogo/bin/torchrun}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

[[ ${#GPUS[@]} -ge 8 ]] || {
  echo "Native TMR comparison requires eight GPUs" >&2
  exit 2
}
for checkpoint in "${CHECKPOINT_200K}" "${CHECKPOINT_250K}"; do
  [[ -f "${checkpoint}" ]] || {
    echo "Missing comparison checkpoint: ${checkpoint}" >&2
    exit 2
  }
done

"${PYTHON_BIN}" - "${CHECKPOINT_200K}" "${CHECKPOINT_250K}" <<'PY'
import sys
import torch

for path, expected in zip(sys.argv[1:], (200_000, 250_000)):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("format") != "hy273_unified_actor_checkpoint_v1":
        raise RuntimeError(f"Not a unified actor checkpoint: {path}")
    if int(checkpoint.get("next_global_step", -1)) != expected:
        raise RuntimeError(f"Expected step {expected}: {path}")
PY

mkdir -p "${EVAL_ROOT}/logs"

CUDA_VISIBLE_DEVICES="${GPUS[0]},${GPUS[1]},${GPUS[2]},${GPUS[3]}" \
  "${TORCHRUN_BIN}" --standalone --master_port=29641 --nproc_per_node=4 \
  tools/eval_hy273_multitask_t2m_tmr.py \
  --checkpoint "${CHECKPOINT_200K}" \
  --output_dir "${EVAL_ROOT}/stageB200k" \
  --weight_source ema --num_steps 32 --text_cfg_scale 3.5 \
  --batch_size 8 --num_workers 2 --seed 3407 \
  >"${EVAL_ROOT}/logs/stageB200k.log" 2>&1 &
pid_200k=$!

CUDA_VISIBLE_DEVICES="${GPUS[4]},${GPUS[5]},${GPUS[6]},${GPUS[7]}" \
  "${TORCHRUN_BIN}" --standalone --master_port=29642 --nproc_per_node=4 \
  tools/eval_hy273_multitask_t2m_tmr.py \
  --checkpoint "${CHECKPOINT_250K}" \
  --output_dir "${EVAL_ROOT}/stageB250k" \
  --weight_source ema --num_steps 32 --text_cfg_scale 3.5 \
  --batch_size 8 --num_workers 2 --seed 3407 \
  >"${EVAL_ROOT}/logs/stageB250k.log" 2>&1 &
pid_250k=$!

status=0
wait "${pid_200k}" || status=1
wait "${pid_250k}" || status=1
[[ "${status}" -eq 0 ]] || {
  echo "Native TMR comparison failed; inspect ${EVAL_ROOT}/logs" >&2
  exit "${status}"
}

"${PYTHON_BIN}" - "${EVAL_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = {}
for label in ("stageB200k", "stageB250k"):
    payload = json.loads((root / label / "summary.json").read_text())
    metrics = payload["metrics"]
    pool = metrics["generated_pool32"]
    rows[label] = {
        "fid": metrics["fid"],
        "paired_text_motion_cosine": metrics["paired_text_motion_cosine"],
        "t2m_r1": pool["t2m_top1_percent"],
        "t2m_r2": pool["t2m_top2_percent"],
        "t2m_r3": pool["t2m_top3_percent"],
        "m2t_r1": pool["m2t_top1_percent"],
        "m2t_r2": pool["m2t_top2_percent"],
        "m2t_r3": pool["m2t_top3_percent"],
    }
delta = {key: rows["stageB250k"][key] - rows["stageB200k"][key] for key in rows["stageB200k"]}
result = {
    "protocol": "native_hy273_tmr_ema_ode32_cfg3p5_seed3407_val1332",
    "stageB200k": rows["stageB200k"],
    "stageB250k": rows["stageB250k"],
    "delta_250k_minus_200k": delta,
}
(root / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY

echo "Native HY273 TMR 200K-vs-250K comparison complete: ${EVAL_ROOT}"
