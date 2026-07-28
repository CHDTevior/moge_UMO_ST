#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:-hy273_kencoder_stageA_ddp4x32_20260727_0832}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/hy273_text_fusion}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/${RUN_NAME}/model/step_00200000.pt}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_DIR}/eval}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

[[ ${#GPUS[@]} -ge 7 ]] || {
  echo "GPU_IDS must provide at least seven GPUs" >&2
  exit 2
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "Missing K-Encoder 200K checkpoint: ${CHECKPOINT}" >&2
  exit 2
}

MAIN_PANEL="${EVAL_ROOT}/kencoder_step200k_paraphrase_panel"
VISUAL16="${EVAL_ROOT}/kencoder_step200k_visual16"
PROBE_EMA="${EVAL_ROOT}/kencoder_step200k_text_path_t0p1_ema.json"
PROBE_RAW="${EVAL_ROOT}/kencoder_step200k_text_path_t0p1_raw.json"
LOG_DIR="${EVAL_ROOT}/kencoder_step200k_logs"
mkdir -p "${LOG_DIR}"

run_panel() {
  local gpu="$1"
  local cfg="$2"
  local label="$3"
  local output="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    tools/eval_hy273_text_paraphrase_panel.py \
    --checkpoint "${label}=${CHECKPOINT}" \
    --output_dir "${output}" \
    --device cuda:0 \
    --weight_source ema \
    --num_steps 32 \
    --cfg_scale "${cfg}" \
    --target_length 150 \
    --seeds 3407,12345,20260725
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_samples.py \
    --sample_dir "${output}/${label}" \
    --output_dir "${output}/${label}/gifs" \
    --max_videos 15 \
    --format gif
}

run_visual16() {
  local gpu="$1"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    tools/eval_hy273_multitask_t2m_visual16.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir "${VISUAL16}" \
    --device cuda:0 \
    --weight_source ema \
    --num_steps 32 \
    --cfg_scale 2.0 \
    --seed 3407 \
    --max_samples 16
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_samples.py \
    --sample_dir "${VISUAL16}" \
    --output_dir "${VISUAL16}/gifs" \
    --max_videos 16 \
    --format gif
}

run_probe() {
  local gpu="$1"
  local weight_source="$2"
  local output="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    tools/diagnose_hy273_t2m_text_path.py \
    --checkpoint "kencoder200k=${CHECKPOINT}" \
    --output "${output}" \
    --device cuda:0 \
    --timesteps 0.1 \
    --seed 20260727 \
    --weight_source "${weight_source}"
}

run_panel "${GPUS[0]}" 2.0 kencoder200k "${MAIN_PANEL}" \
  >"${LOG_DIR}/paraphrase_cfg2.log" 2>&1 &
PIDS=("$!")
run_visual16 "${GPUS[1]}" >"${LOG_DIR}/visual16.log" 2>&1 &
PIDS+=("$!")
run_probe "${GPUS[2]}" ema "${PROBE_EMA}" >"${LOG_DIR}/probe_ema.log" 2>&1 &
PIDS+=("$!")
run_probe "${GPUS[3]}" model "${PROBE_RAW}" >"${LOG_DIR}/probe_raw.log" 2>&1 &
PIDS+=("$!")
run_panel "${GPUS[4]}" 1.0 kencoder200k_cfg1p0 \
  "${EVAL_ROOT}/kencoder_step200k_paraphrase_cfg1p0" \
  >"${LOG_DIR}/paraphrase_cfg1.log" 2>&1 &
PIDS+=("$!")
run_panel "${GPUS[5]}" 1.5 kencoder200k_cfg1p5 \
  "${EVAL_ROOT}/kencoder_step200k_paraphrase_cfg1p5" \
  >"${LOG_DIR}/paraphrase_cfg1p5.log" 2>&1 &
PIDS+=("$!")
run_panel "${GPUS[6]}" 3.0 kencoder200k_cfg3p0 \
  "${EVAL_ROOT}/kencoder_step200k_paraphrase_cfg3p0" \
  >"${LOG_DIR}/paraphrase_cfg3.log" 2>&1 &
PIDS+=("$!")

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" != "0" ]]; then
  echo "One or more K-Encoder 200K evaluations failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

echo "K-Encoder 200K evaluation completed: ${EVAL_ROOT}"
