#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs/mogeflow-control"
PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/conda_path/envs/codeflow/bin/python}"
GPU_ID="${GPU_ID:?GPU_ID must name one idle physical GPU}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/checkpoints/t2m/hy273_redenoise_kimodo_complete_stage2_control_ddp8_20260713_0547/model/step_00250000.pt}"
RUN_NAME="${RUN_NAME:-hy273_complete_s2_step250000_kimodo_style_val16_ode32_cfg2_c2_seed3407_20260713}"
OUT_DIR="${OUT_DIR:-${ROOT}/generation/${RUN_NAME}}"
LOG="${LOG:-${ROOT}/run_logs/${RUN_NAME}.log}"
SHM_CHECKPOINT="/dev/shm/${RUN_NAME}.pt"
TEXT_CFG="${TEXT_CFG:-2.0}"
CONTROL_CFG="${CONTROL_CFG:-2.0}"

check_idle_gpu() {
  local line memory util
  line="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v id="${GPU_ID}" '$1 + 0 == id {print; exit}')"
  [[ -n "${line}" ]] || { echo "GPU ${GPU_ID} not found" >&2; return 1; }
  IFS=',' read -r _ memory util <<< "${line}"
  memory="${memory// /}"
  util="${util// /}"
  (( memory <= 512 && util <= 5 )) || {
    echo "GPU ${GPU_ID} is not idle: ${line}" >&2
    return 1
  }
}

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
[[ ! -e "${OUT_DIR}" ]] || { echo "Output already exists: ${OUT_DIR}" >&2; exit 2; }
if [[ "${ALLOW_BUSY_GPU:-0}" == "1" ]]; then
  echo "[eval] busy-GPU override enabled by user; sharing GPU ${GPU_ID}"
  nvidia-smi --id="${GPU_ID}" \
    --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
else
  check_idle_gpu
  sleep 15
  check_idle_gpu
fi

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1
echo "[eval] start=$(date -Is) gpu=${GPU_ID} checkpoint=${CHECKPOINT}"

if [[ ! -f "${SHM_CHECKPOINT}" ]] || [[ "$(stat -c %s "${SHM_CHECKPOINT}")" != "$(stat -c %s "${CHECKPOINT}")" ]]; then
  rm -f "${SHM_CHECKPOINT}"
  cp "${CHECKPOINT}" "${SHM_CHECKPOINT}"
fi

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" sample_hy273_raw.py \
  --checkpoint "${SHM_CHECKPOINT}" \
  --split val \
  --indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --output_dir "${OUT_DIR}" \
  --num_steps 32 \
  --cfg_scale "${TEXT_CFG}" \
  --control_cfg_scale "${CONTROL_CFG}" \
  --weight_source ema \
  --max_control_keyframes 5 \
  --control_mode_schedule root_dense,root_dense,root_sparse,root_sparse,endpoints,endpoints,endpoints,endpoints,fullpose,fullpose,fullpose,contact,contact,mixed,mixed,mixed \
  --c_dir_mode dataset \
  --seed 3407

"${PYTHON_BIN}" tools/render_hy273_samples.py \
  --sample_dir "${OUT_DIR}" \
  --output_dir "${OUT_DIR}/videos_raw_world_gif" \
  --max_videos 16 \
  --format gif \
  --stride 2

"${PYTHON_BIN}" tools/render_hy273_samples.py \
  --sample_dir "${OUT_DIR}" \
  --output_dir "${OUT_DIR}/videos_raw_centered_gif" \
  --max_videos 16 \
  --format gif \
  --stride 2 \
  --center_root

EXACT_SOURCE="${OUT_DIR}/exact_render_source"
mkdir -p "${EXACT_SOURCE}"
ln -s ../samples_exact_clamped.npy "${EXACT_SOURCE}/samples.npy"
for name in observed.npy mask.npy lengths.npy metadata.json; do
  ln -s "../${name}" "${EXACT_SOURCE}/${name}"
done
"${PYTHON_BIN}" tools/render_hy273_samples.py \
  --sample_dir "${EXACT_SOURCE}" \
  --output_dir "${OUT_DIR}/videos_exact_world_gif" \
  --max_videos 16 \
  --format gif \
  --stride 2

echo "[eval] complete=$(date -Is) output=${OUT_DIR}"
