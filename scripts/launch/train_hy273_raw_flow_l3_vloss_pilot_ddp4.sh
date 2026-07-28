#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != "0" ]]; then
  printf 'L3 v-loss pilot launcher accepts no trailing overrides: %q\n' "$*" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export NUM_GPUS="4"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT="29833"
export OMP_NUM_THREADS="4"

exec 9>/tmp/hy273_gpu_0_3.lock
if ! flock -n 9; then
  printf 'GPU 0-3 are reserved by another HY273 launcher\n' >&2
  exit 2
fi

check_idle_gpus() {
  local gpu_info gpu_id gpu_line gpu_name gpu_memory gpu_util
  local -a visible_gpu_list
  IFS=',' read -r -a visible_gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
  gpu_info="$(nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  for gpu_id in "${visible_gpu_list[@]}"; do
    gpu_line="$(awk -F',' -v id="${gpu_id}" '$1 + 0 == id {print; exit}' <<< "${gpu_info}")"
    if [[ -z "${gpu_line}" ]]; then
      printf 'GPU %s not reported by nvidia-smi\n' "${gpu_id}" >&2
      return 2
    fi
    IFS=',' read -r _ gpu_name gpu_memory gpu_util <<< "${gpu_line}"
    gpu_name="${gpu_name# }"
    gpu_memory="${gpu_memory// /}"
    gpu_util="${gpu_util// /}"
    if [[ "${gpu_name}" != *A100* ]]; then
      printf 'GPU %s is not an A100: %s\n' "${gpu_id}" "${gpu_name}" >&2
      return 2
    fi
    if (( gpu_memory > 512 || gpu_util > 5 )); then
      printf 'GPU %s is not idle: memory=%s MiB utilization=%s%%\n' \
        "${gpu_id}" "${gpu_memory}" "${gpu_util}" >&2
      return 2
    fi
  done
}
check_idle_gpus

/root/miniconda3/envs/mogo/bin/python -c \
  'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' \
  "${MASTER_PORT}"

export PYTHON_BIN="/root/miniconda3/envs/mogo/bin/python"
export BATCH_SIZE="16"
export NUM_WORKERS="4"
export MAX_STEPS="500"
export SAVE_EVERY="0"
export MAX_EPOCHS="4000"
export CONFIG="configs/raw_flow_hy273_hytext_l3_vloss_scratch.yaml"
export DATA_ROOT="/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22"
export TEXT_ROOT="/mnt/afs/mogo_base/datasets/HumanML3D/texts"
export HYTEXT_CACHE_DIR="/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128"
export OUT_DIR="${REPO_ROOT}/checkpoints/t2m"
export CLIP_PATH="${REPO_ROOT}/checkpoints/clip/ViT-B-32.pt"
export LR="0.0001"
export FLOW_LOSS_WEIGHT="1.0"
export CONTACT_LOSS_WEIGHT="0.1"
export CLEAN_ROOT_VEL_LOSS_WEIGHT="0.01"
export CLEAN_JOINT_VEL_LOSS_WEIGHT="0.01"
export FOOT_LOCK_LOSS_WEIGHT="0.01"
export EMA_DECAY="0.995"
export EMA_EVERY="10"
export MAX_TEXT_TOKENS="128"
export SEMANTIC_LOSS_FPS="30.0"
export FOOT_LOCK_CONTACT_THRESHOLD="0.5"

SOURCE_MANIFEST="run_logs/hy273_l3_vloss_source_manifest.sha256"
PAYLOAD_MANIFESTS=(
  run_logs/hy273_l3_vloss_motion_payload.sha256
  run_logs/hy273_l3_vloss_text_payload.sha256
  run_logs/hy273_l3_vloss_hytext_payload.sha256
)
sha256sum --check --quiet --strict "${SOURCE_MANIFEST}"
for PAYLOAD_MANIFEST in "${PAYLOAD_MANIFESTS[@]}"; do
  sha256sum --check --quiet --strict "${PAYLOAD_MANIFEST}"
done
SOURCE_MANIFEST_SHA256="$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')"

PILOT_RUN_NAME="hy273_l3_vloss_jit_bound_pilot500_v2"
if [[ -n "${RUN_NAME+x}" && "${RUN_NAME}" != "${PILOT_RUN_NAME}" ]]; then
  printf 'RUN_NAME is fixed to %s for verifier binding; got %q\n' \
    "${PILOT_RUN_NAME}" "${RUN_NAME}" >&2
  exit 2
fi
RUN_NAME="${PILOT_RUN_NAME}"
if [[ -e "${OUT_DIR}/${RUN_NAME}" ]]; then
  printf 'Run directory already exists: %s\n' "${OUT_DIR}/${RUN_NAME}" >&2
  exit 2
fi
LOG_PATH="${REPO_ROOT}/logs/${RUN_NAME}.log"
if [[ -e "${LOG_PATH}" ]]; then
  printf 'Pilot log already exists: %s\n' "${LOG_PATH}" >&2
  exit 2
fi
export RUN_NAME

check_idle_gpus
"${SCRIPT_DIR}/train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh" \
  --no-save-final \
  --log_every 1 \
  --seed 3407 \
  --gradient_accumulation_steps 2 \
  --representation_loss_mode semantic_weighted \
  --representation_loss_scale 0.09397019716051493 \
  --representation_loss_space velocity \
  --velocity_loss_t_eps 0.05 \
  --time_schedule logit_normal \
  --denoiser_p_mean -0.8 \
  --denoiser_p_std 0.8 \
  --fk_consistency_loss_weight 0.07 \
  --fk_consistency_scale_m 0.05 \
  --fk_consistency_warmup_steps 5000 \
  --deterministic_trace \
  --trace_seed 3407 \
  --trace_hash_steps 100 \
  --expected_initial_model_sha256 808a1639f7ec134887ce07b3d2634849dccfe68e6441bac0addba4f4cc579e59 \
  --source_manifest_sha256 "${SOURCE_MANIFEST_SHA256}" \
  2>&1 | tee "${LOG_PATH}"
