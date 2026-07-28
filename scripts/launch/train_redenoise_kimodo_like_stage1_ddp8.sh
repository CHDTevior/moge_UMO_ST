#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29961}"
CONFIG="${CONFIG:-configs/redenoise_kimodo_like_stage1.yaml}"
RUN_NAME="${RUN_NAME:-hy273_redenoise_kimodo_like_stage1_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/t2m}"
[[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "RUN_NAME must be a safe basename: ${RUN_NAME}" >&2
  exit 2
}
[[ -z "${MAX_STEPS+x}" && -z "${MAX_EPOCHS+x}" && -z "${SAVE_EVERY+x}" ]] || {
  echo "MAX_STEPS/MAX_EPOCHS/SAVE_EVERY overrides are disabled; use PILOT_MAX_STEPS" >&2
  exit 2
}
EXECUTION_CONTRACT="stage1_production"
LIMIT_ARGS=()
if [[ -n "${PILOT_MAX_STEPS:-}" ]]; then
  [[ "${ALLOW_SHORT_PILOT:-0}" == "1" ]] || {
    echo "PILOT_MAX_STEPS requires ALLOW_SHORT_PILOT=1" >&2
    exit 2
  }
  [[ "${PILOT_MAX_STEPS}" =~ ^[0-9]+$ ]] && (( PILOT_MAX_STEPS > 0 && PILOT_MAX_STEPS < 200000 )) || {
    echo "Stage-1 PILOT_MAX_STEPS must be in [1,199999]" >&2
    exit 2
  }
  EXECUTION_CONTRACT="stage1_pilot"
  LIMIT_ARGS=(--max_steps "${PILOT_MAX_STEPS}")
fi

[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 2; }
"${PYTHON_BIN}" - "${CONFIG}" <<'PY'
import pathlib, sys, yaml
cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
def require(condition, message):
    if not condition:
        raise SystemExit(message)
require(cfg["model"]["architecture"] == "redenoise_kimodo_like", "wrong architecture")
require(cfg["model"]["prediction_type"] == "x0", "wrong prediction type")
require(cfg["control"]["training_phase"] == "text_only", "not a Stage-1 config")
require(cfg["control"]["modes"] == ["none"], "Stage-1 controls must be [none]")
require(float(cfg["loss"]["control_cont"]) == 0.0, "Stage-1 control loss must be zero")
require(float(cfg["loss"]["control_contact"]) == 0.0, "Stage-1 contact-control loss must be zero")
require(int(cfg["train"]["max_steps"]) == 200000, "Stage-1 config must target 200K")
for value in (
    pathlib.Path(cfg["normalization"]["motion_stats_dir"]).parent / "manifest.json",
    pathlib.Path(cfg["text"]["hytext_cache_dir"]) / "manifest.json",
    pathlib.Path(cfg["assets"]["manifest_path"]),
):
    require(value.is_file(), f"missing asset: {value}")
require(len(str(cfg["assets"]["manifest_sha256"])) == 64, "invalid asset manifest SHA")
require(len(str(cfg["assets"]["expected_initial_model_sha256"])) == 64, "invalid initial model SHA")
PY
[[ ! -e "${OUT_DIR}/${RUN_NAME}" ]] || { echo "Run already exists: ${OUT_DIR}/${RUN_NAME}" >&2; exit 2; }

exec 9>/tmp/hy273_redenoise_kimodo_like_gpu.lock
flock -n 9 || { echo "redenoise_kimodo_like GPU lock is held" >&2; exit 2; }

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#GPU_IDS[@]} == NUM_GPUS )) || {
  echo "NUM_GPUS=${NUM_GPUS} but CUDA_VISIBLE_DEVICES has ${#GPU_IDS[@]} entries" >&2
  exit 2
}
GPU_INFO="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
for GPU_ID in "${GPU_IDS[@]}"; do
  GPU_LINE="$(awk -F',' -v id="${GPU_ID}" '$1 + 0 == id {print; exit}' <<< "${GPU_INFO}")"
  IFS=',' read -r _ GPU_MEMORY GPU_UTIL <<< "${GPU_LINE}"
  GPU_MEMORY="${GPU_MEMORY// /}"
  GPU_UTIL="${GPU_UTIL// /}"
  if [[ -z "${GPU_LINE}" ]] || (( GPU_MEMORY > 512 || GPU_UTIL > 5 )); then
    echo "GPU ${GPU_ID} is not idle: ${GPU_LINE}" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  train_hy273_raw_flow.py \
  --config "${CONFIG}" \
  --name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --execution_contract "${EXECUTION_CONTRACT}" \
  "${LIMIT_ARGS[@]}" \
  --seed "${SEED:-3407}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --amp \
  --ema
