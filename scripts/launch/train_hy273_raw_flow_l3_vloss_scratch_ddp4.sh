#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" != "0" ]]; then
  printf 'L3 v-loss production launcher accepts no trailing overrides: %q\n' "$*" >&2
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
export MAX_STEPS="200000"
export SAVE_EVERY="50000"
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

REPRESENTATION_SCALE="0.09397019716051493"
VELOCITY_T_EPS="0.05"
TIME_SCHEDULE="logit_normal"
DENOISER_P_MEAN="-0.8"
DENOISER_P_STD="0.8"
FK_WEIGHT="0.07"
FK_WARMUP="5000"
EXPECTED_MODEL_SHA="808a1639f7ec134887ce07b3d2634849dccfe68e6441bac0addba4f4cc579e59"
TRAIN_T_REPORT="run_logs/hy273_l3_vloss_jitpm08ps08_calibration_train_t_seed3407_n4096.json"
BIN_REPORT="run_logs/hy273_l3_vloss_jitpm08ps08_calibration_bins_seed3407_n16_alpha0939702.json"
SOURCE_MANIFEST="run_logs/hy273_l3_vloss_source_manifest.sha256"
PILOT_REPORT="run_logs/hy273_l3_vloss_jit_preflight_report.json"

sha256sum --check --quiet --strict "${SOURCE_MANIFEST}"
SOURCE_MANIFEST_SHA256="$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')"
RUNTIME_PILOT_REPORT="$(mktemp /tmp/hy273_l3_vloss_runtime_preflight.XXXXXX.json)"
trap 'rm -f "${RUNTIME_PILOT_REPORT}"' EXIT
"${PYTHON_BIN}" tools/verify_hy273_l3_vloss_preflight.py \
  --output "${RUNTIME_PILOT_REPORT}"
PILOT_REPORT="${RUNTIME_PILOT_REPORT}"
"${PYTHON_BIN}" -c '
import json
import math
import pathlib
import sys

train_path, bin_path, pilot_path, expected_scale, expected_lambda, expected_sha, manifest_sha = sys.argv[1:]
train = json.loads(pathlib.Path(train_path).read_text(encoding="utf-8"))
bins = json.loads(pathlib.Path(bin_path).read_text(encoding="utf-8"))
pilot = json.loads(pathlib.Path(pilot_path).read_text(encoding="utf-8"))
scale = float(expected_scale)
lam = float(expected_lambda)
if train.get("calibration_loss_space") != "velocity":
    raise SystemExit("training-distribution calibration is not velocity-space")
if train.get("time_schedule") != "logit_normal":
    raise SystemExit("training-distribution calibration is not logit-normal")
if not math.isclose(float(train.get("denoiser_p_mean", "nan")), -0.8):
    raise SystemExit("training-distribution calibration has the wrong P_mean")
if not math.isclose(float(train.get("denoiser_p_std", "nan")), 0.8):
    raise SystemExit("training-distribution calibration has the wrong P_std")
if train.get("sample_training_timesteps") is not True or train.get("passed") is not True:
    raise SystemExit("training-distribution calibration did not pass")
if not math.isclose(float(train["representation_scale_alpha"]), scale, rel_tol=0.0, abs_tol=1e-15):
    raise SystemExit("representation scale does not match calibration")
if not math.isclose(float(train["selected_lambda"]), lam, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit("FK lambda does not match training-distribution calibration")
if train.get("initial_model_sha256") != expected_sha:
    raise SystemExit("scratch initial model SHA mismatch")
if pilot.get("format") != "hy273_l3_clean_head_jit_vloss_preflight_v1":
    raise SystemExit("unsupported L3 v-loss preflight format")
if pilot.get("passed") is not True:
    raise SystemExit("L3 v-loss preflight did not pass")
expected = pilot.get("expected", {})
if expected.get("source_manifest_sha256") != manifest_sha:
    raise SystemExit("pilot report is not bound to the current source manifest")
if expected.get("prediction_type") != "x0" or expected.get("representation_loss_space") != "velocity":
    raise SystemExit("pilot report has the wrong prediction/loss contract")
if not math.isclose(float(expected.get("representation_scale", "nan")), scale, rel_tol=0.0, abs_tol=1e-15):
    raise SystemExit("pilot report representation scale mismatch")
pilot_checks = pilot.get("checks", {})
for name in ("source_binding", "payload_binding", "pilot_log", "pilot_contract", "calibration"):
    if pilot_checks.get(name, {}).get("passed") is not True:
        raise SystemExit(f"pilot preflight check is not passing: {name}")
candidate = next(
    (item for item in bins.get("lambda_candidates", []) if math.isclose(float(item["lambda"]), lam)),
    None,
)
if candidate is None:
    raise SystemExit("fixed-bin calibration has no requested FK lambda")
max_bin = max(float(value) for value in candidate["bin_aggregate_ratios"].values())
if max_bin > 0.15:
    raise SystemExit(f"fixed-bin FK ratio exceeds 15%: {max_bin}")
if bins.get("passed") is not True or bins.get("selection_mode") != "max_bin_only":
    raise SystemExit("fixed-bin ceiling-only calibration did not pass")
binding = pilot_checks["source_binding"]
if binding.get("manifest_sha256") != manifest_sha:
    raise SystemExit("preflight source binding does not match the current manifest")
for field in ("config_checks", "trace_checks"):
    if pilot_checks["pilot_contract"].get(field, {}).get("source_manifest_sha") is not True:
        raise SystemExit(f"pilot contract did not bind source manifest in {field}")
print(json.dumps({"representation_scale": scale, "fk_lambda": lam, "max_bin_ratio": max_bin, "source_manifest_sha256": manifest_sha}))
' "${TRAIN_T_REPORT}" "${BIN_REPORT}" "${PILOT_REPORT}" "${REPRESENTATION_SCALE}" "${FK_WEIGHT}" "${EXPECTED_MODEL_SHA}" "${SOURCE_MANIFEST_SHA256}"

case "${PREFLIGHT_ONLY:-0}" in
  0) ;;
  1)
    printf 'L3 clean-head JiT v-loss production preflight passed.\n'
    exit 0
    ;;
  *)
    printf 'PREFLIGHT_ONLY must be 0 or 1, got %q\n' "${PREFLIGHT_ONLY}" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-hy273_l3_vloss_scratch200k_ddp4_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'RUN_NAME must be a safe basename, got %q\n' "${RUN_NAME}" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}/${RUN_NAME}" ]]; then
  printf 'Run directory already exists: %s\n' "${OUT_DIR}/${RUN_NAME}" >&2
  exit 2
fi
export RUN_NAME

check_idle_gpus
"${SCRIPT_DIR}/train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh" \
  --seed 3407 \
  --gradient_accumulation_steps 2 \
  --representation_loss_mode semantic_weighted \
  --representation_loss_scale "${REPRESENTATION_SCALE}" \
  --representation_loss_space velocity \
  --velocity_loss_t_eps "${VELOCITY_T_EPS}" \
  --time_schedule "${TIME_SCHEDULE}" \
  --denoiser_p_mean "${DENOISER_P_MEAN}" \
  --denoiser_p_std "${DENOISER_P_STD}" \
  --fk_consistency_loss_weight "${FK_WEIGHT}" \
  --fk_consistency_scale_m 0.05 \
  --fk_consistency_warmup_steps "${FK_WARMUP}" \
  --deterministic_trace \
  --trace_seed 3407 \
  --trace_hash_steps 100 \
  --expected_initial_model_sha256 "${EXPECTED_MODEL_SHA}" \
  --source_manifest_sha256 "${SOURCE_MANIFEST_SHA256}"
