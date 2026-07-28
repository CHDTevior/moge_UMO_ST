#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOSS_VARIANT="${LOSS_VARIANT:-}"
case "${LOSS_VARIANT}" in
  L2|l2)
    LOSS_VARIANT="L2"
    FK_CONSISTENCY_WEIGHT="0.0"
    FK_CONSISTENCY_WARMUP="0"
    EXPECTED_GPUS="0,1,2,3"
    EXPECTED_MASTER_PORT="29831"
    ;;
  L3|l3)
    LOSS_VARIANT="L3"
    FK_CONSISTENCY_WEIGHT="0.07"
    FK_CONSISTENCY_WARMUP="5000"
    EXPECTED_GPUS="4,5,6,7"
    EXPECTED_MASTER_PORT="29832"
    ;;
  *)
    printf 'LOSS_VARIANT must be L2 or L3, got %q\n' "${LOSS_VARIANT}" >&2
    exit 2
    ;;
esac

if [[ "$#" != "0" ]]; then
  printf 'Production scratch launcher accepts no trailing overrides: %q\n' "$*" >&2
  exit 2
fi

export NUM_GPUS="4"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${EXPECTED_GPUS}"
export MASTER_PORT="${EXPECTED_MASTER_PORT}"
export OMP_NUM_THREADS="4"
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
GPU_INFO="$(nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits)"
for GPU_ID in "${VISIBLE_GPU_LIST[@]}"; do
  GPU_LINE="$(awk -F',' -v id="${GPU_ID}" '$1 + 0 == id {print; exit}' <<< "${GPU_INFO}")"
  if [[ -z "${GPU_LINE}" ]]; then
    printf 'GPU %s not reported by nvidia-smi\n' "${GPU_ID}" >&2
    exit 2
  fi
  IFS=',' read -r _ GPU_NAME GPU_MEMORY GPU_UTIL <<< "${GPU_LINE}"
  GPU_NAME="${GPU_NAME# }"
  GPU_MEMORY="${GPU_MEMORY// /}"
  GPU_UTIL="${GPU_UTIL// /}"
  if [[ "${GPU_NAME}" != *A100* ]]; then
    printf 'GPU %s is not an A100: %s\n' "${GPU_ID}" "${GPU_NAME}" >&2
    exit 2
  fi
  if (( GPU_MEMORY > 512 || GPU_UTIL > 5 )); then
    printf 'GPU %s is not idle: memory=%s MiB utilization=%s%%\n' \
      "${GPU_ID}" "${GPU_MEMORY}" "${GPU_UTIL}" >&2
    exit 2
  fi
done

/root/miniconda3/envs/mogo/bin/python -c \
  'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' \
  "${MASTER_PORT}"

export PYTHON_BIN="/root/miniconda3/envs/mogo/bin/python"
export BATCH_SIZE="16"
export NUM_WORKERS="4"
export MAX_STEPS="200000"
export SAVE_EVERY="50000"
export MAX_EPOCHS="4000"
export CONFIG="configs/raw_flow_hy273_hytext_l2_l3_scratch.yaml"
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
export RUN_NAME="${RUN_NAME:-hy273_${LOSS_VARIANT,,}_scratch200k_ddp4_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  printf 'RUN_NAME must be a safe basename, got %q\n' "${RUN_NAME}" >&2
  exit 2
fi
RUN_DIR="${OUT_DIR}/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  printf 'Run directory already exists: %s\n' "${RUN_DIR}" >&2
  exit 2
fi

SOURCE_MANIFEST="${REPO_ROOT}/run_logs/hy273_l2_l3_scratch_source_manifest.sha256"
PREFLIGHT_REPORT="${REPO_ROOT}/run_logs/hy273_l2_l3_scratch_preflight_report.json"
(
  cd "${REPO_ROOT}"
  sha256sum --check --strict "${SOURCE_MANIFEST}"
)
"${PYTHON_BIN}" -c '
import hashlib
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
report = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
if report.get("format") != "hy273_l2_l3_scratch_preflight_v1":
    raise SystemExit(f"Unsupported HY273 scratch preflight format: {report.get('"'"'format'"'"')!r}")
if report.get("passed") is not True:
    raise SystemExit("HY273 scratch preflight report is not passing")
checks = report.get("checks")
required_checks = (
    "source_binding",
    "calibration",
    "checkpoint",
    "l2_log",
    "l3_log",
    "paired_trace",
    "resume",
    "trace_contract",
)
if not isinstance(checks, dict):
    raise SystemExit("HY273 scratch preflight report has no checks object")
for name in required_checks:
    check = checks.get(name)
    if not isinstance(check, dict) or check.get("passed") is not True:
        raise SystemExit(f"HY273 scratch preflight check is not passing: {name}")
binding = checks["source_binding"]
if binding.get("manifest_sha256") != actual_manifest_sha:
    raise SystemExit(
        "HY273 scratch preflight/source-manifest mismatch: "
        f"report={binding.get('"'"'manifest_sha256'"'"')}, current={actual_manifest_sha}"
    )
' "${SOURCE_MANIFEST}" "${PREFLIGHT_REPORT}"

"${SCRIPT_DIR}/train_hy273_raw_flow_stage1_x0_hytext_ddp8.sh" \
  --seed 3407 \
  --gradient_accumulation_steps 2 \
  --representation_loss_mode semantic_weighted \
  --representation_loss_scale 0.3247346107295814 \
  --fk_consistency_loss_weight "${FK_CONSISTENCY_WEIGHT}" \
  --fk_consistency_scale_m 0.05 \
  --fk_consistency_warmup_steps "${FK_CONSISTENCY_WARMUP}" \
  --deterministic_trace \
  --trace_seed 3407 \
  --trace_hash_steps 100 \
  --expected_initial_model_sha256 808a1639f7ec134887ce07b3d2634849dccfe68e6441bac0addba4f4cc579e59
