#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/afs/mogeflow-control"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_LOG="${WAIT_LOG:-${ROOT}/run_logs/hy273_complete_s2_step250k_eval_wait.log}"

mkdir -p "$(dirname "${WAIT_LOG}")"
exec > >(tee -a "${WAIT_LOG}") 2>&1

while true; do
  echo "[wait] time=$(date -Is)"
  GPU_ID="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      | awk -F',' '$2 + 0 <= 512 && $3 + 0 <= 5 {gsub(/ /, "", $1); print $1; exit}'
  )"
  if [[ -n "${GPU_ID}" ]]; then
    echo "[wait] candidate_idle_gpu=${GPU_ID}"
    if GPU_ID="${GPU_ID}" \
      PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/conda_path/envs/codeflow/bin/python}" \
      bash "${ROOT}/scripts/launch/eval_hy273_kimodo_style_step250k.sh"; then
      echo "[wait] evaluation_complete"
      exit 0
    fi
    echo "[wait] candidate was lost or evaluation failed; retrying"
  fi
  sleep "${POLL_SECONDS}"
done
