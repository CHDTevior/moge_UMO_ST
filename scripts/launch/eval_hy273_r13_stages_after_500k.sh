#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the shared R13 training run}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/outputs/hy273_multitask/${RUN_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-${ROOT_DIR}/run_logs/${RUN_NAME}.log}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_DIR}/evaluation/stage_benchmarks_ode32_cfg2_obs2_seed3407}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PROFILE="${PROFILE:-research}"
POLL_SECONDS="${POLL_SECONDS:-300}"
DEAD_POLLS_BEFORE_FAIL="${DEAD_POLLS_BEFORE_FAIL:-4}"

FINAL_CHECKPOINT="${RUN_DIR}/model/step_00500000.pt"
TRAIN_PATTERN="^/root/miniconda3/envs/mogo/bin/python3\\.10 -u train_hy273_multitask\\.py .*--name ${RUN_NAME}([[:space:]]|$)"

training_is_running() {
  pgrep -f "${TRAIN_PATTERN}" >/dev/null
}

run_is_complete() {
  [[ -s "${FINAL_CHECKPOINT}" ]] &&
    [[ -f "${TRAIN_LOG}" ]] &&
    grep -q '"event": "stage_complete".*"next_global_step": 500000' "${TRAIN_LOG}"
}

dead_polls=0
while ! run_is_complete; do
  if training_is_running; then
    dead_polls=0
  else
    dead_polls=$((dead_polls + 1))
    if (( dead_polls >= DEAD_POLLS_BEFORE_FAIL )); then
      echo "Training stopped without a validated 500K completion; benchmark not started." >&2
      exit 3
    fi
  fi
  echo "[$(date --iso-8601=seconds)] waiting for the 500K checkpoint and stage_complete event"
  sleep "${POLL_SECONDS}"
done

mkdir -p "${EVAL_ROOT}"
for step in 00200000 00250000 00400000 00450000 00500000; do
  checkpoint="${RUN_DIR}/model/step_${step}.pt"
  output_dir="${EVAL_ROOT}/step_${step}"
  [[ -s "${checkpoint}" ]] || {
    echo "Missing stage-boundary checkpoint: ${checkpoint}" >&2
    exit 4
  }
  if [[ -f "${output_dir}/summary.json" && -f "${output_dir}/artifact_index.json" ]]; then
    echo "[$(date --iso-8601=seconds)] step_${step} benchmark already complete; skipping"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] starting full benchmark for step_${step}"
  CHECKPOINT="${checkpoint}" \
    OUTPUT_DIR="${output_dir}" \
    GPU_IDS="${GPU_IDS}" \
    PROFILE="${PROFILE}" \
    bash scripts/launch/eval_hy273_kimodo_v5_contact_8gpu.sh
done

echo "[$(date --iso-8601=seconds)] all R13 stage-boundary benchmarks complete: ${EVAL_ROOT}"
