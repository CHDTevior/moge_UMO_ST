#!/usr/bin/env bash
set -u

ROOT="/mnt/afs/mogeflow-control"
RUN="kv_control_adapter_semantic_mix_encoder_target_ddp4_20260708"
TRAIN_LOG="${ROOT}/checkpoints/t2m/${RUN}/logs/train.jsonl"
MAIN_LOG="${ROOT}/logs/${RUN}_train.log"
HEALTH_LOG="${ROOT}/logs/${RUN}.health.log"
TRAIN_SESSION="kvctrl_semantic_mix_train_20260708"
FLOW_WEIGHT=0.2
KV_WEIGHT=0.8

cd "${ROOT}" || exit 1

while true; do
  {
    printf '\n[%s]\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
    if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
      echo "session=alive"
    else
      echo "session=missing"
    fi

    echo "gpu=index,mem_mib,util_pct"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits

    if [[ -f "${TRAIN_LOG}" ]]; then
      /mnt/afs/conda_path/envs/codeflow/bin/python - "${TRAIN_LOG}" "${FLOW_WEIGHT}" "${KV_WEIGHT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
flow_weight = float(sys.argv[2])
kv_weight = float(sys.argv[3])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    print("metrics=empty")
    raise SystemExit

for n in (20, 100, 200):
    win = rows[-n:] if len(rows) >= n else rows
    def avg(key):
        return sum(float(row[key]) for row in win) / len(win)
    flow = avg("flow_loss")
    kv = avg("kv_control_loss")
    weighted_flow = flow_weight * flow
    weighted_kv = kv_weight * kv
    denom = max(weighted_flow + weighted_kv, 1e-12)
    print(
        f"last_{len(win)} step={win[0]['step']}->{win[-1]['step']} "
        f"loss={avg('loss'):.5f} flow={flow:.5f} kv={kv:.5f} "
        f"flow_pct={100.0 * weighted_flow / denom:.1f} kv_pct={100.0 * weighted_kv / denom:.1f} "
        f"token_acc={avg('token_acc'):.5f} mask={avg('kv_control_mask_frac'):.6f} "
        f"lr={float(win[-1]['lr']):.8f}"
    )
latest = rows[-1]
print(
    "latest "
    f"epoch={latest['epoch']} step={latest['step']} loss={float(latest['loss']):.5f} "
    f"flow={float(latest['flow_loss']):.5f} kv={float(latest['kv_control_loss']):.5f} "
    f"mask={float(latest['kv_control_mask_frac']):.6f} batch={latest['batch_kind']}"
)
PY
    else
      echo "metrics=missing"
    fi

    echo "recent_errors"
    if [[ -f "${MAIN_LOG}" ]]; then
      rg -n 'Traceback|RuntimeError|CUDA out of memory|NCCL|\b(nan|NaN|inf|Inf)\b' "${MAIN_LOG}" | tail -n 20 || true
    else
      echo "main_log=missing"
    fi
  } >> "${HEALTH_LOG}" 2>&1

  sleep 900
done
