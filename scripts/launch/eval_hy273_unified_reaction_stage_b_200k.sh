#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RUN_NAME="${RUN_NAME:?Set RUN_NAME to the completed Unified Reaction run}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/hy273_unified_reaction}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
STAGE_B_STEP="${STAGE_B_STEP:-200000}"
[[ "${STAGE_B_STEP}" =~ ^[0-9]+$ ]] && (( STAGE_B_STEP > 100000 )) || {
  echo "STAGE_B_STEP must be an integer greater than 100000" >&2
  exit 2
}
(( STAGE_B_STEP % 1000 == 0 )) || {
  echo "STAGE_B_STEP must be divisible by 1000 for stable evaluation labels" >&2
  exit 2
}
STAGE_B_K=$((STAGE_B_STEP / 1000))
STAGE_B_LABEL="stageB${STAGE_B_K}k"
STAGE_B_CHECKPOINT_NAME="$(printf 'step_%08d.pt' "${STAGE_B_STEP}")"
STAGE_A_CHECKPOINT="${STAGE_A_CHECKPOINT:-${RUN_ROOT}/model/step_00100000.pt}"
STAGE_B_CHECKPOINT="${STAGE_B_CHECKPOINT:-${RUN_ROOT}/model/${STAGE_B_CHECKPOINT_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-${RUN_ROOT}/eval_stage_b_${STAGE_B_K}k}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/mogo/bin/python}"
KIMODO_ROOT="${KIMODO_ROOT:-/mnt/afs/mogeflow-control/external_repos/kimodo}"
export KIMODO_ROOT
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "${GPU_IDS}"

SOURCE_CFG=2.0
TEXT_CFG=2.0
ODE_STEPS=32
EVAL_SEED=20260801
CAPTION_POLICY=uid_balanced
T2M_SHARDS="${T2M_SHARDS:-8}"
EDIT_SHARDS="${EDIT_SHARDS:-8}"
EDIT_CFG="${EDIT_CFG:-3.0}"
EDIT_CFG_TAG="${EDIT_CFG%.0}"
EDIT_SYSTEMS="source_copy,source_instruction_model,shuffled_source_instruction_model,source_shuffled_instruction_model,source_only_model,relative_instruction_only_ood"
EVAL_PHASE="${EVAL_PHASE:-all}"
ALLOW_REACTION_LOSS_ABLATION="${ALLOW_REACTION_LOSS_ABLATION:-0}"
ALLOW_SAME_MIX_EXTENSION_AT_STEP="${ALLOW_SAME_MIX_EXTENSION_AT_STEP:-0}"
EDIT_DIAGNOSTIC_BASELINE_CHECKPOINT="${EDIT_DIAGNOSTIC_BASELINE_CHECKPOINT:-${STAGE_A_CHECKPOINT}}"
EDIT_DIAGNOSTIC_BASELINE_STEP="${EDIT_DIAGNOSTIC_BASELINE_STEP:-100000}"
EDIT_DIAGNOSTIC_BASELINE_LABEL="${EDIT_DIAGNOSTIC_BASELINE_LABEL:-stageA100k}"
EDIT_DIAGNOSTIC_ALLOW_REACTION_LOSS_ABLATION="${EDIT_DIAGNOSTIC_ALLOW_REACTION_LOSS_ABLATION:-${ALLOW_REACTION_LOSS_ABLATION}}"
MULTITASK_MANIFEST_ROOT="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/hy273_multitask_v1"
EDIT_COUNTERFACTUAL_ROOT="${EDIT_COUNTERFACTUAL_ROOT:-/mnt/afs/mogeflow-control/outputs/hy273_multitask/gates}"

[[ "${T2M_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "T2M_SHARDS must be a positive integer" >&2
  exit 2
}
[[ "${EDIT_SHARDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EDIT_SHARDS must be a positive integer" >&2
  exit 2
}
[[ "${EDIT_CFG}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "EDIT_CFG must be a non-negative number" >&2
  exit 2
}
required_gpus=8
if [[ "${EVAL_PHASE}" == "all" ]]; then
  (( T2M_SHARDS > required_gpus )) && required_gpus="${T2M_SHARDS}"
  (( EDIT_SHARDS > required_gpus )) && required_gpus="${EDIT_SHARDS}"
elif [[ "${EVAL_PHASE}" == "benchmarks" || "${EVAL_PHASE}" == "postprocess" ]]; then
  required_gpus="${T2M_SHARDS}"
  (( EDIT_SHARDS > required_gpus )) && required_gpus="${EDIT_SHARDS}"
elif [[ "${EVAL_PHASE}" == "edit_benchmarks" ]]; then
  required_gpus="${EDIT_SHARDS}"
elif [[ "${EVAL_PHASE}" == "diagnostics" ]]; then
  required_gpus=6
fi
[[ ${#GPUS[@]} -ge ${required_gpus} ]] || {
  echo "GPU_IDS must provide at least ${required_gpus} GPUs for EVAL_PHASE=${EVAL_PHASE}" >&2
  exit 2
}
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]:0:${required_gpus}}"; do
  [[ -n "${gpu}" && -z "${SEEN_GPUS[${gpu}]:-}" ]] || {
    echo "The first ${required_gpus} GPU_IDS entries must be non-empty and unique" >&2
    exit 2
  }
  SEEN_GPUS["${gpu}"]=1
done
for checkpoint in "${STAGE_A_CHECKPOINT}" "${STAGE_B_CHECKPOINT}" "${EDIT_DIAGNOSTIC_BASELINE_CHECKPOINT}"; do
  [[ -f "${checkpoint}" ]] || {
    echo "Missing Unified Reaction checkpoint: ${checkpoint}" >&2
    exit 2
  }
done
[[ "${EDIT_DIAGNOSTIC_BASELINE_STEP}" =~ ^[1-9][0-9]*$ ]] || {
  echo "EDIT_DIAGNOSTIC_BASELINE_STEP must be a positive integer" >&2
  exit 2
}
[[ "${EDIT_DIAGNOSTIC_BASELINE_LABEL}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "EDIT_DIAGNOSTIC_BASELINE_LABEL contains unsupported characters" >&2
  exit 2
}

"${PYTHON_BIN}" - \
  "${STAGE_A_CHECKPOINT}" \
  "${STAGE_B_CHECKPOINT}" \
  "${STAGE_B_STEP}" \
  "${ALLOW_REACTION_LOSS_ABLATION}" \
  "${ALLOW_SAME_MIX_EXTENSION_AT_STEP}" <<'PY'
import sys
from copy import deepcopy

import torch
from train_hy273_unified_actor import (
    CHECKPOINT_FORMAT,
    validate_config,
    validate_resume_config,
)


def expected_task_counts(segments, checkpoint_step):
    expected = {
        "realized_hml": 0,
        "realized_edit": 0,
        "realized_interaction": 0,
    }
    previous_end = 0
    covered = 0
    paired_key = None
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if start != previous_end or end <= start:
            raise RuntimeError("Task schedule is not contiguous and non-empty")
        current_paired_key = "reaction" if "reaction" in segment else "interaction"
        if paired_key is None:
            paired_key = current_paired_key
        elif paired_key != current_paired_key:
            raise RuntimeError("Task schedule changes paired-task key")
        weights = {
            "realized_hml": int(segment["t2m"]),
            "realized_edit": int(segment["edit"]),
            "realized_interaction": int(segment[current_paired_key]),
        }
        if min(weights.values()) < 0 or sum(weights.values()) != 100:
            raise RuntimeError("Task schedule weights must sum to 100")
        active = max(0, min(end, checkpoint_step) - start)
        if active:
            covered += active
            for key, weight in weights.items():
                numerator = active * weight
                if numerator % 100:
                    raise RuntimeError("Checkpoint has fractional expected task counts")
                expected[key] += numerator // 100
        previous_end = end
    if covered != checkpoint_step:
        raise RuntimeError("Task schedule does not cover the checkpoint step")
    return expected


rows = []
for path, expected_step in zip(sys.argv[1:3], (100_000, int(sys.argv[3]))):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Not a unified checkpoint: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"Checkpoint has no resolved config: {path}")
    validate_config(config)
    if int(checkpoint.get("next_global_step", -1)) != expected_step:
        raise RuntimeError(f"Expected exact step {expected_step}: {path}")
    if config["data"].get("paired_task") != "reaction":
        raise RuntimeError("Final evaluation requires the single-target Reaction model")
    if config["model"].get("source_fusion_mode") != "token_block":
        raise RuntimeError("Final evaluation requires the source token block")
    if config["model"].get("text_token_sequence") != "sentence_plus_context":
        raise RuntimeError("Final evaluation requires the full text token sequence")
    ema_every = int(config["training"]["ema_every"])
    if int(checkpoint.get("ema_update_count", -1)) != expected_step // ema_every:
        raise RuntimeError(f"EMA update count does not match step {expected_step}: {path}")
    batcher = checkpoint.get("batcher")
    if not isinstance(batcher, dict):
        raise RuntimeError(f"Checkpoint has no batcher state: {path}")
    scheduler = batcher.get("scheduler")
    scheduler_state = scheduler.get("state") if isinstance(scheduler, dict) else None
    if not isinstance(scheduler_state, dict) or int(scheduler_state.get("next_step", -1)) != expected_step:
        raise RuntimeError(f"Task scheduler does not match step {expected_step}: {path}")
    config_segments = list(config["schedule"]["segments"])
    if scheduler.get("segments") != config_segments:
        raise RuntimeError(f"Scheduler/config segments differ: {path}")
    expected_counts = expected_task_counts(config_segments, expected_step)
    actual_counts = {
        key: int(scheduler_state.get(key, -1)) for key in expected_counts
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Wrong absolute task counts at {expected_step}: "
            f"{actual_counts} != {expected_counts}"
        )
    if any(
        int(scheduler_state.get(key, -1)) != 0
        for key in ("debt_hml", "debt_edit", "debt_interaction")
    ):
        raise RuntimeError(f"Non-zero task debt at {expected_step}: {path}")
    ordinals = batcher.get("next_global_sample_ordinal")
    cursors = batcher.get("cursors")
    local_batch_sizes = batcher.get("local_batch_sizes")
    world_size = int(batcher.get("world_size", -1))
    if not all(isinstance(value, dict) for value in (ordinals, cursors, local_batch_sizes)):
        raise RuntimeError(f"Incomplete stream state: {path}")
    for state_key, stream_id in (
        ("realized_hml", 0),
        ("realized_edit", 1),
        ("realized_interaction", 2),
    ):
        stream_key = str(stream_id)
        global_batch = int(local_batch_sizes.get(stream_key, -1)) * world_size
        cursor = cursors.get(stream_key)
        expected_ordinal = expected_counts[state_key] * global_batch
        if (
            global_batch <= 0
            or not isinstance(cursor, dict)
            or int(cursor.get("global_batch_size", -1)) != global_batch
            or int(ordinals.get(stream_key, -1)) != expected_ordinal
        ):
            raise RuntimeError(f"Stream {stream_id} is not aligned at {expected_step}")
    rows.append(
        {
            "run_name": str(checkpoint.get("run_name", "")),
            "config": deepcopy(config),
            "normalizer": checkpoint.get("normalizer"),
            "normalization": checkpoint.get("normalization"),
            "batcher_static": {
                key: batcher.get(key)
                for key in (
                    "format",
                    "multitask_manifest",
                    "interaction_root",
                    "run_seed",
                    "world_size",
                    "interaction_exclude_overlength",
                    "paired_task",
                    "local_batch_sizes",
                    "manifest_hashes",
                )
            },
            "schedule": scheduler.get("segments"),
        }
    )


def tensors_equal(first, second):
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if set(first) != set(second):
        return False
    return all(
        torch.equal(first[key], second[key])
        if isinstance(first[key], torch.Tensor) and isinstance(second[key], torch.Tensor)
        else first[key] == second[key]
        for key in first
    )


allow_same_mix_extension_at_step = int(sys.argv[5])
if allow_same_mix_extension_at_step < 0:
    raise RuntimeError("ALLOW_SAME_MIX_EXTENSION_AT_STEP must be non-negative")
if (
    rows[0]["normalization"] != rows[1]["normalization"]
    or not tensors_equal(rows[0]["normalizer"], rows[1]["normalizer"])
    or rows[0]["batcher_static"] != rows[1]["batcher_static"]
):
    raise RuntimeError("Stage-A and Stage-B do not share data or normalization")
allow_reaction_loss_ablation = sys.argv[4] == "1"
stage_a_config = deepcopy(rows[0]["config"])
stage_b_config = deepcopy(rows[1]["config"])
if allow_reaction_loss_ablation:
    stage_a_reaction_loss = stage_a_config.get("reaction_loss")
    stage_b_reaction_loss = stage_b_config.get("reaction_loss")
    if stage_a_reaction_loss == stage_b_reaction_loss:
        raise RuntimeError("Requested Reaction-loss ablation is absent")
    stage_a_config["reaction_loss"] = deepcopy(stage_b_reaction_loss)

if allow_same_mix_extension_at_step:
    if (
        not allow_reaction_loss_ablation
        and rows[0]["run_name"] != rows[1]["run_name"]
    ):
        raise RuntimeError("Same-mix extension checkpoints belong to different runs")
    validate_resume_config(
        stage_a_config,
        stage_b_config,
        allow_same_mix_extension_at_step=allow_same_mix_extension_at_step,
    )
    contracts_match = True
else:
    contracts_match = (
        stage_a_config == stage_b_config
        and (
            allow_reaction_loss_ablation
            or rows[0]["run_name"] == rows[1]["run_name"]
        )
    )
if not contracts_match:
    raise RuntimeError(
        "Stage-A and requested Stage-B checkpoints differ outside the permitted scientific contract"
    )
PY

mkdir -p "${EVAL_ROOT}/logs" "${EVAL_ROOT}/reaction"
PROTOCOL_LOCK="${EVAL_ROOT}/reaction/final_protocol_lock.json"
"${PYTHON_BIN}" - "${STAGE_B_CHECKPOINT}" "${PROTOCOL_LOCK}" <<PY
import json
from pathlib import Path
import sys

checkpoint = str(Path(sys.argv[1]).expanduser().resolve())
output = Path(sys.argv[2]).expanduser().resolve()
payload = {
    "format": "hy273_reaction_eval_cfg_lock_v1",
    "checkpoint": checkpoint,
    "checkpoint_next_global_step": ${STAGE_B_STEP},
    "weight_source": "ema",
    "num_steps": ${ODE_STEPS},
    "source_cfg_scale": ${SOURCE_CFG},
    "text_cfg_scale": ${TEXT_CFG},
    "caption_policy": "${CAPTION_POLICY}",
    "seed": ${EVAL_SEED},
    "selection_policy": "preregistered_fixed_before_val_and_test",
    "splits": ["val", "test"],
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(output)
PY

run_t2m_visual16() {
  local output="${EVAL_ROOT}/t2m/visual16_ema_cfg2"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON_BIN}" \
    tools/eval_hy273_multitask_t2m_visual16.py \
    --checkpoint "${STAGE_B_CHECKPOINT}" \
    --output_dir "${output}" \
    --device cuda:0 \
    --weight_source ema \
    --num_steps "${ODE_STEPS}" \
    --cfg_scale "${TEXT_CFG}" \
    --seed 3407 \
    --max_samples 16
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_samples.py \
    --sample_dir "${output}" \
    --output_dir "${output}/gifs" \
    --max_videos 16 \
    --format gif
}

run_t2m_paraphrase() {
  local output="${EVAL_ROOT}/t2m/paraphrase_ema_cfg2"
  local label="unified_reaction_stage_b_${STAGE_B_K}k_cfg2"
  CUDA_VISIBLE_DEVICES="${GPUS[1]}" "${PYTHON_BIN}" \
    tools/eval_hy273_text_paraphrase_panel.py \
    --checkpoint "${label}=${STAGE_B_CHECKPOINT}" \
    --output_dir "${output}" \
    --device cuda:0 \
    --weight_source ema \
    --num_steps "${ODE_STEPS}" \
    --cfg_scale "${TEXT_CFG}" \
    --target_length 150 \
    --seeds 3407,12345,20260725
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_samples.py \
    --sample_dir "${output}/${label}" \
    --output_dir "${output}/${label}/gifs" \
    --max_videos 15 \
    --format gif
}

run_dynamic_edit() {
  local label="$1"
  local checkpoint="$2"
  local edit_cfg="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    tools/eval_hy273_dynamic_edits.py \
    --mode run \
    --output_dir "${EVAL_ROOT}/edit/dynamic" \
    --checkpoint "${label}=${checkpoint}" \
    --weight_source ema \
    --device cuda:0 \
    --batch_size 4 \
    --limit_per_category 16 \
    --ode_steps "${ODE_STEPS}" \
    --source_cfg_scale "${SOURCE_CFG}" \
    --edit_cfg_scale "${edit_cfg}" \
    --overwrite
}

validate_dynamic_edit_records() {
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" - \
    "${EVAL_ROOT}/edit/dynamic" \
    "${STAGE_A_CHECKPOINT}" \
    "${STAGE_B_CHECKPOINT}" \
    "${STAGE_B_STEP}" \
    "${ODE_STEPS}" \
    "${SOURCE_CFG}" <<'PY'
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve()
stage_a = str(Path(sys.argv[2]).expanduser().resolve())
stage_b = str(Path(sys.argv[3]).expanduser().resolve())
stage_b_step = int(sys.argv[4])
ode_steps = int(sys.argv[5])
source_cfg = float(sys.argv[6])
selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
if (
    int(selection.get("selection_seed", -1)) != 20260724
    or int(selection.get("limit_per_category", -1)) != 16
    or int(selection.get("max_frames", -1)) != 300
):
    raise RuntimeError("Dynamic Edit selection does not match the frozen protocol")
pair_ids = [str(row["pair_id"]) for row in selection["selected_pairs"]]
if len(pair_ids) != len(set(pair_ids)):
    raise RuntimeError("Dynamic Edit selection contains duplicate pairs")
expected = {
    "stageA100k_cfg3": (stage_a, 100000, 3.0),
    f"stageB{stage_b_step // 1000}k_cfg2": (stage_b, stage_b_step, 2.0),
    f"stageB{stage_b_step // 1000}k_cfg3": (stage_b, stage_b_step, 3.0),
}
records = []
for path in sorted((root / "records").glob("*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        records.extend(json.loads(line) for line in handle if line.strip())
if {str(row.get("label")) for row in records} != set(expected):
    raise RuntimeError("Dynamic Edit record labels do not match this evaluation")
seen = set()
for row in records:
    label = str(row["label"])
    pair_id = str(row["pair_id"])
    key = (label, pair_id)
    if key in seen or pair_id not in pair_ids:
        raise RuntimeError(f"Invalid Dynamic Edit record identity: {key}")
    seen.add(key)
    checkpoint, step, edit_cfg = expected[label]
    actual_checkpoint = str(Path(row["checkpoint"]).expanduser().resolve())
    if (
        actual_checkpoint != checkpoint
        or int(row.get("checkpoint_step", -1)) != step
        or row.get("weight_source") != "ema"
        or int(row.get("ode_steps", -1)) != ode_steps
        or not math.isclose(float(row.get("source_cfg_scale", -1.0)), source_cfg)
        or not math.isclose(float(row.get("edit_cfg_scale", -1.0)), edit_cfg)
    ):
        raise RuntimeError(f"Dynamic Edit record has the wrong scientific identity: {key}")
expected_keys = {(label, pair_id) for label in expected for pair_id in pair_ids}
if seen != expected_keys:
    raise RuntimeError("Dynamic Edit records do not cover every selected pair and system")
print(f"Dynamic Edit scientific identity verified: {len(records)} records")
PY
}

run_edit_diagnostics() {
  local root="${EVAL_ROOT}/edit/same_source"
  local ablation_args=()
  if [[ "${EDIT_DIAGNOSTIC_ALLOW_REACTION_LOSS_ABLATION}" == "1" ]]; then
    ablation_args+=(--allow_reaction_loss_ablation)
  fi
  CUDA_VISIBLE_DEVICES="${GPUS[5]}" "${PYTHON_BIN}" \
    tools/eval_hy273_edit_same_source_fixed_t.py \
    --checkpoint "${EDIT_DIAGNOSTIC_BASELINE_LABEL}=${EDIT_DIAGNOSTIC_BASELINE_CHECKPOINT}" \
    --checkpoint "${STAGE_B_LABEL}=${STAGE_B_CHECKPOINT}" \
    --system_expectation "${EDIT_DIAGNOSTIC_BASELINE_LABEL}=${EDIT_DIAGNOSTIC_BASELINE_STEP},none" \
    --system_expectation "${STAGE_B_LABEL}=${STAGE_B_STEP},none" \
    --weight_source ema \
    --timesteps 0,0.05,0.1 \
    --groups_per_batch 4 \
    --ode_steps "${ODE_STEPS}" \
    --ode_groups_per_batch 1 \
    --source_cfg_scale "${SOURCE_CFG}" \
    --edit_cfg_scale 3.0 \
    --direct_comparison "${EDIT_DIAGNOSTIC_BASELINE_LABEL},${STAGE_B_LABEL}" \
    "${ablation_args[@]}" \
    --device cuda:0 \
    --output "${EVAL_ROOT}/edit/same_source_assignment_ema_cfg3.json"
  CUDA_VISIBLE_DEVICES="${GPUS[5]}" "${PYTHON_BIN}" \
    tools/sample_hy273_edit_same_source_visuals.py \
    --checkpoint "${EDIT_DIAGNOSTIC_BASELINE_CHECKPOINT}" \
    --label "${EDIT_DIAGNOSTIC_BASELINE_LABEL}_cfg3" \
    --weight_source ema \
    --ode_steps "${ODE_STEPS}" \
    --source_cfg_scale "${SOURCE_CFG}" \
    --edit_cfg_scale 3.0 \
    --device cuda:0 \
    --output_dir "${root}/samples/${EDIT_DIAGNOSTIC_BASELINE_LABEL}_cfg3"
  CUDA_VISIBLE_DEVICES="${GPUS[5]}" "${PYTHON_BIN}" \
    tools/sample_hy273_edit_same_source_visuals.py \
    --checkpoint "${STAGE_B_CHECKPOINT}" \
    --label "${STAGE_B_LABEL}_cfg3" \
    --weight_source ema \
    --ode_steps "${ODE_STEPS}" \
    --source_cfg_scale "${SOURCE_CFG}" \
    --edit_cfg_scale 3.0 \
    --device cuda:0 \
    --output_dir "${root}/samples/${STAGE_B_LABEL}_cfg3"
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" \
    tools/render_hy273_edit_same_source_comparison.py \
    --system "${EDIT_DIAGNOSTIC_BASELINE_LABEL}_cfg3=${root}/samples/${EDIT_DIAGNOSTIC_BASELINE_LABEL}_cfg3" \
    --system "${STAGE_B_LABEL}_cfg3=${root}/samples/${STAGE_B_LABEL}_cfg3" \
    --branch_system "${STAGE_B_LABEL}_cfg3" \
    --output_dir "${root}/gifs"
}

run_reaction_split() {
  local split="$1"
  local gpu="$2"
  local split_root="${EVAL_ROOT}/reaction/${split}"
  local output_json="${split_root}/reaction_${split}.json"
  local extra=()
  if [[ "${split}" == "test" ]]; then
    extra+=(--save_predictions)
  fi
  mkdir -p "${split_root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" tools/eval_hy273_reaction.py \
    --checkpoint "${STAGE_B_CHECKPOINT}" \
    --split "${split}" \
    --weight_source ema \
    --device cuda:0 \
    --batch_size 8 \
    --num_steps "${ODE_STEPS}" \
    --source_cfg_scale "${SOURCE_CFG}" \
    --text_cfg_scale "${TEXT_CFG}" \
    --seed "${EVAL_SEED}" \
    --caption_policy "${CAPTION_POLICY}" \
    --bootstrap_resamples 2000 \
    --final_protocol_lock "${PROTOCOL_LOCK}" \
    --require_final_protocol \
    --output_json "${output_json}" \
    "${extra[@]}"
  if [[ "${split}" == "test" ]]; then
    CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" tools/render_hy273_reaction_review.py \
      --report_json "${output_json}" \
      --output_dir "${split_root}/gifs_action_balanced" \
      --max_videos 12 \
      --joint_source position \
      --fps 30 \
      --stride 3
  fi
}

wait_for_jobs() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  return "${failed}"
}

run_t2m_full_benchmark() {
  local root="${EVAL_ROOT}/t2m/full_test4042_ema_cfg2"
  local preflight="${root}/preflight_manifest.json"
  local preflight_sha
  local shard
  local shard_pids=()
  local common=(
    --checkpoint "${STAGE_B_CHECKPOINT}"
    --output_dir "${root}"
    --weight_source ema
    --num_shards "${T2M_SHARDS}"
    --batch_size 16
    --num_steps "${ODE_STEPS}"
    --cfg_scale "${TEXT_CFG}"
    --seed 3407
  )
  mkdir -p "${root}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON_BIN}" \
    tools/eval_hy273_t2m_nonregression.py \
    "${common[@]}" \
    --device cuda:0 \
    --preflight_only \
    >"${EVAL_ROOT}/logs/t2m_full_preflight.log" 2>&1
  preflight_sha="$(sha256sum "${preflight}" | cut -d' ' -f1)"
  for ((shard = 0; shard < T2M_SHARDS; shard++)); do
    CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${PYTHON_BIN}" \
      tools/eval_hy273_t2m_nonregression.py \
      "${common[@]}" \
      --device cuda:0 \
      --shard_id "${shard}" \
      --preflight_manifest "${preflight}" \
      --preflight_sha256 "${preflight_sha}" \
      >"${EVAL_ROOT}/logs/t2m_full_shard_${shard}.log" 2>&1 &
    shard_pids+=("$!")
  done
  wait_for_jobs "${shard_pids[@]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON_BIN}" \
    tools/eval_hy273_t2m_nonregression.py \
    "${common[@]}" \
    --device cuda:0 \
    --preflight_manifest "${preflight}" \
    --preflight_sha256 "${preflight_sha}" \
    --aggregate \
    >"${EVAL_ROOT}/logs/t2m_full_aggregate.log" 2>&1
}

run_motionfix_full_benchmark() {
  local protocol="$1"
  local label="$2"
  local root="${EVAL_ROOT}/edit/${label}_ema_sourcecfg2_editcfg${EDIT_CFG_TAG}"
  local preflight="${root}/preflight_manifest.json"
  local preflight_sha
  local manifest
  local counterfactual_manifest
  local counterfactual_summary
  local shard
  local shard_pids=()
  if [[ "${protocol}" == "motionfix_val_selected_internal_k273_v1" ]]; then
    manifest="${MULTITASK_MANIFEST_ROOT}/val.jsonl"
    counterfactual_manifest="${EDIT_COUNTERFACTUAL_ROOT}/motionfix_val_counterfactual_manifest_v1.jsonl"
    counterfactual_summary="${EDIT_COUNTERFACTUAL_ROOT}/motionfix_val_counterfactual_manifest_v1_summary.json"
  elif [[ "${protocol}" == "motionfix_full_requested_length_1013_internal_k273_v1" ]]; then
    manifest="${MULTITASK_MANIFEST_ROOT}/test.jsonl"
    counterfactual_manifest="${EDIT_COUNTERFACTUAL_ROOT}/motionfix_edit_counterfactual_manifest_v1.jsonl"
    counterfactual_summary="${EDIT_COUNTERFACTUAL_ROOT}/motionfix_edit_counterfactual_manifest_v1_summary.json"
  else
    echo "Unsupported full MotionFix protocol: ${protocol}" >&2
    return 2
  fi
  local common=(
    --checkpoint "${STAGE_B_CHECKPOINT}"
    --manifest "${manifest}"
    --counterfactual_manifest "${counterfactual_manifest}"
    --counterfactual_manifest_sha256 ""
    --counterfactual_summary "${counterfactual_summary}"
    --counterfactual_summary_sha256 ""
    --output_dir "${root}"
    --protocol "${protocol}"
    --systems "${EDIT_SYSTEMS}"
    --weight_source ema
    --num_shards "${EDIT_SHARDS}"
    --num_steps "${ODE_STEPS}"
    --source_cfg_scale "${SOURCE_CFG}"
    --edit_cfg_scale "${EDIT_CFG}"
    --generate_text_cfg_scale "${TEXT_CFG}"
    --seed 3407
    --bootstrap_samples 10000
  )
  mkdir -p "${root}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON_BIN}" \
    tools/eval_hy273_motionfix_edit.py \
    "${common[@]}" \
    --device cuda:0 \
    --preflight_only \
    >"${EVAL_ROOT}/logs/edit_${label}_preflight.log" 2>&1
  preflight_sha="$(sha256sum "${preflight}" | cut -d' ' -f1)"
  for ((shard = 0; shard < EDIT_SHARDS; shard++)); do
    CUDA_VISIBLE_DEVICES="${GPUS[${shard}]}" "${PYTHON_BIN}" \
      tools/eval_hy273_motionfix_edit.py \
      "${common[@]}" \
      --device cuda:0 \
      --shard_id "${shard}" \
      --preflight_manifest "${preflight}" \
      --preflight_sha256 "${preflight_sha}" \
      >"${EVAL_ROOT}/logs/edit_${label}_shard_${shard}.log" 2>&1 &
    shard_pids+=("$!")
  done
  wait_for_jobs "${shard_pids[@]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON_BIN}" \
    tools/eval_hy273_motionfix_edit.py \
    "${common[@]}" \
    --device cuda:0 \
    --preflight_manifest "${preflight}" \
    --preflight_sha256 "${preflight_sha}" \
    --aggregate \
    >"${EVAL_ROOT}/logs/edit_${label}_aggregate.log" 2>&1
}

if [[ "${EVAL_PHASE}" == "benchmarks" ]]; then
  run_t2m_full_benchmark
  run_motionfix_full_benchmark motionfix_val_selected_internal_k273_v1 full_val330
  run_motionfix_full_benchmark motionfix_full_requested_length_1013_internal_k273_v1 full_test1013
  echo "Unified Reaction ${STAGE_B_LABEL} benchmark evaluation complete: ${EVAL_ROOT}"
  exit 0
fi
if [[ "${EVAL_PHASE}" == "edit_benchmarks" ]]; then
  run_motionfix_full_benchmark motionfix_val_selected_internal_k273_v1 full_val330
  run_motionfix_full_benchmark motionfix_full_requested_length_1013_internal_k273_v1 full_test1013
  echo "Unified Reaction ${STAGE_B_LABEL} Edit benchmark evaluation complete: ${EVAL_ROOT}"
  exit 0
fi
if [[ "${EVAL_PHASE}" == "diagnostics" ]]; then
  run_edit_diagnostics
  echo "Unified Reaction ${STAGE_B_LABEL} Edit diagnostics complete: ${EVAL_ROOT}"
  exit 0
fi
if [[ "${EVAL_PHASE}" == "postprocess" ]]; then
  validate_dynamic_edit_records
  "${PYTHON_BIN}" tools/eval_hy273_dynamic_edits.py \
    --mode aggregate \
    --output_dir "${EVAL_ROOT}/edit/dynamic" \
    --labels "stageA100k_cfg3,${STAGE_B_LABEL}_cfg2,${STAGE_B_LABEL}_cfg3" \
    --limit_per_category 16 \
    --render_per_category 3 \
    --render_stride 3
  run_t2m_full_benchmark
  run_motionfix_full_benchmark motionfix_val_selected_internal_k273_v1 full_val330
  run_motionfix_full_benchmark motionfix_full_requested_length_1013_internal_k273_v1 full_test1013
  echo "Unified Reaction ${STAGE_B_LABEL} postprocessing complete: ${EVAL_ROOT}"
  exit 0
fi
[[ "${EVAL_PHASE}" == "all" ]] || {
  echo "EVAL_PHASE must be all, benchmarks, edit_benchmarks, diagnostics, or postprocess" >&2
  exit 2
}

"${PYTHON_BIN}" tools/eval_hy273_dynamic_edits.py \
  --mode prepare \
  --output_dir "${EVAL_ROOT}/edit/dynamic" \
  --limit_per_category 16 \
  --max_frames 300 \
  --seed 20260724

run_t2m_visual16 >"${EVAL_ROOT}/logs/t2m_visual16.log" 2>&1 &
pids=("$!")
run_t2m_paraphrase >"${EVAL_ROOT}/logs/t2m_paraphrase.log" 2>&1 &
pids+=("$!")
run_dynamic_edit stageA100k_cfg3 "${STAGE_A_CHECKPOINT}" 3.0 "${GPUS[2]}" \
  >"${EVAL_ROOT}/logs/edit_dynamic_stageA_cfg3.log" 2>&1 &
pids+=("$!")
run_dynamic_edit "${STAGE_B_LABEL}_cfg2" "${STAGE_B_CHECKPOINT}" 2.0 "${GPUS[3]}" \
  >"${EVAL_ROOT}/logs/edit_dynamic_stageB_cfg2.log" 2>&1 &
pids+=("$!")
run_dynamic_edit "${STAGE_B_LABEL}_cfg3" "${STAGE_B_CHECKPOINT}" 3.0 "${GPUS[4]}" \
  >"${EVAL_ROOT}/logs/edit_dynamic_stageB_cfg3.log" 2>&1 &
pids+=("$!")
run_edit_diagnostics >"${EVAL_ROOT}/logs/edit_diagnostics.log" 2>&1 &
pids+=("$!")
run_reaction_split val "${GPUS[6]}" >"${EVAL_ROOT}/logs/reaction_val.log" 2>&1 &
pids+=("$!")
run_reaction_split test "${GPUS[7]}" >"${EVAL_ROOT}/logs/reaction_test.log" 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "One or more Unified Reaction evaluations failed; inspect ${EVAL_ROOT}/logs" >&2
  exit "${status}"
fi

validate_dynamic_edit_records
"${PYTHON_BIN}" tools/eval_hy273_dynamic_edits.py \
  --mode aggregate \
  --output_dir "${EVAL_ROOT}/edit/dynamic" \
  --labels "stageA100k_cfg3,${STAGE_B_LABEL}_cfg2,${STAGE_B_LABEL}_cfg3" \
  --limit_per_category 16 \
  --render_per_category 3 \
  --render_stride 3

run_t2m_full_benchmark
run_motionfix_full_benchmark motionfix_val_selected_internal_k273_v1 full_val330
run_motionfix_full_benchmark motionfix_full_requested_length_1013_internal_k273_v1 full_test1013

echo "Unified Reaction ${STAGE_B_LABEL} evaluation complete: ${EVAL_ROOT}"
