#!/usr/bin/env python3
"""Fixed-role Inter-X Reaction evaluation for unified HY273 checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_reaction_dataset import (
    HY273ReactionDataset,
    ReactionConditionPattern,
    ReactionSamplePlan,
    collate_hy273_reaction,
)
from models.raw_motion.hy273_multitask_condition import ConditionBatch, SourceRole
from models.raw_motion.hy273_reaction_metrics import reaction_fixed_role_metrics
from models.raw_motion.hy273_slices import DIM_HY273
from sample_hy273_multitask import (
    create_model_from_checkpoint,
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from train_hy273_unified_actor import CHECKPOINT_FORMAT, validate_config


VARIANTS = (
    "source_text",
    "source_only",
    "empty",
    "shuffled_text",
    "unrelated_source",
)
CAUSAL_ADVANTAGE_DIRECTIONS = {
    "reactor_position_mpjpe_cm": "lower",
    "reactor_fk_mpjpe_cm": "lower",
    "reactor_root_error_cm": "lower",
    "relative_heading_error_deg": "lower",
    "position_relation_distance_mae_cm": "lower",
    "fk_relation_distance_mae_cm": "lower",
    "close_event_error_20cm": "lower",
    "reactor_contact_accuracy": "higher",
    "reactor_contact_f1": "higher",
    "reactor_fk_jerk_error_mps3": "lower",
}
CAPTION_POLICIES = ("first", "uid_balanced")
FINAL_PROTOCOL_LOCK_FORMAT = "hy273_reaction_eval_cfg_lock_v1"
FINAL_CHECKPOINT_STEP = 200_000
FINAL_WEIGHT_SOURCE = "ema"
FINAL_NUM_STEPS = 32
FINAL_SOURCE_CFG_SCALE = 2.0
FINAL_TEXT_CFG_SCALE = 2.0
FINAL_CAPTION_POLICY = "uid_balanced"
FINAL_SEED = 20260801
FINAL_SELECTION_POLICY = "preregistered_fixed_before_val_and_test"
FINAL_SPLITS = ["val", "test"]


def _validate_final_protocol_runtime(
    *,
    checkpoint_step: int,
    weight_source: str,
    num_steps: int,
    source_cfg_scale: float,
    text_cfg_scale: float,
    caption_policy: str,
    seed: int,
    expected_checkpoint_step: int = FINAL_CHECKPOINT_STEP,
    start_index: int = 0,
    num_samples: int | None = None,
) -> None:
    actual = {
        "checkpoint_next_global_step": int(checkpoint_step),
        "weight_source": str(weight_source),
        "num_steps": int(num_steps),
        "source_cfg_scale": float(source_cfg_scale),
        "text_cfg_scale": float(text_cfg_scale),
        "caption_policy": str(caption_policy),
        "seed": int(seed),
    }
    canonical = {
        "checkpoint_next_global_step": int(expected_checkpoint_step),
        "weight_source": FINAL_WEIGHT_SOURCE,
        "num_steps": FINAL_NUM_STEPS,
        "source_cfg_scale": FINAL_SOURCE_CFG_SCALE,
        "text_cfg_scale": FINAL_TEXT_CFG_SCALE,
        "caption_policy": FINAL_CAPTION_POLICY,
        "seed": FINAL_SEED,
    }
    mismatches = {
        key: {"canonical": value, "runtime": actual[key]}
        for key, value in canonical.items()
        if actual[key] != value
    }
    if mismatches:
        raise RuntimeError(
            "Final Reaction benchmark does not use the canonical protocol: "
            f"{mismatches}"
        )
    if int(start_index) != 0 or num_samples is not None:
        raise RuntimeError(
            "Final Reaction benchmark must evaluate the complete requested split"
        )


def _load_final_protocol_lock(
    path: str | Path,
    *,
    checkpoint_path: Path,
    checkpoint_step: int,
    weight_source: str,
    num_steps: int,
    source_cfg_scale: float,
    text_cfg_scale: float,
    caption_policy: str,
    seed: int,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing Reaction final-protocol lock: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("format") != FINAL_PROTOCOL_LOCK_FORMAT:
        raise RuntimeError("Unsupported Reaction final-protocol lock format")
    locked_step = int(payload.get("checkpoint_next_global_step", -1))
    if locked_step <= 100_000:
        raise RuntimeError("Reaction final-protocol lock requires a Stage-B checkpoint")
    _validate_final_protocol_runtime(
        checkpoint_step=checkpoint_step,
        weight_source=weight_source,
        num_steps=num_steps,
        source_cfg_scale=source_cfg_scale,
        text_cfg_scale=text_cfg_scale,
        caption_policy=caption_policy,
        seed=seed,
        expected_checkpoint_step=locked_step,
    )
    expected = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_next_global_step": int(checkpoint_step),
        "weight_source": str(weight_source),
        "num_steps": int(num_steps),
        "source_cfg_scale": float(source_cfg_scale),
        "text_cfg_scale": float(text_cfg_scale),
        "caption_policy": str(caption_policy),
        "seed": int(seed),
    }
    mismatches = {
        key: {"locked": payload.get(key), "runtime": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Reaction final-protocol lock does not match runtime arguments: "
            f"{mismatches}"
        )
    if payload.get("selection_policy") != FINAL_SELECTION_POLICY:
        raise RuntimeError("Reaction final-protocol selection policy is not fixed")
    if payload.get("splits") != FINAL_SPLITS:
        raise RuntimeError("Reaction final-protocol lock must cover val and test")
    return {**payload, "path": str(resolved)}


def _clear_source(condition: ConditionBatch) -> ConditionBatch:
    roles = torch.full_like(condition.source_role_id, int(SourceRole.NULL))
    output = replace(
        condition,
        source_motion=torch.zeros_like(condition.source_motion),
        source_present=torch.zeros_like(condition.source_present),
        source_time_valid=torch.zeros_like(condition.source_time_valid),
        source_value_mask=torch.zeros_like(condition.source_value_mask),
        source_role_id=roles,
        source_native_lengths=torch.zeros_like(condition.source_native_lengths),
        target_to_source_time_map=(
            None
            if condition.target_to_source_time_map is None
            else torch.zeros_like(condition.target_to_source_time_map)
        ),
    )
    output.validate(v1_strict=False)
    return output


def _unrelated_source(
    condition: ConditionBatch,
    donor_condition: ConditionBatch,
) -> ConditionBatch:
    if condition.batch_size != donor_condition.batch_size:
        raise ValueError("Reaction case and donor batches must have equal size")
    if condition.source_slots != donor_condition.source_slots:
        raise ValueError("Reaction case and donor source-slot counts differ")
    output = replace(
        condition,
        source_motion=donor_condition.source_motion.clone(),
        source_present=donor_condition.source_present.clone(),
        source_time_valid=donor_condition.source_time_valid.clone(),
        source_value_mask=donor_condition.source_value_mask.clone(),
        source_native_lengths=donor_condition.source_native_lengths.clone(),
        target_to_source_time_map=None,
    )
    # Keep the current case's P1/P2 semantic role. Donors are role-matched, while
    # this makes the intended ablation explicit at the ConditionBatch boundary.
    output.validate()
    return output


def _length_bucket(length: int) -> str:
    if length <= 90:
        return "short_le_90"
    if length <= 180:
        return "medium_91_180"
    return "long_gt_180"


def _action_category(uid: str) -> str:
    if len(uid) < 12 or uid[0] != "G" or uid[4] != "T" or uid[8] != "A":
        raise ValueError(f"Invalid Inter-X clip identifier: {uid!r}")
    return uid[8:12]


def _caption_index(
    dataset: HY273ReactionDataset,
    index: int,
    policy: str,
) -> int:
    texts = dataset.rows[int(index)]["texts"]
    if not texts:
        raise ValueError(f"Reaction row has no captions: {dataset.uid(index)}")
    if policy == "first":
        return 0
    if policy == "uid_balanced":
        # The manifest order is fixed, and modulo assignment keeps the three
        # Inter-X captions approximately balanced without multiplying test cost.
        return int(index) % len(texts)
    raise ValueError(f"Unsupported caption policy: {policy!r}")


def _case_noise_seed(base_seed: int, uid: str, caption_index: int) -> int:
    payload = f"{uid}:{int(caption_index)}".encode("utf-8")
    value = (1469598103934665603 ^ int(base_seed)) & ((1 << 64) - 1)
    for byte in payload:
        value ^= int(byte)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return value & ((1 << 63) - 1)


def _build_split_donor_map(
    dataset: HY273ReactionDataset,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Build one deterministic, batch-independent donor derangement per split."""

    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for index, row in enumerate(dataset.rows):
        uid = dataset.uid(index)
        record = {
            "index": int(index),
            "uid": uid,
            "action": _action_category(uid),
            "role": int(dataset.actor_order[uid]),
            "length": min(int(row["frames"]), int(dataset.max_frames)),
        }
        record["length_bucket"] = _length_bucket(int(record["length"]))
        groups.setdefault(
            (int(record["role"]), str(record["length_bucket"])), []
        ).append(record)

    donor_map: dict[int, int] = {}
    group_reports: dict[str, Any] = {}
    for (role, length_bucket), records in sorted(groups.items()):
        ordered = sorted(
            records,
            key=lambda row: (str(row["action"]), int(row["length"]), str(row["uid"])),
        )
        if len(ordered) < 2:
            raise RuntimeError(
                "Cannot construct a donor derangement for "
                f"role={role}, length_bucket={length_bucket}"
            )
        best: tuple[tuple[int, int], int] | None = None
        for shift in range(1, len(ordered)):
            donors = ordered[shift:] + ordered[:shift]
            if any(
                case["uid"] == donor["uid"]
                or case["action"] == donor["action"]
                for case, donor in zip(ordered, donors)
            ):
                continue
            total_length_delta = sum(
                abs(int(case["length"]) - int(donor["length"]))
                for case, donor in zip(ordered, donors)
            )
            score = (-total_length_delta, -shift)
            if best is None or score > best[0]:
                best = (score, shift)
        if best is None:
            raise RuntimeError(
                "No action-disjoint donor derangement for "
                f"role={role}, length_bucket={length_bucket}"
            )
        shift = best[1]
        donors = ordered[shift:] + ordered[:shift]
        for case, donor in zip(ordered, donors):
            donor_map[int(case["index"])] = int(donor["index"])
        group_name = f"person_{role}/{length_bucket}"
        group_reports[group_name] = {
            "count": len(ordered),
            "cyclic_shift": int(shift),
            "mean_absolute_frame_difference": float(
                -best[0][0] / len(ordered)
            ),
        }

    if set(donor_map) != set(range(len(dataset))):
        raise RuntimeError("Donor map does not cover the complete Reaction split")
    if len(set(donor_map.values())) != len(dataset):
        raise RuntimeError("Donor map is not one-to-one")
    for index, donor_index in donor_map.items():
        case_uid = dataset.uid(index)
        donor_uid = dataset.uid(donor_index)
        if (
            case_uid == donor_uid
            or _action_category(case_uid) == _action_category(donor_uid)
            or dataset.actor_order[case_uid] != dataset.actor_order[donor_uid]
            or _length_bucket(int(dataset.rows[index]["frames"]))
            != _length_bucket(int(dataset.rows[donor_index]["frames"]))
        ):
            raise RuntimeError("Constructed donor map violates its hard constraints")
    return donor_map, {
        "format": "interx_split_role_length_action_derangement_v1",
        "scope": "complete_filtered_split",
        "one_to_one": True,
        "hard_constraints": [
            "donor_uid_differs",
            "donor_action_Axxx_differs",
            "source_person_role_matches",
            "length_bucket_matches",
        ],
        "groups": group_reports,
    }


def _bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(int(seed))
    if values.size == 1:
        means = values.copy()
    else:
        means = np.empty(int(resamples), dtype=np.float64)
        chunk = max(1, min(512, int(resamples)))
        for start in range(0, int(resamples), chunk):
            stop = min(start + chunk, int(resamples))
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            means[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }


def _aggregate_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, dict[str, float | int]]:
    metric_names = sorted(
        key
        for key, value in rows[0].items()
        if isinstance(value, float) and np.isfinite(value)
    )
    return {
        metric: _bootstrap_mean(
            np.asarray([float(row[metric]) for row in rows]),
            seed=int(seed) + index,
            resamples=resamples,
            confidence=confidence,
        )
        for index, metric in enumerate(metric_names)
    }


def _stratify(
    rows: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {
        name: {
            "count": len(group),
            "metrics": _aggregate_rows(
                group,
                seed=seed + index * 1000,
                resamples=resamples,
                confidence=confidence,
            ),
        }
        for index, (name, group) in enumerate(sorted(groups.items()))
    }


def _matched_advantage(
    correct: list[dict[str, Any]],
    ablated: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, Any]:
    if [row["uid"] for row in correct] != [row["uid"] for row in ablated]:
        raise ValueError("Matched variants have different case order")
    metric_names = sorted(
        key
        for key, value in correct[0].items()
        if (
            isinstance(value, float)
            and key in ablated[0]
            and key in CAUSAL_ADVANTAGE_DIRECTIONS
        )
    )
    output = {}
    for index, metric in enumerate(metric_names):
        correct_values = np.asarray([float(row[metric]) for row in correct])
        ablated_values = np.asarray([float(row[metric]) for row in ablated])
        direction = CAUSAL_ADVANTAGE_DIRECTIONS[metric]
        advantage = (
            correct_values - ablated_values
            if direction == "higher"
            else ablated_values - correct_values
        )
        output[metric] = {
            **_bootstrap_mean(
                advantage,
                seed=seed + index,
                resamples=resamples,
                confidence=confidence,
            ),
            "metric_direction": direction,
            "positive_means_source_text_is_better": True,
        }
    return output


def _make_sample(
    dataset: HY273ReactionDataset,
    index: int,
    *,
    caption_policy: str,
) -> dict[str, Any]:
    uid = dataset.uid(index)
    caption_index = _caption_index(dataset, index, caption_policy)
    return dataset.materialize(
        ReactionSamplePlan(
            global_step=100_000,
            global_sample_ordinal=int(index),
            row_index=int(index),
            uid=uid,
            caption_index=caption_index,
            crop_start=0,
            yaw_u64=1 << 63,
            condition_pattern=ReactionConditionPattern.SOURCE_AND_TEXT,
        )
    )


def _selected_indices(total: int, start: int, count: int | None) -> list[int]:
    if start < 0 or start >= total:
        raise ValueError("start_index is outside the dataset")
    stop = total if count is None else min(total, start + int(count))
    if stop <= start:
        raise ValueError("num_samples must select at least one case")
    return list(range(start, stop))


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("Reaction benchmark requires a unified checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint has no resolved config")
    validate_config(config)
    if config["data"].get("paired_task") != "reaction":
        raise RuntimeError("Checkpoint is not the single-target Reaction architecture")
    if config["model"].get("source_fusion_mode") != "token_block":
        raise RuntimeError("Reaction checkpoint does not use the source token block")
    if config["model"].get("text_token_sequence") != "sentence_plus_context":
        raise RuntimeError("Reaction checkpoint does not use the full main text sequence")
    next_step = int(checkpoint.get("next_global_step", -1))
    if next_step <= 100_000 and not args.allow_stage_a_diagnostic:
        raise RuntimeError("Reaction benchmark requires a checkpoint trained in Stage B")
    final_protocol_lock = None
    if args.final_protocol_lock:
        final_protocol_lock = _load_final_protocol_lock(
            args.final_protocol_lock,
            checkpoint_path=checkpoint_path,
            checkpoint_step=next_step,
            weight_source=args.weight_source,
            num_steps=args.num_steps,
            source_cfg_scale=args.source_cfg_scale,
            text_cfg_scale=args.text_cfg_scale,
            caption_policy=args.caption_policy,
            seed=args.seed,
        )
    if args.require_final_protocol:
        expected_checkpoint_step = (
            int(final_protocol_lock["checkpoint_next_global_step"])
            if final_protocol_lock is not None
            else FINAL_CHECKPOINT_STEP
        )
        _validate_final_protocol_runtime(
            checkpoint_step=next_step,
            weight_source=args.weight_source,
            num_steps=args.num_steps,
            source_cfg_scale=args.source_cfg_scale,
            text_cfg_scale=args.text_cfg_scale,
            caption_policy=args.caption_policy,
            seed=args.seed,
            expected_checkpoint_step=expected_checkpoint_step,
            start_index=args.start_index,
            num_samples=args.num_samples,
        )
        if final_protocol_lock is None:
            raise RuntimeError(
                "Final Reaction benchmark requires --final_protocol_lock"
            )

    device = torch.device(args.device)
    model = create_model_from_checkpoint(checkpoint).to(device)
    weights = checkpoint[args.weight_source]
    model.load_state_dict(weights, strict=True)
    model.eval()
    normalizer = normalizer_from_checkpoint(checkpoint, device)

    dataset = HY273ReactionDataset(
        args.reaction_root,
        split=args.split,
        max_frames=int(config["data"]["max_target_frames"]),
        exclude_overlength=True,
        exclude_known_test_anomalies=True,
    )
    indices = _selected_indices(len(dataset), args.start_index, args.num_samples)
    donor_map, donor_protocol = _build_split_donor_map(dataset)
    rows_by_variant: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in VARIANTS
    }
    prediction_dir = None
    if args.save_predictions:
        prediction_dir = Path(args.output_json).expanduser().resolve().parent / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(indices), args.batch_size):
        real_indices = indices[batch_start : batch_start + args.batch_size]
        donor_indices = [donor_map[index] for index in real_indices]
        samples = [
            _make_sample(dataset, index, caption_policy=args.caption_policy)
            for index in real_indices
        ]
        donor_samples = [
            _make_sample(dataset, index, caption_policy=args.caption_policy)
            for index in donor_indices
        ]
        collated = collate_hy273_reaction(samples)
        donor_collated = collate_hy273_reaction(donor_samples)
        condition: ConditionBatch = collated["condition"]
        donor_condition: ConditionBatch = donor_collated["condition"]
        source = condition.source_motion[:, 0].clone()
        target = collated["target_motion"].clone()
        texts = list(collated["texts"])
        lengths = condition.requested_target_len.clone()
        frames = condition.target_frames
        initial_noise = torch.stack(
            [
                torch.randn(
                    frames,
                    DIM_HY273,
                    device=device,
                    dtype=next(model.parameters()).dtype,
                    generator=torch.Generator(device=device).manual_seed(
                        _case_noise_seed(
                            int(args.seed),
                            str(sample["uid"]),
                            int(sample["plan"].caption_index),
                        )
                    ),
                )
                for sample in samples
            ],
            dim=0,
        )
        zeros = torch.zeros(condition.batch_size, frames, DIM_HY273)
        zero_mask = torch.zeros_like(zeros, dtype=torch.bool)
        empty_condition = _clear_source(condition)
        unrelated_condition = _unrelated_source(condition, donor_condition)
        shuffled_text = list(donor_collated["texts"])
        specs: dict[str, tuple[ConditionBatch, list[str]]] = {
            "source_text": (condition, texts),
            "source_only": (condition, [""] * len(texts)),
            "empty": (empty_condition, [""] * len(texts)),
            "shuffled_text": (condition, shuffled_text),
            "unrelated_source": (unrelated_condition, texts),
        }
        outputs: dict[str, torch.Tensor] = {}
        for variant, (variant_condition, variant_texts) in specs.items():
            sample = sample_hy273_multitask_ode(
                model,
                normalizer,
                variant_condition,
                variant_texts,
                zeros,
                zero_mask,
                num_steps=args.num_steps,
                text_cfg_scale=args.text_cfg_scale,
                source_cfg_scale=args.source_cfg_scale,
                cfg_apply_contacts=True,
                initial_unified_noise=initial_noise,
                diagnostic_allow_source_absent_edit=(variant == "empty"),
            )
            outputs[variant] = sample.raw_motion.detach().cpu()

        real_count = len(real_indices)
        for variant, prediction in outputs.items():
            result = reaction_fixed_role_metrics(
                source[:real_count],
                prediction[:real_count],
                target[:real_count],
                lengths=lengths[:real_count],
                fps=30.0,
            )
            for local_index, metric_row in enumerate(result["per_sample"]):
                sample = samples[local_index]
                donor_sample = donor_samples[local_index]
                uid = str(sample["uid"])
                metric_row.update(
                    {
                        "uid": uid,
                        "dataset_index": int(real_indices[local_index]),
                        "actor_person_index": int(sample["actor_person_index"]),
                        "action_category": _action_category(uid),
                        "length_bucket": _length_bucket(int(metric_row["length"])),
                        "caption_index": int(sample["plan"].caption_index),
                        "negative_donor_uid": str(donor_sample["uid"]),
                        "negative_donor_action_category": _action_category(
                            str(donor_sample["uid"])
                        ),
                        "variant": variant,
                    }
                )
                rows_by_variant[variant].append(metric_row)
        if prediction_dir is not None:
            for local_index in range(real_count):
                uid = str(samples[local_index]["uid"])
                case_dir = prediction_dir / uid
                case_dir.mkdir(parents=True, exist_ok=True)
                np.save(case_dir / "source.npy", source[local_index].numpy())
                np.save(case_dir / "target.npy", target[local_index].numpy())
                for variant, prediction in outputs.items():
                    np.save(
                        case_dir / f"{variant}.npy",
                        prediction[local_index].numpy(),
                    )
                (case_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "uid": uid,
                            "text": texts[local_index],
                            "caption_index": int(
                                samples[local_index]["plan"].caption_index
                            ),
                            "negative_donor_uid": str(
                                donor_samples[local_index]["uid"]
                            ),
                            "negative_donor_text": shuffled_text[local_index],
                            "actor_person_index": int(
                                samples[local_index]["actor_person_index"]
                            ),
                            "length": int(lengths[local_index]),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        print(
            json.dumps(
                {
                    "event": "reaction_eval_progress",
                    "completed": min(batch_start + len(real_indices), len(indices)),
                    "total": len(indices),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    aggregate = {
        variant: _aggregate_rows(
            rows,
            seed=args.seed + variant_index * 10_000,
            resamples=args.bootstrap_resamples,
            confidence=args.confidence,
        )
        for variant_index, (variant, rows) in enumerate(rows_by_variant.items())
    }
    stratified = {
        variant: {
            field: _stratify(
                rows,
                field,
                seed=args.seed + variant_index * 100_000 + field_index * 10_000,
                resamples=args.bootstrap_resamples,
                confidence=args.confidence,
            )
            for field_index, field in enumerate(
                ("actor_person_index", "action_category", "length_bucket")
            )
        }
        for variant_index, (variant, rows) in enumerate(rows_by_variant.items())
    }
    causal_advantage = {
        variant: _matched_advantage(
            rows_by_variant["source_text"],
            rows_by_variant[variant],
            seed=args.seed + variant_index * 10_000,
            resamples=args.bootstrap_resamples,
            confidence=args.confidence,
        )
        for variant_index, variant in enumerate(VARIANTS[1:])
    }
    return {
        "format": "hy273_fixed_role_reaction_eval_v2",
        "assignment_rule": "fixed_source_actor_to_target_reactor_no_swap",
        "checkpoint": str(checkpoint_path),
        "checkpoint_next_global_step": next_step,
        "weight_source": args.weight_source,
        "dataset": "Inter-X K273",
        "split": args.split,
        "excluded_known_anomalies": sorted(
            ["G046T007A038R019"] if args.split == "test" else []
        ),
        "selection": {
            "start_index": args.start_index,
            "count": len(indices),
            "dataset_count_after_filters": len(dataset),
        },
        "sampling": {
            "num_steps": args.num_steps,
            "source_cfg_scale": args.source_cfg_scale,
            "text_cfg_scale": args.text_cfg_scale,
            "matched_initial_noise_across_variants": True,
            "initial_noise_policy": "deterministic_per_uid_and_caption",
            "seed": args.seed,
        },
        "caption_policy": args.caption_policy,
        "negative_donor_protocol": donor_protocol,
        "protocols": {
            "official_comparable": "source_only actor-motion-conditioned Reaction",
            "unified_extension": "source_text actor-motion-plus-language Reaction",
            "causal_ablations": ["empty", "shuffled_text", "unrelated_source"],
            "test_policy": (
                "select CFG on val; run locked test once"
                if final_protocol_lock is None
                else str(final_protocol_lock["selection_policy"])
            ),
        },
        "final_protocol_lock": final_protocol_lock,
        "aggregate": aggregate,
        "causal_advantage": causal_advantage,
        "stratified": stratified,
        "per_sample": rows_by_variant,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--reaction_root",
        default="/mnt/afs/mogo_base/datasets/InteractionK273/interx",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--weight_source", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--source_cfg_scale", type=float, default=2.0)
    parser.add_argument("--text_cfg_scale", type=float, default=2.0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--caption_policy",
        choices=CAPTION_POLICIES,
        default="uid_balanced",
        help="Use caption 0 or one deterministic balanced caption per Inter-X clip.",
    )
    parser.add_argument("--bootstrap_resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--allow_stage_a_diagnostic", action="store_true")
    parser.add_argument("--final_protocol_lock", default="")
    parser.add_argument("--require_final_protocol", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_steps < 1:
        parser.error("batch_size and num_steps must be positive")
    if args.num_samples is not None and args.num_samples < 1:
        parser.error("num_samples must be positive")
    if args.bootstrap_resamples < 1:
        parser.error("bootstrap_resamples must be positive")
    if args.seed < 0:
        parser.error("seed must be non-negative")
    if not 0.0 < args.confidence < 1.0:
        parser.error("confidence must be in (0,1)")
    return args


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(output_path), "count": result["selection"]["count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
