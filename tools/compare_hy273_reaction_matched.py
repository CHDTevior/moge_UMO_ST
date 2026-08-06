#!/usr/bin/env python3
"""Matched UID-cluster comparison for two HY273 Reaction evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_reaction_metrics import reaction_fixed_role_metrics
from models.raw_motion.hy273_slices import DIM_HY273


COUNT_FIELDS = ("tp", "fp", "fn", "target_positive", "target_negative")
THRESHOLDS_CM = (10, 20, 30)
CAUSAL_VARIANTS = ("source_only", "shuffled_text", "unrelated_source", "empty")
REQUIRED_VARIANTS = ("source_text", *CAUSAL_VARIANTS)
EXPECTED_FORMAT = "hy273_fixed_role_reaction_eval_v2"
EXPECTED_DATASET = "Inter-X K273"
EXPECTED_ASSIGNMENT = "fixed_source_actor_to_target_reactor_no_swap"
EXPECTED_CHECKPOINT_STEP = 150_000
EXPECTED_VAL_COUNT = 522
EXPECTED_SAMPLING = {
    "num_steps": 32,
    "source_cfg_scale": 2.0,
    "text_cfg_scale": 2.0,
    "seed": 20260801,
    "initial_noise_policy": "deterministic_per_uid_and_caption",
    "matched_initial_noise_across_variants": True,
}
MEAN_METRICS = (
    "reactor_fk_mpjpe_cm",
    "reactor_root_error_cm",
    "fk_relation_distance_mae_cm",
    "position_relation_distance_mae_cm",
    "partner_facing_error_deg",
    "relative_root_radius_error_cm",
    "relative_root_bearing_error_deg",
    "relative_heading_error_deg",
    "frame0_relative_root_error_cm",
    "initial_15f_relative_root_error_cm",
    "frame0_relative_heading_error_deg",
    "initial_15f_relative_heading_error_deg",
    "first_close_timing_error_s_20cm",
    "first_close_too_early_s_20cm",
    "first_close_too_late_s_20cm",
)
LAYOUT_PHASE_MEAN_METRICS = (
    "frame0_relative_root_error_cm",
    "initial_15f_relative_root_error_cm",
    "frame0_relative_heading_error_deg",
    "initial_15f_relative_heading_error_deg",
    "first_close_timing_error_s_20cm",
    "first_close_too_early_s_20cm",
    "first_close_too_late_s_20cm",
)
LAYOUT_PHASE_POOLED_METRICS = {
    "precontact_relative_root_error_cm": (
        "precontact_relative_root_error_sum_cm",
        "precontact_valid_frames_20cm",
    ),
    "precontact_relative_heading_error_deg": (
        "precontact_relative_heading_error_sum_deg",
        "precontact_valid_frames_20cm",
    ),
    "precontact_false_close_rate_20cm": (
        "precontact_false_close_frames_20cm",
        "precontact_valid_frames_20cm",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("per_sample"), dict):
        raise ValueError(f"Evaluation has no per_sample variants: {path}")
    return payload


def _rows_by_uid(payload: dict[str, Any], variant: str) -> dict[str, list[dict[str, Any]]]:
    rows = payload["per_sample"].get(variant)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing non-empty per_sample variant {variant!r}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("variant") != variant:
            raise ValueError(
                f"Row variant mismatch: requested {variant!r}, got {row.get('variant')!r}"
            )
        uid = str(row["uid"])
        grouped.setdefault(uid, []).append(row)
    for uid_rows in grouped.values():
        uid_rows.sort(
            key=lambda row: (
                int(row["caption_index"]),
                int(row["dataset_index"]),
            )
        )
    return grouped


def _load_prediction_motion(path: Path, length: int) -> torch.Tensor:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != DIM_HY273 or array.shape[0] < length:
        raise ValueError(
            f"Prediction tensor {path} cannot provide [{length},{DIM_HY273}]: "
            f"{array.shape}"
        )
    selected = np.asarray(array[:length], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError(f"Prediction tensor contains non-finite values: {path}")
    return torch.from_numpy(selected.copy())


def _recompute_metrics_from_predictions(
    payload: dict[str, Any],
    prediction_dir: Path,
) -> dict[str, Any]:
    """Replace row metrics using saved samples and the current metric implementation."""

    prediction_dir = prediction_dir.expanduser().resolve(strict=True)
    grouped = {
        variant: _rows_by_uid(payload, variant) for variant in REQUIRED_VARIANTS
    }
    uids = sorted(grouped["source_text"])
    for variant in REQUIRED_VARIANTS:
        if set(grouped[variant]) != set(uids):
            raise ValueError(f"Saved-report UID set differs for variant {variant!r}")

    for uid in uids:
        rows = [grouped[variant][uid] for variant in REQUIRED_VARIANTS]
        if any(len(variant_rows) != 1 for variant_rows in rows):
            raise ValueError(
                "Prediction metric recomputation expects one uid-balanced caption per UID"
            )
        report_rows = [variant_rows[0] for variant_rows in rows]
        length = int(report_rows[0]["length"])
        case_dir = prediction_dir / uid
        metadata_path = case_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            str(metadata.get("uid")) != uid
            or int(metadata.get("length", -1)) != length
            or int(metadata.get("caption_index", -1))
            != int(report_rows[0]["caption_index"])
        ):
            raise ValueError(f"Prediction metadata does not match report row for {uid}")

        source = _load_prediction_motion(case_dir / "source.npy", length)
        target = _load_prediction_motion(case_dir / "target.npy", length)
        predictions = torch.stack(
            [
                _load_prediction_motion(case_dir / f"{variant}.npy", length)
                for variant in REQUIRED_VARIANTS
            ],
            dim=0,
        )
        count = len(REQUIRED_VARIANTS)
        metrics = reaction_fixed_role_metrics(
            source.unsqueeze(0).expand(count, -1, -1),
            predictions,
            target.unsqueeze(0).expand(count, -1, -1),
            lengths=torch.full((count,), length, dtype=torch.long),
            fps=30.0,
        )["per_sample"]
        for report_row, metric_row in zip(report_rows, metrics):
            report_row.update(
                {
                    key: value
                    for key, value in metric_row.items()
                    if key not in {"index", "length", "assignment"}
                }
            )

    payload["matched_metric_recomputation"] = {
        "prediction_dir": str(prediction_dir),
        "metric_function": "reaction_fixed_role_metrics",
        "fps": 30.0,
        "uids": len(uids),
        "variants": list(REQUIRED_VARIANTS),
    }
    return payload


def _validate_single_protocol(payload: dict[str, Any], label: str) -> None:
    expected_scalars = {
        "format": EXPECTED_FORMAT,
        "dataset": EXPECTED_DATASET,
        "split": "val",
        "caption_policy": "uid_balanced",
        "weight_source": "ema",
        "assignment_rule": EXPECTED_ASSIGNMENT,
        "checkpoint_next_global_step": EXPECTED_CHECKPOINT_STEP,
    }
    for field, expected in expected_scalars.items():
        actual = payload.get(field)
        if actual != expected:
            raise ValueError(
                f"{label} violates fixed protocol for {field}: {actual!r} != {expected!r}"
            )

    sampling = payload.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError(f"{label} has no sampling metadata")
    for field, expected in EXPECTED_SAMPLING.items():
        actual = sampling.get(field)
        if actual != expected:
            raise ValueError(
                f"{label} violates fixed sampling for {field}: {actual!r} != {expected!r}"
            )

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{label} has no selection metadata")
    expected_selection = {
        "count": EXPECTED_VAL_COUNT,
        "dataset_count_after_filters": EXPECTED_VAL_COUNT,
        "start_index": 0,
    }
    for field, expected in expected_selection.items():
        actual = selection.get(field)
        if actual != expected:
            raise ValueError(
                f"{label} is not the complete fixed Val selection for {field}: "
                f"{actual!r} != {expected!r}"
            )

    variants = payload.get("per_sample")
    if not isinstance(variants, dict):
        raise ValueError(f"{label} has no per_sample mapping")
    for variant in REQUIRED_VARIANTS:
        rows = variants.get(variant)
        if not isinstance(rows, list) or len(rows) != EXPECTED_VAL_COUNT:
            raise ValueError(
                f"{label} variant {variant!r} is incomplete: "
                f"{len(rows) if isinstance(rows, list) else None} != {EXPECTED_VAL_COUNT}"
            )

    causal = payload.get("protocols", {}).get("causal_ablations")
    if causal != ["empty", "shuffled_text", "unrelated_source"]:
        raise ValueError(f"{label} has unexpected causal ablations: {causal!r}")
    donor = payload.get("negative_donor_protocol")
    if not isinstance(donor, dict) or donor.get("scope") != "complete_filtered_split":
        raise ValueError(f"{label} lacks the complete negative-donor protocol")


def _validate_protocol(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    _validate_single_protocol(baseline, "baseline")
    _validate_single_protocol(candidate, "candidate")
    scalar_fields = ("split", "caption_policy", "weight_source", "assignment_rule")
    for field in scalar_fields:
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Protocol mismatch for {field}: {baseline.get(field)!r} != {candidate.get(field)!r}")
    sampling_fields = (
        "num_steps",
        "source_cfg_scale",
        "text_cfg_scale",
        "seed",
    )
    for field in sampling_fields:
        left = baseline.get("sampling", {}).get(field)
        right = candidate.get("sampling", {}).get(field)
        if left != right:
            raise ValueError(f"Sampling mismatch for {field}: {left!r} != {right!r}")
    if baseline.get("negative_donor_protocol") != candidate.get("negative_donor_protocol"):
        raise ValueError("Negative-donor protocols differ between evaluation arms")


def _config_differences(
    left: Any,
    right: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[tuple[tuple[str, ...], Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append((path + (str(key),), left.get(key), right.get(key)))
            else:
                differences.extend(
                    _config_differences(left[key], right[key], path + (str(key),))
                )
        return differences
    if left != right:
        return [(path, left, right)]
    return []


def _load_checkpoint_contract(payload: dict[str, Any], label: str) -> dict[str, Any]:
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError as error:
        raise RuntimeError("PyTorch is required to validate checkpoint contracts") from error

    checkpoint_value = payload.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError(f"{label} has no checkpoint path")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve(strict=True)
    if checkpoint_path.name != "step_00150000.pt":
        raise ValueError(f"{label} is not the fixed 150K checkpoint: {checkpoint_path}")
    with FakeTensorMode():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    if checkpoint.get("format") != "hy273_unified_actor_checkpoint_v1":
        raise ValueError(f"{label} has unexpected checkpoint format")
    if checkpoint.get("next_global_step") != EXPECTED_CHECKPOINT_STEP:
        raise ValueError(f"{label} checkpoint is not at 150K")
    if payload.get("checkpoint_next_global_step") != checkpoint.get("next_global_step"):
        raise ValueError(f"{label} evaluation/checkpoint step metadata disagree")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{label} checkpoint has no embedded config")
    batcher = checkpoint.get("batcher")
    if not isinstance(batcher, dict):
        raise ValueError(f"{label} checkpoint has no batcher state")
    return {
        "path": str(checkpoint_path),
        "run_name": checkpoint.get("run_name"),
        "config_path": checkpoint.get("config_path"),
        "config": config,
        "rng_contract": checkpoint.get("rng_contract"),
        "batcher": batcher,
    }


def _validate_training_contract(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    left = _load_checkpoint_contract(baseline, "baseline")
    right = _load_checkpoint_contract(candidate, "candidate")
    differences = _config_differences(left["config"], right["config"])
    if mode == "p_only_ablation":
        expected_differences = [
            (("reaction_loss", "close_joint_vector"), 0.0, 0.01),
        ]
    elif mode == "reaction_v3_adaptive":
        baseline_close_vector = left["config"].get("reaction_loss", {}).get(
            "close_joint_vector"
        )
        if baseline_close_vector not in (0.0, 0.01):
            raise ValueError(
                "Reaction-v3 comparison expects a v2 or v2-P-only baseline, "
                f"got close_joint_vector={baseline_close_vector!r}"
            )
        expected_differences = [
            (("reaction_loss", "adaptive_distance_beta_m"), None, 0.05),
            (("reaction_loss", "adaptive_distance_eps_m"), None, 0.1),
            (
                ("reaction_loss", "close_joint_vector"),
                baseline_close_vector,
                0.00191,
            ),
            (("reaction_loss", "fine_min_flow_t"), 0.55, 0.2),
            (("reaction_loss", "joint_distance"), 0.01, 0.0273),
            (
                ("reaction_loss", "joint_distance_mode"),
                None,
                "adaptive_gt_inverse",
            ),
        ]
    elif mode == "reaction_v4_layout":
        expected_differences = [
            (("reaction_loss", "heading_beta"), None, 0.1),
            (("reaction_loss", "layout_contact_threshold_m"), None, 0.2),
            (("reaction_loss", "layout_initial_frames"), None, 15),
            (("reaction_loss", "layout_initial_multiplier"), None, 3.0),
            (("reaction_loss", "layout_precontact_multiplier"), None, 2.0),
            (("reaction_loss", "relative_heading"), 0.0, 0.0217),
            (("reaction_loss", "relative_root"), 0.0, 0.0195),
        ]
    else:
        raise ValueError(f"Unsupported training-contract mode: {mode!r}")

    if differences != expected_differences:
        formatted = [
            {"path": ".".join(path), "baseline": old, "candidate": new}
            for path, old, new in differences
        ]
        raise ValueError(
            f"Checkpoint configs do not implement {mode!r}: "
            f"{formatted}"
        )
    if left["rng_contract"] != right["rng_contract"]:
        raise ValueError("Checkpoint RNG contracts differ")
    if left["batcher"] != right["batcher"]:
        raise ValueError("Final deterministic batcher states differ between arms")
    return {
        "baseline_checkpoint": left["path"],
        "candidate_checkpoint": right["path"],
        "baseline_run_name": left["run_name"],
        "candidate_run_name": right["run_name"],
        "baseline_config_path": left["config_path"],
        "candidate_config_path": right["config_path"],
        "mode": mode,
        "config_differences": [
            {
                "path": ".".join(path),
                "baseline": old,
                "candidate": new,
            }
            for path, old, new in differences
        ],
        "rng_contract": left["rng_contract"],
        "deterministic_batcher_state_matched": True,
        "parent_lineage_note": (
            "Both legacy runs were launched from the protocol-locked 100K parent; "
            "the child checkpoint format does not embed its resume path."
        ),
    }


def _validate_matched_rows(
    baseline: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
) -> list[str]:
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "UID sets differ: "
            f"missing_candidate={missing_candidate[:5]}, missing_baseline={missing_baseline[:5]}"
        )
    matched_fields = (
        "uid",
        "dataset_index",
        "caption_index",
        "length",
        "length_bucket",
        "action_category",
        "actor_person_index",
        "assignment",
        "negative_donor_uid",
        "negative_donor_action_category",
    )
    for uid in sorted(baseline):
        left_rows = baseline[uid]
        right_rows = candidate[uid]
        if len(left_rows) != len(right_rows):
            raise ValueError(f"Row count differs for UID {uid}")
        for left, right in zip(left_rows, right_rows):
            for field in matched_fields:
                if left.get(field) != right.get(field):
                    raise ValueError(
                        f"Matched field {field!r} differs for UID {uid}: "
                        f"{left.get(field)!r} != {right.get(field)!r}"
                    )
    return sorted(baseline)


def _cluster_counts(
    grouped: dict[str, list[dict[str, Any]]],
    uids: Iterable[str],
    prefix: str,
    threshold_cm: int,
) -> np.ndarray:
    rows = []
    stem = f"{prefix}close_{threshold_cm}cm_"
    for uid in uids:
        values = np.zeros(len(COUNT_FIELDS), dtype=np.float64)
        for row in grouped[uid]:
            local = []
            for field in COUNT_FIELDS:
                key = stem + field
                try:
                    value = float(row[key])
                except KeyError as error:
                    raise ValueError(f"Missing count field {key!r} for UID {uid}") from error
                if not math.isfinite(value) or value < 0 or value != round(value):
                    raise ValueError(f"Invalid integer count {key}={value!r} for UID {uid}")
                local.append(value)
            tp, fp, fn, target_positive, target_negative = local
            if tp + fn != target_positive:
                raise ValueError(f"TP+FN does not equal target_positive for UID {uid}")
            if fp > target_negative:
                raise ValueError(f"FP exceeds target_negative for UID {uid}")
            if target_positive + target_negative != int(row["length"]):
                raise ValueError(f"GT close-event denominator does not equal length for UID {uid}")
            values += np.asarray(local, dtype=np.float64)
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def _count_metrics(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn, target_positive, target_negative = np.moveaxis(counts, -1, 0)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=denominator > 0,
        )

    return np.stack(
        (
            ratio(tp, tp + fp),
            ratio(tp, tp + fn),
            ratio(2.0 * tp, 2.0 * tp + fp + fn),
            ratio(fp, target_negative),
            ratio(fn, target_positive),
        ),
        axis=-1,
    )


def _summary(values: np.ndarray, point: np.ndarray) -> dict[str, dict[str, float]]:
    names = ("precision", "recall", "f1", "false_close", "missed_close")
    output: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        finite = values[np.isfinite(values[:, index]), index]
        if not math.isfinite(float(point[index])):
            raise ValueError(f"Pooled {name} is undefined")
        if len(finite) < max(100, int(0.95 * len(values))):
            raise ValueError(
                f"Too few defined bootstrap replicates for {name}: {len(finite)}/{len(values)}"
            )
        output[name] = {
            "delta": float(point[index]),
            "ci_low": float(np.quantile(finite, 0.025)),
            "ci_high": float(np.quantile(finite, 0.975)),
            "valid_resamples": int(len(finite)),
        }
    return output


def _paired_count_comparison(
    baseline_counts: np.ndarray,
    candidate_counts: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
    chunk_size: int = 512,
) -> dict[str, Any]:
    if baseline_counts.shape != candidate_counts.shape:
        raise ValueError("Count arrays are not matched")
    if baseline_counts.ndim != 2 or baseline_counts.shape[1] != len(COUNT_FIELDS):
        raise ValueError("Count arrays have an invalid shape")
    if not np.array_equal(baseline_counts[:, 3:], candidate_counts[:, 3:]):
        raise ValueError("Matched arms have different target-positive/negative denominators")
    point_baseline = _count_metrics(baseline_counts.sum(axis=0))
    point_candidate = _count_metrics(candidate_counts.sum(axis=0))
    bootstrap = np.empty((resamples, 5), dtype=np.float64)
    cluster_count = baseline_counts.shape[0]
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        left = baseline_counts[indices].sum(axis=1)
        right = candidate_counts[indices].sum(axis=1)
        bootstrap[start:stop] = _count_metrics(right) - _count_metrics(left)
    return {
        "aggregate_counts": {
            "baseline": dict(zip(COUNT_FIELDS, map(int, baseline_counts.sum(axis=0)))),
            "candidate": dict(zip(COUNT_FIELDS, map(int, candidate_counts.sum(axis=0)))),
        },
        "baseline": dict(
            zip(
                ("precision", "recall", "f1", "false_close", "missed_close"),
                map(float, point_baseline),
            )
        ),
        "candidate": dict(
            zip(
                ("precision", "recall", "f1", "false_close", "missed_close"),
                map(float, point_candidate),
            )
        ),
        "candidate_minus_baseline": _summary(
            bootstrap, point_candidate - point_baseline
        ),
    }


def _cluster_means(
    grouped: dict[str, list[dict[str, Any]]],
    uids: Iterable[str],
    metric: str,
) -> np.ndarray:
    values = []
    for uid in uids:
        local = np.asarray([float(row[metric]) for row in grouped[uid]], dtype=np.float64)
        if not np.isfinite(local).all():
            raise ValueError(f"Non-finite {metric} for UID {uid}")
        values.append(float(local.mean()))
    return np.asarray(values, dtype=np.float64)


def _paired_mean_comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, float]:
    delta = candidate - baseline
    indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "baseline": float(baseline.mean()),
        "candidate": float(candidate.mean()),
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def _cluster_ratio_components(
    grouped: dict[str, list[dict[str, Any]]],
    uids: Iterable[str],
    numerator_field: str,
    denominator_field: str,
) -> np.ndarray:
    output = []
    for uid in uids:
        numerator = 0.0
        denominator = 0.0
        for row in grouped[uid]:
            local_numerator = float(row[numerator_field])
            local_denominator = float(row[denominator_field])
            if (
                not math.isfinite(local_numerator)
                or not math.isfinite(local_denominator)
                or local_numerator < 0.0
                or local_denominator < 0.0
            ):
                raise ValueError(
                    f"Invalid pooled metric components for UID {uid}: "
                    f"{numerator_field}={local_numerator}, "
                    f"{denominator_field}={local_denominator}"
                )
            numerator += local_numerator
            denominator += local_denominator
        output.append((numerator, denominator))
    return np.asarray(output, dtype=np.float64)


def _paired_ratio_comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] != 2:
        raise ValueError("Pooled ratio components are not matched [N,2] arrays")
    if not np.array_equal(baseline[:, 1], candidate[:, 1]):
        raise ValueError("Matched pooled metrics have different GT-derived denominators")

    def ratio(values: np.ndarray) -> np.ndarray:
        return np.divide(
            values[..., 0],
            values[..., 1],
            out=np.full(values.shape[:-1], np.nan, dtype=np.float64),
            where=values[..., 1] > 0,
        )

    point_baseline = float(ratio(baseline.sum(axis=0)))
    point_candidate = float(ratio(candidate.sum(axis=0)))
    if not math.isfinite(point_baseline) or not math.isfinite(point_candidate):
        raise ValueError("Pooled ratio is undefined for the complete matched selection")
    cluster_count = baseline.shape[0]
    indices = rng.integers(0, cluster_count, size=(resamples, cluster_count))
    bootstrap = ratio(candidate[indices].sum(axis=1)) - ratio(
        baseline[indices].sum(axis=1)
    )
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if len(bootstrap) < max(100, int(0.95 * resamples)):
        raise ValueError(
            f"Too few defined pooled-ratio bootstrap replicates: "
            f"{len(bootstrap)}/{resamples}"
        )
    return {
        "baseline": point_baseline,
        "candidate": point_candidate,
        "candidate_minus_baseline": point_candidate - point_baseline,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "lower_is_better": True,
        "valid_resamples": int(len(bootstrap)),
        "aggregate_components": {
            "baseline_numerator": float(baseline[:, 0].sum()),
            "candidate_numerator": float(candidate[:, 0].sum()),
            "shared_denominator": float(baseline[:, 1].sum()),
        },
    }


def _causal_advantages(
    candidate: dict[str, Any],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    correct = _rows_by_uid(candidate, "source_text")
    output: dict[str, Any] = {}
    for variant in CAUSAL_VARIANTS:
        comparator = _rows_by_uid(candidate, variant)
        uids = _validate_matched_rows(correct, comparator)
        mean_metrics = {}
        for metric in ("reactor_fk_mpjpe_cm", "fk_relation_distance_mae_cm"):
            correct_values = _cluster_means(correct, uids, metric)
            comparator_values = _cluster_means(comparator, uids, metric)
            # Positive means the correct source+text branch has lower error.
            comparison = _paired_mean_comparison(
                correct_values,
                comparator_values,
                rng=rng,
                resamples=resamples,
            )
            mean_metrics[metric] = {
                "advantage": comparison["delta"],
                "ci_low": comparison["ci_low"],
                "ci_high": comparison["ci_high"],
            }
        correct_counts = _cluster_counts(correct, uids, "fk_", 20)
        comparator_counts = _cluster_counts(comparator, uids, "fk_", 20)
        count_comparison = _paired_count_comparison(
            comparator_counts,
            correct_counts,
            rng=rng,
            resamples=resamples,
        )
        output[variant] = {
            **mean_metrics,
            "fk_close_20cm_f1": count_comparison["candidate_minus_baseline"]["f1"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--baseline_predictions", type=Path)
    parser.add_argument("--candidate_predictions", type=Path)
    parser.add_argument("--variant", choices=("source_text",), default="source_text")
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--training_contract",
        choices=("p_only_ablation", "reaction_v3_adaptive", "reaction_v4_layout"),
        default="p_only_ablation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1000:
        parser.error("bootstrap_resamples must be at least 1000 for a 95% interval")

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    _validate_protocol(baseline, candidate)
    prediction_args = (args.baseline_predictions, args.candidate_predictions)
    if (prediction_args[0] is None) != (prediction_args[1] is None):
        parser.error(
            "--baseline_predictions and --candidate_predictions must be provided together"
        )
    if args.training_contract == "reaction_v4_layout" and prediction_args[0] is None:
        parser.error(
            "reaction_v4_layout comparison requires both prediction directories so "
            "new phase metrics are recomputed under one implementation"
        )
    if prediction_args[0] is not None and prediction_args[1] is not None:
        baseline = _recompute_metrics_from_predictions(baseline, prediction_args[0])
        candidate = _recompute_metrics_from_predictions(candidate, prediction_args[1])
    training_contract = _validate_training_contract(
        baseline,
        candidate,
        mode=args.training_contract,
    )
    baseline_rows = _rows_by_uid(baseline, args.variant)
    candidate_rows = _rows_by_uid(candidate, args.variant)
    uids = _validate_matched_rows(baseline_rows, candidate_rows)
    rng = np.random.default_rng(args.seed)

    close_metrics: dict[str, Any] = {}
    for name, prefix in (("position", ""), ("fk", "fk_")):
        close_metrics[name] = {}
        for threshold_cm in THRESHOLDS_CM:
            left = _cluster_counts(baseline_rows, uids, prefix, threshold_cm)
            right = _cluster_counts(candidate_rows, uids, prefix, threshold_cm)
            close_metrics[name][f"{threshold_cm}cm"] = _paired_count_comparison(
                left,
                right,
                rng=rng,
                resamples=args.bootstrap_resamples,
            )

    mean_metrics = {}
    for metric in MEAN_METRICS:
        left = _cluster_means(baseline_rows, uids, metric)
        right = _cluster_means(candidate_rows, uids, metric)
        mean_metrics[metric] = _paired_mean_comparison(
            left,
            right,
            rng=rng,
            resamples=args.bootstrap_resamples,
        )

    layout_phase_pooled_metrics = {}
    for metric, (numerator_field, denominator_field) in (
        LAYOUT_PHASE_POOLED_METRICS.items()
    ):
        left = _cluster_ratio_components(
            baseline_rows,
            uids,
            numerator_field,
            denominator_field,
        )
        right = _cluster_ratio_components(
            candidate_rows,
            uids,
            numerator_field,
            denominator_field,
        )
        layout_phase_pooled_metrics[metric] = _paired_ratio_comparison(
            left,
            right,
            rng=rng,
            resamples=args.bootstrap_resamples,
        )

    result = {
        "format": "hy273_reaction_matched_uid_cluster_comparison_v2",
        "baseline": {"label": args.baseline_label, "path": str(args.baseline.resolve())},
        "candidate": {"label": args.candidate_label, "path": str(args.candidate.resolve())},
        "training_contract": training_contract,
        "variant": args.variant,
        "split": baseline["split"],
        "uid_clusters": len(uids),
        "rows": sum(len(baseline_rows[uid]) for uid in uids),
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "unit": "uid_cluster",
            "interval": "paired_percentile_95",
            "numpy_version": np.__version__,
        },
        "close_metrics": close_metrics,
        "mean_metrics": mean_metrics,
        "layout_phase_metrics": {
            "per_clip_means": {
                metric: mean_metrics[metric] for metric in LAYOUT_PHASE_MEAN_METRICS
            },
            "pooled_over_valid_precontact_frames": layout_phase_pooled_metrics,
        },
        "metric_recomputation": {
            "baseline": baseline.get("matched_metric_recomputation"),
            "candidate": candidate.get("matched_metric_recomputation"),
        },
        "candidate_causal_advantage": _causal_advantages(
            candidate,
            rng=rng,
            resamples=args.bootstrap_resamples,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "uids": len(uids)}))


if __name__ == "__main__":
    main()
