#!/usr/bin/env python3
"""Matched T2M and MotionFix guardrail comparison across two checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


T2M_METRICS: dict[str, tuple[tuple[str, ...], float, str, bool]] = {
    "r_precision_top1": (("r_top1",), 1.0, "fraction", False),
    "r_precision_top2": (("r_top2",), 1.0, "fraction", False),
    "r_precision_top3": (("r_top3",), 1.0, "fraction", False),
    "mm_distance": (("mm_dist",), 1.0, "embedding_distance", True),
    "fk_jerk": (("quality", "fk_jerk_mps3"), 1.0, "m/s^3", True),
    "foot_skate_ratio": (("quality", "foot_skate_ratio"), 1.0, "fraction", True),
}

EDIT_METRICS: dict[str, tuple[str, float, str, bool]] = {
    "target_joint_error": ("global_joint_target_error_m", 100.0, "cm", True),
    "target_rotation_error": ("global_rotation_target_error_deg", 1.0, "deg", True),
    "changed_region_joint_error": (
        "changed_region_target_error_m",
        100.0,
        "cm",
        True,
    ),
    "changed_region_rotation_error": (
        "changed_region_target_rotation_error_deg",
        1.0,
        "deg",
        True,
    ),
    "prediction_jerk": ("prediction_jerk_mps3", 1.0, "m/s^3", True),
    "foot_skate_ratio": ("foot_skate_ratio", 1.0, "fraction", True),
}

EDIT_COUNTERFACTUAL_SYSTEMS = {
    "text_presence": "source_only_model",
    "instruction_content": "source_shuffled_instruction_model",
    "source_content": "shuffled_source_instruction_model",
}

EDIT_IDENTITY_FIELDS = (
    "case_key",
    "case_uid",
    "pair_id",
    "system",
    "instruction",
    "model_instruction",
    "sample_seed",
    "control_subtype",
    "source_frames",
    "target_frames",
    "source_k273_path",
    "target_k273_path",
    "length_relation",
    "target_length_protocol",
    "frame_policy_id",
    "regional_metric_protocol",
    "source_applied_yaw_delta",
    "target_applied_yaw_delta",
    "model_source_applied_yaw_delta",
    "aligned_source_applied_yaw_delta",
    "output_gauge_phi",
    "condition_provenance",
    "sampling_protocol",
    "seen_strata",
    "assets",
)

EDIT_TARGET_IDENTITY_METRICS = (
    "frames",
    "source_target_position_delta_m",
    "source_target_rotation_delta_deg",
    "target_jerk_mps3",
    "changed_joint_entries",
    "ambiguous_joint_entries",
    "unchanged_joint_entries",
    "changed_position_threshold_m",
    "changed_rotation_threshold_deg",
    "changed_temporal_dilation_frames",
    "unchanged_position_threshold_m",
    "unchanged_rotation_threshold_deg",
)


def _load_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    with resolved.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.expanduser().resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _metric_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _paired_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    bootstrap_resamples: int,
    seed: int,
    label: str,
    unit: str,
    lower_is_better: bool,
) -> dict[str, Any]:
    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    if left.ndim != 1 or left.shape != right.shape or left.size == 0:
        raise ValueError(f"Invalid paired arrays for {label}: {left.shape}/{right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"Non-finite paired values for {label}")
    delta = right - left
    rng = np.random.default_rng(_metric_seed(seed, label))
    bootstrap = np.empty(int(bootstrap_resamples), dtype=np.float64)
    chunk_size = 256
    for start in range(0, bootstrap.size, chunk_size):
        stop = min(bootstrap.size, start + chunk_size)
        indices = rng.integers(0, delta.size, size=(stop - start, delta.size))
        bootstrap[start:stop] = delta[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "count": int(delta.size),
        "baseline": float(left.mean()),
        "candidate": float(right.mean()),
        "candidate_minus_baseline": float(delta.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "unit": unit,
        "lower_is_better": bool(lower_is_better),
    }


def _nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing metric field {'.'.join(path)}")
        value = value[key]
    return value


def _rows_by_key(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], Any],
    *,
    label: str,
) -> dict[Any, dict[str, Any]]:
    output: dict[Any, dict[str, Any]] = {}
    for row in rows:
        key = key_fn(row)
        if key in output:
            raise ValueError(f"Duplicate {label} row: {key!r}")
        output[key] = row
    return output


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-7, atol=1e-9):
        raise ValueError(f"Stale aggregate for {label}: {actual} != {expected}")


def compare_t2m(
    baseline_summary_path: Path,
    candidate_summary_path: Path,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_summary = _load_json(baseline_summary_path)
    candidate_summary = _load_json(candidate_summary_path)
    left_protocol = baseline_summary.get("protocol", {})
    right_protocol = candidate_summary.get("protocol", {})
    protocol_fields = (
        "protocol_version",
        "bridge_protocol",
        "reference_domain",
        "case_count",
        "case_plan_sha256",
        "checkpoint_kind",
        "official_benchmark_claim",
        "sampling",
        "weight_source",
    )
    for field in protocol_fields:
        if left_protocol.get(field) != right_protocol.get(field):
            raise ValueError(f"T2M protocol mismatch for {field}")

    left_path = Path(str(baseline_summary["case_metrics_path"]))
    right_path = Path(str(candidate_summary["case_metrics_path"]))
    left_rows = _rows_by_key(
        _load_jsonl(left_path), lambda row: str(row["case_key"]), label="T2M baseline"
    )
    right_rows = _rows_by_key(
        _load_jsonl(right_path), lambda row: str(row["case_key"]), label="T2M candidate"
    )
    if set(left_rows) != set(right_rows):
        raise ValueError("T2M case-key sets differ")
    if len(left_rows) != int(left_protocol["case_count"]):
        raise ValueError("T2M case rows do not match the frozen case count")
    identity_fields = (
        "case_key",
        "sample_seed",
        "length",
        "native_gt_mm_dist",
        "native_gt_r_top1",
        "native_gt_r_top2",
        "native_gt_r_top3",
        "oracle_gt_mm_dist",
        "oracle_gt_r_top1",
        "oracle_gt_r_top2",
        "oracle_gt_r_top3",
    )
    keys = sorted(left_rows)
    for key in keys:
        for field in identity_fields:
            if left_rows[key].get(field) != right_rows[key].get(field):
                raise ValueError(f"T2M identity mismatch for {key}, field {field}")

    metrics: dict[str, Any] = {}
    for name, (path, scale, unit, lower_is_better) in T2M_METRICS.items():
        left = np.asarray(
            [float(_nested(left_rows[key], path)) * scale for key in keys]
        )
        right = np.asarray(
            [float(_nested(right_rows[key], path)) * scale for key in keys]
        )
        metrics[name] = _paired_summary(
            left,
            right,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            label=f"t2m:{name}",
            unit=unit,
            lower_is_better=lower_is_better,
        )

    left_metrics = baseline_summary["metrics"]
    right_metrics = candidate_summary["metrics"]
    aggregate_paths = {
        "r_precision_top1": (("r_precision", "top1"), "r_precision_top1"),
        "r_precision_top2": (("r_precision", "top2"), "r_precision_top2"),
        "r_precision_top3": (("r_precision", "top3"), "r_precision_top3"),
        "mm_distance": (("mm_dist",), "mm_distance"),
        "fk_jerk": (("quality", "fk_jerk_mps3"), "fk_jerk"),
        "foot_skate_ratio": (
            ("quality", "foot_skate_ratio"),
            "foot_skate_ratio",
        ),
    }
    for name, (summary_path, metric_name) in aggregate_paths.items():
        _assert_close(
            float(_nested(left_metrics, summary_path)),
            float(metrics[metric_name]["baseline"]),
            label=f"T2M baseline {name}",
        )
        _assert_close(
            float(_nested(right_metrics, summary_path)),
            float(metrics[metric_name]["candidate"]),
            label=f"T2M candidate {name}",
        )

    return {
        "case_count": len(keys),
        "identity": {
            "matched": True,
            "fields": list(identity_fields),
            "case_plan_sha256": left_protocol["case_plan_sha256"],
            "baseline_num_shards": int(left_protocol["num_shards"]),
            "candidate_num_shards": int(right_protocol["num_shards"]),
            "sharding_is_not_the_bootstrap_unit": True,
        },
        "metrics": metrics,
        "aggregate_point_estimates": {
            "fid": {
                "baseline": float(left_metrics["fid"]),
                "candidate": float(right_metrics["fid"]),
                "candidate_minus_baseline": float(
                    right_metrics["fid"] - left_metrics["fid"]
                ),
                "paired_ci_available": False,
            }
        },
    }


def _edit_protocol_identity(summary: dict[str, Any]) -> dict[str, Any]:
    protocol = summary.get("protocol", {})
    counterfactual = protocol.get("counterfactual_manifest", {})
    counterfactual_summary = counterfactual.get("summary", {})
    train_manifest = protocol.get("train_manifest", {})
    fields = (
        "format",
        "protocol",
        "pair_count",
        "case_count",
        "systems",
        "weight_source",
        "num_steps",
        "source_cfg_scale",
        "edit_cfg_scale",
        "generate_text_cfg_scale",
        "seed",
        "target_length_protocol",
        "metric_protocol",
        "same_noise_across_systems",
        "plan_sha256",
        "source_copy_unequal_protocol",
        "cfg_apply_contacts",
        "contact_feedback",
        "contact_init",
        "generate_cfg_apply_contacts",
        "editing_contact_protocol_id",
        "generate_control_contact_protocol_id",
        "direct_k273_interpolation",
    )
    return {
        **{field: protocol.get(field) for field in fields},
        "counterfactual_manifest_sha256": counterfactual.get("sha256"),
        "counterfactual_rows_sha256": counterfactual_summary.get("rows_sha256"),
        "train_manifest_sha256": train_manifest.get("sha256"),
    }


def _load_edit_rows(summary_path: Path) -> tuple[dict[str, Any], dict[Any, Any]]:
    summary = _load_json(summary_path)
    protocol = summary.get("protocol", {})
    shard_root = summary_path.expanduser().resolve(strict=True).parent / "shards"
    num_shards = int(protocol.get("num_shards", 0))
    if num_shards <= 0:
        raise ValueError("Edit summary has an invalid shard count")
    rows = []
    for shard_id in range(num_shards):
        rows.extend(_load_jsonl(shard_root / f"shard_{shard_id:02d}.jsonl"))
    mapped = _rows_by_key(
        rows,
        lambda row: (str(row["pair_id"]), str(row["system"])),
        label="Edit",
    )
    if len(mapped) != int(protocol["case_count"]):
        raise ValueError("Edit shard rows do not match the frozen case count")
    return summary, mapped


def _paired_edit_metric(
    left_rows: dict[Any, dict[str, Any]],
    right_rows: dict[Any, dict[str, Any]],
    keys: list[Any],
    metric: str,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    left_values = []
    right_values = []
    missing = 0
    for key in keys:
        left = left_rows[key]["metrics"].get(metric)
        right = right_rows[key]["metrics"].get(metric)
        if (left is None) != (right is None):
            raise ValueError(f"Edit missing-value pattern differs for {key}, {metric}")
        if left is None:
            missing += 1
            continue
        left_values.append(float(left) * scale)
        right_values.append(float(right) * scale)
    return np.asarray(left_values), np.asarray(right_values), missing


def compare_edit(
    baseline_summary_path: Path,
    candidate_summary_path: Path,
    *,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_summary, left_rows = _load_edit_rows(baseline_summary_path)
    candidate_summary, right_rows = _load_edit_rows(candidate_summary_path)
    if _edit_protocol_identity(baseline_summary) != _edit_protocol_identity(
        candidate_summary
    ):
        raise ValueError("Edit scientific protocols differ")
    if set(left_rows) != set(right_rows):
        raise ValueError("Edit pair/system case sets differ")

    keys = sorted(left_rows)
    for key in keys:
        left = left_rows[key]
        right = right_rows[key]
        if left.get("status") != "ok" or right.get("status") != "ok":
            raise ValueError(f"Edit case is not successful: {key}")
        for field in EDIT_IDENTITY_FIELDS:
            if left.get(field) != right.get(field):
                raise ValueError(f"Edit identity mismatch for {key}, field {field}")
        left_reference = left.get("aligned_reference_source") or {}
        right_reference = right.get("aligned_reference_source") or {}
        if left_reference.get("tensor_sha256") != right_reference.get("tensor_sha256"):
            raise ValueError(f"Edit aligned source differs for {key}")
        for field in EDIT_TARGET_IDENTITY_METRICS:
            if left["metrics"].get(field) != right["metrics"].get(field):
                raise ValueError(f"Edit target identity mismatch for {key}, metric {field}")

    correct_keys = [
        key for key in keys if key[1] == "source_instruction_model"
    ]
    pair_count = int(baseline_summary["protocol"]["pair_count"])
    if len(correct_keys) != pair_count:
        raise ValueError("Edit correct-condition rows do not cover every pair")

    metrics: dict[str, Any] = {}
    for name, (field, scale, unit, lower_is_better) in EDIT_METRICS.items():
        left, right, missing = _paired_edit_metric(
            left_rows, right_rows, correct_keys, field, scale
        )
        metrics[name] = {
            **_paired_summary(
                left,
                right,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                label=f"edit:{name}",
                unit=unit,
                lower_is_better=lower_is_better,
            ),
            "missing_pairs": int(missing),
        }

    counterfactuals: dict[str, Any] = {}
    for label, system in EDIT_COUNTERFACTUAL_SYSTEMS.items():
        available_pairs = [
            pair_id
            for pair_id, _ in correct_keys
            if (pair_id, system) in left_rows and (pair_id, system) in right_rows
        ]
        if not available_pairs:
            raise ValueError(f"No Edit counterfactual rows for {system}")
        counterfactuals[label] = {}
        for metric_name, field, scale, unit in (
            ("target_joint_error_advantage", "global_joint_target_error_m", 100.0, "cm"),
            (
                "target_rotation_error_advantage",
                "global_rotation_target_error_deg",
                1.0,
                "deg",
            ),
        ):
            left_advantage = np.asarray(
                [
                    (
                        float(left_rows[(pair_id, system)]["metrics"][field])
                        - float(
                            left_rows[(pair_id, "source_instruction_model")]["metrics"][
                                field
                            ]
                        )
                    )
                    * scale
                    for pair_id in available_pairs
                ]
            )
            right_advantage = np.asarray(
                [
                    (
                        float(right_rows[(pair_id, system)]["metrics"][field])
                        - float(
                            right_rows[(pair_id, "source_instruction_model")][
                                "metrics"
                            ][field]
                        )
                    )
                    * scale
                    for pair_id in available_pairs
                ]
            )
            counterfactuals[label][metric_name] = _paired_summary(
                left_advantage,
                right_advantage,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
                label=f"edit:{label}:{metric_name}",
                unit=unit,
                lower_is_better=False,
            )

    left_gain = []
    right_gain = []
    for pair_id, _ in correct_keys:
        for rows, output in ((left_rows, left_gain), (right_rows, right_gain)):
            values = rows[(pair_id, "source_instruction_model")]["metrics"]
            source_delta = float(values["source_target_position_delta_m"])
            target_error = float(values["global_joint_target_error_m"])
            output.append((source_delta - target_error) / max(source_delta, 1.0e-4))
    edit_gain = _paired_summary(
        np.asarray(left_gain),
        np.asarray(right_gain),
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
        label="edit:normalized_edit_gain",
        unit="fraction",
        lower_is_better=False,
    )

    source_copy_keys = [key for key in keys if key[1] == "source_copy"]
    for key in source_copy_keys:
        if left_rows[key]["metrics"] != right_rows[key]["metrics"]:
            raise ValueError(f"Checkpoint-independent source-copy result differs: {key}")

    return {
        "pair_count": pair_count,
        "case_count": len(keys),
        "identity": {
            "matched": True,
            "fields": list(EDIT_IDENTITY_FIELDS),
            "target_identity_metrics": list(EDIT_TARGET_IDENTITY_METRICS),
            "baseline_num_shards": int(baseline_summary["protocol"]["num_shards"]),
            "candidate_num_shards": int(candidate_summary["protocol"]["num_shards"]),
            "source_copy_checkpoint_invariance_verified": True,
        },
        "metrics": metrics,
        "counterfactual_advantage": counterfactuals,
        "normalized_edit_gain": edit_gain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_t2m", type=Path, required=True)
    parser.add_argument("--candidate_t2m", type=Path, required=True)
    parser.add_argument("--baseline_edit", type=Path, required=True)
    parser.add_argument("--candidate_edit", type=Path, required=True)
    parser.add_argument("--baseline_label", required=True)
    parser.add_argument("--candidate_label", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0 or args.seed < 0:
        parser.error("bootstrap_resamples must be positive and seed non-negative")

    payload = {
        "format": "hy273_t2m_edit_matched_guardrails_v1",
        "baseline": {
            "label": args.baseline_label,
            "t2m_summary": str(args.baseline_t2m.expanduser().resolve(strict=True)),
            "edit_summary": str(args.baseline_edit.expanduser().resolve(strict=True)),
        },
        "candidate": {
            "label": args.candidate_label,
            "t2m_summary": str(args.candidate_t2m.expanduser().resolve(strict=True)),
            "edit_summary": str(args.candidate_edit.expanduser().resolve(strict=True)),
        },
        "bootstrap": {
            "unit": "case_or_pair",
            "resamples": int(args.bootstrap_resamples),
            "seed": int(args.seed),
            "interval": "paired_percentile_95",
        },
        "t2m": compare_t2m(
            args.baseline_t2m,
            args.candidate_t2m,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        ),
        "edit": compare_edit(
            args.baseline_edit,
            args.candidate_edit,
            bootstrap_resamples=args.bootstrap_resamples,
            seed=args.seed,
        ),
        "scope": (
            "Matched checkpoint guardrails; intervals resample evaluation cases/pairs "
            "and do not cover training-run or training-seed variance."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
