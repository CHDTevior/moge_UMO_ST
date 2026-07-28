#!/usr/bin/env python
"""R12 250K decision gate for position-only root control and B2 admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.compare_hy273_nonregression import (
    CASE_IDENTITY_FIELDS,
    COUNT_METRICS,
    METRIC_SPECS,
    PRODUCTION_METRICS_BY_SUBTYPE,
    allowed_degradation,
    paired_bootstrap_mean_ci,
    signed_degradation,
)
from train_hy273_multitask import (
    CHECKPOINT_FORMAT,
    R12_ORIGIN_PARENT_SHA256,
    R12_TRAIN_CONTRACT,
    sha256_file,
    validate_r12_origin_parent_identity,
)


FORMAT = "hy273_r12_step250k_gate_v1"
BASELINE_200K_SHA256 = R12_ORIGIN_PARENT_SHA256
R11_250K_SHA256 = "400ae76a855e65b2df8dbb07f67883c87b5f368c268f5d984d9c18123f8da5ef"
XZ_ONLY_SUBTYPES = ("path_2dpos", "waypoint_2dpos")
TEXT_REGIMES = ("notext", "withtext")
ROOT_METRICS = (
    ("constraint_root2d_err", "lower"),
    ("constraint_root2d_acc", "higher"),
)
SKATE_METRICS = (
    "foot_skate_ratio",
    "foot_skate_from_height",
    "foot_skate_from_pred_contacts",
    "foot_skate_max_vel",
)
FIXED16_FLOORS = {
    "fk_jerk_mps3": ("lower", 0.10, 5.0),
    "position_channel_jerk_mps3": ("lower", 0.10, 5.0),
    "foot_skate_from_height": ("lower", 0.10, 0.02),
    "foot_skate_from_pred_contacts": ("lower", 0.10, 0.02),
    "foot_skate_max_vel": ("lower", 0.10, 0.05),
    "foot_skate_ratio": ("lower", 0.10, 0.03),
    "foot_contact_consistency": ("higher", 0.0, 0.03),
}
SCIENTIFIC_PROTOCOL_FIELDS = (
    "protocol_version",
    "profile",
    "dataset_split",
    "num_shards",
    "ode_steps",
    "text_cfg_scale",
    "control_cfg_scale",
    "seed",
    "max_sparse_keyframes",
    "weight_source",
    "primary_output",
    "case_plan_sha256",
)
FIXED16_PROTOCOL_FIELDS = (
    "fixed_visual_protocol",
    "sampling_protocol_version",
    "contact_protocol_version",
    "weight_source",
    "seed",
    "ode_steps",
    "text_cfg_scale",
    "contact_init",
    "contact_feedback",
    "cfg_apply_contacts",
    "primary_output",
    "route",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(key) for key in CASE_IDENTITY_FIELDS)


def load_eval(eval_dir: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = Path(eval_dir).expanduser().resolve()
    index = _read_json(root / "artifact_index.json")
    _require(index.get("status") == "validated", f"Unvalidated control evidence: {root}")
    records: dict[str, dict[str, Any]] = {}
    for shard in sorted((root / "shards").glob("shard_*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            _require(record.get("status") == "ok", f"Bad case status in {shard}")
            key = str(record.get("case_key", ""))
            _require(key and key not in records, f"Duplicate/empty case key: {key!r}")
            generated = record.get("metrics", {}).get("generated_raw")
            _require(isinstance(generated, dict), f"Missing generated_raw metrics: {key}")
            for metric, value in generated.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    _require(math.isfinite(float(value)), f"Non-finite {key}/{metric}")
            records[key] = record
    _require(len(records) == int(index.get("case_count", -1)), f"Case count mismatch: {root}")
    return index, records


def require_matched_cases(*record_sets: dict[str, dict[str, Any]]) -> list[str]:
    keys = sorted(record_sets[0])
    for records in record_sets[1:]:
        _require(sorted(records) == keys, "Control evidence case-key sets differ")
    for key in keys:
        expected = _record_identity(record_sets[0][key])
        for records in record_sets[1:]:
            _require(_record_identity(records[key]) == expected, f"Case identity mismatch: {key}")
    return keys


def scientific_protocol(eval_dir: str | Path) -> dict[str, Any]:
    protocol = _read_json(Path(eval_dir).expanduser().resolve() / "protocol_manifest.json")
    return {name: protocol.get(name) for name in SCIENTIFIC_PROTOCOL_FIELDS}


def require_matched_scientific_protocol(*eval_dirs: str | Path) -> dict[str, Any]:
    protocols = [scientific_protocol(path) for path in eval_dirs]
    expected = protocols[0]
    for path, protocol in zip(eval_dirs[1:], protocols[1:]):
        _require(protocol == expected, f"Sampling/control protocol differs: {path}")
    return expected


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return (int(seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def one_sided_paired_upper(
    differences: np.ndarray,
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    _require(values.ndim == 1 and values.size > 1, "Paired superiority needs at least two cases")
    _require(np.isfinite(values).all(), "Paired superiority received non-finite values")
    rng = np.random.default_rng(int(seed))
    means: list[np.ndarray] = []
    remaining = int(resamples)
    while remaining:
        count = min(remaining, 512)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    bootstrap = np.concatenate(means)
    return float(values.mean()), float(np.quantile(bootstrap, float(confidence)))


def root_superiority_rows(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for subtype in XZ_ONLY_SUBTYPES:
        for regime in TEXT_REGIMES:
            keys = [
                key
                for key, record in baseline.items()
                if record["subtype"] == subtype and record["text_regime"] == regime
            ]
            _require(keys, f"No cases for {subtype}/{regime}")
            for metric, direction in ROOT_METRICS:
                baseline_values = np.asarray(
                    [baseline[key]["metrics"]["generated_raw"][metric] for key in keys],
                    dtype=np.float64,
                )
                candidate_values = np.asarray(
                    [candidate[key]["metrics"]["generated_raw"][metric] for key in keys],
                    dtype=np.float64,
                )
                directional = (
                    candidate_values - baseline_values
                    if direction == "lower"
                    else baseline_values - candidate_values
                )
                mean, upper = one_sided_paired_upper(
                    directional,
                    resamples=resamples,
                    confidence=confidence,
                    seed=_stable_seed(seed, subtype, regime, metric),
                )
                rows.append(
                    {
                        "subtype": subtype,
                        "text_regime": regime,
                        "metric": metric,
                        "direction": direction,
                        "cases": len(keys),
                        "baseline_mean": float(baseline_values.mean()),
                        "candidate_mean": float(candidate_values.mean()),
                        "directional_paired_mean": mean,
                        "one_sided_upper": upper,
                        "confidence": float(confidence),
                        "passed": bool(mean < 0.0 and upper < 0.0),
                    }
                )
    return rows


def skate_improvement_rows(
    r11: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    keys = [key for key, record in r11.items() if record["subtype"] in XZ_ONLY_SUBTYPES]
    rows = []
    for metric in SKATE_METRICS:
        old = np.asarray(
            [r11[key]["metrics"]["generated_raw"][metric] for key in keys], dtype=np.float64
        )
        new = np.asarray(
            [candidate[key]["metrics"]["generated_raw"][metric] for key in keys],
            dtype=np.float64,
        )
        rows.append(
            {
                "metric": metric,
                "cases": len(keys),
                "r11_250k_mean": float(old.mean()),
                "r12_250k_mean": float(new.mean()),
                "difference": float(new.mean() - old.mean()),
                "passed": bool(new.mean() < old.mean()),
            }
        )
    return rows


def _heading_values(motion: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = motion[1:, (0, 2)] - motion[:-1, (0, 2)]
    norm = np.linalg.norm(delta, axis=-1)
    tangent = delta[support] / np.maximum(norm[support, None], 1e-12)
    heading = motion[:-1, 3:5][support]
    heading_norm = np.linalg.norm(heading, axis=-1)
    forward = np.stack([heading[:, 1], heading[:, 0]], axis=-1)
    forward = forward / np.maximum(heading_norm[:, None], 1e-12)
    cosine = np.clip(np.sum(forward * tangent, axis=-1), -1.0, 1.0)
    return cosine, np.rad2deg(np.arccos(cosine))


def paired_heading_path_diagnostic(
    record_sets: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    names = tuple(record_sets)
    _require(len(names) >= 2, "Paired heading diagnostic needs at least two models")
    keys = require_matched_cases(*(record_sets[name] for name in names))
    sums = {name: {"cosine": 0.0, "angle": 0.0, "moving": 0} for name in names}
    per_case = {name: {} for name in names}
    common_frames = 0
    common_cases = 0
    union_frames = 0

    for key in keys:
        if record_sets[names[0]][key]["subtype"] not in XZ_ONLY_SUBTYPES:
            continue
        motions: dict[str, np.ndarray] = {}
        moving: dict[str, np.ndarray] = {}
        length = int(record_sets[names[0]][key]["length"])
        if length < 2:
            continue
        for name in names:
            record = record_sets[name][key]
            with np.load(record["output_path"], allow_pickle=False) as payload:
                motion = np.asarray(payload["generated_raw"], dtype=np.float64)[:length]
            motions[name] = motion
            delta = motion[1:, (0, 2)] - motion[:-1, (0, 2)]
            moving[name] = np.linalg.norm(delta, axis=-1) * 30.0 >= 0.05
            sums[name]["moving"] += int(moving[name].sum())

        common = np.logical_and.reduce([moving[name] for name in names])
        union = np.logical_or.reduce([moving[name] for name in names])
        union_frames += int(union.sum())
        if not np.any(common):
            continue
        common_cases += 1
        common_frames += int(common.sum())
        for name in names:
            cosine, angle = _heading_values(motions[name], common)
            sums[name]["cosine"] += float(cosine.sum())
            sums[name]["angle"] += float(angle.sum())
            per_case[name][key] = {
                "mean_cosine": float(cosine.mean()),
                "mean_abs_angle_deg": float(angle.mean()),
            }

    _require(common_frames > 0, "Paired heading diagnostic found no common moving frames")
    baseline_name = names[0]
    paired_delta = {}
    for name in names[1:]:
        paired_keys = sorted(set(per_case[baseline_name]) & set(per_case[name]))
        _require(paired_keys, f"No paired heading cases for {name}")
        paired_delta[name] = {
            "cases": len(paired_keys),
            "mean_cosine_delta": float(
                np.mean(
                    [
                        per_case[name][key]["mean_cosine"]
                        - per_case[baseline_name][key]["mean_cosine"]
                        for key in paired_keys
                    ]
                )
            ),
            "mean_abs_angle_deg_delta": float(
                np.mean(
                    [
                        per_case[name][key]["mean_abs_angle_deg"]
                        - per_case[baseline_name][key]["mean_abs_angle_deg"]
                        for key in paired_keys
                    ]
                )
            ),
        }
    return {
        "definition": (
            "forward=[heading_sin,heading_cos] vs generated root-XZ tangent; "
            "all models use the same intersection of speed>=0.05m/s frames"
        ),
        "baseline": baseline_name,
        "cases_with_common_motion": common_cases,
        "common_moving_frames": common_frames,
        "union_moving_frames": union_frames,
        "per_model": {
            name: {
                "model_moving_frames": int(sums[name]["moving"]),
                "mean_cosine_on_common_frames": sums[name]["cosine"] / common_frames,
                "mean_abs_angle_deg_on_common_frames": sums[name]["angle"] / common_frames,
            }
            for name in names
        },
        "paired_case_delta_vs_baseline": paired_delta,
    }


def scientific_nonregression_rows(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    *,
    resamples: int,
    confidence: float,
    relative_tolerance: float,
    seed: int,
) -> list[dict[str, Any]]:
    require_matched_cases(baseline, candidate)
    rows: list[dict[str, Any]] = []
    cells = sorted({(row["text_regime"], row["subtype"]) for row in baseline.values()})
    for regime, subtype in cells:
        keys = sorted(
            key
            for key, row in baseline.items()
            if row["text_regime"] == regime and row["subtype"] == subtype
        )
        expected_metrics = PRODUCTION_METRICS_BY_SUBTYPE[str(subtype)]
        for key in keys:
            old = baseline[key]["metrics"]["generated_raw"]
            new = candidate[key]["metrics"]["generated_raw"]
            _require(frozenset(old) == expected_metrics, f"Baseline metric schema changed: {key}")
            _require(frozenset(new) == expected_metrics, f"Candidate metric schema changed: {key}")
            for metric in COUNT_METRICS:
                _require(old.get(metric, 0) == new.get(metric, 0), f"Count metric changed: {key}/{metric}")
        for metric in sorted(expected_metrics & METRIC_SPECS.keys()):
            old = np.asarray(
                [baseline[key]["metrics"]["generated_raw"][metric] for key in keys],
                dtype=np.float64,
            )
            new = np.asarray(
                [candidate[key]["metrics"]["generated_raw"][metric] for key in keys],
                dtype=np.float64,
            )
            direction, category, absolute, scale_floor = METRIC_SPECS[metric]
            degradation = signed_degradation(old, new, direction)
            mean, low, high = paired_bootstrap_mean_ci(
                degradation,
                resamples=resamples,
                confidence=confidence,
                seed=_stable_seed(seed, regime, subtype, metric),
            )
            tolerance = allowed_degradation(
                float(old.mean()), absolute, relative_tolerance, scale_floor
            )
            rows.append(
                {
                    "text_regime": regime,
                    "subtype": subtype,
                    "metric": metric,
                    "threshold_category": category,
                    "cases": len(keys),
                    "baseline_mean": float(old.mean()),
                    "candidate_mean": float(new.mean()),
                    "signed_degradation_mean": mean,
                    "signed_degradation_ci_low": low,
                    "signed_degradation_ci_high": high,
                    "allowed_degradation": tolerance,
                    "passed": bool(mean <= tolerance and high <= tolerance),
                }
            )
    return rows


def fixed16_floor_rows(
    baseline_quality: dict[str, Any], candidate_quality: dict[str, Any]
) -> list[dict[str, Any]]:
    baseline = baseline_quality["aggregate"]
    candidate = candidate_quality["aggregate"]
    rows = []
    for metric, (direction, relative, absolute) in FIXED16_FLOORS.items():
        old = float(baseline[metric])
        new = float(candidate[metric])
        _require(math.isfinite(old) and math.isfinite(new), f"Non-finite fixed16 {metric}")
        if direction == "lower":
            limit = old * (1.0 + relative) + absolute
            passed = new <= limit
        else:
            limit = old - absolute
            passed = new >= limit
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "baseline": old,
                "candidate": new,
                "limit": limit,
                "passed": bool(passed),
            }
        )
    return rows


def validate_fixed16_pair(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    *,
    candidate_checkpoint_sha: str,
    candidate_run_uuid: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_root = Path(baseline_dir)
    candidate_root = Path(candidate_dir)
    baseline_meta = _read_json(baseline_root / "metadata.json")
    candidate_meta = _read_json(candidate_root / "metadata.json")
    baseline_quality = _read_json(baseline_root / "quality.json")
    candidate_quality = _read_json(candidate_root / "quality.json")

    _require(baseline_meta.get("checkpoint_next_global_step") == 200_000, "Bad fixed16 baseline step")
    _require(candidate_meta.get("checkpoint_next_global_step") == 250_000, "Bad fixed16 candidate step")
    _require(candidate_meta.get("checkpoint_sha256") == candidate_checkpoint_sha, "Visual SHA mismatch")
    _require(candidate_meta.get("run_uuid") == candidate_run_uuid, "Visual UUID mismatch")
    for field in FIXED16_PROTOCOL_FIELDS:
        _require(
            baseline_meta.get(field) == candidate_meta.get(field),
            f"Fixed16 sampling field differs: {field}",
        )
    texts = baseline_meta.get("texts")
    lengths = baseline_meta.get("lengths")
    _require(isinstance(texts, list) and len(texts) == 16, "Fixed16 baseline must contain 16 prompts")
    _require(isinstance(lengths, list) and len(lengths) == 16, "Fixed16 baseline must contain 16 lengths")
    _require(candidate_meta.get("texts") == texts, "Fixed16 prompts differ")
    _require(candidate_meta.get("lengths") == lengths, "Fixed16 lengths differ")
    for name, quality in (("baseline", baseline_quality), ("candidate", candidate_quality)):
        per_sample = quality.get("per_sample")
        _require(isinstance(per_sample, list) and len(per_sample) == 16, f"{name} fixed16 rows != 16")
        for index, row in enumerate(per_sample):
            _require(row.get("index") == index, f"{name} fixed16 index mismatch")
            _require(row.get("text") == texts[index], f"{name} fixed16 prompt mismatch")
            _require(row.get("length") == lengths[index], f"{name} fixed16 length mismatch")
    return baseline_quality, candidate_quality


def _checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    import torch

    return torch.load(Path(path), map_location="cpu", mmap=True, weights_only=False)


def evaluate_gate(args: argparse.Namespace) -> int:
    actual_checkpoint_sha = sha256_file(args.checkpoint)
    _require(actual_checkpoint_sha == args.checkpoint_sha256.lower(), "Candidate SHA mismatch")
    checkpoint = _checkpoint_metadata(args.checkpoint)
    _require(checkpoint.get("format") == CHECKPOINT_FORMAT, "Candidate checkpoint format mismatch")
    _require(checkpoint.get("train_contract") == R12_TRAIN_CONTRACT, "Candidate is not R12")
    _require(checkpoint.get("next_global_step") == 250_000, "Candidate is not step 250K")
    _require(checkpoint.get("context_update_count") == 0, "Context updated before R12 B2")
    checkpoint_origin = validate_r12_origin_parent_identity(
        checkpoint.get("runtime_identity", {}).get("origin_parent")
    )
    run_identity = _read_json(args.run_identity)
    _require(run_identity.get("run_name") == checkpoint.get("run_name"), "Run name mismatch")
    _require(run_identity.get("run_uuid") == checkpoint.get("run_uuid"), "Run UUID mismatch")
    _require(
        validate_r12_origin_parent_identity(run_identity.get("origin_parent"))
        == checkpoint_origin,
        "Run/checkpoint origin mismatch",
    )

    protocol = require_matched_scientific_protocol(
        args.baseline_eval, args.r11_eval, args.candidate_eval
    )
    baseline_index, baseline = load_eval(args.baseline_eval)
    r11_index, r11 = load_eval(args.r11_eval)
    candidate_index, candidate = load_eval(args.candidate_eval)
    require_matched_cases(baseline, r11, candidate)
    _require(baseline_index["checkpoint_sha256"] == BASELINE_200K_SHA256, "Bad 200K evidence")
    _require(r11_index["checkpoint_sha256"] == R11_250K_SHA256, "Bad R11 250K evidence")
    _require(candidate_index["checkpoint_sha256"] == actual_checkpoint_sha, "Bad R12 evidence")

    superiority = root_superiority_rows(
        baseline,
        candidate,
        resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    skate = skate_improvement_rows(r11, candidate)
    heading = paired_heading_path_diagnostic(
        {
            "baseline_200k": baseline,
            "r11_250k": r11,
            "r12_250k": candidate,
        }
    )
    nonregression = scientific_nonregression_rows(
        baseline,
        candidate,
        resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        relative_tolerance=args.relative_tolerance,
        seed=args.seed,
    )
    nonregression_passed = all(row["passed"] for row in nonregression)

    baseline_visual = Path(args.baseline_visual)
    candidate_visual = Path(args.candidate_visual)
    baseline_quality, candidate_quality = validate_fixed16_pair(
        baseline_visual,
        candidate_visual,
        candidate_checkpoint_sha=actual_checkpoint_sha,
        candidate_run_uuid=checkpoint["run_uuid"],
    )
    fixed16 = fixed16_floor_rows(baseline_quality, candidate_quality)

    human = _read_json(args.human_verdict)
    human_passed = bool(
        human.get("format") == "hy273_r12_fixed16_human_verdict_v1"
        and human.get("status") == "passed"
        and human.get("checkpoint_sha256") == actual_checkpoint_sha
        and human.get("run_uuid") == checkpoint.get("run_uuid")
        and isinstance(human.get("review"), str)
        and bool(human["review"].strip())
    )

    checks = {
        "root_superiority": all(row["passed"] for row in superiority),
        "control_nonregression": nonregression_passed,
        "r11_xz_skate_strict_improvement": all(row["passed"] for row in skate),
        "heading_path_diagnostic_measured": heading["common_moving_frames"] > 0,
        "fixed16_physical_floors": all(row["passed"] for row in fixed16),
        "fixed16_human_verdict": human_passed,
    }
    passed = all(checks.values())
    payload = {
        "format": FORMAT,
        "status": "passed" if passed else "failed",
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": actual_checkpoint_sha,
            "next_global_step": 250_000,
            "train_contract": R12_TRAIN_CONTRACT,
            "run_name": checkpoint["run_name"],
            "run_uuid": checkpoint["run_uuid"],
            "origin_parent": checkpoint_origin,
        },
        "checks": checks,
        "root_superiority": {
            "method": "one-sided paired bootstrap; directional upper confidence bound < 0",
            "resamples": args.bootstrap_resamples,
            "confidence": args.confidence,
            "rows": superiority,
        },
        "scientific_protocol": protocol,
        "control_nonregression": {
            "method": "matched-case paired bootstrap without code-SHA gating",
            "relative_tolerance": args.relative_tolerance,
            "rows": nonregression,
        },
        "r11_xz_skate_strict_improvement": skate,
        "heading_path_diagnostic": heading,
        "fixed16_physical_floors": fixed16,
        "fixed16_human_verdict": human,
        "evidence": {
            "baseline_200k": str(Path(args.baseline_eval).resolve()),
            "r11_250k": str(Path(args.r11_eval).resolve()),
            "r12_250k": str(Path(args.candidate_eval).resolve()),
            "baseline_visual": str(baseline_visual.resolve()),
            "candidate_visual": str(candidate_visual.resolve()),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks, "output": str(output.resolve())}, sort_keys=True))
    return 0 if passed else 2


def validate_resume(args: argparse.Namespace) -> int:
    gate = _read_json(args.gate_artifact)
    _require(gate.get("format") == FORMAT and gate.get("status") == "passed", "R12 gate did not pass")
    checkpoint = _checkpoint_metadata(args.checkpoint)
    _require(checkpoint.get("format") == CHECKPOINT_FORMAT, "Resume checkpoint format mismatch")
    _require(checkpoint.get("train_contract") == R12_TRAIN_CONTRACT, "Resume checkpoint is not R12")
    _require(checkpoint.get("next_global_step") == 250_000, "B2 requires exact R12 250K")
    _require(checkpoint.get("run_uuid") == gate["checkpoint"]["run_uuid"], "Gate/resume UUID mismatch")
    _require(args.checkpoint_sha256.lower() == gate["checkpoint"]["sha256"], "Gate/resume SHA mismatch")
    origin = validate_r12_origin_parent_identity(
        checkpoint.get("runtime_identity", {}).get("origin_parent")
    )
    _require(origin == gate["checkpoint"]["origin_parent"], "Gate/resume origin mismatch")
    print(json.dumps({"status": "passed", "gate": str(Path(args.gate_artifact).resolve())}, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    for name in (
        "baseline_eval",
        "r11_eval",
        "candidate_eval",
        "checkpoint",
        "checkpoint_sha256",
        "run_identity",
        "baseline_visual",
        "candidate_visual",
        "human_verdict",
        "output",
    ):
        evaluate.add_argument(f"--{name}", required=True)
    evaluate.add_argument("--bootstrap_resamples", type=int, default=10_000)
    evaluate.add_argument("--confidence", type=float, default=0.95)
    evaluate.add_argument("--relative_tolerance", type=float, default=0.05)
    evaluate.add_argument("--seed", type=int, default=3407)
    validate = sub.add_parser("validate-resume")
    validate.add_argument("--gate_artifact", required=True)
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument("--checkpoint_sha256", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        raise SystemExit(evaluate_gate(args))
    raise SystemExit(validate_resume(args))


if __name__ == "__main__":
    main()
