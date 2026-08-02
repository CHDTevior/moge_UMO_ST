#!/usr/bin/env python3
"""Reproducible case-paired bootstrap for two HY273 control evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    ("constraint_root2d_err", "root_error_m"),
    ("constraint_root2d_acc", "root_accuracy"),
    ("constraint_end_effector", "endpoint_position_error_m"),
    ("constraint_end_effector_rotation_deg", "endpoint_rotation_error_deg"),
    ("constraint_fullbody_keyframe", "fullbody_error_m"),
    ("controlled_contact_accuracy", "controlled_contact_accuracy"),
    ("controlled_contact_bce", "controlled_contact_bce"),
    ("fk_position_rotation_consistency_cm", "fk_consistency_cm"),
    ("foot_contact_consistency", "foot_contact_consistency"),
    ("foot_skate_from_pred_contacts", "contact_foot_velocity_mps"),
    ("foot_skate_max_vel", "max_foot_velocity_mps"),
    ("foot_skate_ratio", "foot_skate_ratio"),
)


def load_records(eval_dir: Path) -> dict[str, dict[str, Any]]:
    paths = sorted((eval_dir / "shards").glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No shard JSONL files under {eval_dir}")
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("status") != "ok":
                    raise RuntimeError(f"Non-ok case in {path}: {row.get('case_key')}")
                key = str(row["case_key"])
                if key in records:
                    raise RuntimeError(f"Duplicate case key: {key}")
                records[key] = row
    return records


def row_seed(base_seed: int, regime: str, metric: str) -> int:
    payload = f"{base_seed}\0{regime}\0{metric}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def paired_percentile_interval(
    deltas: np.ndarray,
    *,
    resamples: int,
    confidence: float,
    seed: int,
    chunk_size: int,
) -> tuple[float, float, float]:
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1 or not len(deltas) or not np.isfinite(deltas).all():
        raise ValueError("Paired deltas must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(
            0,
            len(deltas),
            size=(stop - start, len(deltas)),
            dtype=np.int32,
        )
        means[start:stop] = deltas[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(
        means,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    return float(deltas.mean()), float(low), float(high)


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row["metrics"]["generated_raw"].get(metric)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    baseline_dir = Path(args.baseline_eval_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_eval_dir).expanduser().resolve()
    baseline = load_records(baseline_dir)
    candidate = load_records(candidate_dir)
    if set(baseline) != set(candidate):
        missing_baseline = sorted(set(candidate) - set(baseline))
        missing_candidate = sorted(set(baseline) - set(candidate))
        raise RuntimeError(
            "Case-key mismatch: "
            f"missing_baseline={missing_baseline[:5]} "
            f"missing_candidate={missing_candidate[:5]}"
        )

    rows = []
    for regime in ("withtext", "notext"):
        regime_keys = sorted(
            key
            for key, row in baseline.items()
            if str(row["text_regime"]) == regime
        )
        for metric, label in METRICS:
            keys = [
                key
                for key in regime_keys
                if metric_value(baseline[key], metric) is not None
                and metric_value(candidate[key], metric) is not None
            ]
            if not keys:
                continue
            baseline_values = np.asarray(
                [metric_value(baseline[key], metric) for key in keys],
                dtype=np.float64,
            )
            candidate_values = np.asarray(
                [metric_value(candidate[key], metric) for key in keys],
                dtype=np.float64,
            )
            seed = row_seed(args.seed, regime, metric)
            mean, low, high = paired_percentile_interval(
                candidate_values - baseline_values,
                resamples=args.resamples,
                confidence=args.confidence,
                seed=seed,
                chunk_size=args.chunk_size,
            )
            rows.append(
                {
                    "text_regime": regime,
                    "metric": metric,
                    "label": label,
                    "case_count": len(keys),
                    "baseline_mean": float(baseline_values.mean()),
                    "candidate_mean": float(candidate_values.mean()),
                    "candidate_minus_baseline_mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "row_seed": seed,
                }
            )

    return {
        "format": "hy273_control_case_paired_bootstrap_v1",
        "status": "validated",
        "baseline_eval_dir": str(baseline_dir),
        "candidate_eval_dir": str(candidate_dir),
        "aligned_case_count": len(baseline),
        "estimand": (
            "generated_raw case-macro; candidate minus baseline; each applicable "
            "case contributes one paired scalar"
        ),
        "bootstrap": {
            "method": "paired_percentile_two_sided",
            "resampling_unit": "case_key",
            "resamples": int(args.resamples),
            "confidence": float(args.confidence),
            "base_seed": int(args.seed),
            "row_seed_derivation": "sha256(base_seed, text_regime, metric)",
            "indices_shared_across_metrics": False,
            "quantile_method": "numpy.linear",
            "chunk_size": int(args.chunk_size),
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_eval_dir", required=True)
    parser.add_argument("--candidate_eval_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--chunk_size", type=int, default=256)
    args = parser.parse_args()
    if args.resamples <= 0 or args.chunk_size <= 0:
        parser.error("resamples and chunk_size must be positive")
    if not 0.0 < args.confidence < 1.0:
        parser.error("confidence must be in (0, 1)")
    return args


def main() -> None:
    args = parse_args()
    payload = build_summary(args)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "aligned_case_count": payload["aligned_case_count"],
                "row_count": len(payload["rows"]),
                "output_json": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
