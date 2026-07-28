#!/usr/bin/env python3
"""Verify the HY273 L3 clean-head/JiT-v-loss launch gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAIN_LINE = re.compile(r"^\[train\] epoch=(\d+) step=(\d+) (.*)$")
EXPECTED_INITIAL_SHA = "808a1639f7ec134887ce07b3d2634849dccfe68e6441bac0addba4f4cc579e59"
EXPECTED_SCALE = 0.09397019716051493
EXPECTED_FK_LAMBDA = 0.07
PAYLOAD_MANIFESTS = (
    Path("run_logs/hy273_l3_vloss_motion_payload.sha256"),
    Path("run_logs/hy273_l3_vloss_text_payload.sha256"),
    Path("run_logs/hy273_l3_vloss_hytext_payload.sha256"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot_log",
        type=Path,
        default=Path("logs/hy273_l3_vloss_jit_bound_pilot500_v2.log"),
    )
    parser.add_argument(
        "--pilot_run",
        type=Path,
        default=Path(
            "checkpoints/t2m/hy273_l3_vloss_jit_bound_pilot500_v2"
        ),
    )
    parser.add_argument(
        "--train_calibration",
        type=Path,
        default=Path(
            "run_logs/hy273_l3_vloss_jitpm08ps08_calibration_train_t_seed3407_n4096.json"
        ),
    )
    parser.add_argument(
        "--bin_calibration",
        type=Path,
        default=Path(
            "run_logs/hy273_l3_vloss_jitpm08ps08_calibration_bins_seed3407_n16_alpha0939702.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run_logs/hy273_l3_vloss_jit_preflight_report.json"),
    )
    parser.add_argument(
        "--source_manifest",
        type=Path,
        default=Path("run_logs/hy273_l3_vloss_source_manifest.sha256"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_binding_report(path: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise RuntimeError(f"Malformed source manifest line {line_number}: {line!r}")
        expected, filename = parts
        filename = filename.lstrip("*")
        source_path = Path(filename)
        if not source_path.is_absolute():
            source_path = Path.cwd() / source_path
        actual = sha256_file(source_path)
        entries.append(
            {
                "path": filename,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    return {
        "path": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
        "entries": entries,
        "passed": bool(entries) and all(entry["matches"] for entry in entries),
    }


def close(left: Any, right: float, atol: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return float(ordered[index])


def metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": float(max(values)),
        "last": float(values[-1]),
    }


def parse_pilot_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = TRAIN_LINE.match(line)
        if match is None:
            continue
        metrics: dict[str, float] = {}
        for token in match.group(3).split():
            key, value = token.split("=", maxsplit=1)
            metrics[key] = float(value)
        rows.append(
            {
                "epoch": int(match.group(1)),
                "step": int(match.group(2)),
                "metrics": metrics,
            }
        )

    required = {
        "loss",
        "flow",
        "grad_norm_pre_clip",
        "grad_clip_active",
        "prediction_type_x0",
        "loss_space_velocity",
        "timestep_mean",
        "timestep_min",
        "timestep_max",
        "velocity_x0_weight_scaled_mean",
        "velocity_x0_weight_scaled_max",
    }
    missing = [
        {"step": row["step"], "keys": sorted(required - row["metrics"].keys())}
        for row in rows
        if not required.issubset(row["metrics"])
    ]
    non_finite = [
        {"step": row["step"], "metric": name, "value": value}
        for row in rows
        for name, value in row["metrics"].items()
        if not math.isfinite(value)
    ]
    steps = [row["step"] for row in rows]
    metric_names = sorted(required) if rows and not missing else []
    summaries = {
        name: metric_summary([row["metrics"][name] for row in rows])
        for name in metric_names
    }
    first_flow = statistics.fmean(row["metrics"]["flow"] for row in rows[:50]) if rows else math.nan
    last_flow = statistics.fmean(row["metrics"]["flow"] for row in rows[-50:]) if rows else math.nan
    lower = text.lower()
    fatal_markers = {
        marker: lower.count(marker)
        for marker in ("traceback", "out of memory", "non-finite loss")
    }
    clip_hits = (
        sum(row["metrics"]["grad_clip_active"] > 0.0 for row in rows)
        if rows and not missing
        else -1
    )
    passed = (
        steps == list(range(1, 501))
        and not missing
        and not non_finite
        and not any(fatal_markers.values())
        and clip_hits == 0
        and all(row["metrics"]["prediction_type_x0"] == 1.0 for row in rows)
        and all(row["metrics"]["loss_space_velocity"] == 1.0 for row in rows)
        and last_flow < first_flow
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "logged_steps": len(rows),
        "first_step": steps[0] if steps else None,
        "last_step": steps[-1] if steps else None,
        "missing_metrics": missing,
        "non_finite_metrics": non_finite,
        "fatal_markers": fatal_markers,
        "grad_clip_hits": clip_hits,
        "grad_clip_hit_rate": clip_hits / len(rows) if rows and clip_hits >= 0 else None,
        "first_50_flow_mean": first_flow,
        "last_50_flow_mean": last_flow,
        "metrics": summaries,
        "passed": passed,
    }


def pilot_contract_report(run_dir: Path, source_manifest_sha256: str) -> dict[str, Any]:
    config_path = run_dir / "config_resolved.json"
    trace_path = run_dir / "trace_contract.json"
    config = load_json(config_path)
    trace = load_json(trace_path)
    config_checks = {
        "prediction_type": config.get("prediction_type") == "x0",
        "loss_space": config.get("representation_loss_space") == "velocity",
        "representation_scale": close(config.get("representation_loss_scale"), EXPECTED_SCALE),
        "velocity_t_eps": close(config.get("velocity_loss_t_eps"), 0.05),
        "time_schedule": config.get("time_schedule") == "logit_normal",
        "p_mean": close(config.get("denoiser_p_mean"), -0.8),
        "p_std": close(config.get("denoiser_p_std"), 0.8),
        "fk_lambda": close(config.get("fk_consistency_loss_weight"), EXPECTED_FK_LAMBDA),
        "max_steps": int(config.get("max_steps", -1)) == 500,
        "save_every": int(config.get("save_every", -1)) == 0,
        "save_final": config.get("save_final") is False,
        "batch_size": int(config.get("batch_size", -1)) == 16,
        "accumulation": int(config.get("gradient_accumulation_steps", -1)) == 2,
        "source_manifest_sha": config.get("source_manifest_sha256")
        == source_manifest_sha256,
    }
    trace_checks = {
        "format": trace.get("format") == "hy273_stateless_trace_v1",
        "world_size": int(trace.get("world_size", -1)) == 4,
        "effective_global_batch": int(trace.get("effective_global_batch", -1)) == 128,
        "trace_seed": int(trace.get("trace_seed", -1)) == 3407,
        "trace_hash_steps": int(trace.get("trace_hash_steps", -1)) == 100,
        "initial_model_sha": trace.get("initial_model_sha256") == EXPECTED_INITIAL_SHA,
        "source_manifest_sha": trace.get("source_manifest_sha256")
        == source_manifest_sha256,
        "cuda_visible_devices": trace.get("cuda_visible_devices") == "0,1,2,3",
        "master_port": trace.get("master_port") == "29833",
    }
    rank_reports = []
    for rank in range(4):
        path = run_dir / "logs" / f"trace_rank{rank:02d}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        valid = (
            [int(row.get("optimizer_step", -1)) for row in rows] == list(range(100))
            and all(len(row.get("micro_digests", [])) == 2 for row in rows)
        )
        rank_reports.append(
            {
                "rank": rank,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "passed": valid,
            }
        )
    passed = all(config_checks.values()) and all(trace_checks.values()) and all(
        item["passed"] for item in rank_reports
    )
    return {
        "run_dir": str(run_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "trace_contract_path": str(trace_path.resolve()),
        "trace_contract_sha256": sha256_file(trace_path),
        "config_checks": config_checks,
        "trace_checks": trace_checks,
        "rank_traces": rank_reports,
        "passed": passed,
    }


def calibration_report(train_path: Path, bin_path: Path) -> dict[str, Any]:
    train = load_json(train_path)
    bins = load_json(bin_path)
    train_candidate = next(
        (
            item
            for item in train.get("lambda_candidates", [])
            if close(item.get("lambda"), EXPECTED_FK_LAMBDA)
        ),
        None,
    )
    bin_candidate = next(
        (
            item
            for item in bins.get("lambda_candidates", [])
            if close(item.get("lambda"), EXPECTED_FK_LAMBDA)
        ),
        None,
    )
    aggregate_ratio = (
        float(train_candidate["aggregate_ratio"]) if train_candidate is not None else math.nan
    )
    max_bin_ratio = (
        max(float(value) for value in bin_candidate["bin_aggregate_ratios"].values())
        if bin_candidate is not None
        else math.nan
    )
    common_checks = {
        "loss_space": train.get("calibration_loss_space") == "velocity",
        "time_schedule": train.get("time_schedule") == "logit_normal",
        "p_mean": close(train.get("denoiser_p_mean"), -0.8),
        "p_std": close(train.get("denoiser_p_std"), 0.8),
        "velocity_t_eps": close(train.get("velocity_loss_t_eps"), 0.05),
        "representation_scale": close(train.get("representation_scale_alpha"), EXPECTED_SCALE),
        "initial_model_sha": train.get("initial_model_sha256") == EXPECTED_INITIAL_SHA,
        "training_distribution": train.get("sample_training_timesteps") is True,
        "sample_count": int(train.get("sample_count", -1)) == 4096,
        "selected_lambda": close(train.get("selected_lambda"), EXPECTED_FK_LAMBDA),
        "train_passed": train.get("passed") is True,
        "aggregate_ratio": 0.05 <= aggregate_ratio <= 0.10,
        "max_bin_ratio": max_bin_ratio <= 0.15,
        "bin_scale": close(bins.get("representation_scale_alpha"), EXPECTED_SCALE),
        "bin_loss_space": bins.get("calibration_loss_space") == "velocity",
        "bin_passed": bins.get("passed") is True,
        "bin_selection_mode": bins.get("selection_mode") == "max_bin_only",
        "bin_selected_lambda": close(bins.get("selected_lambda"), EXPECTED_FK_LAMBDA),
        "bin_capped_regime_audited": any(
            close(value, 0.99) for value in bins.get("t_bins", [])
        ),
    }
    return {
        "train_path": str(train_path.resolve()),
        "train_sha256": sha256_file(train_path),
        "bin_path": str(bin_path.resolve()),
        "bin_sha256": sha256_file(bin_path),
        "aggregate_gradient_ratio": aggregate_ratio,
        "max_fixed_timestep_bin_gradient_ratio": max_bin_ratio,
        "checks": common_checks,
        "passed": all(common_checks.values()),
    }


def main() -> None:
    args = parse_args()
    source_binding = source_binding_report(args.source_manifest)
    source_manifest_sha256 = str(source_binding["manifest_sha256"])
    payload_reports = [source_binding_report(path) for path in PAYLOAD_MANIFESTS]
    payload_binding = {
        "manifests": payload_reports,
        "passed": all(report["passed"] for report in payload_reports),
    }
    checks = {
        "source_binding": source_binding,
        "payload_binding": payload_binding,
        "pilot_log": parse_pilot_log(args.pilot_log),
        "pilot_contract": pilot_contract_report(
            args.pilot_run, source_manifest_sha256
        ),
        "calibration": calibration_report(args.train_calibration, args.bin_calibration),
    }
    report = {
        "format": "hy273_l3_clean_head_jit_vloss_preflight_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected": {
            "prediction_type": "x0",
            "representation_loss_space": "velocity",
            "time_schedule": "logit_normal",
            "denoiser_p_mean": -0.8,
            "denoiser_p_std": 0.8,
            "velocity_loss_t_eps": 0.05,
            "representation_scale": EXPECTED_SCALE,
            "fk_consistency_lambda": EXPECTED_FK_LAMBDA,
            "initial_model_sha256": EXPECTED_INITIAL_SHA,
            "source_manifest_sha256": source_manifest_sha256,
        },
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve())}, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
