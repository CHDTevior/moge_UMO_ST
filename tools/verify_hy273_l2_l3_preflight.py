#!/usr/bin/env python3
"""Verify the paired HY273 L2/L3 scratch-training launch gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


TRAIN_LINE = re.compile(r"^\[train\] epoch=(\d+) step=(\d+) (.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("run_logs/hy273_l2_l3_calibration_scratch_seed3407_n16_final.json"),
    )
    parser.add_argument(
        "--l2_run",
        type=Path,
        default=Path("checkpoints/t2m/hy273_l2_scratch_ddp4_preflight101_20260710"),
    )
    parser.add_argument(
        "--l3_run",
        type=Path,
        default=Path("checkpoints/t2m/hy273_l3_scratch_ddp4_preflight101_20260710"),
    )
    parser.add_argument(
        "--l2_log",
        type=Path,
        default=Path("logs/hy273_l2_scratch_ddp4_preflight101_20260710.log"),
    )
    parser.add_argument(
        "--l3_log",
        type=Path,
        default=Path("logs/hy273_l3_scratch_ddp4_preflight101_20260710.log"),
    )
    parser.add_argument(
        "--resume_run",
        type=Path,
        default=Path("checkpoints/t2m/hy273_l2_scratch_ddp4_resume100_to101_20260710"),
    )
    parser.add_argument(
        "--resume_log",
        type=Path,
        default=Path("logs/hy273_l2_scratch_ddp4_resume100_to101_20260710.log"),
    )
    parser.add_argument(
        "--reference_run",
        type=Path,
        default=Path("checkpoints/t2m/hy273_l2_scratch_ddp4_reference101_20260710"),
    )
    parser.add_argument("--trace_steps", type=int, default=100)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--checkpoint_step", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--source_manifest",
        type=Path,
        default=Path("run_logs/hy273_l2_l3_scratch_source_manifest.sha256"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("run_logs/hy273_l2_l3_scratch_preflight_report.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_sha(rows: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        if len(parts) != 2 or len(parts[0]) != 64:
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


def parse_training_log(path: Path, grad_clip: float) -> dict[str, Any]:
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

    grad_norms = [row["metrics"]["grad_norm_pre_clip"] for row in rows]
    non_finite_metrics = [
        {"step": row["step"], "metric": key, "value": value}
        for row in rows
        for key, value in row["metrics"].items()
        if not math.isfinite(value)
    ]
    lower = text.lower()
    fatal_markers = {
        marker: lower.count(marker)
        for marker in ("traceback", "out of memory", "non-finite loss")
    }
    return {
        "path": str(path.resolve()),
        "logged_steps": len(rows),
        "first_step": rows[0]["step"] if rows else None,
        "last_step": rows[-1]["step"] if rows else None,
        "max_grad_norm_pre_clip": max(grad_norms) if grad_norms else None,
        "grad_clip_threshold": grad_clip,
        "grad_clip_hits": sum(value >= grad_clip for value in grad_norms),
        "grad_clip_hit_rate": (
            sum(value >= grad_clip for value in grad_norms) / len(grad_norms)
            if grad_norms
            else None
        ),
        "non_finite_metrics": non_finite_metrics,
        "fatal_markers": fatal_markers,
        "rows": rows,
        "passed": bool(rows)
        and not non_finite_metrics
        and not any(fatal_markers.values()),
    }


def metrics_at_step(log_report: dict[str, Any], step: int) -> dict[str, float]:
    matches = [row["metrics"] for row in log_report["rows"] if row["step"] == step]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one step={step} row in {log_report['path']}, got {len(matches)}"
        )
    return matches[0]


def trace_report(
    l2_run: Path,
    l3_run: Path,
    reference_run: Path,
    resume_run: Path,
    world_size: int,
    trace_steps: int,
) -> dict[str, Any]:
    rank_reports: list[dict[str, Any]] = []
    paired_equal = True
    restart_equal = True
    for rank in range(world_size):
        filename = f"trace_rank{rank:02d}.jsonl"
        l2_rows = load_jsonl(l2_run / "logs" / filename)[:trace_steps]
        l3_rows = load_jsonl(l3_run / "logs" / filename)[:trace_steps]
        expected_steps = list(range(trace_steps))
        l2_steps = [int(row["optimizer_step"]) for row in l2_rows]
        l3_steps = [int(row["optimizer_step"]) for row in l3_rows]
        paired_rank_equal = l2_rows == l3_rows and l2_steps == expected_steps and l3_steps == expected_steps
        paired_equal = paired_equal and paired_rank_equal

        reference_rows = load_jsonl(reference_run / "logs" / filename)
        resume_rows = load_jsonl(resume_run / "logs" / filename)
        reference_step = [row for row in reference_rows if row["optimizer_step"] == trace_steps]
        resume_step = [row for row in resume_rows if row["optimizer_step"] == trace_steps]
        restart_rank_equal = (
            len(reference_step) == 1
            and len(resume_step) == 1
            and reference_step[0] == resume_step[0]
        )
        restart_equal = restart_equal and restart_rank_equal
        rank_reports.append(
            {
                "rank": rank,
                "paired_steps": len(l2_rows),
                "l2_sha256": canonical_sha(l2_rows),
                "l3_sha256": canonical_sha(l3_rows),
                "paired_equal": paired_rank_equal,
                "restart_step": trace_steps,
                "reference_restart_digest": reference_step[0] if len(reference_step) == 1 else None,
                "resumed_restart_digest": resume_step[0] if len(resume_step) == 1 else None,
                "restart_equal": restart_rank_equal,
            }
        )
    return {
        "trace_steps": trace_steps,
        "rank_reports": rank_reports,
        "l2_l3_exact_match": paired_equal,
        "restart_input_exact_match": restart_equal,
        "passed": paired_equal and restart_equal,
    }


def checkpoint_report(path: Path, expected_step: int) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    train_state = checkpoint.get("train_state")
    report = {
        "path": str(path.resolve()),
        "step": int(checkpoint.get("step", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "train_state": train_state,
        "model_keys": len(checkpoint.get("model", {})),
        "optimizer_state_entries": len(checkpoint.get("optimizer", {}).get("state", {})),
        "ema_keys": len(checkpoint.get("ema", {})),
    }
    report["passed"] = (
        report["step"] == expected_step
        and isinstance(train_state, dict)
        and train_state.get("format") == "hy273_train_cursor_v1"
        and int(train_state.get("next_epoch", -1)) == 0
        and int(train_state.get("next_step_in_epoch", -1)) == expected_step
        and int(train_state.get("world_size", -1)) == 4
        and int(train_state.get("batch_size_per_rank", -1)) == 16
        and int(train_state.get("gradient_accumulation_steps", -1)) == 2
        and int(train_state.get("effective_global_batch", -1)) == 128
        and report["model_keys"] > 0
        and report["optimizer_state_entries"] > 0
        and report["ema_keys"] == report["model_keys"]
    )
    del checkpoint
    gc.collect()
    return report


def main() -> None:
    args = parse_args()
    calibration = load_json(args.calibration)
    l2_contract = load_json(args.l2_run / "trace_contract.json")
    l3_contract = load_json(args.l3_run / "trace_contract.json")
    expected_sha = str(calibration["initial_model_sha256"])
    selected_lambda = float(calibration["selected_lambda"])
    selected = next(
        candidate
        for candidate in calibration["lambda_candidates"]
        if float(candidate["lambda"]) == selected_lambda
    )
    calibration_report = {
        "path": str(args.calibration.resolve()),
        "passed_flag": bool(calibration.get("passed")),
        "initial_model_sha256": expected_sha,
        "representation_scale_alpha": float(calibration["representation_scale_alpha"]),
        "selected_lambda": selected_lambda,
        "aggregate_ratio": float(selected["aggregate_ratio"]),
        "max_timestep_bin_ratio": max(float(value) for value in selected["bin_aggregate_ratios"].values()),
    }
    calibration_report["passed"] = (
        calibration_report["passed_flag"]
        and 0.05 <= calibration_report["aggregate_ratio"] <= 0.10
        and calibration_report["max_timestep_bin_ratio"] <= 0.15
    )

    contract_fields = (
        "format",
        "trace_seed",
        "world_size",
        "microbatch_per_rank",
        "gradient_accumulation_steps",
        "effective_global_batch",
        "trace_hash_steps",
        "streams",
    )
    contract_report = {
        "l2": l2_contract,
        "l3": l3_contract,
        "comparable_fields": list(contract_fields),
        "initial_sha_matches_calibration": (
            l2_contract.get("initial_model_sha256")
            == l3_contract.get("initial_model_sha256")
            == expected_sha
        ),
        "paired_contract_equal": all(l2_contract.get(key) == l3_contract.get(key) for key in contract_fields),
    }
    contract_report["passed"] = (
        contract_report["initial_sha_matches_calibration"]
        and contract_report["paired_contract_equal"]
    )

    l2_log = parse_training_log(args.l2_log, args.grad_clip)
    l3_log = parse_training_log(args.l3_log, args.grad_clip)
    resume_log = parse_training_log(args.resume_log, args.grad_clip)
    uninterrupted_metrics = metrics_at_step(l2_log, args.checkpoint_step + 1)
    resumed_metrics = metrics_at_step(resume_log, args.checkpoint_step + 1)
    resume_metrics_equal = uninterrupted_metrics == resumed_metrics

    traces = trace_report(
        args.l2_run,
        args.l3_run,
        args.reference_run,
        args.resume_run,
        args.world_size,
        args.trace_steps,
    )
    checkpoint = checkpoint_report(
        args.l2_run / "model" / f"step_{args.checkpoint_step:08d}.pt",
        args.checkpoint_step,
    )
    resume_report = {
        "uninterrupted_step": args.checkpoint_step + 1,
        "uninterrupted_metrics": uninterrupted_metrics,
        "resumed_metrics": resumed_metrics,
        "logged_metrics_exact_match": resume_metrics_equal,
        "input_trace_exact_match": traces["restart_input_exact_match"],
        "passed": resume_metrics_equal and traces["restart_input_exact_match"],
    }

    checks = {
        "source_binding": source_binding_report(args.source_manifest),
        "calibration": calibration_report,
        "trace_contract": contract_report,
        "paired_trace": traces,
        "l2_log": {key: value for key, value in l2_log.items() if key != "rows"},
        "l3_log": {key: value for key, value in l3_log.items() if key != "rows"},
        "checkpoint": checkpoint,
        "resume": resume_report,
    }
    passed = all(
        check["passed"]
        for check in checks.values()
        if isinstance(check, dict) and "passed" in check
    )
    report = {
        "format": "hy273_l2_l3_scratch_preflight_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
