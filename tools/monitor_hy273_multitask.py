#!/usr/bin/env python
"""One-shot health check for HY273 multitask metrics.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any


LOSS = "loss/overall/backward_total"
GRAD = "grad/total_preclip"
CLIP = "grad/clip_active"
THROUGHPUT = "throughput/samples_per_second"
MEMORY = "memory/max_allocated_gib"
LOSS_PERCENT_METRICS = {
    "representation": "loss/overall/group_representation/percent_total",
    "contact": "loss/overall/group_contact_all/percent_total",
    "control_continuous": "loss/overall/group_control_continuous/percent_total",
    "control_contact": "loss/overall/group_control_contact/percent_total",
    "fk_consistency": "loss/overall/group_fk_consistency/percent_total",
    "clean_root_velocity": "loss/overall/group_clean_root_velocity/percent_total",
    "clean_joint_velocity": "loss/overall/group_clean_joint_velocity/percent_total",
    "foot_lock": "loss/overall/group_foot_lock/percent_total",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row.get("metrics"), dict):
                raise RuntimeError(f"Missing metrics object at {path}:{line_number}")
            rows.append(row)
    return rows


def _metric(row: dict[str, Any], name: str, default: float = float("nan")) -> float:
    return float(row["metrics"].get(name, default))


def _window(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [row for row in rows if start <= int(row["step"]) <= end]


def _require_exact(
    bad: list[str], row: dict[str, Any], expected: dict[str, float], label: str
) -> None:
    for name, target in expected.items():
        value = _metric(row, name)
        if not math.isfinite(value) or abs(value - target) > 1e-12:
            bad.append(f"{label} contract violation: {name}={value}, expected {target}")


def _require_probability_range(
    bad: list[str], row: dict[str, Any], name: str, low: float, high: float, label: str
) -> None:
    value = _metric(row, name)
    if not math.isfinite(value) or value < low - 1e-12 or value > high + 1e-12:
        bad.append(f"{label} schedule violation: {name}={value}, expected [{low}, {high}]")


def evaluate_health(
    rows: list[dict[str, Any]],
    *,
    stale_minutes: float,
    metrics_mtime: float,
    min_throughput: float,
    max_memory_gib: float,
) -> dict[str, Any]:
    if not rows:
        return {"status": "bad", "reasons": ["no train metrics found"], "last": {}}

    last = rows[-1]
    step = int(last["step"])
    metrics = last["metrics"]
    bad: list[str] = []
    warnings: list[str] = []

    required = (LOSS, GRAD, CLIP, THROUGHPUT, MEMORY)
    for name in required:
        value = _metric(last, name)
        if not math.isfinite(value):
            bad.append(f"non-finite or missing metric: {name}")

    age_minutes = max(0.0, (time.time() - metrics_mtime) / 60.0)
    if stale_minutes > 0 and age_minutes > stale_minutes:
        bad.append(f"metrics stale for {age_minutes:.1f} minutes")

    stage = str(last.get("stage", ""))
    if stage == "stage_a_t2m":
        _require_exact(
            bad,
            last,
            {
                "schedule/p_t2m": 1.0,
                "schedule/p_control": 0.0,
                "schedule/p_edit": 0.0,
                "batch/mask_fraction": 0.0,
                "batch/source_present": 0.0,
                "batch/stream_edit": 0.0,
                "grad/context_preclip": 0.0,
                "train/context_update_count": 0.0,
            },
            "Stage A",
        )
    elif stage == "stage_b1_control_bootstrap":
        _require_exact(
            bad,
            last,
            {
                "schedule/p_t2m": 0.1,
                "schedule/p_control": 0.9,
                "schedule/p_edit": 0.0,
                "batch/source_present": 0.0,
                "batch/stream_edit": 0.0,
                "grad/context_preclip": 0.0,
                "update/context/sampled_norm": 0.0,
                "train/context_update_count": 0.0,
            },
            "Stage B1",
        )
        if _metric(last, "batch/mask_fraction", 0.0) <= 0.0:
            bad.append("Stage B1 contract violation: latest batch has an empty control mask")
        if _metric(last, "loss/overall/control_continuous/denominator", 0.0) <= 0.0:
            bad.append("Stage B1 contract violation: continuous control loss has no support")
    elif stage == "stage_b2_joint_adapt":
        _require_probability_range(bad, last, "schedule/p_t2m", 0.1, 0.18, "Stage B2")
        _require_probability_range(bad, last, "schedule/p_control", 0.8, 0.9, "Stage B2")
        _require_probability_range(bad, last, "schedule/p_edit", 0.0, 0.02, "Stage B2")
    elif stage == "stage_c_consolidate":
        _require_probability_range(bad, last, "schedule/p_t2m", 0.18, 0.35, "Stage C")
        _require_probability_range(bad, last, "schedule/p_control", 0.45, 0.8, "Stage C")
        _require_probability_range(bad, last, "schedule/p_edit", 0.02, 0.2, "Stage C")
    elif stage == "stage_c_unified_edit_v2":
        _require_exact(
            bad,
            last,
            {
                "schedule/p_t2m": 0.3,
                "schedule/p_control": 0.4,
                "schedule/p_edit": 0.3,
                "batch/source_present": 0.3,
                "batch/stream_edit": 0.3,
            },
            "Stage C unified Edit v2",
        )
    else:
        bad.append(f"unknown training stage: {stage!r}")

    schedule_sum = sum(
        _metric(last, name)
        for name in ("schedule/p_t2m", "schedule/p_control", "schedule/p_edit")
    )
    if not math.isfinite(schedule_sum) or abs(schedule_sum - 1.0) > 1e-9:
        bad.append(f"task probabilities sum to {schedule_sum}, expected 1.0")

    source_present = _metric(last, "batch/source_present")
    stream_edit = _metric(last, "batch/stream_edit")
    context_grad = _metric(last, "grad/context_preclip")
    context_update = _metric(last, "update/context/sampled_norm")
    if abs(source_present - stream_edit) > 1e-12:
        bad.append(
            "source-present and MOTION_EDIT fractions differ: "
            f"source={source_present}, stream_edit={stream_edit}"
        )
    if source_present <= 1e-12 and (context_grad != 0.0 or context_update != 0.0):
        bad.append(
            "source-absent batch changed context parameters: "
            f"grad={context_grad}, update={context_update}"
        )
    if source_present > 1e-12:
        if stage not in {
            "stage_b2_joint_adapt",
            "stage_c_consolidate",
            "stage_c_unified_edit_v2",
        }:
            bad.append(f"source context appeared in forbidden stage {stage}")
        if context_grad <= 0.0 or context_update <= 0.0:
            bad.append(
                "source-present batch has no context learning signal: "
                f"grad={context_grad}, update={context_update}"
            )

    if step >= 20 and _metric(last, LOSS) > 1.3:
        bad.append(f"loss too high after warm start: {_metric(last, LOSS):.4f}")
    if step >= 20 and _metric(last, GRAD) > 5.0:
        bad.append(f"preclip gradient too high: {_metric(last, GRAD):.4f}")
    if _metric(last, MEMORY) > max_memory_gib:
        bad.append(
            f"max allocated memory {_metric(last, MEMORY):.2f} GiB exceeds "
            f"{max_memory_gib:.2f} GiB"
        )

    stable_rows = [row for row in rows if int(row["step"]) >= 20]
    recent = stable_rows[-3:]
    if len(recent) == 3 and all(_metric(row, THROUGHPUT) < min_throughput for row in recent):
        bad.append(f"throughput below {min_throughput:.1f} samples/s for three windows")
    if recent:
        clip_rate = sum(_metric(row, CLIP, 1.0) for row in recent) / len(recent)
        if step <= 100 and clip_rate > 0.50:
            bad.append(f"clip-active ratio remains high: {clip_rate:.3f}")
        elif step > 100 and clip_rate > 0.25:
            warnings.append(f"clip-active ratio is elevated: {clip_rate:.3f}")

    fk_warmup = _metric(last, "schedule/fk_warmup_factor")
    if step >= 5000 and fk_warmup < 1.0 - 1e-12:
        bad.append(f"FK warmup incomplete at step {step}: {fk_warmup:.6f}")

    trend: dict[str, float] = {}
    if step >= 1000:
        early = _window(rows, 100, 200)
        late = _window(rows, 500, 1000)
        if early and late:
            early_loss = statistics.median(_metric(row, LOSS) for row in early)
            late_loss = statistics.median(_metric(row, LOSS) for row in late)
            relative_drop = 1.0 - late_loss / max(early_loss, 1e-30)
            trend.update(
                {
                    "loss_median_100_200": early_loss,
                    "loss_median_500_1000": late_loss,
                    "loss_relative_drop_500_1000": relative_drop,
                }
            )
            if relative_drop < 0.10:
                bad.append(
                    "loss median from steps 500-1000 did not improve by at least 10% "
                    "over steps 100-200"
                )
        late_clip_rows = _window(rows, max(800, step - 200), step)
        if late_clip_rows:
            late_clip = statistics.mean(_metric(row, CLIP, 1.0) for row in late_clip_rows)
            trend["clip_active_recent_mean"] = late_clip
            if late_clip > 0.25:
                bad.append(f"clip-active ratio exceeds 25% near step {step}: {late_clip:.3f}")
        base_update = _metric(last, "update/base/sampled_norm")
        trend["base_update_sampled_norm"] = base_update
        if not math.isfinite(base_update) or base_update <= 0.0:
            bad.append(f"base parameter update is non-finite or zero: {base_update}")

    status = "bad" if bad else ("warn" if warnings else "ok")
    stage_rows = [row for row in rows if str(row.get("stage", "")) == stage]
    recent_stage_rows = stage_rows[-50:]
    stage_window = {
        "records": len(recent_stage_rows),
        "step_start": int(recent_stage_rows[0]["step"]) if recent_stage_rows else step,
        "step_end": step,
        "loss_mean": (
            statistics.fmean(_metric(row, LOSS) for row in recent_stage_rows)
            if recent_stage_rows
            else _metric(last, LOSS)
        ),
        "throughput_mean": (
            statistics.fmean(_metric(row, THROUGHPUT) for row in recent_stage_rows)
            if recent_stage_rows
            else _metric(last, THROUGHPUT)
        ),
    }
    return {
        "status": status,
        "reasons": bad,
        "warnings": warnings,
        "trend": trend,
        "last": {
            "step": step,
            "stage": stage,
            "loss": _metric(last, LOSS),
            "fk_distance_cm": _metric(last, "loss/overall/fk_distance_cm"),
            "grad_preclip": _metric(last, GRAD),
            "clip_active": _metric(last, CLIP),
            "samples_per_second": _metric(last, THROUGHPUT),
            "max_allocated_gib": _metric(last, MEMORY),
            "metrics_age_minutes": age_minutes,
            "routing": {
                "p_t2m": _metric(last, "schedule/p_t2m"),
                "p_control": _metric(last, "schedule/p_control"),
                "p_edit": _metric(last, "schedule/p_edit"),
                "mask_fraction": _metric(last, "batch/mask_fraction"),
                "source_present": source_present,
                "stream_edit": stream_edit,
                "context_grad_preclip": context_grad,
                "context_update_sampled_norm": context_update,
                "context_update_count": _metric(last, "train/context_update_count"),
            },
            "loss_percentages": {
                label: _metric(last, name) for label, name in LOSS_PERCENT_METRICS.items()
            },
        },
        "stage_window": stage_window,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stale_minutes", type=float, default=15.0)
    parser.add_argument("--min_throughput", type=float, default=140.0)
    parser.add_argument("--max_memory_gib", type=float, default=64.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_path = args.run_dir.expanduser().resolve() / "metrics.jsonl"
    if not metrics_path.is_file():
        payload = {
            "status": "bad",
            "reasons": [f"missing metrics file: {metrics_path}"],
            "last": {},
        }
    else:
        payload = evaluate_health(
            _load_rows(metrics_path),
            stale_minutes=args.stale_minutes,
            metrics_mtime=metrics_path.stat().st_mtime,
            min_throughput=args.min_throughput,
            max_memory_gib=args.max_memory_gib,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] in {"ok", "warn"} else 2)


if __name__ == "__main__":
    main()
