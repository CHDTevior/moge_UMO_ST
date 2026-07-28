"""Health monitor for HY273 raw-flow training logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(r"\[train\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+(?P<body>.*)")


def parse_metrics(line: str) -> dict[str, float] | None:
    match = TRAIN_RE.search(line)
    if not match:
        return None
    out: dict[str, float] = {
        "epoch": float(match.group("epoch")),
        "step": float(match.group("step")),
    }
    for part in match.group("body").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def read_train_metrics(log_path: Path) -> list[dict[str, float]]:
    if not log_path.is_file():
        return []
    metrics: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parsed = parse_metrics(line)
            if parsed is not None:
                metrics.append(parsed)
    return metrics


def load_weights(run_dir: Path) -> dict[str, float]:
    cfg_path = run_dir / "config_resolved.json"
    defaults = {
        "flow": 1.0,
        "contact": 0.1,
        "control_cont": 0.25,
        "control_contact": 0.05,
        "clean_cont": 0.0,
        "clean_root_vel": 0.0,
        "clean_joint_vel": 0.0,
        "foot_lock": 0.0,
    }
    if not cfg_path.is_file():
        return defaults
    cfg = json.loads(cfg_path.read_text())
    return {
        "flow": float(cfg.get("flow_loss_weight", defaults["flow"])),
        "contact": float(cfg.get("contact_loss_weight", defaults["contact"])),
        "control_cont": float(cfg.get("control_cont_loss_weight", defaults["control_cont"])),
        "control_contact": float(cfg.get("control_contact_loss_weight", defaults["control_contact"])),
        "clean_cont": float(cfg.get("clean_cont_loss_weight", defaults["clean_cont"])),
        "clean_root_vel": float(
            cfg.get("clean_root_vel_loss_weight", defaults["clean_root_vel"])
        ),
        "clean_joint_vel": float(
            cfg.get("clean_joint_vel_loss_weight", defaults["clean_joint_vel"])
        ),
        "foot_lock": float(cfg.get("foot_lock_loss_weight", defaults["foot_lock"])),
    }


def gpu_status() -> list[dict[str, int]]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_used_mb": int(parts[1]),
                "memory_total_mb": int(parts[2]),
                "util_gpu_pct": int(parts[3]),
            }
        )
    return rows


def checkpoint_status(run_dir: Path) -> dict[str, Any]:
    model_dir = run_dir / "model"
    if not model_dir.is_dir():
        return {"latest": None, "count": 0}
    ckpts = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return {"latest": None, "count": 0}
    latest = ckpts[-1]
    return {
        "latest": str(latest),
        "latest_mtime": latest.stat().st_mtime,
        "latest_size_gb": latest.stat().st_size / (1024**3),
        "count": len(ckpts),
    }


def weighted_percentages(metric: dict[str, float], weights: dict[str, float]) -> dict[str, float]:
    contrib = {
        "flow": weights["flow"] * metric.get("flow", 0.0),
        "contact": weights["contact"] * metric.get("contact", 0.0),
        "control_cont": weights["control_cont"] * metric.get("control_cont", 0.0),
        "control_contact": weights["control_contact"] * metric.get("control_contact", 0.0),
        "clean_cont": weights["clean_cont"] * metric.get("clean_cont", 0.0),
        "clean_root_vel": weights["clean_root_vel"] * metric.get("clean_root_vel", 0.0),
        "clean_joint_vel": weights["clean_joint_vel"] * metric.get("clean_joint_vel", 0.0),
        "foot_lock": weights["foot_lock"] * metric.get("foot_lock", 0.0),
        "fk_consistency": metric.get("fk_consistency_weighted", 0.0),
    }
    total = sum(contrib.values())
    if total <= 0:
        return {key: 0.0 for key in contrib}
    return {key: value / total * 100.0 for key, value in contrib.items()}


def expected_gpu_indices(run_dir: Path, gpu: list[dict[str, int]]) -> list[int]:
    trace_path = run_dir / "trace_contract.json"
    if trace_path.is_file():
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            visible = str(trace.get("cuda_visible_devices", ""))
            indices = [int(value.strip()) for value in visible.split(",") if value.strip()]
            if indices:
                return indices
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return [row["index"] for row in gpu]


def assess(
    metrics: list[dict[str, float]],
    gpu: list[dict[str, int]],
    expected_gpus: list[int] | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not metrics:
        return "bad", ["no train metrics found"]
    last = metrics[-1]
    for key, value in last.items():
        if isinstance(value, float) and not math.isfinite(value):
            reasons.append(f"non-finite {key}")
    if last.get("loss", 0.0) > 10.0:
        reasons.append(f"loss too large: {last.get('loss')}")
    if len(metrics) >= 10:
        recent = [m["loss"] for m in metrics[-5:]]
        previous = [m["loss"] for m in metrics[-10:-5]]
        if sum(recent) / len(recent) > 2.0 * max(sum(previous) / len(previous), 1e-8):
            reasons.append("recent loss >2x previous window")
    if gpu:
        expected = expected_gpus if expected_gpus is not None else [row["index"] for row in gpu]
        by_index = {row["index"]: row for row in gpu}
        missing = [index for index in expected if index not in by_index]
        idle = [
            index
            for index in expected
            if index in by_index and by_index[index]["memory_used_mb"] <= 1000
        ]
        if missing:
            reasons.append(f"expected GPUs not reported: {missing}")
        if idle:
            reasons.append(f"expected GPUs have <=1GB allocated: {idle}")
    return ("bad" if reasons else "ok"), reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--health-log", default="")
    args = parser.parse_args()

    log_path = Path(args.log)
    run_dir = Path(args.run_dir)
    metrics = read_train_metrics(log_path)
    weights = load_weights(run_dir)
    gpu = gpu_status()
    expected_gpus = expected_gpu_indices(run_dir, gpu)
    ckpt = checkpoint_status(run_dir)
    status, reasons = assess(metrics, gpu, expected_gpus)
    last = metrics[-1] if metrics else {}
    perc = weighted_percentages(last, weights) if last else {}
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "reasons": reasons,
        "last": last,
        "weighted_percent": perc,
        "gpu": gpu,
        "expected_gpu_indices": expected_gpus,
        "checkpoint": ckpt,
        "log": str(log_path),
        "run_dir": str(run_dir),
    }
    text = json.dumps(record, sort_keys=True)
    print(text, flush=True)
    if args.health_log:
        health = Path(args.health_log)
        health.parent.mkdir(parents=True, exist_ok=True)
        with health.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
