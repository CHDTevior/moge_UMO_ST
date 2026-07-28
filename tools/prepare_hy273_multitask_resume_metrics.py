#!/usr/bin/env python
"""Archive metrics newer than a resume checkpoint before restarting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


FORMAT = "hy273_multitask_metrics_resume_segment_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_metric_lines(path: Path) -> tuple[list[bytes], list[int]]:
    lines = path.read_bytes().splitlines(keepends=True)
    steps: list[int] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"Blank metrics row at line {index}")
        try:
            record = json.loads(line)
            step = int(record["step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid metrics row at line {index}") from error
        if steps and step <= steps[-1]:
            raise ValueError(
                f"metrics.jsonl is already non-monotonic at line {index}: "
                f"{steps[-1]} -> {step}"
            )
        steps.append(step)
    return lines, steps


def prepare_resume_metrics(
    run_dir: str | Path,
    checkpoint: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    metrics_path = run_dir / "metrics.jsonl"
    identity_path = run_dir / "run_identity.json"
    if not metrics_path.is_file() or not identity_path.is_file():
        raise FileNotFoundError("run_dir must contain metrics.jsonl and run_identity.json")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint}")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    checkpoint_step = int(payload.get("next_global_step", -1))
    if checkpoint_step < 0:
        raise ValueError("Checkpoint has no valid next_global_step")
    if payload.get("run_name") != run_dir.name:
        raise ValueError("Checkpoint run_name does not match run_dir")
    if payload.get("run_uuid") != identity.get("run_uuid"):
        raise ValueError("Checkpoint run_uuid does not match run_identity.json")

    metrics_sha_before = sha256_file(metrics_path)
    lines, steps = _load_metric_lines(metrics_path)
    split = next((index for index, step in enumerate(steps) if step > checkpoint_step), len(steps))
    kept_lines = lines[:split]
    archived_lines = lines[split:]
    archived_steps = steps[split:]
    result: dict[str, Any] = {
        "format": FORMAT,
        "status": "needs_segmentation" if archived_lines else "clean",
        "applied": False,
        "run_name": run_dir.name,
        "run_uuid": identity["run_uuid"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_next_global_step": checkpoint_step,
        "metrics_sha256_before": metrics_sha_before,
        "metrics_rows_before": len(lines),
        "metrics_tail_step_before": steps[-1] if steps else None,
        "kept_rows": len(kept_lines),
        "kept_tail_step": steps[split - 1] if split else None,
        "archived_rows": len(archived_lines),
        "archived_first_step": archived_steps[0] if archived_steps else None,
        "archived_last_step": archived_steps[-1] if archived_steps else None,
    }
    if not apply or not archived_lines:
        return result

    archived_payload = b"".join(archived_lines)
    archived_sha = hashlib.sha256(archived_payload).hexdigest()
    segment_dir = run_dir / "metrics_segments"
    segment_name = (
        f"after_ckpt_{checkpoint_step:08d}_"
        f"steps_{archived_steps[0]:08d}_{archived_steps[-1]:08d}_"
        f"{archived_sha[:12]}.jsonl"
    )
    segment_path = segment_dir / segment_name
    if segment_path.exists() and sha256_file(segment_path) != archived_sha:
        raise RuntimeError(f"Existing segment content differs: {segment_path}")
    if not segment_path.exists():
        _atomic_write(segment_path, archived_payload)

    # Recheck immediately before replacing the live metrics file. Apply only
    # after all training ranks have stopped; this catches accidental live use.
    if sha256_file(metrics_path) != metrics_sha_before:
        raise RuntimeError("metrics.jsonl changed during preparation; stop training first")
    _atomic_write(metrics_path, b"".join(kept_lines))
    result.update(
        {
            "status": "segmented",
            "applied": True,
            "segment_path": str(segment_path),
            "segment_sha256": archived_sha,
            "metrics_sha256_after": sha256_file(metrics_path),
        }
    )
    manifest_path = segment_path.with_suffix(".json")
    _atomic_write(
        manifest_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    result["segment_manifest"] = str(manifest_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Archive and remove rows newer than the checkpoint; omit for dry-run.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_resume_metrics(args.run_dir, args.checkpoint, apply=args.apply),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
