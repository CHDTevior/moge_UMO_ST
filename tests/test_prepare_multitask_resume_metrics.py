from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.prepare_hy273_multitask_resume_metrics import prepare_resume_metrics


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_identity.json").write_text(
        json.dumps({"run_name": "run", "run_uuid": "uuid-1"}), encoding="utf-8"
    )
    records = [{"step": step, "metrics": {"loss": step / 100.0}} for step in (20, 40, 60)]
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
    )
    checkpoint = run_dir / "model.pt"
    torch.save(
        {"run_name": "run", "run_uuid": "uuid-1", "next_global_step": 40},
        checkpoint,
    )
    return run_dir, checkpoint


def test_resume_metrics_dry_run_and_apply(tmp_path: Path) -> None:
    run_dir, checkpoint = _fixture(tmp_path)
    original = (run_dir / "metrics.jsonl").read_bytes()

    dry_run = prepare_resume_metrics(run_dir, checkpoint)
    assert dry_run["status"] == "needs_segmentation"
    assert dry_run["archived_first_step"] == 60
    assert (run_dir / "metrics.jsonl").read_bytes() == original

    applied = prepare_resume_metrics(run_dir, checkpoint, apply=True)
    assert applied["status"] == "segmented"
    assert applied["archived_rows"] == 1
    remaining = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in remaining] == [20, 40]
    segment = Path(applied["segment_path"])
    assert [json.loads(line)["step"] for line in segment.read_text().splitlines()] == [60]
    assert Path(applied["segment_manifest"]).is_file()

    clean = prepare_resume_metrics(run_dir, checkpoint)
    assert clean["status"] == "clean"
    assert clean["archived_rows"] == 0
