from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tools.gate_hy273_r12_step250k import (
    fixed16_floor_rows,
    paired_heading_path_diagnostic,
    root_superiority_rows,
    scientific_nonregression_rows,
    skate_improvement_rows,
    validate_resume,
)
from train_hy273_multitask import (
    CHECKPOINT_FORMAT,
    R12_TRAIN_CONTRACT,
    r12_origin_parent_identity,
)


def _record(key: str, subtype: str, regime: str, value: float) -> dict:
    return {
        "case_key": key,
        "dataset_index": int(key.rsplit("_", 1)[-1]),
        "motion_id": key,
        "length": 8,
        "subtype": subtype,
        "text_regime": regime,
        "sample_seed": 7,
        "constraint_payload_sha256": "a",
        "initial_noise_sha256": "b",
        "target_tensor_sha256": "c",
        "metrics": {
            "generated_raw": {
                "constraint_root2d_err": value,
                "constraint_root2d_acc": 1.0 - value,
                "foot_skate_ratio": value,
                "foot_skate_from_height": value,
                "foot_skate_from_pred_contacts": value,
                "foot_skate_max_vel": value,
                "foot_contact_consistency": 1.0 - value,
            }
        },
    }


def _matched_records(baseline_value: float, candidate_value: float):
    baseline = {}
    candidate = {}
    index = 0
    for subtype in ("path_2dpos", "waypoint_2dpos"):
        for regime in ("notext", "withtext"):
            for _ in range(4):
                key = f"case_{index}"
                baseline[key] = _record(key, subtype, regime, baseline_value)
                candidate[key] = _record(key, subtype, regime, candidate_value)
                index += 1
    return baseline, candidate


def test_r12_root_superiority_requires_strict_one_sided_gain():
    baseline, better = _matched_records(0.4, 0.2)
    rows = root_superiority_rows(
        baseline, better, resamples=200, confidence=0.95, seed=3
    )
    assert len(rows) == 8
    assert all(row["passed"] for row in rows)

    _, equal = _matched_records(0.4, 0.4)
    rows = root_superiority_rows(
        baseline, equal, resamples=200, confidence=0.95, seed=3
    )
    assert not any(row["passed"] for row in rows)


def test_r12_skate_gate_requires_all_four_point_estimates_to_improve():
    old, new = _matched_records(0.4, 0.3)
    assert all(row["passed"] for row in skate_improvement_rows(old, new))
    first = next(iter(new.values()))
    first["metrics"]["generated_raw"]["foot_skate_max_vel"] = 9.0
    rows = skate_improvement_rows(old, new)
    assert not next(row for row in rows if row["metric"] == "foot_skate_max_vel")["passed"]


def test_heading_path_diagnostic_uses_kimodo_forward_convention(tmp_path: Path):
    motion = np.zeros((8, 273), dtype=np.float32)
    motion[:, 2] = np.arange(8, dtype=np.float32) * 0.1
    motion[:, 3] = 1.0  # [cos=1,sin=0] means +Z forward.
    records = {}
    for name in ("baseline", "candidate"):
        output = tmp_path / f"{name}.npz"
        np.savez(output, generated_raw=motion)
        record = _record("case_0", "path_2dpos", "withtext", 0.1)
        record["output_path"] = str(output)
        records[name] = {"case_0": record}
    diagnostic = paired_heading_path_diagnostic(records)
    assert diagnostic["common_moving_frames"] == 7
    assert diagnostic["per_model"]["baseline"]["mean_cosine_on_common_frames"] == 1.0
    assert (
        diagnostic["per_model"]["candidate"]["mean_abs_angle_deg_on_common_frames"]
        == 0.0
    )
    assert diagnostic["paired_case_delta_vs_baseline"]["candidate"]["mean_cosine_delta"] == 0.0


def test_scientific_nonregression_uses_matched_case_bootstrap_without_code_sha():
    baseline, better = _matched_records(0.4, 0.3)
    rows = scientific_nonregression_rows(
        baseline,
        better,
        resamples=200,
        confidence=0.95,
        relative_tolerance=0.05,
        seed=3,
    )
    assert rows
    assert all(row["passed"] for row in rows)

    _, worse = _matched_records(0.4, 0.8)
    rows = scientific_nonregression_rows(
        baseline,
        worse,
        resamples=200,
        confidence=0.95,
        relative_tolerance=0.05,
        seed=3,
    )
    assert any(not row["passed"] for row in rows)


def test_fixed16_floors_accept_small_changes_and_reject_collapse():
    aggregate = {
        "fk_jerk_mps3": 50.0,
        "position_channel_jerk_mps3": 50.0,
        "foot_skate_from_height": 0.2,
        "foot_skate_from_pred_contacts": 0.1,
        "foot_skate_max_vel": 0.3,
        "foot_skate_ratio": 0.2,
        "foot_contact_consistency": 0.9,
    }
    rows = fixed16_floor_rows({"aggregate": aggregate}, {"aggregate": dict(aggregate)})
    assert all(row["passed"] for row in rows)
    collapsed = dict(aggregate)
    collapsed["foot_skate_max_vel"] = 2.0
    rows = fixed16_floor_rows({"aggregate": aggregate}, {"aggregate": collapsed})
    assert not next(row for row in rows if row["metric"] == "foot_skate_max_vel")["passed"]


def test_b2_resume_gate_binds_checkpoint_uuid_sha_and_origin(tmp_path: Path):
    origin = r12_origin_parent_identity(tmp_path / "r11.pt")
    checkpoint_path = tmp_path / "r12.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "train_contract": R12_TRAIN_CONTRACT,
            "next_global_step": 250_000,
            "run_uuid": "r12-uuid",
            "runtime_identity": {"origin_parent": origin},
        },
        checkpoint_path,
    )
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        __import__("json").dumps(
            {
                "format": "hy273_r12_step250k_gate_v1",
                "status": "passed",
                "checkpoint": {
                    "sha256": "f" * 64,
                    "run_uuid": "r12-uuid",
                    "origin_parent": origin,
                },
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        gate_artifact=str(gate_path),
        checkpoint=str(checkpoint_path),
        checkpoint_sha256="f" * 64,
    )
    assert validate_resume(args) == 0
