from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.compare_hy273_t2m_edit_guardrails import compare_edit, compare_t2m


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _t2m_row(key: str, seed: int, offset: float) -> dict:
    return {
        "case_key": key,
        "sample_seed": seed,
        "length": 30,
        "native_gt_mm_dist": 1.0,
        "native_gt_r_top1": 1.0,
        "native_gt_r_top2": 1.0,
        "native_gt_r_top3": 1.0,
        "oracle_gt_mm_dist": 1.1,
        "oracle_gt_r_top1": 1.0,
        "oracle_gt_r_top2": 1.0,
        "oracle_gt_r_top3": 1.0,
        "r_top1": offset,
        "r_top2": offset + 0.1,
        "r_top3": offset + 0.2,
        "mm_dist": 3.0 - offset,
        "quality": {
            "fk_jerk_mps3": 5.0 - offset,
            "foot_skate_ratio": 0.4 - offset / 10.0,
        },
    }


def _write_t2m_summary(root: Path, rows: list[dict]) -> Path:
    cases = root / "case_metrics.jsonl"
    _write_jsonl(cases, rows)
    summary = {
        "case_metrics_path": str(cases),
        "protocol": {
            "protocol_version": "protocol",
            "bridge_protocol": "bridge",
            "reference_domain": "reference",
            "case_count": len(rows),
            "case_plan_sha256": "plan",
            "checkpoint_kind": "unified_actor",
            "official_benchmark_claim": False,
            "sampling": {"ode_steps": 32, "text_cfg_scale": 2.0},
            "weight_source": "ema",
            "num_shards": 1,
        },
        "metrics": {
            "fid": 2.0,
            "r_precision": {
                "top1": float(np.mean([row["r_top1"] for row in rows])),
                "top2": float(np.mean([row["r_top2"] for row in rows])),
                "top3": float(np.mean([row["r_top3"] for row in rows])),
            },
            "mm_dist": float(np.mean([row["mm_dist"] for row in rows])),
            "quality": {
                "fk_jerk_mps3": float(
                    np.mean([row["quality"]["fk_jerk_mps3"] for row in rows])
                ),
                "foot_skate_ratio": float(
                    np.mean([row["quality"]["foot_skate_ratio"] for row in rows])
                ),
            },
        },
    }
    path = root / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_compare_t2m_checks_identity_and_paired_metrics(tmp_path: Path) -> None:
    baseline = [_t2m_row("a", 1, 0.2), _t2m_row("b", 2, 0.4)]
    candidate = [_t2m_row("a", 1, 0.3), _t2m_row("b", 2, 0.5)]
    left = _write_t2m_summary(tmp_path / "left", baseline)
    right = _write_t2m_summary(tmp_path / "right", candidate)

    result = compare_t2m(left, right, bootstrap_resamples=100, seed=7)

    assert result["identity"]["matched"] is True
    assert result["metrics"]["r_precision_top1"]["candidate_minus_baseline"] == pytest.approx(0.1)
    assert result["metrics"]["foot_skate_ratio"]["candidate_minus_baseline"] == pytest.approx(-0.01)


def _edit_protocol(case_count: int) -> dict:
    return {
        "format": "edit",
        "protocol": "motionfix",
        "pair_count": 2,
        "case_count": case_count,
        "systems": [
            "source_copy",
            "source_instruction_model",
            "source_only_model",
            "source_shuffled_instruction_model",
            "shuffled_source_instruction_model",
            "relative_instruction_only_ood",
        ],
        "weight_source": "ema",
        "num_steps": 32,
        "source_cfg_scale": 2.0,
        "edit_cfg_scale": 2.0,
        "generate_text_cfg_scale": 2.0,
        "seed": 3407,
        "target_length_protocol": "target",
        "metric_protocol": "metric",
        "same_noise_across_systems": True,
        "plan_sha256": "plan",
        "source_copy_unequal_protocol": "slerp",
        "cfg_apply_contacts": False,
        "contact_feedback": "blend",
        "contact_init": "random",
        "generate_cfg_apply_contacts": True,
        "editing_contact_protocol_id": "edit-contact",
        "generate_control_contact_protocol_id": "generate-contact",
        "direct_k273_interpolation": False,
        "counterfactual_manifest": {
            "sha256": "counterfactual",
            "summary": {"rows_sha256": "rows"},
        },
        "train_manifest": {"sha256": "train"},
        "num_shards": 1,
    }


def _edit_row(pair: str, system: str, checkpoint_offset: float) -> dict:
    source_copy = system == "source_copy"
    correct = system == "source_instruction_model"
    system_offset = {
        "source_copy": -0.05,
        "source_instruction_model": 0.0,
        "source_only_model": 0.05,
        "source_shuffled_instruction_model": 0.10,
        "shuffled_source_instruction_model": 0.15,
        "relative_instruction_only_ood": 0.20,
    }[system]
    effective_checkpoint = 0.0 if source_copy else checkpoint_offset
    target_error = 0.30 + system_offset - effective_checkpoint
    identity_metrics = {
        "frames": 30,
        "source_target_position_delta_m": 0.25,
        "source_target_rotation_delta_deg": 25.0,
        "target_jerk_mps3": 20.0,
        "changed_joint_entries": 10,
        "ambiguous_joint_entries": 2,
        "unchanged_joint_entries": 4,
        "changed_position_threshold_m": 0.02,
        "changed_rotation_threshold_deg": 5.0,
        "changed_temporal_dilation_frames": 2,
        "unchanged_position_threshold_m": 0.01,
        "unchanged_rotation_threshold_deg": 3.0,
    }
    return {
        "status": "ok",
        "case_key": f"{pair}:{system}",
        "case_uid": f"motionfix:{pair}",
        "pair_id": pair,
        "system": system,
        "instruction": "move faster",
        "model_instruction": "move faster" if not source_copy else "",
        "sample_seed": int(pair),
        "control_subtype": None,
        "source_frames": 30,
        "target_frames": 30,
        "source_k273_path": f"/{pair}_source.npy",
        "target_k273_path": f"/{pair}_target.npy",
        "length_relation": "equal",
        "target_length_protocol": "target",
        "frame_policy_id": "frames",
        "regional_metric_protocol": "regions",
        "source_applied_yaw_delta": 0.1,
        "target_applied_yaw_delta": 0.2,
        "model_source_applied_yaw_delta": None if source_copy else 0.1,
        "aligned_source_applied_yaw_delta": 0.1,
        "output_gauge_phi": 0.3,
        "condition_provenance": {"system": system},
        "sampling_protocol": {"system": system},
        "seen_strata": {"seen": False},
        "assets": {"source": pair, "target": pair},
        "aligned_reference_source": {"tensor_sha256": f"source-{pair}"},
        "metrics": {
            **identity_metrics,
            "global_joint_target_error_m": target_error,
            "global_rotation_target_error_deg": target_error * 100.0,
            "changed_region_target_error_m": target_error,
            "changed_region_target_rotation_error_deg": target_error * 100.0,
            "prediction_jerk_mps3": 10.0 + target_error,
            "foot_skate_ratio": target_error,
            "global_joint_source_error_m": 0.1 if correct else 0.2,
        },
    }


def _write_edit_summary(root: Path, checkpoint_offset: float) -> Path:
    systems = _edit_protocol(0)["systems"]
    rows = [
        _edit_row(pair, system, checkpoint_offset)
        for pair in ("1", "2")
        for system in systems
    ]
    _write_jsonl(root / "shards" / "shard_00.jsonl", rows)
    path = root / "summary.json"
    path.write_text(
        json.dumps({"protocol": _edit_protocol(len(rows))}), encoding="utf-8"
    )
    return path


def test_compare_edit_checks_protocol_identity_and_condition_advantage(
    tmp_path: Path,
) -> None:
    left = _write_edit_summary(tmp_path / "left", checkpoint_offset=0.0)
    right = _write_edit_summary(tmp_path / "right", checkpoint_offset=0.05)

    result = compare_edit(left, right, bootstrap_resamples=100, seed=7)

    assert result["identity"]["matched"] is True
    assert result["metrics"]["target_joint_error"]["candidate_minus_baseline"] == pytest.approx(-5.0)
    assert result["identity"]["source_copy_checkpoint_invariance_verified"] is True


def test_compare_edit_rejects_source_target_identity_change(tmp_path: Path) -> None:
    left = _write_edit_summary(tmp_path / "left", checkpoint_offset=0.0)
    right = _write_edit_summary(tmp_path / "right", checkpoint_offset=0.05)
    shard = right.parent / "shards" / "shard_00.jsonl"
    rows = _load_rows(shard)
    rows[0]["target_k273_path"] = "/different.npy"
    _write_jsonl(shard, rows)

    with pytest.raises(ValueError, match="target_k273_path"):
        compare_edit(left, right, bootstrap_resamples=10, seed=7)


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
