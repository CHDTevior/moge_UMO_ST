from __future__ import annotations

from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import socket
import time

import pytest
import torch

from eval_hy273_kimodo_full_test import (
    PROTOCOL_VERSION,
    aggregate_output,
    ensure_protocol_manifest,
    evaluation_code_identity,
    load_gpu_inventory_identity,
    plan_digest,
    record_launch_provenance,
    record_provenance,
    validate_success_record,
)
from models.raw_motion.asset_integrity import sha256_file

from models.raw_motion.hy273_kimodo_benchmark import (
    KIMODO_CONTROL_SUBTYPES,
    aggregate_case_records,
    build_kimodo_case_plan,
    compile_kimodo_constraint,
    evaluate_kimodo_constraint_case,
    kimodo_motion_quality_metrics,
    shard_kimodo_case_plan,
)
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    KIMODO_EE_GROUPS,
    KIMODO_EE_ROT_GROUPS,
    NUM_JOINTS,
    ROOT_SLICE,
    global_rot_slice_for,
    joint_pos_slice_for,
    load_smplx22_neutral_joints,
    matrix_to_cont6d,
)


def test_balanced_plan_covers_full_test_once_per_text_regime() -> None:
    plan = build_kimodo_case_plan(4042, seed=3407)
    assert len(plan) == 8084
    by_index = defaultdict(list)
    for case in plan:
        by_index[case.dataset_index].append(case)
    assert set(by_index) == set(range(4042))
    for paired in by_index.values():
        assert {case.text_regime for case in paired} == {"withtext", "notext"}
        assert len({case.subtype for case in paired}) == 1
        assert len({case.sample_seed for case in paired}) == 1
    counts = Counter(
        case.subtype for case in plan if case.text_regime == "withtext"
    )
    assert set(counts) == set(KIMODO_CONTROL_SUBTYPES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_pair_sharding_keeps_text_regimes_on_same_worker() -> None:
    plan = build_kimodo_case_plan(41, seed=3407)
    assigned_keys = []
    for shard_id in range(8):
        shard = shard_kimodo_case_plan(plan, shard_id=shard_id, num_shards=8)
        grouped = defaultdict(set)
        for case in shard:
            grouped[(case.dataset_index, case.subtype, case.sample_seed)].add(
                case.text_regime
            )
            assigned_keys.append(case.key)
        assert all(regimes == {"withtext", "notext"} for regimes in grouped.values())
    assert sorted(assigned_keys) == sorted(case.key for case in plan)


def test_evaluation_code_identity_pins_dynamic_backbone_and_launcher() -> None:
    paths = {entry["path"] for entry in evaluation_code_identity()["files"]}
    assert "models/codeflow/dit_blocks.py" in paths
    assert "scripts/launch/eval_hy273_kimodo_full_test_ddp8.sh" in paths
    assert "tools/validate_gpu_inventory.py" in paths


def _write_gpu_inventory(
    path: Path,
    *,
    checked_unix: float,
    pid: int,
    uuid: str = "GPU-test-a",
) -> dict:
    payload = {
        "format": "hy273_gpu_inventory_v1",
        "phase": "before_launch",
        "host": socket.gethostname(),
        "checked_unix": checked_unix,
        "pid": pid,
        "requested_selectors": ["0"],
        "physical_device_count": 1,
        "homogeneous_signature": {
            "name": "NVIDIA A100-SXM4-80GB",
            "memory_total_mib": 81920,
        },
        "idle_thresholds": {
            "max_memory_used_mib": 499,
            "max_utilization_percent": 5,
            "require_no_compute_pids": True,
        },
        "devices": [
            {
                "selected_id": "0",
                "index": 0,
                "uuid": uuid,
                "pci_bus_id": "0000:01:00.0",
                "name": "NVIDIA A100-SXM4-80GB",
                "memory_used_mib": 0,
                "memory_total_mib": 81920,
                "utilization_percent": 0,
                "compute_pids": [],
            }
        ],
        "passed": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def test_resume_record_rejects_protocol_or_weight_provenance_mismatch(
    tmp_path: Path,
) -> None:
    case = build_kimodo_case_plan(1, seed=3407)[0]
    protocol_sha = "protocol-sha"
    inventory_path = tmp_path / "launch_attempts" / "attempt_1" / "before_launch.json"
    _write_gpu_inventory(
        inventory_path,
        checked_unix=time.time() - 3600,
        pid=101,
    )
    launch_identity = load_gpu_inventory_identity(
        inventory_path,
        expected_num_shards=1,
        require_fresh=False,
    )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "weight_source": "ema",
        "caption_policy": "first_full_motion",
        "num_shards": 1,
        "gpu_inventory": launch_identity["stable"],
        "checkpoint_sha256": "checkpoint-sha",
        "plan_sha256": "plan-sha",
        "preflight_manifest": {
            "code_sha256": "code-sha",
            "dataset_assets_sha256": "assets-sha",
            "dataset_order_sha256": "order-sha",
        },
    }
    text = "a person walks" if case.text_regime == "withtext" else ""
    record = {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "weight_source": "ema",
        "case_key": case.key,
        "dataset_index": case.dataset_index,
        "subtype": case.subtype,
        "family": case.family,
        "text_regime": case.text_regime,
        "sample_seed": case.sample_seed,
        "shard_id": 0,
        "caption_is_full_motion": True,
        "text": text,
        "source_caption": "a person walks",
        "metrics": {
            "generated_raw": {"metric": 0.0},
            "ground_truth": {"metric": 0.0},
            "diagnostic_exact_clamp": {"metric": 0.0},
        },
        **record_provenance(protocol, protocol_sha),
        **record_launch_provenance(launch_identity),
    }
    validate_success_record(
        record,
        case=case,
        expected_shard_id=0,
        protocol=protocol,
        protocol_manifest_sha256=protocol_sha,
    )

    forged_protocol = copy.deepcopy(record)
    forged_protocol["protocol_version"] = "forged-old"
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate_success_record(
            forged_protocol,
            case=case,
            expected_shard_id=0,
            protocol=protocol,
            protocol_manifest_sha256=protocol_sha,
        )

    forged_weight = copy.deepcopy(record)
    forged_weight["weight_source"] = "model"
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate_success_record(
            forged_weight,
            case=case,
            expected_shard_id=0,
            protocol=protocol,
            protocol_manifest_sha256=protocol_sha,
        )


def test_gpu_attestations_allow_restart_after_freshness_window_and_aggregation(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "benchmark"
    old_path = output_dir / "launch_attempts" / "attempt_old" / "before_launch.json"
    new_path = output_dir / "launch_attempts" / "attempt_new" / "before_launch.json"
    _write_gpu_inventory(
        old_path,
        checked_unix=time.time() - 3600,
        pid=201,
    )
    _write_gpu_inventory(
        new_path,
        checked_unix=time.time(),
        pid=202,
    )
    old_launch = load_gpu_inventory_identity(
        old_path,
        expected_num_shards=1,
        require_fresh=False,
    )
    new_launch = load_gpu_inventory_identity(
        new_path,
        expected_num_shards=1,
        require_fresh=True,
    )
    assert old_launch["stable"] == new_launch["stable"]
    assert old_launch["attestation"] != new_launch["attestation"]

    plan = build_kimodo_case_plan(
        1,
        seed=3407,
        subtypes=("path_2dpos",),
        text_regimes=("withtext", "notext"),
    )
    keys = [case.key for case in plan]
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": "/fake/step_00400000.pt",
        "checkpoint_sha256": "checkpoint-sha",
        "dataset_size": 1,
        "expected_case_count": len(plan),
        "expected_case_keys": keys,
        "num_steps": 32,
        "cfg_scale_text": 2.0,
        "cfg_scale_control": 2.0,
        "caption_policy": "first_full_motion",
        "assignment": "balanced_partition",
        "cases_per_subtype": 0,
        "seed": 3407,
        "subtypes": ["path_2dpos"],
        "text_regimes": ["withtext", "notext"],
        "num_shards": 1,
        "plan_sha256": plan_digest(keys),
        "weight_source": "ema",
        "gpu_inventory": old_launch["stable"],
        "preflight_manifest": {
            "code_sha256": "code-sha",
            "dataset_assets_sha256": "assets-sha",
            "dataset_order_sha256": "order-sha",
        },
    }
    ensure_protocol_manifest(output_dir, protocol)
    restart_protocol = copy.deepcopy(protocol)
    restart_protocol["gpu_inventory"] = new_launch["stable"]
    ensure_protocol_manifest(output_dir, restart_protocol)
    protocol_sha = sha256_file(output_dir / "protocol_manifest.json")

    records = []
    for case, launch in zip(plan, (old_launch, new_launch)):
        text = "a person walks" if case.text_regime == "withtext" else ""
        records.append(
            {
                "status": "ok",
                "protocol_version": PROTOCOL_VERSION,
                "weight_source": "ema",
                "case_key": case.key,
                "dataset_index": case.dataset_index,
                "subtype": case.subtype,
                "family": case.family,
                "text_regime": case.text_regime,
                "sample_seed": case.sample_seed,
                "shard_id": 0,
                "caption_is_full_motion": True,
                "text": text,
                "source_caption": "a person walks",
                "metrics": {
                    "generated_raw": {"foot_skate_ratio": 0.0},
                    "ground_truth": {"foot_skate_ratio": 0.0},
                    "diagnostic_exact_clamp": {"foot_skate_ratio": 0.0},
                },
                **record_provenance(protocol, protocol_sha),
                **record_launch_provenance(launch),
            }
        )

    shard_path = output_dir / "shards" / "shard_00.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    summary = aggregate_output(output_dir, allow_incomplete=False)
    assert summary["complete"] is True
    assert summary["actual_case_count"] == 2


@pytest.fixture
def source_motion() -> torch.Tensor:
    return torch.arange(64 * DIM_HY273, dtype=torch.float32).reshape(64, DIM_HY273)


@pytest.mark.parametrize("subtype", KIMODO_CONTROL_SUBTYPES)
def test_every_public_subtype_compiles_same_space_constraints(
    source_motion: torch.Tensor,
    subtype: str,
) -> None:
    compiled = compile_kimodo_constraint(source_motion, subtype, seed=17)
    assert compiled.motion_mask.any()
    assert torch.equal(
        compiled.observed_motion[compiled.motion_mask],
        source_motion[compiled.motion_mask],
    )
    assert torch.count_nonzero(compiled.observed_motion[~compiled.motion_mask]) == 0
    assert not compiled.motion_mask[..., CONTACT_SLICE].any()


def test_root_leaf_position_and_heading_masks_are_distinct(
    source_motion: torch.Tensor,
) -> None:
    path_pos = compile_kimodo_constraint(source_motion, "path_2dpos", seed=3)
    path_rot = compile_kimodo_constraint(source_motion, "path_2dposrot", seed=3)
    waypoint_pos = compile_kimodo_constraint(
        source_motion, "waypoint_2dpos", seed=3
    )
    assert path_pos.root_metric_frames.all()
    assert path_pos.motion_mask[:, [ROOT_SLICE.start, ROOT_SLICE.start + 2]].all()
    assert not path_pos.motion_mask[..., HEADING_SLICE].any()
    assert path_rot.motion_mask[..., HEADING_SLICE].all()
    assert waypoint_pos.root_metric_frames.any()
    assert not waypoint_pos.root_metric_frames.all()
    assert not waypoint_pos.motion_mask[..., HEADING_SLICE].any()


def test_inbetweening_is_first_and_last_full_body_keyframe(
    source_motion: torch.Tensor,
) -> None:
    compiled = compile_kimodo_constraint(source_motion, "inbetweening", seed=9)
    expected_frames = torch.zeros(source_motion.shape[0], dtype=torch.bool)
    expected_frames[[0, source_motion.shape[0] - 1]] = True
    assert torch.equal(compiled.fullbody_metric_frames, expected_frames)
    assert compiled.motion_mask[expected_frames, ROOT_SLICE.start : HEADING_SLICE.stop].all()
    assert compiled.motion_mask[expected_frames, JOINT_POS_SLICE].all()
    assert not compiled.motion_mask[..., GLOBAL_ROT_SLICE].any()


@pytest.mark.parametrize(
    ("subtype", "position_groups", "rotation_groups"),
    [
        ("feet_posrot", (0, 1), (0, 1)),
        ("hands_posrot", (2, 3), (2, 3)),
        ("hands_feet_posrot", (0, 1, 2, 3), (0, 1, 2, 3)),
    ],
)
def test_endpoint_leaf_uses_kimodo_root_position_rotation_contract(
    source_motion: torch.Tensor,
    subtype: str,
    position_groups: tuple[int, ...],
    rotation_groups: tuple[int, ...],
) -> None:
    compiled = compile_kimodo_constraint(source_motion, subtype, seed=23)
    frames = compiled.endpoint_position_metric_mask.any(dim=-1)
    position_joints = tuple(
        joint for group in position_groups for joint in KIMODO_EE_GROUPS[group]
    )
    rotation_joints = tuple(
        joint for group in rotation_groups for joint in KIMODO_EE_ROT_GROUPS[group]
    )
    assert frames.any()
    assert compiled.motion_mask[frames, ROOT_SLICE.start : HEADING_SLICE.stop].all()
    assert compiled.motion_mask[frames][:, joint_pos_slice_for(position_joints)].all()
    assert compiled.motion_mask[frames][:, global_rot_slice_for(rotation_joints)].all()
    assert not compiled.root_metric_frames.any()
    assert not compiled.fullbody_metric_frames.any()


def test_public_foot_metrics_match_reference_equations() -> None:
    joints = torch.zeros(3, NUM_JOINTS, 3)
    joints[1, [7, 10, 8, 11], 0] = 0.01
    joints[2, [7, 10, 8, 11], 0] = 0.03
    contacts = torch.ones(3, 4, dtype=torch.bool)
    metrics = kimodo_motion_quality_metrics(joints, contacts, fps=30.0)
    assert metrics["foot_skate_from_height"] == pytest.approx(0.45)
    assert metrics["foot_skate_from_pred_contacts"] == pytest.approx(0.45)
    assert metrics["foot_skate_max_vel"] == pytest.approx(0.60)
    assert metrics["foot_skate_ratio"] == pytest.approx(1.0)
    assert metrics["foot_contact_consistency"] == pytest.approx(0.0)


def _static_identity_motion(length: int = 8) -> torch.Tensor:
    motion = torch.zeros(length, DIM_HY273)
    neutral = load_smplx22_neutral_joints()
    motion[:, JOINT_POS_SLICE] = neutral.reshape(1, -1)
    identity = torch.eye(3).expand(length, NUM_JOINTS, 3, 3)
    motion[:, GLOBAL_ROT_SLICE] = matrix_to_cont6d(identity).reshape(length, -1)
    motion[:, HEADING_SLICE.start] = 1.0
    motion[:, CONTACT_SLICE] = 1.0
    return motion


@pytest.mark.parametrize(
    "subtype",
    [
        "path_2dposrot",
        "inbetweening",
        "hands_feet_posrot",
        "root_ee_hands_feet_posrot_fullbody",
    ],
)
def test_ground_truth_pass_is_exact_except_root_smoothing_semantics(
    subtype: str,
) -> None:
    motion = _static_identity_motion()
    compiled = compile_kimodo_constraint(motion, subtype, seed=41)
    metrics = evaluate_kimodo_constraint_case(motion, motion, compiled)
    if compiled.fullbody_metric_frames.any():
        assert metrics["constraint_fullbody_keyframe"] == pytest.approx(0.0)
    if compiled.endpoint_position_metric_mask.any():
        assert metrics["constraint_end_effector"] == pytest.approx(0.0)
        assert metrics["constraint_end_effector_rotation_deg"] == pytest.approx(
            0.0, abs=1e-4
        )
    if compiled.root_metric_frames.any():
        assert metrics["constraint_root2d_err"] == pytest.approx(0.0)
        assert metrics["constraint_root2d_acc"] == pytest.approx(1.0)


def test_aggregation_uses_per_motion_mean_and_root_p95() -> None:
    def record(key: str, root_error: float) -> dict:
        passes = {
            "constraint_root2d_err": root_error,
            "constraint_root2d_acc": 1.0,
        }
        return {
            "status": "ok",
            "case_key": key,
            "text_regime": "withtext",
            "subtype": "path_2dpos",
            "family": "root",
            "metrics": {
                "generated_raw": passes,
                "ground_truth": passes,
                "diagnostic_exact_clamp": passes,
            },
        }

    summary = aggregate_case_records([record("a", 1.0), record("b", 3.0)])
    subtype = next(
        row
        for row in summary["rows"]
        if row["level"] == "subtype" and row["name"] == "path_2dpos"
    )
    assert subtype["generated_raw"]["constraint_root2d_err"] == pytest.approx(2.0)
    assert subtype["generated_raw"]["constraint_root2d_err_p95"] == pytest.approx(2.9)
    assert subtype["generated_raw"]["constraint_root2d_err__count"] == 2
