from __future__ import annotations

import itertools
from argparse import Namespace

import pytest
import torch

from models.raw_motion.hy273_constraints import (
    KimodoControlCurriculum,
    build_kimodo_control_curriculum_batch,
    build_synthetic_control_batch,
)
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    KIMODO_EE_GROUPS,
    KIMODO_EE_ROT_GROUPS,
    ROOT_SLICE,
    joint_pos_slice_for,
    global_rot_slice_for,
    matrix_to_cont6d,
    yaw_matrix,
)
from sample_hy273_raw import resolve_endpoint_protocol, resolve_sampling_control_modes
from eval_hy273_raw_control import (
    MetricAccumulator,
    foot_skate_from_joints_per_sample,
    masked_contact_metrics,
    masked_l2_per_sample,
    masked_rotation_error_deg,
    phase2_distribution_conditioned_on_control,
)
from train_hy273_raw_flow import (
    apply_deterministic_text_dropout,
    build_arg_parser,
    create_model,
    explicit_cli_destinations,
    load_yaml,
    merge_config,
    validate_ema_contract,
    validate_execution_contract,
    validate_resume_contract,
    validate_run_name,
)


def _selected_joint_ids(mask_at_frame: torch.Tensor) -> tuple[int, ...]:
    position_mask = mask_at_frame[JOINT_POS_SLICE].reshape(22, 3)
    assert torch.equal(position_mask, position_mask[:, :1].expand_as(position_mask))
    return tuple(torch.where(position_mask[:, 0])[0].tolist())


def _valid_kimodo_subsets() -> set[tuple[int, ...]]:
    expected_groups = ((7, 10), (8, 11), (20,), (21,))
    assert KIMODO_EE_GROUPS == expected_groups
    subsets: set[tuple[int, ...]] = set()
    for enabled in itertools.product((False, True), repeat=len(expected_groups)):
        if not any(enabled):
            continue
        subsets.add(
            tuple(sorted(
                joint_id
                for use_group, group in zip(enabled, expected_groups)
                if use_group
                for joint_id in group
            ))
        )
    return subsets


def test_kimodo_endpoint_all_masks_six_position_joints_and_root_reference() -> None:
    motion = torch.arange(12 * DIM_HY273, dtype=torch.float32).reshape(1, 12, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.tensor([12]),
        modes=("endpoints",),
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="all",
        min_keyframes=1,
        max_keyframes=1,
        generator=torch.Generator().manual_seed(7),
    )

    frames = torch.where(result.motion_mask[0].any(dim=-1))[0]
    assert frames.numel() == 1
    frame = int(frames.item())
    expected = torch.zeros(DIM_HY273, dtype=torch.bool)
    expected[ROOT_SLICE.start : HEADING_SLICE.stop] = True
    expected[joint_pos_slice_for(joint_id for group in KIMODO_EE_GROUPS for joint_id in group)] = True
    assert torch.equal(result.motion_mask[0, frame], expected)
    assert torch.equal(result.observed_motion[result.motion_mask], motion[result.motion_mask])
    assert torch.count_nonzero(result.observed_motion[~result.motion_mask]) == 0


def test_kimodo_random_endpoint_control_samples_only_nonempty_logical_groups() -> None:
    batch_size = 64
    motion = torch.randn(batch_size, 10, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.full((batch_size,), 10),
        modes=("endpoints",),
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="random_nonempty",
        min_keyframes=1,
        max_keyframes=1,
        generator=torch.Generator().manual_seed(11),
    )

    valid_subsets = _valid_kimodo_subsets()
    sampled: set[tuple[int, ...]] = set()
    for batch_idx in range(batch_size):
        frame = int(torch.where(result.motion_mask[batch_idx].any(dim=-1))[0].item())
        selected = _selected_joint_ids(result.motion_mask[batch_idx, frame])
        assert selected in valid_subsets
        sampled.add(selected)
        assert result.motion_mask[batch_idx, frame, ROOT_SLICE.start : HEADING_SLICE.stop].all()
    assert len(sampled) > 1


def test_kimodo_complete_endpoint_masks_positions_rotations_and_root_reference() -> None:
    motion = torch.randn(1, 12, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.tensor([12]),
        modes=("endpoints",),
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="all",
        min_keyframes=1,
        max_keyframes=1,
        include_endpoint_rotations=True,
        generator=torch.Generator().manual_seed(17),
    )
    frame = int(torch.where(result.motion_mask[0].any(dim=-1))[0].item())
    endpoint_joints = tuple(
        joint_id for group in KIMODO_EE_GROUPS for joint_id in group
    )
    endpoint_rotation_joints = tuple(
        joint_id for group in KIMODO_EE_ROT_GROUPS for joint_id in group
    )
    expected = torch.zeros(DIM_HY273, dtype=torch.bool)
    expected[ROOT_SLICE.start : HEADING_SLICE.stop] = True
    expected[joint_pos_slice_for(endpoint_joints)] = True
    expected[global_rot_slice_for(endpoint_rotation_joints)] = True
    assert torch.equal(result.motion_mask[0, frame], expected)
    assert torch.equal(result.observed_motion[result.motion_mask], motion[result.motion_mask])


def test_kimodo_complete_contact_pattern_masks_all_four_binary_contacts_only() -> None:
    motion = torch.randn(8, 20, DIM_HY273)
    motion[..., CONTACT_SLICE] = torch.randint(0, 2, (8, 20, 4)).float()
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.full((8,), 20),
        modes=("contact",),
        min_keyframes=1,
        max_keyframes=4,
        include_contact_pattern=True,
        generator=torch.Generator().manual_seed(23),
    )
    assert all(name == "contact" for name in result.mode_ids)
    assert result.motion_mask[..., CONTACT_SLICE].any()
    assert not result.motion_mask[..., : CONTACT_SLICE.start].any()
    contact_mask = result.motion_mask[..., CONTACT_SLICE]
    assert torch.equal(contact_mask, contact_mask[..., :1].expand_as(contact_mask))
    assert torch.equal(result.observed_motion[result.motion_mask], motion[result.motion_mask])


def test_kimodo_complete_curriculum_exercises_contact_and_endpoint_rotations() -> None:
    motion = torch.randn(2048, 32, DIM_HY273)
    motion[..., CONTACT_SLICE] = torch.randint(0, 2, (2048, 32, 4)).float()
    result = build_kimodo_control_curriculum_batch(
        motion,
        lengths=torch.full((2048,), 32),
        progress=1.0,
        config=KimodoControlCurriculum(
            none_prob=0.0,
            mixed_prob=0.25,
            endpoint_preset="kimodo_ee",
            include_endpoint_rotations=True,
            include_contact_pattern=True,
        ),
        generator=torch.Generator().manual_seed(31),
    )
    contact_samples = [
        idx for idx, name in enumerate(result.mode_ids) if "contact" in name
    ]
    endpoint_samples = [
        idx for idx, name in enumerate(result.mode_ids) if "endpoints" in name
    ]
    assert contact_samples and endpoint_samples
    assert result.motion_mask[contact_samples, :, CONTACT_SLICE].any()
    endpoint_rotation_mask = result.motion_mask[
        endpoint_samples, :, GLOBAL_ROT_SLICE
    ]
    assert endpoint_rotation_mask.any()


def test_endpoint_root_reference_can_only_be_disabled_explicitly() -> None:
    motion = torch.randn(1, 8, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.tensor([8]),
        modes=("endpoints",),
        endpoint_subset_mode="all",
        include_root_ref_for_endpoints=False,
        min_keyframes=1,
        max_keyframes=1,
        generator=torch.Generator().manual_seed(3),
    )
    frame = int(torch.where(result.motion_mask[0].any(dim=-1))[0].item())
    assert not result.motion_mask[0, frame, ROOT_SLICE.start : HEADING_SLICE.stop].any()
    assert _selected_joint_ids(result.motion_mask[0, frame])


def test_kimodo_control_curriculum_probabilities_and_progressive_keyframes() -> None:
    batch = 4000
    motion = torch.randn(batch, 40, DIM_HY273)
    lengths = torch.full((batch,), 40)
    config = KimodoControlCurriculum(
        endpoint_preset="five_point",
        max_sparse_keyframes=20,
    )
    start = build_kimodo_control_curriculum_batch(
        motion,
        lengths,
        progress=0.0,
        config=config,
        generator=torch.Generator().manual_seed(123),
    )
    none_fraction = sum(name == "none" for name in start.mode_ids) / batch
    mixed_fraction = sum(name.startswith("mixed:") for name in start.mode_ids) / batch
    assert none_fraction == pytest.approx(0.10, abs=0.025)
    assert mixed_fraction == pytest.approx(0.25, abs=0.03)
    for idx, name in enumerate(start.mode_ids):
        if name == "root_sparse":
            frames = torch.where(start.motion_mask[idx].any(dim=-1))[0]
            assert frames.numel() == 1
            assert not start.motion_mask[idx, frames[0], ROOT_SLICE.start + 1]

    end = build_kimodo_control_curriculum_batch(
        motion[:512],
        lengths[:512],
        progress=1.0,
        config=config,
        generator=torch.Generator().manual_seed(321),
    )
    sparse_counts = [
        int(end.motion_mask[idx].any(dim=-1).sum())
        for idx, name in enumerate(end.mode_ids)
        if name in {"root_sparse", "endpoints", "fullpose"}
    ]
    assert sparse_counts and max(sparse_counts) > 1
    assert max(sparse_counts) <= 20


def test_kimodo_complete_dense_root_covers_every_valid_frame() -> None:
    motion = torch.randn(2, 10, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.tensor([7, 10]),
        modes=("root_dense",),
        dense_min_fraction=1.0,
        generator=torch.Generator().manual_seed(19),
        track_pattern_masks=True,
    )
    assert result.pattern_masks is not None
    dense = result.pattern_masks["root_dense"]
    assert dense[0, :7].any(dim=-1).all()
    assert not dense[0, 7:].any()
    assert dense[1].any(dim=-1).all()
    assert not dense[..., ROOT_SLICE.start + 1].any()


def test_root_heading_probability_covers_position_only_and_position_rotation() -> None:
    batch = 4096
    motion = torch.randn(batch, 24, DIM_HY273)
    schedule = tuple(
        "root_dense" if index % 2 else "root_sparse" for index in range(batch)
    )
    result = build_kimodo_control_curriculum_batch(
        motion,
        lengths=torch.full((batch,), 24),
        progress=1.0,
        config=KimodoControlCurriculum(
            none_prob=0.0,
            mixed_prob=0.0,
            dense_min_fraction=1.0,
            root_heading_probability=0.5,
        ),
        generator=torch.Generator().manual_seed(101),
        root_heading_generator=torch.Generator().manual_seed(202),
        mode_schedule=schedule,
    )

    position_only = [
        index for index, name in enumerate(result.mode_ids) if name.endswith("_2dpos")
    ]
    position_rotation = [
        index for index, name in enumerate(result.mode_ids) if name.endswith("_2dposrot")
    ]
    assert len(position_only) / batch == pytest.approx(0.5, abs=0.03)
    assert len(position_rotation) / batch == pytest.approx(0.5, abs=0.03)
    assert result.motion_mask[:, :, ROOT_SLICE.start].any(dim=-1).all()
    assert result.motion_mask[:, :, ROOT_SLICE.start + 2].any(dim=-1).all()
    assert not result.motion_mask[:, :, ROOT_SLICE.start + 1].any()
    assert not result.motion_mask[position_only, :, HEADING_SLICE].any()
    assert result.motion_mask[position_rotation, :, HEADING_SLICE].any(dim=-1).any(dim=-1).all()


def test_root_heading_draw_does_not_perturb_existing_control_random_stream() -> None:
    batch = 512
    motion = torch.randn(batch, 32, DIM_HY273)
    lengths = torch.full((batch,), 32)
    schedule = tuple("mixed" for _ in range(batch))
    common = dict(
        none_prob=0.0,
        mixed_prob=1.0,
        max_sparse_keyframes=20,
        dense_min_fraction=1.0,
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="random_nonempty",
        include_root_ref_for_endpoints=True,
        include_endpoint_rotations=True,
        include_contact_pattern=True,
    )
    legacy = build_kimodo_control_curriculum_batch(
        motion,
        lengths,
        progress=1.0,
        config=KimodoControlCurriculum(**common, root_heading_probability=1.0),
        generator=torch.Generator().manual_seed(303),
        mode_schedule=schedule,
    )
    rootmask = build_kimodo_control_curriculum_batch(
        motion,
        lengths,
        progress=1.0,
        config=KimodoControlCurriculum(**common, root_heading_probability=0.5),
        generator=torch.Generator().manual_seed(303),
        root_heading_generator=torch.Generator().manual_seed(404),
        mode_schedule=schedule,
    )

    outside_heading = torch.ones(DIM_HY273, dtype=torch.bool)
    outside_heading[HEADING_SLICE] = False
    assert torch.equal(
        legacy.motion_mask[..., outside_heading],
        rootmask.motion_mask[..., outside_heading],
    )
    assert torch.equal(
        legacy.observed_motion[..., outside_heading],
        rootmask.observed_motion[..., outside_heading],
    )
    assert any("root_" in name and "2dpos" in name for name in rootmask.mode_ids)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_root_heading_probability_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="root_heading_probability"):
        KimodoControlCurriculum(root_heading_probability=value).validate()


def test_partial_root_heading_probability_requires_independent_generator() -> None:
    with pytest.raises(ValueError, match="independent root_heading_generator"):
        build_kimodo_control_curriculum_batch(
            torch.randn(1, 12, DIM_HY273),
            lengths=torch.tensor([12]),
            progress=1.0,
            config=KimodoControlCurriculum(
                none_prob=0.0,
                mixed_prob=0.0,
                root_heading_probability=0.5,
            ),
            generator=torch.Generator().manual_seed(17),
            mode_schedule=("root_sparse",),
        )


def test_inference_mode_schedule_assigns_one_requested_mode_per_sample() -> None:
    schedule = (
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
        "contact",
        "mixed",
    )
    motion = torch.randn(len(schedule), 12, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.full((len(schedule),), 12),
        modes=schedule,
        mode_schedule=schedule,
        include_contact_pattern=True,
        dense_min_fraction=1.0,
        generator=torch.Generator().manual_seed(41),
    )
    assert result.mode_ids[:5] == list(schedule[:5])
    assert result.mode_ids[5].startswith("mixed:")
    assert result.motion_mask[0].any()
    assert result.motion_mask[1].any(dim=-1).all()
    assert result.motion_mask[4, :, CONTACT_SLICE].any()

    with pytest.raises(ValueError, match="exactly one mode"):
        build_synthetic_control_batch(
            motion,
            lengths=torch.full((len(schedule),), 12),
            modes=schedule,
            mode_schedule=schedule[:-1],
            include_contact_pattern=True,
        )


def test_pattern_masks_keep_fullbody_separate_from_endpoints_and_hidden_root() -> None:
    motion = torch.randn(2, 12, DIM_HY273)
    fullbody = build_synthetic_control_batch(
        motion[:1],
        lengths=torch.tensor([12]),
        modes=("fullpose",),
        min_keyframes=1,
        max_keyframes=1,
        track_pattern_masks=True,
        generator=torch.Generator().manual_seed(21),
    )
    endpoints = build_synthetic_control_batch(
        motion[1:],
        lengths=torch.tensor([12]),
        modes=("endpoints",),
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="all",
        include_endpoint_rotations=True,
        min_keyframes=1,
        max_keyframes=1,
        track_pattern_masks=True,
        generator=torch.Generator().manual_seed(22),
    )
    assert fullbody.pattern_masks is not None
    assert endpoints.pattern_masks is not None
    assert fullbody.pattern_masks["fullpose"][..., JOINT_POS_SLICE].any()
    assert not fullbody.pattern_masks["endpoints"].any()
    endpoint_pattern = endpoints.pattern_masks["endpoints"]
    assert endpoint_pattern[..., ROOT_SLICE.start : HEADING_SLICE.stop].any()
    assert endpoint_pattern[..., JOINT_POS_SLICE].any()
    assert endpoint_pattern[..., GLOBAL_ROT_SLICE].any()
    assert not endpoints.pattern_masks["fullpose"].any()


def test_last_executed_phase2_update_reaches_full_curriculum_progress() -> None:
    start_step = 200_000
    curriculum_steps = 200_000
    final_optimizer_step = 399_999
    progress = (final_optimizer_step - start_step + 1) / curriculum_steps
    assert progress == 1.0


def test_kimodo_like_architecture_rejects_velocity_prediction() -> None:
    args = build_arg_parser().parse_args([])
    args.architecture = "redenoise_kimodo_like"
    args.prediction_type = "velocity"
    with pytest.raises(ValueError, match="requires --prediction_type x0"):
        create_model(args)


def test_stage2_config_does_not_claim_untrained_contact_control() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "configs/redenoise_kimodo_like_stage2_control.yaml"])
    args = merge_config(
        args,
        {
            "model": {"architecture": "redenoise_kimodo_like"},
            "control": {"training_phase": "control"},
            "loss": {"control_contact": 0.0},
        },
        explicit_cli={"config"},
    )
    assert args.control_contact_loss_weight == 0.0


def test_kimodo_complete_stage2_config_enables_full_control_protocol() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        ["--config", "configs/redenoise_kimodo_complete_stage2_control.yaml"]
    )
    args = merge_config(args, load_yaml(args.config), explicit_cli={"config"})
    args.execution_contract = "stage2_kimodo_complete_production"
    validate_execution_contract(args)
    assert set(args.control_modes.split(",")) == {
        "none",
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
        "contact",
        "mixed",
    }
    assert args.endpoint_preset == "kimodo_ee"
    assert args.control_include_endpoint_rotations is True
    assert args.control_contact_loss_weight > 0.0


def test_public_v1_control_builder_never_injects_root_y_or_contacts() -> None:
    motion = torch.randn(512, 40, DIM_HY273)
    result = build_synthetic_control_batch(
        motion,
        lengths=torch.full((512,), 40),
        modes=("root_sparse", "root_dense", "endpoints", "fullpose", "mixed"),
        endpoint_preset="five_point",
        generator=torch.Generator().manual_seed(9),
    )
    assert not result.motion_mask[..., CONTACT_SLICE].any()
    for batch_idx, mode in enumerate(result.mode_ids):
        if mode in {"root_sparse", "root_dense"}:
            assert not result.motion_mask[batch_idx, :, ROOT_SLICE.start + 1].any()
    assert any(mode == "root_dense" for mode in result.mode_ids)
    assert any(mode.startswith("mixed:") for mode in result.mode_ids)


def test_control_evaluation_uses_phase2_distribution_conditioned_on_control() -> None:
    assert phase2_distribution_conditioned_on_control(0.10, 0.25) == pytest.approx(
        0.25 / 0.90
    )


def test_complete_control_metrics_measure_rotation_and_contact() -> None:
    identity = torch.eye(3).reshape(1, 1, 1, 3, 3)
    quarter_turn = yaw_matrix(torch.tensor(torch.pi / 2)).reshape(1, 1, 1, 3, 3)
    error = masked_rotation_error_deg(
        matrix_to_cont6d(identity),
        matrix_to_cont6d(quarter_turn),
        torch.ones(1, 1, 1, dtype=torch.bool),
    )
    assert float(error) == pytest.approx(90.0, abs=1e-4)

    abs_error, accuracy = masked_contact_metrics(
        torch.tensor([[[0.9, 0.4, 0.7, 0.1]]]),
        torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]),
        torch.ones(1, 1, 4, dtype=torch.bool),
    )
    assert float(abs_error) == pytest.approx(0.375)
    assert float(accuracy) == pytest.approx(0.5)


def test_control_metric_reduction_is_per_sample_and_ignores_absent_cases() -> None:
    pred = torch.tensor(
        [
            [[[1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],
            [[[3.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]],
        ]
    )
    target = torch.zeros_like(pred)
    mask = torch.tensor(
        [
            [[True], [False], [False]],
            [[True], [True], [True]],
        ]
    )
    values, valid = masked_l2_per_sample(pred, target, mask)
    assert values.tolist() == pytest.approx([1.0, 3.0])
    assert valid.tolist() == [True, True]

    together = MetricAccumulator()
    together.add("metric", values, valid)
    split = MetricAccumulator()
    split.add("metric", values[:1], valid[:1])
    split.add("metric", values[1:], valid[1:])
    split.add("metric", torch.tensor([0.0]), torch.tensor([False]))
    assert together.means()["metric"] == pytest.approx(2.0)
    assert split.means()["metric"] == pytest.approx(2.0)
    assert split.counts["metric"] == 2


def test_foot_skate_reports_meters_per_second_from_contacting_feet() -> None:
    joints = torch.zeros(1, 3, 22, 3)
    for joint_id in (7, 10, 8, 11):
        joints[0, :, joint_id, 0] = torch.tensor([0.0, 1.0, 2.0])
    contacts = torch.ones(1, 3, 4, dtype=torch.bool)
    speed = foot_skate_from_joints_per_sample(
        joints, contacts, torch.tensor([3]), fps=30.0
    )
    assert speed.tolist() == pytest.approx([30.0])


def test_text_dropout_is_shared_even_without_deterministic_trace() -> None:
    texts, dropped, internal_probability = apply_deterministic_text_dropout(
        ["walk", "run"], 1.0, torch.device("cpu"), None
    )
    assert texts == ["", ""]
    assert dropped is not None and dropped.all()
    assert internal_probability == 0.0


def test_run_name_rejects_paths() -> None:
    assert validate_run_name("hy273.valid-run_01") == "hy273.valid-run_01"
    for invalid in ("/tmp/run", "../run", "a/b", ""):
        with pytest.raises(ValueError, match="safe run basename"):
            validate_run_name(invalid)


def test_final_execution_contract_rejects_schedule_and_phase_bypasses() -> None:
    parser = build_arg_parser()
    stage1 = parser.parse_args(["--config", "configs/redenoise_kimodo_like_stage1.yaml"])
    stage1 = merge_config(
        stage1,
        load_yaml(stage1.config),
        explicit_cli={"config"},
    )
    stage1.execution_contract = "stage1_production"
    validate_execution_contract(stage1)
    stage1.max_steps = 1
    with pytest.raises(RuntimeError, match="max_steps=200000"):
        validate_execution_contract(stage1)
    stage1.execution_contract = "stage1_pilot"
    validate_execution_contract(stage1)

    stage2 = parser.parse_args(["--config", "configs/redenoise_kimodo_like_stage2_control.yaml"])
    stage2 = merge_config(
        stage2,
        load_yaml(stage2.config),
        explicit_cli={"config"},
    )
    stage2.execution_contract = "stage2_production"
    validate_execution_contract(stage2)
    stage2.training_phase = "text_only"
    with pytest.raises(RuntimeError, match="requires the control phase"):
        validate_execution_contract(stage2)
    stage2.training_phase = "control"
    stage2.control_modes = "none,root_sparse"
    with pytest.raises(RuntimeError, match="frozen v1 control mode set"):
        validate_execution_contract(stage2)


def test_resume_contract_rejects_shape_preserving_text_semantic_change() -> None:
    requested = build_arg_parser().parse_args([])
    requested.data_root = "/data"
    requested.text_root = "/text"
    requested.text_encoder = "hy_cache"
    requested.max_text_tokens = 128
    requested.hytext_cache_dir = "/cache"
    requested.random_first_heading = True
    requested.root_origin_shift = True
    saved = vars(requested).copy()
    validate_resume_contract(requested, saved, "checkpoint.pt")
    for field, changed in (
        ("max_text_tokens", 50),
        ("split", "val"),
        ("time_schedule", "uniform"),
        ("representation_loss_space", "velocity"),
        ("velocity_loss_t_eps", 0.01),
        ("source_manifest_sha256", "different-manifest"),
        ("hytext_allow_cache_miss", True),
    ):
        incompatible = saved.copy()
        incompatible[field] = changed
        with pytest.raises(RuntimeError, match=field):
            validate_resume_contract(requested, incompatible, "checkpoint.pt")


def test_strict_resume_requires_v2_objective_and_source_contract() -> None:
    requested = build_arg_parser().parse_args([])
    saved = vars(requested).copy()
    legacy_fields = (
        "resume_contract_version",
        "representation_loss_space",
        "velocity_loss_t_eps",
        "source_manifest_sha256",
    )
    for field in legacy_fields:
        saved.pop(field)
    with pytest.raises(RuntimeError, match="missing fields"):
        validate_resume_contract(requested, saved, "legacy.pt")
    validate_resume_contract(
        requested,
        saved,
        "legacy.pt",
        allowed_missing_fields=legacy_fields,
    )


def test_ema_contract_rejects_missing_parameter() -> None:
    model_state = {"a": torch.zeros(2), "b": torch.zeros(3)}
    with pytest.raises(RuntimeError, match="missing=.*b"):
        validate_ema_contract(model_state, {"a": torch.zeros(2)}, "checkpoint.pt")


def test_explicit_cli_default_value_overrides_conflicting_yaml() -> None:
    parser = build_arg_parser()
    argv = [
        "--control_modes",
        "none,root,endpoints,fullpose,mixed",
        "--endpoint_subset_mode",
        "random_nonempty",
    ]
    args = parser.parse_args(argv)
    cfg = {
        "control": {
            "modes": ["none"],
            "endpoint_subset_mode": "all",
        }
    }
    merged = merge_config(
        args,
        cfg,
        explicit_cli=explicit_cli_destinations(parser, argv),
    )
    assert merged.control_modes == "none,root,endpoints,fullpose,mixed"
    assert merged.endpoint_subset_mode == "random_nonempty"


def test_training_cli_rejects_abbreviated_options() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--endpoint_subset", "random_nonempty"])


def test_sampling_reuses_checkpoint_endpoint_protocol_and_allows_override() -> None:
    checkpoint_args = Namespace(
        endpoint_preset="five_point",
        endpoint_subset_mode="all",
        endpoint_root_ref_mode="none",
        max_control_keyframes=5,
    )
    inherited = resolve_endpoint_protocol(checkpoint_args)
    assert inherited == {
        "endpoint_preset": "five_point",
        "endpoint_subset_mode": "all",
        "endpoint_root_ref_mode": "none",
        "max_control_keyframes": 5,
        "include_root_ref_for_endpoints": False,
        "include_endpoint_rotations": False,
        "include_contact_pattern": False,
    }
    overridden = resolve_endpoint_protocol(
        checkpoint_args,
        endpoint_preset="kimodo_ee",
        endpoint_subset_mode="random_nonempty",
        endpoint_root_ref_mode="kimodo_hidden_root",
        max_control_keyframes=8,
    )
    assert overridden["endpoint_preset"] == "kimodo_ee"
    assert overridden["endpoint_subset_mode"] == "random_nonempty"
    assert overridden["include_root_ref_for_endpoints"] is True


def test_sampling_reuses_complete_checkpoint_control_modes() -> None:
    checkpoint_args = Namespace(
        control_modes="none,root_sparse,root_dense,endpoints,fullpose,contact,mixed"
    )
    assert resolve_sampling_control_modes(checkpoint_args) == (
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
        "contact",
        "mixed",
    )
