from __future__ import annotations

import torch

from data.hy273_multitask_scheduler import EditConditionPattern, SamplePlan
from models.raw_motion.hy273_multitask_condition import CapabilityId, TrainStream
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import (
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_SLICE,
    VELOCITY_SLICE,
)
from models.raw_motion.hy273_unified_edit_losses import (
    build_source_target_discrepancy_mask,
    compute_physical_temporal_edit_loss,
    compute_source_anchor_loss,
    compute_source_target_discrepancy_x0_loss,
)
from sample_hy273_multitask import make_edit_condition
from train_hy273_multitask import (
    build_equal_length_source_identity_flow,
    build_stateless_unified_273_flow_inputs,
    load_same_source_instruction_donors,
    resolve_edit_research_treatment,
    same_source_instruction_texts,
    source_fusion_mode_from_checkpoint,
    validate_research_resume_objective,
)


def _plan(stream: TrainStream, ordinal: int) -> SamplePlan:
    return SamplePlan(
        global_step=400_000,
        global_sample_ordinal=ordinal,
        train_stream_id=stream,
        capability_id=(
            CapabilityId.MOTION_EDIT
            if stream == TrainStream.MOTION_EDIT
            else CapabilityId.T2M
        ),
        row_index=ordinal,
        uid=f"row:{ordinal}",
        caption_index=0 if stream == TrainStream.HML_MIXED else None,
        yaw_u64=11 + ordinal,
        control_u64=17 + ordinal,
        text_drop=False,
        edit_pattern=(
            EditConditionPattern.SOURCE_TEXT
            if stream == TrainStream.MOTION_EDIT
            else None
        ),
    )


def _config() -> dict:
    return {
        "flow": {
            "timestep_schedule": "logit_normal",
            "timestep_mean": -0.8,
            "timestep_std": 0.8,
        }
    }


def test_edit_low_t_mixture_is_stateless_and_edit_only() -> None:
    x0 = torch.zeros(4, 5, DIM_HY273)
    edit_plans = [_plan(TrainStream.MOTION_EDIT, index) for index in range(4)]
    hml_plans = [_plan(TrainStream.HML_MIXED, index) for index in range(4)]
    kwargs = dict(
        x0_norm=x0,
        manifest_sha256="manifest",
        run_seed=123,
        config=_config(),
    )

    edit_baseline, noise_baseline, selected_baseline = build_stateless_unified_273_flow_inputs(
        plans=edit_plans, **kwargs
    )
    edit_repeat, noise_repeat, selected_repeat = build_stateless_unified_273_flow_inputs(
        plans=edit_plans, **kwargs
    )
    torch.testing.assert_close(edit_repeat, edit_baseline)
    torch.testing.assert_close(noise_repeat, noise_baseline)
    torch.testing.assert_close(selected_repeat, selected_baseline)
    assert not bool(selected_baseline.any())

    edit_low, noise_low, selected_low = build_stateless_unified_273_flow_inputs(
        plans=edit_plans,
        edit_low_t_mix_prob=1.0,
        edit_low_t_max=0.2,
        **kwargs,
    )
    assert bool((edit_low <= 0.2).all())
    assert bool(selected_low.all())
    torch.testing.assert_close(noise_low, noise_baseline)

    hml_baseline, hml_noise, hml_selected = build_stateless_unified_273_flow_inputs(
        plans=hml_plans, **kwargs
    )
    hml_override, hml_override_noise, hml_override_selected = build_stateless_unified_273_flow_inputs(
        plans=hml_plans,
        edit_low_t_mix_prob=1.0,
        edit_low_t_max=0.2,
        **kwargs,
    )
    torch.testing.assert_close(hml_override, hml_baseline)
    torch.testing.assert_close(hml_override_noise, hml_noise)
    torch.testing.assert_close(hml_override_selected, hml_selected)
    assert not bool(hml_selected.any())


def test_edit_low_t_half_selection_is_replayable_and_near_half() -> None:
    count = 2_048
    plans = [_plan(TrainStream.MOTION_EDIT, index) for index in range(count)]
    kwargs = dict(
        plans=plans,
        x0_norm=torch.zeros(count, 1, DIM_HY273),
        manifest_sha256="manifest",
        run_seed=987,
        config=_config(),
        edit_low_t_mix_prob=0.5,
        edit_low_t_max=0.2,
    )
    timesteps, noise, selected = build_stateless_unified_273_flow_inputs(**kwargs)
    repeated_t, repeated_noise, repeated_selected = (
        build_stateless_unified_273_flow_inputs(**kwargs)
    )
    torch.testing.assert_close(repeated_t, timesteps)
    torch.testing.assert_close(repeated_noise, noise)
    torch.testing.assert_close(repeated_selected, selected)
    assert 0.47 < float(selected.float().mean()) < 0.53
    assert bool((timesteps[selected] <= 0.2).all())


def test_named_research_treatments_are_single_resolved_objectives() -> None:
    baseline = resolve_edit_research_treatment("baseline")
    identity = resolve_edit_research_treatment("anchored_identity")
    identity_low_t = resolve_edit_research_treatment("anchored_identity_low_t")
    clean = resolve_edit_research_treatment("clean_x0_mse")
    low_t = resolve_edit_research_treatment("low_t_only")
    factorial = resolve_edit_research_treatment("clean_x0_mse_low_t")
    same_source = resolve_edit_research_treatment("same_source_contrast")
    same_source_hinge = resolve_edit_research_treatment("same_source_hinge_only")
    same_source_softplus = resolve_edit_research_treatment(
        "same_source_softplus_only"
    )
    positive_only = resolve_edit_research_treatment("no_rank_positive_only")
    changed_positive_only = resolve_edit_research_treatment(
        "same_source_changed_positive_only"
    )
    physical_temporal = resolve_edit_research_treatment(
        "physical_temporal_positive_only"
    )
    source_token = resolve_edit_research_treatment(
        "source_token_block_positive_only"
    )
    discrepancy = resolve_edit_research_treatment("source_target_discrepancy_x0")
    assert baseline["representation_loss_space"] == "velocity_mse"
    assert baseline["contact_loss_space"] == "velocity_mse"
    assert baseline["secondary_branch"] == "shuffled_instruction"
    assert identity["secondary_branch"] == "source_identity"
    assert identity["identity_base_scale"] > 0.0
    assert identity["source_anchor_scale"] > 0.0
    assert identity_low_t["secondary_branch"] == identity["secondary_branch"]
    assert identity_low_t["identity_base_scale"] == identity["identity_base_scale"]
    assert identity_low_t["source_anchor_scale"] == identity["source_anchor_scale"]
    assert identity_low_t["representation_loss_space"] == identity["representation_loss_space"]
    assert identity_low_t["contact_loss_space"] == identity["contact_loss_space"]
    assert identity_low_t["low_t_mix_prob"] == 0.5
    assert identity_low_t["low_t_max"] == 0.2
    assert clean["representation_loss_space"] == "clean_x0_mse"
    assert clean["contact_loss_space"] == "velocity_mse"
    assert clean["low_t_mix_prob"] == 0.0
    assert low_t["representation_loss_space"] == "velocity_mse"
    assert low_t["low_t_mix_prob"] == 0.5
    assert factorial["representation_loss_space"] == "clean_x0_mse"
    assert factorial["low_t_mix_prob"] == 0.5
    assert same_source["representation_loss_space"] == baseline[
        "representation_loss_space"
    ]
    assert same_source["low_t_mix_prob"] == baseline["low_t_mix_prob"]
    assert same_source["secondary_branch"] == "same_source_instruction"
    assert same_source_hinge["secondary_branch"] == "same_source_instruction"
    assert same_source_hinge["instruction_negative_scope"] == "same_source_only"
    assert same_source_hinge["instruction_rank_mode"] == "hinge"
    assert same_source_softplus["instruction_negative_scope"] == "same_source_only"
    assert same_source_softplus["instruction_rank_mode"] == "softplus"
    assert same_source_softplus["instruction_rank_temperature"] == 0.01
    assert same_source_softplus["instruction_rank_multiplier"] == 0.575
    assert positive_only["secondary_branch"] == "none"
    assert positive_only["instruction_rank_multiplier"] == 0.0
    assert positive_only["discrepancy_x0_scale"] == 0.0
    assert positive_only["temporal_scale"] == 0.0
    assert physical_temporal["secondary_branch"] == "none"
    assert physical_temporal["instruction_rank_multiplier"] == 0.0
    assert physical_temporal["discrepancy_x0_scale"] == 0.0
    assert physical_temporal["temporal_scale"] > 0.0
    temporal_diff = {
        key: (positive_only.get(key), physical_temporal.get(key))
        for key in sorted(set(positive_only) | set(physical_temporal))
        if positive_only.get(key) != physical_temporal.get(key)
    }
    assert temporal_diff == {
        "name": ("no_rank_positive_only", "physical_temporal_positive_only"),
        "temporal_scale": (0.0, 0.00053),
    }
    assert changed_positive_only["secondary_branch"] == "none"
    assert changed_positive_only["instruction_rank_multiplier"] == 0.0
    assert changed_positive_only["discrepancy_sample_scope"] == "same_source_only"
    assert changed_positive_only["discrepancy_x0_scale"] == 0.05
    assert changed_positive_only["discrepancy_fraction"] == 0.20
    treatment_diff = {
        key: (positive_only.get(key), changed_positive_only.get(key))
        for key in sorted(set(positive_only) | set(changed_positive_only))
        if positive_only.get(key) != changed_positive_only.get(key)
    }
    assert treatment_diff == {
        "discrepancy_sample_scope": ("all", "same_source_only"),
        "discrepancy_x0_scale": (0.0, 0.05),
        "name": ("no_rank_positive_only", "same_source_changed_positive_only"),
    }
    source_token_diff = {
        key: (positive_only.get(key), source_token.get(key))
        for key in sorted(set(positive_only) | set(source_token))
        if positive_only.get(key) != source_token.get(key)
    }
    assert source_token_diff == {
        "name": (
            "no_rank_positive_only",
            "source_token_block_positive_only",
        ),
        "source_fusion_mode": ("additive", "token_block"),
    }
    assert baseline["discrepancy_x0_scale"] == 0.0
    assert discrepancy["secondary_branch"] == baseline["secondary_branch"]
    assert discrepancy["discrepancy_x0_scale"] == 0.01
    assert discrepancy["discrepancy_fraction"] == 0.20


def test_source_fusion_mode_is_recovered_from_checkpoint_runtime() -> None:
    assert source_fusion_mode_from_checkpoint({}) == "additive"
    checkpoint = {
        "runtime_identity": {
            "research_overrides": {
                "research_treatment": {
                    "source_fusion_mode": "token_block",
                }
            }
        }
    }
    assert source_fusion_mode_from_checkpoint(checkpoint) == "token_block"


def _identity_k273(batch: int, frames: int) -> torch.Tensor:
    motion = torch.zeros(batch, frames, DIM_HY273)
    motion[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    rotations = motion[..., GLOBAL_ROT_SLICE].reshape(batch, frames, 22, 6)
    rotations[...] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return motion


def test_source_target_discrepancy_mask_separates_root_body_and_velocity() -> None:
    source = _identity_k273(3, 5)
    target = source.clone()
    target[1, 4, ROOT_SLICE.start] = 1.0
    joint = 1
    joint_pos_start = JOINT_POS_SLICE.start + 3 * joint
    target[2, 2, joint_pos_start] = 1.0
    valid = torch.ones(3, 5, dtype=torch.bool)
    hard = torch.zeros_like(target, dtype=torch.bool)
    hard[2, 2, joint_pos_start : joint_pos_start + 3] = True

    result = build_source_target_discrepancy_mask(
        source_physical=source[:, None],
        source_lengths=torch.full((3, 1), 5),
        target_physical=target,
        target_valid=valid,
        hard_mask=hard,
        fraction=0.20,
    )

    assert result.exact_identity.tolist() == [True, False, False]
    assert not result.pre_intersection_mask[0].any()
    assert result.root_time_mask[1].nonzero().flatten().tolist() == [4]
    assert not result.root_time_mask[2].any()
    assert result.body_joint_time_mask[2, :, joint].nonzero().flatten().tolist() == [2]
    assert result.velocity_joint_time_mask[2, :, joint].nonzero().flatten().tolist() == [1, 2]
    assert result.pre_intersection_mask[2, 2, joint_pos_start]
    assert not result.mask[2, 2, joint_pos_start]
    velocity_start = VELOCITY_SLICE.start + 3 * joint
    assert result.mask[2, 1, velocity_start]
    assert result.mask[2, 2, velocity_start]


def test_source_target_discrepancy_random_rotations_preserve_exact_identity() -> None:
    torch.manual_seed(7)
    source = _identity_k273(8, 11)
    source[..., GLOBAL_ROT_SLICE] = torch.randn_like(
        source[..., GLOBAL_ROT_SLICE]
    )
    result = build_source_target_discrepancy_mask(
        source_physical=source,
        source_lengths=torch.full((8,), 11),
        target_physical=source.clone(),
        target_valid=torch.ones(8, 11, dtype=torch.bool),
        hard_mask=torch.zeros_like(source, dtype=torch.bool),
        fraction=0.20,
    )

    assert result.exact_identity.tolist() == [True] * 8
    assert not bool(result.pre_intersection_mask.any())
    assert not bool(result.mask.any())


def test_source_target_discrepancy_ignores_source_free_cfg_rows() -> None:
    source = torch.zeros(2, 1, 5, DIM_HY273)
    target = torch.zeros(2, 5, DIM_HY273)
    source[0, 0, :, 5] = torch.linspace(0.0, 0.25, 5)
    target[0, :, 5] = torch.linspace(0.0, 0.5, 5)
    proxy = build_source_target_discrepancy_mask(
        source_physical=source,
        source_lengths=torch.tensor([[5], [0]]),
        target_physical=target,
        target_valid=torch.ones(2, 5, dtype=torch.bool),
        hard_mask=torch.zeros_like(target, dtype=torch.bool),
        fraction=0.20,
    )

    assert bool(proxy.mask[0].any())
    assert not bool(proxy.mask[1].any())
    assert not bool(proxy.pre_intersection_mask[1].any())
    assert not bool(proxy.exact_identity[1])
    assert not bool(proxy.equal_length[1])


def test_source_target_discrepancy_x0_zero_scale_is_gradient_neutral() -> None:
    target = _identity_k273(1, 4)
    source = target.clone()
    target[0, 2, JOINT_POS_SLICE.start + 3] = 0.5
    proxy = build_source_target_discrepancy_mask(
        source_physical=source,
        source_lengths=torch.tensor([4]),
        target_physical=target,
        target_valid=torch.ones(1, 4, dtype=torch.bool),
        hard_mask=torch.zeros_like(target, dtype=torch.bool),
        fraction=0.25,
    )
    prediction = (target[..., :269] + 0.25).detach().requires_grad_(True)
    baseline = compute_source_target_discrepancy_x0_loss(
        correct_x0_hat_cont=prediction,
        x0_target_norm=target,
        discrepancy_mask=proxy.mask,
        scale=0.0,
    )
    assert baseline.raw > 0.0
    assert baseline.weighted == 0.0
    baseline.total.backward()
    assert prediction.grad is not None
    assert not bool(torch.count_nonzero(prediction.grad))


def test_physical_temporal_edit_loss_is_zero_for_exact_target() -> None:
    source = _identity_k273(2, 5)
    target = source.clone()
    joint_offset = JOINT_POS_SLICE.start + 3
    target[0, :, joint_offset] = torch.linspace(0.0, 0.4, 5)
    prediction = target.clone().requires_grad_(True)
    normalizer = HY273Normalizer(
        torch.zeros(DIM_HY273),
        torch.ones(DIM_HY273),
        normalize_contacts=True,
    )
    bundle = compute_physical_temporal_edit_loss(
        x0_hat_norm=prediction,
        x0_target_physical=target,
        source_physical=source[:, None],
        source_lengths=torch.tensor([[5], [5]]),
        target_valid=torch.ones(2, 5, dtype=torch.bool),
        sample_mask=torch.tensor([True, True]),
        normalizer=normalizer,
        scale=0.01,
    )

    assert bundle.raw == 0.0
    assert bundle.vector_raw == 0.0
    assert bundle.speed_raw == 0.0
    assert bundle.active_fraction == 1.0
    assert bundle.importance_mean > 0.0
    bundle.total.backward()
    assert prediction.grad is not None
    assert not bool(torch.count_nonzero(prediction.grad))


def test_physical_temporal_edit_loss_targets_changed_joint_speed() -> None:
    source = _identity_k273(1, 5)
    target = source.clone()
    joint_offset = JOINT_POS_SLICE.start + 3
    target[0, :, joint_offset] = torch.linspace(0.0, 0.4, 5)
    prediction = source.clone().requires_grad_(True)
    normalizer = HY273Normalizer(
        torch.zeros(DIM_HY273),
        torch.ones(DIM_HY273),
        normalize_contacts=True,
    )
    bundle = compute_physical_temporal_edit_loss(
        x0_hat_norm=prediction,
        x0_target_physical=target,
        source_physical=source,
        source_lengths=torch.tensor([5]),
        target_valid=torch.ones(1, 5, dtype=torch.bool),
        sample_mask=torch.tensor([True]),
        normalizer=normalizer,
        scale=0.01,
        change_scale_mps=0.25,
    )

    assert bundle.raw > 0.0
    assert bundle.vector_raw > 0.0
    assert bundle.speed_raw > 0.0
    assert bundle.high_importance_fraction > 0.0
    assert bundle.source_target_velocity_delta_mps > 0.0
    bundle.total.backward()
    assert prediction.grad is not None
    assert bool((prediction.grad[0, :, joint_offset] != 0).any())


def test_physical_temporal_edit_loss_excludes_unequal_length_rows() -> None:
    source = _identity_k273(2, 5)
    target = source.clone()
    joint_offset = JOINT_POS_SLICE.start + 3
    target[:, :, joint_offset] = torch.linspace(0.0, 0.4, 5)
    prediction = source.clone().requires_grad_(True)
    normalizer = HY273Normalizer(
        torch.zeros(DIM_HY273),
        torch.ones(DIM_HY273),
        normalize_contacts=True,
    )
    bundle = compute_physical_temporal_edit_loss(
        x0_hat_norm=prediction,
        x0_target_physical=target,
        source_physical=source,
        source_lengths=torch.tensor([5, 4]),
        target_valid=torch.ones(2, 5, dtype=torch.bool),
        sample_mask=torch.tensor([True, True]),
        normalizer=normalizer,
        scale=0.01,
    )

    assert bundle.active_fraction == 0.5
    assert bundle.equal_length_fraction == 0.5
    bundle.total.backward()
    assert prediction.grad is not None
    assert bool((prediction.grad[0] != 0).any())
    assert not bool((prediction.grad[1] != 0).any())



def test_same_source_instruction_donors_filter_and_fallback(tmp_path) -> None:
    groups = tmp_path / "groups.json"
    groups.write_text(
        """[
          {"source_sha256":"same-a","pair_ids":["000001","000002"],
           "texts":["move faster","move slower"],"target_pair_mse":0.25},
          {"source_sha256":"same-b","pair_ids":["000003","000004"],
           "texts":["turn left","turn right"],"target_pair_mse":0.05}
        ]""",
        encoding="utf-8",
    )
    donors, spec = load_same_source_instruction_donors(
        groups, minimum_target_pair_mse=0.10
    )
    assert donors == {
        "motionfix:000001": "move slower",
        "motionfix:000002": "move faster",
    }
    assert spec["selected_groups"] == 1
    assert spec["eligible_rows"] == 2

    negatives, used = same_source_instruction_texts(
        ["move faster", "unrelated edit", "third edit"],
        ["motionfix:000001", "motionfix:000099", "motionfix:000100"],
        donors,
    )
    assert negatives[0] == "move slower"
    assert negatives[1:] == ["third edit", "move faster"]
    assert used == [True, False, False]

    negatives, used = same_source_instruction_texts(
        ["move faster", "unrelated edit", "third edit"],
        ["motionfix:000001", "motionfix:000099", "motionfix:000100"],
        donors,
        fallback_mode="self",
    )
    assert negatives == ["move slower", "unrelated edit", "third edit"]
    assert used == [True, False, False]


def test_source_identity_flow_uses_only_exact_equal_length_pairs() -> None:
    source = torch.zeros(2, 5, DIM_HY273)
    source[0, :4] = 2.0
    source[1, :5] = 3.0
    source[..., 269:273] = 0.0
    condition = make_edit_condition(
        source,
        source_lengths=torch.tensor([4, 5]),
        target_lengths=torch.tensor([4, 4]),
        target_frames=4,
    )
    normalizer = HY273Normalizer(
        torch.zeros(DIM_HY273),
        torch.ones(DIM_HY273),
        normalize_contacts=True,
    )
    timesteps = torch.full((2,), 0.5)
    noise = torch.zeros(2, 4, DIM_HY273)
    state = build_equal_length_source_identity_flow(
        condition=condition,
        normalizer=normalizer,
        timesteps=timesteps,
        unified_noise=noise,
    )

    assert state["exact_pair"].tolist() == [True, False]
    assert state["target_valid"][0].tolist() == [True] * 4
    assert not bool(state["target_valid"][1].any())
    torch.testing.assert_close(
        state["model_in"][0, :, :269],
        torch.full((4, 269), 1.0),
    )
    assert not bool(state["model_in"][0, :, 269:DIM_HY273].any())


def test_source_anchor_uses_detached_exact_copy_comparator() -> None:
    target = torch.zeros(2, 3, DIM_HY273)
    source = torch.ones_like(target, requires_grad=True)
    correct = torch.ones(2, 3, 269, requires_grad=True)
    bundle = compute_source_anchor_loss(
        correct_x0_hat_cont=correct,
        source_x0_norm=source,
        x0_target_norm=target,
        target_valid=torch.ones(2, 3, dtype=torch.bool),
        hard_mask=torch.zeros_like(target, dtype=torch.bool),
        sample_mask=torch.tensor([True, False]),
        scale=1.0,
        relative_margin=0.1,
    )
    assert bundle.raw > 0
    assert bundle.active_fraction == 0.5
    bundle.total.backward()
    assert correct.grad is not None and bool((correct.grad[0] != 0).any())
    assert not bool((correct.grad[1] != 0).any())
    assert source.grad is None


def test_research_resume_locks_treatment_and_edit_objective() -> None:
    legacy_overrides = {
        "research_treatment": resolve_edit_research_treatment("anchored_identity"),
        "edit_objective": {
            "target_x0_scale": 0.05,
            "hard_x0_scale": 0.02,
        },
    }
    overrides = {
        **legacy_overrides,
        "base_representation_loss_space": "velocity_mse",
        "base_contact_loss_space": "velocity_mse",
        "text_global_conditioning": "pooled_adaln",
        "text_fusion_mode": "f00",
        "conditioning_architecture": "hytext_flux",
        "llm2vec_cache_dir": "",
    }
    validate_research_resume_objective(
        {"research_overrides": legacy_overrides}, overrides
    )
    changed = {
        **overrides,
        "edit_objective": {
            **overrides["edit_objective"],
            "target_x0_scale": 0.10,
        },
    }
    try:
        validate_research_resume_objective(
            {"research_overrides": overrides}, changed
        )
    except RuntimeError as exc:
        assert "Research objective changed" in str(exc)
    else:
        raise AssertionError("Research resume must reject objective drift")

    clean_x0 = {
        **overrides,
        "base_representation_loss_space": "clean_x0_smooth_l1",
        "base_contact_loss_space": "clean_x0_smooth_l1",
    }
    try:
        validate_research_resume_objective(
            {"research_overrides": legacy_overrides}, clean_x0
        )
    except RuntimeError as exc:
        assert "Research objective changed" in str(exc)
    else:
        raise AssertionError("Legacy velocity checkpoint must reject clean-x0 drift")
