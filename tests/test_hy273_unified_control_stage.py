from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from data.hy273_multitask_manifest_dataset import build_global_sample_plans
from data.hy273_multitask_scheduler import (
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    orthogonal_control_from_ordinal,
)
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    TaskId,
    TrainStream,
)
from models.raw_motion.hy273_slices import CONTACT_SLICE, DIM_HY273
from sample_hy273_multitask import (
    _guided_prediction,
    make_edit_condition,
    make_reaction_condition,
)
from tools.eval_hy273_unified_task_control import (
    CONTROL_SUBTYPES,
    _case_plan,
    _summarize,
)
from train_hy273_unified_actor import (
    FULLTEXT_REACTION_V5_1_CONTROL_CONTRACT,
    FULLTEXT_STAGE_C_CONTROL_CONTRACT,
    REACTION_V5_1_CONTROL_CONFIG,
    build_orthogonal_hard_controls,
    load_config,
    make_loss_weights,
    validate_config,
    validate_fulltext_phase_contract,
    validate_resume_config,
)


class _PlanDataset:
    def __init__(self, stream: TrainStream) -> None:
        self.stream = stream
        self.manifest_sha256 = f"manifest-{int(stream)}"

    def uid(self, row_index: int) -> str:
        return f"row-{int(row_index)}"

    def caption_count(self, row_index: int) -> int:
        return 3


@pytest.mark.parametrize(
    ("stream", "control_capability", "plain_capability"),
    [
        (TrainStream.HML_MIXED, CapabilityId.KIMODO_CONTROL, CapabilityId.T2M),
        (
            TrainStream.MOTION_EDIT,
            CapabilityId.MOTION_EDIT_CONTROL,
            CapabilityId.MOTION_EDIT,
        ),
    ],
)
def test_stage_c_control_is_exact_90_10_inside_each_task(
    stream: TrainStream,
    control_capability: CapabilityId,
    plain_capability: CapabilityId,
) -> None:
    dataset = _PlanDataset(stream)
    start = 22_400_000 if stream == TrainStream.HML_MIXED else 5_600_000
    plans = build_global_sample_plans(
        dataset=dataset,
        row_indices=list(range(100)),
        global_step=350_000,
        first_global_ordinal=start,
        run_seed=20260801,
        schedule_version=KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
        orthogonal_control_probability=0.90,
    )
    assert sum(plan.control_present for plan in plans) == 90
    assert sum(plan.capability_id == control_capability for plan in plans) == 90
    assert sum(plan.capability_id == plain_capability for plan in plans) == 10
    replay = build_global_sample_plans(
        dataset=dataset,
        row_indices=list(range(100)),
        global_step=350_000,
        first_global_ordinal=start,
        run_seed=20260801,
        schedule_version=KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
        orthogonal_control_probability=0.90,
    )
    assert replay == plans


def test_orthogonal_control_sequence_is_exact_for_reaction_stream() -> None:
    start = 5_600_000
    values = [
        orthogonal_control_from_ordinal(
            start + offset,
            0.90,
            phase=int(TrainStream.REACTION) * 3,
        )
        for offset in range(100)
    ]
    assert sum(values) == 90


@pytest.mark.parametrize(
    ("stream", "control_capability", "plain_capability"),
    [
        (TrainStream.HML_MIXED, CapabilityId.KIMODO_CONTROL, CapabilityId.T2M),
        (
            TrainStream.MOTION_EDIT,
            CapabilityId.MOTION_EDIT_CONTROL,
            CapabilityId.MOTION_EDIT,
        ),
    ],
)
def test_reaction_v5_1_control_is_exact_80_20_inside_each_task(
    stream: TrainStream,
    control_capability: CapabilityId,
    plain_capability: CapabilityId,
) -> None:
    dataset = _PlanDataset(stream)
    plans = build_global_sample_plans(
        dataset=dataset,
        row_indices=list(range(100)),
        global_step=300_000,
        first_global_ordinal=20_480_000 if stream == TrainStream.HML_MIXED else 4_480_000,
        run_seed=20260801,
        schedule_version=KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
        orthogonal_control_probability=0.80,
    )
    assert sum(plan.control_present for plan in plans) == 80
    assert sum(plan.capability_id == control_capability for plan in plans) == 80
    assert sum(plan.capability_id == plain_capability for plan in plans) == 20


def test_reaction_v5_1_control_is_exact_80_20_for_reaction() -> None:
    values = [
        orthogonal_control_from_ordinal(
            4_480_000 + offset,
            0.80,
            phase=int(TrainStream.REACTION) * 3,
        )
        for offset in range(100)
    ]
    assert sum(values) == 80


@pytest.mark.parametrize(
    ("config_path", "global_step"),
    [
        (
            "configs/hy273_unified_fulltext_reaction_v1_stage_c_control.yaml",
            350_000,
        ),
        (
            "configs/hy273_unified_fulltext_reaction_v5_1_control80_continue400k.yaml",
            300_000,
        ),
    ],
)
def test_reaction_control_compiles_from_reactor_target_and_never_padding(
    config_path: str,
    global_step: int,
) -> None:
    config, _ = load_config(
        config_path
    )
    source = torch.zeros(1, 8, DIM_HY273)
    source[..., CONTACT_SLICE] = 0.0
    condition = make_reaction_condition(
        source,
        target_lengths=torch.tensor([6]),
        target_frames=8,
        source_person_index=0,
        capability=CapabilityId.REACTION_CONTROL,
    )
    target = torch.randn(1, 8, DIM_HY273)
    target[..., CONTACT_SLICE] = torch.randint(0, 2, (1, 8, 4)).float()
    plan = SimpleNamespace(
        global_sample_ordinal=5_600_000,
        uid="reaction-case",
        control_present=True,
        control_u64=1234,
    )
    observed, mask, modes, present = build_orthogonal_hard_controls(
        target_physical=target,
        condition=condition,
        plans=[plan],
        global_step=global_step,
        config=config,
        manifest_sha256="reaction-manifest",
        run_seed=20260801,
        stream=TrainStream.REACTION,
    )
    assert present.tolist() == [True]
    assert modes[0] != "none"
    assert mask[0, :6].any()
    assert not mask[0, 6:].any()
    torch.testing.assert_close(observed[mask], target[mask], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(observed[~mask]) == 0
    assert not torch.equal(observed[mask], source.expand_as(target)[mask])


@pytest.mark.parametrize(
    ("identity_branch", "expected_value"),
    [(False, 1.0), (True, 0.0)],
)
def test_edit_control_uses_each_cfg_branch_effective_target(
    identity_branch: bool,
    expected_value: float,
) -> None:
    config, _ = load_config(
        "configs/hy273_unified_fulltext_reaction_v5_1_control80_continue400k.yaml"
    )
    source = torch.zeros(1, 8, DIM_HY273)
    motionfix_target = torch.ones(1, 8, DIM_HY273)
    effective_target = source.clone() if identity_branch else motionfix_target
    condition = make_edit_condition(
        source,
        target_lengths=torch.tensor([8]),
        capability=CapabilityId.MOTION_EDIT_CONTROL,
    )
    plan = SimpleNamespace(
        global_sample_ordinal=4_480_010 if identity_branch else 4_480_011,
        uid="edit-source-identity" if identity_branch else "edit-source-text",
        control_present=True,
        control_u64=1234,
    )
    observed, mask, modes, present = build_orthogonal_hard_controls(
        target_physical=effective_target,
        condition=condition,
        plans=[plan],
        global_step=300_000,
        config=config,
        manifest_sha256="edit-manifest",
        run_seed=20260801,
        stream=TrainStream.MOTION_EDIT,
    )
    assert present.tolist() == [True]
    assert modes[0] != "none"
    assert mask.any()
    torch.testing.assert_close(
        observed[mask],
        torch.full_like(observed[mask], expected_value),
        rtol=0.0,
        atol=0.0,
    )


def test_reaction_control_layered_cfg_unit_scales_equal_all_branch() -> None:
    branches = {
        "empty": torch.full((1, 2, DIM_HY273), 1.0),
        "source": torch.full((1, 2, DIM_HY273), 2.0),
        "joint": torch.full((1, 2, DIM_HY273), 4.0),
        "all": torch.full((1, 2, DIM_HY273), 8.0),
    }
    output = _guided_prediction(
        branches,
        route="reaction",
        has_control=True,
        text_cfg_scale=1.0,
        source_cfg_scale=1.0,
        edit_cfg_scale=1.0,
        control_cfg_scale=1.0,
        cfg_apply_contacts=True,
    )
    torch.testing.assert_close(output, branches["all"], rtol=0.0, atol=0.0)


def test_stage_c_config_and_350k_transition_are_narrow() -> None:
    parent, _ = load_config(
        "configs/hy273_unified_fulltext_reaction_v1_continue350k.yaml"
    )
    stage_c, _ = load_config(
        "configs/hy273_unified_fulltext_reaction_v1_stage_c_control.yaml"
    )
    validate_config(stage_c)
    validate_resume_config(
        parent,
        stage_c,
        allow_same_mix_extension_at_step=350_000,
        allow_control_stage_transition_at_step=350_000,
    )
    changed = {
        **stage_c,
        "control": {**stage_c["control"], "present_probability": 0.80},
    }
    with pytest.raises(ValueError, match="config changed"):
        validate_resume_config(
            parent,
            changed,
            allow_same_mix_extension_at_step=350_000,
            allow_control_stage_transition_at_step=350_000,
        )
    weights = make_loss_weights(stage_c)
    assert weights.control_continuous == 0.25
    assert weights.control_contact == pytest.approx(0.02857142857142857)


def test_reaction_v5_1_control_config_and_300k_transition_are_narrow() -> None:
    parent, _ = load_config(
        "configs/hy273_unified_fulltext_reaction_v5_1_full_contact_continue300k.yaml"
    )
    control, _ = load_config(
        "configs/hy273_unified_fulltext_reaction_v5_1_control80_continue400k.yaml"
    )
    validate_config(control)
    assert control["control"] == REACTION_V5_1_CONTROL_CONFIG
    assert "ease" not in control
    validate_resume_config(
        parent,
        control,
        allow_control_stage_transition_at_step=300_000,
    )
    changed = {
        **control,
        "control": {**control["control"], "present_probability": 0.90},
    }
    with pytest.raises(ValueError, match="config changed"):
        validate_resume_config(
            parent,
            changed,
            allow_control_stage_transition_at_step=300_000,
        )
    weights = make_loss_weights(control)
    assert weights.control_continuous == 0.25
    assert weights.control_contact == pytest.approx(0.02857142857142857)


def test_stage_c_phase_is_strictly_segmented_in_50k_gates() -> None:
    for stop in (400_000, 450_000, 500_000):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_C_CONTROL_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=stop,
            global_step=stop - 50_000,
        )
    with pytest.raises(ValueError, match=r"\[400000,450000\)"):
        validate_fulltext_phase_contract(
            FULLTEXT_STAGE_C_CONTROL_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=450_000,
            global_step=350_000,
        )


def test_reaction_v5_1_control_phase_is_one_100k_stage() -> None:
    for step in (300_000, 350_000, 399_999):
        validate_fulltext_phase_contract(
            FULLTEXT_REACTION_V5_1_CONTROL_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=400_000,
            global_step=step,
        )
    with pytest.raises(ValueError, match="stop_step=400000"):
        validate_fulltext_phase_contract(
            FULLTEXT_REACTION_V5_1_CONTROL_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=350_000,
            global_step=300_000,
        )
    with pytest.raises(ValueError, match=r"\[300000,400000\)"):
        validate_fulltext_phase_contract(
            FULLTEXT_REACTION_V5_1_CONTROL_CONTRACT,
            has_resume=True,
            run_dir_exists=True,
            declared_stop_step=400_000,
            global_step=299_999,
        )


def test_reaction_control_remains_reaction_task_not_new_task() -> None:
    source = torch.zeros(1, 4, DIM_HY273)
    condition = make_reaction_condition(
        source,
        target_lengths=torch.tensor([4]),
        source_person_index=1,
        capability=CapabilityId.REACTION_CONTROL,
    )
    assert condition.task_id.tolist() == [int(TaskId.REACTION)]
    assert condition.capability_id.tolist() == [int(CapabilityId.REACTION_CONTROL)]
    assert not bool(condition.ease_present.any())


def test_control_gate_case_plan_covers_every_subtype_replayably() -> None:
    plan = _case_plan(
        1_000,
        task="reaction",
        seed=20260801,
        cases_per_subtype=2,
    )
    assert len(plan) == 2 * len(CONTROL_SUBTYPES)
    assert {case.subtype for case in plan} == set(CONTROL_SUBTYPES)
    assert plan == _case_plan(
        1_000,
        task="reaction",
        seed=20260801,
        cases_per_subtype=2,
    )


def test_control_gate_improvement_uses_metric_direction() -> None:
    record = {
        "task": "t2m",
        "subtype": "path_2dpos",
        "controlled": {
            "constraint_root2d_err": 0.10,
            "constraint_root2d_acc": 0.90,
        },
        "control_zero": {
            "constraint_root2d_err": 0.30,
            "constraint_root2d_acc": 0.40,
        },
        "controlled_task_metrics": {"target_fk_mpjpe_cm": 10.0},
        "control_zero_task_metrics": {"target_fk_mpjpe_cm": 11.0},
    }
    summary = _summarize([record])["t2m/all"]
    improvement = summary["positive_means_control_helped"]
    assert improvement["constraint_root2d_err"] == pytest.approx(0.20)
    assert improvement["constraint_root2d_acc"] == pytest.approx(0.50)
