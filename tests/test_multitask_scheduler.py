from __future__ import annotations

from collections import Counter

import pytest

from data.hy273_multitask_scheduler import (
    CONTEXT_ONLY_EDIT_SCHEDULE_VERSION,
    PROB_SCALE,
    EditConditionPattern,
    HIGH_LEVEL_SCHEDULE_VERSION,
    R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
    STAGE_C_EDIT20_SCHEDULE_VERSION,
    STAGE_C_SAFE_MIX_SCHEDULE_VERSION,
    STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION,
    UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION,
    UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
    UNIFIED_EDIT_V2_SCHEDULE_VERSION,
    UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION,
    WeightedDeficitScheduler,
    edit_pattern_from_draw,
    optimizer_group_hparams,
    phase_for_step,
    probability_units_for_step,
)
from models.raw_motion.hy273_multitask_condition import TrainStream


def test_phase_boundaries_and_optimizer_groups():
    assert phase_for_step(199_999).name == "STAGE_A"
    assert phase_for_step(200_000).name == "STAGE_B1"
    assert phase_for_step(250_000).name == "STAGE_B2"
    assert phase_for_step(400_000).name == "STAGE_C"
    assert optimizer_group_hparams(249_999)["G1_context_weight"]["lr"] == 0.0
    assert optimizer_group_hparams(250_000)["G1_context_weight"]["lr"] == 1e-4
    assert optimizer_group_hparams(400_000)["G0_existing"]["lr"] == 2e-5


def test_probability_units_are_exact_at_ramp_boundaries():
    assert probability_units_for_step(0).__dict__ == {"t2m": PROB_SCALE, "control": 0, "edit": 0}
    assert probability_units_for_step(249_999).__dict__ == {
        "t2m": 500_000,
        "control": 4_500_000,
        "edit": 0,
    }
    assert probability_units_for_step(250_000).__dict__ == {
        "t2m": 500_040,
        "control": 4_499_950,
        "edit": 10,
    }
    assert probability_units_for_step(259_999).__dict__ == {
        "t2m": 900_000,
        "control": 4_000_000,
        "edit": 100_000,
    }
    assert probability_units_for_step(424_999).__dict__ == {
        "t2m": 1_750_000,
        "control": 2_250_000,
        "edit": 1_000_000,
    }


def test_r16_stage_b_is_fixed_t2m_control_without_edit():
    expected = {
        "t2m": 500_000,
        "control": 4_500_000,
        "edit": 0,
    }
    for step in (200_000, 249_999, 250_000, 259_999, 399_999):
        assert probability_units_for_step(
            step,
            R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
        ).__dict__ == expected

    assert (
        optimizer_group_hparams(
            399_999,
            R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
        )["G1_context_weight"]["lr"]
        == 0.0
    )


def test_r16_stage_b_schedule_forks_at_200k_and_stage_c_at_400k():
    stage_a = WeightedDeficitScheduler()
    for _ in range(200_000):
        stage_a.choose()

    stage_b = WeightedDeficitScheduler(
        schedule_version=R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION
    )
    stage_b.load_state_dict(
        stage_a.state_dict(),
        allow_schedule_fork_at_step=200_000,
    )
    assert stage_b.state.next_step == 200_000
    for _ in range(200_000):
        assert stage_b.choose() == TrainStream.HML_MIXED
    assert stage_b.state.realized_edit == 0

    stage_c = WeightedDeficitScheduler(
        schedule_version=UNIFIED_EDIT_V2_SCHEDULE_VERSION
    )
    stage_c.load_state_dict(
        stage_b.state_dict(),
        allow_schedule_fork_at_step=400_000,
    )
    assert stage_c.state.next_step == 400_000


def test_stage_c_safe_mix_profile_changes_only_stage_c():
    assert probability_units_for_step(
        399_999, STAGE_C_SAFE_MIX_SCHEDULE_VERSION
    ) == probability_units_for_step(399_999, HIGH_LEVEL_SCHEDULE_VERSION)
    for step in (400_000, 404_999, 500_000):
        assert probability_units_for_step(
            step, STAGE_C_SAFE_MIX_SCHEDULE_VERSION
        ).__dict__ == {
            "t2m": 900_000,
            "control": 4_000_000,
            "edit": 100_000,
        }


def test_stage_c_edit20_profile_is_fixed_after_400k():
    assert probability_units_for_step(
        399_999, STAGE_C_EDIT20_SCHEDULE_VERSION
    ) == probability_units_for_step(399_999, HIGH_LEVEL_SCHEDULE_VERSION)
    for step in (400_000, 449_999, 500_000):
        assert probability_units_for_step(
            step, STAGE_C_EDIT20_SCHEDULE_VERSION
        ).__dict__ == {
            "t2m": 1_750_000,
            "control": 2_250_000,
            "edit": 1_000_000,
        }


def test_stage_d_edit_calibration_keeps_mix_for_10k_then_reweights():
    assert probability_units_for_step(
        499_999, STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
    ).__dict__ == {
        "t2m": 1_750_000,
        "control": 2_250_000,
        "edit": 1_000_000,
    }
    assert probability_units_for_step(
        500_000, STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
    ).__dict__ == {
        "t2m": 1_750_000,
        "control": 2_250_000,
        "edit": 1_000_000,
    }
    assert probability_units_for_step(
        509_999, STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
    ).__dict__ == {
        "t2m": 1_750_000,
        "control": 2_250_000,
        "edit": 1_000_000,
    }
    assert probability_units_for_step(
        510_000, STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
    ).__dict__ == {
        "t2m": 1_500_000,
        "control": 2_000_000,
        "edit": 1_500_000,
    }


def test_context_only_schedule_is_edit_only_at_every_step():
    for step in (0, 249_999, 400_000, 10_000_000):
        assert probability_units_for_step(
            step, CONTEXT_ONLY_EDIT_SCHEDULE_VERSION
        ).__dict__ == {"t2m": 0, "control": 0, "edit": PROB_SCALE}

    scheduler = WeightedDeficitScheduler(
        schedule_version=CONTEXT_ONLY_EDIT_SCHEDULE_VERSION
    )
    assert [scheduler.choose(step) for step in range(20)] == [
        TrainStream.MOTION_EDIT
    ] * 20


def test_unified_edit_schedule_and_condition_patterns():
    for step in (400_000, 500_000, 10_000_000):
        assert probability_units_for_step(
            step, UNIFIED_EDIT_V2_SCHEDULE_VERSION
        ).__dict__ == {
            "t2m": 1_500_000,
            "control": 2_000_000,
            "edit": 1_500_000,
        }
    draws = [0, (1 << 64) * 89 // 100, (1 << 64) * 91 // 100]
    assert [
        edit_pattern_from_draw(draw, UNIFIED_EDIT_V2_SCHEDULE_VERSION)
        for draw in draws
    ] == [
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
    ]


def test_unified_edit40_schedule_and_condition_patterns():
    for step in (450_000, 499_999, 500_000):
        assert probability_units_for_step(
            step, UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
        ).__dict__ == {
            "t2m": 1_500_000,
            "control": 1_500_000,
            "edit": 2_000_000,
        }
    draws = [0, (1 << 64) * 89 // 100, (1 << 64) * 91 // 100]
    assert [
        edit_pattern_from_draw(draw, UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION)
        for draw in draws
    ] == [
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
    ]
    boundary = (9 * (1 << 64) + 9) // 10
    assert edit_pattern_from_draw(
        boundary - 1, UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
    ) == EditConditionPattern.SOURCE_TEXT
    assert edit_pattern_from_draw(
        boundary, UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
    ) == EditConditionPattern.SOURCE_TEXT_CONTROL


def test_unified_decomposed_cfg_schedule_and_condition_patterns():
    for step in (400_000, 449_999, 500_000):
        assert probability_units_for_step(
            step, UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION
        ).__dict__ == {
            "t2m": 1_500_000,
            "control": 2_000_000,
            "edit": 1_500_000,
        }
    draws = [
        0,
        (1 << 64) * 76 // 100,
        (1 << 64) * 86 // 100,
        (1 << 64) * 91 // 100,
        (1 << 64) * 96 // 100,
    ]
    assert [
        edit_pattern_from_draw(draw, UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION)
        for draw in draws
    ] == [
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_IDENTITY,
        EditConditionPattern.TEXT_ONLY,
        EditConditionPattern.UNCONDITIONAL,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
    ]


def test_unified_decomposed_cfg_edit80_preserves_inner_patterns():
    for step in (450_000, 499_999, 500_000):
        assert probability_units_for_step(
            step, UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION
        ).__dict__ == {
            "t2m": 500_000,
            "control": 500_000,
            "edit": 4_000_000,
        }
    draws = [
        0,
        (1 << 64) * 76 // 100,
        (1 << 64) * 86 // 100,
        (1 << 64) * 91 // 100,
        (1 << 64) * 96 // 100,
    ]
    assert [
        edit_pattern_from_draw(
            draw, UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION
        )
        for draw in draws
    ] == [
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_IDENTITY,
        EditConditionPattern.TEXT_ONLY,
        EditConditionPattern.UNCONDITIONAL,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
    ]


def test_unified_decomposed_cfg_edit80_forks_at_450k():
    parent = WeightedDeficitScheduler(
        schedule_version=UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION
    )
    parent.state.next_step = 450_000
    parent.state.debt_hml = -50_000
    parent.state.debt_edit = 50_000
    parent.state.realized_hml = 432_100
    parent.state.realized_edit = 17_900

    fork = WeightedDeficitScheduler(
        schedule_version=UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION
    )
    fork.load_state_dict(
        parent.state_dict(), allow_schedule_fork_at_step=450_000
    )
    assert fork.state.__dict__ == parent.state.__dict__

    for wrong_boundary in (449_999, 450_001):
        rejected = WeightedDeficitScheduler(
            schedule_version=UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION
        )
        with pytest.raises(ValueError, match="version mismatch"):
            rejected.load_state_dict(
                parent.state_dict(), allow_schedule_fork_at_step=wrong_boundary
            )


def test_unified_decomposed_cfg_forks_only_at_exact_400k_boundary():
    parent = WeightedDeficitScheduler()
    for _ in range(400_000):
        parent.choose()
    state = parent.state_dict()
    fork = WeightedDeficitScheduler(
        schedule_version=UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION
    )
    fork.load_state_dict(state, allow_schedule_fork_at_step=400_000)
    assert fork.state.next_step == 400_000


def test_stage_c_safe_mix_forks_only_at_exact_400k_boundary():
    parent = WeightedDeficitScheduler()
    for _ in range(400_000):
        parent.choose()
    state = parent.state_dict()

    fork = WeightedDeficitScheduler(
        schedule_version=STAGE_C_SAFE_MIX_SCHEDULE_VERSION
    )
    fork.load_state_dict(state, allow_schedule_fork_at_step=400_000)
    assert fork.state.next_step == 400_000
    assert probability_units_for_step(
        fork.state.next_step, fork.schedule_version
    ).__dict__ == {
        "t2m": 900_000,
        "control": 4_000_000,
        "edit": 100_000,
    }


def test_stage_c_edit20_forks_only_at_exact_400k_boundary():
    parent = WeightedDeficitScheduler()
    for _ in range(400_000):
        parent.choose()
    state = parent.state_dict()

    fork = WeightedDeficitScheduler(
        schedule_version=STAGE_C_EDIT20_SCHEDULE_VERSION
    )
    fork.load_state_dict(state, allow_schedule_fork_at_step=400_000)
    assert fork.state.next_step == 400_000
    assert probability_units_for_step(
        fork.state.next_step, fork.schedule_version
    ).__dict__ == {
        "t2m": 1_750_000,
        "control": 2_250_000,
        "edit": 1_000_000,
    }


def test_unified_edit_forks_only_at_exact_400k_boundary():
    parent = WeightedDeficitScheduler()
    for _ in range(400_000):
        parent.choose()
    state = parent.state_dict()

    fork = WeightedDeficitScheduler(
        schedule_version=UNIFIED_EDIT_V2_SCHEDULE_VERSION
    )
    fork.load_state_dict(state, allow_schedule_fork_at_step=400_000)
    assert fork.state.next_step == 400_000
    assert probability_units_for_step(
        fork.state.next_step, fork.schedule_version
    ).__dict__ == {
        "t2m": 1_500_000,
        "control": 2_000_000,
        "edit": 1_500_000,
    }


def test_unified_edit40_forks_only_at_exact_450k_boundary():
    parent = WeightedDeficitScheduler(
        schedule_version=UNIFIED_EDIT_V2_SCHEDULE_VERSION
    )
    parent.state.next_step = 450_000
    parent.state.debt_hml = -50_000
    parent.state.debt_edit = 50_000
    parent.state.realized_hml = 432_100
    parent.state.realized_edit = 17_900

    fork = WeightedDeficitScheduler(
        schedule_version=UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
    )
    fork.load_state_dict(
        parent.state_dict(), allow_schedule_fork_at_step=450_000
    )
    assert fork.state.__dict__ == parent.state.__dict__
    assert probability_units_for_step(
        fork.state.next_step, fork.schedule_version
    ).__dict__ == {
        "t2m": 1_500_000,
        "control": 1_500_000,
        "edit": 2_000_000,
    }

    for wrong_boundary in (449_999, 450_001):
        rejected = WeightedDeficitScheduler(
            schedule_version=UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
        )
        with pytest.raises(ValueError, match="version mismatch"):
            rejected.load_state_dict(
                parent.state_dict(), allow_schedule_fork_at_step=wrong_boundary
            )

    wrong_step = parent.state_dict()
    wrong_step["next_step"] = 449_999
    with pytest.raises(ValueError, match="version mismatch"):
        fork.load_state_dict(wrong_step, allow_schedule_fork_at_step=450_000)

    wrong_parent = parent.state_dict()
    wrong_parent["format"] = HIGH_LEVEL_SCHEDULE_VERSION
    with pytest.raises(ValueError, match="version mismatch"):
        fork.load_state_dict(wrong_parent, allow_schedule_fork_at_step=450_000)


def test_stage_d_calibration_forks_only_at_exact_500k_boundary():
    parent = WeightedDeficitScheduler(
        schedule_version=STAGE_C_EDIT20_SCHEDULE_VERSION
    )
    for _ in range(500_000):
        parent.choose()
    state = parent.state_dict()

    fork = WeightedDeficitScheduler(
        schedule_version=STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
    )
    fork.load_state_dict(state, allow_schedule_fork_at_step=500_000)
    assert fork.state.next_step == 500_000
    assert probability_units_for_step(
        fork.state.next_step, fork.schedule_version
    ).__dict__ == {
        "t2m": 1_750_000,
        "control": 2_250_000,
        "edit": 1_000_000,
    }


def test_weighted_deficit_full_schedule_and_resume_replay():
    scheduler = WeightedDeficitScheduler()
    prefix = [scheduler.choose() for _ in range(300_123)]
    state = scheduler.state_dict()
    continuation = [scheduler.choose() for _ in range(10_000)]

    resumed = WeightedDeficitScheduler()
    resumed.load_state_dict(state)
    assert continuation == [resumed.choose() for _ in range(10_000)]
    assert Counter(prefix)[TrainStream.MOTION_EDIT] == 902


def test_edit_pattern_integer_bins():
    assert edit_pattern_from_draw(0) == EditConditionPattern.SOURCE_TEXT
    draws = [0, (1 << 64) * 71 // 100, (1 << 64) * 81 // 100, (1 << 64) * 96 // 100]
    assert [edit_pattern_from_draw(draw) for draw in draws] == [
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_ONLY,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
        EditConditionPattern.SOURCE_CONTROL,
    ]


def test_stage_d_edit_pattern_bins_train_hierarchical_cfg_branches():
    draws = [
        0,
        (1 << 64) * 76 // 100,
        (1 << 64) * 86 // 100,
        (1 << 64) * 91 // 100,
        (1 << 64) * 96 // 100,
    ]
    assert [
        edit_pattern_from_draw(draw, STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION)
        for draw in draws
    ] == [
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_IDENTITY,
        EditConditionPattern.TEXT_ONLY,
        EditConditionPattern.UNCONDITIONAL,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
    ]

    assert [
        edit_pattern_from_draw(draw, CONTEXT_ONLY_EDIT_SCHEDULE_VERSION)
        for draw in draws
    ] == [
        EditConditionPattern.SOURCE_TEXT,
        EditConditionPattern.SOURCE_IDENTITY,
        EditConditionPattern.TEXT_ONLY,
        EditConditionPattern.UNCONDITIONAL,
        EditConditionPattern.SOURCE_TEXT_CONTROL,
    ]
