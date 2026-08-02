from __future__ import annotations

import torch

from models.raw_motion.hy273_reaction_metrics import reaction_fixed_role_metrics
from models.raw_motion.hy273_slices import (
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_SLICE,
)


def _motion(batch: int = 1, frames: int = 6) -> torch.Tensor:
    value = torch.zeros(batch, frames, DIM_HY273)
    value[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    value[..., GLOBAL_ROT_SLICE] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ).repeat(22)
    return value


def test_fixed_role_reaction_metrics_are_zero_for_exact_reactor() -> None:
    source = _motion()
    target = _motion()
    target[..., ROOT_SLICE.start] = 1.0
    target[..., JOINT_POS_SLICE] = 0.1
    result = reaction_fixed_role_metrics(source, target, target)
    assert result["assignment_rule"] == "fixed_source_actor_to_target_reactor_no_swap"
    assert result["aggregate"]["reactor_position_mpjpe_cm"] == 0.0
    assert result["aggregate"]["reactor_fk_mpjpe_cm"] == 0.0
    assert result["per_sample"][0]["assignment"] == "fixed_actor_to_reactor"


def test_copying_source_is_not_hidden_by_actor_swap_or_pair_average() -> None:
    source = _motion()
    target = _motion()
    target[..., ROOT_SLICE.start] = 2.0
    copied_source = source.clone()
    result = reaction_fixed_role_metrics(source, copied_source, target)
    assert result["aggregate"]["reactor_position_mpjpe_cm"] > 190.0
    assert result["aggregate"]["reactor_root_error_cm"] > 190.0
    assert result["aggregate"]["prediction_to_source_position_mpjpe_cm"] == 0.0


def test_reaction_metrics_respect_lengths() -> None:
    source = _motion(batch=2)
    target = source.clone()
    prediction = target.clone()
    prediction[0, 3:, ROOT_SLICE.start] = 100.0
    result = reaction_fixed_role_metrics(
        source,
        prediction,
        target,
        lengths=torch.tensor([3, 6]),
    )
    assert result["per_sample"][0]["reactor_root_error_cm"] == 0.0
