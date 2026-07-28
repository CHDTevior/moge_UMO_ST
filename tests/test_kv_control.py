from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.codeflow.kv_control import (  # noqa: E402
    build_joint_control_condition,
    masked_joint_position_loss,
    sample_random_joint_position_control,
)


def test_build_joint_control_condition_shape_and_values() -> None:
    current = torch.zeros(1, 2, 2, 3)
    target = torch.ones(1, 2, 2, 3)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 0] = True
    cond = build_joint_control_condition(current, target, mask)
    assert cond.shape == (1, 2, 12)
    assert torch.all(cond[..., :3] == 1)
    assert torch.all(cond[..., 3:6] == 0)
    assert torch.all(cond[..., 6:9] == 1)
    assert torch.all(cond[..., 9:12] == 0)


def test_masked_joint_position_loss() -> None:
    pred = torch.zeros(1, 1, 2, 3)
    target = torch.ones(1, 1, 2, 3)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[:, :, 0] = True
    loss = masked_joint_position_loss(pred, target, mask, loss_type="l1")
    assert torch.isclose(loss, torch.tensor(1.0))


def test_sample_random_joint_position_control() -> None:
    generator = torch.Generator().manual_seed(123)
    motion = torch.randn(2, 16, 272)
    mean = torch.zeros(272)
    std = torch.ones(272)
    batch = sample_random_joint_position_control(
        motion,
        torch.tensor([16, 8]),
        mean,
        std,
        min_keyframes=2,
        max_keyframes=3,
        min_joints=2,
        max_joints=4,
        generator=generator,
    )
    assert batch["target_joints"].shape == (2, 16, 22, 3)
    assert batch["target_mask"].shape == (2, 16, 22, 3)
    assert batch["target_mask"][0, 16:].sum() == 0
    assert batch["target_mask"][1, 8:].sum() == 0
    assert batch["target_mask"].sum() > 0


if __name__ == "__main__":
    test_build_joint_control_condition_shape_and_values()
    test_masked_joint_position_loss()
    test_sample_random_joint_position_control()
