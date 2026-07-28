import json

import numpy as np
import pytest
import torch

from models.raw_motion.hy273_ease import (
    EASE_STATS_FORMAT,
    HY273EaseConditioner,
    ease_from_k273,
)
from models.raw_motion.hy273_normalizer import apply_yaw_rotation
from models.raw_motion.hy273_slices import DIM_HY273, yaw_rotate_positions


def _motion_from_root(root: torch.Tensor) -> torch.Tensor:
    motion = torch.zeros(root.shape[0], DIM_HY273)
    motion[:, :3] = root
    motion[:, 3] = 1.0
    return motion


def test_endpoint_exact_straight_motion_is_zero():
    root = torch.stack(
        [
            torch.linspace(0.0, 2.0, 12),
            torch.full((12,), 1.0),
            torch.linspace(0.0, -1.0, 12),
        ],
        dim=-1,
    )
    label = ease_from_k273(_motion_from_root(root))
    assert torch.allclose(label, torch.zeros(6), atol=2e-7, rtol=0.0)


def test_ease_is_translation_invariant_and_yaw_equivariant():
    u = torch.linspace(0.0, 1.0, 14)
    root = torch.stack([u.square(), 1.0 + 0.1 * u.square(), u**3], dim=-1)
    motion = _motion_from_root(root)
    label = ease_from_k273(motion)

    translated = motion.clone()
    translated[:, 0] += 17.0
    translated[:, 1] -= 3.0
    translated[:, 2] += 4.0
    assert torch.allclose(
        ease_from_k273(translated), label, atol=2e-6, rtol=0.0
    )

    angle = torch.tensor(0.73)
    rotated = ease_from_k273(apply_yaw_rotation(motion, angle))
    expected = yaw_rotate_positions(label.reshape(2, 3), angle).reshape(6)
    assert torch.allclose(rotated, expected, atol=2e-6, rtol=0.0)


def test_padding_is_excluded_and_must_be_prefix():
    root = torch.stack(
        [torch.linspace(0.0, 1.0, 9) ** 2, torch.ones(9), torch.zeros(9)],
        dim=-1,
    )
    motion = _motion_from_root(root)
    padded = torch.cat([motion, torch.randn(5, DIM_HY273)], dim=0)
    valid = torch.zeros(14, dtype=torch.bool)
    valid[:9] = True
    assert torch.allclose(
        ease_from_k273(padded, valid), ease_from_k273(motion), atol=1e-7
    )
    valid[10] = True
    with pytest.raises(ValueError, match="prefix"):
        ease_from_k273(padded, valid)


def test_zero_initialized_conditioner_is_exact_identity(tmp_path):
    np.save(tmp_path / "Mean.npy", np.zeros(6, dtype=np.float32))
    np.save(tmp_path / "Std.npy", np.ones(6, dtype=np.float32))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"format": EASE_STATS_FORMAT, "feature_dim": 6})
    )
    conditioner = HY273EaseConditioner(32, tmp_path)
    values = torch.randn(4, 6)
    present = torch.tensor([True, False, True, False])
    output = conditioner(values, present, dtype=torch.float32)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in conditioner.parameters())
