from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.codeflow.motionstreamer272 import (
    decode_motionstreamer272_joints_from_embeddings,
    recover_motionstreamer272_positions_from_normalized,
    recover_motionstreamer272_positions_raw,
)


class TinyDecoder(torch.nn.Module):
    def __init__(self, code_dim: int) -> None:
        super().__init__()
        weight = torch.linspace(-0.05, 0.05, code_dim * 272, dtype=torch.float32).view(code_dim, 272)
        self.register_buffer("weight", weight)

    def decode_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        return torch.matmul(z.mean(dim=2), self.weight.to(device=z.device, dtype=z.dtype))


def test_motionstreamer272_recover_matches_existing_eval_helper() -> None:
    from models.codeflow.eval_inpainting import _recover_motionstreamer272_positions

    motion = torch.randn(2, 5, 272)
    expected = _recover_motionstreamer272_positions(motion, 22)
    actual = recover_motionstreamer272_positions_raw(motion, 22)
    torch.testing.assert_close(actual, expected)


def test_motionstreamer272_normalized_recover_shape() -> None:
    motion = torch.randn(2, 7, 272)
    mean = torch.randn(272)
    std = torch.rand(272).clamp_min(0.1)
    joints = recover_motionstreamer272_positions_from_normalized(motion, mean, std)
    assert joints.shape == (2, 7, 22, 3)
    assert torch.isfinite(joints).all()


def test_decode_to_joints_has_live_embedding_gradient() -> None:
    decoder = TinyDecoder(code_dim=4)
    z = torch.randn(2, 7, 6, 4, requires_grad=True)
    mean = torch.zeros(272)
    std = torch.ones(272)
    joints = decode_motionstreamer272_joints_from_embeddings(decoder, z, mean, std)
    loss = joints.square().mean()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_motionstreamer272_recover_gradcheck() -> None:
    motion = torch.randn(1, 3, 272, dtype=torch.float64, requires_grad=True)
    motion.data[:, :, 2:8].add_(0.5)

    def fn(x: torch.Tensor) -> torch.Tensor:
        return recover_motionstreamer272_positions_raw(x, 22)

    assert torch.autograd.gradcheck(fn, (motion,), eps=1e-6, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    test_motionstreamer272_recover_matches_existing_eval_helper()
    test_motionstreamer272_normalized_recover_shape()
    test_decode_to_joints_has_live_embedding_gradient()
    test_motionstreamer272_recover_gradcheck()
