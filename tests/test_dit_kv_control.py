from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.codeflow.dit_blocks import FrameMotionTextDiT  # noqa: E402


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    motion = torch.randn(2, 5, 24)
    text = torch.randn(2, 3, 24)
    cond = torch.randn(2, 24)
    motion_valid = torch.ones(2, 5, dtype=torch.bool)
    text_padding = torch.zeros(2, 3, dtype=torch.bool)
    pos = torch.arange(5, dtype=torch.float32).view(1, 5, 1).expand(2, -1, -1)
    return motion, text, cond, motion_valid, text_padding, pos


def _open_adaln_gates(model: torch.nn.Module) -> None:
    for module in model.modules():
        if module.__class__.__name__ != "AdaLNModulation":
            continue
        hidden = module.linear.bias.numel() // (3 * module.num)
        with torch.no_grad():
            for idx in range(module.num):
                start = (idx * 3 + 2) * hidden
                module.linear.bias[start : start + hidden].fill_(1.0)


def test_frame_dit_control_none_matches_base() -> None:
    torch.manual_seed(11)
    base = FrameMotionTextDiT(
        hidden_size=24,
        num_heads=3,
        depth_double=1,
        depth_single=1,
        dropout=0.0,
        rope_axes_dims=[8],
    )
    torch.manual_seed(11)
    controlled = FrameMotionTextDiT(
        hidden_size=24,
        num_heads=3,
        depth_double=1,
        depth_single=1,
        dropout=0.0,
        rope_axes_dims=[8],
        control_input_dim=12,
        control_rank=4,
        control_encoder_width=16,
    )
    inputs = _inputs()
    out_base = base(*inputs)
    out_control_none = controlled(*inputs, control_cond=None)
    torch.testing.assert_close(out_control_none, out_base)


def test_frame_dit_control_down_has_live_gradient() -> None:
    torch.manual_seed(22)
    model = FrameMotionTextDiT(
        hidden_size=24,
        num_heads=3,
        depth_double=1,
        depth_single=1,
        dropout=0.0,
        rope_axes_dims=[8],
        control_input_dim=12,
        control_rank=4,
        control_encoder_width=16,
    )
    _open_adaln_gates(model)
    inputs = _inputs()
    control_cond = torch.randn(2, 20, 12)
    out = model(*inputs, control_cond=control_cond)
    out.square().mean().backward()
    grad_norms = [
        float(layer.weight.grad.detach().abs().sum().item())
        for layer in model.control_kv_down
        if layer.weight.grad is not None
    ]
    assert len(grad_norms) == 2
    assert all(value > 0.0 for value in grad_norms)


if __name__ == "__main__":
    test_frame_dit_control_none_matches_base()
    test_frame_dit_control_down_has_live_gradient()
