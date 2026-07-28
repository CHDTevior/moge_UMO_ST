from __future__ import annotations

import numpy as np
import pytest
import torch

from models.raw_motion.kimodo_context_flow_dit import HY273KimodoContextFlow
from train_hy273_multitask import text_global_conditioning_from_checkpoint


def _model(tmp_path, mode: str) -> HY273KimodoContextFlow:
    full = tmp_path / mode / "full"
    local = tmp_path / mode / "local"
    full.mkdir(parents=True)
    local.mkdir(parents=True)
    np.save(full / "Mean.npy", np.zeros(273, dtype=np.float32))
    np.save(full / "Std.npy", np.ones(273, dtype=np.float32))
    np.save(local / "Mean.npy", np.zeros(4, dtype=np.float32))
    np.save(local / "Std.npy", np.ones(4, dtype=np.float32))
    return HY273KimodoContextFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        max_text_tokens=4,
        text_encoder="null",
        motion_stats_dir=full,
        local_root_stats_dir=local,
        text_global_conditioning=mode,
    )


def test_qwen_tokens_only_removes_pooled_value_but_keeps_zero_gradient(tmp_path):
    model = _model(tmp_path, "qwen_tokens_only")
    timestep = torch.tensor([0.4, 0.7])
    direction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    pooled = torch.randn(2, 16, requires_grad=True)

    actual = model._compose_global_condition(
        timestep, direction, pooled, dtype=torch.float32
    )
    changed = model._compose_global_condition(
        timestep, direction, pooled.detach() + 100.0, dtype=torch.float32
    )

    torch.testing.assert_close(actual, changed, rtol=0.0, atol=0.0)
    actual.sum().backward()
    assert pooled.grad is not None
    assert torch.count_nonzero(pooled.grad) == 0


def test_pooled_adaln_preserves_legacy_global_text_path(tmp_path):
    model = _model(tmp_path, "pooled_adaln")
    timestep = torch.tensor([0.4])
    direction = torch.tensor([[1.0, 0.0]])
    pooled = torch.randn(1, 16)
    delta = torch.randn(1, 16)

    before = model._compose_global_condition(
        timestep, direction, pooled, dtype=torch.float32
    )
    after = model._compose_global_condition(
        timestep, direction, pooled + delta, dtype=torch.float32
    )

    torch.testing.assert_close(after - before, delta, rtol=1e-5, atol=1e-6)


def test_text_global_conditioning_checkpoint_recovery():
    assert text_global_conditioning_from_checkpoint({}) == "pooled_adaln"
    checkpoint = {
        "runtime_identity": {
            "research_overrides": {
                "text_global_conditioning": "qwen_tokens_only"
            }
        }
    }
    assert (
        text_global_conditioning_from_checkpoint(checkpoint)
        == "qwen_tokens_only"
    )
    checkpoint["runtime_identity"]["research_overrides"][
        "text_global_conditioning"
    ] = "unknown"
    with pytest.raises(ValueError, match="unknown text global conditioning"):
        text_global_conditioning_from_checkpoint(checkpoint)
