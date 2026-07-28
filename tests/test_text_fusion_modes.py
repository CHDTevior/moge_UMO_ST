from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from models.codeflow.dit_blocks import (
    DoubleStreamBlock,
    FrameMotionTextDiT,
    SingleStreamBlock,
    TEXT_FUSION_MODES,
)
from train_hy273_multitask import (
    create_model,
    create_model_from_checkpoint,
    current_code_identity,
    text_fusion_mode_from_checkpoint,
    validate_checkpoint_text_fusion_implementation,
    validate_checkpoint_text_fusion_mode,
)


def _open_adaln_gates(module: torch.nn.Module) -> None:
    for child in module.modules():
        if child.__class__.__name__ != "AdaLNModulation":
            continue
        hidden = child.linear.bias.numel() // (3 * child.num)
        with torch.no_grad():
            for idx in range(child.num):
                gate_start = (idx * 3 + 2) * hidden
                child.linear.bias[gate_start : gate_start + hidden].fill_(1.0)


def _frame_inputs():
    torch.manual_seed(123)
    motion = torch.randn(2, 5, 24)
    text = torch.randn(2, 3, 24)
    cond = torch.randn(2, 24)
    motion_valid = torch.ones(2, 5, dtype=torch.bool)
    text_padding = torch.zeros(2, 3, dtype=torch.bool)
    pos = torch.arange(5).view(1, 5, 1).expand(2, -1, -1)
    return motion, text, cond, motion_valid, text_padding, pos


def _load_legacy_dit_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "external_repos/273_motion_raw_diffusion_plan/models/codeflow/dit_blocks.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_hy273_legacy_dit_blocks", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_f00_default_is_exactly_legacy_compatible() -> None:
    legacy_module = _load_legacy_dit_module()
    kwargs = dict(
        hidden_size=24,
        num_heads=3,
        depth_double=2,
        depth_single=2,
        mlp_ratio=1.5,
        dropout=0.0,
        rope_axes_dims=[8],
    )
    torch.manual_seed(71)
    legacy = legacy_module.FrameMotionTextDiT(**kwargs)
    torch.manual_seed(71)
    default = FrameMotionTextDiT(**kwargs)
    torch.manual_seed(71)
    explicit = FrameMotionTextDiT(**kwargs, text_fusion_mode="f00")

    assert tuple(legacy.state_dict()) == tuple(default.state_dict())
    assert tuple(default.state_dict()) == tuple(explicit.state_dict())
    for name, value in default.state_dict().items():
        torch.testing.assert_close(
            value, legacy.state_dict()[name], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            value, explicit.state_dict()[name], rtol=0.0, atol=0.0
        )
    _open_adaln_gates(legacy)
    _open_adaln_gates(default)
    _open_adaln_gates(explicit)
    legacy_output = legacy(*_frame_inputs())
    default_output = default(*_frame_inputs())
    explicit_output = explicit(*_frame_inputs())
    torch.testing.assert_close(
        legacy_output,
        default_output,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        default_output, explicit_output, rtol=0.0, atol=0.0
    )
    legacy_output.square().mean().backward()
    default_output.square().mean().backward()
    for (legacy_name, legacy_parameter), (name, parameter) in zip(
        legacy.named_parameters(), default.named_parameters()
    ):
        assert legacy_name == name
        assert legacy_parameter.grad is not None
        assert parameter.grad is not None
        torch.testing.assert_close(
            legacy_parameter.grad, parameter.grad, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize(
    ("shared_mode", "separate_mode"),
    (("f00", "f01"), ("f10", "f11")),
)
def test_separate_projection_starts_from_tied_shared_solution_without_aliasing(
    shared_mode: str,
    separate_mode: str,
) -> None:
    kwargs = dict(
        hidden_size=24,
        num_heads=3,
        depth_double=2,
        depth_single=2,
        mlp_ratio=1.5,
        dropout=0.0,
        rope_axes_dims=[8],
    )
    torch.manual_seed(72)
    shared = FrameMotionTextDiT(**kwargs, text_fusion_mode=shared_mode)
    shared_rng = torch.get_rng_state().clone()
    torch.manual_seed(72)
    separate = FrameMotionTextDiT(**kwargs, text_fusion_mode=separate_mode)
    separate_rng = torch.get_rng_state().clone()

    assert torch.equal(shared_rng, separate_rng)
    shared_state = shared.state_dict()
    separate_state = separate.state_dict()
    for name, value in shared_state.items():
        if ".joint_attn." in name:
            for stream in ("motion", "text"):
                mapped = name.replace(".joint_attn.", f".{stream}_attn.")
                torch.testing.assert_close(
                    value, separate_state[mapped], rtol=0.0, atol=0.0
                )
        else:
            torch.testing.assert_close(
                value, separate_state[name], rtol=0.0, atol=0.0
            )

    for block in separate.double_blocks:
        assert block.motion_attn is not None
        assert block.text_attn is not None
        for motion_parameter, text_parameter in zip(
            block.motion_attn.parameters(), block.text_attn.parameters()
        ):
            assert motion_parameter is not text_parameter
            assert motion_parameter.data_ptr() != text_parameter.data_ptr()

    _open_adaln_gates(shared)
    _open_adaln_gates(separate)
    torch.testing.assert_close(
        shared(*_frame_inputs()),
        separate(*_frame_inputs()),
        rtol=1e-5,
        atol=1e-6,
    )


def test_fusion_parameter_bundles_match_the_2x2_contract() -> None:
    counts = {}
    keys = {}
    kwargs = dict(
        hidden_size=24,
        num_heads=3,
        depth_double=2,
        depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        rope_axes_dims=[8],
    )
    for mode in TEXT_FUSION_MODES:
        model = FrameMotionTextDiT(**kwargs, text_fusion_mode=mode)
        counts[mode] = sum(parameter.numel() for parameter in model.parameters())
        keys[mode] = tuple(model.state_dict())

    assert counts["f00"] == counts["f10"]
    assert counts["f01"] == counts["f11"]
    assert counts["f01"] > counts["f00"]
    assert keys["f00"] == keys["f10"]
    assert keys["f01"] == keys["f11"]
    assert keys["f00"] != keys["f01"]


@pytest.mark.parametrize("mode", ("f10", "f11"))
def test_double_stream_asymmetric_text_is_motion_invariant(mode: str) -> None:
    torch.manual_seed(81)
    block = DoubleStreamBlock(
        hidden_size=24,
        num_heads=3,
        mlp_ratio=1.0,
        dropout=0.0,
        text_fusion_mode=mode,
    ).eval()
    _open_adaln_gates(block)
    motion = torch.randn(2, 5, 24)
    text = torch.randn(2, 3, 24)
    cond = torch.randn(2, 24)
    motion_valid = torch.ones(2, 5, dtype=torch.bool)
    text_valid = torch.ones(2, 3, dtype=torch.bool)
    pos = torch.arange(5).view(1, 5, 1).expand(2, -1, -1)

    motion_a, text_a = block(
        motion,
        text,
        cond,
        motion_valid,
        text_valid,
        pos,
        [8],
    )
    motion_b, text_b = block(
        motion + 3.0 * torch.randn_like(motion),
        text,
        cond,
        motion_valid,
        text_valid,
        pos,
        [8],
    )
    torch.testing.assert_close(text_a, text_b, rtol=0.0, atol=0.0)
    assert not torch.allclose(motion_a, motion_b)

    changed_text_motion, _ = block(
        motion,
        text + 3.0 * torch.randn_like(text),
        cond,
        motion_valid,
        text_valid,
        pos,
        [8],
    )
    assert not torch.allclose(motion_a, changed_text_motion)


@pytest.mark.parametrize("mode", ("f10", "f11"))
def test_single_stream_asymmetric_text_is_motion_invariant(mode: str) -> None:
    torch.manual_seed(91)
    block = SingleStreamBlock(
        hidden_size=24,
        num_heads=3,
        mlp_ratio=1.0,
        dropout=0.0,
        text_fusion_mode=mode,
    ).eval()
    _open_adaln_gates(block)
    motion = torch.randn(2, 5, 24)
    text = torch.randn(2, 3, 24)
    cond = torch.randn(2, 24)
    valid = torch.ones(2, 8, dtype=torch.bool)
    motion_pos = torch.arange(5).view(1, 5, 1).expand(2, -1, -1)
    text_pos = torch.zeros(2, 3, 1, dtype=torch.long)
    pos = torch.cat([motion_pos, text_pos], dim=1)

    output_a = block(
        torch.cat([motion, text], dim=1),
        cond,
        valid,
        pos,
        [8],
        motion_token_count=5,
    )
    output_b = block(
        torch.cat(
            [motion + 3.0 * torch.randn_like(motion), text], dim=1
        ),
        cond,
        valid,
        pos,
        [8],
        motion_token_count=5,
    )
    torch.testing.assert_close(
        output_a[:, 5:], output_b[:, 5:], rtol=0.0, atol=0.0
    )

    output_changed_text = block(
        torch.cat(
            [motion, text + 3.0 * torch.randn_like(text)], dim=1
        ),
        cond,
        valid,
        pos,
        [8],
        motion_token_count=5,
    )
    assert not torch.allclose(
        output_a[:, :5], output_changed_text[:, :5]
    )


@pytest.mark.parametrize("mode", TEXT_FUSION_MODES)
def test_control_extra_kv_respects_attention_direction(mode: str) -> None:
    torch.manual_seed(101)
    block = DoubleStreamBlock(
        hidden_size=24,
        num_heads=3,
        mlp_ratio=1.0,
        dropout=0.0,
        text_fusion_mode=mode,
    ).eval()
    _open_adaln_gates(block)
    motion, text, cond, motion_valid, text_padding, pos = _frame_inputs()
    text_valid = ~text_padding
    base_motion, base_text = block(
        motion,
        text,
        cond,
        motion_valid,
        text_valid,
        pos,
        [8],
    )
    control = torch.randn_like(motion)
    controlled_motion, controlled_text = block(
        motion,
        text,
        cond,
        motion_valid,
        text_valid,
        pos,
        [8],
        control_k=control,
        control_v=control,
        control_valid=motion_valid,
        control_pos=pos,
    )

    assert not torch.equal(base_motion, controlled_motion)
    if mode in {"f00", "f01"}:
        assert not torch.equal(base_text, controlled_text)
    else:
        torch.testing.assert_close(
            base_text, controlled_text, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("mode", TEXT_FUSION_MODES)
def test_padded_tokens_and_positions_do_not_change_valid_motion(mode: str) -> None:
    torch.manual_seed(111)
    model = FrameMotionTextDiT(
        hidden_size=24,
        num_heads=3,
        depth_double=2,
        depth_single=2,
        mlp_ratio=1.0,
        dropout=0.0,
        rope_axes_dims=[8],
        text_fusion_mode=mode,
    ).eval()
    _open_adaln_gates(model)
    motion = torch.randn(2, 5, 24)
    text = torch.randn(2, 4, 24)
    cond = torch.randn(2, 24)
    motion_valid = torch.tensor(
        [[True] * 5, [True, True, True, False, False]]
    )
    text_padding = torch.tensor(
        [[False] * 4, [False, False, True, True]]
    )
    pos = torch.arange(5).view(1, 5, 1).expand(2, -1, -1).clone()

    perturbed_motion = motion.clone()
    perturbed_motion[1, 3:] = 1000.0 * torch.randn_like(
        perturbed_motion[1, 3:]
    )
    perturbed_text = text.clone()
    perturbed_text[1, 2:] = 1000.0 * torch.randn_like(
        perturbed_text[1, 2:]
    )
    perturbed_pos = pos.clone()
    perturbed_pos[1, 3:] += 10_000

    expected = model(
        motion, text, cond, motion_valid, text_padding, pos
    )
    actual = model(
        perturbed_motion,
        perturbed_text,
        cond,
        motion_valid,
        text_padding,
        perturbed_pos,
    )
    torch.testing.assert_close(
        expected[0], actual[0], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        expected[1, :3], actual[1, :3], rtol=0.0, atol=0.0
    )


def test_text_fusion_checkpoint_recovery_defaults_old_models_to_f00() -> None:
    assert text_fusion_mode_from_checkpoint({}) == "f00"
    checkpoint = {
        "runtime_identity": {
            "research_overrides": {"text_fusion_mode": "f11"}
        }
    }
    assert text_fusion_mode_from_checkpoint(checkpoint) == "f11"
    checkpoint["runtime_identity"]["research_overrides"][
        "text_fusion_mode"
    ] = "unknown"
    with pytest.raises(ValueError, match="unknown text fusion mode"):
        text_fusion_mode_from_checkpoint(checkpoint)


def test_resume_rejects_fusion_change_even_for_fresh_fork() -> None:
    checkpoint = {
        "runtime_identity": {
            "research_overrides": {"text_fusion_mode": "f10"}
        }
    }
    validate_checkpoint_text_fusion_mode(checkpoint, "f10")
    with pytest.raises(RuntimeError, match="changed across checkpoint load"):
        validate_checkpoint_text_fusion_mode(checkpoint, "f00")


def test_explicit_fusion_checkpoint_rejects_implementation_drift() -> None:
    current = current_code_identity()
    checkpoint = {
        "runtime_identity": {
            "research_overrides": {"text_fusion_mode": "f10"}
        },
        "code_identity": current,
    }
    assert (
        validate_checkpoint_text_fusion_implementation(checkpoint, current)
        == "exact"
    )
    for name in (
        "train_hy273_multitask.py",
        "models/codeflow/dit_blocks.py",
        "models/raw_motion/kimodo_context_flow_dit.py",
        "models/raw_motion/kimodo_like_flow_dit.py",
        "models/raw_motion/raw_flow_dit.py",
    ):
        drifted = {
            **current,
            "files": {**current["files"], name: "changed"},
        }
        with pytest.raises(
            RuntimeError, match="Text fusion implementation changed"
        ):
            validate_checkpoint_text_fusion_implementation(
                checkpoint, drifted
            )


def test_legacy_implicit_f00_skips_fusion_implementation_guard() -> None:
    assert (
        validate_checkpoint_text_fusion_implementation(
            {}, current_code_identity()
        )
        == "legacy_f00_unpinned"
    )


@pytest.mark.parametrize("mode", TEXT_FUSION_MODES)
def test_checkpoint_round_trip_rebuilds_fusion_model(
    tmp_path, mode: str
) -> None:
    stats_root = tmp_path / "stats"
    full = stats_root / "full"
    local = stats_root / "local_root"
    full.mkdir(parents=True)
    local.mkdir(parents=True)
    np.save(full / "Mean.npy", np.zeros(273, dtype=np.float32))
    np.save(full / "Std.npy", np.ones(273, dtype=np.float32))
    np.save(local / "Mean.npy", np.zeros(4, dtype=np.float32))
    np.save(local / "Std.npy", np.ones(4, dtype=np.float32))
    config = {
        "data": {
            "stats_root": str(stats_root),
            "max_target_frames": 8,
        },
        "flow": {"contact_protocol": "unified_273_clean_flow_v1"},
        "loss": {"fps": 30.0},
        "model": {
            "hidden_dim": 16,
            "num_heads": 4,
            "root_depth_double": 1,
            "root_depth_single": 1,
            "body_depth_double": 1,
            "body_depth_single": 1,
            "mlp_ratio": 1.0,
            "dropout": 0.0,
            "stats_variance_eps": 0.0,
            "detach_root_bridge": True,
            "self_conditioning": False,
        },
        "text": {
            "encoder": "null",
            "cache_dir": "",
            "max_tokens": 4,
            "ctxt_dim": 16,
            "vtxt_dim": 8,
            "max_open_shards": 1,
            "strict_cache": True,
        },
    }
    original = create_model(config, text_fusion_mode=mode)
    checkpoint = {
        "config": config,
        "model": original.state_dict(),
        "code_identity": current_code_identity(),
        "runtime_identity": {
            "research_overrides": {
                "research_treatment": {"source_fusion_mode": "additive"},
                "text_global_conditioning": "pooled_adaln",
                "text_fusion_mode": mode,
            }
        },
    }
    checkpoint_path = tmp_path / f"{mode}.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = create_model_from_checkpoint(loaded)
    restored.load_state_dict(loaded["model"], strict=True)
    assert restored.root_backbone.text_fusion_mode == mode
    assert restored.body_backbone.text_fusion_mode == mode
    assert tuple(original.state_dict()) == tuple(restored.state_dict())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(
            value, restored.state_dict()[name], rtol=0.0, atol=0.0
        )
