from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from models.raw_motion.hytext_cache import (
    LLM2VEC_CACHE_FORMAT,
    hytext_profile_key,
)
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    make_absent_condition,
)
from models.raw_motion.hy273_slices import DIM_HY273, LOCAL_ROOT_DIM, ROOT_DIM
from models.raw_motion.kimodo_context_flow_dit import HY273KimodoContextFlow
from models.raw_motion.kimodo_prefix_transformer import (
    KimodoPrefixTransformer,
)
from models.raw_motion.llm2vec_cache import CachedLLM2VecTextEncoder
from train_hy273_multitask import (
    base_loss_spaces_from_checkpoint,
    conditioning_architecture_from_checkpoint,
)
from tools.hy273_runtime_text_encoding import (
    encode_missing_text_rows,
    register_runtime_text_rows,
)


PROFILE = "hytext_absolute_motion_v1"
ENCODER_ID = "test-llm2vec"
PROMPT_VERSION = "test-raw-text-v1"


def _build_cache(root: Path) -> None:
    shard = root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    ctxt = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.save(shard / "ctxt.npy", ctxt)
    np.save(shard / "vtxt.npy", np.zeros((3, 1, 1), dtype=np.float32))
    np.save(shard / "ctxt_len.npy", np.ones(3, dtype=np.int16))
    rows = ("walk", "dance", "")
    index = {}
    for row, text in enumerate(rows):
        key = hytext_profile_key(
            text,
            PROFILE,
            encoder_identity=ENCODER_ID,
            prompt_template_version=PROMPT_VERSION,
        )
        index[key] = {
            "shard": "shard_00000",
            "row": row,
            "text": text,
            "profile": PROFILE,
        }
    (root / "index.json").write_text(json.dumps(index))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": LLM2VEC_CACHE_FORMAT,
                "encoder_identity": ENCODER_ID,
                "prompt_template_version": PROMPT_VERSION,
                "ctxt_dim": 6,
                "vtxt_dim": 1,
                "max_length_llm": 1,
                "encoding_batch_size": 1,
            }
        )
    )


def _build_stats(root: Path) -> tuple[Path, Path]:
    motion = root / "full"
    local_root = root / "local_root"
    motion.mkdir(parents=True)
    local_root.mkdir(parents=True)
    np.save(motion / "Mean.npy", np.zeros(DIM_HY273, dtype=np.float32))
    np.save(motion / "Std.npy", np.ones(DIM_HY273, dtype=np.float32))
    np.save(
        local_root / "Mean.npy",
        np.zeros(LOCAL_ROOT_DIM, dtype=np.float32),
    )
    np.save(
        local_root / "Std.npy",
        np.ones(LOCAL_ROOT_DIM, dtype=np.float32),
    )
    return motion, local_root


def test_llm2vec_cache_returns_one_sentence_token_and_zero_dropout(
    tmp_path: Path,
) -> None:
    _build_cache(tmp_path)
    encoder = CachedLLM2VecTextEncoder(
        hidden_dim=8,
        cache_dir=tmp_path,
        embedding_dim=6,
        project_tokens=False,
    )
    condition = encoder(
        ["walk", "dance"],
        profiles=[PROFILE, PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition.tokens.shape == (2, 1, 6)
    assert not torch.equal(condition.tokens[0], condition.tokens[1])
    assert condition.pooled.shape == (2, 8)
    assert torch.count_nonzero(condition.pooled) == 0
    assert not condition.padding_mask.any()

    dropped = encoder(
        ["walk", "dance"],
        profiles=[PROFILE, PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
        force_drop=True,
    )
    assert torch.count_nonzero(dropped.tokens) == 0


def test_llm2vec_flux_projection_is_trainable(tmp_path: Path) -> None:
    _build_cache(tmp_path)
    encoder = CachedLLM2VecTextEncoder(
        hidden_dim=8,
        cache_dir=tmp_path,
        embedding_dim=6,
        project_tokens=True,
    )
    output = encoder(
        ["walk"],
        profiles=[PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert output.tokens.shape == (1, 1, 8)
    output.tokens.square().mean().backward()
    assert encoder.token_proj.weight.grad is not None
    assert torch.count_nonzero(encoder.token_proj.weight.grad) > 0


def test_runtime_llm2vec_rows_preserve_profile_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_cache(tmp_path)
    encoder = CachedLLM2VecTextEncoder(
        hidden_dim=8,
        cache_dir=tmp_path,
        embedding_dim=6,
    )

    def fake_encode(_cache, pairs, _device):
        count = len(pairs)
        return (
            torch.zeros(count, 1, 1),
            torch.arange(count * 6, dtype=torch.float32).view(count, 1, 6),
            torch.ones(count, dtype=torch.long),
        )

    monkeypatch.setattr(
        "tools.hy273_runtime_text_encoding._encode_llm2vec",
        fake_encode,
    )
    rows = encode_missing_text_rows(
        encoder.cache,
        ["walk", "novel", "novel"],
        [PROFILE, PROFILE, PROFILE],
        torch.device("cpu"),
    )
    assert rows.texts == ("novel",)
    assert rows.profiles == (PROFILE,)
    assert register_runtime_text_rows(encoder.cache, rows) == 1
    assert encoder.cache.text_source("novel", PROFILE) == "runtime_encoder"


def test_kimodo_prefix_transformer_uses_text_time_and_heading() -> None:
    torch.manual_seed(9)
    model = KimodoPrefixTransformer(
        hidden_size=16,
        num_heads=4,
        num_layers=2,
        mlp_ratio=1.0,
        dropout=0.0,
        text_dim=6,
        num_register_tokens=3,
        max_sequence_length=64,
    ).eval()
    motion = torch.randn(2, 5, 16)
    text = torch.randn(2, 1, 6)
    cond = torch.randn(2, 2, 16)
    valid = torch.ones(2, 5, dtype=torch.bool)
    padding = torch.zeros(2, 1, dtype=torch.bool)
    pos = torch.arange(5).view(1, 5, 1).expand(2, -1, -1)

    baseline = model(motion, text, cond, valid, padding, pos)
    changed_text = model(motion, text + 1.0, cond, valid, padding, pos)
    changed_time = model(
        motion,
        text,
        cond + torch.tensor([[[1.0] * 16, [0.0] * 16]]),
        valid,
        padding,
        pos,
    )
    assert baseline.shape == motion.shape
    assert not torch.allclose(baseline, changed_text)
    assert not torch.allclose(baseline, changed_time)
    baseline.square().mean().backward()
    assert model.text_proj.weight.grad is not None
    assert torch.count_nonzero(model.text_proj.weight.grad) > 0


def test_kimodo_prefix_rejects_projected_text_tokens() -> None:
    model = KimodoPrefixTransformer(
        hidden_size=16,
        num_heads=4,
        num_layers=1,
        text_dim=6,
    )
    with pytest.raises(ValueError, match="one raw LLM2Vec token"):
        model(
            torch.randn(1, 3, 16),
            torch.randn(1, 1, 16),
            torch.randn(1, 2, 16),
            torch.ones(1, 3, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.arange(3).view(1, 3, 1),
        )


def test_full_kimodo_prefix_model_routes_text_to_root_and_body(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    _build_cache(cache)
    motion_stats, local_root_stats = _build_stats(tmp_path / "stats")
    model = HY273KimodoContextFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        mlp_ratio=1.0,
        max_text_tokens=1,
        text_encoder="llm2vec_cache",
        hytext_cache_dir=str(cache),
        hytext_ctxt_dim=6,
        hytext_vtxt_dim=1,
        motion_stats_dir=str(motion_stats),
        local_root_stats_dir=str(local_root_stats),
        max_frames=8,
        text_global_conditioning="llm2vec_tokens_only",
        backbone_type="kimodo_prefix",
    )
    frames = 4
    condition = make_absent_condition(
        batch_size=1,
        target_frames=frames,
        target_lengths=torch.tensor([frames]),
        capability=CapabilityId.T2M,
    )
    model_in = torch.randn(1, frames, DIM_HY273 * 2)
    kwargs = {
        "t": torch.tensor([0.4]),
        "c_dir": torch.tensor([[1.0, 0.0]]),
        "length_mask": torch.ones(1, frames, dtype=torch.bool),
        "condition": condition,
    }
    walk = model(model_in, text=["walk"], **kwargs)
    dance = model(model_in, text=["dance"], **kwargs)
    assert walk.shape == (1, frames, DIM_HY273)
    assert not torch.allclose(walk, dance)
    walk.square().mean().backward()
    for backbone in (model.root_backbone, model.body_backbone):
        assert backbone.text_proj.weight.grad is not None
        assert torch.count_nonzero(backbone.text_proj.weight.grad) > 0


def test_full_llm2vec_flux_model_opens_causal_root_and_body_text_paths(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    _build_cache(cache)
    motion_stats, local_root_stats = _build_stats(tmp_path / "stats")
    model = HY273KimodoContextFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        mlp_ratio=1.0,
        max_text_tokens=1,
        text_encoder="llm2vec_cache",
        hytext_cache_dir=str(cache),
        hytext_ctxt_dim=6,
        hytext_vtxt_dim=1,
        motion_stats_dir=str(motion_stats),
        local_root_stats_dir=str(local_root_stats),
        max_frames=8,
        text_global_conditioning="llm2vec_tokens_only",
        backbone_type="flux",
    )
    frames = 4
    condition = make_absent_condition(
        batch_size=2,
        target_frames=frames,
        target_lengths=torch.tensor([frames, frames]),
        capability=CapabilityId.T2M,
    )
    model_in = torch.randn(2, frames, DIM_HY273 * 2)
    target = torch.randn(2, frames, DIM_HY273)
    kwargs = {
        "t": torch.tensor([0.3, 0.7]),
        "c_dir": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "length_mask": torch.ones(2, frames, dtype=torch.bool),
        "condition": condition,
    }

    # AdaLN-Zero deliberately starts text-independent. Two real optimizer
    # updates open its residual gates before testing end-to-end causality.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(model_in, text=["walk", "dance"], **kwargs)
        (prediction - target).square().mean().backward()
        optimizer.step()

    with torch.no_grad():
        correct = model(model_in, text=["walk", "dance"], **kwargs)
        permuted = model(model_in, text=["dance", "walk"], **kwargs)
        dropped = model(
            model_in,
            text=["walk", "dance"],
            force_drop_text=True,
            **kwargs,
        )
    assert not torch.allclose(
        correct[..., :ROOT_DIM],
        permuted[..., :ROOT_DIM],
    )
    assert not torch.allclose(
        correct[..., ROOT_DIM:],
        permuted[..., ROOT_DIM:],
    )
    assert not torch.allclose(correct, dropped)

    for output_slice, backbone in (
        (slice(None, ROOT_DIM), model.root_backbone),
        (slice(ROOT_DIM, None), model.body_backbone),
    ):
        model.zero_grad(set_to_none=True)
        prediction = model(model_in, text=["walk", "dance"], **kwargs)
        prediction[..., output_slice].square().mean().backward()
        text_grads = [
            parameter.grad
            for parameter in model.text_encoder.token_proj.parameters()
            if parameter.grad is not None
        ]
        backbone_grads = [
            parameter.grad
            for parameter in backbone.parameters()
            if parameter.grad is not None
        ]
        assert text_grads and backbone_grads
        assert all(torch.isfinite(gradient).all() for gradient in text_grads)
        assert all(torch.isfinite(gradient).all() for gradient in backbone_grads)
        assert sum(torch.count_nonzero(gradient) for gradient in text_grads) > 0
        assert sum(torch.count_nonzero(gradient) for gradient in backbone_grads) > 0


def test_conditioning_architecture_checkpoint_recovery() -> None:
    assert conditioning_architecture_from_checkpoint({}) == "hytext_flux"
    checkpoint = {
        "runtime_identity": {
            "research_overrides": {
                "conditioning_architecture": "llm2vec_kimodo_prefix"
            }
        }
    }
    assert (
        conditioning_architecture_from_checkpoint(checkpoint)
        == "llm2vec_kimodo_prefix"
    )
    checkpoint["runtime_identity"]["research_overrides"][
        "conditioning_architecture"
    ] = "unknown"
    with pytest.raises(ValueError, match="unknown conditioning architecture"):
        conditioning_architecture_from_checkpoint(checkpoint)


def test_base_loss_space_checkpoint_recovery_is_backward_compatible() -> None:
    assert base_loss_spaces_from_checkpoint({}) == (
        "velocity_mse",
        "velocity_mse",
    )
    checkpoint = {
        "runtime_identity": {
            "research_overrides": {
                "base_representation_loss_space": "clean_x0_smooth_l1",
                "base_contact_loss_space": "clean_x0_smooth_l1",
            }
        }
    }
    assert base_loss_spaces_from_checkpoint(checkpoint) == (
        "clean_x0_smooth_l1",
        "clean_x0_smooth_l1",
    )
    checkpoint["runtime_identity"]["research_overrides"][
        "base_contact_loss_space"
    ] = "unknown"
    with pytest.raises(ValueError, match="unknown base_contact_loss_space"):
        base_loss_spaces_from_checkpoint(checkpoint)
