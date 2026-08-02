from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from models.codeflow.dit_blocks import FrameMotionTextDiT
from models.raw_motion.hytext_cache import (
    LLM2VEC_CACHE_FORMAT,
    hytext_profile_key,
)
from models.raw_motion.hy273_actor_exchange import (
    ActorExchangeFrameMotionTextDiT,
)
from models.raw_motion.llm2vec_cache import CachedLLM2VecTextEncoder
from models.raw_motion.llm2vec_context_cache import (
    LLM2VEC_CONTEXT_CACHE_FORMAT,
    LLM2VecContextMemmapCache,
)
from tools.hy273_runtime_text_encoding import (
    encode_missing_text_rows,
    register_runtime_text_rows,
)
from train_hy273_unified_actor import _is_adaptation_parameter


ENCODER_ID = "context-xattn-test-encoder"
PROMPT_VERSION = "context-xattn-test-prompt"
PROFILE = "hytext_absolute_motion_v1"
ROWS = (
    ("short", np.arange(12, dtype=np.float32).reshape(2, 6)),
    ("long", np.arange(24, dtype=np.float32).reshape(4, 6) + 100.0),
    ("", np.empty((0, 6), dtype=np.float32)),
)


def _key(text: str) -> str:
    return hytext_profile_key(
        text,
        PROFILE,
        encoder_identity=ENCODER_ID,
        prompt_template_version=PROMPT_VERSION,
    )


def _build_global_cache(root: Path) -> None:
    shard = root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    sentence = np.stack(
        [
            values.mean(axis=0) if len(values) else np.zeros(6)
            for _, values in ROWS
        ],
        axis=0,
    )[:, None]
    np.save(shard / "ctxt.npy", sentence.astype(np.float16))
    np.save(shard / "vtxt.npy", np.zeros((len(ROWS), 1, 1), np.float16))
    np.save(shard / "ctxt_len.npy", np.ones(len(ROWS), np.int16))
    index = {
        _key(text): {
            "shard": "shard_00000",
            "row": row,
            "text": text,
            "profile": PROFILE,
        }
        for row, (text, _) in enumerate(ROWS)
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
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
                "model_dtype": "bf16",
                "storage_dtype": "fp16",
            }
        ),
        encoding="utf-8",
    )


def _build_context_cache(root: Path) -> None:
    index = {}
    for shard_id, shard_rows in enumerate((ROWS[:1], ROWS[1:])):
        shard_name = f"shard_{shard_id:05d}"
        shard = root / "shards" / shard_name
        shard.mkdir(parents=True)
        nonempty = [values for _, values in shard_rows if len(values)]
        packed = (
            np.concatenate(nonempty, axis=0)
            if nonempty
            else np.empty((0, 6), dtype=np.float32)
        )
        np.save(shard / "tokens.npy", packed.astype(np.float16))
        offset = 0
        for text, values in shard_rows:
            index[_key(text)] = {
                "shard": shard_name,
                "offset": offset,
                "length": len(values),
                "text": text,
                "profile": PROFILE,
            }
            offset += len(values)
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": LLM2VEC_CONTEXT_CACHE_FORMAT,
                "encoder_identity": ENCODER_ID,
                "prompt_template_version": PROMPT_VERSION,
                "embedding_dim": 6,
                "encoding_batch_size": 1,
                "model_dtype": "bf16",
                "storage_dtype": "fp16",
                "pooling_mode": "mean",
                "skip_instruction": True,
                "num_texts": len(ROWS),
            }
        ),
        encoding="utf-8",
    )


def test_ragged_context_cache_preserves_order_across_shards(
    tmp_path: Path,
) -> None:
    _build_context_cache(tmp_path)
    cache = LLM2VecContextMemmapCache(
        tmp_path,
        embedding_dim=6,
        max_open_shards=1,
    )
    tokens, lengths = cache.lookup_rows(
        ["long", "", "short"],
        [PROFILE, PROFILE, PROFILE],
    )
    assert tokens.shape == (3, 4, 6)
    assert lengths.tolist() == [4, 0, 2]
    torch.testing.assert_close(
        tokens[0, :4],
        torch.from_numpy(ROWS[1][1]).to(torch.float16),
    )
    assert torch.count_nonzero(tokens[1]) == 0
    torch.testing.assert_close(
        tokens[2, :2],
        torch.from_numpy(ROWS[0][1]).to(torch.float16),
    )
    assert torch.count_nonzero(tokens[2, 2:]) == 0


def test_encoder_returns_variable_local_memory_and_dropout(
    tmp_path: Path,
) -> None:
    global_cache = tmp_path / "global"
    context_cache = tmp_path / "context"
    _build_global_cache(global_cache)
    _build_context_cache(context_cache)
    encoder = CachedLLM2VecTextEncoder(
        hidden_dim=8,
        cache_dir=global_cache,
        embedding_dim=6,
        context_cache_dir=context_cache,
    )
    condition = encoder(
        ["short", "long"],
        profiles=[PROFILE, PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition.tokens.shape == (2, 1, 8)
    assert condition.local_tokens is not None
    assert condition.local_tokens.shape == (2, 4, 8)
    assert condition.local_padding_mask is not None
    assert condition.local_padding_mask.tolist() == [
        [False, False, True, True],
        [False, False, False, False],
    ]
    assert torch.count_nonzero(condition.local_tokens[0, 2:]) == 0

    dropped = encoder(
        ["short", "long"],
        profiles=[PROFILE, PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
        force_drop=True,
    )
    assert dropped.local_padding_mask is not None
    assert bool(dropped.local_padding_mask.all())
    assert dropped.local_tokens is not None
    assert torch.count_nonzero(dropped.local_tokens) == 0


def test_sentence_plus_context_returns_one_main_variable_length_sequence(
    tmp_path: Path,
) -> None:
    global_cache = tmp_path / "global"
    context_cache = tmp_path / "context"
    _build_global_cache(global_cache)
    _build_context_cache(context_cache)
    encoder = CachedLLM2VecTextEncoder(
        hidden_dim=8,
        cache_dir=global_cache,
        embedding_dim=6,
        context_cache_dir=context_cache,
        token_sequence_mode="sentence_plus_context",
    )
    condition = encoder(
        ["short", "long"],
        profiles=[PROFILE, PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition.tokens.shape == (2, 5, 8)
    assert condition.padding_mask.tolist() == [
        [False, False, False, True, True],
        [False, False, False, False, False],
    ]
    assert condition.local_tokens is None
    assert condition.local_padding_mask is None


def test_runtime_context_registers_novel_prompt_and_empty_cfg_branch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    global_cache = tmp_path / "global"
    context_cache = tmp_path / "context"
    _build_global_cache(global_cache)
    _build_context_cache(context_cache)
    encoder = CachedLLM2VecTextEncoder(
        hidden_dim=8,
        cache_dir=global_cache,
        embedding_dim=6,
        context_cache_dir=context_cache,
    )
    contextual = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6)

    def fake_encode(_cache, _context_cache, pairs, _device):
        assert pairs == [("novel prompt", PROFILE)]
        return (
            torch.zeros(1, 1, 1),
            contextual.mean(dim=1, keepdim=True),
            torch.ones(1, dtype=torch.long),
            contextual,
            torch.tensor([3], dtype=torch.long),
        )

    monkeypatch.setattr(
        "tools.hy273_runtime_text_encoding._encode_llm2vec_with_context",
        fake_encode,
    )
    rows = encode_missing_text_rows(
        encoder.cache,
        ["novel prompt", ""],
        [PROFILE, PROFILE],
        torch.device("cpu"),
        context_cache=encoder.context_cache,
    )
    assert rows.count == 1
    assert rows.contextual is not None
    assert rows.contextual_lengths is not None
    assert register_runtime_text_rows(
        encoder.cache,
        rows,
        context_cache=encoder.context_cache,
    ) == 1

    condition = encoder(
        ["novel prompt", ""],
        profiles=[PROFILE, PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert condition.local_tokens is not None
    assert condition.local_padding_mask is not None
    assert condition.local_padding_mask.tolist() == [
        [False, False, False],
        [True, True, True],
    ]
    assert torch.count_nonzero(condition.local_tokens[0]) > 0
    assert torch.count_nonzero(condition.local_tokens[1]) == 0


def test_open_gate_backpropagates_into_context_projection(
    tmp_path: Path,
) -> None:
    global_cache = tmp_path / "global"
    context_cache = tmp_path / "context"
    _build_global_cache(global_cache)
    _build_context_cache(context_cache)
    encoder = CachedLLM2VecTextEncoder(
        hidden_dim=16,
        cache_dir=global_cache,
        embedding_dim=6,
        context_cache_dir=context_cache,
    )
    condition = encoder(
        ["short", "long"],
        profiles=[PROFILE, PROFILE],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    model = FrameMotionTextDiT(
        hidden_size=16,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        local_text_cross_attention=True,
    )
    model.local_text_gates[0].data.fill_(0.1)
    output = model(
        torch.randn(2, 4, 16),
        condition.tokens,
        torch.randn(2, 16),
        torch.ones(2, 4, dtype=torch.bool),
        condition.padding_mask,
        torch.arange(4).view(1, 4, 1).expand(2, -1, -1),
        local_text=condition.local_tokens,
        local_text_padding_mask=condition.local_padding_mask,
    )
    output.square().mean().backward()
    assert encoder.local_proj is not None
    projection = encoder.local_proj[1]
    assert isinstance(projection, torch.nn.Linear)
    assert projection.weight.grad is not None
    assert torch.count_nonzero(projection.weight.grad) > 0


def _backbone(local: bool) -> FrameMotionTextDiT:
    return FrameMotionTextDiT(
        hidden_size=16,
        num_heads=4,
        depth_double=2,
        depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        local_text_cross_attention=local,
    )


def _backbone_inputs() -> dict[str, torch.Tensor]:
    return {
        "motion": torch.randn(2, 5, 16),
        "text": torch.randn(2, 1, 16),
        "cond": torch.randn(2, 16),
        "motion_valid": torch.tensor(
            [
                [True, True, True, True, True],
                [True, True, True, False, False],
            ]
        ),
        "text_padding_mask": torch.zeros(2, 1, dtype=torch.bool),
        "motion_pos_ids": torch.arange(5).view(1, 5, 1).expand(2, -1, -1),
    }


def test_zero_gate_is_exact_identity_and_first_step_trains_gate() -> None:
    torch.manual_seed(31)
    baseline = _backbone(local=False).eval()
    torch.manual_seed(31)
    fulltext = _backbone(local=True).eval()
    full_state = fulltext.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(
            full_state[name],
            value,
            rtol=0.0,
            atol=0.0,
        )
    inputs = _backbone_inputs()
    local = torch.randn(2, 4, 16, requires_grad=True)
    local_padding = torch.tensor(
        [
            [False, False, True, True],
            [False, False, False, False],
        ]
    )
    expected = baseline(**inputs)
    actual = fulltext(
        **inputs,
        local_text=local,
        local_text_padding_mask=local_padding,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    actual.square().mean().backward()
    assert all(gate.grad is not None for gate in fulltext.local_text_gates)
    assert any(
        float(gate.grad.abs()) > 0.0
        for gate in fulltext.local_text_gates
        if gate.grad is not None
    )
    assert local.grad is not None
    assert torch.count_nonzero(local.grad) == 0


def test_open_gate_changes_output_and_backpropagates_to_local_tokens() -> None:
    torch.manual_seed(41)
    model = _backbone(local=True).eval()
    for gate in model.local_text_gates:
        gate.data.fill_(0.2)
    inputs = _backbone_inputs()
    local = torch.randn(2, 3, 16, requires_grad=True)
    local_padding = torch.tensor(
        [[False, False, True], [False, False, False]]
    )
    output = model(
        **inputs,
        local_text=local,
        local_text_padding_mask=local_padding,
    )
    no_local_effect = _backbone(local=False)
    no_local_effect.load_state_dict(
        {
            name: value
            for name, value in model.state_dict().items()
            if name in no_local_effect.state_dict()
        }
    )
    baseline = no_local_effect.eval()(**inputs)
    assert not torch.allclose(output, baseline)
    output.square().mean().backward()
    assert local.grad is not None
    assert torch.count_nonzero(local.grad) > 0


def test_local_text_parameters_are_stage_a_base_parameters() -> None:
    assert not _is_adaptation_parameter(
        "text_encoder.local_proj.1.weight"
    )
    assert not _is_adaptation_parameter(
        "root_backbone.local_text_gates.0"
    )


def test_actor_backbone_handles_one_and_two_actors_with_local_memory() -> None:
    torch.manual_seed(53)
    model = ActorExchangeFrameMotionTextDiT(
        hidden_size=16,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        actor_exchange_dim=8,
        actor_exchange_heads=2,
        local_text_cross_attention=True,
    )
    model.local_text_gates[0].data.fill_(0.1)
    frames = 4
    for actors in (1, 2):
        scenes = 2
        flat_batch = scenes * actors
        motion = torch.randn(flat_batch, frames, 16, requires_grad=True)
        text = torch.randn(flat_batch, 1, 16)
        cond = torch.randn(flat_batch, 16)
        valid = torch.ones(flat_batch, frames, dtype=torch.bool)
        pos = torch.arange(frames).view(1, frames, 1).expand(
            flat_batch, -1, -1
        )
        scene_local = torch.randn(scenes, 3, 16)
        local = scene_local.repeat_interleave(actors, dim=0)
        local_padding = torch.tensor(
            [[False, False, True], [False, False, False]]
        ).repeat_interleave(actors, dim=0)
        output = model(
            motion,
            text,
            cond,
            valid,
            torch.zeros(flat_batch, 1, dtype=torch.bool),
            pos,
            local_text=local,
            local_text_padding_mask=local_padding,
            scene_batch_size=scenes,
            actor_count=actors,
        )
        assert output.shape == motion.shape
        output.square().mean().backward()
        assert motion.grad is not None
        model.zero_grad(set_to_none=True)


def test_two_actor_local_text_path_is_actor_swap_equivariant() -> None:
    torch.manual_seed(59)
    model = ActorExchangeFrameMotionTextDiT(
        hidden_size=16,
        num_heads=4,
        depth_double=1,
        depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        actor_exchange_dim=8,
        actor_exchange_heads=2,
        local_text_cross_attention=True,
    ).eval()
    model.local_text_gates[0].data.fill_(0.15)
    torch.nn.init.normal_(model.double_actor_exchange[0].up.weight, std=0.05)
    torch.nn.init.normal_(model.single_actor_exchange[0].up.weight, std=0.05)
    scenes, actors, frames = 2, 2, 4
    motion = torch.randn(scenes, actors, frames, 16)
    text = torch.randn(scenes, 1, 16)
    cond = torch.randn(scenes, 16)
    valid = torch.ones(scenes, actors, frames, dtype=torch.bool)
    local = torch.randn(scenes, 3, 16)
    local_padding = torch.tensor(
        [[False, False, True], [False, False, False]]
    )

    def run(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(scenes * actors, frames, 16)
        flat_valid = mask.reshape(scenes * actors, frames)
        return model(
            flat,
            text.repeat_interleave(actors, dim=0),
            cond.repeat_interleave(actors, dim=0),
            flat_valid,
            torch.zeros(scenes * actors, 1, dtype=torch.bool),
            torch.arange(frames).view(1, frames, 1).expand(
                scenes * actors, -1, -1
            ),
            local_text=local.repeat_interleave(actors, dim=0),
            local_text_padding_mask=local_padding.repeat_interleave(
                actors, dim=0
            ),
            scene_batch_size=scenes,
            actor_count=actors,
        ).reshape(scenes, actors, frames, 16)

    baseline = run(motion, valid)
    swapped = run(motion.flip(1), valid.flip(1))
    torch.testing.assert_close(
        swapped,
        baseline.flip(1),
        rtol=1e-5,
        atol=1e-6,
    )
