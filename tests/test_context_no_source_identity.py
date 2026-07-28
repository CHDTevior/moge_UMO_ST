from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from models.raw_motion.hy273_multitask_condition import (
    RELATIVE_EDIT_TEXT_PROFILE,
    CapabilityId,
    SourceRole,
    TargetOp,
    TaskId,
    TrainStream,
    make_absent_condition,
)
from models.raw_motion.kimodo_context_flow_dit import (
    HY273KimodoContextFlow,
    build_source_token_block,
)
from models.raw_motion.kimodo_like_flow_dit import HY273RedenoiseKimodoLike


def _stats(tmp_path):
    full = tmp_path / "full"
    local = tmp_path / "local"
    full.mkdir()
    local.mkdir()
    np.save(full / "Mean.npy", np.zeros(273, dtype=np.float32))
    np.save(full / "Std.npy", np.ones(273, dtype=np.float32))
    np.save(local / "Mean.npy", np.zeros(4, dtype=np.float32))
    np.save(local / "Std.npy", np.ones(4, dtype=np.float32))
    return str(full), str(local)


def _kwargs(tmp_path):
    full, local = _stats(tmp_path)
    return dict(
        hidden_dim=32,
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
    )


def test_context_model_preserves_legacy_initialization_and_no_source_output(tmp_path):
    kwargs = _kwargs(tmp_path)
    torch.manual_seed(1234)
    legacy = HY273RedenoiseKimodoLike(**kwargs).eval()
    torch.manual_seed(1234)
    context = HY273KimodoContextFlow(**kwargs).eval()

    legacy_state = legacy.state_dict()
    context_state = context.state_dict()
    for key, value in legacy_state.items():
        assert key in context_state
        assert torch.equal(value, context_state[key]), key

    model_in = torch.randn(2, 6, 546)
    timestep = torch.tensor([0.25, 0.75])
    valid = torch.tensor([[1] * 6, [1] * 4 + [0] * 2], dtype=torch.bool)
    condition = make_absent_condition(
        batch_size=2,
        target_frames=6,
        target_lengths=valid.sum(dim=1),
    )
    with torch.no_grad():
        expected = legacy(model_in, timestep, text=["", ""], length_mask=valid)
        details = context(
            model_in,
            timestep,
            text=["", ""],
            length_mask=valid,
            condition=condition,
            return_details=True,
        )
    assert torch.equal(expected, details.prediction)
    assert torch.count_nonzero(details.context_root) == 0
    assert torch.count_nonzero(details.context_body) == 0
    assert not details.context_present.any()


def test_source_token_mode_preserves_source_free_generate_path(tmp_path):
    kwargs = _kwargs(tmp_path)
    torch.manual_seed(4321)
    legacy = HY273RedenoiseKimodoLike(**kwargs).eval()
    torch.manual_seed(4321)
    token_block = HY273KimodoContextFlow(
        **kwargs, source_fusion_mode="token_block"
    ).eval()

    model_in = torch.randn(2, 6, 546)
    timestep = torch.tensor([0.2, 0.8])
    valid = torch.tensor([[1] * 6, [1] * 4 + [0] * 2], dtype=torch.bool)
    condition = make_absent_condition(
        batch_size=2,
        target_frames=6,
        target_lengths=valid.sum(dim=1),
    )
    with torch.no_grad():
        expected = legacy(model_in, timestep, text=["", ""], length_mask=valid)
        actual = token_block(
            model_in,
            timestep,
            text=["", ""],
            length_mask=valid,
            condition=condition,
        )
    assert torch.equal(expected, actual)


def test_source_token_block_layout_and_validity():
    target = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    source = target + 100.0
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    present = torch.tensor([True, False])
    pos = torch.arange(4).view(1, 4, 1).expand(2, 4, 1)

    tokens, token_valid, token_pos, target_slice = build_source_token_block(
        target, source, valid, present, pos
    )

    assert tokens.shape == (2, 9, 3)
    assert token_valid.shape == (2, 9)
    assert token_pos.shape == (2, 9, 1)
    assert torch.equal(tokens[:, :4], source)
    assert torch.count_nonzero(tokens[:, 4]) == 0
    assert torch.equal(tokens[:, target_slice], target)
    assert token_valid[0].tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert token_valid[1].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert torch.equal(token_pos[:, :4], pos)
    assert torch.equal(token_pos[:, target_slice], pos)
    assert token_pos[:, 4].tolist() == [[3], [2]]


def _source_present_edit_condition(
    source: torch.Tensor,
    lengths: torch.Tensor,
):
    batch, source_frames, _ = source.shape
    target_frames = int(lengths.max().item())
    base = make_absent_condition(
        batch_size=batch,
        target_frames=target_frames,
        target_lengths=lengths,
    )
    source_valid = (
        torch.arange(source_frames)[None] < lengths[:, None]
    )
    condition = replace(
        base,
        train_stream_id=torch.full(
            (batch,), int(TrainStream.MOTION_EDIT), dtype=torch.long
        ),
        task_id=torch.full((batch,), int(TaskId.EDIT), dtype=torch.long),
        capability_id=torch.full(
            (batch,), int(CapabilityId.MOTION_EDIT), dtype=torch.long
        ),
        text_encoding_profile=(RELATIVE_EDIT_TEXT_PROFILE,) * batch,
        target_op_id=torch.where(
            base.target_valid,
            torch.full_like(base.target_op_id, int(TargetOp.EDIT)),
            base.target_op_id,
        ),
        source_motion=source[:, None],
        source_present=torch.ones(batch, 1, dtype=torch.bool),
        source_time_valid=source_valid[:, None],
        source_value_mask=source_valid[:, None, :, None].expand(
            batch, 1, source_frames, 273
        ),
        source_role_id=torch.full(
            (batch, 1), int(SourceRole.SELF), dtype=torch.long
        ),
        source_native_lengths=lengths[:, None],
    )
    condition.validate()
    return condition


def test_source_token_output_is_invariant_to_batch_padding(tmp_path):
    torch.manual_seed(2468)
    model = HY273KimodoContextFlow(
        **_kwargs(tmp_path), source_fusion_mode="token_block"
    ).eval()

    short_frames = 3
    long_frames = 6
    short_input = torch.randn(1, short_frames, 546)
    short_source = torch.randn(1, short_frames, 273)
    short_source[..., 269:273] = 0.0
    short_condition = _source_present_edit_condition(
        short_source, torch.tensor([short_frames])
    )

    padded_short_input = torch.zeros(1, long_frames, 546)
    padded_short_input[:, :short_frames] = short_input
    long_input = torch.randn(1, long_frames, 546)
    mixed_input = torch.cat([padded_short_input, long_input], dim=0)
    mixed_valid = torch.tensor(
        [
            [True] * short_frames + [False] * (long_frames - short_frames),
            [True] * long_frames,
        ]
    )
    padded_short_source = torch.zeros(1, long_frames, 273)
    padded_short_source[:, :short_frames] = short_source
    long_source = torch.randn(1, long_frames, 273)
    long_source[..., 269:273] = 0.0
    mixed_condition = _source_present_edit_condition(
        torch.cat([padded_short_source, long_source], dim=0),
        torch.tensor([short_frames, long_frames]),
    )

    with torch.no_grad():
        short_output = model(
            short_input,
            torch.tensor([0.4]),
            text=["edit"],
            length_mask=torch.ones(1, short_frames, dtype=torch.bool),
            condition=short_condition,
        )
        mixed_output = model(
            mixed_input,
            torch.tensor([0.4, 0.7]),
            text=["edit", "edit"],
            length_mask=mixed_valid,
            condition=mixed_condition,
        )

    torch.testing.assert_close(
        short_output,
        mixed_output[:1, :short_frames],
        atol=2e-5,
        rtol=2e-5,
    )


def test_context_constructor_preserves_legacy_rng_state(tmp_path):
    kwargs = _kwargs(tmp_path)
    torch.manual_seed(991)
    HY273RedenoiseKimodoLike(**kwargs)
    expected_cpu_rng = torch.get_rng_state().clone()

    torch.manual_seed(991)
    HY273KimodoContextFlow(**kwargs)
    assert torch.equal(torch.get_rng_state(), expected_cpu_rng)


def test_source_free_edit_keeps_task_and_operation_context(tmp_path):
    model = HY273KimodoContextFlow(**_kwargs(tmp_path)).eval()
    generate = make_absent_condition(batch_size=1, target_frames=6)
    edit = replace(
        generate,
        train_stream_id=torch.tensor([int(TrainStream.MOTION_EDIT)]),
        task_id=torch.tensor([int(TaskId.EDIT)]),
        capability_id=torch.tensor([int(CapabilityId.MOTION_EDIT)]),
        text_encoding_profile=(RELATIVE_EDIT_TEXT_PROFILE,),
        target_op_id=torch.full((1, 6), int(TargetOp.EDIT)),
        source_role_id=torch.full((1, 1), int(SourceRole.NULL)),
    )
    edit.validate(v1_strict=False)

    with torch.no_grad():
        model.source_context.task_embed.weight[int(TaskId.EDIT)].fill_(0.25)
        model.source_context.op_embed.weight[int(TargetOp.EDIT)].fill_(0.5)
        edit_output = model.source_context(edit, target_frames=6, dtype=torch.float32)
        generate_output = model.source_context(
            generate, target_frames=6, dtype=torch.float32
        )

    assert edit_output.context_present.tolist() == [True]
    assert torch.count_nonzero(edit_output.root) > 0
    assert torch.count_nonzero(edit_output.body) > 0
    assert generate_output.context_present.tolist() == [False]
    assert torch.count_nonzero(generate_output.root) == 0
    assert torch.count_nonzero(generate_output.body) == 0
