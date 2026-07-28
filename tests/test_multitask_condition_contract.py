from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.raw_motion.hy273_multitask_condition import (
    ABSOLUTE_TEXT_PROFILE,
    RELATIVE_EDIT_TEXT_PROFILE,
    CapabilityId,
    ConditionBatch,
    FramePolicy,
    SourceRole,
    TargetOp,
    TaskId,
    TrainStream,
    make_absent_condition,
)
from models.raw_motion.hy273_slices import DIM_HY273


def _edit_condition(batch: int = 2, source_frames: int = 7, target_frames: int = 5):
    target_lengths = torch.tensor([target_frames, target_frames - 1], dtype=torch.long)
    source_lengths = torch.tensor([[source_frames], [source_frames - 2]], dtype=torch.long)
    target_valid = torch.arange(target_frames)[None] < target_lengths[:, None]
    source_valid = torch.arange(source_frames)[None, None] < source_lengths[..., None]
    target_op = torch.full((batch, target_frames), int(TargetOp.PRESERVE), dtype=torch.long)
    target_op[target_valid] = int(TargetOp.EDIT)
    source = torch.randn(batch, 1, source_frames, DIM_HY273)
    source[..., 269:273] = 0.0
    source = torch.where(source_valid[..., None], source, torch.zeros_like(source))
    return ConditionBatch(
        train_stream_id=torch.full((batch,), int(TrainStream.MOTION_EDIT), dtype=torch.long),
        task_id=torch.full((batch,), int(TaskId.EDIT), dtype=torch.long),
        capability_id=torch.full((batch,), int(CapabilityId.MOTION_EDIT), dtype=torch.long),
        text_encoding_profile=(RELATIVE_EDIT_TEXT_PROFILE,) * batch,
        target_valid=target_valid,
        target_op_id=target_op,
        source_motion=source,
        source_present=torch.ones(batch, 1, dtype=torch.bool),
        source_time_valid=source_valid,
        source_value_mask=source_valid[..., None].expand(-1, -1, -1, DIM_HY273).clone(),
        source_role_id=torch.full((batch, 1), int(SourceRole.SELF), dtype=torch.long),
        source_native_lengths=source_lengths,
        requested_target_len=target_lengths,
        frame_gauge_dir=torch.tensor([[1.0, 0.0]]).expand(batch, -1).clone(),
        frame_policy_id=torch.full(
            (batch,), int(FramePolicy.INDEPENDENT_SEQUENCE), dtype=torch.long
        ),
    )


def test_absent_sentinel_is_finite_and_valid():
    condition = make_absent_condition(batch_size=3, target_frames=8)
    condition.validate()
    assert condition.source_motion.shape == (3, 1, 1, DIM_HY273)
    assert torch.count_nonzero(condition.source_motion) == 0
    assert condition.text_encoding_profile == (ABSOLUTE_TEXT_PROFILE,) * 3


def test_edit_contract_and_device_transfer():
    condition = _edit_condition()
    condition.validate()
    moved = condition.to("cpu")
    moved.validate()
    assert moved.source_native_lengths.tolist() == [[7], [5]]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda c: replace(c, text_encoding_profile=(ABSOLUTE_TEXT_PROFILE,) * c.batch_size),
            "relative-edit",
        ),
        (
            lambda c: replace(c, task_id=torch.zeros_like(c.task_id)),
            "GENERATE samples cannot carry source",
        ),
        (
            lambda c: replace(
                c,
                source_time_valid=torch.ones_like(c.source_time_valid),
            ),
            "source_time_valid",
        ),
        (
            lambda c: replace(
                c,
                source_native_lengths=torch.zeros_like(c.source_native_lengths),
                source_time_valid=torch.zeros_like(c.source_time_valid),
                source_value_mask=torch.zeros_like(c.source_value_mask),
            ),
            "positive native length",
        ),
        (
            lambda c: replace(
                c,
                source_value_mask=torch.zeros_like(c.source_value_mask),
            ),
            "at least one source value",
        ),
        (
            lambda c: replace(
                c,
                source_role_id=torch.full_like(c.source_role_id, int(SourceRole.OTHER_ACTOR)),
            ),
            "SourceRole.SELF",
        ),
        (
            lambda c: replace(
                c,
                frame_policy_id=torch.full_like(c.frame_policy_id, int(FramePolicy.SHARED_WORLD)),
            ),
            "FramePolicy.INDEPENDENT_SEQUENCE",
        ),
        (
            lambda c: replace(
                c,
                source_motion=c.source_motion.clone().index_put(
                    (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([269])),
                    torch.tensor([0.5]),
                ),
            ),
            "binary 0/1",
        ),
        (
            lambda c: replace(
                c,
                frame_gauge_dir=torch.zeros_like(c.frame_gauge_dir),
            ),
            "unit",
        ),
    ],
)
def test_fail_closed_invalid_combinations(mutator, message):
    with pytest.raises(ValueError, match=message):
        mutator(_edit_condition()).validate()


def test_time_map_must_be_monotonic_bounded_and_zero_on_padding():
    condition = _edit_condition()
    mapping = torch.zeros(
        condition.batch_size,
        condition.source_slots,
        condition.target_frames,
    )
    for index in range(condition.batch_size):
        target_len = int(condition.requested_target_len[index])
        source_len = int(condition.source_native_lengths[index, 0])
        mapping[index, 0, :target_len] = torch.linspace(0, source_len - 1, target_len)
    replace(condition, target_to_source_time_map=mapping).validate()

    nonmonotonic = mapping.clone()
    nonmonotonic[0, 0, 2] = 0.0
    with pytest.raises(ValueError, match="monotonic"):
        replace(condition, target_to_source_time_map=nonmonotonic).validate()

    out_of_bounds = mapping.clone()
    out_of_bounds[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="outside source bounds"):
        replace(condition, target_to_source_time_map=out_of_bounds).validate()

    padded = mapping.clone()
    padded[1, 0, -1] = 1.0
    with pytest.raises(ValueError, match="Inactive/padded"):
        replace(condition, target_to_source_time_map=padded).validate()
