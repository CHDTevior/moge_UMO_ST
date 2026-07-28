"""Typed condition contract for HY273 multi-capability generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Sequence

import torch

from .hy273_slices import DIM_HY273


ABSOLUTE_TEXT_PROFILE = "hytext_absolute_motion_v1"
RELATIVE_EDIT_TEXT_PROFILE = "hytext_relative_edit_v1"


class TrainStream(IntEnum):
    HML_MIXED = 0
    MOTION_EDIT = 1


class TaskId(IntEnum):
    GENERATE = 0
    EDIT = 1
    REACTION = 2


class CapabilityId(IntEnum):
    T2M = 0
    KIMODO_CONTROL = 1
    MOTION_EDIT = 2
    MOTION_EDIT_CONTROL = 3


class TargetOp(IntEnum):
    PRESERVE = 0
    GENERATE = 1
    EDIT = 2


class SourceRole(IntEnum):
    NULL = 0
    SELF = 1
    OTHER_ACTOR = 2
    SCENE = 3


class TextKind(IntEnum):
    ABSOLUTE_MOTION = 0
    RELATIVE_EDIT = 1
    INTERACTION = 2


class FramePolicy(IntEnum):
    INDEPENDENT_SEQUENCE = 0
    SHARED_WORLD = 1


NUM_TASKS = len(TaskId)
NUM_CAPABILITIES = len(CapabilityId)
NUM_TRAIN_STREAMS = len(TrainStream)
NUM_TARGET_OPS = len(TargetOp)
NUM_SOURCE_ROLES = len(SourceRole)
NUM_FRAME_POLICIES = len(FramePolicy)


def _require_tensor(name: str, value: torch.Tensor, shape: Sequence[int | None]) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != len(shape):
        raise ValueError(f"{name} must have rank {len(shape)}, got {tuple(value.shape)}")
    for axis, expected in enumerate(shape):
        if expected is not None and value.shape[axis] != expected:
            raise ValueError(
                f"{name} axis {axis} must be {expected}, got {tuple(value.shape)}"
            )


def _is_prefix_mask(mask: torch.Tensor, lengths: torch.Tensor) -> bool:
    positions = torch.arange(mask.shape[-1], device=mask.device)
    expected = positions.view(*([1] * (mask.ndim - 1)), -1) < lengths.unsqueeze(-1)
    return bool(torch.equal(mask, expected))


@dataclass(frozen=True)
class ConditionBatch:
    """All non-text conditions for one source-topology-homogeneous batch.

    ``source_motion`` remains in unnormalized physical K273 space. The model owns
    source normalization so inference cannot accidentally use a different scale.
    """

    train_stream_id: torch.Tensor
    task_id: torch.Tensor
    capability_id: torch.Tensor
    text_encoding_profile: tuple[str, ...]
    target_valid: torch.Tensor
    target_op_id: torch.Tensor
    source_motion: torch.Tensor
    source_present: torch.Tensor
    source_time_valid: torch.Tensor
    source_value_mask: torch.Tensor
    source_role_id: torch.Tensor
    source_native_lengths: torch.Tensor
    requested_target_len: torch.Tensor
    frame_gauge_dir: torch.Tensor
    frame_policy_id: torch.Tensor
    ease_physical: torch.Tensor
    ease_present: torch.Tensor
    target_to_source_time_map: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.target_valid.shape[0])

    @property
    def target_frames(self) -> int:
        return int(self.target_valid.shape[1])

    @property
    def source_slots(self) -> int:
        return int(self.source_motion.shape[1])

    @property
    def source_frames(self) -> int:
        return int(self.source_motion.shape[2])

    def to(self, device: torch.device | str) -> "ConditionBatch":
        values = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            values[field_name] = value.to(device) if torch.is_tensor(value) else value
        return replace(self, **values)

    def validate(self, *, max_target_frames: int = 300, v1_strict: bool = True) -> None:
        bsz, target_frames = self.target_valid.shape
        _require_tensor("train_stream_id", self.train_stream_id, (bsz,))
        _require_tensor("task_id", self.task_id, (bsz,))
        _require_tensor("capability_id", self.capability_id, (bsz,))
        _require_tensor("target_op_id", self.target_op_id, (bsz, target_frames))
        _require_tensor("requested_target_len", self.requested_target_len, (bsz,))
        _require_tensor("frame_gauge_dir", self.frame_gauge_dir, (bsz, 2))
        _require_tensor("frame_policy_id", self.frame_policy_id, (bsz,))
        _require_tensor("ease_physical", self.ease_physical, (bsz, 6))
        _require_tensor("ease_present", self.ease_present, (bsz,))
        if len(self.text_encoding_profile) != bsz:
            raise ValueError(
                f"Expected {bsz} text profiles, got {len(self.text_encoding_profile)}"
            )
        _require_tensor(
            "source_motion", self.source_motion, (bsz, None, None, DIM_HY273)
        )
        slots, source_frames = self.source_motion.shape[1:3]
        _require_tensor("source_present", self.source_present, (bsz, slots))
        _require_tensor(
            "source_time_valid", self.source_time_valid, (bsz, slots, source_frames)
        )
        _require_tensor(
            "source_value_mask",
            self.source_value_mask,
            (bsz, slots, source_frames, DIM_HY273),
        )
        _require_tensor("source_role_id", self.source_role_id, (bsz, slots))
        _require_tensor(
            "source_native_lengths", self.source_native_lengths, (bsz, slots)
        )
        if self.target_to_source_time_map is not None:
            _require_tensor(
                "target_to_source_time_map",
                self.target_to_source_time_map,
                (bsz, slots, target_frames),
            )

        bool_fields = {
            "target_valid": self.target_valid,
            "source_present": self.source_present,
            "source_time_valid": self.source_time_valid,
            "source_value_mask": self.source_value_mask,
            "ease_present": self.ease_present,
        }
        for name, value in bool_fields.items():
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must be bool, got {value.dtype}")
        for name, value in {
            "source_motion": self.source_motion,
            "frame_gauge_dir": self.frame_gauge_dir,
            "ease_physical": self.ease_physical,
            "target_to_source_time_map": self.target_to_source_time_map,
        }.items():
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains non-finite values")

        target_lengths = self.requested_target_len.long()
        if bool((target_lengths < 1).any()) or bool((target_lengths > target_frames).any()):
            raise ValueError("requested_target_len must be in [1, padded target length]")
        if bool((target_lengths > int(max_target_frames)).any()):
            raise ValueError(f"requested_target_len exceeds max_target_frames={max_target_frames}")
        if not _is_prefix_mask(self.target_valid, target_lengths):
            raise ValueError("target_valid must be an exact prefix mask")

        source_lengths = self.source_native_lengths.long()
        if bool((source_lengths < 0).any()) or bool((source_lengths > source_frames).any()):
            raise ValueError("source_native_lengths are outside padded source bounds")
        if not _is_prefix_mask(self.source_time_valid, source_lengths):
            raise ValueError("source_time_valid must match source_native_lengths exactly")
        if bool((self.source_time_valid & ~self.source_present[..., None]).any()):
            raise ValueError("Absent source slots cannot contain valid source frames")
        if bool((self.source_value_mask & ~self.source_time_valid[..., None]).any()):
            raise ValueError("source_value_mask cannot expose padding")

        absent = ~self.source_present
        if bool((source_lengths[absent] != 0).any()):
            raise ValueError("Absent source slots must have native length zero")
        if bool((self.source_role_id[absent] != int(SourceRole.NULL)).any()):
            raise ValueError("Absent source slots must use SourceRole.NULL")
        absent_values = absent[..., None, None].expand_as(self.source_motion)
        if bool(torch.count_nonzero(self.source_motion.masked_select(absent_values))):
            raise ValueError("Absent source sentinel motion must be exact zero")
        if bool((self.source_role_id[self.source_present] == int(SourceRole.NULL)).any()):
            raise ValueError("Present source slots cannot use SourceRole.NULL")
        if bool((source_lengths[self.source_present] <= 0).any()):
            raise ValueError("Present source slots must have positive native length")
        if bool(torch.count_nonzero(self.ease_physical[~self.ease_present])):
            raise ValueError("Absent Ease sentinel must be exact physical zero")
        visible_by_slot = self.source_value_mask.reshape(bsz, slots, -1).any(dim=-1)
        if bool((self.source_present & ~visible_by_slot).any()):
            raise ValueError("Present source slots must expose at least one source value")
        source_contact_mask = self.source_value_mask[..., 269:273]
        source_contacts = self.source_motion[..., 269:273]
        invalid_source_contacts = source_contact_mask & (source_contacts != 0.0) & (
            source_contacts != 1.0
        )
        if bool(invalid_source_contacts.any()):
            raise ValueError("Visible source contacts must be exact binary 0/1")
        if self.target_to_source_time_map is not None:
            time_map = self.target_to_source_time_map
            inactive_map = (~self.source_present[..., None]) | (~self.target_valid[:, None, :])
            if bool(torch.count_nonzero(time_map.masked_select(inactive_map))):
                raise ValueError("Inactive/padded target_to_source_time_map values must be zero")
            for batch_index in range(bsz):
                target_len = int(target_lengths[batch_index].item())
                for slot_index in range(slots):
                    if not bool(self.source_present[batch_index, slot_index]):
                        continue
                    source_len = int(source_lengths[batch_index, slot_index].item())
                    coords = time_map[batch_index, slot_index, :target_len]
                    if bool(((coords < 0) | (coords > source_len - 1)).any()):
                        raise ValueError("target_to_source_time_map is outside source bounds")
                    if coords.numel() > 1 and bool((coords[1:] < coords[:-1]).any()):
                        raise ValueError("target_to_source_time_map must be monotonic")

        if bool(((self.task_id < 0) | (self.task_id >= NUM_TASKS)).any()):
            raise ValueError("task_id contains an unknown value")
        if bool(
            ((self.train_stream_id < 0) | (self.train_stream_id >= NUM_TRAIN_STREAMS)).any()
        ):
            raise ValueError("train_stream_id contains an unknown value")
        if bool(((self.capability_id < 0) | (self.capability_id >= NUM_CAPABILITIES)).any()):
            raise ValueError("capability_id contains an unknown value")
        if bool(((self.target_op_id < 0) | (self.target_op_id >= NUM_TARGET_OPS)).any()):
            raise ValueError("target_op_id contains an unknown value")
        if bool(
            ((self.source_role_id < 0) | (self.source_role_id >= NUM_SOURCE_ROLES)).any()
        ):
            raise ValueError("source_role_id contains an unknown value")
        if bool(
            ((self.frame_policy_id < 0) | (self.frame_policy_id >= NUM_FRAME_POLICIES)).any()
        ):
            raise ValueError("frame_policy_id contains an unknown value")
        gauge_norm = torch.linalg.vector_norm(self.frame_gauge_dir.float(), dim=-1)
        if not bool(torch.allclose(gauge_norm, torch.ones_like(gauge_norm), atol=1e-5, rtol=1e-5)):
            raise ValueError("frame_gauge_dir must be a unit [cos,sin] direction")

        valid_ops = self.target_op_id.masked_select(self.target_valid)
        padding_ops = self.target_op_id.masked_select(~self.target_valid)
        if bool((padding_ops != int(TargetOp.PRESERVE)).any()):
            raise ValueError("Target padding must use TargetOp.PRESERVE")

        for index in range(bsz):
            stream = TrainStream(int(self.train_stream_id[index]))
            task = TaskId(int(self.task_id[index]))
            capability = CapabilityId(int(self.capability_id[index]))
            has_source = bool(self.source_present[index].any())
            profile = self.text_encoding_profile[index]
            ops = valid_ops.new_empty(0)
            ops = self.target_op_id[index][self.target_valid[index]]

            if task == TaskId.GENERATE:
                if has_source and v1_strict:
                    raise ValueError("v1 GENERATE samples cannot carry source context")
                if bool((ops != int(TargetOp.GENERATE)).any()):
                    raise ValueError("GENERATE samples require TargetOp.GENERATE")
                if profile != ABSOLUTE_TEXT_PROFILE:
                    raise ValueError("GENERATE samples require the absolute-motion text profile")
                if stream != TrainStream.HML_MIXED:
                    raise ValueError("GENERATE samples must come from HML_MIXED")
                if capability not in {CapabilityId.T2M, CapabilityId.KIMODO_CONTROL}:
                    raise ValueError("GENERATE capability is inconsistent with HML_MIXED")
            elif task == TaskId.EDIT:
                # Source-free EDIT rows are the task-local text-only and
                # unconditional branches required to train hierarchical CFG;
                # callers must opt into the extended (non-v1-strict) contract.
                if bool(self.ease_present[index]):
                    raise ValueError("EDIT samples cannot carry Ease conditioning")
                if not has_source and v1_strict:
                    raise ValueError("EDIT samples require a source motion")
                if bool((ops != int(TargetOp.EDIT)).any()):
                    raise ValueError("EDIT samples require TargetOp.EDIT")
                if profile != RELATIVE_EDIT_TEXT_PROFILE:
                    raise ValueError("EDIT samples require the relative-edit text profile")
                if stream != TrainStream.MOTION_EDIT:
                    raise ValueError("EDIT samples must come from MOTION_EDIT")
                if capability not in {
                    CapabilityId.MOTION_EDIT,
                    CapabilityId.MOTION_EDIT_CONTROL,
                }:
                    raise ValueError("EDIT capability is inconsistent with MOTION_EDIT")
                if v1_strict and has_source and bool(
                    (self.source_role_id[index][self.source_present[index]] != int(SourceRole.SELF)).any()
                ):
                    raise ValueError("v1 MotionFix EDIT sources must use SourceRole.SELF")
                if v1_strict and slots != 1:
                    raise ValueError("v1 MotionFix EDIT requires exactly one source slot")
                if v1_strict and FramePolicy(int(self.frame_policy_id[index])) != FramePolicy.INDEPENDENT_SEQUENCE:
                    raise ValueError(
                        "v1 MotionFix EDIT requires FramePolicy.INDEPENDENT_SEQUENCE"
                    )
            elif v1_strict:
                raise ValueError("REACTION is reserved but not trainable in v1")


def make_absent_condition(
    *,
    batch_size: int,
    target_frames: int,
    target_lengths: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    capability: CapabilityId = CapabilityId.T2M,
) -> ConditionBatch:
    """Create the finite K=1/Ts=1 no-source sentinel used by T2M/control."""

    if target_lengths is None:
        target_lengths = torch.full(
            (batch_size,), target_frames, device=device, dtype=torch.long
        )
    else:
        target_lengths = target_lengths.to(device=device, dtype=torch.long)
    positions = torch.arange(target_frames, device=device)[None]
    target_valid = positions < target_lengths[:, None]
    target_op = torch.full(
        (batch_size, target_frames), int(TargetOp.PRESERVE), device=device, dtype=torch.long
    )
    target_op[target_valid] = int(TargetOp.GENERATE)
    gauge = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
    gauge[:, 0] = 1.0
    condition = ConditionBatch(
        train_stream_id=torch.full(
            (batch_size,), int(TrainStream.HML_MIXED), device=device, dtype=torch.long
        ),
        task_id=torch.full(
            (batch_size,), int(TaskId.GENERATE), device=device, dtype=torch.long
        ),
        capability_id=torch.full(
            (batch_size,), int(capability), device=device, dtype=torch.long
        ),
        text_encoding_profile=(ABSOLUTE_TEXT_PROFILE,) * batch_size,
        target_valid=target_valid,
        target_op_id=target_op,
        source_motion=torch.zeros(
            batch_size, 1, 1, DIM_HY273, device=device, dtype=torch.float32
        ),
        source_present=torch.zeros(batch_size, 1, device=device, dtype=torch.bool),
        source_time_valid=torch.zeros(batch_size, 1, 1, device=device, dtype=torch.bool),
        source_value_mask=torch.zeros(
            batch_size, 1, 1, DIM_HY273, device=device, dtype=torch.bool
        ),
        source_role_id=torch.full(
            (batch_size, 1), int(SourceRole.NULL), device=device, dtype=torch.long
        ),
        source_native_lengths=torch.zeros(batch_size, 1, device=device, dtype=torch.long),
        requested_target_len=target_lengths,
        frame_gauge_dir=gauge,
        frame_policy_id=torch.full(
            (batch_size,), int(FramePolicy.INDEPENDENT_SEQUENCE), device=device, dtype=torch.long
        ),
        ease_physical=torch.zeros(
            batch_size, 6, device=device, dtype=torch.float32
        ),
        ease_present=torch.zeros(batch_size, device=device, dtype=torch.bool),
        target_to_source_time_map=torch.zeros(
            batch_size, 1, target_frames, device=device, dtype=torch.float32
        ),
    )
    condition.validate()
    return condition
