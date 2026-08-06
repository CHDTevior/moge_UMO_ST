"""Unified x0-flow sampler for HY273 generation, control, and editing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from models.raw_motion.flow_schedule import (
    LEGACY_SPLIT_CONTACT_PROTOCOL,
    UNIFIED_273_CONTACT_PROTOCOL,
    clean_x0_euler_step,
    make_ode_grid,
    uses_unified_273_flow,
)
from models.raw_motion.hy273_multitask_condition import (
    ABSOLUTE_TEXT_PROFILE,
    INTERACTION_TEXT_PROFILE,
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
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_yaw_rotation
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    CONT_DIM,
    DIM_HY273,
    HEADING_SLICE,
    ROOT_SLICE,
)
from train_hy273_multitask import (
    create_model_from_checkpoint as create_legacy_model_from_checkpoint,
    load_config,
    validate_frozen_contract,
)
from train_hy273_unified_actor import (
    CHECKPOINT_FORMAT as UNIFIED_ACTOR_CHECKPOINT_FORMAT,
    create_model as create_unified_actor_model,
    validate_config as validate_unified_actor_config,
)
from tools.hy273_runtime_text_encoding import (
    encode_missing_text_rows,
    register_runtime_text_rows,
)


SAMPLING_PROTOCOL_VERSION = "hy273_multitask_layered_cfg_x0_v2"
CONTACT_PROTOCOL_VERSION = "hy273_multitask_selected_full_condition_logits_v1"
UNIFIED_CONTACT_PROTOCOL_VERSION = "hy273_unified_273_clean_flow_v1"
EDIT_SOURCE_BASELINE_MODES = ("learned", "exact")


@dataclass
class MultitaskODESampleOutput:
    raw_motion: torch.Tensor
    exact_clamped_motion: torch.Tensor
    final_clean_prediction: torch.Tensor
    final_branch_predictions: dict[str, torch.Tensor]
    branch_names: tuple[str, ...]
    protocol: dict[str, object]


@dataclass(frozen=True)
class ReactionGauge:
    """Invertible source-centered frame used by Reaction training and sampling."""

    source_motion: torch.Tensor
    frame_gauge_dir: torch.Tensor
    root_anchor_xz: torch.Tensor
    yaw_delta: torch.Tensor


def prepare_reaction_source(
    source_motion: torch.Tensor,
    *,
    heading_rad: torch.Tensor | float | None = None,
) -> ReactionGauge:
    """Center the observed actor and optionally rotate it to a requested heading."""

    if source_motion.ndim == 2:
        source_motion = source_motion.unsqueeze(0)
    if source_motion.ndim != 3 or source_motion.shape[-1] != DIM_HY273:
        raise ValueError(f"source_motion must be [B,T,{DIM_HY273}]")
    source = source_motion.float().clone()
    if source.shape[1] < 1:
        raise ValueError("Reaction source motion cannot be empty")
    anchor = source[:, 0, [ROOT_SLICE.start, ROOT_SLICE.start + 2]].clone()
    source[..., ROOT_SLICE.start] -= anchor[:, 0, None]
    source[..., ROOT_SLICE.start + 2] -= anchor[:, 1, None]
    current = torch.atan2(
        source[:, 0, HEADING_SLICE.start + 1],
        source[:, 0, HEADING_SLICE.start],
    )
    if heading_rad is None:
        target = current
    else:
        target = torch.as_tensor(
            heading_rad,
            device=source.device,
            dtype=source.dtype,
        )
        if target.ndim == 0:
            target = target.expand(source.shape[0])
        if target.shape != (source.shape[0],):
            raise ValueError("heading_rad must be scalar or have shape [B]")
    delta = target - current
    source = apply_yaw_rotation(source, delta)
    return ReactionGauge(
        source_motion=source,
        frame_gauge_dir=source[:, 0, HEADING_SLICE].clone(),
        root_anchor_xz=anchor,
        yaw_delta=delta,
    )


def restore_reaction_world(
    motion: torch.Tensor,
    gauge: ReactionGauge,
) -> torch.Tensor:
    """Map a generated reactor from the training gauge back to source world space."""

    if motion.ndim != 3 or motion.shape[0] != gauge.source_motion.shape[0]:
        raise ValueError("Reaction output must have shape [B,T,273] with matching B")
    restored = apply_yaw_rotation(motion, -gauge.yaw_delta.to(motion.device))
    anchor = gauge.root_anchor_xz.to(device=restored.device, dtype=restored.dtype)
    restored[..., ROOT_SLICE.start] += anchor[:, 0, None]
    restored[..., ROOT_SLICE.start + 2] += anchor[:, 1, None]
    return restored


def apply_reaction_gauge(
    motion: torch.Tensor,
    gauge: ReactionGauge,
) -> torch.Tensor:
    """Map target-side Reaction observations into the source-centered gauge."""

    if (
        motion.ndim != 3
        or motion.shape[0] != gauge.source_motion.shape[0]
        or motion.shape[-1] != DIM_HY273
    ):
        raise ValueError(
            "Reaction target observation must be [B,T,273] with matching B"
        )
    transformed = motion.float().clone()
    anchor = gauge.root_anchor_xz.to(
        device=transformed.device, dtype=transformed.dtype
    )
    transformed[..., ROOT_SLICE.start] -= anchor[:, 0, None]
    transformed[..., ROOT_SLICE.start + 2] -= anchor[:, 1, None]
    return apply_yaw_rotation(
        transformed, gauge.yaw_delta.to(device=transformed.device)
    )


def restore_reaction_sample_output(
    output: MultitaskODESampleOutput,
    gauge: ReactionGauge,
) -> MultitaskODESampleOutput:
    """Restore every public Reaction output and record the applied gauge."""

    output.raw_motion = restore_reaction_world(output.raw_motion, gauge)
    output.exact_clamped_motion = restore_reaction_world(
        output.exact_clamped_motion, gauge
    )
    output.final_clean_prediction = restore_reaction_world(
        output.final_clean_prediction, gauge
    )
    output.final_branch_predictions = {
        name: restore_reaction_world(value, gauge)
        for name, value in output.final_branch_predictions.items()
    }
    output.protocol.update(
        {
            "reaction_source_gauge": "source_root_centered_then_shared_yaw",
            "reaction_output_frame": "restored_source_world",
            "reaction_yaw_delta_rad": [
                float(value) for value in gauge.yaw_delta.detach().cpu()
            ],
            "reaction_root_anchor_xz": gauge.root_anchor_xz.detach().cpu().tolist(),
        }
    )
    return output


def create_model_from_checkpoint(
    checkpoint: dict[str, object],
) -> torch.nn.Module:
    if checkpoint.get("format") == UNIFIED_ACTOR_CHECKPOINT_FORMAT:
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise ValueError("Unified actor checkpoint is missing its config")
        validate_unified_actor_config(config)
        return create_unified_actor_model(config)
    return create_legacy_model_from_checkpoint(checkpoint)


def validate_sampling_mode_checkpoint(
    mode: str,
    checkpoint: Mapping[str, object],
    config: Mapping[str, object],
    *,
    allow_stage_a_reaction_diagnostic: bool = False,
) -> None:
    """Reject task routes that the checkpoint architecture never trained."""

    data = config.get("data")
    model = config.get("model")
    data = data if isinstance(data, Mapping) else {}
    model = model if isinstance(model, Mapping) else {}
    paired_task = str(data.get("paired_task", "interaction"))
    if mode in {"reaction", "reaction_control"}:
        if checkpoint.get("format") != UNIFIED_ACTOR_CHECKPOINT_FORMAT:
            raise RuntimeError("Reaction mode requires a unified checkpoint")
        if paired_task != "reaction":
            raise RuntimeError("Reaction mode requires paired_task=reaction")
        if model.get("source_fusion_mode") != "token_block":
            raise RuntimeError("Reaction mode requires source_fusion_mode=token_block")
        if model.get("text_token_sequence") != "sentence_plus_context":
            raise RuntimeError(
                "Reaction mode requires text_token_sequence=sentence_plus_context"
            )
        next_step = int(checkpoint.get("next_global_step", -1))
        if mode == "reaction_control":
            control = config.get("control")
            control = control if isinstance(control, Mapping) else {}
            if not bool(control.get("enabled", False)) or next_step <= 350_000:
                raise RuntimeError(
                    "Reaction-Control requires a checkpoint trained after 350K in Control Stage C"
                )
        if next_step <= 100_000 and not allow_stage_a_reaction_diagnostic:
            raise RuntimeError(
                "Reaction mode requires a Stage-B checkpoint; use the explicit "
                "diagnostic flag only for connectivity checks"
            )
    elif mode == "interaction" and paired_task != "interaction":
        raise RuntimeError(
            "Two-actor Interaction mode requires an archived paired_task=interaction checkpoint"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalizer_from_checkpoint(
    checkpoint: dict[str, object], device: torch.device
) -> HY273Normalizer:
    state = checkpoint.get("normalizer")
    if not isinstance(state, dict) or not {"mean", "std", "variance_eps"} <= set(state):
        raise RuntimeError("Multitask checkpoint has no complete normalizer state")
    variance = state["variance_eps"]
    variance_eps = float(variance.item()) if torch.is_tensor(variance) else float(variance)
    config = checkpoint.get("config")
    if checkpoint.get("format") == UNIFIED_ACTOR_CHECKPOINT_FORMAT:
        normalization = checkpoint.get("normalization")
        normalize_contacts = bool(
            normalization.get("normalize_contacts", False)
            if isinstance(normalization, dict)
            else False
        )
        if not normalize_contacts:
            raise RuntimeError(
                "Unified actor checkpoint must use normalized 273D contacts"
            )
    else:
        contact_protocol = (
            config.get("flow", {}).get(
                "contact_protocol", LEGACY_SPLIT_CONTACT_PROTOCOL
            )
            if isinstance(config, dict)
            else LEGACY_SPLIT_CONTACT_PROTOCOL
        )
        normalize_contacts = uses_unified_273_flow(str(contact_protocol))
    return HY273Normalizer(
        torch.as_tensor(state["mean"]).reshape(DIM_HY273),
        torch.as_tensor(state["std"]).reshape(DIM_HY273),
        variance_eps=variance_eps,
        normalize_contacts=normalize_contacts,
    ).to(device)


def make_edit_condition(
    source_motion: torch.Tensor,
    *,
    target_lengths: torch.Tensor,
    source_lengths: torch.Tensor | None = None,
    target_frames: int | None = None,
    capability: CapabilityId = CapabilityId.MOTION_EDIT,
    frame_gauge_dir: torch.Tensor | None = None,
) -> ConditionBatch:
    """Create the v1 K=1 MotionFix-style source condition in physical K273."""

    if source_motion.ndim == 2:
        source_motion = source_motion.unsqueeze(0)
    if source_motion.ndim != 3 or source_motion.shape[-1] != DIM_HY273:
        raise ValueError(f"source_motion must be [B,Ts,{DIM_HY273}]")
    source_motion = source_motion.float()
    bsz, source_frames, _ = source_motion.shape
    target_lengths = target_lengths.to(device=source_motion.device, dtype=torch.long)
    if target_lengths.shape != (bsz,):
        raise ValueError("target_lengths must have shape [B]")
    target_frames = int(target_frames or target_lengths.max().item())
    if target_frames < int(target_lengths.max().item()):
        raise ValueError("target_frames cannot be shorter than target_lengths")
    if source_lengths is None:
        source_lengths = torch.full(
            (bsz,), source_frames, device=source_motion.device, dtype=torch.long
        )
    else:
        source_lengths = source_lengths.to(
            device=source_motion.device, dtype=torch.long
        )
        if source_lengths.shape != (bsz,):
            raise ValueError("source_lengths must have shape [B]")
        if bool(((source_lengths < 1) | (source_lengths > source_frames)).any()):
            raise ValueError("source_lengths are outside the padded source bounds")
    source_valid = (
        torch.arange(source_frames, device=source_motion.device)[None]
        < source_lengths[:, None]
    )
    target_valid = (
        torch.arange(target_frames, device=source_motion.device)[None]
        < target_lengths[:, None]
    )
    target_op = torch.full(
        (bsz, target_frames),
        int(TargetOp.PRESERVE),
        device=source_motion.device,
        dtype=torch.long,
    )
    target_op[target_valid] = int(TargetOp.EDIT)
    if frame_gauge_dir is None:
        frame_gauge_dir = torch.zeros(
            bsz, 2, device=source_motion.device, dtype=torch.float32
        )
        frame_gauge_dir[:, 0] = 1.0
    condition = ConditionBatch(
        train_stream_id=torch.full(
            (bsz,), int(TrainStream.MOTION_EDIT), device=source_motion.device, dtype=torch.long
        ),
        task_id=torch.full(
            (bsz,), int(TaskId.EDIT), device=source_motion.device, dtype=torch.long
        ),
        capability_id=torch.full(
            (bsz,), int(capability), device=source_motion.device, dtype=torch.long
        ),
        text_encoding_profile=(RELATIVE_EDIT_TEXT_PROFILE,) * bsz,
        target_valid=target_valid,
        target_op_id=target_op,
        source_motion=source_motion[:, None],
        source_present=torch.ones(bsz, 1, device=source_motion.device, dtype=torch.bool),
        source_time_valid=source_valid[:, None],
        source_value_mask=source_valid[:, None, :, None].expand(
            bsz, 1, source_frames, DIM_HY273
        ),
        source_role_id=torch.full(
            (bsz, 1), int(SourceRole.SELF), device=source_motion.device, dtype=torch.long
        ),
        source_native_lengths=source_lengths[:, None],
        requested_target_len=target_lengths,
        frame_gauge_dir=frame_gauge_dir.to(
            device=source_motion.device, dtype=torch.float32
        ),
        frame_policy_id=torch.full(
            (bsz,),
            int(FramePolicy.INDEPENDENT_SEQUENCE),
            device=source_motion.device,
            dtype=torch.long,
        ),
        ease_physical=torch.zeros(
            bsz, 6, device=source_motion.device, dtype=torch.float32
        ),
        ease_present=torch.zeros(
            bsz, device=source_motion.device, dtype=torch.bool
        ),
        target_to_source_time_map=None,
    )
    condition.validate()
    return condition


def make_reaction_condition(
    source_motion: torch.Tensor,
    *,
    target_lengths: torch.Tensor,
    source_lengths: torch.Tensor | None = None,
    target_frames: int | None = None,
    frame_gauge_dir: torch.Tensor | None = None,
    source_person_index: int | torch.Tensor,
    capability: CapabilityId = CapabilityId.TEXT_REACTION,
) -> ConditionBatch:
    """Create an observed actor condition for one reactor target.

    ``source_person_index`` identifies whether the observed source is the first
    (0) or second (1) person referred to by the pair-level Inter-X caption.
    """

    capability = CapabilityId(capability)
    if capability not in {
        CapabilityId.TEXT_REACTION,
        CapabilityId.REACTION_CONTROL,
    }:
        raise ValueError("Reaction condition requires a Reaction capability")
    base = make_edit_condition(
        source_motion,
        target_lengths=target_lengths,
        source_lengths=source_lengths,
        target_frames=target_frames,
        capability=CapabilityId.MOTION_EDIT,
        frame_gauge_dir=frame_gauge_dir,
    )
    target_op = torch.full_like(base.target_op_id, int(TargetOp.PRESERVE))
    target_op[base.target_valid] = int(TargetOp.GENERATE)
    person_index = torch.as_tensor(
        source_person_index,
        device=base.source_role_id.device,
        dtype=torch.long,
    )
    if person_index.ndim == 0:
        person_index = person_index.expand(base.batch_size)
    if person_index.shape != (base.batch_size,):
        raise ValueError("source_person_index must be scalar or have shape [B]")
    if bool(((person_index < 0) | (person_index > 1)).any()):
        raise ValueError("source_person_index must contain only 0 (P1) or 1 (P2)")
    source_roles = torch.where(
        person_index == 0,
        torch.full_like(person_index, int(SourceRole.OTHER_ACTOR_FIRST_PERSON)),
        torch.full_like(person_index, int(SourceRole.OTHER_ACTOR_SECOND_PERSON)),
    )[:, None]
    condition = replace(
        base,
        train_stream_id=torch.full_like(
            base.train_stream_id, int(TrainStream.REACTION)
        ),
        task_id=torch.full_like(base.task_id, int(TaskId.REACTION)),
        capability_id=torch.full_like(
            base.capability_id, int(capability)
        ),
        text_encoding_profile=(INTERACTION_TEXT_PROFILE,) * base.batch_size,
        target_op_id=target_op,
        source_role_id=source_roles,
        frame_policy_id=torch.full_like(
            base.frame_policy_id, int(FramePolicy.SHARED_WORLD)
        ),
    )
    condition.validate()
    return condition


def make_instruction_only_edit_diagnostic_condition(
    *,
    target_lengths: torch.Tensor,
    target_frames: int | None = None,
    frame_gauge_dir: torch.Tensor | None = None,
) -> ConditionBatch:
    """Build the explicit relative-instruction-only OOD diagnostic route.

    This route is forbidden in training and is accepted by the sampler only when
    its diagnostic flag is set. It keeps EDIT task/op/profile semantics while the
    source context is the exact absent sentinel.
    """

    target_lengths = target_lengths.long()
    batch_size = int(target_lengths.shape[0])
    target_frames = int(target_frames or target_lengths.max().item())
    base = make_absent_condition(
        batch_size=batch_size,
        target_frames=target_frames,
        target_lengths=target_lengths,
        device=target_lengths.device,
        capability=CapabilityId.T2M,
    )
    target_op = torch.full_like(base.target_op_id, int(TargetOp.PRESERVE))
    target_op[base.target_valid] = int(TargetOp.EDIT)
    if frame_gauge_dir is None:
        frame_gauge_dir = base.frame_gauge_dir
    condition = replace(
        base,
        train_stream_id=torch.full_like(
            base.train_stream_id, int(TrainStream.MOTION_EDIT)
        ),
        task_id=torch.full_like(base.task_id, int(TaskId.EDIT)),
        capability_id=torch.full_like(
            base.capability_id, int(CapabilityId.MOTION_EDIT)
        ),
        text_encoding_profile=(RELATIVE_EDIT_TEXT_PROFILE,) * batch_size,
        target_op_id=target_op,
        frame_gauge_dir=frame_gauge_dir.to(dtype=torch.float32),
    )
    condition.validate(v1_strict=False)
    return condition


def _repeat_condition(
    condition: ConditionBatch, count: int, *, v1_strict: bool = True
) -> ConditionBatch:
    values: dict[str, object] = {}
    for field_name in condition.__dataclass_fields__:
        value = getattr(condition, field_name)
        if torch.is_tensor(value):
            values[field_name] = torch.cat([value] * count, dim=0)
        elif value is None:
            values[field_name] = None
        elif field_name == "text_encoding_profile":
            values[field_name] = tuple(value) * count
        else:
            raise TypeError(f"Unsupported ConditionBatch field {field_name}: {type(value)}")
    repeated = replace(condition, **values)
    repeated.validate(v1_strict=v1_strict)
    return repeated


def _clear_source_branch(
    condition: ConditionBatch,
    *,
    branch_index: int,
    branch_count: int,
) -> ConditionBatch:
    """Replace one repeated branch with the explicit source-absent sentinel."""

    if not 0 <= int(branch_index) < int(branch_count):
        raise ValueError("branch_index must identify one repeated condition branch")
    batch_size = condition.batch_size // int(branch_count)
    if batch_size * int(branch_count) != condition.batch_size:
        raise ValueError("Repeated condition batch is not divisible by branch_count")
    start = int(branch_index) * batch_size
    stop = start + batch_size

    def zero_rows(value: torch.Tensor) -> torch.Tensor:
        output = value.clone()
        output[start:stop] = 0
        return output

    source_roles = condition.source_role_id.clone()
    source_roles[start:stop] = int(SourceRole.NULL)
    cleared = replace(
        condition,
        source_motion=zero_rows(condition.source_motion),
        source_present=zero_rows(condition.source_present),
        source_time_valid=zero_rows(condition.source_time_valid),
        source_value_mask=zero_rows(condition.source_value_mask),
        source_role_id=source_roles,
        source_native_lengths=zero_rows(condition.source_native_lengths),
        target_to_source_time_map=(
            None
            if condition.target_to_source_time_map is None
            else zero_rows(condition.target_to_source_time_map)
        ),
    )
    cleared.validate(v1_strict=False)
    return cleared


def _route(
    condition: ConditionBatch,
    hard_mask: torch.Tensor,
    *,
    diagnostic_allow_source_absent_edit: bool = False,
) -> tuple[str, bool]:
    condition.validate(v1_strict=not diagnostic_allow_source_absent_edit)
    task_values = set(int(value) for value in condition.task_id.tolist())
    capability_values = set(int(value) for value in condition.capability_id.tolist())
    if len(task_values) != 1 or len(capability_values) != 1:
        raise ValueError("Sampling batches must be task/capability homogeneous")
    task = TaskId(next(iter(task_values)))
    capability = CapabilityId(next(iter(capability_values)))
    has_control = bool(hard_mask.any().item())
    expected_control = capability in {
        CapabilityId.KIMODO_CONTROL,
        CapabilityId.MOTION_EDIT_CONTROL,
        CapabilityId.REACTION_CONTROL,
    }
    if has_control != expected_control:
        raise ValueError(
            f"Capability/control-mask mismatch: capability={capability.name}, mask={has_control}"
        )
    if task == TaskId.GENERATE:
        if capability not in {CapabilityId.T2M, CapabilityId.KIMODO_CONTROL}:
            raise ValueError("GENERATE route has an editing capability")
        return "generate", has_control
    if task == TaskId.EDIT:
        if capability not in {CapabilityId.MOTION_EDIT, CapabilityId.MOTION_EDIT_CONTROL}:
            raise ValueError("EDIT route has a generation capability")
        return "edit", has_control
    if task == TaskId.REACTION:
        if capability not in {
            CapabilityId.TEXT_REACTION,
            CapabilityId.REACTION_CONTROL,
        }:
            raise ValueError("REACTION route has an invalid capability")
        return "reaction", has_control
    raise ValueError("Two-actor INTERACTION uses its archived sampler")


def _branch_spec(
    route: str,
    has_control: bool,
    texts: list[str],
) -> tuple[tuple[str, ...], list[list[str]], tuple[bool, ...]]:
    empty = [""] * len(texts)
    if route == "generate" and not has_control:
        return ("text", "empty"), [texts, empty], (False, False)
    if route == "generate":
        return (
            ("joint", "text", "control", "empty"),
            [texts, texts, empty, empty],
            (True, False, True, False),
        )
    if route in {"edit", "reaction"} and not has_control:
        return (
            ("empty", "source", "joint"),
            [empty, empty, texts],
            (False, False, False),
        )
    if route == "reaction":
        return (
            ("empty", "source", "joint", "all"),
            [empty, empty, texts, texts],
            (False, False, False, True),
        )
    return (
        ("empty", "source", "edit", "all"),
        [empty, empty, texts, texts],
        (False, False, False, True),
    )


def _guided_prediction(
    branches: dict[str, torch.Tensor],
    *,
    route: str,
    has_control: bool,
    text_cfg_scale: float,
    source_cfg_scale: float,
    edit_cfg_scale: float,
    control_cfg_scale: float,
    cfg_apply_contacts: bool,
) -> torch.Tensor:
    if route == "generate" and not has_control:
        guided = branches["empty"] + float(text_cfg_scale) * (
            branches["text"] - branches["empty"]
        )
        selected = guided if cfg_apply_contacts else branches["text"].clone()
    elif route == "generate":
        guided = (
            branches["empty"]
            + float(text_cfg_scale) * (branches["text"] - branches["empty"])
            + float(control_cfg_scale) * (branches["control"] - branches["empty"])
        )
        selected = guided if cfg_apply_contacts else branches["joint"].clone()
    elif route == "reaction" and not has_control:
        guided = (
            branches["empty"]
            + float(source_cfg_scale)
            * (branches["source"] - branches["empty"])
            + float(text_cfg_scale)
            * (branches["joint"] - branches["source"])
        )
        selected = guided if cfg_apply_contacts else branches["joint"].clone()
    elif route == "reaction":
        guided = (
            branches["empty"]
            + float(source_cfg_scale)
            * (branches["source"] - branches["empty"])
            + float(text_cfg_scale)
            * (branches["joint"] - branches["source"])
            + float(control_cfg_scale)
            * (branches["all"] - branches["joint"])
        )
        selected = guided if cfg_apply_contacts else branches["all"].clone()
    elif not has_control:
        if float(source_cfg_scale) == 1.0 and float(edit_cfg_scale) == 1.0:
            # Unified Edit is trained directly on source+instruction. At unit
            # scales the hierarchical expression is algebraically the joint
            # branch, so use it directly and exclude untrained CFG branches.
            guided = branches["joint"]
        else:
            guided = (
                branches["empty"]
                + float(source_cfg_scale)
                * (branches["source"] - branches["empty"])
                + float(edit_cfg_scale)
                * (branches["joint"] - branches["source"])
            )
        selected = guided if cfg_apply_contacts else branches["joint"].clone()
    else:
        if float(source_cfg_scale) == 1.0 and float(edit_cfg_scale) == 1.0:
            guided = branches["edit"] + float(control_cfg_scale) * (
                branches["all"] - branches["edit"]
            )
        else:
            guided = (
                branches["empty"]
                + float(source_cfg_scale)
                * (branches["source"] - branches["empty"])
                + float(edit_cfg_scale)
                * (branches["edit"] - branches["source"])
                + float(control_cfg_scale)
                * (branches["all"] - branches["edit"])
            )
        selected = guided if cfg_apply_contacts else branches["all"].clone()
    if not cfg_apply_contacts:
        selected[..., :CONT_DIM] = guided[..., :CONT_DIM]
    return selected


def _apply_exact_edit_source_baseline(
    branches: dict[str, torch.Tensor],
    *,
    source_anchor_norm: torch.Tensor,
    texts: list[str],
    has_control: bool,
    unified_273_flow: bool,
) -> None:
    """Replace the learned Edit source branch with the observed clean source x0."""

    source_prediction = branches["source"].clone()
    if unified_273_flow:
        source_prediction.copy_(source_anchor_norm)
    else:
        source_prediction[..., :CONT_DIM] = source_anchor_norm[..., :CONT_DIM]
        source_contacts = source_anchor_norm[..., CONTACT_SLICE].clamp(1e-4, 1.0 - 1e-4)
        source_prediction[..., CONTACT_SLICE] = torch.logit(source_contacts).to(
            dtype=source_prediction.dtype
        )
    branches["source"] = source_prediction

    # An empty edit instruction means preserve. Avoid extrapolating the stale
    # learned empty-text branch away from the exact source anchor.
    instruction_branch = "edit" if has_control else "joint"
    empty_instruction = torch.tensor(
        [not value.strip() for value in texts],
        device=source_prediction.device,
        dtype=torch.bool,
    )
    if bool(empty_instruction.any()):
        mask = empty_instruction[:, None, None]
        branches[instruction_branch] = torch.where(
            mask, source_prediction, branches[instruction_branch]
        )


@torch.no_grad()
def sample_hy273_multitask_ode(
    model: torch.nn.Module,
    normalizer: HY273Normalizer,
    condition: ConditionBatch,
    texts: Iterable[str],
    observed_physical: torch.Tensor,
    hard_mask: torch.Tensor,
    *,
    num_steps: int = 32,
    text_cfg_scale: float | None = None,
    source_cfg_scale: float | None = None,
    edit_cfg_scale: float | None = None,
    control_cfg_scale: float | None = None,
    contact_init: str = "random",
    contact_feedback: str = "blend",
    cfg_apply_contacts: bool | None = None,
    velocity_t_eps: float = 1e-4,
    generator: torch.Generator | None = None,
    initial_unified_noise: torch.Tensor | None = None,
    initial_continuous_noise: torch.Tensor | None = None,
    initial_contact_noise: torch.Tensor | None = None,
    diagnostic_allow_source_absent_edit: bool = False,
    edit_source_baseline: str = "learned",
    edit_source_anchor_physical: torch.Tensor | None = None,
) -> MultitaskODESampleOutput:
    """Integrate one homogeneous capability batch with layered CFG.

    The ODE state is never persistently clamped. Hard observations overwrite only
    the denoiser inputs of branches marked as controlled; the final exact-clamped
    tensor is emitted separately from learned raw adherence.
    """

    if num_steps <= 0 or velocity_t_eps <= 0:
        raise ValueError("num_steps and velocity_t_eps must be positive")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    unified_273_flow = bool(normalizer.normalize_contacts)
    condition = condition.to(device)
    condition.validate(v1_strict=not diagnostic_allow_source_absent_edit)
    texts = list(texts)
    if len(texts) != condition.batch_size:
        raise ValueError("texts must contain one entry per sample")
    observed_physical = observed_physical.to(device=device, dtype=torch.float32)
    hard_mask = hard_mask.to(device=device, dtype=torch.bool)
    expected_shape = (condition.batch_size, condition.target_frames, DIM_HY273)
    if observed_physical.shape != expected_shape or hard_mask.shape != expected_shape:
        raise ValueError(f"observed_physical/hard_mask must have shape {expected_shape}")
    if bool((hard_mask & ~condition.target_valid[..., None]).any()):
        raise ValueError("Hard control mask cannot touch target padding")
    route, has_control = _route(
        condition,
        hard_mask,
        diagnostic_allow_source_absent_edit=diagnostic_allow_source_absent_edit,
    )
    text_cfg_scale = float(
        text_cfg_scale
        if text_cfg_scale is not None
        else 2.0
    )
    source_cfg_scale = float(1.0 if source_cfg_scale is None else source_cfg_scale)
    edit_cfg_scale = float(
        1.0 if edit_cfg_scale is None else edit_cfg_scale
    )
    control_cfg_scale = float(
        2.0 if control_cfg_scale is None else control_cfg_scale
    )
    cfg_apply_contacts = (
        True
        if unified_273_flow
        else bool(
            route == "generate" and has_control
            if cfg_apply_contacts is None
            else cfg_apply_contacts
        )
    )
    branch_names, branch_texts, branch_controlled = _branch_spec(
        route, has_control, texts
    )

    edit_source_baseline = str(edit_source_baseline)
    if edit_source_baseline not in EDIT_SOURCE_BASELINE_MODES:
        raise ValueError(
            f"edit_source_baseline must be one of {EDIT_SOURCE_BASELINE_MODES}"
        )
    if edit_source_baseline == "exact":
        if route != "edit" or diagnostic_allow_source_absent_edit:
            raise ValueError("Exact source baseline requires a source-present Edit route")
        if source_cfg_scale != 1.0:
            raise ValueError("Exact source baseline requires source_cfg_scale=1.0")
        if edit_source_anchor_physical is None:
            raise ValueError("Exact source baseline requires edit_source_anchor_physical")
        if edit_source_anchor_physical.shape != expected_shape:
            raise ValueError(
                "edit_source_anchor_physical must have shape "
                f"{expected_shape}, got {tuple(edit_source_anchor_physical.shape)}"
            )
        if not bool(torch.isfinite(edit_source_anchor_physical).all()):
            raise ValueError("edit_source_anchor_physical contains non-finite values")
        edit_source_anchor_physical = edit_source_anchor_physical.to(
            device=device, dtype=torch.float32
        )
    elif edit_source_anchor_physical is not None:
        raise ValueError(
            "edit_source_anchor_physical is only valid with edit_source_baseline='exact'"
        )

    observed_norm = normalizer.normalize(observed_physical).to(dtype=dtype)
    source_anchor_norm = (
        None
        if edit_source_anchor_physical is None
        else normalizer.normalize(edit_source_anchor_physical).to(dtype=dtype)
    )
    mask_f = hard_mask.to(dtype=dtype)
    bsz, frames, _ = observed_norm.shape
    expected_cont_shape = (bsz, frames, CONT_DIM)
    expected_contact_shape = (bsz, frames, 4)
    expected_unified_shape = (bsz, frames, DIM_HY273)
    if unified_273_flow:
        if initial_continuous_noise is not None or initial_contact_noise is not None:
            raise ValueError(
                "R13 unified flow accepts only initial_unified_noise [B,T,273]; "
                "legacy split noise tensors are forbidden"
            )
        if initial_unified_noise is not None:
            if initial_unified_noise.shape != expected_unified_shape:
                raise ValueError(
                    "initial_unified_noise must have shape "
                    f"{expected_unified_shape}, got {tuple(initial_unified_noise.shape)}"
                )
            state = initial_unified_noise.to(device=device, dtype=dtype).clone()
            initial_noise_source = "provided_unified_273d"
        else:
            state = torch.randn(
                *expected_unified_shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            initial_noise_source = "torch_randn_unified_273d"
        final_clean_norm = state.clone()
    else:
        if initial_unified_noise is not None:
            raise ValueError("initial_unified_noise requires the R13 unified protocol")
        if initial_continuous_noise is None:
            z_cont = torch.randn(
                *expected_cont_shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            if initial_continuous_noise.shape != expected_cont_shape:
                raise ValueError(
                    "initial_continuous_noise must have shape "
                    f"{expected_cont_shape}, got {tuple(initial_continuous_noise.shape)}"
                )
            z_cont = initial_continuous_noise.to(device=device, dtype=dtype).clone()
        if initial_contact_noise is not None:
            if initial_contact_noise.shape != expected_contact_shape:
                raise ValueError(
                    f"initial_contact_noise must have shape {expected_contact_shape}, "
                    f"got {tuple(initial_contact_noise.shape)}"
                )
            contact_noise = initial_contact_noise.to(device=device, dtype=dtype).clone()
        elif contact_init == "random":
            contact_noise = torch.rand(
                bsz, frames, 4, device=device, dtype=dtype, generator=generator
            )
        elif contact_init == "half":
            contact_noise = torch.full(
                (bsz, frames, 4), 0.5, device=device, dtype=dtype
            )
        elif contact_init == "zeros":
            contact_noise = torch.zeros(bsz, frames, 4, device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown contact_init={contact_init!r}")
        if contact_feedback not in {"blend", "prob", "fixed"}:
            raise ValueError(f"Unknown contact_feedback={contact_feedback!r}")
        contact_aux = contact_noise.clone()
        final_clean_norm = torch.cat([z_cont, contact_aux], dim=-1)
    final_branches: dict[str, torch.Tensor] = {}
    grid = make_ode_grid(num_steps, device=device).to(dtype=dtype)

    repeated_condition = _repeat_condition(
        condition,
        len(branch_names),
        v1_strict=not diagnostic_allow_source_absent_edit,
    )
    if route in {"edit", "reaction"}:
        repeated_condition = _clear_source_branch(
            repeated_condition,
            branch_index=branch_names.index("empty"),
            branch_count=len(branch_names),
        )
    branch_text_flat = [text for rows in branch_texts for text in rows]
    branch_valid = condition.target_valid.repeat(len(branch_names), 1)
    branch_c_dir = condition.frame_gauge_dir.to(dtype=dtype).repeat(
        len(branch_names), 1
    )
    zero_mask = torch.zeros_like(mask_f)
    for step_index in range(num_steps):
        t = grid[step_index].expand(bsz)
        dt = grid[step_index + 1] - grid[step_index]
        if not unified_273_flow:
            state = torch.cat([z_cont, contact_aux], dim=-1)
        controlled_state = state * (1.0 - mask_f) + observed_norm * mask_f
        model_inputs = []
        for controlled in branch_controlled:
            branch_state = controlled_state if controlled else state
            branch_mask = mask_f if controlled else zero_mask
            model_inputs.append(torch.cat([branch_state, branch_mask], dim=-1))
        prediction = model(
            torch.cat(model_inputs, dim=0),
            t=t.repeat(len(branch_names)),
            c_dir=branch_c_dir,
            text=branch_text_flat,
            length_mask=branch_valid,
            x_self_cond=None,
            text_drop_prob=0.0,
            condition=repeated_condition,
        )
        chunks = prediction.chunk(len(branch_names), dim=0)
        final_branches = dict(zip(branch_names, chunks))
        if source_anchor_norm is not None:
            _apply_exact_edit_source_baseline(
                final_branches,
                source_anchor_norm=source_anchor_norm,
                texts=texts,
                has_control=has_control,
                unified_273_flow=unified_273_flow,
            )
        selected = _guided_prediction(
            final_branches,
            route=route,
            has_control=has_control,
            text_cfg_scale=text_cfg_scale,
            source_cfg_scale=source_cfg_scale,
            edit_cfg_scale=edit_cfg_scale,
            control_cfg_scale=control_cfg_scale,
            cfg_apply_contacts=cfg_apply_contacts,
        )
        if unified_273_flow:
            final_clean_norm = selected
            state, velocity = clean_x0_euler_step(
                state,
                selected,
                timestep=t,
                dt=dt,
                velocity_t_eps=velocity_t_eps,
            )
        else:
            x0_cont = selected[..., :CONT_DIM]
            contact_prob = torch.sigmoid(selected[..., CONTACT_SLICE])
            final_clean_norm = torch.cat([x0_cont, contact_prob], dim=-1)
            z_cont, velocity = clean_x0_euler_step(
                state[..., :CONT_DIM],
                x0_cont,
                timestep=t,
                dt=dt,
                velocity_t_eps=velocity_t_eps,
            )
            if contact_feedback == "blend":
                t_next = grid[step_index + 1].view(1, 1, 1)
                contact_aux = t_next * contact_prob + (1.0 - t_next) * contact_noise
            elif contact_feedback == "prob":
                contact_aux = contact_prob
            else:
                contact_aux = contact_noise

    final_norm = state if unified_273_flow else torch.cat([z_cont, contact_aux], dim=-1)
    def decode_model_state(value: torch.Tensor) -> torch.Tensor:
        if unified_273_flow:
            physical = normalizer.denormalize(value.float())
            physical[..., CONTACT_SLICE] = (
                physical[..., CONTACT_SLICE] >= 0.5
            ).to(physical.dtype)
            return physical
        return normalizer.denormalize(value.float())

    raw = decode_model_state(final_norm)
    # Clamp in physical space so every requested value, including binary
    # contacts, is bit-exact rather than a normalize/denormalize round trip.
    exact = torch.where(
        hard_mask,
        observed_physical.to(device=raw.device, dtype=raw.dtype),
        raw,
    )
    clean = decode_model_state(final_clean_norm)
    branch_physical = {}
    for name, value in final_branches.items():
        branch_state = (
            value
            if unified_273_flow
            else torch.cat(
                [value[..., :CONT_DIM], torch.sigmoid(value[..., CONTACT_SLICE])],
                dim=-1,
            )
        )
        branch_physical[name] = decode_model_state(branch_state)
    for batch_index, length_value in enumerate(condition.requested_target_len):
        length = int(length_value.item())
        if length < frames:
            for value in (raw, exact, clean, *branch_physical.values()):
                value[batch_index, length:] = value[batch_index, length - 1 : length]
    protocol = {
        "sampling_protocol_version": SAMPLING_PROTOCOL_VERSION,
        "contact_protocol_version": (
            UNIFIED_CONTACT_PROTOCOL_VERSION
            if unified_273_flow
            else CONTACT_PROTOCOL_VERSION
        ),
        "contact_protocol": (
            UNIFIED_273_CONTACT_PROTOCOL
            if unified_273_flow
            else LEGACY_SPLIT_CONTACT_PROTOCOL
        ),
        "route": route,
        "has_control": has_control,
        "source_fusion_mode": str(
            getattr(model, "source_fusion_mode", "additive")
        ),
        "text_global_conditioning": str(
            getattr(model, "text_global_conditioning", "pooled_adaln")
        ),
        "text_fusion_mode": str(
            getattr(model, "text_fusion_mode", "f00")
        ),
        "branch_names": list(branch_names),
        "ode_steps": int(num_steps),
        "ode_state_persistent_clamp": False,
        "text_cfg_scale": float(text_cfg_scale),
        "source_cfg_scale": float(source_cfg_scale),
        "edit_cfg_scale": float(edit_cfg_scale),
        "control_cfg_scale": float(control_cfg_scale),
        "cfg_apply_contacts": bool(cfg_apply_contacts),
        "contact_init": "unified_273d_state" if unified_273_flow else contact_init,
        "contact_feedback": "ode_273d" if unified_273_flow else contact_feedback,
        "initial_noise_source": (
            initial_noise_source if unified_273_flow else "legacy_split_state"
        ),
        "primary_output": "raw_pre_exact_clamp",
        "diagnostic_allow_source_absent_edit": bool(
            diagnostic_allow_source_absent_edit
        ),
        "edit_source_baseline": edit_source_baseline,
    }
    return MultitaskODESampleOutput(
        raw_motion=raw,
        exact_clamped_motion=exact,
        final_clean_prediction=clean,
        final_branch_predictions=branch_physical,
        branch_names=branch_names,
        protocol=protocol,
    )


@torch.no_grad()
def sample_hy273_interaction_ode(
    model: torch.nn.Module,
    normalizer: HY273Normalizer,
    texts: Iterable[str],
    target_lengths: torch.Tensor,
    *,
    num_steps: int = 32,
    cfg_scale: float = 3.5,
    c_dir: torch.Tensor | None = None,
    velocity_t_eps: float = 1e-4,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
) -> MultitaskODESampleOutput:
    """Jointly generate two actors with InterGen-style two-way text CFG."""

    if not normalizer.normalize_contacts:
        raise ValueError("Interaction sampling requires unified 273D flow")
    if num_steps <= 0 or velocity_t_eps <= 0:
        raise ValueError("num_steps and velocity_t_eps must be positive")
    if not torch.isfinite(torch.tensor(float(cfg_scale))):
        raise ValueError("cfg_scale must be finite")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    text_rows = tuple(texts)
    batch = len(text_rows)
    if batch == 0:
        raise ValueError("At least one interaction text is required")
    lengths = target_lengths.to(device=device, dtype=torch.long).reshape(-1)
    if lengths.shape != (batch,) or bool((lengths <= 0).any()):
        raise ValueError("target_lengths must contain one positive length per text")
    frames = int(lengths.max().item())
    if frames > int(getattr(model, "max_frames", frames)):
        raise ValueError(
            f"Requested {frames} frames exceeds model max_frames="
            f"{getattr(model, 'max_frames', 'unknown')}"
        )
    scene_valid = (
        torch.arange(frames, device=device).view(1, 1, frames)
        < lengths.view(batch, 1, 1)
    ).expand(batch, 2, frames)
    expected_shape = (batch, 2, frames, DIM_HY273)
    if initial_noise is None:
        state = torch.randn(
            expected_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        initial_noise_source = "torch_randn_independent_actor_273d"
    else:
        if initial_noise.shape != expected_shape:
            raise ValueError(
                f"initial_noise must have shape {expected_shape}, "
                f"got {tuple(initial_noise.shape)}"
            )
        state = initial_noise.to(device=device, dtype=dtype).clone()
        initial_noise_source = "provided_independent_actor_273d"
    if c_dir is None:
        c_dir = torch.zeros(batch, 2, device=device, dtype=dtype)
        c_dir[:, 0] = 1.0
    else:
        c_dir = c_dir.to(device=device, dtype=dtype)
        if c_dir.shape != (batch, 2):
            raise ValueError("c_dir must have shape [B,2]")
        norm = torch.linalg.vector_norm(c_dir.float(), dim=-1)
        if bool((norm < 1e-6).any()):
            raise ValueError("c_dir rows must be non-zero")
        c_dir = c_dir / norm.to(dtype=dtype).unsqueeze(-1)

    task_id = torch.full(
        (batch,), int(TaskId.INTERACTION), device=device, dtype=torch.long
    )
    zero_mask = torch.zeros_like(state)
    grid = make_ode_grid(num_steps, device=device).to(dtype=dtype)
    final_clean = state.clone()
    final_branches: dict[str, torch.Tensor] = {}
    branch_text = ("",) * batch + text_rows
    branch_profiles = (INTERACTION_TEXT_PROFILE,) * (2 * batch)
    for step_index in range(num_steps):
        t = grid[step_index].expand(batch)
        dt = grid[step_index + 1] - grid[step_index]
        prediction = model(
            torch.cat(
                [
                    torch.cat([state, zero_mask], dim=-1),
                    torch.cat([state, zero_mask], dim=-1),
                ],
                dim=0,
            ),
            t=t.repeat(2),
            c_dir=c_dir.repeat(2, 1),
            text=branch_text,
            length_mask=scene_valid.repeat(2, 1, 1),
            task_id=task_id.repeat(2),
            text_profiles=branch_profiles,
        )
        unconditional, conditional = prediction.chunk(2, dim=0)
        final_branches = {
            "empty": unconditional,
            "text": conditional,
        }
        final_clean = unconditional + float(cfg_scale) * (
            conditional - unconditional
        )
        state, _ = clean_x0_euler_step(
            state,
            final_clean,
            timestep=t,
            dt=dt,
            velocity_t_eps=velocity_t_eps,
        )

    def decode(value: torch.Tensor) -> torch.Tensor:
        physical = normalizer.denormalize(value.float())
        physical[..., CONTACT_SLICE] = (
            physical[..., CONTACT_SLICE] >= 0.5
        ).to(dtype=physical.dtype)
        return physical

    raw = decode(state)
    clean = decode(final_clean)
    branch_physical = {
        name: decode(value) for name, value in final_branches.items()
    }
    for batch_index, length_value in enumerate(lengths):
        length = int(length_value.item())
        if length < frames:
            for value in (raw, clean, *branch_physical.values()):
                value[batch_index, :, length:] = value[
                    batch_index, :, length - 1 : length
                ]
    protocol = {
        "sampling_protocol": "hy273_unified_actor_interaction_cfg_v1",
        "mode": "interaction",
        "ode_steps": int(num_steps),
        "text_cfg_scale": float(cfg_scale),
        "initial_noise_source": initial_noise_source,
        "actor_noise": "independent",
        "scene_timestep_text_c_dir": "shared",
        "contact_feedback": "ode_273d",
    }
    return MultitaskODESampleOutput(
        raw_motion=raw,
        exact_clamped_motion=raw.clone(),
        final_clean_prediction=clean,
        final_branch_predictions=branch_physical,
        branch_names=("empty", "text"),
        protocol=protocol,
    )


def _load_array(path: str, name: str) -> torch.Tensor:
    if not path:
        raise ValueError(f"--{name} is required for this mode")
    value = torch.from_numpy(np.load(path)).float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", default="")
    parser.add_argument("--weight_source", choices=["ema", "raw"], default="ema")
    parser.add_argument(
        "--mode",
        choices=[
            "t2m",
            "control",
            "edit",
            "edit_control",
            "reaction",
            "reaction_control",
            "interaction",
        ],
        required=True,
    )
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument(
        "--target_length",
        type=int,
        default=None,
        help=(
            "Target frames. Defaults to source length for Edit/Reaction and 150 "
            "for source-free generation."
        ),
    )
    parser.add_argument("--source_npy", default="")
    parser.add_argument(
        "--source_fps",
        type=float,
        default=None,
        help="Physical FPS of source_npy. Reaction currently accepts exact 30 FPS K273 only.",
    )
    parser.add_argument(
        "--reaction_source_person_index",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "For Reaction, whether source_npy is the first (0) or second (1) "
            "person referred to by the interaction text. Required in Reaction mode."
        ),
    )
    parser.add_argument("--reaction_clip_id", default="")
    parser.add_argument(
        "--allow_stage_a_reaction_diagnostic",
        action="store_true",
        help="Allow Reaction connectivity diagnostics before its Stage-B training begins.",
    )
    parser.add_argument("--source_anchor_npy", default="")
    parser.add_argument("--observed_npy", default="")
    parser.add_argument("--mask_npy", default="")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--text_cfg_scale", type=float, default=None)
    parser.add_argument("--source_cfg_scale", type=float, default=None)
    parser.add_argument("--edit_cfg_scale", type=float, default=None)
    parser.add_argument(
        "--edit_source_baseline",
        choices=EDIT_SOURCE_BASELINE_MODES,
        default="learned",
    )
    parser.add_argument("--control_cfg_scale", type=float, default=None)
    parser.add_argument(
        "--ease",
        nargs=6,
        type=float,
        default=None,
        metavar=("EIX", "EIY", "EIZ", "EOX", "EOY", "EOZ"),
        help="Physical Ease [E_in_xyz,E_out_xyz] in meters.",
    )
    parser.add_argument("--contact_init", choices=["random", "half", "zeros"], default="random")
    parser.add_argument("--contact_feedback", choices=["blend", "prob", "fixed"], default="blend")
    parser.add_argument(
        "--cfg_apply_contacts", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--heading_deg",
        type=float,
        default=0.0,
        help=(
            "Scene reference direction in degrees. Reaction rotates the observed actor "
            "to this heading during denoising and restores outputs to source world space."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if args.checkpoint_sha256 and sha256_file(checkpoint_path) != args.checkpoint_sha256:
        raise RuntimeError("Checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint does not contain resolved multitask config")
    if checkpoint.get("format") == UNIFIED_ACTOR_CHECKPOINT_FORMAT:
        validate_unified_actor_config(config)
    else:
        validate_frozen_contract(config)
    validate_sampling_mode_checkpoint(
        args.mode,
        checkpoint,
        config,
        allow_stage_a_reaction_diagnostic=args.allow_stage_a_reaction_diagnostic,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model_from_checkpoint(checkpoint).to(device)
    weights = checkpoint["ema"] if args.weight_source == "ema" else checkpoint["model"]
    model.load_state_dict(weights, strict=True)
    model.eval()
    normalizer = normalizer_from_checkpoint(checkpoint, device)

    source = None
    reaction_gauge = None
    if args.mode.startswith("edit") or args.mode in {"reaction", "reaction_control"}:
        source = _load_array(args.source_npy, "source_npy")
        bsz = source.shape[0]
    else:
        if args.mode == "interaction" and not args.text:
            raise ValueError("Interaction mode requires at least one --text")
        bsz = max(len(args.text), 1)
    target_length = (
        int(args.target_length)
        if args.target_length is not None
        else int(source.shape[1])
        if source is not None
        else 150
    )
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if args.mode in {"reaction", "reaction_control"}:
        if source is None:
            raise RuntimeError("Reaction source was not loaded")
        if args.reaction_source_person_index is None:
            raise ValueError(
                "Reaction requires explicit --reaction_source_person_index 0 or 1"
            )
        if args.source_fps is None:
            raise ValueError("Reaction requires explicit --source_fps 30")
        if float(args.source_fps) != 30.0:
            raise ValueError(
                "Reaction source must be exact 30 FPS K273; resample rotations/root "
                "and recompute heading, velocities, and contacts before sampling"
            )
        if target_length != int(source.shape[1]):
            raise ValueError(
                "Reaction uses synchronized Inter-X actor/reactor frames; "
                "target_length must equal the source length"
            )
        reaction_gauge = prepare_reaction_source(
            source,
            heading_rad=float(args.heading_deg) * torch.pi / 180.0,
        )
        source = reaction_gauge.source_motion
    texts = args.text or ([""] * bsz)
    if len(texts) == 1 and bsz > 1:
        texts = texts * bsz
    if len(texts) != bsz:
        raise ValueError("Number of --text values must match the batch size")
    lengths = torch.full((bsz,), target_length, dtype=torch.long)
    controlled = args.mode in {"control", "edit_control", "reaction_control"}
    condition = None
    if args.mode != "interaction":
        capability = {
            "t2m": CapabilityId.T2M,
            "control": CapabilityId.KIMODO_CONTROL,
            "edit": CapabilityId.MOTION_EDIT,
            "edit_control": CapabilityId.MOTION_EDIT_CONTROL,
            "reaction": CapabilityId.TEXT_REACTION,
            "reaction_control": CapabilityId.REACTION_CONTROL,
        }[args.mode]
        if source is None:
            condition = make_absent_condition(
                batch_size=bsz,
                target_frames=target_length,
                target_lengths=lengths,
                capability=capability,
            )
        elif args.mode in {"reaction", "reaction_control"}:
            condition = make_reaction_condition(
                source,
                target_lengths=lengths,
                frame_gauge_dir=reaction_gauge.frame_gauge_dir,
                source_person_index=args.reaction_source_person_index,
                capability=capability,
            )
        else:
            condition = make_edit_condition(
                source,
                target_lengths=lengths,
                capability=capability,
            )
    if args.ease is not None:
        if condition is None or args.mode.startswith("edit") or args.mode in {
            "reaction",
            "reaction_control",
        }:
            raise ValueError("The first Ease protocol supports single-actor GENERATE routes only")
        ease = torch.tensor(args.ease, dtype=torch.float32).view(1, 6)
        condition = replace(
            condition,
            ease_physical=ease.expand(bsz, 6).clone(),
            ease_present=torch.ones(bsz, dtype=torch.bool),
        )
        condition.validate()
    text_profiles = (
        [RELATIVE_EDIT_TEXT_PROFILE] * bsz
        if args.mode.startswith("edit")
        else [INTERACTION_TEXT_PROFILE] * bsz
        if args.mode in {"reaction", "reaction_control", "interaction"}
        else [ABSOLUTE_TEXT_PROFILE] * bsz
    )
    context_cache = getattr(model.text_encoder, "context_cache", None)
    runtime_text_rows = encode_missing_text_rows(
        model.text_encoder.cache,
        texts,
        text_profiles,
        device,
        context_cache=context_cache,
    )
    register_runtime_text_rows(
        model.text_encoder.cache,
        runtime_text_rows,
        context_cache=context_cache,
    )
    if args.mode == "interaction":
        heading = torch.deg2rad(
            torch.full((bsz,), float(args.heading_deg), dtype=torch.float32)
        )
        interaction_c_dir = torch.stack(
            [torch.cos(heading), torch.sin(heading)], dim=-1
        )
        generator = torch.Generator(device=device).manual_seed(int(args.seed))
        output = sample_hy273_interaction_ode(
            model,
            normalizer,
            texts,
            lengths,
            num_steps=args.num_steps,
            cfg_scale=(
                3.5 if args.text_cfg_scale is None else args.text_cfg_scale
            ),
            c_dir=interaction_c_dir,
            generator=generator,
        )
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "samples_raw.npy", output.raw_motion.cpu().numpy())
        np.save(
            output_dir / "final_clean_prediction.npy",
            output.final_clean_prediction.cpu().numpy(),
        )
        for name, value in output.final_branch_predictions.items():
            np.save(output_dir / f"final_branch_{name}.npy", value.cpu().numpy())
        metadata = {
            **output.protocol,
            "checkpoint": str(checkpoint_path),
            "weight_source": args.weight_source,
            "seed": int(args.seed),
            "texts": texts,
            "runtime_encoded_text_rows": int(runtime_text_rows.count),
            "lengths": lengths.tolist(),
            "heading_deg": float(args.heading_deg),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"output_dir": str(output_dir), **output.protocol},
                sort_keys=True,
            )
        )
        return
    source_anchor = None
    if args.edit_source_baseline == "exact":
        if args.source_anchor_npy:
            source_anchor = _load_array(args.source_anchor_npy, "source_anchor_npy")
        elif source is not None and source.shape[1] == target_length:
            source_anchor = source
        else:
            raise ValueError(
                "Exact Edit baseline requires --source_anchor_npy when source and target lengths differ"
            )
    shape = (bsz, target_length, DIM_HY273)
    if controlled:
        observed = _load_array(args.observed_npy, "observed_npy")
        mask = _load_array(args.mask_npy, "mask_npy").bool()
        if observed.shape != shape or mask.shape != shape:
            raise ValueError(f"Control arrays must have shape {shape}")
        if args.mode == "reaction_control":
            if reaction_gauge is None:
                raise RuntimeError("Reaction-Control gauge was not prepared")
            observed = apply_reaction_gauge(observed, reaction_gauge)
    else:
        observed = torch.zeros(shape)
        mask = torch.zeros(shape, dtype=torch.bool)

    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    output = sample_hy273_multitask_ode(
        model,
        normalizer,
        condition,
        texts,
        observed,
        mask,
        num_steps=args.num_steps,
        text_cfg_scale=args.text_cfg_scale,
        source_cfg_scale=args.source_cfg_scale,
        edit_cfg_scale=args.edit_cfg_scale,
        control_cfg_scale=args.control_cfg_scale,
        contact_init=args.contact_init,
        contact_feedback=args.contact_feedback,
        cfg_apply_contacts=args.cfg_apply_contacts,
        generator=generator,
        edit_source_baseline=args.edit_source_baseline,
        edit_source_anchor_physical=source_anchor,
    )
    if args.mode in {"reaction", "reaction_control"}:
        if reaction_gauge is None:
            raise RuntimeError("Reaction gauge was not prepared")
        output = restore_reaction_sample_output(output, reaction_gauge)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "samples_raw.npy", output.raw_motion.cpu().numpy())
    np.save(output_dir / "samples_exact_clamped.npy", output.exact_clamped_motion.cpu().numpy())
    np.save(output_dir / "final_clean_prediction.npy", output.final_clean_prediction.cpu().numpy())
    for name, value in output.final_branch_predictions.items():
        np.save(output_dir / f"final_branch_{name}.npy", value.cpu().numpy())
    metadata = {
        **output.protocol,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "weight_source": args.weight_source,
        "seed": int(args.seed),
        "texts": texts,
        "runtime_encoded_text_rows": int(runtime_text_rows.count),
        "lengths": lengths.tolist(),
        "heading_deg": float(args.heading_deg),
        "reaction_source_person_index": (
            int(args.reaction_source_person_index)
            if args.mode in {"reaction", "reaction_control"}
            else None
        ),
        "reaction_clip_id": (
            args.reaction_clip_id
            if args.mode in {"reaction", "reaction_control"}
            else None
        ),
        "source_fps": float(args.source_fps) if args.source_fps is not None else None,
        "ease_physical": (
            None if args.ease is None else [float(value) for value in args.ease]
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **output.protocol}, sort_keys=True))


if __name__ == "__main__":
    main()
