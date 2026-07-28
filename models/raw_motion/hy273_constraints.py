"""Synthetic HY273 observed_motion / motion_mask construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import torch

from .hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    FIVE_POINT_GROUPS,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    KIMODO_EE_GROUPS,
    KIMODO_EE_ROT_GROUPS,
    ROOT_SLICE,
    global_rot_slice_for,
    joint_pos_slice_for,
)

ControlMode = Literal[
    "none",
    "root",
    "root_sparse",
    "root_dense",
    "endpoints",
    "fullpose",
    "contact",
    "mixed",
]
KimodoPattern = Literal["root_sparse", "root_dense", "endpoints", "fullpose", "contact"]
EndpointPreset = Literal["kimodo_ee", "five_point"]
EndpointSubsetMode = Literal["all", "random_nonempty"]


@dataclass
class ControlBatch:
    observed_motion: torch.Tensor
    motion_mask: torch.Tensor
    mode_ids: list[str]
    pattern_masks: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class KimodoControlCurriculum:
    none_prob: float = 0.10
    mixed_prob: float = 0.25
    max_sparse_keyframes: int = 20
    dense_min_fraction: float = 0.25
    endpoint_preset: EndpointPreset = "five_point"
    endpoint_subset_mode: EndpointSubsetMode = "random_nonempty"
    include_root_ref_for_endpoints: bool = True
    include_endpoint_rotations: bool = False
    include_contact_pattern: bool = False
    root_heading_probability: float = 1.0

    def validate(self) -> None:
        if self.none_prob < 0 or self.mixed_prob < 0 or self.none_prob + self.mixed_prob > 1:
            raise ValueError("Expected non-negative none/mixed probabilities summing to at most one")
        if self.max_sparse_keyframes < 1:
            raise ValueError("max_sparse_keyframes must be positive")
        if not 0 < self.dense_min_fraction <= 1:
            raise ValueError("dense_min_fraction must be in (0,1]")
        if not 0.0 <= self.root_heading_probability <= 1.0:
            raise ValueError("root_heading_probability must be in [0,1]")


def _select_frames(
    length: int,
    count: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if length <= 0 or count <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    count = min(int(count), int(length))
    if count == 1:
        return torch.randint(0, int(length), (1,), device=device, generator=generator)
    lin = torch.linspace(0, int(length) - 1, count, device=device)
    jitter = torch.randint(-1, 2, (count,), device=device, generator=generator)
    return (lin.round().long() + jitter).clamp(0, int(length) - 1).unique()


def _endpoint_groups(
    endpoint_preset: EndpointPreset,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    if endpoint_preset == "kimodo_ee":
        return KIMODO_EE_GROUPS, KIMODO_EE_ROT_GROUPS
    if endpoint_preset == "five_point":
        return FIVE_POINT_GROUPS, FIVE_POINT_GROUPS
    raise ValueError(f"Unknown endpoint preset: {endpoint_preset}")


def _sample_endpoint_joints(
    position_groups: tuple[tuple[int, ...], ...],
    rotation_groups: tuple[tuple[int, ...], ...],
    subset_mode: EndpointSubsetMode,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(position_groups) != len(rotation_groups):
        raise ValueError("Endpoint position/rotation groups must have the same length")
    if subset_mode == "all":
        return (
            tuple(joint_id for group in position_groups for joint_id in group),
            tuple(joint_id for group in rotation_groups for joint_id in group),
        )
    if subset_mode != "random_nonempty":
        raise ValueError(f"Unknown endpoint subset mode: {subset_mode}")

    selected = torch.rand(len(position_groups), device=device, generator=generator) < 0.5
    if not bool(selected.any().item()):
        selected[
            torch.randint(
                0, len(position_groups), (1,), device=device, generator=generator
            ).item()
        ] = True
    return (
        tuple(
            joint_id
            for group_idx, group in enumerate(position_groups)
            if bool(selected[group_idx].item())
            for joint_id in group
        ),
        tuple(
            joint_id
            for group_idx, group in enumerate(rotation_groups)
            if bool(selected[group_idx].item())
            for joint_id in group
        ),
    )


def _apply_indices(
    obs: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
    batch_idx: int,
    frames: torch.Tensor,
    indices: Sequence[int],
) -> None:
    if frames.numel() == 0 or not indices:
        return
    idx = torch.as_tensor(indices, device=source.device, dtype=torch.long)
    obs[batch_idx, frames[:, None], idx[None, :]] = source[batch_idx, frames[:, None], idx[None, :]]
    mask[batch_idx, frames[:, None], idx[None, :]] = True


def _apply_pattern_mask(
    pattern_masks: dict[str, torch.Tensor] | None,
    pattern: str,
    batch_idx: int,
    frames: torch.Tensor,
    indices: Sequence[int],
) -> None:
    if pattern_masks is None or frames.numel() == 0 or not indices:
        return
    idx = torch.as_tensor(indices, device=frames.device, dtype=torch.long)
    pattern_masks[pattern][batch_idx, frames[:, None], idx[None, :]] = True


def _low_biased_keyframe_count(
    max_count: int,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> int:
    if max_count <= 1:
        return 1
    draw = torch.rand((), device=device, generator=generator)
    return 1 + min(int((draw.square() * max_count).floor().item()), max_count - 1)


def _dense_interval_frames(
    length: int,
    min_fraction: float,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    min_span = max(1, min(length, int(round(length * float(min_fraction)))))
    if min_span >= length:
        return torch.arange(length, device=device, dtype=torch.long)
    span = int(torch.randint(min_span, length + 1, (1,), device=device, generator=generator).item())
    start = int(torch.randint(0, length - span + 1, (1,), device=device, generator=generator).item())
    return torch.arange(start, start + span, device=device, dtype=torch.long)


def build_kimodo_control_curriculum_batch(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    progress: float,
    config: KimodoControlCurriculum,
    generator: Optional[torch.Generator] = None,
    root_heading_generator: Optional[torch.Generator] = None,
    track_pattern_masks: bool = False,
    mode_schedule: Sequence[ControlMode] | None = None,
) -> ControlBatch:
    """Build project-domain Kimodo-like constraints from the paired clean motion."""
    if motion.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected motion [B,T,{DIM_HY273}], got {tuple(motion.shape)}")
    config.validate()
    progress = min(max(float(progress), 0.0), 1.0)
    current_kmax = 1 + int(progress * (config.max_sparse_keyframes - 1))
    bsz = motion.shape[0]
    if mode_schedule is not None and len(mode_schedule) != bsz:
        raise ValueError("mode_schedule must contain exactly one mode per batch element")
    device = motion.device
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    mode_ids: list[str] = []
    patterns: tuple[KimodoPattern, ...] = (
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
    )
    if config.include_contact_pattern:
        patterns = (*patterns, "contact")
    pattern_masks = (
        {pattern: torch.zeros_like(motion, dtype=torch.bool) for pattern in patterns}
        if track_pattern_masks
        else None
    )
    root_position_indices = [ROOT_SLICE.start, ROOT_SLICE.start + 2]
    root_path_indices = root_position_indices + list(
        range(HEADING_SLICE.start, HEADING_SLICE.stop)
    )
    root_full_indices = list(range(ROOT_SLICE.start, HEADING_SLICE.stop))
    fullpose_indices = list(range(JOINT_POS_SLICE.start, JOINT_POS_SLICE.stop))
    endpoint_position_groups, endpoint_rotation_groups = _endpoint_groups(
        config.endpoint_preset
    )

    for batch_idx in range(bsz):
        length = max(1, int(lengths[batch_idx].item()))
        forced_mode = None if mode_schedule is None else mode_schedule[batch_idx]
        mode_draw = float(torch.rand((), device=device, generator=generator).item())
        if forced_mode == "none" or (forced_mode is None and mode_draw < config.none_prob):
            chosen: tuple[KimodoPattern, ...] = ()
        elif forced_mode == "mixed" or (
            forced_mode is None and mode_draw < config.none_prob + config.mixed_prob
        ):
            order = torch.randperm(len(patterns), device=device, generator=generator)
            chosen = (patterns[int(order[0].item())], patterns[int(order[1].item())])
        elif forced_mode == "root":
            chosen = ("root_sparse",)
        elif forced_mode is not None:
            if forced_mode not in patterns:
                raise ValueError(f"Forced curriculum mode is unavailable: {forced_mode}")
            chosen = (forced_mode,)
        else:
            chosen = (
                patterns[int(torch.randint(0, len(patterns), (1,), device=device, generator=generator).item())],
            )

        labeled_patterns: list[str] = []
        for pattern in chosen:
            root_indices = root_path_indices
            root_label = pattern
            if pattern in {"root_sparse", "root_dense"}:
                probability = float(config.root_heading_probability)
                if probability >= 1.0:
                    include_heading = True
                elif probability <= 0.0:
                    include_heading = False
                else:
                    if root_heading_generator is None:
                        raise ValueError(
                            "0 < root_heading_probability < 1 requires an independent "
                            "root_heading_generator"
                        )
                    include_heading = bool(
                        torch.rand(
                            (), device=device, generator=root_heading_generator
                        ).item()
                        < probability
                    )
                root_indices = (
                    root_path_indices if include_heading else root_position_indices
                )
                if probability < 1.0:
                    suffix = "2dposrot" if include_heading else "2dpos"
                    root_label = f"{pattern}_{suffix}"
            labeled_patterns.append(root_label)

            if pattern == "root_dense":
                frames = _dense_interval_frames(
                    length,
                    config.dense_min_fraction,
                    device,
                    generator,
                )
                _apply_indices(obs, mask, motion, batch_idx, frames, root_indices)
                _apply_pattern_mask(
                    pattern_masks, pattern, batch_idx, frames, root_indices
                )
                continue

            count = _low_biased_keyframe_count(min(current_kmax, length), device, generator)
            frames = _select_frames(length, count, device, generator=generator)
            if pattern == "root_sparse":
                _apply_indices(obs, mask, motion, batch_idx, frames, root_indices)
                _apply_pattern_mask(
                    pattern_masks, pattern, batch_idx, frames, root_indices
                )
            elif pattern == "fullpose":
                _apply_indices(obs, mask, motion, batch_idx, frames, root_full_indices)
                _apply_indices(obs, mask, motion, batch_idx, frames, fullpose_indices)
                _apply_pattern_mask(
                    pattern_masks, pattern, batch_idx, frames, root_full_indices
                )
                _apply_pattern_mask(
                    pattern_masks, pattern, batch_idx, frames, fullpose_indices
                )
            elif pattern == "endpoints":
                endpoint_joints, endpoint_rotation_joints = _sample_endpoint_joints(
                    endpoint_position_groups,
                    endpoint_rotation_groups,
                    config.endpoint_subset_mode,
                    device,
                    generator,
                )
                if config.include_root_ref_for_endpoints:
                    _apply_indices(obs, mask, motion, batch_idx, frames, root_full_indices)
                    _apply_pattern_mask(
                        pattern_masks, pattern, batch_idx, frames, root_full_indices
                    )
                position_indices = joint_pos_slice_for(endpoint_joints)
                _apply_indices(
                    obs,
                    mask,
                    motion,
                    batch_idx,
                    frames,
                    position_indices,
                )
                _apply_pattern_mask(
                    pattern_masks, pattern, batch_idx, frames, position_indices
                )
                if config.include_endpoint_rotations:
                    rotation_indices = global_rot_slice_for(
                        endpoint_rotation_joints
                    )
                    _apply_indices(
                        obs,
                        mask,
                        motion,
                        batch_idx,
                        frames,
                        rotation_indices,
                    )
                    _apply_pattern_mask(
                        pattern_masks, pattern, batch_idx, frames, rotation_indices
                    )
            elif pattern == "contact":
                contact_indices = range(CONTACT_SLICE.start, CONTACT_SLICE.stop)
                _apply_indices(
                    obs,
                    mask,
                    motion,
                    batch_idx,
                    frames,
                    contact_indices,
                )
                _apply_pattern_mask(
                    pattern_masks, pattern, batch_idx, frames, contact_indices
                )
            else:
                raise AssertionError(f"Unhandled control pattern: {pattern}")
        if not labeled_patterns:
            mode_ids.append("none")
        elif len(labeled_patterns) == 1:
            mode_ids.append(labeled_patterns[0])
        else:
            mode_ids.append(f"mixed:{labeled_patterns[0]}+{labeled_patterns[1]}")
    return ControlBatch(
        observed_motion=obs,
        motion_mask=mask,
        mode_ids=mode_ids,
        pattern_masks=pattern_masks,
    )


def build_synthetic_control_batch(
    motion: torch.Tensor,
    lengths: torch.Tensor,
    modes: Sequence[ControlMode] = (
        "none",
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
        "mixed",
    ),
    endpoint_preset: EndpointPreset = "kimodo_ee",
    endpoint_subset_mode: EndpointSubsetMode = "random_nonempty",
    min_keyframes: int = 1,
    max_keyframes: int = 8,
    include_root_ref_for_endpoints: bool = True,
    include_endpoint_rotations: bool = False,
    include_contact_pattern: bool = False,
    dense_min_fraction: float = 0.25,
    generator: Optional[torch.Generator] = None,
    track_pattern_masks: bool = False,
    mode_schedule: Sequence[ControlMode] | None = None,
) -> ControlBatch:
    """Sample observed controls from the same clean motion.

    Inputs/outputs are in the same domain as ``motion``; call before normalization for
    source-domain controls, or after normalization for model-domain controls.
    """
    if motion.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected motion [B,T,{DIM_HY273}], got {tuple(motion.shape)}")
    if not modes:
        raise ValueError("At least one control mode is required")
    if mode_schedule is not None and len(mode_schedule) != motion.shape[0]:
        raise ValueError(
            "mode_schedule must contain exactly one mode per batch element"
        )
    if not 0 < float(dense_min_fraction) <= 1:
        raise ValueError("dense_min_fraction must be in (0,1]")
    if min_keyframes < 1 or max_keyframes < min_keyframes:
        raise ValueError(
            f"Expected 1 <= min_keyframes <= max_keyframes, got {min_keyframes}, {max_keyframes}"
        )
    bsz, _frames, _ = motion.shape
    device = motion.device
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    mode_names: list[str] = []
    endpoint_position_groups, endpoint_rotation_groups = _endpoint_groups(endpoint_preset)
    root_path_indices = [ROOT_SLICE.start, ROOT_SLICE.start + 2] + list(
        range(HEADING_SLICE.start, HEADING_SLICE.stop)
    )
    root_full_indices = list(range(ROOT_SLICE.start, HEADING_SLICE.stop))
    fullpose_indices = list(range(JOINT_POS_SLICE.start, JOINT_POS_SLICE.stop))
    available_patterns: tuple[KimodoPattern, ...] = (
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
    )
    if include_contact_pattern:
        available_patterns = (*available_patterns, "contact")
    pattern_masks = (
        {
            pattern: torch.zeros_like(motion, dtype=torch.bool)
            for pattern in available_patterns
        }
        if track_pattern_masks
        else None
    )

    for b in range(bsz):
        mode = (
            mode_schedule[b]
            if mode_schedule is not None
            else modes[
                int(
                    torch.randint(
                        0,
                        len(modes),
                        (1,),
                        device=device,
                        generator=generator,
                    ).item()
                )
            ]
        )
        length = int(lengths[b].item())
        if mode == "none":
            mode_names.append("none")
            continue
        if mode == "mixed":
            order = torch.randperm(len(available_patterns), device=device, generator=generator)
            selected_patterns = (
                available_patterns[int(order[0].item())],
                available_patterns[int(order[1].item())],
            )
            mode_names.append(f"mixed:{selected_patterns[0]}+{selected_patterns[1]}")
        elif mode == "root":
            selected_patterns = ("root_sparse",)
            mode_names.append("root_sparse")
        elif mode in available_patterns:
            selected_patterns = (mode,)
            mode_names.append(str(mode))
        else:
            raise ValueError(f"Unknown control mode: {mode}")

        for pattern in selected_patterns:
            if pattern == "root_dense":
                dense_frames = _dense_interval_frames(
                    length, dense_min_fraction, device, generator
                )
                _apply_indices(obs, mask, motion, b, dense_frames, root_path_indices)
                _apply_pattern_mask(
                    pattern_masks, pattern, b, dense_frames, root_path_indices
                )
                continue
            k = int(
                torch.randint(
                    min_keyframes,
                    max_keyframes + 1,
                    (1,),
                    device=device,
                    generator=generator,
                ).item()
            )
            keyframes = _select_frames(length, k, device, generator=generator)
            if pattern == "root_sparse":
                _apply_indices(obs, mask, motion, b, keyframes, root_path_indices)
                _apply_pattern_mask(
                    pattern_masks, pattern, b, keyframes, root_path_indices
                )
            elif pattern == "endpoints":
                endpoint_joints, endpoint_rotation_joints = _sample_endpoint_joints(
                    endpoint_position_groups,
                    endpoint_rotation_groups,
                    endpoint_subset_mode,
                    device,
                    generator,
                )
                if include_root_ref_for_endpoints:
                    _apply_indices(obs, mask, motion, b, keyframes, root_full_indices)
                    _apply_pattern_mask(
                        pattern_masks, pattern, b, keyframes, root_full_indices
                    )
                position_indices = joint_pos_slice_for(endpoint_joints)
                _apply_indices(
                    obs,
                    mask,
                    motion,
                    b,
                    keyframes,
                    position_indices,
                )
                _apply_pattern_mask(
                    pattern_masks, pattern, b, keyframes, position_indices
                )
                if include_endpoint_rotations:
                    rotation_indices = global_rot_slice_for(
                        endpoint_rotation_joints
                    )
                    _apply_indices(
                        obs,
                        mask,
                        motion,
                        b,
                        keyframes,
                        rotation_indices,
                    )
                    _apply_pattern_mask(
                        pattern_masks, pattern, b, keyframes, rotation_indices
                    )
            elif pattern == "fullpose":
                _apply_indices(obs, mask, motion, b, keyframes, root_full_indices)
                _apply_indices(obs, mask, motion, b, keyframes, fullpose_indices)
                _apply_pattern_mask(
                    pattern_masks, pattern, b, keyframes, root_full_indices
                )
                _apply_pattern_mask(
                    pattern_masks, pattern, b, keyframes, fullpose_indices
                )
            elif pattern == "contact":
                contact_indices = range(CONTACT_SLICE.start, CONTACT_SLICE.stop)
                _apply_indices(
                    obs,
                    mask,
                    motion,
                    b,
                    keyframes,
                    contact_indices,
                )
                _apply_pattern_mask(
                    pattern_masks, pattern, b, keyframes, contact_indices
                )
            else:
                raise AssertionError(f"Unhandled control pattern: {pattern}")
    return ControlBatch(
        observed_motion=obs,
        motion_mask=mask,
        mode_ids=mode_names,
        pattern_masks=pattern_masks,
    )


def compile_root_control(motion: torch.Tensor, frames: Iterable[int]) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.zeros_like(motion)
    mask = torch.zeros_like(motion, dtype=torch.bool)
    f = torch.as_tensor(list(frames), device=motion.device, dtype=torch.long)
    idx = torch.as_tensor(
        [ROOT_SLICE.start, ROOT_SLICE.start + 2, HEADING_SLICE.start, HEADING_SLICE.start + 1],
        device=motion.device,
        dtype=torch.long,
    )
    obs[f[:, None], idx[None, :]] = motion[f[:, None], idx[None, :]]
    mask[f[:, None], idx[None, :]] = True
    return obs, mask
