"""Edit-specific objectives for the shared HY273 multitask model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .hy273_normalizer import HY273Normalizer
from .hy273_multitask_losses import (
    SEMANTIC_SLICES,
    SEMANTIC_WEIGHTS,
    SEMANTIC_WEIGHT_SUM,
)
from .hy273_slices import CONT_DIM, DIM_HY273
from .hy273_slices import (
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    NUM_JOINTS,
    ROOT_SLICE,
    VELOCITY_SLICE,
    cont6d_to_matrix,
    reconstruct_global_joints_from_features,
    split_global_rot6d,
)


@dataclass(frozen=True)
class UnifiedEditLossWeights:
    target_x0_scale: float = 0.05
    hard_x0_scale: float = 0.02
    hard_fraction: float = 0.20
    instruction_rank_scale: float = 0.05
    instruction_relative_margin: float = 0.10

    def validate(self) -> None:
        if min(
            self.target_x0_scale,
            self.hard_x0_scale,
            self.instruction_rank_scale,
            self.instruction_relative_margin,
        ) < 0.0:
            raise ValueError("Unified Edit loss coefficients must be non-negative")
        if not 0.0 < self.hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be in (0,1]")


@dataclass
class UnifiedEditLossBundle:
    total: torch.Tensor
    target_x0_raw: torch.Tensor
    hard_x0_raw: torch.Tensor
    instruction_rank_raw: torch.Tensor
    instruction_rank_ce_raw: torch.Tensor
    target_x0_weighted: torch.Tensor
    hard_x0_weighted: torch.Tensor
    instruction_rank_weighted: torch.Tensor
    correct_distance: torch.Tensor
    shuffled_distance: torch.Tensor
    instruction_gap: torch.Tensor
    instruction_rank_active_fraction: torch.Tensor
    instruction_rank_active_all_fraction: torch.Tensor
    instruction_rank_eligible_fraction: torch.Tensor
    instruction_rank_slope: torch.Tensor
    instruction_rank_margin: torch.Tensor


@dataclass
class SourceAnchorLossBundle:
    total: torch.Tensor
    raw: torch.Tensor
    weighted: torch.Tensor
    correct_distance: torch.Tensor
    source_baseline_distance: torch.Tensor
    active_fraction: torch.Tensor


@dataclass
class SourceTargetDiscrepancyMask:
    """A weak source/target discrepancy proxy, not semantic edit annotation."""

    pre_intersection_mask: torch.Tensor
    mask: torch.Tensor
    root_time_mask: torch.Tensor
    body_joint_time_mask: torch.Tensor
    velocity_joint_time_mask: torch.Tensor
    exact_identity: torch.Tensor
    equal_length: torch.Tensor


@dataclass
class SourceTargetDiscrepancyLossBundle:
    total: torch.Tensor
    raw: torch.Tensor
    weighted: torch.Tensor
    active_fraction: torch.Tensor


@dataclass
class PhysicalTemporalEditLossBundle:
    total: torch.Tensor
    raw: torch.Tensor
    vector_raw: torch.Tensor
    speed_raw: torch.Tensor
    weighted: torch.Tensor
    active_fraction: torch.Tensor
    equal_length_fraction: torch.Tensor
    importance_mean: torch.Tensor
    high_importance_fraction: torch.Tensor
    source_target_velocity_delta_mps: torch.Tensor
    source_target_speed_delta_mps: torch.Tensor


def _top_fraction_time_mask(
    scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    fraction: float,
    activity_eps: float,
) -> torch.Tensor:
    """Select a fixed fraction along the final time axis for each score row."""

    if scores.ndim < 2 or valid.shape != (scores.shape[0], scores.shape[-1]):
        raise ValueError("scores must be [B,...,T] and valid must be [B,T]")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    if not math.isfinite(float(activity_eps)) or float(activity_eps) < 0.0:
        raise ValueError("activity_eps must be finite and non-negative")

    time = scores.shape[-1]
    valid_view = valid.view(valid.shape[0], *([1] * (scores.ndim - 2)), time)
    masked = scores.masked_fill(~valid_view, float("-inf"))
    order = masked.argsort(dim=-1, descending=True)
    lengths = valid.sum(dim=-1)
    counts = torch.ceil(lengths.float() * float(fraction)).long().clamp_min(1)
    count_view = counts.view(counts.shape[0], *([1] * (scores.ndim - 2)), 1)
    selected_by_rank = torch.arange(time, device=scores.device).view(
        *([1] * (scores.ndim - 1)), time
    ) < count_view
    selected_by_rank = selected_by_rank.expand_as(scores)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(dim=-1, index=order, src=selected_by_rank)
    active = masked.amax(dim=-1) > float(activity_eps)
    return selected & valid_view & active.unsqueeze(-1)


@torch.no_grad()
def build_source_target_discrepancy_mask(
    *,
    source_physical: torch.Tensor,
    source_lengths: torch.Tensor,
    target_physical: torch.Tensor,
    target_valid: torch.Tensor,
    hard_mask: torch.Tensor,
    fraction: float = 0.20,
    position_scale_m: float = 0.05,
    rotation_scale_deg: float = 10.0,
    activity_eps: float = 1e-6,
) -> SourceTargetDiscrepancyMask:
    """Rank source/target discrepancies after paired gauge augmentation.

    MotionFix pairs are independent sequences, so normalized-progress nearest
    neighbors are only a weak alignment proxy. Root time and each body joint's
    time axis are ranked independently to prevent distal joints from starving
    root supervision. K273 velocity masks follow the finite-difference edge
    convention rather than copying the same-frame position mask.
    """

    if source_physical.ndim == 4:
        if source_physical.shape[1] != 1:
            raise ValueError("Discrepancy proxy requires exactly one source slot")
        source_physical = source_physical[:, 0]
    if source_lengths.ndim == 2:
        if source_lengths.shape[1] != 1:
            raise ValueError("Discrepancy proxy requires exactly one source length")
        source_lengths = source_lengths[:, 0]
    if source_physical.ndim != 3 or source_physical.shape[-1] != DIM_HY273:
        raise ValueError("source_physical must have shape [B,Ts,273]")
    if target_physical.ndim != 3 or target_physical.shape[-1] != DIM_HY273:
        raise ValueError("target_physical must have shape [B,T,273]")
    if source_physical.shape[0] != target_physical.shape[0]:
        raise ValueError("source and target batch sizes differ")
    if source_lengths.shape != (target_physical.shape[0],):
        raise ValueError("source_lengths must have shape [B]")
    if target_valid.shape != target_physical.shape[:2]:
        raise ValueError("target_valid must have shape [B,T]")
    if hard_mask.shape != target_physical.shape:
        raise ValueError("hard_mask must have shape [B,T,273]")
    if not math.isfinite(float(position_scale_m)) or float(position_scale_m) <= 0.0:
        raise ValueError("position_scale_m must be finite and positive")
    if not math.isfinite(float(rotation_scale_deg)) or float(rotation_scale_deg) <= 0.0:
        raise ValueError("rotation_scale_deg must be finite and positive")

    source = source_physical.detach().float()
    target = target_physical.detach().float()
    valid = target_valid.detach().to(device=target.device, dtype=torch.bool)
    source_lengths = source_lengths.detach().to(device=target.device, dtype=torch.long)
    target_lengths = valid.sum(dim=-1).long()
    if bool((source_lengths < 0).any()) or bool((source_lengths > source.shape[1]).any()):
        raise ValueError("source_lengths are outside the padded source tensor")
    if bool((target_lengths < 1).any()):
        raise ValueError("Every discrepancy sample needs at least one target frame")

    batch, frames = target.shape[:2]
    source_active = source_lengths > 0
    safe_source_lengths = source_lengths.clamp_min(1)
    discrepancy_valid = valid & source_active.unsqueeze(1)
    target_index = torch.arange(frames, device=target.device, dtype=torch.float64)
    numerator = target_index.unsqueeze(0) * (safe_source_lengths - 1).double().unsqueeze(1)
    denominator = (target_lengths - 1).clamp_min(1).double().unsqueeze(1)
    source_indices = torch.floor(numerator / denominator + 0.5).long()
    source_indices = source_indices.clamp_min(0)
    source_indices = torch.minimum(
        source_indices, (safe_source_lengths - 1).unsqueeze(1)
    )
    aligned_source = torch.gather(
        source,
        dim=1,
        index=source_indices.unsqueeze(-1).expand(batch, frames, DIM_HY273),
    )

    source_root = aligned_source[..., ROOT_SLICE]
    target_root = target[..., ROOT_SLICE]
    root_position_distance = torch.linalg.vector_norm(source_root - target_root, dim=-1)
    source_heading = aligned_source[..., HEADING_SLICE]
    target_heading = target[..., HEADING_SLICE]
    heading_dot = (source_heading * target_heading).sum(dim=-1)
    heading_cross = (
        source_heading[..., 0] * target_heading[..., 1]
        - source_heading[..., 1] * target_heading[..., 0]
    )
    heading_angle_deg = torch.rad2deg(
        torch.atan2(heading_cross.abs(), heading_dot.clamp(min=-1.0, max=1.0))
    )
    root_score = (
        root_position_distance / float(position_scale_m)
        + heading_angle_deg / float(rotation_scale_deg)
    )
    root_time_mask = _top_fraction_time_mask(
        root_score,
        discrepancy_valid,
        fraction=fraction,
        activity_eps=activity_eps,
    )

    source_joints = reconstruct_global_joints_from_features(aligned_source)
    target_joints = reconstruct_global_joints_from_features(target)
    joint_position_distance = torch.linalg.vector_norm(
        source_joints - target_joints, dim=-1
    )
    source_rot = cont6d_to_matrix(split_global_rot6d(aligned_source))
    target_rot = cont6d_to_matrix(split_global_rot6d(target))
    relative_rot = source_rot.transpose(-1, -2) @ target_rot
    rotation_cos = (
        relative_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    # atan2 avoids acos' square-root amplification near identity. In float32,
    # acos(trace(R.T @ R)) otherwise reports about 0.05 degrees for R vs itself.
    rotation_skew = torch.stack(
        (
            relative_rot[..., 2, 1] - relative_rot[..., 1, 2],
            relative_rot[..., 0, 2] - relative_rot[..., 2, 0],
            relative_rot[..., 1, 0] - relative_rot[..., 0, 1],
        ),
        dim=-1,
    )
    rotation_sin = 0.5 * torch.linalg.vector_norm(rotation_skew, dim=-1)
    rotation_angle_deg = torch.rad2deg(
        torch.atan2(
            rotation_sin.clamp(min=0.0, max=1.0),
            rotation_cos.clamp(min=-1.0, max=1.0),
        )
    )
    body_score = (
        joint_position_distance / float(position_scale_m)
        + rotation_angle_deg / float(rotation_scale_deg)
    )
    body_joint_time_mask = _top_fraction_time_mask(
        body_score.transpose(1, 2),
        discrepancy_valid,
        fraction=fraction,
        activity_eps=activity_eps,
    ).transpose(1, 2)

    velocity_joint_time_mask = body_joint_time_mask.clone()
    if frames > 1:
        velocity_joint_time_mask[:, :-1] = (
            body_joint_time_mask[:, :-1] | body_joint_time_mask[:, 1:]
        )
        for sample_index, length in enumerate(target_lengths.tolist()):
            if length > 1:
                velocity_joint_time_mask[sample_index, length - 1] = (
                    velocity_joint_time_mask[sample_index, length - 2]
                )

    pre = torch.zeros(batch, frames, CONT_DIM, device=target.device, dtype=torch.bool)
    pre[..., ROOT_SLICE] = root_time_mask.unsqueeze(-1)
    pre[..., HEADING_SLICE] = root_time_mask.unsqueeze(-1)
    pre[..., JOINT_POS_SLICE] = body_joint_time_mask.unsqueeze(-1).expand(
        -1, -1, -1, 3
    ).reshape(batch, frames, -1)
    pre[..., GLOBAL_ROT_SLICE] = body_joint_time_mask.unsqueeze(-1).expand(
        -1, -1, -1, 6
    ).reshape(batch, frames, -1)
    pre[..., VELOCITY_SLICE] = velocity_joint_time_mask.unsqueeze(-1).expand(
        -1, -1, -1, 3
    ).reshape(batch, frames, -1)
    exact_identity = source_active & ~(
        root_time_mask.any(dim=1) | body_joint_time_mask.any(dim=(1, 2))
    )
    pre &= valid.unsqueeze(-1)
    post = pre & ~hard_mask[..., :CONT_DIM].detach().to(
        device=target.device, dtype=torch.bool
    )
    return SourceTargetDiscrepancyMask(
        pre_intersection_mask=pre,
        mask=post,
        root_time_mask=root_time_mask,
        body_joint_time_mask=body_joint_time_mask,
        velocity_joint_time_mask=velocity_joint_time_mask,
        exact_identity=exact_identity,
        equal_length=source_active & (source_lengths == target_lengths),
    )


def compute_source_target_discrepancy_x0_loss(
    *,
    correct_x0_hat_cont: torch.Tensor,
    x0_target_norm: torch.Tensor,
    discrepancy_mask: torch.Tensor,
    scale: float,
    active_denominator: torch.Tensor | float | None = None,
) -> SourceTargetDiscrepancyLossBundle:
    """Apply normalized clean-x0 SmoothL1 only on discrepancy-ranked coordinates."""

    if x0_target_norm.ndim != 3 or x0_target_norm.shape[-1] != DIM_HY273:
        raise ValueError("x0_target_norm must have shape [B,T,273]")
    target = x0_target_norm[..., :CONT_DIM].to(
        device=correct_x0_hat_cont.device,
        dtype=correct_x0_hat_cont.dtype,
    )
    if correct_x0_hat_cont.shape != target.shape:
        raise ValueError("correct prediction shape does not match target")
    if discrepancy_mask.shape != target.shape or discrepancy_mask.dtype != torch.bool:
        raise ValueError("discrepancy_mask must be bool [B,T,269]")
    if not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError("scale must be finite and non-negative")
    mask = discrepancy_mask.to(device=target.device)
    values = F.smooth_l1_loss(
        correct_x0_hat_cont, target, reduction="none", beta=1.0
    )
    per_sample = _semantic_distance_per_sample(values, mask)
    active = mask.reshape(mask.shape[0], -1).any(dim=-1)
    active_f = active.to(dtype=per_sample.dtype)
    if active_denominator is None:
        denominator = active_f.sum().clamp_min(1.0)
    else:
        denominator = torch.as_tensor(
            active_denominator,
            device=per_sample.device,
            dtype=per_sample.dtype,
        )
        if denominator.numel() != 1:
            raise ValueError("active_denominator must be scalar")
        if not bool(torch.isfinite(denominator.detach())):
            raise ValueError("active_denominator must be finite")
        if bool((denominator.detach() < 0).item()):
            raise ValueError("active_denominator must be non-negative")
        denominator = torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
    raw = (per_sample * active_f).sum() / denominator
    weighted = raw * float(scale)
    return SourceTargetDiscrepancyLossBundle(
        total=weighted,
        raw=raw,
        weighted=weighted,
        active_fraction=active_f.mean(),
    )


def compute_physical_temporal_edit_loss(
    *,
    x0_hat_norm: torch.Tensor,
    x0_target_physical: torch.Tensor,
    source_physical: torch.Tensor,
    source_lengths: torch.Tensor,
    target_valid: torch.Tensor,
    sample_mask: torch.Tensor,
    normalizer: HY273Normalizer,
    scale: float,
    vector_weight: float = 0.5,
    speed_weight: float = 0.5,
    background_weight: float = 0.10,
    change_scale_mps: float = 0.25,
    smooth_l1_beta_mps: float = 0.10,
    fps: float = 30.0,
    active_denominator: torch.Tensor | float | None = None,
) -> PhysicalTemporalEditLossBundle:
    """Supervise instruction-bearing Edit rows in physical joint-velocity space.

    The source only determines a detached soft importance map. The optimized
    target remains the paired target motion, so the loss cannot reward copying
    source or satisfy itself by changing a negative branch. V1 deliberately
    uses exact equal-length pairs; unequal-length temporal alignment is a
    separate modeling question and is left to the base objective.
    """

    if x0_hat_norm.ndim != 3 or x0_hat_norm.shape[-1] != DIM_HY273:
        raise ValueError("x0_hat_norm must have shape [B,T,273]")
    if x0_target_physical.shape != x0_hat_norm.shape:
        raise ValueError("x0_target_physical must match x0_hat_norm")
    if source_physical.ndim == 4:
        if source_physical.shape[1] != 1:
            raise ValueError("Physical temporal Edit loss requires one source slot")
        source_physical = source_physical[:, 0]
    if source_lengths.ndim == 2:
        if source_lengths.shape[1] != 1:
            raise ValueError("Physical temporal Edit loss requires one source length")
        source_lengths = source_lengths[:, 0]
    if (
        source_physical.ndim != 3
        or source_physical.shape[0] != x0_hat_norm.shape[0]
        or source_physical.shape[-1] != DIM_HY273
    ):
        raise ValueError("source_physical must have shape [B,Ts,273]")
    batch, frames = x0_hat_norm.shape[:2]
    if source_lengths.shape != (batch,):
        raise ValueError("source_lengths must have shape [B]")
    if target_valid.shape != (batch, frames):
        raise ValueError("target_valid must have shape [B,T]")
    if sample_mask.shape != (batch,):
        raise ValueError("sample_mask must have shape [B]")

    scalar_values = {
        "scale": scale,
        "vector_weight": vector_weight,
        "speed_weight": speed_weight,
        "background_weight": background_weight,
        "change_scale_mps": change_scale_mps,
        "smooth_l1_beta_mps": smooth_l1_beta_mps,
        "fps": fps,
    }
    if any(not math.isfinite(float(value)) for value in scalar_values.values()):
        raise ValueError(f"Physical temporal loss values must be finite: {scalar_values}")
    if float(scale) < 0.0:
        raise ValueError("scale must be non-negative")
    if float(vector_weight) < 0.0 or float(speed_weight) < 0.0:
        raise ValueError("vector_weight and speed_weight must be non-negative")
    weight_sum = float(vector_weight) + float(speed_weight)
    if weight_sum <= 0.0:
        raise ValueError("At least one temporal component weight must be positive")
    if not 0.0 <= float(background_weight) <= 1.0:
        raise ValueError("background_weight must be in [0,1]")
    if min(float(change_scale_mps), float(smooth_l1_beta_mps), float(fps)) <= 0.0:
        raise ValueError("change scale, SmoothL1 beta, and fps must be positive")

    device = x0_hat_norm.device
    valid = target_valid.to(device=device, dtype=torch.bool)
    target_lengths = valid.sum(dim=-1).long()
    source_lengths = source_lengths.to(device=device, dtype=torch.long)
    if bool((source_lengths < 0).any()) or bool(
        (source_lengths > source_physical.shape[1]).any()
    ):
        raise ValueError("source_lengths are outside source_physical")
    requested = sample_mask.to(device=device, dtype=torch.bool)
    equal_length = (
        (source_lengths > 0)
        & (source_lengths == target_lengths)
        & (target_lengths >= 2)
    )
    active = requested & equal_length
    pair_valid = valid[:, 1:] & valid[:, :-1] & active[:, None]

    source = source_physical.detach().to(device=device, dtype=torch.float32)
    target = x0_target_physical.detach().to(device=device, dtype=torch.float32)
    source_on_target = source.new_zeros((batch, frames, DIM_HY273))
    copied_frames = min(frames, source.shape[1])
    source_on_target[:, :copied_frames] = source[:, :copied_frames]

    with torch.autocast(device_type=device.type, enabled=False):
        prediction_physical = normalizer.denormalize(x0_hat_norm.float())
        pred_joints = reconstruct_global_joints_from_features(prediction_physical)
        target_joints = reconstruct_global_joints_from_features(target)
        source_joints = reconstruct_global_joints_from_features(source_on_target)

        pred_velocity = (pred_joints[:, 1:] - pred_joints[:, :-1]) * float(fps)
        target_velocity = (
            target_joints[:, 1:] - target_joints[:, :-1]
        ) * float(fps)
        source_velocity = (
            source_joints[:, 1:] - source_joints[:, :-1]
        ) * float(fps)

        with torch.no_grad():
            velocity_delta = torch.linalg.vector_norm(
                target_velocity - source_velocity, dim=-1
            )
            target_speed = torch.linalg.vector_norm(target_velocity, dim=-1)
            source_speed = torch.linalg.vector_norm(source_velocity, dim=-1)
            speed_delta = (target_speed - source_speed).abs()
            importance = (
                velocity_delta / float(change_scale_mps)
            ).clamp(min=0.0, max=1.0)
            soft_importance = float(background_weight) + (
                1.0 - float(background_weight)
            ) * importance

        pair_weight = soft_importance * pair_valid[..., None].float()
        per_sample_denominator = pair_weight.sum(dim=(1, 2)).clamp_min(1e-12)
        vector_values = F.smooth_l1_loss(
            pred_velocity,
            target_velocity,
            reduction="none",
            beta=float(smooth_l1_beta_mps),
        ).mean(dim=-1)
        pred_speed = torch.linalg.vector_norm(pred_velocity, dim=-1)
        speed_values = F.smooth_l1_loss(
            pred_speed,
            target_speed,
            reduction="none",
            beta=float(smooth_l1_beta_mps),
        )
        vector_per_sample = (
            vector_values * pair_weight
        ).sum(dim=(1, 2)) / per_sample_denominator
        speed_per_sample = (
            speed_values * pair_weight
        ).sum(dim=(1, 2)) / per_sample_denominator

        active_f = active.float()
        if active_denominator is None:
            denominator = active_f.sum().clamp_min(1.0)
        else:
            denominator = torch.as_tensor(
                active_denominator,
                device=device,
                dtype=torch.float32,
            )
            if denominator.numel() != 1:
                raise ValueError("active_denominator must be scalar")
            if not bool(torch.isfinite(denominator.detach())) or bool(
                (denominator.detach() < 0).item()
            ):
                raise ValueError("active_denominator must be finite and non-negative")
            denominator = torch.where(
                denominator > 0, denominator, torch.ones_like(denominator)
            )

        vector_raw = (vector_per_sample * active_f).sum() / denominator
        speed_raw = (speed_per_sample * active_f).sum() / denominator
        normalized_vector_weight = float(vector_weight) / weight_sum
        normalized_speed_weight = float(speed_weight) / weight_sum
        raw = (
            normalized_vector_weight * vector_raw
            + normalized_speed_weight * speed_raw
        )
        weighted = raw * float(scale)

        diagnostic_mask = pair_valid[..., None].expand_as(importance)
        diagnostic_count = diagnostic_mask.float().sum().clamp_min(1.0)
        importance_mean = (
            importance * diagnostic_mask.float()
        ).sum() / diagnostic_count
        high_importance_fraction = (
            (importance >= 0.5) & diagnostic_mask
        ).float().sum() / diagnostic_count
        velocity_delta_mean = (
            velocity_delta * diagnostic_mask.float()
        ).sum() / diagnostic_count
        speed_delta_mean = (
            speed_delta * diagnostic_mask.float()
        ).sum() / diagnostic_count

    requested_count = requested.float().sum().clamp_min(1.0)
    return PhysicalTemporalEditLossBundle(
        total=weighted,
        raw=raw,
        vector_raw=vector_raw,
        speed_raw=speed_raw,
        weighted=weighted,
        active_fraction=active_f.mean(),
        equal_length_fraction=(equal_length & requested).float().sum()
        / requested_count,
        importance_mean=importance_mean,
        high_importance_fraction=high_importance_fraction,
        source_target_velocity_delta_mps=velocity_delta_mean,
        source_target_speed_delta_mps=speed_delta_mean,
    )


def _semantic_distance_per_sample(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    rows = []
    zero = values.sum(dim=(1, 2)) * 0.0
    for name, block_slice in SEMANTIC_SLICES.items():
        block_values = values[..., block_slice]
        block_mask = mask[..., block_slice].to(dtype=block_values.dtype)
        numerator = (block_values * block_mask).sum(dim=(1, 2))
        denominator = block_mask.sum(dim=(1, 2))
        block_mean = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(1.0),
            zero,
        )
        rows.append(block_mean * float(SEMANTIC_WEIGHTS[name]))
    return torch.stack(rows, dim=0).sum(dim=0) / SEMANTIC_WEIGHT_SUM


def _semantic_hard_distance_per_sample(
    values: torch.Tensor,
    mask: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    per_block = []
    for name, block_slice in SEMANTIC_SLICES.items():
        block_values = values[..., block_slice]
        block_mask = mask[..., block_slice]
        per_sample = []
        for sample_index in range(values.shape[0]):
            selected = block_values[sample_index][block_mask[sample_index]]
            if selected.numel() == 0:
                per_sample.append(block_values[sample_index].sum() * 0.0)
                continue
            count = max(1, int(math.ceil(selected.numel() * float(fraction))))
            per_sample.append(selected.topk(count, sorted=False).values.mean())
        per_block.append(
            torch.stack(per_sample) * float(SEMANTIC_WEIGHTS[name])
        )
    return torch.stack(per_block, dim=0).sum(dim=0) / SEMANTIC_WEIGHT_SUM


def compute_unified_edit_loss(
    *,
    correct_x0_hat_cont: torch.Tensor,
    shuffled_x0_hat_cont: torch.Tensor | None,
    x0_target_norm: torch.Tensor,
    target_valid: torch.Tensor,
    hard_mask: torch.Tensor,
    weights: UnifiedEditLossWeights | None = None,
    instruction_rank_mode: str = "hinge",
    instruction_rank_temperature: float = 0.01,
    instruction_rank_sample_mask: torch.Tensor | None = None,
    instruction_rank_denominator: torch.Tensor | float | None = None,
) -> UnifiedEditLossBundle:
    """Emphasize hard target regions and require instruction sensitivity.

    Both auxiliary reconstruction terms use only the aligned target prediction
    and target x0. When a shuffled prediction is provided, the ranking term
    compares it with the correct instruction for the same source/noisy state.
    """

    weights = weights or UnifiedEditLossWeights()
    weights.validate()
    if x0_target_norm.ndim != 3 or x0_target_norm.shape[-1] != DIM_HY273:
        raise ValueError("x0_target_norm must have shape [B,T,273]")
    expected = x0_target_norm[..., :CONT_DIM]
    if correct_x0_hat_cont.shape != expected.shape:
        raise ValueError("correct prediction shape does not match target")
    if shuffled_x0_hat_cont is not None and shuffled_x0_hat_cont.shape != expected.shape:
        raise ValueError("shuffled prediction shape does not match target")
    if target_valid.shape != x0_target_norm.shape[:2]:
        raise ValueError("target_valid must have shape [B,T]")
    if hard_mask.shape != x0_target_norm.shape:
        raise ValueError("hard_mask must have shape [B,T,273]")
    if instruction_rank_mode not in {"hinge", "softplus"}:
        raise ValueError("instruction_rank_mode must be 'hinge' or 'softplus'")
    if (
        not math.isfinite(float(instruction_rank_temperature))
        or float(instruction_rank_temperature) <= 0.0
    ):
        raise ValueError("instruction_rank_temperature must be finite and positive")
    if instruction_rank_sample_mask is None:
        rank_eligible = torch.ones(
            correct_x0_hat_cont.shape[0],
            device=correct_x0_hat_cont.device,
            dtype=torch.bool,
        )
    else:
        if instruction_rank_sample_mask.shape != (correct_x0_hat_cont.shape[0],):
            raise ValueError("instruction_rank_sample_mask must have shape [B]")
        rank_eligible = instruction_rank_sample_mask.to(
            device=correct_x0_hat_cont.device, dtype=torch.bool
        )

    target = expected.to(
        device=correct_x0_hat_cont.device,
        dtype=correct_x0_hat_cont.dtype,
    )
    valid = target_valid.to(device=target.device, dtype=torch.bool)
    unobserved = valid[..., None] & ~hard_mask[..., :CONT_DIM].to(
        device=target.device, dtype=torch.bool
    )
    correct_values = F.smooth_l1_loss(
        correct_x0_hat_cont, target, reduction="none", beta=1.0
    )
    correct_distance = _semantic_distance_per_sample(correct_values, unobserved)
    if shuffled_x0_hat_cont is None:
        shuffled_distance = correct_distance.detach()
    else:
        shuffled_values = F.smooth_l1_loss(
            shuffled_x0_hat_cont, target, reduction="none", beta=1.0
        )
        shuffled_distance = _semantic_distance_per_sample(shuffled_values, unobserved)
    hard_distance = _semantic_hard_distance_per_sample(
        correct_values, unobserved, weights.hard_fraction
    )

    rank_margin = (
        (1.0 + float(weights.instruction_relative_margin)) * correct_distance
        - shuffled_distance
    )
    if shuffled_x0_hat_cont is None:
        rank_per_sample = torch.zeros_like(correct_distance)
        rank_ce_per_sample = rank_per_sample
        rank_slope = torch.zeros_like(correct_distance)
        rank_eligible = torch.zeros_like(rank_eligible)
    elif instruction_rank_mode == "hinge":
        rank_per_sample = torch.relu(rank_margin)
        rank_ce_per_sample = rank_per_sample
        rank_slope = (rank_margin > 0).to(dtype=rank_margin.dtype)
    else:
        temperature = float(instruction_rank_temperature)
        # Keep the objective in distance units while replacing the hard hinge
        # with a smooth two-way assignment loss. Float32 avoids bf16 tail loss.
        scaled_margin = rank_margin.float() / temperature
        rank_ce_per_sample = temperature * F.softplus(scaled_margin)
        # Subtracting chance entropy changes no gradient and keeps the reported
        # auxiliary centered at zero when the two instructions are indistinct.
        rank_per_sample = rank_ce_per_sample - temperature * math.log(2.0)
        rank_slope = torch.sigmoid(scaled_margin)
    rank_weight = rank_eligible.to(dtype=rank_per_sample.dtype)
    if instruction_rank_denominator is None:
        rank_denominator = rank_weight.sum().clamp_min(1.0)
    else:
        rank_denominator = torch.as_tensor(
            instruction_rank_denominator,
            device=rank_per_sample.device,
            dtype=rank_per_sample.dtype,
        )
        if rank_denominator.numel() != 1:
            raise ValueError("instruction_rank_denominator must be scalar")
        if not bool(torch.isfinite(rank_denominator.detach())):
            raise ValueError("instruction_rank_denominator must be finite")
        if bool((rank_denominator.detach() < 0).item()):
            raise ValueError("instruction_rank_denominator must be non-negative")
        rank_denominator = torch.where(
            rank_denominator > 0,
            rank_denominator,
            torch.ones_like(rank_denominator),
        )
    rank_active = (rank_margin > 0) & rank_eligible
    target_x0_raw = correct_distance.mean()
    hard_x0_raw = hard_distance.mean()
    instruction_rank_raw = (rank_per_sample * rank_weight).sum() / rank_denominator
    instruction_rank_ce_raw = (
        (rank_ce_per_sample * rank_weight).sum() / rank_denominator
    )
    target_x0_weighted = target_x0_raw * float(weights.target_x0_scale)
    hard_x0_weighted = hard_x0_raw * float(weights.hard_x0_scale)
    instruction_rank_weighted = instruction_rank_raw * float(
        weights.instruction_rank_scale
    )
    total = target_x0_weighted + hard_x0_weighted + instruction_rank_weighted
    return UnifiedEditLossBundle(
        total=total,
        target_x0_raw=target_x0_raw,
        hard_x0_raw=hard_x0_raw,
        instruction_rank_raw=instruction_rank_raw,
        instruction_rank_ce_raw=instruction_rank_ce_raw,
        target_x0_weighted=target_x0_weighted,
        hard_x0_weighted=hard_x0_weighted,
        instruction_rank_weighted=instruction_rank_weighted,
        correct_distance=correct_distance.mean(),
        shuffled_distance=shuffled_distance.mean(),
        instruction_gap=(shuffled_distance - correct_distance).mean(),
        instruction_rank_active_fraction=(
            rank_active.float().sum() / rank_denominator
        ),
        instruction_rank_active_all_fraction=rank_active.float().mean(),
        instruction_rank_eligible_fraction=rank_eligible.float().mean(),
        instruction_rank_slope=(rank_slope * rank_weight).sum() / rank_denominator,
        instruction_rank_margin=(rank_margin * rank_weight).sum() / rank_denominator,
    )


def compute_source_anchor_loss(
    *,
    correct_x0_hat_cont: torch.Tensor,
    source_x0_norm: torch.Tensor,
    x0_target_norm: torch.Tensor,
    target_valid: torch.Tensor,
    hard_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    scale: float,
    relative_margin: float,
) -> SourceAnchorLossBundle:
    """Require the instructed prediction to improve over exact source copying.

    The source baseline is ground truth and detached by construction, so the
    objective cannot be satisfied by making an auxiliary negative branch worse.
    V1 uses only equal-length pairs, where source and target share an exact frame
    correspondence after the paired root/yaw transform.
    """

    if x0_target_norm.ndim != 3 or x0_target_norm.shape[-1] != DIM_HY273:
        raise ValueError("x0_target_norm must have shape [B,T,273]")
    if source_x0_norm.shape != x0_target_norm.shape:
        raise ValueError("source_x0_norm must match x0_target_norm")
    expected = x0_target_norm[..., :CONT_DIM]
    if correct_x0_hat_cont.shape != expected.shape:
        raise ValueError("correct prediction shape does not match target")
    if target_valid.shape != x0_target_norm.shape[:2]:
        raise ValueError("target_valid must have shape [B,T]")
    if hard_mask.shape != x0_target_norm.shape:
        raise ValueError("hard_mask must have shape [B,T,273]")
    if sample_mask.shape != (x0_target_norm.shape[0],):
        raise ValueError("sample_mask must have shape [B]")
    if not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError("scale must be finite and non-negative")
    if not math.isfinite(float(relative_margin)) or float(relative_margin) < 0.0:
        raise ValueError("relative_margin must be finite and non-negative")

    target = expected.to(
        device=correct_x0_hat_cont.device,
        dtype=correct_x0_hat_cont.dtype,
    )
    source = source_x0_norm[..., :CONT_DIM].to(
        device=target.device,
        dtype=target.dtype,
    )
    valid = target_valid.to(device=target.device, dtype=torch.bool)
    unobserved = valid[..., None] & ~hard_mask[..., :CONT_DIM].to(
        device=target.device,
        dtype=torch.bool,
    )
    active = sample_mask.to(device=target.device, dtype=torch.bool)
    active = active & unobserved.reshape(unobserved.shape[0], -1).any(dim=-1)
    correct_values = F.smooth_l1_loss(
        correct_x0_hat_cont, target, reduction="none", beta=1.0
    )
    source_values = F.smooth_l1_loss(source, target, reduction="none", beta=1.0)
    correct_distance = _semantic_distance_per_sample(correct_values, unobserved)
    source_distance = _semantic_distance_per_sample(source_values, unobserved).detach()
    per_sample = torch.relu(
        (1.0 + float(relative_margin)) * correct_distance - source_distance
    )
    active_f = active.to(dtype=per_sample.dtype)
    denominator = active_f.sum().clamp_min(1.0)
    raw = (per_sample * active_f).sum() / denominator
    weighted = raw * float(scale)
    active_correct = (correct_distance * active_f).sum() / denominator
    active_source = (source_distance * active_f).sum() / denominator
    return SourceAnchorLossBundle(
        total=weighted,
        raw=raw,
        weighted=weighted,
        correct_distance=active_correct,
        source_baseline_distance=active_source,
        active_fraction=active_f.mean(),
    )
