"""Physical relation objectives for paired-actor HY273 generation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .hy273_multitask_losses import RatioLossTerm
from .hy273_slices import (
    DIM_HY273,
    HEADING_SLICE,
    ROOT_SLICE,
    reconstruct_global_joints_from_features,
)


@dataclass(frozen=True)
class HY273InteractionLossWeights:
    relative_root: float = 0.02
    relative_heading: float = 0.01
    joint_distance: float = 0.01
    close_joint_vector: float = 0.01
    root_scale_m: float = 0.25
    distance_scale_m: float = 0.10
    close_vector_scale_m: float = 0.05
    distance_gt_threshold_m: float = 1.0
    close_gt_threshold_m: float = 0.20
    min_flow_t: float = 0.20

    def validate(self) -> None:
        for name in (
            "relative_root",
            "relative_heading",
            "joint_distance",
            "close_joint_vector",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "root_scale_m",
            "distance_scale_m",
            "close_vector_scale_m",
            "distance_gt_threshold_m",
            "close_gt_threshold_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= float(self.min_flow_t) < 1.0:
            raise ValueError("min_flow_t must be in [0,1)")


@dataclass
class HY273InteractionLossBundle:
    total: torch.Tensor
    terms: dict[str, RatioLossTerm]
    distance_mask_fraction: torch.Tensor
    close_mask_fraction: torch.Tensor
    active_scene_fraction: torch.Tensor


def _ratio(
    name: str,
    values: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> RatioLossTerm:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values).to(dtype=values.dtype)
    return RatioLossTerm(
        name=name,
        group="interaction_relation",
        numerator=(values * expanded).sum(),
        denominator=expanded.sum(),
        weight=float(weight),
    )


def _relative_heading(heading: torch.Tensor) -> torch.Tensor:
    h0 = heading[:, 0]
    h1 = heading[:, 1]
    return torch.stack(
        (
            h1[..., 0] * h0[..., 0] + h1[..., 1] * h0[..., 1],
            h1[..., 1] * h0[..., 0] - h1[..., 0] * h0[..., 1],
        ),
        dim=-1,
    )


def compute_hy273_interaction_loss(
    *,
    prediction_physical: torch.Tensor,
    target_physical: torch.Tensor,
    actor_valid: torch.Tensor,
    timesteps: torch.Tensor | None = None,
    weights: HY273InteractionLossWeights | None = None,
) -> HY273InteractionLossBundle:
    """Compare actor-relative geometry using masks derived only from GT."""

    weights = weights or HY273InteractionLossWeights()
    weights.validate()
    if prediction_physical.shape != target_physical.shape:
        raise ValueError("Interaction prediction and target shapes differ")
    if (
        prediction_physical.ndim != 4
        or prediction_physical.shape[1] != 2
        or prediction_physical.shape[-1] != DIM_HY273
    ):
        raise ValueError("Interaction tensors must have shape [B,2,T,273]")
    if actor_valid.shape != prediction_physical.shape[:3]:
        raise ValueError("actor_valid must have shape [B,2,T]")
    if timesteps is None:
        active_scene = torch.ones(
            prediction_physical.shape[0],
            device=prediction_physical.device,
            dtype=torch.bool,
        )
    else:
        if timesteps.shape != (prediction_physical.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        active_scene = timesteps.to(device=prediction_physical.device) >= float(
            weights.min_flow_t
        )
    pair_time_valid = actor_valid[:, 0] & actor_valid[:, 1]
    relation_active_scene = active_scene & pair_time_valid.any(dim=1)
    # InterGen applies relation losses only outside its highest-noise region.
    # HY273 flow time runs in the opposite direction, so t >= min_flow_t is the
    # corresponding clean-to-mid-noise subset.
    valid = pair_time_valid & relation_active_scene[:, None]
    prediction = prediction_physical.float()
    target = target_physical.float()

    pred_root_relative = (
        prediction[:, 1, :, ROOT_SLICE] - prediction[:, 0, :, ROOT_SLICE]
    )
    target_root_relative = (
        target[:, 1, :, ROOT_SLICE] - target[:, 0, :, ROOT_SLICE]
    )
    root_values = F.smooth_l1_loss(
        (pred_root_relative - target_root_relative) / float(weights.root_scale_m),
        torch.zeros_like(pred_root_relative),
        reduction="none",
        beta=1.0,
    )

    pred_heading = _relative_heading(prediction[..., HEADING_SLICE])
    target_heading = _relative_heading(target[..., HEADING_SLICE])
    heading_values = F.smooth_l1_loss(
        pred_heading,
        target_heading,
        reduction="none",
        beta=1.0,
    )

    pred_joints = reconstruct_global_joints_from_features(prediction)
    target_joints = reconstruct_global_joints_from_features(target)
    batch, _, frames, joints, _ = pred_joints.shape
    pred_a = pred_joints[:, 0].reshape(batch * frames, joints, 3)
    pred_b = pred_joints[:, 1].reshape(batch * frames, joints, 3)
    target_a = target_joints[:, 0].reshape(batch * frames, joints, 3)
    target_b = target_joints[:, 1].reshape(batch * frames, joints, 3)
    pred_distance = torch.cdist(pred_a, pred_b).reshape(
        batch, frames, joints, joints
    )
    target_distance = torch.cdist(target_a, target_b).reshape(
        batch, frames, joints, joints
    )
    distance_mask = (
        valid[..., None, None]
        & (target_distance.detach() < float(weights.distance_gt_threshold_m))
    )
    distance_values = F.smooth_l1_loss(
        (pred_distance - target_distance) / float(weights.distance_scale_m),
        torch.zeros_like(pred_distance),
        reduction="none",
        beta=1.0,
    )

    pred_vector = (
        pred_joints[:, 0, :, :, None, :]
        - pred_joints[:, 1, :, None, :, :]
    )
    target_vector = (
        target_joints[:, 0, :, :, None, :]
        - target_joints[:, 1, :, None, :, :]
    )
    close_mask = (
        valid[..., None, None]
        & (target_distance.detach() < float(weights.close_gt_threshold_m))
    )
    close_values = F.smooth_l1_loss(
        (pred_vector - target_vector) / float(weights.close_vector_scale_m),
        torch.zeros_like(pred_vector),
        reduction="none",
        beta=1.0,
    )

    terms = {
        "interaction_relative_root": _ratio(
            "interaction_relative_root",
            root_values,
            valid,
            weights.relative_root,
        ),
        "interaction_relative_heading": _ratio(
            "interaction_relative_heading",
            heading_values,
            valid,
            weights.relative_heading,
        ),
        "interaction_joint_distance": _ratio(
            "interaction_joint_distance",
            distance_values,
            distance_mask,
            weights.joint_distance,
        ),
        "interaction_close_joint_vector": _ratio(
            "interaction_close_joint_vector",
            close_values,
            close_mask,
            weights.close_joint_vector,
        ),
    }
    zero = prediction.sum() * 0.0
    total = sum((term.weighted for term in terms.values()), zero)
    valid_pair_count = valid.sum().clamp_min(1).to(dtype=torch.float32)
    pair_entries = valid_pair_count * float(joints * joints)
    return HY273InteractionLossBundle(
        total=total,
        terms=terms,
        distance_mask_fraction=distance_mask.sum().float() / pair_entries,
        close_mask_fraction=close_mask.sum().float() / pair_entries,
        active_scene_fraction=relation_active_scene.float().mean(),
    )
