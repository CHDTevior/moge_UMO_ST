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
    JOINT_POS_SLICE,
    ROOT_SLICE,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
    yaw_rotate_positions,
)


@dataclass(frozen=True)
class HY273InteractionLossWeights:
    # Legacy v1 terms. Reaction-v2 sets the algebraically redundant terms to zero.
    relative_root: float = 0.02
    relative_heading: float = 0.01
    joint_distance: float = 0.01
    close_joint_vector: float = 0.01
    relative_root_radius: float = 0.0
    relative_root_bearing: float = 0.0
    partner_facing: float = 0.0
    soft_proximity: float = 0.0
    false_close: float = 0.0
    scene_proximity: float = 0.0
    precontact_false_close: float = 0.0
    first_contact_cdf: float = 0.0
    fk_contact_map_positive: float = 0.0
    fk_contact_map_negative: float = 0.0
    fk_contact_vector: float = 0.0
    fk_contact_transition: float = 0.0
    root_scale_m: float = 0.25
    heading_beta: float = 1.0
    layout_initial_frames: int = 0
    layout_initial_multiplier: float = 1.0
    layout_precontact_multiplier: float = 1.0
    layout_contact_threshold_m: float = 0.20
    root_radius_scale_m: float = 0.25
    distance_scale_m: float = 0.10
    joint_distance_mode: str = "thresholded_scaled"
    adaptive_distance_eps_m: float = 0.10
    adaptive_distance_beta_m: float = 0.05
    close_vector_scale_m: float = 0.05
    distance_gt_threshold_m: float = 1.0
    close_gt_threshold_m: float = 0.20
    bearing_min_radius_m: float = 0.10
    bearing_eps_m: float = 0.05
    proximity_threshold_m: float = 0.20
    proximity_temperature_m: float = 0.03
    false_close_margin_m: float = 0.08
    false_close_gt_threshold_m: float = 0.20
    false_close_directional_strength: float = 0.025
    precontact_directional_strength: float = 0.25
    overlap_root_fallback_m: float = 1e-4
    fk_contact_threshold_m: float = 0.15
    fk_contact_temperature_m: float = 0.02
    fk_contact_vector_scale_m: float = 0.05
    fk_contact_transition_beta: float = 0.10
    distance_include_predicted_near: bool = False
    min_flow_t: float = 0.20
    coarse_min_flow_t: float | None = None
    fine_min_flow_t: float | None = None

    def validate(self) -> None:
        for name in (
            "relative_root",
            "relative_heading",
            "joint_distance",
            "close_joint_vector",
            "relative_root_radius",
            "relative_root_bearing",
            "partner_facing",
            "soft_proximity",
            "false_close",
            "scene_proximity",
            "precontact_false_close",
            "first_contact_cdf",
            "fk_contact_map_positive",
            "fk_contact_map_negative",
            "fk_contact_vector",
            "fk_contact_transition",
            "false_close_directional_strength",
            "precontact_directional_strength",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "root_scale_m",
            "heading_beta",
            "layout_initial_multiplier",
            "layout_precontact_multiplier",
            "layout_contact_threshold_m",
            "root_radius_scale_m",
            "distance_scale_m",
            "adaptive_distance_eps_m",
            "adaptive_distance_beta_m",
            "close_vector_scale_m",
            "distance_gt_threshold_m",
            "close_gt_threshold_m",
            "bearing_min_radius_m",
            "bearing_eps_m",
            "proximity_threshold_m",
            "proximity_temperature_m",
            "false_close_margin_m",
            "false_close_gt_threshold_m",
            "overlap_root_fallback_m",
            "fk_contact_threshold_m",
            "fk_contact_temperature_m",
            "fk_contact_vector_scale_m",
            "fk_contact_transition_beta",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.layout_initial_frames) != self.layout_initial_frames:
            raise ValueError("layout_initial_frames must be an integer")
        if int(self.layout_initial_frames) < 0:
            raise ValueError("layout_initial_frames must be non-negative")
        for name in ("layout_initial_multiplier", "layout_precontact_multiplier"):
            if float(getattr(self, name)) < 1.0:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 <= float(self.min_flow_t) < 1.0:
            raise ValueError("min_flow_t must be in [0,1)")
        for name in ("coarse_min_flow_t", "fine_min_flow_t"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0,1)")
        if float(self.false_close_margin_m) >= float(
            self.false_close_gt_threshold_m
        ):
            raise ValueError(
                "false_close_margin_m must be below false_close_gt_threshold_m"
            )
        if self.joint_distance_mode not in {
            "thresholded_scaled",
            "adaptive_gt_inverse",
        }:
            raise ValueError(
                "joint_distance_mode must be 'thresholded_scaled' or "
                "'adaptive_gt_inverse'"
            )

    @property
    def coarse_gate(self) -> float:
        return float(
            self.min_flow_t
            if self.coarse_min_flow_t is None
            else self.coarse_min_flow_t
        )

    @property
    def fine_gate(self) -> float:
        return float(
            self.min_flow_t if self.fine_min_flow_t is None else self.fine_min_flow_t
        )


@dataclass
class HY273InteractionLossBundle:
    total: torch.Tensor
    terms: dict[str, RatioLossTerm]
    distance_mask_fraction: torch.Tensor
    close_mask_fraction: torch.Tensor
    active_scene_fraction: torch.Tensor
    coarse_active_scene_fraction: torch.Tensor
    fine_active_scene_fraction: torch.Tensor
    proximity_positive_fraction: torch.Tensor
    false_close_fraction: torch.Tensor
    diagnostic_ratios: dict[str, tuple[torch.Tensor, torch.Tensor]]


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
    h0 = F.normalize(heading[:, 0].float(), dim=-1, eps=1e-6)
    h1 = F.normalize(heading[:, 1].float(), dim=-1, eps=1e-6)
    return torch.stack(
        (
            h1[..., 0] * h0[..., 0] + h1[..., 1] * h0[..., 1],
            h1[..., 1] * h0[..., 0] - h1[..., 0] * h0[..., 1],
        ),
        dim=-1,
    )


def _source_local_root(
    relative_root: torch.Tensor,
    source_heading: torch.Tensor,
) -> torch.Tensor:
    """Express a signed world-space root vector in the source heading frame."""

    return _vectors_in_source_heading_frame(relative_root, source_heading)


def _vectors_in_source_heading_frame(
    vectors: torch.Tensor,
    source_heading: torch.Tensor,
) -> torch.Tensor:
    """Rotate 3D vectors into the observed source actor's local yaw frame."""

    heading = F.normalize(source_heading.float(), dim=-1, eps=1e-6)
    source_yaw = torch.atan2(heading[..., 1], heading[..., 0])
    return yaw_rotate_positions(vectors.float(), -source_yaw)


def _layout_phase_weights(
    *,
    pair_time_valid: torch.Tensor,
    target_distance: torch.Tensor,
    initial_frames: int,
    initial_multiplier: float,
    precontact_multiplier: float,
    contact_threshold_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weight initial layout and the phase before the first GT close event."""

    if target_distance.shape[:2] != pair_time_valid.shape:
        raise ValueError("target_distance and pair_time_valid shapes disagree")
    frames = pair_time_valid.shape[1]
    frame_index = torch.arange(frames, device=pair_time_valid.device)[None, :]
    gt_close = (
        target_distance.detach().amin(dim=(-1, -2)) < float(contact_threshold_m)
    ) & pair_time_valid
    first_close = torch.where(gt_close, frame_index, frames).amin(dim=1)
    precontact = pair_time_valid & (frame_index < first_close[:, None])
    initial = pair_time_valid & (frame_index < min(int(initial_frames), frames))

    phase = torch.ones_like(pair_time_valid, dtype=target_distance.dtype)
    phase = torch.where(
        precontact,
        phase.new_tensor(float(precontact_multiplier)),
        phase,
    )
    phase = torch.where(
        initial,
        torch.maximum(phase, phase.new_tensor(float(initial_multiplier))),
        phase,
    )
    return phase, initial, precontact, gt_close.any(dim=1)


def _soft_unit(vector: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    radius = torch.linalg.vector_norm(vector, dim=-1)
    return vector / radius.clamp_min(float(eps)).unsqueeze(-1), radius


def _bernoulli_kl_from_logits(
    prediction_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(Bernoulli(target) || Bernoulli(prediction)) without FP32 cancellation."""

    prediction = prediction_logits.double()
    target = target_logits.detach().double()
    target_probability = torch.sigmoid(target)
    kl = target_probability * (
        F.logsigmoid(target) - F.logsigmoid(prediction)
    ) + (1.0 - target_probability) * (
        F.logsigmoid(-target) - F.logsigmoid(-prediction)
    )
    return kl.clamp_min(0.0).to(dtype=prediction_logits.dtype)


def _facing_partner_descriptor(
    local_root: torch.Tensor,
    relative_heading: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Cos/sin of reactor facing relative to the direction toward the source."""

    to_source, _ = _soft_unit(-local_root, eps)
    heading = F.normalize(relative_heading, dim=-1, eps=1e-6)
    forward = torch.stack((heading[..., 1], heading[..., 0]), dim=-1)
    dot = (forward * to_source).sum(dim=-1)
    cross = forward[..., 0] * to_source[..., 1] - forward[..., 1] * to_source[..., 0]
    return torch.stack((dot, cross), dim=-1)


def compute_hy273_interaction_loss(
    *,
    prediction_physical: torch.Tensor,
    target_physical: torch.Tensor,
    actor_valid: torch.Tensor,
    timesteps: torch.Tensor | None = None,
    weights: HY273InteractionLossWeights | None = None,
) -> HY273InteractionLossBundle:
    """Compare fixed-role actor geometry with separate coarse and fine gates."""

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
        legacy_active_scene = torch.ones(
            prediction_physical.shape[0],
            device=prediction_physical.device,
            dtype=torch.bool,
        )
        coarse_active_scene = legacy_active_scene
        fine_active_scene = legacy_active_scene
    else:
        if timesteps.shape != (prediction_physical.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        local_timesteps = timesteps.to(device=prediction_physical.device)
        legacy_active_scene = local_timesteps >= float(
            weights.min_flow_t
        )
        coarse_active_scene = local_timesteps >= weights.coarse_gate
        fine_active_scene = local_timesteps >= weights.fine_gate
    pair_time_valid = actor_valid[:, 0] & actor_valid[:, 1]
    pair_scene_valid = pair_time_valid.any(dim=1)
    legacy_active_scene = legacy_active_scene & pair_scene_valid
    coarse_active_scene = coarse_active_scene & pair_scene_valid
    fine_active_scene = fine_active_scene & pair_scene_valid
    legacy_valid = pair_time_valid & legacy_active_scene[:, None]
    coarse_valid = pair_time_valid & coarse_active_scene[:, None]
    fine_valid = pair_time_valid & fine_active_scene[:, None]
    prediction = prediction_physical.float()
    target = target_physical.float()

    pred_root_relative = (
        prediction[:, 1, :, ROOT_SLICE] - prediction[:, 0, :, ROOT_SLICE]
    )
    target_root_relative = (
        target[:, 1, :, ROOT_SLICE] - target[:, 0, :, ROOT_SLICE]
    )
    # Reaction observes the source actor. Both prediction and target use the same
    # detached source-heading frame, including signed vertical displacement. The
    # source translation still cancels algebraically; the useful intervention is
    # source-aligned axes plus the phase reweighting applied below.
    source_heading_frame = target[:, 0, :, HEADING_SLICE].detach()
    pred_root_local_3d = _source_local_root(
        pred_root_relative, source_heading_frame
    )
    target_root_local_3d = _source_local_root(
        target_root_relative, source_heading_frame
    ).detach()
    root_values = F.smooth_l1_loss(
        (pred_root_local_3d - target_root_local_3d)
        / float(weights.root_scale_m),
        torch.zeros_like(pred_root_local_3d),
        reduction="none",
        beta=1.0,
    )

    pred_root_local = pred_root_local_3d[..., (0, 2)]
    target_root_local = target_root_local_3d[..., (0, 2)]
    pred_bearing, pred_radius = _soft_unit(
        pred_root_local, float(weights.bearing_eps_m)
    )
    target_bearing, target_radius = _soft_unit(
        target_root_local, float(weights.bearing_eps_m)
    )
    radius_values = F.smooth_l1_loss(
        (pred_radius - target_radius) / float(weights.root_radius_scale_m),
        torch.zeros_like(pred_radius),
        reduction="none",
        beta=1.0,
    )
    bearing_values = F.smooth_l1_loss(
        pred_bearing,
        target_bearing,
        reduction="none",
        beta=1.0,
    )
    bearing_valid = coarse_valid & (
        target_radius.detach() >= float(weights.bearing_min_radius_m)
    )

    pred_heading = _relative_heading(prediction[..., HEADING_SLICE])
    target_heading = _relative_heading(target[..., HEADING_SLICE])
    heading_values = F.smooth_l1_loss(
        pred_heading,
        target_heading,
        reduction="none",
        beta=float(weights.heading_beta),
    )
    pred_facing = _facing_partner_descriptor(
        pred_root_local, pred_heading, float(weights.bearing_eps_m)
    )
    target_facing = _facing_partner_descriptor(
        target_root_local, target_heading, float(weights.bearing_eps_m)
    )
    facing_values = F.smooth_l1_loss(
        pred_facing,
        target_facing,
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
    detached_target_distance = target_distance.detach()
    (
        layout_phase_weight,
        layout_initial,
        layout_precontact,
        gt_contact_scene,
    ) = _layout_phase_weights(
        pair_time_valid=pair_time_valid,
        target_distance=detached_target_distance,
        initial_frames=int(weights.layout_initial_frames),
        initial_multiplier=float(weights.layout_initial_multiplier),
        precontact_multiplier=float(weights.layout_precontact_multiplier),
        contact_threshold_m=float(weights.layout_contact_threshold_m),
    )
    # These low-dimensional layout anchors use the coarse flow-time gate. Float
    # phase weights are normalized by _ratio, so the configured coefficients
    # retain their average scale while supervision shifts toward setup frames.
    layout_ratio_mask = (
        coarse_valid.to(dtype=layout_phase_weight.dtype) * layout_phase_weight
    )
    bearing_layout_mask = (
        bearing_valid.to(dtype=layout_phase_weight.dtype) * layout_phase_weight
    )
    target_distance_near = detached_target_distance < float(
        weights.distance_gt_threshold_m
    )
    distance_near = target_distance_near
    if weights.distance_include_predicted_near:
        distance_near = distance_near | (
            pred_distance.detach() < float(weights.distance_gt_threshold_m)
        )
    if weights.joint_distance_mode == "adaptive_gt_inverse":
        # Every inter-actor joint pair remains supervised. GT-near pairs receive
        # more weight without turning GT-far false collisions into blind spots.
        distance_mask = fine_valid[..., None, None].expand_as(pred_distance)
        adaptive_distance_weight = 1.0 / (
            detached_target_distance + float(weights.adaptive_distance_eps_m)
        )
        distance_ratio_mask = (
            distance_mask.to(dtype=pred_distance.dtype) * adaptive_distance_weight
        )
        distance_values = F.smooth_l1_loss(
            pred_distance,
            detached_target_distance,
            reduction="none",
            beta=float(weights.adaptive_distance_beta_m),
        )
    else:
        distance_mask = fine_valid[..., None, None] & distance_near
        distance_ratio_mask = distance_mask
        distance_values = F.smooth_l1_loss(
            (pred_distance - detached_target_distance)
            / float(weights.distance_scale_m),
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
    # Match the source recipe exactly: coordinate-wise SmoothL1 is evaluated in
    # the fixed observed source's heading frame, not the random world-yaw frame.
    close_source_heading = target[:, 0, :, HEADING_SLICE].detach()
    pred_close_vector = _vectors_in_source_heading_frame(
        pred_vector, close_source_heading
    )
    target_close_vector = _vectors_in_source_heading_frame(
        target_vector, close_source_heading
    ).detach()
    close_mask = (
        fine_valid[..., None, None]
        & (target_distance.detach() < float(weights.close_gt_threshold_m))
    )
    close_values = F.smooth_l1_loss(
        (pred_close_vector - target_close_vector)
        / float(weights.close_vector_scale_m),
        torch.zeros_like(pred_close_vector),
        reduction="none",
        beta=1.0,
    )

    # Full-contact lifecycle supervision follows the FK path used by rendering
    # and headline Reaction metrics. Positive and negative pair-time entries are
    # normalized independently because true 15cm contacts are extremely sparse.
    fk_contact_enabled = any(
        float(value) > 0.0
        for value in (
            weights.fk_contact_map_positive,
            weights.fk_contact_map_negative,
            weights.fk_contact_vector,
            weights.fk_contact_transition,
        )
    )
    if fk_contact_enabled:
        pred_fk_joints = fk_positions_from_global_rot6d(
            prediction.reshape(batch * 2, frames, DIM_HY273)
        ).reshape(batch, 2, frames, joints, 3)
        target_fk_joints = fk_positions_from_global_rot6d(
            target.reshape(batch * 2, frames, DIM_HY273)
        ).reshape(batch, 2, frames, joints, 3).detach()
        pred_fk_vector = (
            pred_fk_joints[:, 0, :, :, None, :]
            - pred_fk_joints[:, 1, :, None, :, :]
        )
        target_fk_vector = (
            target_fk_joints[:, 0, :, :, None, :]
            - target_fk_joints[:, 1, :, None, :, :]
        )
        pred_fk_distance = torch.linalg.vector_norm(pred_fk_vector, dim=-1)
        target_fk_distance = torch.linalg.vector_norm(
            target_fk_vector, dim=-1
        ).detach()
        fk_contact_threshold = float(weights.fk_contact_threshold_m)
        fk_contact_temperature = float(weights.fk_contact_temperature_m)
        # A Euclidean norm has no separation direction at exact overlap. For
        # GT-negative pairs only, preserve the true forward distance while
        # routing the overlap gradient through the reactor root along the GT
        # pair direction. Normal non-overlap gradients remain unchanged.
        target_fk_direction = target_fk_vector / target_fk_distance.clamp_min(
            1e-6
        ).unsqueeze(-1)
        # FK translation is carried by smooth-root XZ and pelvis-position Y.
        # Route overlap separation through those channels, avoiding unrelated
        # local-joint deformation and the FK-inert smooth-root Y channel.
        pred_fk_root_carrier = torch.stack(
            (
                prediction[..., ROOT_SLICE.start],
                prediction[..., JOINT_POS_SLICE.start + 1],
                prediction[..., ROOT_SLICE.start + 2],
            ),
            dim=-1,
        )
        pred_root_pair_vector = (
            pred_fk_root_carrier[:, 0].detach() - pred_fk_root_carrier[:, 1]
        )
        root_target_separation = (
            pred_root_pair_vector[..., None, None, :] * target_fk_direction
        ).sum(dim=-1)
        root_distance_surrogate = (
            pred_fk_distance.detach()
            + root_target_separation
            - root_target_separation.detach()
        )
        fk_overlap_negative = (
            pred_fk_distance.detach() < float(weights.overlap_root_fallback_m)
        ) & (target_fk_distance >= fk_contact_threshold)
        pred_fk_distance = torch.where(
            fk_overlap_negative,
            root_distance_surrogate,
            pred_fk_distance,
        )
        pred_fk_contact_logits = (
            fk_contact_threshold - pred_fk_distance
        ) / fk_contact_temperature
        target_fk_contact_logits = (
            fk_contact_threshold - target_fk_distance
        ) / fk_contact_temperature
        fk_contact_map_values = _bernoulli_kl_from_logits(
            pred_fk_contact_logits,
            target_fk_contact_logits,
        )
        target_fk_contact = target_fk_distance < fk_contact_threshold
        predicted_fk_contact = (
            pred_fk_distance.detach() < fk_contact_threshold
        )
        fk_contact_positive_mask = (
            fine_valid[..., None, None] & target_fk_contact
        )
        fk_contact_negative_mask = (
            fine_valid[..., None, None] & ~target_fk_contact
        )
        fk_predicted_contact_mask = (
            fine_valid[..., None, None] & predicted_fk_contact
        )

        pred_fk_vector_local = _vectors_in_source_heading_frame(
            pred_fk_vector, close_source_heading
        )
        target_fk_vector_local = _vectors_in_source_heading_frame(
            target_fk_vector, close_source_heading
        ).detach()
        fk_contact_vector_values = F.smooth_l1_loss(
            (pred_fk_vector_local - target_fk_vector_local)
            / float(weights.fk_contact_vector_scale_m),
            torch.zeros_like(pred_fk_vector_local),
            reduction="none",
            beta=1.0,
        )

        pred_fk_contact_probability = torch.sigmoid(pred_fk_contact_logits)
        target_fk_contact_probability = torch.sigmoid(
            target_fk_contact_logits
        ).detach()
        pred_fk_contact_delta = (
            pred_fk_contact_probability[:, 1:]
            - pred_fk_contact_probability[:, :-1]
        )
        target_fk_contact_delta = (
            target_fk_contact_probability[:, 1:]
            - target_fk_contact_probability[:, :-1]
        )
        fk_contact_transition_values = F.smooth_l1_loss(
            pred_fk_contact_delta,
            target_fk_contact_delta,
            reduction="none",
            beta=float(weights.fk_contact_transition_beta),
        )
        fk_transition_valid = fine_valid[:, 1:] & fine_valid[:, :-1]
        fk_transition_context = (
            target_fk_contact[:, 1:]
            | target_fk_contact[:, :-1]
            | predicted_fk_contact[:, 1:]
            | predicted_fk_contact[:, :-1]
        )
        fk_contact_transition_mask = (
            fk_transition_valid[..., None, None] & fk_transition_context
        )
    else:
        fk_contact_map_values = pred_distance * 0.0
        fk_contact_vector_values = pred_vector * 0.0
        fk_contact_transition_values = pred_distance[:, 1:] * 0.0
        fk_contact_positive_mask = torch.zeros_like(
            pred_distance, dtype=torch.bool
        )
        fk_contact_negative_mask = torch.zeros_like(
            pred_distance, dtype=torch.bool
        )
        fk_predicted_contact_mask = torch.zeros_like(
            pred_distance, dtype=torch.bool
        )
        fk_contact_transition_mask = torch.zeros_like(
            pred_distance[:, 1:], dtype=torch.bool
        )

    # Event-level objectives use the same 20 cm scene definition as evaluation.
    # Unlike the 22x22 pair losses, these terms cannot be diluted by hundreds of
    # easy far-apart joint pairs.  They use the coarse gate so low-t samples must
    # infer layout and contact timing from source/text rather than target leakage.
    scene_threshold = float(weights.layout_contact_threshold_m)
    scene_temperature = float(weights.proximity_temperature_m)
    pred_distance_flat = pred_distance.reshape(batch, frames, joints * joints)
    target_distance_flat = detached_target_distance.reshape(
        batch, frames, joints * joints
    )
    pred_scene_distance, pred_closest_pair = pred_distance_flat.min(dim=-1)
    target_scene_distance = target_distance_flat.amin(dim=-1)
    target_scene_close = target_scene_distance < scene_threshold
    scene_contact_logits = (
        scene_threshold - pred_scene_distance
    ) / scene_temperature
    target_scene_contact_logits = (
        scene_threshold - target_scene_distance
    ) / scene_temperature
    scene_proximity_values = _bernoulli_kl_from_logits(
        scene_contact_logits,
        target_scene_contact_logits,
    )
    scene_proximity_positive = coarse_valid & target_scene_close
    scene_proximity_negative = coarse_valid & ~target_scene_close

    predicted_close_hard = (
        (pred_scene_distance < scene_threshold) & pair_time_valid
    )
    predicted_first_contact_hard_cdf = (
        predicted_close_hard.to(dtype=torch.int64).cumsum(dim=1) > 0
    ).to(dtype=pred_scene_distance.dtype)
    # Forward uses the benchmark's exact hard 20 cm event. Backward uses the
    # closest distance seen in each prefix. Applying the straight-through
    # estimator after the prefix reduction avoids a hard cumprod Jacobian whose
    # gradient changes when identical contacts repeat after the first event.
    valid_scene_distance = torch.where(
        pair_time_valid,
        pred_scene_distance,
        torch.full_like(pred_scene_distance, torch.inf),
    )
    predicted_prefix_min_distance = torch.cummin(
        valid_scene_distance,
        dim=1,
    ).values
    predicted_first_contact_soft_cdf = torch.sigmoid(
        (scene_threshold - predicted_prefix_min_distance) / scene_temperature
    )
    predicted_first_contact_cdf = (
        predicted_first_contact_hard_cdf
        + predicted_first_contact_soft_cdf
        - predicted_first_contact_soft_cdf.detach()
    )
    target_first_contact_hard_cdf = (
        (target_scene_close & pair_time_valid).to(dtype=torch.int64).cumsum(dim=1)
        > 0
    )
    target_first_contact_cdf = target_first_contact_hard_cdf.to(
        dtype=predicted_first_contact_cdf.dtype
    )
    first_contact_cdf_values = F.smooth_l1_loss(
        predicted_first_contact_cdf,
        target_first_contact_cdf,
        reduction="none",
        beta=0.10,
    )

    flat_pred_vector = pred_vector.reshape(batch, frames, joints * joints, 3)
    flat_target_vector = target_vector.reshape(batch, frames, joints * joints, 3)
    closest_index = pred_closest_pair[..., None, None].expand(-1, -1, 1, 3)
    closest_pred_vector = torch.gather(
        flat_pred_vector, dim=2, index=closest_index
    ).squeeze(2)
    closest_target_vector = torch.gather(
        flat_target_vector, dim=2, index=closest_index
    ).squeeze(2).detach()
    closest_target_direction = closest_target_vector / torch.linalg.vector_norm(
        closest_target_vector, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    predicted_target_separation_scene = (
        closest_pred_vector * closest_target_direction
    ).sum(dim=-1)
    # At exact overlap cdist has no radial direction. Route the target-directed
    # fallback through the reactor root only, so resolving a collision cannot
    # win by distorting whichever local joint happened to be the argmin tie.
    pred_root_pair_vector = (
        prediction[:, 0, :, ROOT_SLICE].detach()
        - prediction[:, 1, :, ROOT_SLICE]
    )
    root_only_target_separation = (
        pred_root_pair_vector * closest_target_direction
    ).sum(dim=-1)
    root_only_target_separation = (
        predicted_target_separation_scene.detach()
        + root_only_target_separation
        - root_only_target_separation.detach()
    )
    predicted_target_separation_scene = torch.where(
        pred_scene_distance.detach() < float(weights.overlap_root_fallback_m),
        root_only_target_separation,
        predicted_target_separation_scene,
    )
    precontact_radial = (
        F.relu(scene_threshold - pred_scene_distance) / scene_threshold
    ).square()
    precontact_directional = (
        F.relu(scene_threshold - predicted_target_separation_scene)
        / scene_threshold
    ).square()
    precontact_directional = precontact_directional * (
        pred_scene_distance.detach() < scene_threshold
    ).to(dtype=precontact_directional.dtype)
    precontact_false_close_values = precontact_radial + float(
        weights.precontact_directional_strength
    ) * precontact_directional
    precontact_false_close_mask = coarse_valid & layout_precontact

    proximity_target = torch.sigmoid(
        (float(weights.proximity_threshold_m) - target_distance)
        / float(weights.proximity_temperature_m)
    ).detach()
    proximity_prediction = torch.sigmoid(
        (float(weights.proximity_threshold_m) - pred_distance)
        / float(weights.proximity_temperature_m)
    )
    proximity_values = F.smooth_l1_loss(
        proximity_prediction,
        proximity_target,
        reduction="none",
        beta=1.0,
    )
    proximity_positive = fine_valid[..., None, None] & (
        target_distance.detach() < float(weights.proximity_threshold_m)
    )
    proximity_negative = (
        fine_valid[..., None, None]
        & (
            target_distance.detach()
            >= float(weights.proximity_threshold_m)
        )
        & distance_near
    )
    false_close_mask = (
        fine_valid[..., None, None]
        & (
            target_distance.detach()
            >= float(weights.false_close_gt_threshold_m)
        )
        & (pred_distance.detach() < float(weights.false_close_margin_m))
    )
    # A pure distance hinge has zero gradient at exact overlap because cdist has no
    # preferred separation direction there.  The GT-far mask gives us a physically
    # meaningful direction: push the offending pair apart along its target vector.
    target_direction = target_vector / target_distance.clamp_min(1e-6).unsqueeze(-1)
    predicted_target_separation = (pred_vector * target_direction).sum(dim=-1)
    false_close_radial = (
        F.relu(float(weights.false_close_margin_m) - pred_distance)
        / float(weights.false_close_margin_m)
    ).square()
    false_close_directional = (
        F.relu(
            float(weights.false_close_margin_m) - predicted_target_separation
        )
        / float(weights.false_close_margin_m)
    ).square()
    false_close_values = false_close_radial + float(
        weights.false_close_directional_strength
    ) * false_close_directional

    terms = {
        "interaction_relative_root": _ratio(
            "interaction_relative_root",
            root_values,
            layout_ratio_mask,
            weights.relative_root,
        ),
        "interaction_relative_heading": _ratio(
            "interaction_relative_heading",
            heading_values,
            layout_ratio_mask,
            weights.relative_heading,
        ),
        "interaction_joint_distance": _ratio(
            "interaction_joint_distance",
            distance_values,
            distance_ratio_mask,
            weights.joint_distance,
        ),
        "interaction_close_joint_vector": _ratio(
            "interaction_close_joint_vector",
            close_values,
            close_mask,
            weights.close_joint_vector,
        ),
        "interaction_relative_root_radius": _ratio(
            "interaction_relative_root_radius",
            radius_values,
            layout_ratio_mask,
            weights.relative_root_radius,
        ),
        "interaction_relative_root_bearing": _ratio(
            "interaction_relative_root_bearing",
            bearing_values,
            bearing_layout_mask,
            weights.relative_root_bearing,
        ),
        "interaction_partner_facing": _ratio(
            "interaction_partner_facing",
            facing_values,
            bearing_layout_mask,
            weights.partner_facing,
        ),
        "interaction_soft_proximity_positive": _ratio(
            "interaction_soft_proximity_positive",
            proximity_values,
            proximity_positive,
            0.5 * float(weights.soft_proximity),
        ),
        "interaction_soft_proximity_negative": _ratio(
            "interaction_soft_proximity_negative",
            proximity_values,
            proximity_negative,
            0.5 * float(weights.soft_proximity),
        ),
        "interaction_false_close": _ratio(
            "interaction_false_close",
            false_close_values,
            false_close_mask,
            weights.false_close,
        ),
        "interaction_scene_proximity_positive": _ratio(
            "interaction_scene_proximity_positive",
            scene_proximity_values,
            scene_proximity_positive,
            0.5 * float(weights.scene_proximity),
        ),
        "interaction_scene_proximity_negative": _ratio(
            "interaction_scene_proximity_negative",
            scene_proximity_values,
            scene_proximity_negative,
            0.5 * float(weights.scene_proximity),
        ),
        "interaction_precontact_false_close": _ratio(
            "interaction_precontact_false_close",
            precontact_false_close_values,
            precontact_false_close_mask,
            weights.precontact_false_close,
        ),
        "interaction_first_contact_cdf": _ratio(
            "interaction_first_contact_cdf",
            first_contact_cdf_values,
            coarse_valid,
            weights.first_contact_cdf,
        ),
        "interaction_fk_contact_map_positive": _ratio(
            "interaction_fk_contact_map_positive",
            fk_contact_map_values,
            fk_contact_positive_mask,
            weights.fk_contact_map_positive,
        ),
        "interaction_fk_contact_map_negative": _ratio(
            "interaction_fk_contact_map_negative",
            fk_contact_map_values,
            fk_contact_negative_mask,
            weights.fk_contact_map_negative,
        ),
        "interaction_fk_contact_vector": _ratio(
            "interaction_fk_contact_vector",
            fk_contact_vector_values,
            fk_contact_positive_mask,
            weights.fk_contact_vector,
        ),
        "interaction_fk_contact_transition": _ratio(
            "interaction_fk_contact_transition",
            fk_contact_transition_values,
            fk_contact_transition_mask,
            weights.fk_contact_transition,
        ),
    }
    zero = prediction.sum() * 0.0
    total = sum((term.weighted for term in terms.values()), zero)
    fine_pair_count = fine_valid.sum().to(dtype=torch.float32)
    pair_entries = fine_pair_count * float(joints * joints)
    safe_pair_entries = pair_entries.clamp_min(1.0)
    scene_entries = prediction.new_tensor(float(batch), dtype=torch.float32)
    diagnostic_ratios = {
        "layout_initial_frame_fraction": (
            layout_initial.sum().float(),
            pair_time_valid.sum().float(),
        ),
        "layout_precontact_frame_fraction": (
            layout_precontact.sum().float(),
            pair_time_valid.sum().float(),
        ),
        "layout_phase_weight_mean": (
            (layout_phase_weight * pair_time_valid).sum().float(),
            pair_time_valid.sum().float(),
        ),
        "gt_contact_scene_fraction": (
            gt_contact_scene.sum().float(),
            pair_scene_valid.sum().float(),
        ),
        "distance_mask_fraction": (distance_mask.sum().float(), pair_entries),
        "adaptive_distance_weight_mean": (
            distance_ratio_mask.sum().float(),
            distance_mask.sum().float(),
        ),
        "close_mask_fraction": (close_mask.sum().float(), pair_entries),
        "relation_active_scene_fraction": (
            fine_active_scene.sum().float(),
            scene_entries,
        ),
        "coarse_active_scene_fraction": (
            coarse_active_scene.sum().float(),
            scene_entries,
        ),
        "fine_active_scene_fraction": (
            fine_active_scene.sum().float(),
            scene_entries,
        ),
        "proximity_positive_fraction": (
            proximity_positive.sum().float(),
            pair_entries,
        ),
        "false_close_fraction": (false_close_mask.sum().float(), pair_entries),
        "scene_close_positive_frame_fraction": (
            scene_proximity_positive.sum().float(),
            coarse_valid.sum().float(),
        ),
        "precontact_false_close_frame_fraction": (
            (
                precontact_false_close_mask
                & (pred_scene_distance.detach() < scene_threshold)
            ).sum().float(),
            precontact_false_close_mask.sum().float(),
        ),
        "first_contact_cdf_positive_frame_fraction": (
            (target_first_contact_hard_cdf & coarse_valid).sum().float(),
            coarse_valid.sum().float(),
        ),
        "fk_contact_positive_pair_fraction": (
            fk_contact_positive_mask.sum().float(),
            pair_entries,
        ),
        "fk_contact_predicted_positive_pair_fraction": (
            fk_predicted_contact_mask.sum().float(),
            pair_entries,
        ),
        "fk_contact_transition_pair_fraction": (
            fk_contact_transition_mask.sum().float(),
            (
                (fine_valid[:, 1:] & fine_valid[:, :-1]).sum().float()
                * float(joints * joints)
            ),
        ),
    }
    return HY273InteractionLossBundle(
        total=total,
        terms=terms,
        distance_mask_fraction=distance_mask.sum().float() / safe_pair_entries,
        close_mask_fraction=close_mask.sum().float() / safe_pair_entries,
        active_scene_fraction=fine_active_scene.float().mean(),
        coarse_active_scene_fraction=coarse_active_scene.float().mean(),
        fine_active_scene_fraction=fine_active_scene.float().mean(),
        proximity_positive_fraction=(
            proximity_positive.sum().float() / safe_pair_entries
        ),
        false_close_fraction=false_close_mask.sum().float() / safe_pair_entries,
        diagnostic_ratios=diagnostic_ratios,
    )
