"""Fixed-role physical metrics for actor-conditioned HY273 Reaction."""

from __future__ import annotations

from typing import Any

import torch

from .hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    HEADING_SLICE,
    ROOT_SLICE,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)


def _batched(value: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    squeeze = value.ndim == 2
    if squeeze:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[-1] != DIM_HY273:
        raise ValueError(f"{name} must have shape [B,T,{DIM_HY273}] or [T,{DIM_HY273}]")
    return value.float(), squeeze


def _sample_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    numerator = (values * mask.to(values.dtype)).flatten(1).sum(dim=1)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1)
    return numerator / denominator


def _heading_angle(features: torch.Tensor) -> torch.Tensor:
    heading = features[..., HEADING_SLICE]
    return torch.atan2(heading[..., 1], heading[..., 0])


def _angular_error_deg(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    delta = left - right
    return torch.atan2(torch.sin(delta), torch.cos(delta)).abs() * (180.0 / torch.pi)


def _jerk(joints: torch.Tensor, fps: float) -> torch.Tensor:
    return torch.diff(joints, n=3, dim=1) * float(fps) ** 3


def reaction_fixed_role_metrics(
    source_actor: torch.Tensor,
    prediction_reactor: torch.Tensor,
    target_reactor: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    fps: float = 30.0,
) -> dict[str, Any]:
    """Evaluate ``source actor -> reactor`` without actor permutation matching.

    The observed source is never scored as a generated output. Pairwise metrics use
    it only as a fixed reference, so copying the source cannot be hidden by swapping
    actor identities or averaging one exact source actor with one failed reactor.
    """

    source, source_unbatched = _batched(source_actor, "source_actor")
    prediction, prediction_unbatched = _batched(
        prediction_reactor, "prediction_reactor"
    )
    target, target_unbatched = _batched(target_reactor, "target_reactor")
    if len({source_unbatched, prediction_unbatched, target_unbatched}) != 1:
        raise ValueError("Reaction tensors must use the same batching convention")
    if source.shape != prediction.shape or source.shape != target.shape:
        raise ValueError("Reaction source, prediction, and target shapes differ")
    if not all(bool(torch.isfinite(value).all()) for value in (source, prediction, target)):
        raise ValueError("Reaction metrics require finite tensors")
    if not torch.isfinite(torch.tensor(float(fps))) or float(fps) <= 0.0:
        raise ValueError("fps must be finite and positive")

    batch, frames, _ = source.shape
    if lengths is None:
        lengths = torch.full(
            (batch,), frames, device=source.device, dtype=torch.long
        )
    else:
        lengths = lengths.to(device=source.device, dtype=torch.long).reshape(-1)
        if lengths.shape != (batch,):
            raise ValueError("lengths must have shape [B]")
        if bool(((lengths < 1) | (lengths > frames)).any()):
            raise ValueError("lengths must be in [1,T]")
    valid = torch.arange(frames, device=source.device)[None] < lengths[:, None]

    source_pos = reconstruct_global_joints_from_features(source)
    prediction_pos = reconstruct_global_joints_from_features(prediction)
    target_pos = reconstruct_global_joints_from_features(target)
    source_fk = fk_positions_from_global_rot6d(source)
    prediction_fk = fk_positions_from_global_rot6d(prediction)
    target_fk = fk_positions_from_global_rot6d(target)

    position_mpjpe = _sample_mean(
        (prediction_pos - target_pos).norm(dim=-1), valid
    )
    fk_mpjpe = _sample_mean((prediction_fk - target_fk).norm(dim=-1), valid)
    root_error = _sample_mean(
        (prediction[..., ROOT_SLICE] - target[..., ROOT_SLICE]).norm(dim=-1),
        valid,
    )

    source_heading = _heading_angle(source)
    prediction_heading = _heading_angle(prediction)
    target_heading = _heading_angle(target)
    relative_heading_error = _sample_mean(
        _angular_error_deg(
            prediction_heading - source_heading,
            target_heading - source_heading,
        ),
        valid,
    )
    prediction_heading_norm = torch.linalg.vector_norm(
        prediction[..., HEADING_SLICE], dim=-1
    )
    heading_norm_error = _sample_mean(
        (prediction_heading_norm - 1.0).abs(), valid
    )

    flat_shape = (batch * frames, 22, 3)
    pred_position_distance = torch.cdist(
        source_pos.reshape(flat_shape), prediction_pos.reshape(flat_shape)
    ).reshape(batch, frames, 22, 22)
    target_position_distance = torch.cdist(
        source_pos.reshape(flat_shape), target_pos.reshape(flat_shape)
    ).reshape(batch, frames, 22, 22)
    pred_fk_distance = torch.cdist(
        source_fk.reshape(flat_shape), prediction_fk.reshape(flat_shape)
    ).reshape(batch, frames, 22, 22)
    target_fk_distance = torch.cdist(
        source_fk.reshape(flat_shape), target_fk.reshape(flat_shape)
    ).reshape(batch, frames, 22, 22)

    prediction_min_distance = pred_position_distance.flatten(2).amin(dim=-1)
    target_min_distance = target_position_distance.flatten(2).amin(dim=-1)
    relation_position_mae = _sample_mean(
        (pred_position_distance - target_position_distance).abs(), valid
    )
    relation_fk_mae = _sample_mean(
        (pred_fk_distance - target_fk_distance).abs(), valid
    )
    close_event_error = _sample_mean(
        (
            (prediction_min_distance < 0.20)
            != (target_min_distance < 0.20)
        ).float(),
        valid,
    )

    contact_probability = prediction[..., CONTACT_SLICE].clamp(0.0, 1.0)
    target_contact = target[..., CONTACT_SLICE] >= 0.5
    predicted_contact = contact_probability >= 0.5
    contact_accuracy = _sample_mean(
        (predicted_contact == target_contact).float(), valid
    )
    contact_valid = valid[..., None]
    tp = (predicted_contact & target_contact & contact_valid).sum((1, 2)).float()
    fp = (predicted_contact & ~target_contact & contact_valid).sum((1, 2)).float()
    fn = (~predicted_contact & target_contact & contact_valid).sum((1, 2)).float()
    contact_f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)

    if frames >= 4:
        jerk_valid = valid[:, 3:] & valid[:, 2:-1] & valid[:, 1:-2] & valid[:, :-3]
        prediction_jerk = _jerk(prediction_fk, float(fps))
        target_jerk = _jerk(target_fk, float(fps))
        jerk_error = _sample_mean(
            (prediction_jerk - target_jerk).norm(dim=-1), jerk_valid
        )
        prediction_jerk_magnitude = _sample_mean(
            prediction_jerk.norm(dim=-1), jerk_valid
        )
    else:
        jerk_error = source.new_zeros(batch)
        prediction_jerk_magnitude = source.new_zeros(batch)

    prediction_to_source = _sample_mean(
        (prediction_pos - source_pos).norm(dim=-1), valid
    )
    target_to_source = _sample_mean((target_pos - source_pos).norm(dim=-1), valid)
    fk_consistency = _sample_mean(
        (prediction_fk - prediction_pos).norm(dim=-1), valid
    )

    metrics: dict[str, torch.Tensor] = {
        "reactor_position_mpjpe_cm": position_mpjpe * 100.0,
        "reactor_fk_mpjpe_cm": fk_mpjpe * 100.0,
        "reactor_root_error_cm": root_error * 100.0,
        "relative_heading_error_deg": relative_heading_error,
        "reactor_heading_unit_norm_error": heading_norm_error,
        "position_relation_distance_mae_cm": relation_position_mae * 100.0,
        "fk_relation_distance_mae_cm": relation_fk_mae * 100.0,
        "close_event_error_20cm": close_event_error,
        "prediction_min_inter_actor_joint_cm": _sample_mean(
            prediction_min_distance, valid
        )
        * 100.0,
        "target_min_inter_actor_joint_cm": _sample_mean(target_min_distance, valid)
        * 100.0,
        "reactor_contact_accuracy": contact_accuracy,
        "reactor_contact_f1": contact_f1,
        "reactor_fk_jerk_error_mps3": jerk_error,
        "reactor_prediction_fk_jerk_mps3": prediction_jerk_magnitude,
        "prediction_to_source_position_mpjpe_cm": prediction_to_source * 100.0,
        "target_to_source_position_mpjpe_cm": target_to_source * 100.0,
        "prediction_position_fk_disagreement_cm": fk_consistency * 100.0,
    }
    per_sample: list[dict[str, float | int | str]] = []
    for index in range(batch):
        row: dict[str, float | int | str] = {
            "index": index,
            "length": int(lengths[index].item()),
            "assignment": "fixed_actor_to_reactor",
        }
        row.update({name: float(value[index].item()) for name, value in metrics.items()})
        per_sample.append(row)
    return {
        "assignment_rule": "fixed_source_actor_to_target_reactor_no_swap",
        "aggregate": {
            name: float(value.mean().item()) for name, value in metrics.items()
        },
        "per_sample": per_sample,
    }
