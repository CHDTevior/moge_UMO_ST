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


INITIAL_LAYOUT_FRAMES = 15
CONTACT_TIMING_THRESHOLD_M = 0.20
FK_PAIR_CONTACT_THRESHOLD_M = 0.15


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


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator.float() / denominator.float().clamp_min(1.0)


def _close_event_statistics(
    prediction_min_distance: torch.Tensor,
    target_min_distance: torch.Tensor,
    valid: torch.Tensor,
    *,
    name_prefix: str = "",
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    metrics: dict[str, torch.Tensor] = {}
    counts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    sample_counts: dict[str, torch.Tensor] = {}
    for threshold_cm in (10, 20, 30):
        threshold_m = float(threshold_cm) / 100.0
        prediction_close = prediction_min_distance < threshold_m
        target_close = target_min_distance < threshold_m
        true_positive = (prediction_close & target_close & valid).sum(dim=1)
        false_positive = (prediction_close & ~target_close & valid).sum(dim=1)
        false_negative = (~prediction_close & target_close & valid).sum(dim=1)
        target_negative = (~target_close & valid).sum(dim=1)
        target_positive = (target_close & valid).sum(dim=1)
        close_prefix = f"{name_prefix}close_{threshold_cm}cm"
        false_close_name = f"{name_prefix}false_close_rate_{threshold_cm}cm"
        missed_close_name = f"{name_prefix}missed_close_rate_{threshold_cm}cm"
        metrics.update(
            {
                f"{close_prefix}_precision": _safe_ratio(
                    true_positive, true_positive + false_positive
                ),
                f"{close_prefix}_recall": _safe_ratio(
                    true_positive, true_positive + false_negative
                ),
                f"{close_prefix}_f1": _safe_ratio(
                    2.0 * true_positive,
                    2.0 * true_positive + false_positive + false_negative,
                ),
                false_close_name: _safe_ratio(false_positive, target_negative),
                missed_close_name: _safe_ratio(false_negative, target_positive),
            }
        )
        counts.update(
            {
                f"{close_prefix}_precision": (
                    true_positive,
                    true_positive + false_positive,
                ),
                f"{close_prefix}_recall": (
                    true_positive,
                    true_positive + false_negative,
                ),
                f"{close_prefix}_f1": (
                    2.0 * true_positive,
                    2.0 * true_positive + false_positive + false_negative,
                ),
                false_close_name: (false_positive, target_negative),
                missed_close_name: (false_negative, target_positive),
            }
        )
        sample_counts.update(
            {
                f"{close_prefix}_tp": true_positive,
                f"{close_prefix}_fp": false_positive,
                f"{close_prefix}_fn": false_negative,
                f"{close_prefix}_target_positive": target_positive,
                f"{close_prefix}_target_negative": target_negative,
            }
        )
    return metrics, counts, sample_counts


def _binary_event_statistics(
    prediction_positive: torch.Tensor,
    target_positive: torch.Tensor,
    valid: torch.Tensor,
    *,
    event_prefix: str,
    false_positive_name: str,
    false_negative_name: str,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """Return per-sample and pooled-count statistics for a binary event map."""

    if prediction_positive.shape != target_positive.shape:
        raise ValueError("Binary prediction and target event maps differ")
    expanded_valid = valid
    while expanded_valid.ndim < prediction_positive.ndim:
        expanded_valid = expanded_valid.unsqueeze(-1)
    expanded_valid = expanded_valid.expand_as(prediction_positive)
    reduce_dims = tuple(range(1, prediction_positive.ndim))
    true_positive = (
        prediction_positive & target_positive & expanded_valid
    ).sum(dim=reduce_dims)
    false_positive = (
        prediction_positive & ~target_positive & expanded_valid
    ).sum(dim=reduce_dims)
    false_negative = (
        ~prediction_positive & target_positive & expanded_valid
    ).sum(dim=reduce_dims)
    target_positive_count = (target_positive & expanded_valid).sum(dim=reduce_dims)
    target_negative_count = (~target_positive & expanded_valid).sum(dim=reduce_dims)
    metrics = {
        f"{event_prefix}_precision": _safe_ratio(
            true_positive, true_positive + false_positive
        ),
        f"{event_prefix}_recall": _safe_ratio(
            true_positive, true_positive + false_negative
        ),
        f"{event_prefix}_f1": _safe_ratio(
            2.0 * true_positive,
            2.0 * true_positive + false_positive + false_negative,
        ),
        false_positive_name: _safe_ratio(false_positive, target_negative_count),
        false_negative_name: _safe_ratio(false_negative, target_positive_count),
    }
    counts = {
        f"{event_prefix}_precision": (
            true_positive,
            true_positive + false_positive,
        ),
        f"{event_prefix}_recall": (
            true_positive,
            true_positive + false_negative,
        ),
        f"{event_prefix}_f1": (
            2.0 * true_positive,
            2.0 * true_positive + false_positive + false_negative,
        ),
        false_positive_name: (false_positive, target_negative_count),
        false_negative_name: (false_negative, target_positive_count),
    }
    sample_counts = {
        f"{event_prefix}_tp": true_positive,
        f"{event_prefix}_fp": false_positive,
        f"{event_prefix}_fn": false_negative,
        f"{event_prefix}_target_positive": target_positive_count,
        f"{event_prefix}_target_negative": target_negative_count,
    }
    return metrics, counts, sample_counts


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
    frame_index = torch.arange(frames, device=source.device)[None]
    initial_valid = valid & (frame_index < min(INITIAL_LAYOUT_FRAMES, frames))
    frame0_valid = valid & (frame_index == 0)

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
    root_error_per_frame = (
        prediction[..., ROOT_SLICE] - target[..., ROOT_SLICE]
    ).norm(dim=-1)
    root_error = _sample_mean(root_error_per_frame, valid)
    prediction_root_delta = (
        prediction[..., ROOT_SLICE][..., (0, 2)]
        - source[..., ROOT_SLICE][..., (0, 2)]
    )
    target_root_delta = (
        target[..., ROOT_SLICE][..., (0, 2)]
        - source[..., ROOT_SLICE][..., (0, 2)]
    )
    prediction_root_radius = torch.linalg.vector_norm(
        prediction_root_delta, dim=-1
    )
    target_root_radius = torch.linalg.vector_norm(target_root_delta, dim=-1)
    root_radius_error = _sample_mean(
        (prediction_root_radius - target_root_radius).abs(), valid
    )
    bearing_dot = (
        prediction_root_delta * target_root_delta
    ).sum(dim=-1) / (
        prediction_root_radius * target_root_radius
    ).clamp_min(1e-6)
    root_bearing_error = torch.rad2deg(
        torch.acos(bearing_dot.clamp(-1.0, 1.0))
    )
    root_bearing_error = torch.where(
        prediction_root_radius >= 0.10,
        root_bearing_error,
        torch.full_like(root_bearing_error, 180.0),
    )
    bearing_valid = valid & (target_root_radius >= 0.10)
    root_bearing_error = _sample_mean(root_bearing_error, bearing_valid)
    bearing_valid_frames = bearing_valid.sum(dim=1)
    bearing_valid_fraction = _safe_ratio(bearing_valid_frames, lengths)

    source_heading = _heading_angle(source)
    prediction_heading = _heading_angle(prediction)
    target_heading = _heading_angle(target)
    relative_heading_error_per_frame = _angular_error_deg(
        prediction_heading - source_heading,
        target_heading - source_heading,
    )
    relative_heading_error = _sample_mean(relative_heading_error_per_frame, valid)
    prediction_to_source_angle = torch.atan2(
        -prediction_root_delta[..., 0], -prediction_root_delta[..., 1]
    )
    target_to_source_angle = torch.atan2(
        -target_root_delta[..., 0], -target_root_delta[..., 1]
    )
    partner_facing_error = _angular_error_deg(
        prediction_heading - prediction_to_source_angle,
        target_heading - target_to_source_angle,
    )
    partner_facing_error = torch.where(
        prediction_root_radius >= 0.10,
        partner_facing_error,
        torch.full_like(partner_facing_error, 180.0),
    )
    partner_facing_error = _sample_mean(partner_facing_error, bearing_valid)
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
    prediction_fk_min_distance = pred_fk_distance.flatten(2).amin(dim=-1)
    target_fk_min_distance = target_fk_distance.flatten(2).amin(dim=-1)
    target_close_20cm = target_min_distance < CONTACT_TIMING_THRESHOLD_M
    prediction_close_20cm = prediction_min_distance < CONTACT_TIMING_THRESHOLD_M
    sentinel = lengths[:, None].expand(batch, frames)
    first_target_close = torch.where(
        target_close_20cm & valid, frame_index, sentinel
    ).amin(dim=1)
    first_prediction_close = torch.where(
        prediction_close_20cm & valid, frame_index, sentinel
    ).amin(dim=1)
    first_close_delta = first_prediction_close - first_target_close
    precontact_valid = valid & (frame_index < first_target_close[:, None])
    precontact_frames = precontact_valid.sum(dim=1)
    precontact_root_error_sum_cm = (
        root_error_per_frame * precontact_valid.to(root_error_per_frame.dtype)
    ).sum(dim=1) * 100.0
    precontact_heading_error_sum_deg = (
        relative_heading_error_per_frame
        * precontact_valid.to(relative_heading_error_per_frame.dtype)
    ).sum(dim=1)
    precontact_false_close_frames = (
        prediction_close_20cm & precontact_valid
    ).sum(dim=1)
    precontact_false_close = _safe_ratio(
        precontact_false_close_frames,
        precontact_frames,
    )
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
    fk_close_event_error = _sample_mean(
        (
            (prediction_fk_min_distance < 0.20)
            != (target_fk_min_distance < 0.20)
        ).float(),
        valid,
    )
    close_event_metrics, close_event_counts, close_event_sample_counts = (
        _close_event_statistics(
            prediction_min_distance,
            target_min_distance,
            valid,
        )
    )
    fk_metrics, fk_counts, fk_sample_counts = _close_event_statistics(
        prediction_fk_min_distance,
        target_fk_min_distance,
        valid,
        name_prefix="fk_",
    )
    close_event_metrics.update(fk_metrics)
    close_event_counts.update(fk_counts)
    close_event_sample_counts.update(fk_sample_counts)

    fk_pair_prediction_contact = (
        pred_fk_distance < FK_PAIR_CONTACT_THRESHOLD_M
    )
    fk_pair_target_contact = target_fk_distance < FK_PAIR_CONTACT_THRESHOLD_M
    pair_metrics, pair_counts, pair_sample_counts = _binary_event_statistics(
        fk_pair_prediction_contact,
        fk_pair_target_contact,
        valid,
        event_prefix="fk_pair_close_15cm",
        false_positive_name="fk_pair_false_close_rate_15cm",
        false_negative_name="fk_pair_missed_close_rate_15cm",
    )
    # Onset and release are distinct lifecycle events. Keeping direction as an
    # event axis prevents a predicted release from matching a target onset at
    # the same pair and frame boundary.
    fk_pair_prediction_transition = torch.stack(
        (
            ~fk_pair_prediction_contact[:, :-1]
            & fk_pair_prediction_contact[:, 1:],
            fk_pair_prediction_contact[:, :-1]
            & ~fk_pair_prediction_contact[:, 1:],
        ),
        dim=-1,
    )
    fk_pair_target_transition = torch.stack(
        (
            ~fk_pair_target_contact[:, :-1] & fk_pair_target_contact[:, 1:],
            fk_pair_target_contact[:, :-1] & ~fk_pair_target_contact[:, 1:],
        ),
        dim=-1,
    )
    pair_transition_valid = valid[:, 1:] & valid[:, :-1]
    transition_metrics, transition_counts, transition_sample_counts = (
        _binary_event_statistics(
            fk_pair_prediction_transition,
            fk_pair_target_transition,
            pair_transition_valid,
            event_prefix="fk_pair_transition_15cm",
            false_positive_name="fk_pair_false_transition_rate_15cm",
            false_negative_name="fk_pair_missed_transition_rate_15cm",
        )
    )
    close_event_metrics.update(pair_metrics)
    close_event_metrics.update(transition_metrics)
    close_event_counts.update(pair_counts)
    close_event_counts.update(transition_counts)
    close_event_sample_counts.update(pair_sample_counts)
    close_event_sample_counts.update(transition_sample_counts)

    prediction_fk_pair_vector = (
        source_fk[:, :, :, None, :] - prediction_fk[:, :, None, :, :]
    )
    target_fk_pair_vector = (
        source_fk[:, :, :, None, :] - target_fk[:, :, None, :, :]
    )
    fk_contact_vector_error = torch.linalg.vector_norm(
        prediction_fk_pair_vector - target_fk_pair_vector,
        dim=-1,
    )
    fk_contact_vector_valid = valid[..., None, None] & fk_pair_target_contact
    fk_contact_vector_error_sum_cm = (
        fk_contact_vector_error
        * fk_contact_vector_valid.to(dtype=fk_contact_vector_error.dtype)
    ).sum(dim=(1, 2, 3)) * 100.0
    fk_contact_vector_count = fk_contact_vector_valid.sum(dim=(1, 2, 3))
    fk_contact_vector_error_cm = _safe_ratio(
        fk_contact_vector_error_sum_cm,
        fk_contact_vector_count,
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
        "frame0_relative_root_error_cm": (
            _sample_mean(root_error_per_frame, frame0_valid) * 100.0
        ),
        "initial_15f_relative_root_error_cm": (
            _sample_mean(root_error_per_frame, initial_valid) * 100.0
        ),
        "precontact_relative_root_error_cm": (
            _safe_ratio(precontact_root_error_sum_cm, precontact_frames)
        ),
        "relative_root_radius_error_cm": root_radius_error * 100.0,
        "relative_root_bearing_error_deg": root_bearing_error,
        "relative_root_bearing_valid_frame_fraction": bearing_valid_fraction,
        "partner_facing_error_deg": partner_facing_error,
        "relative_heading_error_deg": relative_heading_error,
        "frame0_relative_heading_error_deg": _sample_mean(
            relative_heading_error_per_frame, frame0_valid
        ),
        "initial_15f_relative_heading_error_deg": _sample_mean(
            relative_heading_error_per_frame, initial_valid
        ),
        "precontact_relative_heading_error_deg": _safe_ratio(
            precontact_heading_error_sum_deg, precontact_frames
        ),
        "precontact_frame_fraction": _safe_ratio(precontact_frames, lengths),
        # A missing close event is represented by the sequence endpoint. This
        # makes premature false contact and missed contact contribute finite,
        # directionally interpretable timing errors.
        "first_close_timing_error_s_20cm": first_close_delta.abs().float()
        / float(fps),
        "first_close_too_early_s_20cm": (-first_close_delta).clamp_min(0).float()
        / float(fps),
        "first_close_too_late_s_20cm": first_close_delta.clamp_min(0).float()
        / float(fps),
        "precontact_false_close_rate_20cm": precontact_false_close,
        "reactor_heading_unit_norm_error": heading_norm_error,
        "position_relation_distance_mae_cm": relation_position_mae * 100.0,
        "fk_relation_distance_mae_cm": relation_fk_mae * 100.0,
        "close_event_error_20cm": close_event_error,
        "fk_close_event_error_20cm": fk_close_event_error,
        "prediction_min_inter_actor_joint_cm": _sample_mean(
            prediction_min_distance, valid
        )
        * 100.0,
        "target_min_inter_actor_joint_cm": _sample_mean(target_min_distance, valid)
        * 100.0,
        "prediction_fk_min_inter_actor_joint_cm": _sample_mean(
            prediction_fk_min_distance, valid
        )
        * 100.0,
        "target_fk_min_inter_actor_joint_cm": _sample_mean(
            target_fk_min_distance, valid
        )
        * 100.0,
        "reactor_contact_accuracy": contact_accuracy,
        "reactor_contact_f1": contact_f1,
        "reactor_fk_jerk_error_mps3": jerk_error,
        "reactor_prediction_fk_jerk_mps3": prediction_jerk_magnitude,
        "prediction_to_source_position_mpjpe_cm": prediction_to_source * 100.0,
        "target_to_source_position_mpjpe_cm": target_to_source * 100.0,
        "prediction_position_fk_disagreement_cm": fk_consistency * 100.0,
        "fk_contact_vector_error_cm_15cm": fk_contact_vector_error_cm,
    }
    metrics.update(close_event_metrics)
    per_sample: list[dict[str, float | int | str | None]] = []
    for index in range(batch):
        row: dict[str, float | int | str | None] = {
            "index": index,
            "length": int(lengths[index].item()),
            "assignment": "fixed_actor_to_reactor",
        }
        row.update({name: float(value[index].item()) for name, value in metrics.items()})
        row["relative_root_bearing_valid_frames"] = int(
            bearing_valid_frames[index].item()
        )
        row["precontact_false_close_frames_20cm"] = int(
            precontact_false_close_frames[index].item()
        )
        row["precontact_valid_frames_20cm"] = int(precontact_frames[index].item())
        row["precontact_relative_root_error_sum_cm"] = float(
            precontact_root_error_sum_cm[index].item()
        )
        row["precontact_relative_heading_error_sum_deg"] = float(
            precontact_heading_error_sum_deg[index].item()
        )
        row["fk_contact_vector_error_sum_cm_15cm"] = float(
            fk_contact_vector_error_sum_cm[index].item()
        )
        row["fk_contact_vector_target_pairs_15cm"] = int(
            fk_contact_vector_count[index].item()
        )
        if int(precontact_frames[index].item()) == 0:
            row["precontact_relative_root_error_cm"] = None
            row["precontact_relative_heading_error_deg"] = None
            row["precontact_false_close_rate_20cm"] = None
        if int(fk_contact_vector_count[index].item()) == 0:
            row["fk_contact_vector_error_cm_15cm"] = None
        row.update(
            {
                name: int(value[index].item())
                for name, value in close_event_sample_counts.items()
            }
        )
        for name, (_, denominator) in close_event_counts.items():
            if int(denominator[index].item()) == 0:
                row[name] = None
        per_sample.append(row)
    aggregate = {name: float(value.mean().item()) for name, value in metrics.items()}
    total_precontact_frames = precontact_frames.sum()
    if int(total_precontact_frames.item()) > 0:
        aggregate["precontact_relative_root_error_cm"] = float(
            _safe_ratio(
                precontact_root_error_sum_cm.sum(), total_precontact_frames
            ).item()
        )
        aggregate["precontact_relative_heading_error_deg"] = float(
            _safe_ratio(
                precontact_heading_error_sum_deg.sum(), total_precontact_frames
            ).item()
        )
    else:
        aggregate["precontact_relative_root_error_cm"] = None
        aggregate["precontact_relative_heading_error_deg"] = None
    aggregate["precontact_false_close_rate_20cm"] = (
        float(
            _safe_ratio(
                precontact_false_close_frames.sum(), total_precontact_frames
            ).item()
        )
        if int(total_precontact_frames.item()) > 0
        else None
    )
    for name, (numerator, denominator) in close_event_counts.items():
        total_denominator = denominator.sum()
        aggregate[name] = (
            float(_safe_ratio(numerator.sum(), total_denominator).item())
            if int(total_denominator.item()) > 0
            else None
        )
    total_fk_contact_pairs = fk_contact_vector_count.sum()
    aggregate["fk_contact_vector_error_cm_15cm"] = (
        float(
            _safe_ratio(
                fk_contact_vector_error_sum_cm.sum(),
                total_fk_contact_pairs,
            ).item()
        )
        if int(total_fk_contact_pairs.item()) > 0
        else None
    )
    return {
        "assignment_rule": "fixed_source_actor_to_target_reactor_no_swap",
        "aggregate": aggregate,
        "per_sample": per_sample,
    }
