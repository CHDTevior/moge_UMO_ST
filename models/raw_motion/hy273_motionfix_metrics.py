"""Internal K273 MotionFix metrics and physically valid source-copy timewarp."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
import torch.nn.functional as F

from .hy273_kimodo_benchmark import (
    kimodo_motion_quality_metrics,
    load_smplx22_metric_joints,
)
from .hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    ROOT_SLICE,
    cont6d_to_matrix,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
    split_global_rot6d,
)


HY201_REPO = Path("/mnt/afs/UMO_debug/hy201_to_kimodo273")
KIMODO_REPO = Path("/mnt/afs/UMO_debug/outside_material/kimodo")
PHYSICAL_TIMEWARP_PROTOCOL = "hy201_root_linear_local_so3_slerp_reextract_k273_v1"
INTERNAL_METRIC_PROTOCOL = "hy273_motionfix_internal_metrics_v2_changed_regions"


def _load_external_conversion_api() -> tuple[Any, Any, Any]:
    repo = HY201_REPO.expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"HY201 conversion repository is missing: {repo}")
    repo_string = str(repo)
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
    from hy201_to_kimodo273.convert import build_kimodo273_tensors
    from hy201_to_kimodo273.geometry import decode_hy201_rot_trans
    from hy201_to_kimodo273.kimodo_bridge import load_kimodo_motion_rep

    return decode_hy201_rot_trans, build_kimodo273_tensors, load_kimodo_motion_rep


@lru_cache(maxsize=1)
def load_motion_rep() -> Any:
    _, _, loader = _load_external_conversion_api()
    _, motion_rep = loader(KIMODO_REPO, fps=30)
    return motion_rep


def _slerp_local_rotations(rotations: np.ndarray, target_frames: int) -> np.ndarray:
    source_frames, joints = rotations.shape[:2]
    if target_frames < 2 or source_frames < 2:
        return np.repeat(rotations[:1], target_frames, axis=0)
    source_time = np.arange(source_frames, dtype=np.float64)
    target_time = np.linspace(0.0, float(source_frames - 1), target_frames)
    output = np.empty((target_frames, joints, 3, 3), dtype=np.float64)
    for joint in range(joints):
        interpolator = Slerp(source_time, Rotation.from_matrix(rotations[:, joint]))
        output[:, joint] = interpolator(target_time).as_matrix()
    return output


def physical_timewarp_hy201_to_k273(
    hy201: np.ndarray,
    target_frames: int,
    *,
    device: str | torch.device = "cpu",
    motion_rep: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Timewarp root/rotations, then recompute every derived K273 channel.

    This function deliberately has no K273 input. It therefore cannot silently
    interpolate cont6d, velocity, contact, heading, or smooth-root channels.
    """

    hy201 = np.asarray(hy201)
    if hy201.ndim != 2 or hy201.shape[1] != 201 or not np.isfinite(hy201).all():
        raise ValueError(f"Expected finite HY201 [T,201], got {hy201.shape}")
    source_frames = int(hy201.shape[0])
    target_frames = int(target_frames)
    if source_frames < 1 or target_frames < 1:
        raise ValueError("Source and target frame counts must be positive")

    decode, build, _ = _load_external_conversion_api()
    local_rotations, root_positions = decode(hy201)
    target_time = np.linspace(0.0, float(max(source_frames - 1, 0)), target_frames)
    source_time = np.arange(source_frames, dtype=np.float64)
    if source_frames == 1:
        resampled_root = np.repeat(root_positions[:1], target_frames, axis=0)
    else:
        resampled_root = np.stack(
            [np.interp(target_time, source_time, root_positions[:, axis]) for axis in range(3)],
            axis=-1,
        )
    resampled_rotations = _slerp_local_rotations(local_rotations, target_frames)
    motion_rep = load_motion_rep() if motion_rep is None else motion_rep
    tensor_device = torch.device(device)
    with torch.no_grad():
        features, _, conversion_info = build(
            torch.from_numpy(resampled_rotations).to(tensor_device, torch.float32)[None],
            torch.from_numpy(resampled_root).to(tensor_device, torch.float32)[None],
            motion_rep,
        )
    features = features[0].detach().cpu().float().contiguous()
    if features.shape != (target_frames, DIM_HY273) or not torch.isfinite(features).all():
        raise RuntimeError(f"Physical timewarp produced invalid K273 {tuple(features.shape)}")
    contacts = features[:, CONTACT_SLICE]
    if not bool(((contacts == 0.0) | (contacts == 1.0)).all()):
        raise RuntimeError("Re-extracted K273 contacts are not exact binary values")
    return features, {
        "protocol": PHYSICAL_TIMEWARP_PROTOCOL,
        "source_domain": "HY201 root translation + local SO(3) rotations",
        "forbidden_operation": "direct K273/cont6d interpolation",
        "source_frames": source_frames,
        "target_frames": target_frames,
        "smooth_root_fallback": bool(conversion_info.get("smooth_root_fallback", False)),
        "velocity_single_frame_fallback": bool(
            conversion_info.get("velocity_single_frame_fallback", False)
        ),
    }


def _rotation_error_deg(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first_matrix = cont6d_to_matrix(first)
    second_matrix = cont6d_to_matrix(second)
    relative = first_matrix.transpose(-1, -2) @ second_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return torch.rad2deg(torch.acos(cosine))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    count = int(expanded.sum().item())
    if count == 0:
        return None
    return float(values[expanded].mean().item())


def _jerk_mps3(joints: torch.Tensor, fps: float) -> float:
    if joints.shape[0] < 4:
        return 0.0
    jerk = torch.diff(joints, n=3, dim=0) * float(fps) ** 3
    return float(jerk.norm(dim=-1).mean().item())


def evaluate_motionfix_internal_case(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_aligned: torch.Tensor,
    *,
    fps: float = 30.0,
    changed_position_threshold_m: float = 0.02,
    changed_rotation_threshold_deg: float = 5.0,
    unchanged_position_threshold_m: float = 0.01,
    unchanged_rotation_threshold_deg: float = 3.0,
    changed_temporal_dilation_frames: int = 2,
) -> dict[str, float | int | None]:
    """Evaluate one prediction in a common requested-target-length gauge."""

    expected = target.shape
    if prediction.shape != expected or source_aligned.shape != expected:
        raise ValueError(
            "Prediction, target, and aligned source must share [T,273]: "
            f"{prediction.shape}/{target.shape}/{source_aligned.shape}"
        )
    if target.ndim != 2 or target.shape[1] != DIM_HY273 or target.shape[0] < 2:
        raise ValueError("MotionFix metrics require [T,273] with T >= 2")
    if not all(torch.isfinite(value).all() for value in (prediction, target, source_aligned)):
        raise ValueError("MotionFix metric inputs must be finite")

    prediction = prediction.float()
    target = target.float()
    source_aligned = source_aligned.float()
    neutral_joints = load_smplx22_metric_joints(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    pred_joints = fk_positions_from_global_rot6d(
        prediction, neutral_joints=neutral_joints
    )
    target_joints = fk_positions_from_global_rot6d(
        target, neutral_joints=neutral_joints
    )
    source_joints = fk_positions_from_global_rot6d(
        source_aligned, neutral_joints=neutral_joints
    )
    pred_position_joints = reconstruct_global_joints_from_features(prediction)
    pred_target_pos = (pred_joints - target_joints).norm(dim=-1)
    source_target_pos = (source_joints - target_joints).norm(dim=-1)
    pred_source_pos = (pred_joints - source_joints).norm(dim=-1)

    pred_target_rot = _rotation_error_deg(
        split_global_rot6d(prediction), split_global_rot6d(target)
    )
    pred_source_rot = _rotation_error_deg(
        split_global_rot6d(prediction), split_global_rot6d(source_aligned)
    )
    source_target_rot = _rotation_error_deg(
        split_global_rot6d(source_aligned), split_global_rot6d(target)
    )
    if not (
        0.0 <= float(unchanged_position_threshold_m)
        < float(changed_position_threshold_m)
    ):
        raise ValueError("Position thresholds must satisfy 0 <= unchanged < changed")
    if not (
        0.0 <= float(unchanged_rotation_threshold_deg)
        < float(changed_rotation_threshold_deg)
    ):
        raise ValueError("Rotation thresholds must satisfy 0 <= unchanged < changed")
    if int(changed_temporal_dilation_frames) < 0:
        raise ValueError("changed_temporal_dilation_frames must be non-negative")

    changed_seed = (source_target_pos > float(changed_position_threshold_m)) | (
        source_target_rot > float(changed_rotation_threshold_deg)
    )
    dilation = int(changed_temporal_dilation_frames)
    if dilation:
        changed = F.max_pool1d(
            changed_seed.transpose(0, 1).float().unsqueeze(0),
            kernel_size=2 * dilation + 1,
            stride=1,
            padding=dilation,
        ).squeeze(0).transpose(0, 1).bool()
    else:
        changed = changed_seed
    unchanged = (
        (source_target_pos < float(unchanged_position_threshold_m))
        & (source_target_rot < float(unchanged_rotation_threshold_deg))
        & ~changed
    )
    ambiguous = ~(changed | unchanged)
    root_error = (
        pred_position_joints[:, 0]
        - reconstruct_global_joints_from_features(target)[:, 0]
    ).norm(dim=-1)
    contact_prediction = prediction[:, CONTACT_SLICE] >= 0.5
    contact_target = target[:, CONTACT_SLICE] >= 0.5
    quality = kimodo_motion_quality_metrics(
        pred_joints, contact_prediction, fps=float(fps)
    )
    result: dict[str, float | int | None] = {
        "protocol": INTERNAL_METRIC_PROTOCOL,
        "frames": int(target.shape[0]),
        "global_joint_target_error_m": float(pred_target_pos.mean().item()),
        "root_target_error_m": float(root_error.mean().item()),
        "global_rotation_target_error_deg": float(pred_target_rot.mean().item()),
        "contact_target_accuracy": float(
            (contact_prediction == contact_target).float().mean().item()
        ),
        "changed_joint_entries": int(changed.sum().item()),
        "unchanged_joint_entries": int(unchanged.sum().item()),
        "ambiguous_joint_entries": int(ambiguous.sum().item()),
        "changed_region_target_error_m": _masked_mean(pred_target_pos, changed),
        "unchanged_region_source_error_m": _masked_mean(pred_source_pos, unchanged),
        "changed_region_target_rotation_error_deg": _masked_mean(
            pred_target_rot, changed
        ),
        "unchanged_region_source_rotation_error_deg": _masked_mean(
            pred_source_rot, unchanged
        ),
        "global_joint_source_error_m": float(pred_source_pos.mean().item()),
        "global_rotation_source_error_deg": float(pred_source_rot.mean().item()),
        "source_target_position_delta_m": float(source_target_pos.mean().item()),
        "source_target_rotation_delta_deg": float(source_target_rot.mean().item()),
        "changed_position_threshold_m": float(changed_position_threshold_m),
        "changed_rotation_threshold_deg": float(changed_rotation_threshold_deg),
        "unchanged_position_threshold_m": float(unchanged_position_threshold_m),
        "unchanged_rotation_threshold_deg": float(
            unchanged_rotation_threshold_deg
        ),
        "changed_temporal_dilation_frames": int(
            changed_temporal_dilation_frames
        ),
        "prediction_jerk_mps3": _jerk_mps3(pred_joints, float(fps)),
        "target_jerk_mps3": _jerk_mps3(target_joints, float(fps)),
        "fk_position_rotation_consistency_cm": float(
            (pred_position_joints - pred_joints).norm(dim=-1).mean().item()
            * 100.0
        ),
    }
    result.update(quality)
    return result
