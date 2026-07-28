"""KV-Control style geometry condition helpers for CodeFlow."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .motionstreamer272 import recover_motionstreamer272_positions_from_normalized


SEMANTIC_CONTROL_JOINTS = {
    "root": [0],
    "endpoints5": [10, 11, 15, 20, 21],
    "root_endpoints6": [0, 10, 11, 15, 20, 21],
    "full_pose": list(range(22)),
}

SEMANTIC_ENDPOINT_JOINTS = [10, 11, 15, 20, 21]
SEMANTIC_MIX_PROFILES: Tuple[str, ...] = (
    "root",
    "endpoints_random_subset",
    "endpoints5",
    "root_endpoints_random_subset",
    "root_endpoints6",
    "full_pose",
)
SEMANTIC_MIX_WEIGHTS = (0.15, 0.20, 0.15, 0.20, 0.15, 0.15)
SEMANTIC_MIX_GROUP_COUNT_WEIGHTS = (0.55, 0.35, 0.10)


def build_joint_control_condition(
    current_joints: torch.Tensor,
    target_joints: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Build residual+absolute frame control channels.

    Shapes:
      current_joints / target_joints / target_mask: [B,F,J,3]
      return: [B,F,2*J*3]
    """
    if current_joints.shape != target_joints.shape or current_joints.shape != target_mask.shape:
        raise ValueError(
            "current_joints, target_joints, and target_mask must have identical shapes, "
            f"got {tuple(current_joints.shape)}, {tuple(target_joints.shape)}, {tuple(target_mask.shape)}"
        )
    if current_joints.ndim != 4 or current_joints.shape[-1] != 3:
        raise ValueError(f"Expected joint tensors [B,F,J,3], got {tuple(current_joints.shape)}")
    mask = target_mask.to(device=current_joints.device, dtype=current_joints.dtype)
    target = target_joints.to(device=current_joints.device, dtype=current_joints.dtype)
    residual = (target - current_joints) * mask
    absolute = target * mask
    return torch.cat(
        [
            residual.reshape(current_joints.shape[0], current_joints.shape[1], -1),
            absolute.reshape(current_joints.shape[0], current_joints.shape[1], -1),
        ],
        dim=-1,
    )


def semantic_control_joint_ids(
    profile: str,
    *,
    device: torch.device,
    joint_count_total: int,
    min_joints: int = 1,
    max_joints: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return joint ids for a real user-facing control profile."""
    profile = str(profile)
    min_joints = max(int(min_joints), 1)
    max_joints = int(max_joints) if max_joints is not None else int(joint_count_total)
    max_joints = max(max_joints, min_joints)
    if profile in SEMANTIC_CONTROL_JOINTS:
        ids = [idx for idx in SEMANTIC_CONTROL_JOINTS[profile] if idx < int(joint_count_total)]
        return torch.as_tensor(ids, device=device, dtype=torch.long)
    if profile == "semantic_random_subset":
        pool = [idx for idx in SEMANTIC_CONTROL_JOINTS["root_endpoints6"] if idx < int(joint_count_total)]
        if not pool:
            return torch.empty((0,), device=device, dtype=torch.long)
        order = torch.randperm(len(pool), device=device, generator=generator)
        count_hi = min(len(pool), max_joints)
        count_lo = min(min_joints, count_hi)
        count = int(torch.randint(count_lo, count_hi + 1, (1,), device=device, generator=generator).item())
        return torch.as_tensor(pool, device=device, dtype=torch.long)[order[:count]]
    if profile == "endpoints_random_subset":
        pool = [idx for idx in SEMANTIC_ENDPOINT_JOINTS if idx < int(joint_count_total)]
        if not pool:
            return torch.empty((0,), device=device, dtype=torch.long)
        order = torch.randperm(len(pool), device=device, generator=generator)
        count_hi = min(len(pool), max_joints)
        count_lo = min(min_joints, count_hi)
        count = int(torch.randint(count_lo, count_hi + 1, (1,), device=device, generator=generator).item())
        return torch.as_tensor(pool, device=device, dtype=torch.long)[order[:count]]
    if profile == "root_endpoints_random_subset":
        endpoints = [idx for idx in SEMANTIC_ENDPOINT_JOINTS if idx < int(joint_count_total)]
        root = [0] if int(joint_count_total) > 0 else []
        if not endpoints:
            return torch.as_tensor(root, device=device, dtype=torch.long)
        order = torch.randperm(len(endpoints), device=device, generator=generator)
        endpoint_hi = min(len(endpoints), max(1, max_joints - len(root)))
        endpoint_lo = min(max(1, min_joints - len(root)), endpoint_hi)
        count = int(torch.randint(endpoint_lo, endpoint_hi + 1, (1,), device=device, generator=generator).item())
        ids = root + [endpoints[int(idx.item())] for idx in order[:count]]
        return torch.as_tensor(ids, device=device, dtype=torch.long)
    raise ValueError(f"Unsupported semantic control profile: {profile}")


def sample_semantic_profile(
    profile: str,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> str:
    profile = str(profile)
    if profile != "semantic_mix":
        return profile
    weights = torch.as_tensor(SEMANTIC_MIX_WEIGHTS, device=device, dtype=torch.float32)
    index = int(torch.multinomial(weights, 1, replacement=True, generator=generator).item())
    return SEMANTIC_MIX_PROFILES[index]


def sample_semantic_profile_group(
    profile: str,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[str, ...]:
    """Return one or more semantic control groups for a sequence.

    semantic_mix is intentionally a union of user-facing control families, so a
    single motion may contain root controls on some frames and endpoint/full-pose
    keyframes on other frames.
    """
    profile = str(profile)
    if profile != "semantic_mix":
        return (profile,)
    count_weights = torch.as_tensor(SEMANTIC_MIX_GROUP_COUNT_WEIGHTS, device=device, dtype=torch.float32)
    group_count = int(torch.multinomial(count_weights, 1, replacement=True, generator=generator).item()) + 1
    profile_weights = torch.as_tensor(SEMANTIC_MIX_WEIGHTS, device=device, dtype=torch.float32)
    profile_ids = torch.multinomial(
        profile_weights,
        min(group_count, len(SEMANTIC_MIX_PROFILES)),
        replacement=False,
        generator=generator,
    )
    return tuple(SEMANTIC_MIX_PROFILES[int(idx.item())] for idx in profile_ids)


def sample_control_frame_ids(
    valid_len: int,
    frame_count: int,
    *,
    device: torch.device,
    strategy: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    valid_len = max(1, int(valid_len))
    frame_count = max(1, min(int(frame_count), valid_len))
    if str(strategy) == "uniform":
        if frame_count == 1:
            return torch.zeros((1,), device=device, dtype=torch.long)
        return torch.linspace(0, valid_len - 1, frame_count, device=device).round().long().unique()
    if str(strategy) == "endpoints":
        if frame_count == 1:
            return torch.zeros((1,), device=device, dtype=torch.long)
        ids = torch.linspace(0, valid_len - 1, frame_count, device=device).round().long().unique()
        if ids[-1].item() != valid_len - 1:
            ids = torch.cat([ids, torch.as_tensor([valid_len - 1], device=device, dtype=torch.long)])
        return ids.unique()
    if str(strategy) == "random":
        return torch.randperm(valid_len, device=device, generator=generator)[:frame_count].sort().values
    if str(strategy) == "mixed":
        weights = torch.as_tensor([0.60, 0.30, 0.10], device=device, dtype=torch.float32)
        index = int(torch.multinomial(weights, 1, replacement=True, generator=generator).item())
        strategy_name = ("random", "uniform", "endpoints")[index]
        return sample_control_frame_ids(
            valid_len,
            frame_count,
            device=device,
            strategy=strategy_name,
            generator=generator,
        )
    raise ValueError(f"Unsupported control keyframe strategy: {strategy}")


def masked_joint_position_loss(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    target_mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if pred_joints.shape != target_joints.shape or pred_joints.shape != target_mask.shape:
        raise ValueError(
            "pred_joints, target_joints, and target_mask must have identical shapes, "
            f"got {tuple(pred_joints.shape)}, {tuple(target_joints.shape)}, {tuple(target_mask.shape)}"
        )
    mask = target_mask.to(device=pred_joints.device, dtype=pred_joints.dtype)
    denom = mask.sum().clamp_min(1.0)
    delta = (pred_joints - target_joints.to(device=pred_joints.device, dtype=pred_joints.dtype)) * mask
    if loss_type == "l1":
        return delta.abs().sum() / denom
    if loss_type == "mse":
        return delta.square().sum() / denom
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="sum") / denom
    raise ValueError(f"Unsupported control loss_type: {loss_type}")


@torch.no_grad()
def sample_random_joint_position_control(
    motion_norm: torch.Tensor,
    frame_lengths: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    min_keyframes: int = 1,
    max_keyframes: int = 5,
    min_joints: int = 1,
    max_joints: int = 6,
    dropout_prob: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Sample sparse GT joint position controls from normalized 272D motion."""
    if motion_norm.ndim != 3 or motion_norm.shape[-1] != 272:
        raise ValueError(f"Expected motion_norm [B,F,272], got {tuple(motion_norm.shape)}")
    bsz, seq_len, _feat_dim = motion_norm.shape
    target_joints = recover_motionstreamer272_positions_from_normalized(motion_norm, mean, std)
    target_mask = torch.zeros_like(target_joints, dtype=torch.bool)

    frame_lengths = frame_lengths.to(device=motion_norm.device, dtype=torch.long).clamp(min=1, max=seq_len)
    joint_count_total = target_joints.shape[2]
    min_keyframes = max(int(min_keyframes), 1)
    max_keyframes = max(int(max_keyframes), min_keyframes)
    min_joints = max(int(min_joints), 1)
    max_joints = max(int(max_joints), min_joints)

    for batch_idx in range(bsz):
        if float(dropout_prob) > 0.0:
            keep = torch.rand((), device=motion_norm.device, generator=generator) >= float(dropout_prob)
            if not bool(keep.item()):
                continue
        valid_len = int(frame_lengths[batch_idx].item())
        keyframe_hi = min(max_keyframes, valid_len)
        keyframe_lo = min(min_keyframes, keyframe_hi)
        keyframe_count = int(
            torch.randint(
                keyframe_lo,
                keyframe_hi + 1,
                (1,),
                device=motion_norm.device,
                generator=generator,
            ).item()
        )
        joint_hi = min(max_joints, joint_count_total)
        joint_lo = min(min_joints, joint_hi)
        joint_count = int(
            torch.randint(
                joint_lo,
                joint_hi + 1,
                (1,),
                device=motion_norm.device,
                generator=generator,
            ).item()
        )
        frame_ids = torch.randperm(valid_len, device=motion_norm.device, generator=generator)[:keyframe_count]
        joint_ids = torch.randperm(joint_count_total, device=motion_norm.device, generator=generator)[:joint_count]
        target_mask[batch_idx, frame_ids[:, None], joint_ids[None, :], :] = True

    return {
        "target_joints": target_joints,
        "target_mask": target_mask,
    }


@torch.no_grad()
def sample_semantic_joint_position_control(
    motion_norm: torch.Tensor,
    frame_lengths: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    profile: str = "root_endpoints6",
    min_keyframes: int = 1,
    max_keyframes: int = 5,
    min_joints: int = 1,
    max_joints: int = 6,
    keyframe_strategy: str = "random",
    dropout_prob: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Sample root/end-effector/full-pose controls that match user-facing protocols."""
    if motion_norm.ndim != 3 or motion_norm.shape[-1] != 272:
        raise ValueError(f"Expected motion_norm [B,F,272], got {tuple(motion_norm.shape)}")
    bsz, seq_len, _feat_dim = motion_norm.shape
    target_joints = recover_motionstreamer272_positions_from_normalized(motion_norm, mean, std)
    target_mask = torch.zeros_like(target_joints, dtype=torch.bool)
    frame_lengths = frame_lengths.to(device=motion_norm.device, dtype=torch.long).clamp(min=1, max=seq_len)
    joint_count_total = target_joints.shape[2]
    min_keyframes = max(int(min_keyframes), 1)
    max_keyframes = max(int(max_keyframes), min_keyframes)
    min_joints = max(int(min_joints), 1)
    max_joints = max(int(max_joints), min_joints)

    for batch_idx in range(bsz):
        if float(dropout_prob) > 0.0:
            keep = torch.rand((), device=motion_norm.device, generator=generator) >= float(dropout_prob)
            if not bool(keep.item()):
                continue
        valid_len = int(frame_lengths[batch_idx].item())
        keyframe_hi = min(max_keyframes, valid_len)
        keyframe_lo = min(min_keyframes, keyframe_hi)
        sample_profiles = sample_semantic_profile_group(
            str(profile),
            device=motion_norm.device,
            generator=generator,
        )
        for sample_profile in sample_profiles:
            keyframe_count = int(
                torch.randint(
                    keyframe_lo,
                    keyframe_hi + 1,
                    (1,),
                    device=motion_norm.device,
                    generator=generator,
                ).item()
            )
            joint_ids = semantic_control_joint_ids(
                sample_profile,
                device=motion_norm.device,
                joint_count_total=joint_count_total,
                min_joints=min_joints,
                max_joints=max_joints,
                generator=generator,
            )
            if joint_ids.numel() == 0:
                continue
            frame_ids = sample_control_frame_ids(
                valid_len,
                keyframe_count,
                device=motion_norm.device,
                strategy=str(keyframe_strategy),
                generator=generator,
            )
            target_mask[batch_idx, frame_ids[:, None], joint_ids[None, :], :] = True

    return {
        "target_joints": target_joints,
        "target_mask": target_mask,
    }


@torch.no_grad()
def sample_joint_position_control(
    motion_norm: torch.Tensor,
    frame_lengths: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    profile: str = "random_joints",
    min_keyframes: int = 1,
    max_keyframes: int = 5,
    min_joints: int = 1,
    max_joints: int = 6,
    dropout_prob: float = 0.0,
    keyframe_strategy: str = "random",
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    if str(profile) == "random_joints":
        return sample_random_joint_position_control(
            motion_norm,
            frame_lengths,
            mean,
            std,
            min_keyframes=min_keyframes,
            max_keyframes=max_keyframes,
            min_joints=min_joints,
            max_joints=max_joints,
            dropout_prob=dropout_prob,
            generator=generator,
        )
    return sample_semantic_joint_position_control(
        motion_norm,
        frame_lengths,
        mean,
        std,
        profile=str(profile),
        min_keyframes=min_keyframes,
        max_keyframes=max_keyframes,
        min_joints=min_joints,
        max_joints=max_joints,
        keyframe_strategy=str(keyframe_strategy),
        dropout_prob=dropout_prob,
        generator=generator,
    )
