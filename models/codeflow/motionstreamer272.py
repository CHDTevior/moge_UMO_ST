"""Differentiable MotionStreamer-272 geometry helpers."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class ContinuousEmbeddingDecoder(Protocol):
    def decode_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        ...


def _stats_to_tensor(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.dim() == 1:
        value = value.view(1, 1, -1)
    return value


def motionstreamer272_denormalize(
    motion_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    if motion_norm.ndim != 3 or motion_norm.shape[-1] != 272:
        raise ValueError(f"Expected normalized MotionStreamer motion [B,T,272], got {tuple(motion_norm.shape)}")
    mean = _stats_to_tensor(mean, motion_norm)
    std = _stats_to_tensor(std, motion_norm)
    if mean.shape[-1] != 272 or std.shape[-1] != 272:
        raise ValueError(f"Expected 272D mean/std, got mean={tuple(mean.shape)} std={tuple(std.shape)}")
    return motion_norm * std + mean


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    if d6.shape[-1] != 6:
        raise ValueError(f"Expected 6D rotation features, got last dim {d6.shape[-1]}")
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def recover_motionstreamer272_positions_raw(
    motion_raw: torch.Tensor,
    num_joints: int = 22,
) -> torch.Tensor:
    """Recover world-space joint positions from denormalized 272D features.

    The 272D layout used by this project is:
      root x/z velocity [0:2], root heading delta 6D [2:8],
      local joint positions [8:8+3J], local velocities, local 6D rotations.
    """
    if motion_raw.ndim != 3 or motion_raw.shape[-1] != 272:
        raise ValueError(f"Expected raw MotionStreamer motion [B,T,272], got {tuple(motion_raw.shape)}")
    if int(num_joints) <= 0:
        raise ValueError(f"num_joints must be positive, got {num_joints}")
    num_joints = int(num_joints)
    if 8 + 3 * num_joints > motion_raw.shape[-1]:
        raise ValueError(f"num_joints={num_joints} is incompatible with feature dim {motion_raw.shape[-1]}")

    bsz, seq_len, _feat_dim = motion_raw.shape
    local_positions = motion_raw[:, :, 8 : 8 + 3 * num_joints].reshape(bsz, seq_len, num_joints, 3)
    root_velocity = motion_raw[:, :, :2]
    heading_delta_matrix = rotation_6d_to_matrix(motion_raw[:, :, 2:8])

    headings = []
    running = heading_delta_matrix[:, 0]
    headings.append(running)
    for frame in range(1, seq_len):
        running = heading_delta_matrix[:, frame] @ running
        headings.append(running)
    heading = torch.stack(headings, dim=1)
    inv_heading = heading.transpose(-1, -2)

    positions = torch.matmul(
        inv_heading[:, :, None].expand(-1, -1, num_joints, -1, -1),
        local_positions[..., None],
    ).squeeze(-1)

    velocity_zeros = root_velocity.new_zeros(root_velocity.shape[:-1])
    velocity_xyz = torch.stack(
        (root_velocity[..., 0], velocity_zeros, root_velocity[..., 1]),
        dim=-1,
    )
    if seq_len > 1:
        velocity_tail = torch.matmul(inv_heading[:, :-1], velocity_xyz[:, 1:, :, None]).squeeze(-1)
        velocity_xyz = torch.cat([velocity_xyz[:, :1], velocity_tail], dim=1)
    root_translation = torch.cumsum(velocity_xyz, dim=1)
    offset = torch.stack(
        (
            root_translation[..., 0],
            root_translation.new_zeros(root_translation.shape[:-1]),
            root_translation[..., 2],
        ),
        dim=-1,
    )
    return positions + offset[:, :, None, :]


def recover_motionstreamer272_positions_from_normalized(
    motion_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_joints: int = 22,
) -> torch.Tensor:
    motion_raw = motionstreamer272_denormalize(motion_norm, mean, std)
    return recover_motionstreamer272_positions_raw(motion_raw, num_joints=num_joints)


def decode_motionstreamer272_joints_from_embeddings(
    decoder: ContinuousEmbeddingDecoder,
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_joints: int = 22,
) -> torch.Tensor:
    """Decode continuous RVQ embeddings and recover differentiable joints."""
    motion_norm = decoder.decode_embeddings(embeddings)
    return recover_motionstreamer272_positions_from_normalized(
        motion_norm,
        mean=mean,
        std=std,
        num_joints=num_joints,
    )
