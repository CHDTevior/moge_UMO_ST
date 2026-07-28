"""Constants and tensor helpers for the Kimodo273 / HY273 representation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

DIM_HY273 = 273
CONT_DIM = 269
NUM_JOINTS = 22

ROOT_SLICE = slice(0, 3)
HEADING_SLICE = slice(3, 5)
JOINT_POS_SLICE = slice(5, 71)
GLOBAL_ROT_SLICE = slice(71, 203)
VELOCITY_SLICE = slice(203, 269)
CONTACT_SLICE = slice(269, 273)

ROOT_DIM = 5
BODY_DIM = 268
LOCAL_ROOT_DIM = 4

SMPLX22_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

SMPLX22_PARENTS = torch.tensor(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=torch.long,
)

LEFT_ANKLE = 7
RIGHT_ANKLE = 8
LEFT_FOOT = 10
RIGHT_FOOT = 11
HEAD = 15
LEFT_WRIST = 20
RIGHT_WRIST = 21

CONTACT_JOINTS = (LEFT_ANKLE, LEFT_FOOT, RIGHT_ANKLE, RIGHT_FOOT)
KIMODO_EE_JOINTS = (LEFT_ANKLE, LEFT_FOOT, RIGHT_ANKLE, RIGHT_FOOT, LEFT_WRIST, RIGHT_WRIST)
FIVE_POINT_JOINTS = (HEAD, LEFT_WRIST, RIGHT_WRIST, LEFT_FOOT, RIGHT_FOOT)
KIMODO_EE_GROUPS = (
    (LEFT_ANKLE, LEFT_FOOT),
    (RIGHT_ANKLE, RIGHT_FOOT),
    (LEFT_WRIST,),
    (RIGHT_WRIST,),
)
KIMODO_EE_ROT_GROUPS = (
    (LEFT_ANKLE,),
    (RIGHT_ANKLE,),
    (LEFT_WRIST,),
    (RIGHT_WRIST,),
)
FIVE_POINT_GROUPS = tuple((joint_id,) for joint_id in FIVE_POINT_JOINTS)

FALLBACK_SHORT_CLIPS = {
    "motion_data/000990.npy",
    "motion_data/005836.npy",
    "motion_data/M000990.npy",
    "motion_data/M005836.npy",
}

_NEUTRAL_JOINTS_CPU_CACHE: dict[str, torch.Tensor] = {}


def check_hy273(x: torch.Tensor, name: str = "motion") -> None:
    if x.shape[-1] != DIM_HY273:
        raise ValueError(f"{name} must end with dim {DIM_HY273}, got {tuple(x.shape)}")


def split_joints_pos(x: torch.Tensor) -> torch.Tensor:
    check_hy273(x)
    return x[..., JOINT_POS_SLICE].reshape(*x.shape[:-1], NUM_JOINTS, 3)


def split_global_rot6d(x: torch.Tensor) -> torch.Tensor:
    check_hy273(x)
    return x[..., GLOBAL_ROT_SLICE].reshape(*x.shape[:-1], NUM_JOINTS, 6)


def split_velocities(x: torch.Tensor) -> torch.Tensor:
    check_hy273(x)
    return x[..., VELOCITY_SLICE].reshape(*x.shape[:-1], NUM_JOINTS, 3)


def reconstruct_global_joints_from_features(x: torch.Tensor) -> torch.Tensor:
    """Recover global joints from HY273 position channels.

    HY273 stores joints x/z relative to smooth_root x/z, while y is already global.
    """
    joints = split_joints_pos(x)
    root = x[..., ROOT_SLICE]
    global_joints = joints.clone()
    global_joints[..., 0] = joints[..., 0] + root[..., None, 0]
    global_joints[..., 2] = joints[..., 2] + root[..., None, 2]
    return global_joints


def cont6d_to_matrix(cont6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Kimodo cont6d: first matrix column followed by second matrix column."""
    if cont6d.shape[-1] != 6:
        raise ValueError(f"Expected cont6d last dim 6, got {tuple(cont6d.shape)}")
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = F.normalize(x_raw, dim=-1, eps=eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps=eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matrix [...,3,3], got {tuple(matrix.shape)}")
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def yaw_matrix(angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack([cos, zero, sin, zero, one, zero, -sin, zero, cos], dim=-1).reshape(
        angle.shape + (3, 3)
    )


def yaw_rotate_positions(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate row-vector positions exactly like Kimodo RotateFeatures.rotate_positions."""
    mat_t = yaw_matrix(angle).transpose(-2, -1)
    while mat_t.ndim < x.ndim + 1:
        mat_t = mat_t.unsqueeze(-3)
    return torch.matmul(x.unsqueeze(-2), mat_t).squeeze(-2)


def yaw_rotate_heading(heading: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    mat = torch.stack([cos, sin, -sin, cos], dim=-1).reshape(angle.shape + (2, 2))
    while mat.ndim < heading.ndim + 1:
        mat = mat.unsqueeze(-3)
    return torch.matmul(heading.unsqueeze(-2), mat).squeeze(-2)


def yaw_rotate_rot6d(rot6d: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Left-multiply global rotation matrices by a Y-axis rotation."""
    mats = cont6d_to_matrix(rot6d)
    rot_y = yaw_matrix(angle)
    while rot_y.ndim < mats.ndim:
        rot_y = rot_y.unsqueeze(-3)
    return matrix_to_cont6d(torch.matmul(rot_y, mats))


def joint_pos_slice_for(joint_ids: Iterable[int]) -> list[int]:
    indices: list[int] = []
    for joint_id in joint_ids:
        base = JOINT_POS_SLICE.start + int(joint_id) * 3
        indices.extend([base, base + 1, base + 2])
    return indices


def global_rot_slice_for(joint_ids: Iterable[int]) -> list[int]:
    indices: list[int] = []
    for joint_id in joint_ids:
        base = GLOBAL_ROT_SLICE.start + int(joint_id) * 6
        indices.extend(range(base, base + 6))
    return indices


def load_smplx22_neutral_joints(
    path: str | Path = "external_repos/kimodo/kimodo/assets/skeletons/smplx22/joints.p",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in _NEUTRAL_JOINTS_CPU_CACHE:
        joints_cpu = torch.load(resolved, map_location="cpu").squeeze().float()
        _NEUTRAL_JOINTS_CPU_CACHE[resolved] = (joints_cpu - joints_cpu[0:1]).contiguous()
    joints = _NEUTRAL_JOINTS_CPU_CACHE[resolved].to(dtype=dtype)
    if device is not None:
        joints = joints.to(device)
    return joints


def fk_positions_from_global_rot6d(
    features: torch.Tensor,
    neutral_joints: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate FK positions from HY273 global rotations and rest offsets.

    Kimodo stores global rotations. For each child joint, the rest offset is rotated by
    the parent's global rotation, then accumulated down the tree.
    """
    check_hy273(features)
    squeeze = False
    if features.ndim == 2:
        features = features.unsqueeze(0)
        squeeze = True
    bsz, frames = features.shape[:2]
    device = features.device
    dtype = features.dtype
    if neutral_joints is None:
        neutral_joints = load_smplx22_neutral_joints(device=device, dtype=dtype)
    else:
        neutral_joints = neutral_joints.to(device=device, dtype=dtype)
    parents = SMPLX22_PARENTS.to(device)
    root_pos = reconstruct_global_joints_from_features(features)[:, :, 0]
    global_rot = cont6d_to_matrix(split_global_rot6d(features))
    out = torch.zeros((bsz, frames, NUM_JOINTS, 3), device=device, dtype=dtype)
    out[:, :, 0] = root_pos
    offsets = neutral_joints - neutral_joints[parents.clamp_min(0)]
    for joint in range(1, NUM_JOINTS):
        parent = int(parents[joint].item())
        offset = offsets[joint].view(1, 1, 3, 1)
        rotated = torch.matmul(global_rot[:, :, parent], offset).squeeze(-1)
        out[:, :, joint] = out[:, :, parent] + rotated
    return out.squeeze(0) if squeeze else out


def indices_mask(shape: Sequence[int], indices: Sequence[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(*shape, DIM_HY273, device=device, dtype=torch.bool)
    if indices:
        mask[..., torch.as_tensor(indices, device=device, dtype=torch.long)] = True
    return mask
