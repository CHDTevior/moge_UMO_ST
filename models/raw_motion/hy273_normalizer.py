"""Normalization and Kimodo-style sequence transforms for HY273."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_SLICE,
    VELOCITY_SLICE,
    check_hy273,
    reconstruct_global_joints_from_features,
    yaw_rotate_heading,
    yaw_rotate_positions,
    yaw_rotate_rot6d,
)


@dataclass
class TransformResult:
    motion: torch.Tensor
    c_dir: torch.Tensor
    yaw_delta: torch.Tensor


class HY273Normalizer:
    """Normalize HY273 under either legacy or unified-contact model coordinates."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        eps: float = 1e-6,
        variance_eps: float = 0.0,
        normalize_contacts: bool = False,
    ) -> None:
        if mean.shape[-1] != DIM_HY273 or std.shape[-1] != DIM_HY273:
            raise ValueError(f"Mean/std must have dim {DIM_HY273}, got {mean.shape} {std.shape}")
        self.mean = mean.detach().float().view(1, 1, DIM_HY273)
        self.std = std.detach().float().clamp_min(float(eps)).view(1, 1, DIM_HY273)
        self.normalize_contacts = bool(normalize_contacts)
        if not self.normalize_contacts:
            self.mean[..., CONTACT_SLICE] = 0.0
            self.std[..., CONTACT_SLICE] = 1.0
        self.variance_eps = float(variance_eps)

    @classmethod
    def from_data_root(
        cls,
        data_root: str | Path,
        eps: float = 1e-6,
        stats_dir: str | Path | None = None,
        variance_eps: float = 0.0,
        normalize_contacts: bool = False,
    ) -> "HY273Normalizer":
        root = Path(stats_dir).expanduser() if stats_dir else Path(data_root)
        mean_path = root / "stats" / "Mean.npy"
        std_path = root / "stats" / "Std.npy"
        if stats_dir:
            mean_path = root / "Mean.npy"
            std_path = root / "Std.npy"
        if not mean_path.is_file():
            mean_path = root / "Mean.npy"
        if not std_path.is_file():
            std_path = root / "Std.npy"
        mean = torch.from_numpy(np.load(mean_path)).float()
        std = torch.from_numpy(np.load(std_path)).float()
        return cls(
            mean,
            std,
            eps=eps,
            variance_eps=variance_eps,
            normalize_contacts=normalize_contacts,
        )

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> "HY273Normalizer":
        other = HY273Normalizer(
            self.mean.squeeze(0).squeeze(0),
            self.std.squeeze(0).squeeze(0),
            variance_eps=self.variance_eps,
            normalize_contacts=self.normalize_contacts,
        )
        other.mean = other.mean.to(device=device, dtype=dtype or self.mean.dtype)
        other.std = other.std.to(device=device, dtype=dtype or self.std.dtype)
        return other

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().clone(),
            "std": self.std.detach().clone(),
            "variance_eps": torch.tensor(self.variance_eps, dtype=torch.float64),
        }

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        check_hy273(x)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        scale = torch.sqrt(std.square() + self.variance_eps) if self.variance_eps > 0 else std
        out = (x - mean) / scale
        if not self.normalize_contacts:
            out[..., CONTACT_SLICE] = x[..., CONTACT_SLICE]
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        check_hy273(x)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        scale = torch.sqrt(std.square() + self.variance_eps) if self.variance_eps > 0 else std
        out = x * scale + mean
        if not self.normalize_contacts:
            out[..., CONTACT_SLICE] = x[..., CONTACT_SLICE]
        return out


def root_origin_shift(x: torch.Tensor) -> torch.Tensor:
    """Shift smooth_root x/z so frame 0 starts at the origin; body-relative channels stay unchanged."""
    check_hy273(x)
    out = x.clone()
    delta_x = -out[..., :1, ROOT_SLICE.start]
    delta_z = -out[..., :1, ROOT_SLICE.start + 2]
    out[..., ROOT_SLICE.start] = out[..., ROOT_SLICE.start] + delta_x
    out[..., ROOT_SLICE.start + 2] = out[..., ROOT_SLICE.start + 2] + delta_z
    return out


def apply_yaw_rotation(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Apply Kimodo-style whole-sequence yaw rotation in unnormalized HY273 space."""
    check_hy273(x)
    out = x.clone()
    root = out[..., ROOT_SLICE].reshape(*out.shape[:-1], 3)
    joints = out[..., JOINT_POS_SLICE].reshape(*out.shape[:-1], 22, 3)
    rots = out[..., GLOBAL_ROT_SLICE].reshape(*out.shape[:-1], 22, 6)
    vel = out[..., VELOCITY_SLICE].reshape(*out.shape[:-1], 22, 3)
    out[..., ROOT_SLICE] = yaw_rotate_positions(root, angle)
    out[..., HEADING_SLICE] = yaw_rotate_heading(out[..., HEADING_SLICE], angle)
    out[..., JOINT_POS_SLICE] = yaw_rotate_positions(joints, angle).reshape(*out.shape[:-1], 66)
    out[..., GLOBAL_ROT_SLICE] = yaw_rotate_rot6d(rots, angle).reshape(*out.shape[:-1], 132)
    out[..., VELOCITY_SLICE] = yaw_rotate_positions(vel, angle).reshape(*out.shape[:-1], 66)
    return out


def apply_kimodo_training_transform(
    x: torch.Tensor,
    random_heading: bool = True,
    root_shift: bool = True,
    generator: Optional[torch.Generator] = None,
) -> TransformResult:
    """Apply first-frame root shift and random first-heading augmentation."""
    check_hy273(x)
    out = root_origin_shift(x) if root_shift else x.clone()
    batch_shape = out.shape[:-2]
    if out.ndim == 2:
        out = out.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    bsz = out.shape[0]
    first_heading = out[:, 0, HEADING_SLICE]
    current_angle = torch.atan2(first_heading[:, 1], first_heading[:, 0])
    if random_heading:
        target = (torch.rand((bsz,), device=out.device, dtype=out.dtype, generator=generator) * 2.0 - 1.0) * torch.pi
    else:
        target = current_angle
    delta = target - current_angle
    out = apply_yaw_rotation(out, delta)
    c_dir = out[:, 0, HEADING_SLICE].clone()
    if squeeze:
        return TransformResult(out.squeeze(0), c_dir.squeeze(0), delta.squeeze(0))
    if batch_shape:
        return TransformResult(out, c_dir, delta)
    return TransformResult(out, c_dir, delta)


def transform_equivariance_error(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    before = reconstruct_global_joints_from_features(x)
    after = reconstruct_global_joints_from_features(apply_yaw_rotation(x, angle))
    expected = yaw_rotate_positions(before, angle)
    return (after - expected).abs().amax()
