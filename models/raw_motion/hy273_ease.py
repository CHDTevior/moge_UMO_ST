"""Physical ease-in/out labels and conditioning for HY273 motion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .hy273_slices import DIM_HY273, reconstruct_global_joints_from_features


EASE_DIM = 6
EASE_STATS_FORMAT = "hy273_ease_stats_v1"


def _endpoint_exact_mean_residual(trajectory: torch.Tensor) -> torch.Tensor:
    """Return mean residual from endpoint-exact linear interpolation."""

    if trajectory.ndim != 2 or trajectory.shape[-1] != 3:
        raise ValueError(
            f"Expected trajectory [T,3], got {tuple(trajectory.shape)}"
        )
    frames = int(trajectory.shape[0])
    if frames < 2:
        raise ValueError("Each Ease half must contain at least two frames")
    u = torch.linspace(
        0.0,
        1.0,
        frames,
        device=trajectory.device,
        dtype=trajectory.dtype,
    )[:, None]
    linear = (1.0 - u) * trajectory[0] + u * trajectory[-1]
    return (trajectory - linear).mean(dim=0)


def ease_from_centroid_trajectory(trajectory: torch.Tensor) -> torch.Tensor:
    """Compute ``[E_in, E_out]`` from one physical centroid trajectory."""

    if trajectory.ndim != 2 or trajectory.shape[-1] != 3:
        raise ValueError(
            f"Expected trajectory [T,3], got {tuple(trajectory.shape)}"
        )
    frames = int(trajectory.shape[0])
    if frames < 4:
        raise ValueError("Ease labels require at least four valid frames")
    split = frames // 2
    return torch.cat(
        [
            _endpoint_exact_mean_residual(trajectory[:split]),
            _endpoint_exact_mean_residual(trajectory[split:]),
        ],
        dim=0,
    )


def ease_from_k273(
    physical_motion: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute physical Ease labels from unnormalized K273 motion.

    The global centroid is reconstructed from smooth-root translation plus local
    joint positions. Padding is excluded using an exact prefix validity mask.
    """

    squeeze = physical_motion.ndim == 2
    if squeeze:
        physical_motion = physical_motion.unsqueeze(0)
    if physical_motion.ndim != 3 or physical_motion.shape[-1] != DIM_HY273:
        raise ValueError(
            f"Expected physical K273 [B,T,{DIM_HY273}], got "
            f"{tuple(physical_motion.shape)}"
        )
    batch_size, frames, _ = physical_motion.shape
    if valid is None:
        valid = torch.ones(
            batch_size,
            frames,
            device=physical_motion.device,
            dtype=torch.bool,
        )
    elif valid.ndim == 1 and batch_size == 1:
        valid = valid.unsqueeze(0)
    if valid.shape != (batch_size, frames) or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [B,T]")
    lengths = valid.sum(dim=-1)
    positions = torch.arange(frames, device=valid.device)[None]
    if not torch.equal(valid, positions < lengths[:, None]):
        raise ValueError("Ease validity must be an exact prefix mask")

    joints = reconstruct_global_joints_from_features(physical_motion.float())
    centroid = joints.mean(dim=-2)
    labels = [
        ease_from_centroid_trajectory(centroid[index, : int(length.item())])
        for index, length in enumerate(lengths)
    ]
    output = torch.stack(labels).to(dtype=physical_motion.dtype)
    return output[0] if squeeze else output


class HY273EaseNormalizer(nn.Module):
    """Independent six-dimensional physical Ease normalization."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(EASE_DIM)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(EASE_DIM)
        if not bool(torch.isfinite(mean).all()):
            raise ValueError("Ease mean contains non-finite values")
        if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
            raise ValueError("Ease std must be finite and strictly positive")
        if not torch.equal(mean[[0, 2, 3, 5]], torch.zeros(4)):
            raise ValueError("Yaw-isotropic Ease stats require exact zero X/Z means")
        if not torch.equal(std[[0, 2]], std[[0, 0]]) or not torch.equal(
            std[[3, 5]], std[[3, 3]]
        ):
            raise ValueError("Each Ease half must share its X/Z scale")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @classmethod
    def from_directory(cls, stats_dir: str | Path) -> "HY273EaseNormalizer":
        root = Path(stats_dir).expanduser().resolve()
        mean_path = root / "Mean.npy"
        std_path = root / "Std.npy"
        metadata_path = root / "metadata.json"
        for path in (mean_path, std_path, metadata_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing Ease stats asset: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != EASE_STATS_FORMAT:
            raise ValueError(
                f"Ease stats format mismatch: {metadata.get('format')!r}"
            )
        if int(metadata.get("feature_dim", -1)) != EASE_DIM:
            raise ValueError("Ease stats feature_dim mismatch")
        return cls(np.load(mean_path), np.load(std_path))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != EASE_DIM:
            raise ValueError(f"Ease values must end in {EASE_DIM} channels")
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)
        return (value - mean) / std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != EASE_DIM:
            raise ValueError(f"Ease values must end in {EASE_DIM} channels")
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)
        return value * std + mean


class HY273EaseConditioner(nn.Module):
    """Zero-initialized 6D physical Ease to hidden-state additive bias."""

    def __init__(self, hidden_dim: int, stats_dir: str | Path) -> None:
        super().__init__()
        self.normalizer = HY273EaseNormalizer.from_directory(stats_dir)
        self.input = nn.Linear(EASE_DIM, int(hidden_dim))
        self.activation = nn.SiLU()
        self.output = nn.Linear(int(hidden_dim), int(hidden_dim))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def weight_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.input.weight, self.output.weight

    def bias_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.input.bias, self.output.bias

    def forward(
        self,
        ease_physical: torch.Tensor,
        ease_present: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if ease_physical.ndim != 2 or ease_physical.shape[-1] != EASE_DIM:
            raise ValueError("ease_physical must be [B,6]")
        if ease_present.shape != ease_physical.shape[:1]:
            raise ValueError("ease_present must be [B]")
        if ease_present.dtype != torch.bool:
            raise TypeError("ease_present must be bool")
        if not bool(torch.isfinite(ease_physical).all()):
            raise ValueError("ease_physical contains non-finite values")
        normalized = self.normalizer.normalize(ease_physical.float())
        hidden = self.output(self.activation(self.input(normalized)))
        hidden = hidden * ease_present[:, None].to(dtype=hidden.dtype)
        return hidden.to(dtype=dtype)
