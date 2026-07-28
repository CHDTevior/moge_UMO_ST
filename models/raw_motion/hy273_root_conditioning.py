"""Kimodo-compatible global-root to local-root conditioning utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .hy273_slices import HEADING_SLICE, LOCAL_ROOT_DIM, ROOT_DIM, ROOT_SLICE


def _load_stat(path: Path, *names: str) -> torch.Tensor:
    for name in names:
        candidate = path / name
        if candidate.is_file():
            return torch.from_numpy(np.load(candidate)).float()
    raise FileNotFoundError(f"Missing statistics under {path}: expected one of {names}")


class KimodoRootConditioner(nn.Module):
    """Convert normalized global root x0 into normalized finite-difference root features."""

    def __init__(
        self,
        motion_stats_dir: str | Path,
        local_root_stats_dir: str | Path,
        fps: float = 30.0,
        eps: float = 1e-6,
        variance_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        motion_dir = Path(motion_stats_dir).expanduser().resolve()
        local_dir = Path(local_root_stats_dir).expanduser().resolve()
        motion_mean = _load_stat(motion_dir, "Mean.npy", "mean.npy")
        motion_std = _load_stat(motion_dir, "Std.npy", "std.npy")
        local_mean = _load_stat(local_dir, "Mean.npy", "mean.npy")
        local_std = _load_stat(local_dir, "Std.npy", "std.npy")
        if motion_mean.numel() < ROOT_DIM or motion_std.numel() < ROOT_DIM:
            raise ValueError("Motion statistics do not contain the five global-root channels")
        if local_mean.numel() != LOCAL_ROOT_DIM or local_std.numel() != LOCAL_ROOT_DIM:
            raise ValueError(
                f"Local-root statistics must have {LOCAL_ROOT_DIM} entries, got "
                f"{local_mean.numel()} and {local_std.numel()}"
            )
        self.register_buffer("global_root_mean", motion_mean[:ROOT_DIM].view(1, 1, ROOT_DIM))
        self.register_buffer(
            "global_root_std", motion_std[:ROOT_DIM].clamp_min(float(eps)).view(1, 1, ROOT_DIM)
        )
        self.register_buffer("local_root_mean", local_mean.view(1, 1, LOCAL_ROOT_DIM))
        self.register_buffer(
            "local_root_std", local_std.clamp_min(float(eps)).view(1, 1, LOCAL_ROOT_DIM)
        )
        self.fps = float(fps)
        self.variance_eps = float(variance_eps)

    def _scale(self, std: torch.Tensor) -> torch.Tensor:
        if self.variance_eps <= 0:
            return std.float()
        return torch.sqrt(std.float().square() + self.variance_eps)

    def unnormalize_global_root(self, root_norm: torch.Tensor) -> torch.Tensor:
        if root_norm.shape[-1] != ROOT_DIM:
            raise ValueError(f"Expected global root [...,{ROOT_DIM}], got {tuple(root_norm.shape)}")
        return root_norm.float() * self._scale(self.global_root_std) + self.global_root_mean.float()

    def normalize_local_root(self, local_root: torch.Tensor) -> torch.Tensor:
        return (local_root - self.local_root_mean.float()) / self._scale(self.local_root_std)

    def forward(self, root_norm: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if root_norm.ndim != 3 or root_norm.shape[-1] != ROOT_DIM:
            raise ValueError(f"Expected root_norm [B,T,{ROOT_DIM}], got {tuple(root_norm.shape)}")
        bsz, frames, _ = root_norm.shape
        lengths = lengths.to(device=root_norm.device, dtype=torch.long).clamp(min=1, max=frames)

        # The bridge stays FP32 under BF16 training because atan2 and finite differences
        # are sensitive to small heading vectors.
        root = self.unnormalize_global_root(root_norm)
        pos = root[..., ROOT_SLICE]
        heading = root[..., HEADING_SLICE]
        cos = heading[..., 0]
        sin = heading[..., 1]

        local = root.new_zeros((bsz, frames, LOCAL_ROOT_DIM))
        if frames > 1:
            dot = cos[:, 1:] * cos[:, :-1] + sin[:, 1:] * sin[:, :-1]
            cross = sin[:, 1:] * cos[:, :-1] - cos[:, 1:] * sin[:, :-1]
            local[:, :-1, 0] = torch.atan2(cross, dot) * self.fps
            delta = (pos[:, 1:] - pos[:, :-1]) * self.fps
            local[:, :-1, 1] = delta[..., 0]
            local[:, :-1, 2] = delta[..., 2]

            batch = torch.arange(bsz, device=root.device)
            has_pair = lengths > 1
            if bool(has_pair.any()):
                dst = lengths[has_pair] - 1
                src = lengths[has_pair] - 2
                local[batch[has_pair], dst, :3] = local[batch[has_pair], src, :3]
            if bool((~has_pair).any()):
                local[batch[~has_pair], 0, :3] = 0.0
        local[..., 3] = pos[..., 1]
        return self.normalize_local_root(local).to(dtype=root_norm.dtype)
