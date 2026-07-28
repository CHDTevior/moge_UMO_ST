#!/usr/bin/env python3
"""Build train-only HY273 and Kimodo local-root statistics with yaw quadrature."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_slices import (
    BODY_DIM,
    CONTACT_SLICE,
    DIM_HY273,
    FALLBACK_SHORT_CLIPS,
    HEADING_SLICE,
    ROOT_DIM,
    ROOT_SLICE,
)


class RunningMoments:
    def __init__(self, dim: int) -> None:
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        self.sum += x64.sum(axis=0)
        self.sum_sq += np.square(x64).sum(axis=0)
        self.count += int(x64.shape[0])

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0:
            raise RuntimeError("No frames were accumulated")
        mean = self.sum / self.count
        var = np.maximum(self.sum_sq / self.count - np.square(mean), 1e-12)
        std = np.sqrt(var)
        std[std < 1e-6] = 1.0
        return mean.astype(np.float32), std.astype(np.float32)


def global_root_to_local(root: torch.Tensor, fps: float) -> torch.Tensor:
    if root.ndim != 2 or root.shape[-1] != ROOT_DIM:
        raise ValueError(f"Expected root [T,{ROOT_DIM}], got {tuple(root.shape)}")
    frames = root.shape[0]
    out = root.new_zeros((frames, 4), dtype=torch.float32)
    if frames > 1:
        heading = root[:, HEADING_SLICE]
        cos, sin = heading[:, 0], heading[:, 1]
        dot = cos[1:] * cos[:-1] + sin[1:] * sin[:-1]
        cross = sin[1:] * cos[:-1] - cos[1:] * sin[:-1]
        out[:-1, 0] = torch.atan2(cross, dot) * float(fps)
        delta = (root[1:, ROOT_SLICE] - root[:-1, ROOT_SLICE]) * float(fps)
        out[:-1, 1] = delta[:, 0]
        out[:-1, 2] = delta[:, 2]
        out[-1, :3] = out[-2, :3]
    out[:, 3] = root[:, ROOT_SLICE.start + 1]
    return out


def save_stats(path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "Mean.npy", mean)
    np.save(path / "Std.npy", std)
    np.save(path / "mean.npy", mean)
    np.save(path / "std.npy", std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_files", type=int, default=0)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_file = data_root / "split_existing" / "train.txt"
    ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if args.max_files > 0:
        ids = ids[: args.max_files]

    full_moments = RunningMoments(DIM_HY273)
    local_moments = RunningMoments(4)
    processed: list[str] = []
    skipped: list[str] = []
    targets = torch.tensor([0.0, 0.5 * torch.pi, torch.pi, 1.5 * torch.pi], dtype=torch.float32)

    for motion_id in ids:
        rel = f"motion_data/{motion_id}.npy"
        path = data_root / rel
        if rel in FALLBACK_SHORT_CLIPS or not path.is_file():
            skipped.append(rel)
            continue
        raw = torch.from_numpy(np.load(path).astype(np.float32, copy=False))
        if raw.ndim != 2 or raw.shape[-1] != DIM_HY273 or raw.shape[0] < 2:
            skipped.append(rel)
            continue
        shifted = root_origin_shift(raw)
        first_heading = shifted[0, HEADING_SLICE]
        current = torch.atan2(first_heading[1], first_heading[0])
        for target in targets:
            augmented = apply_yaw_rotation(shifted, target - current)
            full_moments.update(augmented.numpy())
            local_moments.update(global_root_to_local(augmented[:, :ROOT_DIM], args.fps).numpy())
        processed.append(rel)

    full_mean, full_std = full_moments.finalize()
    local_mean, local_std = local_moments.finalize()
    full_mean[CONTACT_SLICE] = 0.0
    full_std[CONTACT_SLICE] = 1.0

    # Four equally-spaced headings make the planar moments isotropic. Enforce the
    # exact invariant to avoid tiny accumulation-order differences.
    local_planar_var = float((local_std[1] ** 2 + local_mean[1] ** 2 + local_std[2] ** 2 + local_mean[2] ** 2) / 2.0)
    local_mean[1:3] = 0.0
    local_std[1:3] = np.sqrt(local_planar_var)

    save_stats(output_dir / "full", full_mean, full_std)
    save_stats(output_dir / "global_root", full_mean[:ROOT_DIM], full_std[:ROOT_DIM])
    save_stats(output_dir / "body", full_mean[ROOT_DIM:], full_std[ROOT_DIM:])
    save_stats(output_dir / "local_root", local_mean, local_std)

    split_hash = hashlib.sha256(split_file.read_bytes()).hexdigest()
    manifest = {
        "format": "hy273_redenoise_kimodo_like_stats_v1",
        "data_root": str(data_root),
        "split": "train",
        "split_file": str(split_file),
        "split_sha256": split_hash,
        "fps": float(args.fps),
        "yaw_quadrature_targets_rad": [float(x) for x in targets.tolist()],
        "root_origin_shift": True,
        "contacts_normalized": False,
        "files_processed": len(processed),
        "files_skipped": len(skipped),
        "augmented_frames": full_moments.count,
        "dims": {"full": DIM_HY273, "global_root": ROOT_DIM, "body": BODY_DIM, "local_root": 4},
        "skipped_examples": skipped[:20],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
