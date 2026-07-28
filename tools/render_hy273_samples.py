"""Render HY273 raw-flow sample arrays to lightweight skeleton videos."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hy273_slices import (
    KIMODO_EE_JOINTS,
    SMPLX22_PARENTS,
    reconstruct_global_joints_from_features,
)


def _joint_edges() -> list[tuple[int, int]]:
    parents = SMPLX22_PARENTS.tolist()
    return [(parent, child) for child, parent in enumerate(parents) if parent >= 0]


def _load_features(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 3 or data.shape[-1] != 273:
        raise ValueError(f"{path} must have shape [B,T,273], got {data.shape}")
    return data.astype(np.float32, copy=False)


def _load_prediction(sample_dir: Path) -> np.ndarray:
    for name in ("samples_raw.npy", "samples.npy"):
        path = sample_dir / name
        if path.is_file():
            return _load_features(path)
    raise FileNotFoundError(
        f"No HY273 prediction found under {sample_dir}; expected samples_raw.npy or samples.npy"
    )


def _load_lengths(sample_dir: Path, metadata: dict, batch_size: int, frames: int) -> np.ndarray:
    lengths_path = sample_dir / "lengths.npy"
    if lengths_path.is_file():
        lengths = np.load(lengths_path).astype(np.int64, copy=False)
    else:
        lengths = np.asarray(metadata.get("lengths", []), dtype=np.int64)
    if lengths.shape == (batch_size,):
        return np.clip(lengths, 1, frames)
    return np.full((batch_size,), frames, dtype=np.int64)


def _features_to_joints(features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        joints = reconstruct_global_joints_from_features(torch.from_numpy(features)).cpu().numpy()
    return joints.astype(np.float32, copy=False)


def _masked_observed_joints(observed: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joints = _features_to_joints(observed)
    joint_mask = mask[..., 5:71].reshape(mask.shape[0], mask.shape[1], 22, 3).any(axis=-1)
    root_mask = mask[..., 0:3].any(axis=-1)
    joint_mask[..., 0] |= root_mask
    return joints, joint_mask


def _axis_limits(*motions: np.ndarray, radius: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([m.reshape(-1, 3) for m in motions], axis=0)
    finite = stacked[np.isfinite(stacked).all(axis=-1)]
    if finite.size == 0:
        return np.array([-2.0, 0.0, -2.0]), np.array([2.0, 2.0, 2.0])
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    center = (mins + maxs) * 0.5
    half = float(np.max(maxs - mins) * 0.55)
    if radius is not None:
        half = max(half, float(radius) * 0.5)
    half = max(half, 1.0)
    lower = center - half
    upper = center + half
    lower[1] = min(lower[1], 0.0)
    upper[1] = max(upper[1], lower[1] + 1.0)
    return lower, upper


def _draw_skeleton(ax, joints: np.ndarray, edges: Sequence[tuple[int, int]], color: str, alpha: float, lw: float) -> None:
    for parent, child in edges:
        seg = joints[[parent, child]]
        ax.plot(seg[:, 0], seg[:, 2], seg[:, 1], color=color, alpha=alpha, linewidth=lw)


def render_sample(
    pred_joints: np.ndarray,
    observed_joints: np.ndarray,
    joint_mask: np.ndarray,
    save_path: Path,
    title: str,
    fps: int,
    stride: int,
    radius: float | None,
    center_root: bool,
    hold_last_seconds: float,
) -> None:
    pred_joints = pred_joints[::stride]
    observed_joints = observed_joints[::stride]
    joint_mask = joint_mask[::stride]
    if center_root:
        pred_root = pred_joints[:, 0:1, [0, 2]]
        obs_root = observed_joints[:, 0:1, [0, 2]]
        pred_joints = pred_joints.copy()
        observed_joints = observed_joints.copy()
        pred_joints[:, :, 0] -= pred_root[:, :, 0]
        pred_joints[:, :, 2] -= pred_root[:, :, 1]
        observed_joints[:, :, 0] -= obs_root[:, :, 0]
        observed_joints[:, :, 2] -= obs_root[:, :, 1]
    if hold_last_seconds > 0.0 and pred_joints.shape[0] > 0:
        hold_frames = max(int(round(float(hold_last_seconds) * max(fps, 1) / max(stride, 1))), 0)
        if hold_frames > 0:
            pred_joints = np.concatenate(
                [pred_joints, np.repeat(pred_joints[-1:], hold_frames, axis=0)],
                axis=0,
            )
            observed_joints = np.concatenate(
                [observed_joints, np.repeat(observed_joints[-1:], hold_frames, axis=0)],
                axis=0,
            )
            joint_mask = np.concatenate(
                [joint_mask, np.repeat(joint_mask[-1:], hold_frames, axis=0)],
                axis=0,
            )
    lower, upper = _axis_limits(pred_joints, observed_joints, radius=radius)
    edges = _joint_edges()
    ee_ids = np.array(list(KIMODO_EE_JOINTS), dtype=np.int64)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    frame_count = pred_joints.shape[0]

    def update(index: int) -> None:
        ax.clear()
        ax.view_init(elev=18, azim=-68)
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[2], upper[2])
        ax.set_zlim(lower[1], upper[1])
        ax.set_title(textwrap.fill(title, width=58), fontsize=9, pad=14)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        ax.grid(False)

        _draw_skeleton(ax, observed_joints[index], edges, color="#9ca3af", alpha=0.35, lw=1.5)
        _draw_skeleton(ax, pred_joints[index], edges, color="#2563eb", alpha=0.95, lw=2.2)

        mask_now = joint_mask[index]
        if mask_now.any():
            pts = observed_joints[index, mask_now]
            ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], color="#dc2626", s=28, depthshade=False)
        ee_pts = pred_joints[index, ee_ids]
        ax.scatter(ee_pts[:, 0], ee_pts[:, 2], ee_pts[:, 1], color="#059669", s=14, depthshade=False)

    ani = FuncAnimation(fig, update, frames=frame_count, interval=1000.0 / max(fps, 1), repeat=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_fps = max(fps // max(stride, 1), 1)
    if save_path.suffix == ".gif":
        writer = animation.PillowWriter(fps=out_fps)
        ani.save(save_path, writer=writer)
    else:
        ani.save(save_path, fps=out_fps)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_videos", type=int, default=4)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.0)
    parser.add_argument("--format", choices=["auto", "mp4", "gif"], default="auto")
    parser.add_argument("--center_root", action="store_true")
    parser.add_argument("--hold_last_seconds", type=float, default=0.0)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    pred = _load_prediction(sample_dir)
    observed_path = sample_dir / "observed.npy"
    mask_path = sample_dir / "mask.npy"
    observed = _load_features(observed_path) if observed_path.is_file() else np.zeros_like(pred)
    mask = (
        np.load(mask_path).astype(bool, copy=False)
        if mask_path.is_file()
        else np.zeros_like(pred, dtype=bool)
    )
    if observed.shape != pred.shape:
        raise ValueError(f"observed shape {observed.shape} does not match samples shape {pred.shape}")
    if mask.shape != pred.shape:
        raise ValueError(f"mask shape {mask.shape} does not match samples shape {pred.shape}")

    pred_joints = _features_to_joints(pred)
    observed_joints, joint_mask = _masked_observed_joints(observed, mask)
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    lengths = _load_lengths(sample_dir, metadata, pred.shape[0], pred.shape[1])
    texts = metadata.get("texts", [])
    modes = metadata.get("control_modes", [])

    output_dir = Path(args.output_dir) if args.output_dir else sample_dir / "videos"
    limit = min(int(args.max_videos), pred.shape[0])
    radius = None if args.radius <= 0 else float(args.radius)
    video_format = args.format
    if video_format == "auto":
        video_format = "mp4" if shutil.which("ffmpeg") else "gif"
    written: list[str] = []
    for idx in range(limit):
        title = f"{idx:02d}"
        if idx < len(modes):
            title += f" {modes[idx]}"
        if idx < len(texts):
            title += f" | {texts[idx]}"
        save_path = output_dir / f"sample_{idx:02d}.{video_format}"
        length = int(lengths[idx])
        render_sample(
            pred_joints[idx, :length],
            observed_joints[idx, :length],
            joint_mask[idx, :length],
            save_path=save_path,
            title=title,
            fps=args.fps,
            stride=max(int(args.stride), 1),
            radius=radius,
            center_root=bool(args.center_root),
            hold_last_seconds=float(args.hold_last_seconds),
        )
        written.append(str(save_path))
    print(json.dumps({"sample_dir": str(sample_dir), "videos": written}, indent=2))


if __name__ == "__main__":
    main()
