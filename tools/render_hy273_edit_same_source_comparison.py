#!/usr/bin/env python3
"""Render matched-noise same-source Edit outputs as side-by-side GIFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_slices import (
    SMPLX22_PARENTS,
    reconstruct_global_joints_from_features,
)


def parse_system(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("system must be LABEL=DIR")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip() or not path.is_dir():
        raise argparse.ArgumentTypeError(f"invalid system: {value}")
    return label.strip(), path


def joints(values: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return (
            reconstruct_global_joints_from_features(torch.from_numpy(values).float())
            .cpu()
            .numpy()
        )


def edges() -> list[tuple[int, int]]:
    return [
        (int(parent), child)
        for child, parent in enumerate(SMPLX22_PARENTS.tolist())
        if int(parent) >= 0
    ]


def common_limits(motions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([motion.reshape(-1, 3) for motion in motions], axis=0)
    points = points[np.isfinite(points).all(axis=-1)]
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    half = max(float((upper - lower).max()) * 0.55, 1.0)
    lower = center - half
    upper = center + half
    lower[1] = min(float(lower[1]), 0.0)
    upper[1] = max(float(upper[1]), float(lower[1]) + 1.0)
    return lower, upper


def draw(ax: Any, pose: np.ndarray, color: str) -> None:
    for parent, child in edges():
        segment = pose[[parent, child]]
        ax.plot(
            segment[:, 0], segment[:, 2], segment[:, 1],
            color=color, linewidth=2.0, alpha=0.95,
        )


def render(
    panels: list[tuple[str, np.ndarray, str]],
    *,
    title: str,
    path: Path,
    fps: int,
    stride: int,
) -> None:
    motions = [motion[::stride] for _, motion, _ in panels]
    lower, upper = common_limits(motions)
    figure = plt.figure(figsize=(4.0 * len(panels), 4.4))
    axes = [figure.add_subplot(1, len(panels), index + 1, projection="3d") for index in range(len(panels))]
    figure.suptitle(textwrap.fill(title, width=110), fontsize=11)

    def update(frame: int) -> None:
        for axis, (panel_title, _, color), motion in zip(axes, panels, motions):
            axis.clear()
            axis.view_init(elev=18, azim=-68)
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[2], upper[2])
            axis.set_zlim(lower[1], upper[1])
            axis.set_title(panel_title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([])
            axis.grid(False)
            draw(axis, motion[min(frame, len(motion) - 1)], color)

    animation_obj = FuncAnimation(
        figure,
        update,
        frames=max(len(motion) for motion in motions),
        interval=1000.0 / max(fps // stride, 1),
        repeat=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    animation_obj.save(path, writer=animation.PillowWriter(fps=max(fps // stride, 1)))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", type=parse_system, required=True)
    parser.add_argument("--branch_system", default="softplus405k")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()

    systems = dict(args.system)
    if len(systems) != len(args.system):
        raise ValueError("System labels must be unique")
    if args.branch_system not in systems:
        raise ValueError("--branch_system must name one of --system")
    loaded: dict[str, dict[str, Any]] = {}
    reference_rows = None
    for label, directory in systems.items():
        metadata = json.loads((directory / "metadata.json").read_text())
        rows = metadata["rows"]
        identity = [
            (row["pair_id"], row["instruction"], row["sibling_instruction"], row["frames"])
            for row in rows
        ]
        if reference_rows is None:
            reference_rows = identity
        elif identity != reference_rows:
            raise RuntimeError(f"Row identity differs for system {label}")
        loaded[label] = {
            "metadata": metadata,
            "source": np.load(directory / "source.npy"),
            "target": np.load(directory / "target.npy"),
            "correct": np.load(directory / "correct.npy"),
            "sibling": np.load(directory / "sibling.npy"),
            "empty": np.load(directory / "empty.npy"),
        }

    first_label = next(iter(systems))
    source_joints = joints(loaded[first_label]["source"])
    target_joints = joints(loaded[first_label]["target"])
    correct_joints = {
        label: joints(values["correct"]) for label, values in loaded.items()
    }
    branch_label = str(args.branch_system)
    branch_joints = {
        branch: joints(loaded[branch_label][branch])
        for branch in ("correct", "sibling", "empty")
    }
    output_dir = args.output_dir.expanduser().resolve()
    rows = loaded[first_label]["metadata"]["rows"]
    for row_index, row in enumerate(rows):
        frames = int(row["frames"])
        pair_id = str(row["pair_id"])
        instruction = str(row["instruction"])
        sibling_instruction = str(row["sibling_instruction"])
        system_panels = [
            ("source", source_joints[row_index, :frames], "#6b7280"),
            ("target", target_joints[row_index, :frames], "#059669"),
        ]
        for label in systems:
            mse = float(
                loaded[label]["metadata"]["rows"][row_index][
                    "continuous_target_mse"
                ]["correct"]
            )
            system_panels.append(
                (f"{label}\ncorrect MSE={mse:.3f}", correct_joints[label][row_index, :frames], "#2563eb")
            )
        render(
            system_panels,
            title=f"pair {pair_id} | {instruction}",
            path=output_dir / "systems_correct" / f"{row_index:02d}_{pair_id}.gif",
            fps=int(args.fps),
            stride=int(args.stride),
        )

        branch_panels = [
            ("source", source_joints[row_index, :frames], "#6b7280"),
            ("target", target_joints[row_index, :frames], "#059669"),
        ]
        for branch in ("correct", "sibling", "empty"):
            mse = float(
                loaded[branch_label]["metadata"]["rows"][row_index][
                    "continuous_target_mse"
                ][branch]
            )
            branch_panels.append(
                (f"{branch_label} {branch}\nMSE={mse:.3f}", branch_joints[branch][row_index, :frames], "#7c3aed")
            )
        render(
            branch_panels,
            title=(
                f"pair {pair_id} | correct: {instruction} | "
                f"sibling: {sibling_instruction}"
            ),
            path=output_dir / f"{branch_label}_text_branches" / f"{row_index:02d}_{pair_id}.gif",
            fps=int(args.fps),
            stride=int(args.stride),
        )

    summary = {
        "format": "hy273_r13_same_source_visual_comparison_v1",
        "systems": {label: str(path) for label, path in systems.items()},
        "branch_system": branch_label,
        "rows": rows,
        "fps": int(args.fps),
        "stride": int(args.stride),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "gifs": 2 * len(rows)}))


if __name__ == "__main__":
    main()
