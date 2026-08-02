#!/usr/bin/env python3
"""Render fixed-role Reaction predictions as source/reactor comparison GIFs."""

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
import torch
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_slices import (
    DIM_HY273,
    SMPLX22_PARENTS,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)


SOURCE_COLOR = "#2563eb"
PANELS = (
    ("Ground-truth reactor", "target.npy", "#16a34a"),
    ("Source + text", "source_text.npy", "#dc2626"),
    ("Source only", "source_only.npy", "#ea580c"),
    ("Shuffled text", "shuffled_text.npy", "#7c3aed"),
)


def _load_motion(path: Path) -> np.ndarray:
    value = np.load(path)
    if value.ndim != 2 or value.shape[-1] != DIM_HY273:
        raise ValueError(f"{path} must have shape [T,{DIM_HY273}], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path} contains non-finite values")
    return value.astype(np.float32, copy=False)


def _to_joints(features: np.ndarray, source: str) -> np.ndarray:
    tensor = torch.from_numpy(features)
    with torch.no_grad():
        joints = (
            fk_positions_from_global_rot6d(tensor)
            if source == "fk"
            else reconstruct_global_joints_from_features(tensor)
        )
    return joints.cpu().numpy().astype(np.float32, copy=False)


def _select_rows(
    report: dict[str, Any],
    *,
    max_videos: int,
    requested_uids: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows = list(report["per_sample"]["source_text"])
    by_uid = {str(row["uid"]): row for row in rows}
    if requested_uids:
        missing = [uid for uid in requested_uids if uid not in by_uid]
        if missing:
            raise KeyError(f"Reaction report does not contain requested UIDs: {missing}")
        return [by_uid[uid] for uid in requested_uids[:max_videos]]

    ordered = sorted(
        rows,
        key=lambda row: (str(row["action_category"]), str(row["uid"])),
    )
    selected: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    selected_uids: set[str] = set()
    for row in ordered:
        action = str(row["action_category"])
        if action in seen_actions:
            continue
        selected.append(row)
        seen_actions.add(action)
        selected_uids.add(str(row["uid"]))
        if len(selected) == max_videos:
            return selected
    for row in ordered:
        if str(row["uid"]) in selected_uids:
            continue
        selected.append(row)
        if len(selected) == max_videos:
            break
    return selected


def _axis_limits(joints: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    finite = np.concatenate([value.reshape(-1, 3) for value in joints], axis=0)
    finite = finite[np.isfinite(finite).all(axis=-1)]
    if not finite.size:
        return np.asarray([-2.0, 0.0, -2.0]), np.asarray([2.0, 2.0, 2.0])
    lower = finite.min(axis=0)
    upper = finite.max(axis=0)
    center = (lower + upper) * 0.5
    horizontal_half = max(
        float(upper[0] - lower[0]) * 0.55,
        float(upper[2] - lower[2]) * 0.55,
        1.25,
    )
    lower[0], upper[0] = center[0] - horizontal_half, center[0] + horizontal_half
    lower[2], upper[2] = center[2] - horizontal_half, center[2] + horizontal_half
    lower[1] = min(float(lower[1]) - 0.1, 0.0)
    upper[1] = max(float(upper[1]) + 0.1, float(lower[1]) + 1.5)
    return lower, upper


def _draw_actor(axis, joints: np.ndarray, color: str, alpha: float) -> None:
    for child, parent in enumerate(SMPLX22_PARENTS.tolist()):
        if parent < 0:
            continue
        segment = joints[[parent, child]]
        axis.plot(
            segment[:, 0],
            segment[:, 2],
            segment[:, 1],
            color=color,
            linewidth=2.0,
            alpha=alpha,
        )
    axis.scatter(
        joints[0, 0],
        joints[0, 2],
        joints[0, 1],
        color=color,
        s=18,
        depthshade=False,
        alpha=alpha,
    )


def _render_case(
    *,
    source: np.ndarray,
    reactors: list[tuple[str, np.ndarray, str]],
    title: str,
    output: Path,
    fps: int,
    stride: int,
) -> None:
    source = source[::stride]
    reactors = [(name, value[::stride], color) for name, value, color in reactors]
    frame_count = min([source.shape[0], *[value.shape[0] for _, value, _ in reactors]])
    lower, upper = _axis_limits([source, *[value for _, value, _ in reactors]])
    fig = plt.figure(figsize=(12.4, 9.2), facecolor="white")
    axes = [fig.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    fig.suptitle(textwrap.fill(title, width=110), fontsize=11, y=0.98)

    def update(frame_index: int) -> None:
        for axis, (panel_name, reactor, reactor_color) in zip(axes, reactors):
            axis.clear()
            axis.view_init(elev=18, azim=-68)
            axis.set_xlim(lower[0], upper[0])
            axis.set_ylim(lower[2], upper[2])
            axis.set_zlim(lower[1], upper[1])
            axis.set_title(panel_name, fontsize=10, pad=7)
            axis.set_xlabel("x", fontsize=7)
            axis.set_ylabel("z", fontsize=7)
            axis.set_zlabel("y", fontsize=7)
            axis.tick_params(labelsize=6)
            axis.grid(False)
            _draw_actor(axis, source[frame_index], SOURCE_COLOR, 0.9)
            _draw_actor(axis, reactor[frame_index], reactor_color, 0.95)
            for actor, color in ((source, SOURCE_COLOR), (reactor, reactor_color)):
                trail = actor[: frame_index + 1, 0]
                axis.plot(
                    trail[:, 0],
                    trail[:, 2],
                    trail[:, 1],
                    color=color,
                    linewidth=1.0,
                    alpha=0.35,
                )

    rendered = FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=1000.0 * stride / max(int(fps), 1),
        repeat=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(
        output,
        writer=animation.PillowWriter(fps=max(int(fps) // stride, 1)),
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--prediction_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_videos", type=int, default=12)
    parser.add_argument("--uids", default="")
    parser.add_argument("--joint_source", choices=("position", "fk"), default="position")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    if args.max_videos < 1 or args.fps < 1 or args.stride < 1:
        parser.error("max_videos, fps, and stride must be positive")

    report_path = Path(args.report_json).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("format") != "hy273_fixed_role_reaction_eval_v2":
        raise RuntimeError("Renderer requires a fixed-role Reaction v2 report")
    prediction_dir = (
        Path(args.prediction_dir).expanduser().resolve()
        if args.prediction_dir
        else report_path.parent / "predictions"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    requested_uids = tuple(uid.strip() for uid in args.uids.split(",") if uid.strip())
    rows = _select_rows(
        report,
        max_videos=int(args.max_videos),
        requested_uids=requested_uids,
    )

    written: list[str] = []
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        uid = str(row["uid"])
        case_dir = prediction_dir / uid
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
        length = int(metadata["length"])
        source = _to_joints(_load_motion(case_dir / "source.npy")[:length], args.joint_source)
        reactors = [
            (
                panel_name,
                _to_joints(_load_motion(case_dir / filename)[:length], args.joint_source),
                color,
            )
            for panel_name, filename, color in PANELS
        ]
        output = output_dir / f"{index:03d}_{uid}.gif"
        _render_case(
            source=source,
            reactors=reactors,
            title=(
                f"{uid} | source actor: person {int(metadata['actor_person_index']) + 1} | "
                f"{metadata['text']}"
            ),
            output=output,
            fps=int(args.fps),
            stride=int(args.stride),
        )
        written.append(str(output))
        selected.append(
            {
                "uid": uid,
                "action_category": str(row["action_category"]),
                "text": str(metadata["text"]),
                "actor_person_index": int(metadata["actor_person_index"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "report_json": str(report_path),
                "prediction_dir": str(prediction_dir),
                "selection": "requested_uids" if requested_uids else "action_balanced",
                "joint_source": args.joint_source,
                "source_actor_color": SOURCE_COLOR,
                "selected_cases": selected,
                "videos": written,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "videos": len(written)}))


if __name__ == "__main__":
    main()
