"""Render the formally replayed HY273 Kimodo-control gallery.

The renderer consumes only saved gallery arrays. It visualizes the raw model
output before terminal exact clamping, plus the constraint targets that were
actually supplied to the sampler.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hy273_slices import (  # noqa: E402
    CONTACT_JOINTS,
    CONTACT_SLICE,
    SMPLX22_PARENTS,
    cont6d_to_matrix,
    fk_positions_from_global_rot6d,
    split_global_rot6d,
)


COLORS = {
    "ink": (24, 31, 42, 255),
    "muted": (91, 101, 116, 255),
    "grid": (208, 215, 224, 170),
    "grid_major": (174, 184, 197, 210),
    "blue": (31, 111, 235, 255),
    "blue_shadow": (9, 52, 116, 110),
    "trail": (88, 178, 231, 72),
    "orange": (232, 128, 30, 255),
    "orange_faint": (232, 128, 30, 90),
    "target": (220, 55, 55, 255),
    "target_faint": (220, 55, 55, 75),
    "pose": (240, 174, 38, 255),
    "pose_faint": (240, 174, 38, 55),
    "green": (20, 143, 86, 255),
    "axis_x": (220, 55, 55, 255),
    "axis_y": (24, 158, 88, 255),
    "axis_z": (38, 103, 214, 255),
    "danger": (202, 42, 42, 255),
}

SUBTYPE_LABELS = {
    "path_2dpos": "Dense root path",
    "path_2dposrot": "Dense root path + heading",
    "waypoint_2dpos": "Sparse root waypoints",
    "waypoint_2dposrot": "Sparse waypoints + heading",
    "inbetweening": "Full-body inbetweening",
    "random": "Sparse full-body keyframes",
    "feet_posrot": "Feet position + rotation",
    "hands_posrot": "Hands position + rotation",
    "hands_feet_posrot": "Hands + feet position + rotation",
    "root_ee_hands_feet_posrot_fullbody": "Root + hands/feet + full-body",
    "root_ee_hands_posrot": "Root + hands position/rotation",
    "root_ee_hands_posrot_fullbody": "Root + hands + full-body",
    "root_path_fullbody": "Dense root path + full-body",
    "contact_only_sparse": "Sparse foot-contact control",
    "root_sparse_contact": "Sparse root + contact control",
    "root_dense_contact": "Dense root + contact control",
    "endpoints_contact": "End effectors + contact control",
    "fullpose_contact": "Full-pose + contact control",
    "mixed_contact": "Mixed root/end-effector/full-pose/contact control",
}


@dataclass(frozen=True)
class RenderTask:
    gallery_dir: str
    case_dir: str
    width: int
    height: int
    fps: int
    stride: int
    gif_fps: int
    make_mp4: bool
    make_gif: bool
    overwrite: bool


class WorldProjector:
    def __init__(
        self,
        points: np.ndarray,
        viewport: tuple[int, int, int, int],
        *,
        azimuth_degrees: float = 38.0,
        elevation_degrees: float = 19.0,
    ) -> None:
        self.viewport = viewport
        azimuth = math.radians(azimuth_degrees)
        elevation = math.radians(elevation_degrees)
        self.cos_a = math.cos(azimuth)
        self.sin_a = math.sin(azimuth)
        self.cos_e = math.cos(elevation)
        self.sin_e = math.sin(elevation)
        camera = self.camera(np.asarray(points, dtype=np.float32).reshape(-1, 3))
        finite = camera[np.isfinite(camera).all(axis=-1)]
        if not finite.size:
            finite = np.asarray([[-2.0, -1.0], [2.0, 2.0]], dtype=np.float32)
        low = finite.min(axis=0)
        high = finite.max(axis=0)
        span = np.maximum(high - low, 1e-3)
        low -= np.maximum(span * np.asarray([0.07, 0.10]), [0.25, 0.25])
        high += np.maximum(span * np.asarray([0.07, 0.10]), [0.25, 0.25])
        left, top, right, bottom = viewport
        available_w = max(right - left, 1)
        available_h = max(bottom - top, 1)
        scale = min(available_w / max(high[0] - low[0], 1e-4), available_h / max(high[1] - low[1], 1e-4))
        used_w = (high[0] - low[0]) * scale
        used_h = (high[1] - low[1]) * scale
        self.scale = float(scale)
        self.origin_x = float(left + (available_w - used_w) * 0.5 - low[0] * scale)
        self.origin_y = float(top + (available_h - used_h) * 0.5 + high[1] * scale)

    def camera(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        u = points[..., 0] * self.cos_a - points[..., 2] * self.sin_a
        depth = points[..., 0] * self.sin_a + points[..., 2] * self.cos_a
        v = points[..., 1] * self.cos_e - depth * self.sin_e
        return np.stack([u, v], axis=-1)

    def map(self, points: np.ndarray) -> np.ndarray:
        camera = self.camera(points)
        x = self.origin_x + camera[..., 0] * self.scale
        y = self.origin_y - camera[..., 1] * self.scale
        return np.stack([x, y], axis=-1)


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size=size)
    return ImageFont.load_default()


def fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    text = " ".join(str(text).split())
    if draw.textlength(text, font=font) <= max_width:
        return text
    suffix = "..."
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if draw.textlength(text[:middle] + suffix, font=font) <= max_width:
            low = middle
        else:
            high = middle - 1
    return text[:low].rstrip() + suffix


def joint_edges() -> list[tuple[int, int]]:
    return [
        (int(parent), child)
        for child, parent in enumerate(SMPLX22_PARENTS.tolist())
        if parent >= 0
    ]


def rgba_layer(image: Image.Image) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    return layer, ImageDraw.Draw(layer)


def composite(image: Image.Image, layer: Image.Image) -> None:
    image.alpha_composite(layer)


def screen_points(projector: WorldProjector, points: np.ndarray) -> list[tuple[int, int]]:
    mapped = projector.map(points)
    return [(int(round(x)), int(round(y))) for x, y in mapped]


def draw_skeleton(
    draw: ImageDraw.ImageDraw,
    projector: WorldProjector,
    joints: np.ndarray,
    *,
    color: tuple[int, int, int, int],
    width: int,
    joint_radius: int = 0,
) -> None:
    mapped = screen_points(projector, joints)
    for parent, child in joint_edges():
        draw.line([mapped[parent], mapped[child]], fill=color, width=width)
    if joint_radius > 0:
        for x, y in mapped:
            draw.ellipse(
                (x - joint_radius, y - joint_radius, x + joint_radius, y + joint_radius),
                fill=color,
            )


def draw_marker(
    draw: ImageDraw.ImageDraw,
    projector: WorldProjector,
    point: np.ndarray,
    *,
    color: tuple[int, int, int, int],
    radius: int,
    width: int = 0,
) -> None:
    x, y = screen_points(projector, np.asarray(point)[None])[0]
    box = (x - radius, y - radius, x + radius, y + radius)
    if width:
        draw.ellipse(box, outline=color, width=width)
    else:
        draw.ellipse(box, fill=color)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    projector: WorldProjector,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: tuple[int, int, int, int],
    width: int = 3,
) -> None:
    p0, p1 = screen_points(projector, np.stack([start, end]))
    draw.line([p0, p1], fill=color, width=width)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    norm = max(math.hypot(dx, dy), 1e-6)
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    size = 7.0 + width
    wing = 0.55 * size
    tip = (p1[0], p1[1])
    left = (int(p1[0] - ux * size + px * wing), int(p1[1] - uy * size + py * wing))
    right = (int(p1[0] - ux * size - px * wing), int(p1[1] - uy * size - py * wing))
    draw.polygon([tip, left, right], fill=color)


def evenly_spaced(values: np.ndarray, max_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.size <= max_count:
        return values
    indices = np.linspace(0, values.size - 1, max_count).round().astype(np.int64)
    return values[np.unique(indices)]


def nearest_control_frame(frames: np.ndarray, current: int) -> int | None:
    if not frames.size:
        return None
    upcoming = frames[frames >= current]
    return int(upcoming[0] if upcoming.size else frames[-1])


def metric_text(metrics: dict[str, float]) -> str:
    values: list[str] = []
    for key, label in (
        ("constraint_root2d_err", "root"),
        ("constraint_fullbody_keyframe", "full body"),
        ("constraint_end_effector", "end effector"),
    ):
        if key in metrics:
            values.append(f"{label} {100.0 * float(metrics[key]):.1f} cm")
    if "constraint_end_effector_rotation_deg" in metrics:
        values.append(f"rotation {float(metrics['constraint_end_effector_rotation_deg']):.1f} deg")
    if "controlled_contact_accuracy" in metrics:
        values.append(
            f"contact {100.0 * float(metrics['controlled_contact_accuracy']):.1f}%"
        )
    return "  |  ".join(values) if values else "No geometric control metric"


def load_case(case_dir: Path) -> dict[str, Any]:
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    arrays = {
        name: np.load(case_dir / f"{name}.npy")
        for name in (
            "generated_raw",
            "target",
            "mask",
            "root_metric_frames",
            "fullbody_metric_frames",
            "endpoint_position_metric_mask",
            "endpoint_rotation_metric_mask",
        )
    }
    length = int(metadata["length"])
    contact_mask_path = case_dir / "contact_metric_mask.npy"
    arrays["contact_metric_mask"] = (
        np.load(contact_mask_path)
        if contact_mask_path.is_file()
        else np.zeros((length, 4), dtype=bool)
    )
    for name in ("generated_raw", "target", "mask"):
        if arrays[name].shape != (length, 273):
            raise RuntimeError(f"Unexpected {name} shape in {case_dir}: {arrays[name].shape}")
    if arrays["contact_metric_mask"].shape != (length, 4):
        raise RuntimeError(
            f"Unexpected contact metric mask shape in {case_dir}: "
            f"{arrays['contact_metric_mask'].shape}"
        )
    return {"metadata": metadata, **arrays}


def decode_case(case: dict[str, Any]) -> dict[str, np.ndarray]:
    generated = torch.from_numpy(case["generated_raw"].astype(np.float32, copy=False))
    target = torch.from_numpy(case["target"].astype(np.float32, copy=False))
    with torch.no_grad():
        generated_joints = fk_positions_from_global_rot6d(generated).cpu().numpy()
        target_joints = fk_positions_from_global_rot6d(target).cpu().numpy()
        target_rotations = cont6d_to_matrix(split_global_rot6d(target)).cpu().numpy()
    return {
        "generated_joints": generated_joints.astype(np.float32, copy=False),
        "target_joints": target_joints.astype(np.float32, copy=False),
        "target_rotations": target_rotations.astype(np.float32, copy=False),
    }


def scene_extents(
    generated_joints: np.ndarray,
    target_joints: np.ndarray,
    target: np.ndarray,
    control_frames: np.ndarray,
    root_frames: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    parts = [generated_joints.reshape(-1, 3)]
    if control_frames.size:
        parts.append(target_joints[control_frames].reshape(-1, 3))
    if root_frames.size:
        root_path = np.stack(
            [target[root_frames, 0], target[root_frames, 1], target[root_frames, 2]],
            axis=-1,
        ).astype(np.float32)
        parts.append(root_path)
    points = np.concatenate(parts, axis=0)
    finite = points[np.isfinite(points).all(axis=-1)]
    if not finite.size:
        finite = np.asarray([[-2.0, 0.0, -2.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    x_min, z_min = finite[:, [0, 2]].min(axis=0)
    x_max, z_max = finite[:, [0, 2]].max(axis=0)
    span = max(float(x_max - x_min), float(z_max - z_min), 2.0)
    pad = max(0.65, 0.08 * span)
    bounds = (float(x_min - pad), float(x_max + pad), float(z_min - pad), float(z_max + pad))
    ground = np.asarray(
        [
            [bounds[0], 0.0, bounds[2]],
            [bounds[0], 0.0, bounds[3]],
            [bounds[1], 0.0, bounds[2]],
            [bounds[1], 0.0, bounds[3]],
        ],
        dtype=np.float32,
    )
    return np.concatenate([finite, ground], axis=0), bounds


def grid_step(bounds: tuple[float, float, float, float]) -> float:
    extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    if extent <= 4.0:
        return 0.5
    if extent <= 12.0:
        return 1.0
    return 2.0


def draw_ground_grid(
    draw: ImageDraw.ImageDraw,
    projector: WorldProjector,
    bounds: tuple[float, float, float, float],
) -> None:
    x_min, x_max, z_min, z_max = bounds
    step = grid_step(bounds)
    x_values = np.arange(math.floor(x_min / step) * step, x_max + step, step)
    z_values = np.arange(math.floor(z_min / step) * step, z_max + step, step)
    for value in x_values:
        major = abs(value / (step * 5.0) - round(value / (step * 5.0))) < 1e-5
        color = COLORS["grid_major"] if major else COLORS["grid"]
        points = np.asarray([[value, 0.0, z_min], [value, 0.0, z_max]], dtype=np.float32)
        draw.line(screen_points(projector, points), fill=color, width=2 if major else 1)
    for value in z_values:
        major = abs(value / (step * 5.0) - round(value / (step * 5.0))) < 1e-5
        color = COLORS["grid_major"] if major else COLORS["grid"]
        points = np.asarray([[x_min, 0.0, value], [x_max, 0.0, value]], dtype=np.float32)
        draw.line(screen_points(projector, points), fill=color, width=2 if major else 1)


def draw_header_and_footer(
    image: Image.Image,
    metadata: dict[str, Any],
    width: int,
    height: int,
) -> None:
    draw = ImageDraw.Draw(image)
    eyebrow_font = load_font(13, bold=True)
    title_font = load_font(24, bold=True)
    body_font = load_font(15)
    small_font = load_font(13)
    draw.text((22, 13), "KIMODO-LIKE CONTROL  /  RAW MODEL OUTPUT", fill=COLORS["muted"], font=eyebrow_font)
    label = SUBTYPE_LABELS.get(str(metadata["subtype"]), str(metadata["subtype"]))
    draw.text((22, 34), label, fill=COLORS["ink"], font=title_font)
    quantile = int(round(100.0 * float(metadata["target_quantile"])))
    badge = f"P{quantile:02d}"
    badge_x = min(22 + int(draw.textlength(label, font=title_font)) + 16, width - 90)
    draw.rounded_rectangle((badge_x, 37, badge_x + 54, 63), radius=4, fill=(232, 240, 252, 255))
    draw.text((badge_x + 11, 41), badge, fill=COLORS["blue"], font=eyebrow_font)
    caption = fit_text(draw, str(metadata["caption"]), body_font, width - 44)
    draw.text((22, 68), caption, fill=COLORS["muted"], font=body_font)

    footer_top = height - 70
    draw.line((22, footer_top, width - 22, footer_top), fill=(214, 220, 228, 255), width=1)
    metrics = metadata["regenerated_metrics"]
    draw.text((22, footer_top + 10), metric_text(metrics), fill=COLORS["ink"], font=body_font)
    skate = 100.0 * float(metrics.get("foot_skate_ratio", float("nan")))
    contact = 100.0 * float(metrics.get("foot_contact_consistency", float("nan")))
    skate_color = COLORS["danger"] if skate > 20.0 else COLORS["muted"]
    quality = f"foot skate {skate:.1f}%  |  contact consistency {contact:.1f}%"
    draw.text((22, footer_top + 36), quality, fill=skate_color, font=small_font)
    protocol = metadata["protocol"]
    checkpoint_label = str(metadata.get("checkpoint_label", "EMA 400K"))
    protocol_text = (
        f"{checkpoint_label}  /  ODE{int(protocol['num_steps'])}  /  "
        f"text CFG {float(protocol['cfg_scale_text']):g}  /  "
        f"control CFG {float(protocol['cfg_scale_control']):g}"
    )
    right = width - 22 - int(draw.textlength(protocol_text, font=small_font))
    draw.text((max(right, width // 2), footer_top + 36), protocol_text, fill=COLORS["muted"], font=small_font)


def draw_static_controls(
    image: Image.Image,
    projector: WorldProjector,
    case: dict[str, Any],
    decoded: dict[str, np.ndarray],
    ground_bounds: tuple[float, float, float, float],
) -> None:
    layer, draw = rgba_layer(image)
    draw_ground_grid(draw, projector, ground_bounds)
    target = case["target"]
    target_joints = decoded["target_joints"]
    mask = case["mask"].astype(bool, copy=False)
    root_frames = np.flatnonzero(case["root_metric_frames"])
    fullbody_frames = np.flatnonzero(case["fullbody_metric_frames"])
    endpoint_mask = case["endpoint_position_metric_mask"].astype(bool, copy=False)
    endpoint_frames = np.flatnonzero(endpoint_mask.any(axis=-1))

    if root_frames.size:
        root_points = np.stack(
            [target[root_frames, 0], np.zeros(root_frames.size), target[root_frames, 2]],
            axis=-1,
        )
        if root_frames.size > 2:
            draw.line(screen_points(projector, root_points), fill=COLORS["orange"], width=4)
        marker_frames = evenly_spaced(root_frames, 24)
        for frame in marker_frames:
            point = np.asarray([target[frame, 0], 0.0, target[frame, 2]], dtype=np.float32)
            draw_marker(draw, projector, point, color=COLORS["orange"], radius=4)

        heading_frames = root_frames[mask[root_frames, 3:5].any(axis=-1)]
        for frame in evenly_spaced(heading_frames, 12):
            heading = target[frame, 3:5]
            norm = max(float(np.linalg.norm(heading)), 1e-6)
            start = np.asarray([target[frame, 0], 0.04, target[frame, 2]], dtype=np.float32)
            end = start + np.asarray([heading[0], 0.0, heading[1]], dtype=np.float32) * (0.38 / norm)
            draw_arrow(draw, projector, start, end, color=COLORS["orange_faint"], width=2)

    for frame in fullbody_frames:
        draw_skeleton(
            draw,
            projector,
            target_joints[frame],
            color=COLORS["pose_faint"],
            width=2,
        )

    for frame in endpoint_frames:
        for joint in np.flatnonzero(endpoint_mask[frame]):
            draw_marker(
                draw,
                projector,
                target_joints[frame, joint],
                color=COLORS["target_faint"],
                radius=3,
            )
    composite(image, layer)


def draw_rotation_axes(
    draw: ImageDraw.ImageDraw,
    projector: WorldProjector,
    base: np.ndarray,
    rotation: np.ndarray,
    axis_length: float = 0.20,
) -> None:
    for axis, color in enumerate((COLORS["axis_x"], COLORS["axis_y"], COLORS["axis_z"])):
        end = base + rotation[:, axis] * axis_length
        draw_arrow(draw, projector, base, end, color=color, width=2)


def draw_dynamic_frame(
    base: Image.Image,
    projector: WorldProjector,
    case: dict[str, Any],
    decoded: dict[str, np.ndarray],
    source_frame: int,
    fps: int,
) -> Image.Image:
    image = base.copy()
    layer, draw = rgba_layer(image)
    generated_joints = decoded["generated_joints"]
    target_joints = decoded["target_joints"]
    target_rotations = decoded["target_rotations"]
    target = case["target"]
    mask = case["mask"].astype(bool, copy=False)
    root_frames = np.flatnonzero(case["root_metric_frames"])
    fullbody_frames = np.flatnonzero(case["fullbody_metric_frames"])
    endpoint_mask = case["endpoint_position_metric_mask"].astype(bool, copy=False)
    endpoint_rotation_mask = case["endpoint_rotation_metric_mask"].astype(bool, copy=False)
    endpoint_frames = np.flatnonzero(endpoint_mask.any(axis=-1))
    contact_mask = case["contact_metric_mask"].astype(bool, copy=False)
    contact_frames = np.flatnonzero(contact_mask.any(axis=-1))

    for offset, alpha in ((36, 35), (24, 50), (12, 72)):
        trail_frame = source_frame - offset
        if trail_frame >= 0:
            color = (*COLORS["trail"][:3], alpha)
            draw_skeleton(draw, projector, generated_joints[trail_frame], color=color, width=3)

    next_full = nearest_control_frame(fullbody_frames, source_frame)
    if next_full is not None:
        distance = abs(next_full - source_frame)
        alpha = 235 if distance <= 2 else 130
        draw_skeleton(
            draw,
            projector,
            target_joints[next_full],
            color=(*COLORS["pose"][:3], alpha),
            width=3,
            joint_radius=2,
        )

    next_endpoint = nearest_control_frame(endpoint_frames, source_frame)
    if next_endpoint is not None:
        for joint in np.flatnonzero(endpoint_mask[next_endpoint]):
            point = target_joints[next_endpoint, joint]
            draw_marker(draw, projector, point, color=COLORS["target"], radius=8, width=3)
            if endpoint_rotation_mask[next_endpoint, joint]:
                draw_rotation_axes(
                    draw,
                    projector,
                    point,
                    target_rotations[next_endpoint, joint],
                )

    next_root = nearest_control_frame(root_frames, source_frame)
    if next_root is not None:
        point = np.asarray([target[next_root, 0], 0.0, target[next_root, 2]], dtype=np.float32)
        draw_marker(draw, projector, point, color=COLORS["orange"], radius=7, width=3)
        if mask[next_root, 3:5].any():
            heading = target[next_root, 3:5]
            norm = max(float(np.linalg.norm(heading)), 1e-6)
            start = point + np.asarray([0.0, 0.05, 0.0], dtype=np.float32)
            end = start + np.asarray([heading[0], 0.0, heading[1]], dtype=np.float32) * (0.45 / norm)
            draw_arrow(draw, projector, start, end, color=COLORS["orange"], width=3)

    next_contact = nearest_control_frame(contact_frames, source_frame)
    if next_contact is not None:
        for contact_index in np.flatnonzero(contact_mask[next_contact]):
            joint = CONTACT_JOINTS[int(contact_index)]
            active = bool(target[next_contact, CONTACT_SLICE.start + int(contact_index)] > 0.5)
            draw_marker(
                draw,
                projector,
                target_joints[next_contact, joint],
                color=COLORS["green"] if active else COLORS["danger"],
                radius=10,
                width=4,
            )

    draw_skeleton(
        draw,
        projector,
        generated_joints[source_frame],
        color=COLORS["blue_shadow"],
        width=7,
    )
    draw_skeleton(
        draw,
        projector,
        generated_joints[source_frame],
        color=COLORS["blue"],
        width=4,
        joint_radius=3,
    )

    contacts = case["generated_raw"][source_frame, CONTACT_SLICE] > 0.5
    previous = max(source_frame - 1, 0)
    foot_speed = np.linalg.norm(
        generated_joints[source_frame, list(CONTACT_JOINTS)]
        - generated_joints[previous, list(CONTACT_JOINTS)],
        axis=-1,
    ) * float(fps)
    for contact_index, active in enumerate(contacts):
        if not active:
            continue
        joint = CONTACT_JOINTS[contact_index]
        color = COLORS["danger"] if foot_speed[contact_index] > 0.2 else COLORS["green"]
        draw_marker(
            draw,
            projector,
            generated_joints[source_frame, joint],
            color=color,
            radius=7,
            width=3,
        )
    composite(image, layer)

    text_draw = ImageDraw.Draw(image)
    frame_font = load_font(14, bold=True)
    frame_text = f"frame {source_frame + 1:03d}/{case['metadata']['length']:03d}  |  {source_frame / float(fps):.1f}s"
    x = image.width - 22 - int(text_draw.textlength(frame_text, font=frame_font))
    text_draw.text((x, 18), frame_text, fill=COLORS["ink"], font=frame_font)
    target_parts: list[str] = []
    for label, frame in (
        ("root", next_root),
        ("pose", next_full),
        ("endpoint", next_endpoint),
        ("contact", next_contact),
    ):
        if frame is not None:
            delta = int(frame - source_frame)
            target_parts.append(f"{label} {'now' if delta == 0 else f'{delta:+d}f'}")
    if target_parts:
        target_text = "next target: " + " / ".join(target_parts)
        x = image.width - 22 - int(text_draw.textlength(target_text, font=load_font(12)))
        text_draw.text((x, 42), target_text, fill=COLORS["muted"], font=load_font(12))

    legend_font = load_font(12)
    legend_y = image.height - 94
    text_draw.text(
        (22, legend_y),
        "blue generated  /  pale blue history  /  orange root  /  red endpoint  /  "
        "yellow target pose  /  green-red contact target",
        fill=COLORS["muted"],
        font=legend_font,
    )
    return image


def run_ffmpeg(command: Sequence[str], label: str) -> None:
    result = subprocess.run(
        list(command),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-30:])
        raise RuntimeError(f"ffmpeg failed for {label}:\n{tail}")


def encode_media(
    frame_pattern: str,
    frame_rate: float,
    media_dir: Path,
    *,
    make_mp4: bool,
    make_gif: bool,
    gif_fps: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for gallery rendering")
    if make_mp4:
        temporary = media_dir / ".control_raw.tmp.mp4"
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-framerate",
                f"{frame_rate:.8g}",
                "-i",
                frame_pattern,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            "MP4",
        )
        os.replace(temporary, media_dir / "control_raw.mp4")
    if make_gif:
        temporary = media_dir / ".control_raw.tmp.gif"
        filter_graph = (
            f"fps={max(gif_fps, 1)},scale=720:-2:flags=lanczos,"
            "split[s0][s1];[s0]palettegen=max_colors=160[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
        )
        run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-framerate",
                f"{frame_rate:.8g}",
                "-i",
                frame_pattern,
                "-vf",
                filter_graph,
                "-loop",
                "0",
                str(temporary),
            ],
            "GIF",
        )
        os.replace(temporary, media_dir / "control_raw.gif")


def render_case(task: RenderTask) -> dict[str, Any]:
    torch.set_num_threads(1)
    gallery_dir = Path(task.gallery_dir)
    case_dir = gallery_dir / task.case_dir
    media_dir = case_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = media_dir / "control_raw.mp4"
    gif_path = media_dir / "control_raw.gif"
    poster_path = media_dir / "poster.jpg"
    requested = [poster_path]
    if task.make_mp4:
        requested.append(mp4_path)
    if task.make_gif:
        requested.append(gif_path)

    case = load_case(case_dir)
    metadata = case["metadata"]
    if not task.overwrite and all(path.is_file() for path in requested):
        return render_record(gallery_dir, case_dir, metadata)

    decoded = decode_case(case)
    all_control_frames = np.unique(
        np.concatenate(
            [
                np.flatnonzero(case["root_metric_frames"]),
                np.flatnonzero(case["fullbody_metric_frames"]),
                np.flatnonzero(case["endpoint_position_metric_mask"].any(axis=-1)),
            ]
        )
    )
    projection_points, ground_bounds = scene_extents(
        decoded["generated_joints"],
        decoded["target_joints"],
        case["target"],
        all_control_frames,
        np.flatnonzero(case["root_metric_frames"]),
    )
    viewport = (24, 100, task.width - 24, task.height - 112)
    projector = WorldProjector(projection_points, viewport)
    base = Image.new("RGBA", (task.width, task.height), (249, 250, 252, 255))
    draw_header_and_footer(base, metadata, task.width, task.height)
    draw_static_controls(base, projector, case, decoded, ground_bounds)

    frame_indices = list(range(0, int(metadata["length"]), max(task.stride, 1)))
    if frame_indices[-1] != int(metadata["length"]) - 1:
        frame_indices.append(int(metadata["length"]) - 1)
    with tempfile.TemporaryDirectory(prefix="hy273_gallery_frames_") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for render_position, source_frame in enumerate(frame_indices):
            frame = draw_dynamic_frame(
                base,
                projector,
                case,
                decoded,
                source_frame,
                task.fps,
            )
            frame_path = temporary_root / f"frame_{render_position:05d}.png"
            frame.convert("RGB").save(frame_path, quality=95)
            if render_position == 0:
                frame.convert("RGB").save(poster_path, quality=90, optimize=True)
        encode_media(
            str(temporary_root / "frame_%05d.png"),
            task.fps / float(max(task.stride, 1)),
            media_dir,
            make_mp4=task.make_mp4,
            make_gif=task.make_gif,
            gif_fps=task.gif_fps,
        )
    return render_record(gallery_dir, case_dir, metadata)


def relative_media_path(gallery_dir: Path, path: Path) -> str:
    return path.relative_to(gallery_dir).as_posix() if path.is_file() else ""


def render_record(gallery_dir: Path, case_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    media_dir = case_dir / "media"
    metrics = metadata["regenerated_metrics"]
    return {
        "selection_index": int(metadata["selection_index"]),
        "case_key": str(metadata["case_key"]),
        "subtype": str(metadata["subtype"]),
        "subtype_label": SUBTYPE_LABELS.get(str(metadata["subtype"]), str(metadata["subtype"])),
        "family": str(metadata["family"]),
        "quantile": int(round(100.0 * float(metadata["target_quantile"]))),
        "composite_percentile": float(metadata["composite_percentile"]),
        "motion_id": str(metadata["motion_id"]),
        "caption": str(metadata["caption"]),
        "length": int(metadata["length"]),
        "metrics": metrics,
        "control_metric_text": metric_text(metrics),
        "foot_skate_percent": 100.0 * float(metrics.get("foot_skate_ratio", float("nan"))),
        "contact_consistency_percent": 100.0 * float(
            metrics.get("foot_contact_consistency", float("nan"))
        ),
        "mp4": relative_media_path(gallery_dir, media_dir / "control_raw.mp4"),
        "gif": relative_media_path(gallery_dir, media_dir / "control_raw.gif"),
        "poster": relative_media_path(gallery_dir, media_dir / "poster.jpg"),
        "metadata": (case_dir / "metadata.json").relative_to(gallery_dir).as_posix(),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_html(records: list[dict[str, Any]], gallery_dir: Path) -> None:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HY273 Kimodo-like Control Gallery</title>
<style>
:root { color-scheme: light; --ink:#18202c; --muted:#647084; --line:#d8dee7; --blue:#1f6feb; --red:#dc3737; --orange:#e8801e; --surface:#f6f8fa; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); background:#fff; font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }
header { border-bottom:1px solid var(--line); background:#fff; }
.header-inner,.toolbar,main { width:min(1500px,calc(100% - 32px)); margin:0 auto; }
.header-inner { padding:24px 0 18px; display:flex; align-items:end; justify-content:space-between; gap:24px; }
h1 { margin:0; font-size:26px; line-height:1.15; letter-spacing:0; }
.lede { margin:7px 0 0; color:var(--muted); max-width:800px; }
.summary { color:var(--muted); white-space:nowrap; }
.toolbar-wrap { position:sticky; top:0; z-index:20; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); }
.toolbar { padding:12px 0; display:grid; grid-template-columns:minmax(210px,1fr) 170px 260px auto; gap:10px; align-items:center; }
input,select,button { min-height:38px; border:1px solid #cbd3df; border-radius:4px; background:#fff; color:var(--ink); font:inherit; letter-spacing:0; }
input,select { width:100%; padding:0 11px; }
button { padding:0 13px; cursor:pointer; }
button:hover,button:focus-visible { border-color:var(--blue); outline:none; }
.segments { display:flex; }
.segments button { border-radius:0; border-right-width:0; }
.segments button:first-child { border-radius:4px 0 0 4px; }
.segments button:last-child { border-radius:0 4px 4px 0; border-right-width:1px; }
.segments button.active { color:#fff; background:var(--blue); border-color:var(--blue); }
main { padding:18px 0 42px; }
.grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
.card { min-width:0; border:1px solid var(--line); border-radius:6px; overflow:hidden; background:#fff; }
.media { aspect-ratio:3/2; background:#eef1f5; border-bottom:1px solid var(--line); }
.media video { width:100%; height:100%; display:block; object-fit:contain; background:#f9fafc; }
.body { padding:13px 14px 14px; }
.topline { display:flex; justify-content:space-between; align-items:start; gap:10px; }
.title { margin:0; font-size:16px; line-height:1.25; }
.badge { flex:none; padding:2px 7px; border-radius:3px; color:var(--blue); background:#e8f0fc; font-weight:700; }
.raw { margin:4px 0 0; color:var(--muted); font:12px ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }
.caption { min-height:42px; margin:10px 0; color:#3f4856; }
.metrics { padding-top:9px; border-top:1px solid #e7ebf0; color:#3f4856; }
.quality { margin-top:5px; color:var(--muted); }
.quality.bad { color:var(--red); font-weight:650; }
.actions { margin-top:11px; display:flex; gap:8px; align-items:center; }
.actions a { color:var(--blue); text-decoration:none; }
.actions a:hover { text-decoration:underline; }
.actions .inspect { margin-left:auto; min-height:32px; }
.empty { display:none; padding:80px 0; text-align:center; color:var(--muted); }
dialog { width:min(1100px,calc(100% - 32px)); border:1px solid var(--line); border-radius:6px; padding:0; box-shadow:0 20px 60px rgba(17,24,39,.22); }
dialog::backdrop { background:rgba(20,27,38,.62); }
.dialog-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; border-bottom:1px solid var(--line); }
.dialog-head h2 { margin:0; font-size:17px; }
.close { width:36px; padding:0; font-size:24px; line-height:1; }
.dialog-video { width:100%; display:block; max-height:76vh; background:#f6f8fa; }
@media (max-width:1050px) { .grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .toolbar { grid-template-columns:1fr 160px 1fr; } .segments { grid-column:1/-1; } }
@media (max-width:680px) { .header-inner { display:block; } .summary { margin-top:12px; } .toolbar { grid-template-columns:1fr; } .segments { grid-column:auto; } .grid { grid-template-columns:1fr; } h1 { font-size:22px; } }
</style>
</head>
<body>
<header><div class="header-inner"><div><h1>HY273 Kimodo-like control gallery</h1><p class="lede">Matched raw model outputs before terminal exact clamping. Target overlays are the actual constraints supplied during sampling.</p></div><div class="summary" id="summary"></div></div></header>
<div class="toolbar-wrap"><div class="toolbar">
  <input id="search" type="search" placeholder="Search caption, subtype, or motion ID" aria-label="Search gallery">
  <select id="family" aria-label="Control family"><option value="all">All families</option></select>
  <select id="subtype" aria-label="Control subtype"><option value="all">All 13 subtypes</option></select>
  <div class="segments" aria-label="Percentile filter"><button class="active" data-q="all">All</button><button data-q="25">P25</button><button data-q="50">P50</button><button data-q="75">P75</button></div>
</div></div>
<main><div class="grid" id="grid"></div><div class="empty" id="empty">No cases match the current filters.</div></main>
<dialog id="viewer"><div class="dialog-head"><h2 id="viewerTitle"></h2><button class="close" id="closeViewer" aria-label="Close">&times;</button></div><video class="dialog-video" id="viewerVideo" controls muted loop playsinline></video></dialog>
<script>
const CASES=__CASES_JSON__;
const grid=document.querySelector('#grid'); const empty=document.querySelector('#empty');
const family=document.querySelector('#family'); const subtype=document.querySelector('#subtype'); const search=document.querySelector('#search');
const viewer=document.querySelector('#viewer'); const viewerVideo=document.querySelector('#viewerVideo'); const viewerTitle=document.querySelector('#viewerTitle');
let quantile='all';
const esc=(value)=>String(value).replace(/[&<>'"]/g,(char)=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const families=[...new Set(CASES.map(x=>x.family))].sort(); families.forEach(value=>family.insertAdjacentHTML('beforeend',`<option value="${esc(value)}">${esc(value)}</option>`));
const subtypes=[...new Map(CASES.map(x=>[x.subtype,x.subtype_label])).entries()].sort((a,b)=>a[1].localeCompare(b[1])); subtypes.forEach(([value,label])=>subtype.insertAdjacentHTML('beforeend',`<option value="${esc(value)}">${esc(label)}</option>`));
function card(row){ const bad=row.foot_skate_percent>20; const media=row.mp4?`<video muted loop playsinline preload="metadata" poster="${esc(row.poster)}"><source src="${esc(row.mp4)}" type="video/mp4"></video>`:`<img src="${esc(row.gif)}" alt="${esc(row.subtype_label)}">`; return `<article class="card" data-index="${row.selection_index}"><div class="media">${media}</div><div class="body"><div class="topline"><div><h2 class="title">${esc(row.subtype_label)}</h2><div class="raw">${esc(row.subtype)} / ${esc(row.motion_id)}</div></div><span class="badge">P${row.quantile}</span></div><p class="caption">${esc(row.caption)}</p><div class="metrics">${esc(row.control_metric_text)}<div class="quality ${bad?'bad':''}">Foot skate ${row.foot_skate_percent.toFixed(1)}% / contact ${row.contact_consistency_percent.toFixed(1)}%</div></div><div class="actions">${row.gif?`<a href="${esc(row.gif)}">GIF</a>`:''}<a href="${esc(row.metadata)}">metadata</a>${row.mp4?`<button class="inspect" data-open="${row.selection_index}">Inspect</button>`:''}</div></div></article>`; }
function render(){ const needle=search.value.trim().toLowerCase(); const rows=CASES.filter(row=>(family.value==='all'||row.family===family.value)&&(subtype.value==='all'||row.subtype===subtype.value)&&(quantile==='all'||String(row.quantile)===quantile)&&(!needle||`${row.caption} ${row.subtype} ${row.subtype_label} ${row.motion_id}`.toLowerCase().includes(needle))); grid.innerHTML=rows.map(card).join(''); empty.style.display=rows.length?'none':'block'; document.querySelector('#summary').textContent=`${rows.length} / ${CASES.length} cases`; observeVideos(); }
let observer; function observeVideos(){ if(observer) observer.disconnect(); observer=new IntersectionObserver(entries=>entries.forEach(entry=>{ const video=entry.target; if(entry.isIntersecting) video.play().catch(()=>{}); else video.pause(); }),{rootMargin:'80px',threshold:.25}); document.querySelectorAll('.media video').forEach(video=>observer.observe(video)); }
[family,subtype,search].forEach(node=>node.addEventListener('input',render)); document.querySelectorAll('[data-q]').forEach(button=>button.addEventListener('click',()=>{ document.querySelectorAll('[data-q]').forEach(item=>item.classList.remove('active')); button.classList.add('active'); quantile=button.dataset.q; render(); }));
grid.addEventListener('click',event=>{ const button=event.target.closest('[data-open]'); if(!button) return; const row=CASES.find(item=>item.selection_index===Number(button.dataset.open)); viewerTitle.textContent=`${row.subtype_label} / P${row.quantile} / ${row.motion_id}`; viewerVideo.src=row.mp4; viewer.showModal(); viewerVideo.play().catch(()=>{}); });
function closeViewer(){ viewerVideo.pause(); viewerVideo.removeAttribute('src'); viewerVideo.load(); viewer.close(); } document.querySelector('#closeViewer').addEventListener('click',closeViewer); viewer.addEventListener('click',event=>{ if(event.target===viewer) closeViewer(); });
render();
</script>
</body>
</html>
'''
    (gallery_dir / "index.html").write_text(
        template.replace("__CASES_JSON__", payload), encoding="utf-8"
    )


def parse_indices(value: str) -> set[int] | None:
    if not value.strip():
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gallery_dir",
        default=(
            "/mnt/afs/mogeflow-control/generation/"
            "hy273_step400k_kimodo_gallery_39"
        ),
    )
    parser.add_argument("--case_indices", default="")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--gif_fps", type=int, default=10)
    parser.add_argument("--no_mp4", action="store_true")
    parser.add_argument("--no_gif", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.width < 640 or args.height < 480:
        raise ValueError("Gallery frames must be at least 640x480")
    if args.fps < 1 or args.stride < 1 or args.gif_fps < 1:
        raise ValueError("fps, stride, and gif_fps must be positive")
    gallery_dir = Path(args.gallery_dir).expanduser().resolve()
    manifest_path = gallery_dir / "gallery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("cases")
    if (
        not isinstance(rows, list)
        or not rows
        or int(manifest.get("num_cases", -1)) != len(rows)
    ):
        raise RuntimeError("Gallery manifest has an invalid case list")
    selected_indices = parse_indices(args.case_indices)
    if selected_indices is not None:
        rows = [row for row in rows if int(row["selection_index"]) in selected_indices]
        missing = selected_indices - {int(row["selection_index"]) for row in rows}
        if missing:
            raise ValueError(f"Unknown case indices: {sorted(missing)}")
    tasks = [
        RenderTask(
            gallery_dir=str(gallery_dir),
            case_dir=str(row["case_dir"]),
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            stride=int(args.stride),
            gif_fps=int(args.gif_fps),
            make_mp4=not args.no_mp4,
            make_gif=not args.no_gif,
            overwrite=bool(args.overwrite),
        )
        for row in rows
    ]
    records: list[dict[str, Any]] = []
    worker_count = max(1, min(int(args.workers), len(tasks)))
    if worker_count == 1:
        for task in tasks:
            record = render_case(task)
            records.append(record)
            print(f"RENDER_DONE {record['selection_index']:02d} {record['subtype']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(render_case, task): task for task in tasks}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(f"RENDER_DONE {record['selection_index']:02d} {record['subtype']}", flush=True)
    records.sort(key=lambda row: int(row["selection_index"]))
    render_manifest = {
        "format": "hy273_kimodo_render_gallery_v1",
        "source_manifest": "gallery_manifest.json",
        "num_cases": len(records),
        "render": {
            "width": int(args.width),
            "height": int(args.height),
            "source_fps": int(args.fps),
            "stride": int(args.stride),
            "video_fps": float(args.fps) / float(args.stride),
            "gif_fps": int(args.gif_fps),
            "primary_output": "generated_raw_pre_terminal_clamp",
        },
        "cases": records,
    }
    atomic_json(gallery_dir / "render_manifest.json", render_manifest)
    build_html(records, gallery_dir)
    print(
        json.dumps(
            {
                "gallery": str(gallery_dir),
                "cases": len(records),
                "manifest": str(gallery_dir / "render_manifest.json"),
                "html": str(gallery_dir / "index.html"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
