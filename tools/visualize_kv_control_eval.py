"""Visualize sparse joint-control eval samples for KV-Control CodeFlow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_codeflow_kv_control import (  # noqa: E402
    CONTROL_PROTOCOL,
    decode_motion_from_embeddings,
    load_model,
    make_device,
    sample_controlled_embeddings,
)
from models.codeflow.eval_motionstreamer272_t2m import (  # noqa: E402
    MotionStreamer272T2MEvalDataset,
    collate_motionstreamer272_t2m_eval,
)
from models.codeflow.kv_control import sample_joint_position_control  # noqa: E402
from models.codeflow.motionstreamer272 import recover_motionstreamer272_positions_from_normalized  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402
from utils.paramUtil import t2m_kinematic_chain  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="dataset/HumanML3D_272")
    parser.add_argument("--mean_path", type=str, default="dataset/HumanML3D_272/Mean.npy")
    parser.add_argument("--std_path", type=str, default="dataset/HumanML3D_272/Std.npy")
    parser.add_argument("--vq_checkpoint", type=str, default="")
    parser.add_argument("--vq_partition", type=str, default="")
    parser.add_argument("--clip_path", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_indices", type=str, default="0,1,2")
    parser.add_argument("--random_samples", action="store_true")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cond_scale", type=float, default=3.0)
    parser.add_argument("--decode_mode", type=str, default="continuous", choices=["continuous", "nearest", "ids"])
    parser.add_argument("--terminal_mode", type=str, default="")
    parser.add_argument("--min_keyframes", type=int, default=1)
    parser.add_argument("--max_keyframes", type=int, default=8)
    parser.add_argument("--min_joints", type=int, default=1)
    parser.add_argument("--max_joints", type=int, default=6)
    parser.add_argument(
        "--control_profile",
        type=str,
        default="semantic_mix",
        choices=[
            "random_joints",
            "root",
            "endpoints5",
            "root_endpoints6",
            "full_pose",
            "semantic_random_subset",
            "endpoints_random_subset",
            "root_endpoints_random_subset",
            "semantic_mix",
        ],
    )
    parser.add_argument(
        "--control_keyframe_strategy",
        type=str,
        default="mixed",
        choices=["random", "uniform", "endpoints", "mixed"],
    )
    parser.add_argument("--control_dropout_prob", type=float, default=0.0)
    parser.add_argument("--guidance_mode", type=str, default="none", choices=["none", "gradient"])
    parser.add_argument("--guidance_eta", type=float, default=0.05)
    parser.add_argument("--guidance_variable", type=str, default="clean", choices=["clean", "state"])
    parser.add_argument("--guidance_optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--guidance_adam_beta1", type=float, default=0.5)
    parser.add_argument("--guidance_adam_beta2", type=float, default=0.9)
    parser.add_argument("--guidance_weight_decay", type=float, default=1e-6)
    parser.add_argument("--guidance_anchor", type=float, default=0.0)
    parser.add_argument("--guidance_recompute_cond", type=str, default="step", choices=["step", "inner"])
    parser.add_argument("--guidance_start", type=float, default=0.5)
    parser.add_argument("--guidance_end", type=float, default=1.0)
    parser.add_argument("--guidance_inner_iters", type=int, default=1)
    parser.add_argument("--guidance_total_iters", type=int, default=0)
    parser.add_argument(
        "--guidance_iter_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear_increase", "linear_decrease"],
    )
    parser.add_argument(
        "--guidance_eta_schedule",
        type=str,
        default="linear_decay",
        choices=["constant", "linear_decay"],
    )
    parser.add_argument("--guidance_grad_clip", type=float, default=1.0)
    parser.add_argument("--guidance_loss", type=str, default="l2", choices=["l1", "l2", "dist"])
    parser.add_argument("--guidance_joint_anchor", type=float, default=0.0)
    parser.add_argument("--guidance_foot_skate_weight", type=float, default=0.0)
    parser.add_argument("--guidance_floor_weight", type=float, default=0.0)
    parser.add_argument("--guidance_smooth_weight", type=float, default=0.0)
    parser.add_argument("--guidance_foot_height", type=float, default=0.08)
    parser.add_argument("--guidance_foot_temp", type=float, default=0.02)
    parser.add_argument("--max_render_frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--no_ema", action="store_true")
    return parser.parse_args()


def _as_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(args.checkpoint),
        output_dir=str(args.output_dir),
        eval_dir="",
        data_root=str(args.data_root),
        mean_path=str(args.mean_path),
        std_path=str(args.std_path),
        vq_checkpoint=str(args.vq_checkpoint),
        vq_partition=str(args.vq_partition),
        clip_path=str(args.clip_path),
        gpu_id=int(args.gpu_id),
        device=str(args.device),
        batch_size=int(args.num_samples),
        num_workers=0,
        steps=int(args.steps),
        cond_scale=float(args.cond_scale),
        repeat_times=1,
        seed=int(args.seed),
        max_samples=0,
        decode_mode=str(args.decode_mode),
        terminal_mode=str(args.terminal_mode),
        no_ema=bool(args.no_ema),
        min_keyframes=int(args.min_keyframes),
        max_keyframes=int(args.max_keyframes),
        min_joints=int(args.min_joints),
        max_joints=int(args.max_joints),
        control_profile=str(args.control_profile),
        control_keyframe_strategy=str(args.control_keyframe_strategy),
        control_dropout_prob=float(args.control_dropout_prob),
        guidance_mode=str(args.guidance_mode),
        guidance_eta=float(args.guidance_eta),
        guidance_variable=str(args.guidance_variable),
        guidance_optimizer=str(args.guidance_optimizer),
        guidance_adam_beta1=float(args.guidance_adam_beta1),
        guidance_adam_beta2=float(args.guidance_adam_beta2),
        guidance_weight_decay=float(args.guidance_weight_decay),
        guidance_anchor=float(args.guidance_anchor),
        guidance_recompute_cond=str(args.guidance_recompute_cond),
        guidance_start=float(args.guidance_start),
        guidance_end=float(args.guidance_end),
        guidance_inner_iters=int(args.guidance_inner_iters),
        guidance_total_iters=int(args.guidance_total_iters),
        guidance_iter_schedule=str(args.guidance_iter_schedule),
        guidance_eta_schedule=str(args.guidance_eta_schedule),
        guidance_grad_clip=float(args.guidance_grad_clip),
        guidance_loss=str(args.guidance_loss),
        guidance_joint_anchor=float(args.guidance_joint_anchor),
        guidance_foot_skate_weight=float(args.guidance_foot_skate_weight),
        guidance_floor_weight=float(args.guidance_floor_weight),
        guidance_smooth_weight=float(args.guidance_smooth_weight),
        guidance_foot_height=float(args.guidance_foot_height),
        guidance_foot_temp=float(args.guidance_foot_temp),
        post_guidance_iters=0,
        post_guidance_lr=0.05,
        post_guidance_loss="dist",
        post_guidance_anchor=0.0,
        post_guidance_grad_clip=0.0,
        save_json_name="",
    )


def _sample_indices(args: argparse.Namespace, dataset_len: int) -> List[int]:
    if bool(args.random_samples):
        rng = np.random.default_rng(int(args.seed))
        count = min(int(args.num_samples), int(dataset_len))
        return sorted(rng.choice(dataset_len, size=count, replace=False).astype(int).tolist())
    return [int(part) for part in args.sample_indices.replace(" ", "").split(",") if part]


def _project(points: np.ndarray, angle_degrees: float = 35.0) -> np.ndarray:
    theta = math.radians(angle_degrees)
    x = points[..., 0] * math.cos(theta) - points[..., 2] * math.sin(theta)
    y = points[..., 1]
    return np.stack([x, y], axis=-1)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _shorten(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."


def _panel_transform(
    bounds: Tuple[float, float, float, float],
    box: Tuple[int, int, int, int],
):
    umin, umax, vmin, vmax = bounds
    left, top, right, bottom = box
    width = max(1, right - left)
    height = max(1, bottom - top)
    span_u = max(umax - umin, 1e-5)
    span_v = max(vmax - vmin, 1e-5)
    scale = min((width - 24) / span_u, (height - 24) / span_v)
    used_w = span_u * scale
    used_h = span_v * scale
    x0 = left + (width - used_w) * 0.5
    y0 = top + (height - used_h) * 0.5

    def map_point(point: np.ndarray) -> Tuple[int, int]:
        x = x0 + (float(point[0]) - umin) * scale
        y = y0 + (vmax - float(point[1])) * scale
        return int(round(x)), int(round(y))

    return map_point


def _draw_skeleton(
    draw: ImageDraw.ImageDraw,
    points_2d: np.ndarray,
    *,
    box: Tuple[int, int, int, int],
    bounds: Tuple[float, float, float, float],
    color: Tuple[int, int, int],
) -> None:
    if not np.isfinite(points_2d).all():
        return
    map_point = _panel_transform(bounds, box)
    shadow = (220, 220, 220)
    for chain in t2m_kinematic_chain:
        pts = [map_point(points_2d[j]) for j in chain if j < points_2d.shape[0]]
        if len(pts) >= 2:
            draw.line(pts, fill=shadow, width=5, joint="curve")
            draw.line(pts, fill=color, width=3, joint="curve")
    for joint_idx in range(points_2d.shape[0]):
        x, y = map_point(points_2d[joint_idx])
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def _draw_control_targets(
    draw: ImageDraw.ImageDraw,
    target_points_2d: np.ndarray,
    mask: np.ndarray,
    frame_idx: int,
    *,
    box: Tuple[int, int, int, int],
    bounds: Tuple[float, float, float, float],
) -> None:
    if frame_idx >= mask.shape[0]:
        return
    joints = np.where(mask[frame_idx].any(axis=-1))[0]
    if joints.size == 0:
        return
    map_point = _panel_transform(bounds, box)
    for joint_idx in joints.tolist():
        x, y = map_point(target_points_2d[frame_idx, joint_idx])
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(255, 196, 0), width=3)


def _root_center(joints: np.ndarray) -> np.ndarray:
    centered = joints.copy()
    centered[..., 0] -= centered[:, None, 0, 0]
    centered[..., 2] -= centered[:, None, 0, 2]
    min_y = float(centered[..., 1].min())
    centered[..., 1] -= min_y
    return centered


def _mean_kps_cm(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    common = min(pred.shape[0], target.shape[0], mask.shape[0])
    pred = pred[:common]
    target = target[:common]
    mask = mask[:common].astype(bool)
    joint_mask = mask.any(axis=-1)
    if not joint_mask.any():
        return float("nan")
    return float(np.linalg.norm(pred - target, axis=-1)[joint_mask].mean() * 100.0)


def render_gif(record: Dict[str, object], out_path: Path, fps: float, max_render_frames: int) -> None:
    labels = (
        ("GT", "gt", (29, 143, 86)),
        ("NO CONTROL", "no_control", (80, 87, 96)),
        ("CONTROL", "control", (33, 111, 219)),
    )
    length = int(record["length"])
    if int(max_render_frames) > 0:
        length = min(length, int(max_render_frames))

    joints = {
        "gt": _root_center(np.asarray(record["gt_joints"], dtype=np.float32)[:length]),
        "no_control": _root_center(np.asarray(record["no_control_joints"], dtype=np.float32)[:length]),
        "control": _root_center(np.asarray(record["control_joints"], dtype=np.float32)[:length]),
    }
    mask = np.asarray(record["target_mask"], dtype=bool)[:length]
    projected = {key: _project(value) for key, value in joints.items()}
    all_proj = np.concatenate([value.reshape(-1, 2) for value in projected.values()], axis=0)
    umin, vmin = np.nanmin(all_proj, axis=0)
    umax, vmax = np.nanmax(all_proj, axis=0)
    pad_u = max((umax - umin) * 0.08, 0.25)
    pad_v = max((vmax - vmin) * 0.12, 0.25)
    bounds = (float(umin - pad_u), float(umax + pad_u), float(vmin - pad_v), float(vmax + pad_v))

    panel_w, panel_h = 330, 360
    header_h, footer_h = 82, 44
    width = panel_w * len(labels)
    height = header_h + panel_h + footer_h
    title_font = _font(16)
    small_font = _font(13)
    panel_font = _font(15)
    duration = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    frames: List[Image.Image] = []
    caption = _shorten(str(record["caption"]), 160)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(length):
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((14, 10), f"KV-control sample {record['sample_index']} | frame {frame_idx + 1}/{length}", fill=(20, 24, 28), font=title_font)
        draw.text((14, 38), caption, fill=(68, 74, 82), font=small_font)
        for col, (label, key, color) in enumerate(labels):
            left = col * panel_w
            box = (left + 8, header_h, left + panel_w - 8, header_h + panel_h)
            draw.text((left + 14, header_h - 28), label, fill=color, font=panel_font)
            draw.rectangle(box, outline=(224, 228, 232), width=1)
            _draw_skeleton(draw, projected[key][frame_idx], box=box, bounds=bounds, color=color)
            _draw_control_targets(draw, projected["gt"], mask, frame_idx, box=box, bounds=bounds)
        draw.text(
            (14, header_h + panel_h + 12),
            "yellow rings mark controlled target joints on controlled frames; root-centered render",
            fill=(92, 99, 107),
            font=small_font,
        )
        frames.append(image)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2, optimize=False)


def main() -> None:
    args = parse_args()
    fixseed(int(args.seed))
    device = make_device(args)
    eval_args = _as_eval_args(args)
    model, opt, ckpt, weight_source = load_model(Path(args.checkpoint), eval_args, device)
    dataset = MotionStreamer272T2MEvalDataset(
        args.data_root or str(getattr(opt, "data_root", "")),
        args.split,
        unit_length=int(getattr(opt, "unit_length", 4)),
        max_motion_length=int(getattr(opt, "motion_length", 300)),
        max_samples=0,
    )
    sample_indices = _sample_indices(args, len(dataset))
    batch = collate_motionstreamer272_t2m_eval([dataset[idx] for idx in sample_indices])
    raw_motion = batch["motion"].to(device=model.device, dtype=torch.float32)
    lengths = batch["length"].to(device=model.device, dtype=torch.long)
    mean = torch.from_numpy(np.load(args.mean_path).astype(np.float32)).to(model.device)
    std = torch.from_numpy(np.load(args.std_path).astype(np.float32)).to(model.device)
    target_norm = (raw_motion - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    token_lengths = (lengths // int(getattr(opt, "unit_length", 4))).clamp(min=1)
    control_batch = sample_joint_position_control(
        target_norm,
        lengths,
        mean=mean,
        std=std,
        profile=str(args.control_profile),
        min_keyframes=int(args.min_keyframes),
        max_keyframes=int(args.max_keyframes),
        min_joints=int(args.min_joints),
        max_joints=int(args.max_joints),
        keyframe_strategy=str(args.control_keyframe_strategy),
        dropout_prob=float(args.control_dropout_prob),
    )
    target_joints = control_batch["target_joints"]
    target_mask = control_batch["target_mask"]
    latent_len = int(token_lengths.max().item())
    init_noise = torch.randn(
        raw_motion.shape[0],
        latent_len,
        int(model.config.num_parts),
        int(model.config.code_dim),
        device=model.device,
    ) * float(model.config.noise_scale)
    with torch.no_grad():
        no_control_embeddings = sample_controlled_embeddings(
            model,
            batch["caption"],
            token_lengths,
            target_joints,
            target_mask,
            mean,
            std,
            steps=int(args.steps),
            cond_scale=float(args.cond_scale),
            init_noise=init_noise,
            enable_control=False,
            args=eval_args,
        )
    if str(args.guidance_mode) == "gradient":
        with torch.enable_grad():
            control_embeddings = sample_controlled_embeddings(
                model,
                batch["caption"],
                token_lengths,
                target_joints,
                target_mask,
                mean,
                std,
                steps=int(args.steps),
                cond_scale=float(args.cond_scale),
                init_noise=init_noise,
                enable_control=True,
                args=eval_args,
            )
    else:
        with torch.no_grad():
            control_embeddings = sample_controlled_embeddings(
                model,
                batch["caption"],
                token_lengths,
                target_joints,
                target_mask,
                mean,
                std,
                steps=int(args.steps),
                cond_scale=float(args.cond_scale),
                init_noise=init_noise,
                enable_control=True,
                args=eval_args,
            )
    terminal_mode = args.terminal_mode or None
    no_control_norm = decode_motion_from_embeddings(model, no_control_embeddings, terminal_mode=terminal_mode, decode_mode=str(args.decode_mode))
    control_norm = decode_motion_from_embeddings(model, control_embeddings, terminal_mode=terminal_mode, decode_mode=str(args.decode_mode))
    no_control_joints = recover_motionstreamer272_positions_from_normalized(no_control_norm, mean, std)
    control_joints = recover_motionstreamer272_positions_from_normalized(control_norm, mean, std)

    out_dir = Path(args.out_dir).expanduser().resolve()
    gif_dir = out_dir / "gifs"
    cache_dir = out_dir / "caches"
    records: List[Dict[str, object]] = []
    for local_idx, sample_idx in enumerate(sample_indices):
        length = int(lengths[local_idx].item())
        record = {
            "sample_index": int(sample_idx),
            "sample_id": str(batch["name"][local_idx]),
            "caption": str(batch["caption"][local_idx]),
            "length": int(length),
            "gt_joints": target_joints[local_idx, :length].detach().cpu().numpy(),
            "no_control_joints": no_control_joints[local_idx, :length].detach().cpu().numpy(),
            "control_joints": control_joints[local_idx, :length].detach().cpu().numpy(),
            "target_mask": target_mask[local_idx, :length].detach().cpu().numpy(),
        }
        control_kps = _mean_kps_cm(record["control_joints"], record["gt_joints"], record["target_mask"])
        no_control_kps = _mean_kps_cm(record["no_control_joints"], record["gt_joints"], record["target_mask"])
        cache_path = cache_dir / f"sample{sample_idx:04d}_kv_control.npz"
        gif_path = gif_dir / f"sample{sample_idx:04d}_gt_nocontrol_control.gif"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **record)
        render_gif(record, gif_path, float(args.fps), int(args.max_render_frames))
        row = {
            "sample_index": int(sample_idx),
            "sample_id": record["sample_id"],
            "caption": record["caption"],
            "length": int(length),
            "control_kps_cm": float(control_kps),
            "no_control_kps_cm": float(no_control_kps),
            "improvement_cm": float(no_control_kps - control_kps),
            "gif": str(gif_path),
            "cache": str(cache_path),
        }
        records.append(row)
        print(
            f"VIS_KV sample={sample_idx} control_kps={control_kps:.3f}cm "
            f"no_control_kps={no_control_kps:.3f}cm gif={gif_path}",
            flush=True,
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": int(ckpt.get("epoch", 0)),
        "step": int(ckpt.get("step", 0)),
        "weight_source": str(weight_source),
        "control_protocol": CONTROL_PROTOCOL,
        "stage": "B_plus_C_visual_guidance" if str(args.guidance_mode) == "gradient" else "B_kv_adapter_visual_gate",
        "config": {
            "steps": int(args.steps),
            "cond_scale": float(args.cond_scale),
            "decode_mode": str(args.decode_mode),
            "seed": int(args.seed),
            "sample_indices": sample_indices,
            "control_profile": str(args.control_profile),
            "control_keyframe_strategy": str(args.control_keyframe_strategy),
            "guidance_mode": str(args.guidance_mode),
            "guidance_eta": float(args.guidance_eta),
            "guidance_variable": str(args.guidance_variable),
            "guidance_optimizer": str(args.guidance_optimizer),
            "guidance_total_iters": int(args.guidance_total_iters),
            "guidance_iter_schedule": str(args.guidance_iter_schedule),
            "guidance_eta_schedule": str(args.guidance_eta_schedule),
            "guidance_loss": str(args.guidance_loss),
            "guidance_joint_anchor": float(args.guidance_joint_anchor),
            "guidance_foot_skate_weight": float(args.guidance_foot_skate_weight),
            "guidance_floor_weight": float(args.guidance_floor_weight),
            "guidance_smooth_weight": float(args.guidance_smooth_weight),
            "guidance_foot_height": float(args.guidance_foot_height),
            "guidance_foot_temp": float(args.guidance_foot_temp),
        },
        "records": records,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VIS_KV_DONE out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
