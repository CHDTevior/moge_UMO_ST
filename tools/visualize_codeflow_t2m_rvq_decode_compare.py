"""Render T2M RVQ reconstruction vs nearest/continuous CodeFlow decode GIFs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_codeflow_t2m import load_codeflow_model, make_device
from models.codeflow import PartStructuredMotionCodeFlow
from models.codeflow.eval_motionstreamer272_t2m import (
    MotionStreamer272T2MEvalDataset,
    collate_motionstreamer272_t2m_eval,
)
from tools.visualize_codeflow_umo_edit_gifs import (
    decoded_vq_to_raw,
    draw_skeleton,
    load_recover_fn,
    pad_motion,
    project_points,
    safe_font,
    shorten,
    tensor_norm,
    zero_pad_motion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--random_samples", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="dataset/HumanML3D_272")
    parser.add_argument("--hymotion_root", type=Path, default=Path("/mnt/afs/HY-Motion-1.0"))
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--cond_scale", type=float, default=6.0)
    parser.add_argument("--unit_length", type=int, default=0)
    parser.add_argument("--max_eval_frames", type=int, default=300)
    parser.add_argument("--max_render_frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--kv_root", type=str, default="")
    parser.add_argument("--vq_checkpoint", type=str, default="")
    parser.add_argument("--vq_partition", type=str, default="")
    parser.add_argument("--mean_path", type=str, default="")
    parser.add_argument("--std_path", type=str, default="")
    parser.add_argument("--clip_path", type=str, default="")
    return parser.parse_args()


def recover_positions(record: Dict[str, object], recover_fn) -> Dict[str, np.ndarray]:
    length = int(record["length"])
    out: Dict[str, np.ndarray] = {}
    for key in ("rvq_272", "nearest_272", "continuous_272", "gt_272"):
        motion = np.asarray(record[key], dtype=np.float32)[:length]
        pos = recover_fn(motion)
        pos = pos - pos[:, [0]]
        out[key] = pos
    return out


def render_four_panel_gif(record: Dict[str, object], out_path: Path, recover_fn, fps: float, max_render_frames: int) -> None:
    length = int(record["length"])
    if max_render_frames > 0:
        length = min(length, int(max_render_frames))
    positions = recover_positions(record, recover_fn)
    for key in positions:
        positions[key] = positions[key][:length]

    all_proj = np.concatenate([project_points(pos) for pos in positions.values()], axis=0)
    umin, vmin = np.nanmin(all_proj.reshape(-1, 2), axis=0)
    umax, vmax = np.nanmax(all_proj.reshape(-1, 2), axis=0)
    pad_u = max((umax - umin) * 0.08, 0.25)
    pad_v = max((vmax - vmin) * 0.12, 0.25)
    bounds = (float(umin - pad_u), float(umax + pad_u), float(vmin - pad_v), float(vmax + pad_v))

    labels = (
        ("RVQ", "rvq_272", (125, 88, 35)),
        ("GEN nearest", "nearest_272", (33, 111, 219)),
        ("GEN continuous", "continuous_272", (114, 70, 170)),
        ("GT", "gt_272", (29, 143, 86)),
    )
    panel_w, panel_h = 320, 350
    header_h, footer_h = 78, 44
    width = panel_w * len(labels)
    height = header_h + panel_h + footer_h
    font = safe_font(17)
    small_font = safe_font(13)
    title_font = safe_font(15)
    task = str(record["task"])
    sample_id = str(record["sample_id"])
    text = shorten(str(record["text"]), 170)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1, int(round(1000.0 / max(float(fps), 1e-6))))
    frames: List[Image.Image] = []
    for frame_idx in range(length):
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        header = f"{task} | sample {sample_id} | frame {frame_idx + 1}/{length}"
        draw.text((14, 10), header, fill=(20, 24, 28), font=font)
        draw.text((14, 38), text, fill=(68, 74, 82), font=small_font)
        for col, (label, key, color) in enumerate(labels):
            x0 = col * panel_w
            draw.text((x0 + 14, header_h - 28), label, fill=color, font=title_font)
            draw.rectangle((x0 + 8, header_h, x0 + panel_w - 8, header_h + panel_h), outline=(224, 228, 232), width=1)
            box = (x0 + 8, header_h, x0 + panel_w - 8, header_h + panel_h)
            draw_skeleton(draw, project_points(positions[key][frame_idx]), box=box, bounds=bounds, color=color)
        draw.text(
            (14, header_h + panel_h + 12),
            "root-centered render; 272D recovered with HY-Motion MotionStreamer representation",
            fill=(92, 99, 107),
            font=small_font,
        )
        frames.append(image)

    if not frames:
        raise RuntimeError(f"No frames rendered for {out_path}")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration, loop=0, disposal=2, optimize=False)


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = make_device(args)
    recover_fn = load_recover_fn(args.hymotion_root)
    model, opt, ckpt, weight_source = load_codeflow_model(
        args.checkpoint,
        args,
        device,
        model_cls=PartStructuredMotionCodeFlow,
    )
    if int(getattr(opt, "motion_dim", 0) or 0) != 272:
        raise ValueError(f"Expected a 272D checkpoint, got motion_dim={getattr(opt, 'motion_dim', None)}")

    unit_length = int(getattr(opt, "unit_length", 4))
    load_full_split = bool(args.random_samples or args.sample_indices.strip())
    dataset = MotionStreamer272T2MEvalDataset(
        args.data_root or str(getattr(opt, "data_root", "")),
        args.split,
        unit_length=unit_length,
        max_motion_length=int(args.max_eval_frames),
        max_samples=0 if load_full_split else int(args.sample_index) + int(args.num_samples),
    )
    if args.sample_indices.strip():
        sample_indices = [int(part) for part in args.sample_indices.replace(" ", "").split(",") if part]
    elif args.random_samples:
        rng = np.random.default_rng(int(args.seed))
        sample_count = min(int(args.num_samples), len(dataset))
        sample_indices = sorted(rng.choice(len(dataset), size=sample_count, replace=False).astype(int).tolist())
    else:
        sample_indices = [int(args.sample_index) + offset for offset in range(int(args.num_samples))]
    if not sample_indices:
        raise ValueError("No samples selected for visualization")
    for sample_idx in sample_indices:
        if sample_idx < 0 or sample_idx >= len(dataset):
            raise IndexError(f"sample_index {sample_idx} >= dataset size {len(dataset)}")

    vq_mean_np = np.load(opt.mean_path).astype(np.float32)
    vq_std_np = np.load(opt.std_path).astype(np.float32)
    vq_mean = tensor_norm(vq_mean_np, device)
    vq_std = tensor_norm(vq_std_np, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gif_dir = args.out_dir / "gifs"
    cache_dir = args.out_dir / "caches"
    records: List[Dict[str, object]] = []
    ckpt_epoch = int(ckpt.get("epoch", 0) or 0)
    ckpt_step = int(ckpt.get("step", 0) or 0)

    with torch.no_grad():
        for sample_idx in sample_indices:
            batch = collate_motionstreamer272_t2m_eval([dataset[sample_idx]])
            raw_motion = batch["motion"].to(device=device, dtype=torch.float32)
            lengths = batch["length"].to(device=device, dtype=torch.long)
            captions = list(batch["caption"])
            names = list(batch["name"])
            length = int(min(lengths[0].item(), int(args.max_eval_frames)))
            token_lengths = (lengths // unit_length).clamp(min=1)
            gt_vq_motion = zero_pad_motion((raw_motion - vq_mean) / vq_std, lengths)

            ids, _embeddings = model.tokenizer.encode(gt_vq_motion)
            rvq_norm = model.tokenizer.decode_ids(ids)
            rvq_raw = decoded_vq_to_raw(rvq_norm, raw_motion, vq_mean, vq_std)

            nearest_norm, _nearest_ids = model.generate_motion(
                captions,
                token_lengths=token_lengths,
                steps=int(args.steps),
                cond_scale=float(args.cond_scale),
                terminal_mode=None,
                decode_mode="nearest",
            )
            continuous_norm, _continuous_ids = model.generate_motion(
                captions,
                token_lengths=token_lengths,
                steps=int(args.steps),
                cond_scale=float(args.cond_scale),
                terminal_mode=None,
                decode_mode="continuous",
            )
            nearest_raw = nearest_norm.to(torch.float32) * vq_std + vq_mean
            continuous_raw = continuous_norm.to(torch.float32) * vq_std + vq_mean

            record = {
                "task": "t2m_rvq_decode_compare",
                "sample_index": sample_idx,
                "sample_id": str(names[0]),
                "text": str(captions[0]),
                "length": length,
                "rvq_272": pad_motion(rvq_raw[0, :length].detach().cpu().numpy(), int(args.max_eval_frames)),
                "nearest_272": pad_motion(nearest_raw[0, :length].detach().cpu().numpy(), int(args.max_eval_frames)),
                "continuous_272": pad_motion(continuous_raw[0, :length].detach().cpu().numpy(), int(args.max_eval_frames)),
                "gt_272": pad_motion(raw_motion[0, :length].detach().cpu().numpy(), int(args.max_eval_frames)),
            }
            cache_path = cache_dir / f"sample{sample_idx:03d}_compare.npz"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                **{
                    k: np.asarray(v, dtype=np.float32) if k.endswith("_272") else v
                    for k, v in record.items()
                },
                checkpoint=str(args.checkpoint),
                checkpoint_epoch=ckpt_epoch,
                checkpoint_step=ckpt_step,
                weight_source=str(weight_source),
            )
            gif_path = gif_dir / f"sample{sample_idx:03d}_rvq_nearest_continuous_gt.gif"
            render_four_panel_gif(record, gif_path, recover_fn, float(args.fps), int(args.max_render_frames))
            row = {
                "sample_index": sample_idx,
                "sample_id": str(names[0]),
                "length": length,
                "caption": str(captions[0]),
                "gif": str(gif_path),
                "cache": str(cache_path),
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": ckpt_epoch,
                "checkpoint_step": ckpt_step,
                "weight_source": str(weight_source),
            }
            records.append(row)
            print(f"VIS_DONE_SAMPLE sample={sample_idx:03d} gif={gif_path}", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_step": ckpt_step,
        "weight_source": str(weight_source),
        "split": str(args.split),
        "sample_indices": sample_indices,
        "seed": int(args.seed),
        "order": ["RVQ", "GEN nearest", "GEN continuous", "GT"],
        "records": records,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"VIS_DONE out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
