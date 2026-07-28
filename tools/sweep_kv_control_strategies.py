"""Sweep KV-control inference strategies on a fixed sparse-control panel."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

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
from models.codeflow.motion_code_flow import lengths_to_mask  # noqa: E402
from models.codeflow.motionstreamer272 import recover_motionstreamer272_positions_from_normalized  # noqa: E402
from utils.fixseed import fixseed  # noqa: E402


FOOT_IDS = [7, 10, 8, 11]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--data_root", type=str, default="dataset/HumanML3D_272")
    parser.add_argument("--mean_path", type=str, default="dataset/HumanML3D_272/Mean.npy")
    parser.add_argument("--std_path", type=str, default="dataset/HumanML3D_272/Std.npy")
    parser.add_argument("--vq_checkpoint", type=str, default="")
    parser.add_argument("--vq_partition", type=str, default="")
    parser.add_argument("--clip_path", type=str, default="")
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--sample_indices", type=str, default="")
    parser.add_argument("--sample_seed", type=int, default=3407)
    parser.add_argument("--control_seed", type=int, default=93407)
    parser.add_argument("--noise_seed", type=int, default=13407)
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
    parser.add_argument("--preset", type=str, default="quick", choices=["quick", "broad"])
    parser.add_argument("--strategies", type=str, default="")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--foot_height", type=float, default=0.08)
    parser.add_argument("--gt_contact_speed", type=float, default=0.03)
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _strategy(
    name: str,
    *,
    guidance_mode: str = "none",
    guidance_eta: float = 0.0,
    guidance_total_iters: int = 0,
    guidance_iter_schedule: str = "constant",
    guidance_eta_schedule: str = "constant",
    guidance_loss: str = "l2",
    guidance_anchor: float = 0.0,
    guidance_joint_anchor: float = 0.0,
    guidance_foot_skate_weight: float = 0.0,
    guidance_floor_weight: float = 0.0,
    guidance_smooth_weight: float = 0.0,
    guidance_start: float = 0.0,
    guidance_end: float = 1.0,
    guidance_grad_clip: float = 0.0,
    guidance_recompute_cond: str = "step",
) -> Dict[str, object]:
    return {
        "name": name,
        "guidance_mode": guidance_mode,
        "guidance_eta": float(guidance_eta),
        "guidance_total_iters": int(guidance_total_iters),
        "guidance_iter_schedule": guidance_iter_schedule,
        "guidance_eta_schedule": guidance_eta_schedule,
        "guidance_loss": guidance_loss,
        "guidance_anchor": float(guidance_anchor),
        "guidance_joint_anchor": float(guidance_joint_anchor),
        "guidance_foot_skate_weight": float(guidance_foot_skate_weight),
        "guidance_floor_weight": float(guidance_floor_weight),
        "guidance_smooth_weight": float(guidance_smooth_weight),
        "guidance_start": float(guidance_start),
        "guidance_end": float(guidance_end),
        "guidance_grad_clip": float(guidance_grad_clip),
        "guidance_recompute_cond": guidance_recompute_cond,
    }


def preset_strategies(preset: str) -> List[Dict[str, object]]:
    quick = [
        _strategy("B_only"),
        _strategy(
            "clean_eta006_total512_inc_l2",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=512,
            guidance_iter_schedule="linear_increase",
            guidance_loss="l2",
        ),
        _strategy(
            "clean_eta008_total1000_inc_l2_reference",
            guidance_mode="gradient",
            guidance_eta=0.08,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_increase",
            guidance_loss="l2",
        ),
        _strategy(
            "clean_eta006_total1000_inc_l2_jointanchor001",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_increase",
            guidance_loss="l2",
            guidance_joint_anchor=0.01,
        ),
        _strategy(
            "clean_eta006_total1000_inc_l2_footsafe",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_increase",
            guidance_loss="l2",
            guidance_joint_anchor=0.005,
            guidance_foot_skate_weight=0.5,
            guidance_floor_weight=1.0,
            guidance_smooth_weight=0.01,
        ),
        _strategy(
            "clean_eta006_total1000_const_l1_footsafe",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=1000,
            guidance_iter_schedule="constant",
            guidance_loss="l1",
            guidance_joint_anchor=0.005,
            guidance_foot_skate_weight=0.5,
            guidance_floor_weight=1.0,
            guidance_smooth_weight=0.01,
        ),
    ]
    if preset == "quick":
        return quick
    return quick + [
        _strategy(
            "clean_eta004_total1000_inc_l2_conservative",
            guidance_mode="gradient",
            guidance_eta=0.04,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_increase",
            guidance_loss="l2",
            guidance_joint_anchor=0.01,
        ),
        _strategy(
            "clean_eta006_total1000_dec_l2",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_decrease",
            guidance_loss="l2",
        ),
        _strategy(
            "clean_eta006_total1000_inc_dist",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_increase",
            guidance_loss="dist",
        ),
        _strategy(
            "clean_eta006_total1000_late_l2",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_increase",
            guidance_start=0.25,
            guidance_loss="l2",
        ),
        _strategy(
            "clean_eta006_total1000_inc_l2_strong_footsafe",
            guidance_mode="gradient",
            guidance_eta=0.06,
            guidance_total_iters=1000,
            guidance_iter_schedule="linear_increase",
            guidance_loss="l2",
            guidance_joint_anchor=0.01,
            guidance_foot_skate_weight=1.0,
            guidance_floor_weight=2.0,
            guidance_smooth_weight=0.02,
        ),
    ]


def select_strategies(args: argparse.Namespace) -> List[Dict[str, object]]:
    strategies = preset_strategies(str(args.preset))
    if not args.strategies:
        return strategies
    wanted = [name.strip() for name in args.strategies.split(",") if name.strip()]
    by_name = {str(item["name"]): item for item in strategies}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"Unknown strategies: {missing}. Available: {sorted(by_name)}")
    return [by_name[name] for name in wanted]


def make_eval_args(args: argparse.Namespace, strategy: Dict[str, object]) -> argparse.Namespace:
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
        seed=int(args.sample_seed),
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
        guidance_mode=str(strategy["guidance_mode"]),
        guidance_eta=float(strategy["guidance_eta"]),
        guidance_variable="clean",
        guidance_optimizer="adamw",
        guidance_adam_beta1=0.5,
        guidance_adam_beta2=0.9,
        guidance_weight_decay=1e-6,
        guidance_anchor=float(strategy["guidance_anchor"]),
        guidance_recompute_cond=str(strategy["guidance_recompute_cond"]),
        guidance_start=float(strategy["guidance_start"]),
        guidance_end=float(strategy["guidance_end"]),
        guidance_inner_iters=1,
        guidance_total_iters=int(strategy["guidance_total_iters"]),
        guidance_iter_schedule=str(strategy["guidance_iter_schedule"]),
        guidance_eta_schedule=str(strategy["guidance_eta_schedule"]),
        guidance_grad_clip=float(strategy["guidance_grad_clip"]),
        guidance_loss=str(strategy["guidance_loss"]),
        guidance_joint_anchor=float(strategy["guidance_joint_anchor"]),
        guidance_foot_skate_weight=float(strategy["guidance_foot_skate_weight"]),
        guidance_floor_weight=float(strategy["guidance_floor_weight"]),
        guidance_smooth_weight=float(strategy["guidance_smooth_weight"]),
        guidance_foot_height=float(args.foot_height),
        guidance_foot_temp=0.02,
        post_guidance_iters=0,
        post_guidance_lr=0.05,
        post_guidance_loss="dist",
        post_guidance_anchor=0.0,
        post_guidance_grad_clip=0.0,
        save_json_name="",
    )


def pick_sample_indices(args: argparse.Namespace, dataset_len: int) -> List[int]:
    if args.sample_indices:
        return [int(part) for part in args.sample_indices.replace(" ", "").split(",") if part]
    rng = np.random.default_rng(int(args.sample_seed))
    count = min(int(args.num_samples), int(dataset_len))
    return sorted(rng.choice(dataset_len, size=count, replace=False).astype(int).tolist())


def make_cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def _valid_frame_mask(lengths: torch.Tensor, frame_count: int) -> torch.Tensor:
    return lengths_to_mask(lengths.to(dtype=torch.long).clamp(min=1, max=int(frame_count)), int(frame_count))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        return float("nan")
    return float(values[mask].detach().float().mean().cpu().item())


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, q: float) -> float:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any().item()):
        return float("nan")
    return float(torch.quantile(values[mask].detach().float().cpu(), float(q)).item())


def foot_floor(target_joints: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    frame_count = target_joints.shape[1]
    valid = _valid_frame_mask(lengths, frame_count).to(device=target_joints.device)
    foot_ids = torch.as_tensor(FOOT_IDS, device=target_joints.device, dtype=torch.long)
    foot_y = target_joints[:, :, foot_ids, 1]
    masked = foot_y.masked_fill(~valid[:, :, None], float("inf"))
    floor = masked.amin(dim=(1, 2))
    fallback = foot_y.amin(dim=(1, 2))
    return torch.where(torch.isfinite(floor), floor, fallback)


def foot_metrics(
    joints: torch.Tensor,
    target_joints: torch.Tensor,
    lengths: torch.Tensor,
    *,
    foot_height: float,
    gt_contact_speed: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    frame_count = min(joints.shape[1], target_joints.shape[1])
    joints = joints[:, :frame_count]
    target = target_joints[:, :frame_count].to(device=joints.device, dtype=joints.dtype)
    valid = _valid_frame_mask(lengths, frame_count).to(device=joints.device)
    floor = foot_floor(target, lengths).to(device=joints.device, dtype=joints.dtype).view(-1, 1, 1)
    foot_ids = torch.as_tensor(FOOT_IDS, device=joints.device, dtype=torch.long)
    foot = joints[:, :, foot_ids]
    target_foot = target[:, :, foot_ids]
    height = foot[..., 1] - floor
    low = height <= float(foot_height)
    valid_foot = valid[:, :, None].expand_as(low)
    penetration = torch.relu(floor - foot[..., 1]) * valid_foot.to(dtype=joints.dtype)

    out: Dict[str, float] = {
        "foot_contact_frame_frac": float((low & valid_foot).float().sum().detach().cpu() / valid_foot.float().sum().detach().cpu().clamp_min(1.0)),
        "foot_penetration_cm": _masked_mean(penetration * 100.0, valid_foot),
        "foot_penetration_p95_cm": _masked_quantile(penetration * 100.0, valid_foot, 0.95),
    }
    per_sample: List[Dict[str, float]] = []
    if frame_count > 1:
        horiz = torch.linalg.norm(foot[:, 1:, :, [0, 2]] - foot[:, :-1, :, [0, 2]], dim=-1)
        pair_valid = (valid[:, 1:] & valid[:, :-1])[:, :, None].expand_as(horiz)
        pair_low = (low[:, 1:] & low[:, :-1]) & pair_valid
        out["foot_skate_cm_per_frame"] = _masked_mean(horiz * 100.0, pair_low)
        out["foot_skate_p95_cm_per_frame"] = _masked_quantile(horiz * 100.0, pair_low, 0.95)
        out["foot_skate_pair_frac"] = float(pair_low.float().sum().detach().cpu() / pair_valid.float().sum().detach().cpu().clamp_min(1.0))

        target_horiz = torch.linalg.norm(target_foot[:, 1:, :, [0, 2]] - target_foot[:, :-1, :, [0, 2]], dim=-1)
        target_height = target_foot[..., 1] - floor
        target_low = target_height <= float(foot_height)
        gt_contact = (target_low[:, 1:] & target_low[:, :-1]) & (target_horiz <= float(gt_contact_speed)) & pair_valid
        gen_height_pair = 0.5 * (height[:, 1:] + height[:, :-1])
        out["foot_float_on_gt_contact_cm"] = _masked_mean(torch.relu(gen_height_pair - float(foot_height)) * 100.0, gt_contact)
        out["gt_contact_pair_frac"] = float(gt_contact.float().sum().detach().cpu() / pair_valid.float().sum().detach().cpu().clamp_min(1.0))
    else:
        out.update(
            {
                "foot_skate_cm_per_frame": float("nan"),
                "foot_skate_p95_cm_per_frame": float("nan"),
                "foot_skate_pair_frac": 0.0,
                "foot_float_on_gt_contact_cm": float("nan"),
                "gt_contact_pair_frac": 0.0,
            }
        )

    if frame_count > 2:
        accel = torch.linalg.norm(foot[:, 2:] - 2.0 * foot[:, 1:-1] + foot[:, :-2], dim=-1)
        accel_valid = (valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2])[:, :, None].expand_as(accel)
        out["foot_accel_cm_per_frame2"] = _masked_mean(accel * 100.0, accel_valid)
        all_accel = torch.linalg.norm(joints[:, 2:] - 2.0 * joints[:, 1:-1] + joints[:, :-2], dim=-1)
        all_valid = (valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2])[:, :, None].expand_as(all_accel)
        out["joint_accel_cm_per_frame2"] = _masked_mean(all_accel * 100.0, all_valid)
    else:
        out["foot_accel_cm_per_frame2"] = float("nan")
        out["joint_accel_cm_per_frame2"] = float("nan")

    for idx in range(joints.shape[0]):
        sample_lengths = lengths[idx : idx + 1]
        sample_out, _ = foot_metrics(
            joints[idx : idx + 1],
            target_joints[idx : idx + 1],
            sample_lengths,
            foot_height=foot_height,
            gt_contact_speed=gt_contact_speed,
        ) if joints.shape[0] > 1 else ({}, [])
        per_sample.append(sample_out)
    return out, per_sample


def control_metrics(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    target_mask: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    frame_count = min(pred_joints.shape[1], target_joints.shape[1], target_mask.shape[1])
    pred = pred_joints[:, :frame_count]
    target = target_joints[:, :frame_count].to(device=pred.device, dtype=pred.dtype)
    mask = target_mask[:, :frame_count].to(device=pred.device, dtype=torch.bool)
    joint_mask = mask.any(dim=-1)
    dist = torch.linalg.norm(pred - target, dim=-1)
    valid = _valid_frame_mask(lengths, frame_count).to(device=pred.device)
    valid_joint = valid[:, :, None].expand_as(dist)

    control_values = dist[joint_mask]
    mpjpe_values = dist[valid_joint]
    if control_values.numel() == 0:
        aggregate = {
            "kps_cm": float("nan"),
            "kps_median_cm": float("nan"),
            "success_5cm": float("nan"),
            "success_10cm": float("nan"),
            "success_20cm": float("nan"),
            "control_points": 0,
        }
    else:
        aggregate = {
            "kps_cm": float(control_values.float().mean().detach().cpu() * 100.0),
            "kps_median_cm": float(control_values.float().median().detach().cpu() * 100.0),
            "success_5cm": float((control_values < 0.05).float().mean().detach().cpu()),
            "success_10cm": float((control_values < 0.10).float().mean().detach().cpu()),
            "success_20cm": float((control_values < 0.20).float().mean().detach().cpu()),
            "control_points": int(control_values.numel()),
        }
    aggregate["mpjpe_cm"] = float(mpjpe_values.float().mean().detach().cpu() * 100.0) if mpjpe_values.numel() else float("nan")

    per_sample: List[Dict[str, float]] = []
    for idx in range(pred.shape[0]):
        sample_mask = joint_mask[idx]
        sample_dist = dist[idx]
        sample_control = sample_dist[sample_mask]
        sample_valid = valid_joint[idx]
        sample_mpjpe = sample_dist[sample_valid]
        per_sample.append(
            {
                "kps_cm": float(sample_control.float().mean().detach().cpu() * 100.0) if sample_control.numel() else float("nan"),
                "kps_median_cm": float(sample_control.float().median().detach().cpu() * 100.0) if sample_control.numel() else float("nan"),
                "success_10cm": float((sample_control < 0.10).float().mean().detach().cpu()) if sample_control.numel() else float("nan"),
                "mpjpe_cm": float(sample_mpjpe.float().mean().detach().cpu() * 100.0) if sample_mpjpe.numel() else float("nan"),
                "control_points": int(sample_control.numel()),
            }
        )
    return aggregate, per_sample


def evaluate_joints(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    target_mask: torch.Tensor,
    lengths: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    control_agg, control_per_sample = control_metrics(pred_joints, target_joints, target_mask, lengths)
    foot_agg, foot_per_sample = foot_metrics(
        pred_joints,
        target_joints,
        lengths,
        foot_height=float(args.foot_height),
        gt_contact_speed=float(args.gt_contact_speed),
    )
    aggregate = copy.deepcopy(control_agg)
    aggregate.update(foot_agg)
    per_sample = []
    for c_item, f_item in zip(control_per_sample, foot_per_sample):
        item = copy.deepcopy(c_item)
        item.update(f_item)
        per_sample.append(item)
    return aggregate, per_sample


def decode_embeddings_to_joints(
    model,
    embeddings: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    terminal_mode = args.terminal_mode or None
    motion_norm = decode_motion_from_embeddings(
        model,
        embeddings,
        terminal_mode=terminal_mode,
        decode_mode=str(args.decode_mode),
    )
    return recover_motionstreamer272_positions_from_normalized(motion_norm, mean, std)


def naturalness_score(aggregate: Dict[str, float], no_control: Dict[str, float]) -> float:
    kps = float(aggregate.get("kps_cm", float("inf")))
    skate = float(aggregate.get("foot_skate_cm_per_frame", 0.0))
    no_skate = float(no_control.get("foot_skate_cm_per_frame", 0.0))
    penetration = float(aggregate.get("foot_penetration_cm", 0.0))
    accel = float(aggregate.get("foot_accel_cm_per_frame2", 0.0))
    no_accel = float(no_control.get("foot_accel_cm_per_frame2", 0.0))
    return kps + 5.0 * max(0.0, skate - no_skate) + 3.0 * penetration + 0.5 * max(0.0, accel - no_accel)


def main() -> None:
    args = parse_args()
    fixseed(int(args.sample_seed))
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    run_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint.parents[1]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "logs" / f"strategy_sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = select_strategies(args)
    device = make_device(args)
    load_args = make_eval_args(args, strategies[0])
    model, opt, ckpt, weight_source = load_model(checkpoint, load_args, device)
    dataset = MotionStreamer272T2MEvalDataset(
        args.data_root or str(getattr(opt, "data_root", "")),
        args.split,
        unit_length=int(getattr(opt, "unit_length", 4)),
        max_motion_length=int(getattr(opt, "motion_length", 300)),
        max_samples=0,
    )
    sample_indices = pick_sample_indices(args, len(dataset))
    batch = collate_motionstreamer272_t2m_eval([dataset[idx] for idx in sample_indices])
    raw_motion = batch["motion"].to(device=model.device, dtype=torch.float32)
    lengths = batch["length"].to(device=model.device, dtype=torch.long)
    mean = torch.from_numpy(np.load(args.mean_path).astype(np.float32)).to(model.device)
    std = torch.from_numpy(np.load(args.std_path).astype(np.float32)).to(model.device)
    target_norm = (raw_motion - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    token_lengths = (lengths // int(getattr(opt, "unit_length", 4))).clamp(min=1)

    control_gen = make_cuda_generator(model.device, int(args.control_seed))
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
        dropout_prob=float(args.control_dropout_prob),
        keyframe_strategy=str(args.control_keyframe_strategy),
        generator=control_gen,
    )
    target_joints = control_batch["target_joints"]
    target_mask = control_batch["target_mask"]
    latent_len = int(token_lengths.max().item())
    noise_gen = make_cuda_generator(model.device, int(args.noise_seed))
    init_noise = torch.randn(
        raw_motion.shape[0],
        latent_len,
        int(model.config.num_parts),
        int(model.config.code_dim),
        device=model.device,
        generator=noise_gen,
    ) * float(model.config.noise_scale)

    started = time.time()
    no_control_args = make_eval_args(args, _strategy("no_control"))
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
            args=no_control_args,
        )
        no_control_joints = decode_embeddings_to_joints(model, no_control_embeddings, mean, std, args)
    no_control_agg, no_control_per_sample = evaluate_joints(no_control_joints, target_joints, target_mask, lengths, args)

    payload: Dict[str, object] = {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "epoch": int(ckpt.get("epoch", 0)),
        "step": int(ckpt.get("step", 0)),
        "weight_source": str(weight_source),
        "control_protocol": CONTROL_PROTOCOL,
        "split": str(args.split),
        "sample_indices": sample_indices,
        "sample_ids": [str(item) for item in batch["name"]],
        "captions": [str(item) for item in batch["caption"]],
        "seeds": {
            "sample_seed": int(args.sample_seed),
            "control_seed": int(args.control_seed),
            "noise_seed": int(args.noise_seed),
        },
        "config": {
            "steps": int(args.steps),
            "cond_scale": float(args.cond_scale),
            "decode_mode": str(args.decode_mode),
            "min_keyframes": int(args.min_keyframes),
            "max_keyframes": int(args.max_keyframes),
            "min_joints": int(args.min_joints),
            "max_joints": int(args.max_joints),
            "control_profile": str(args.control_profile),
            "control_keyframe_strategy": str(args.control_keyframe_strategy),
            "foot_height": float(args.foot_height),
            "gt_contact_speed": float(args.gt_contact_speed),
        },
        "no_control": {
            "aggregate": no_control_agg,
            "per_sample": no_control_per_sample,
        },
        "strategies": [],
    }
    write_json(out_dir / "sweep_summary.json", payload)

    if bool(args.save_npz):
        np.savez_compressed(
            out_dir / "fixed_panel_inputs.npz",
            sample_indices=np.asarray(sample_indices, dtype=np.int64),
            lengths=lengths.detach().cpu().numpy(),
            target_joints=target_joints.detach().cpu().numpy(),
            target_mask=target_mask.detach().cpu().numpy(),
            no_control_joints=no_control_joints.detach().cpu().numpy(),
        )

    print(
        f"SWEEP_START epoch={payload['epoch']} step={payload['step']} "
        f"samples={len(sample_indices)} out_dir={out_dir}",
        flush=True,
    )
    print(
        "SWEEP_NO_CONTROL "
        f"kps={no_control_agg['kps_cm']:.3f}cm "
        f"skate={no_control_agg['foot_skate_cm_per_frame']:.3f}cm/frame "
        f"penetration={no_control_agg['foot_penetration_cm']:.3f}cm",
        flush=True,
    )

    for strategy in strategies:
        strategy_args = make_eval_args(args, strategy)
        strategy_start = time.time()
        enable_grad = str(strategy["guidance_mode"]) == "gradient"
        context = torch.enable_grad() if enable_grad else torch.no_grad()
        with context:
            embeddings = sample_controlled_embeddings(
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
                args=strategy_args,
            )
        with torch.no_grad():
            pred_joints = decode_embeddings_to_joints(model, embeddings, mean, std, args)
            aggregate, per_sample = evaluate_joints(pred_joints, target_joints, target_mask, lengths, args)
        aggregate["score_lower_is_better"] = float(naturalness_score(aggregate, no_control_agg))
        row = {
            "name": str(strategy["name"]),
            "strategy": strategy,
            "elapsed_sec": float(time.time() - strategy_start),
            "aggregate": aggregate,
            "per_sample": per_sample,
        }
        payload["strategies"].append(row)
        ranked = sorted(
            payload["strategies"],
            key=lambda item: (
                float(item["aggregate"].get("score_lower_is_better", float("inf"))),
                float(item["aggregate"].get("kps_cm", float("inf"))),
            ),
        )
        payload["ranking"] = [
            {
                "name": str(item["name"]),
                "score_lower_is_better": float(item["aggregate"]["score_lower_is_better"]),
                "kps_cm": float(item["aggregate"]["kps_cm"]),
                "success_10cm": float(item["aggregate"]["success_10cm"]),
                "foot_skate_cm_per_frame": float(item["aggregate"]["foot_skate_cm_per_frame"]),
                "foot_penetration_cm": float(item["aggregate"]["foot_penetration_cm"]),
                "mpjpe_cm": float(item["aggregate"]["mpjpe_cm"]),
            }
            for item in ranked
        ]
        payload["elapsed_sec"] = float(time.time() - started)
        write_json(out_dir / "sweep_summary.json", payload)
        write_json(out_dir / f"{strategy['name']}.json", row)
        if bool(args.save_npz):
            np.savez_compressed(
                out_dir / f"{strategy['name']}.npz",
                pred_joints=pred_joints.detach().cpu().numpy(),
                lengths=lengths.detach().cpu().numpy(),
                target_joints=target_joints.detach().cpu().numpy(),
                target_mask=target_mask.detach().cpu().numpy(),
            )
        print(
            f"SWEEP_STRATEGY name={strategy['name']} "
            f"kps={aggregate['kps_cm']:.3f}cm "
            f"succ10={aggregate['success_10cm']:.3f} "
            f"mpjpe={aggregate['mpjpe_cm']:.3f}cm "
            f"skate={aggregate['foot_skate_cm_per_frame']:.3f}cm/frame "
            f"penetration={aggregate['foot_penetration_cm']:.3f}cm "
            f"score={aggregate['score_lower_is_better']:.3f} "
            f"elapsed={row['elapsed_sec']:.1f}s",
            flush=True,
        )
        del embeddings, pred_joints
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"SWEEP_DONE result={out_dir / 'sweep_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
