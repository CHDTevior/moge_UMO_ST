"""Train HY273 raw-space rectified-flow model."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text
from models.raw_motion.flow_schedule import (
    bce_logits_masked,
    build_flow_state,
    clean_from_velocity,
    mse_masked,
    sample_timesteps,
    smooth_l1_masked,
)
from models.raw_motion.asset_integrity import verify_asset_manifest
from models.raw_motion.hy273_constraints import (
    KimodoControlCurriculum,
    build_kimodo_control_curriculum_batch,
    build_synthetic_control_batch,
)
from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_kimodo_training_transform
from models.raw_motion.hy273_slices import (
    CONTACT_JOINTS,
    CONTACT_SLICE,
    CONT_DIM,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_DIM,
    ROOT_SLICE,
    VELOCITY_SLICE,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)
from models.raw_motion.raw_flow_dit import HY273RawFlow


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    return data


def flatten_get(cfg: dict[str, Any], dotted: str, default: Any) -> Any:
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> tuple[torch.device, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size(), local_rank
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, -1


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def seed_all(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank) * 100003
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_run_name(name: str) -> str:
    value = str(name)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("--name must be a safe run basename")
    return value


def validate_execution_contract(args: argparse.Namespace) -> None:
    contract = str(getattr(args, "execution_contract", "none"))
    if contract == "none":
        return
    phase = str(args.training_phase)
    modes = {item.strip() for item in str(args.control_modes).split(",") if item.strip()}
    max_steps = int(args.max_steps)
    if contract.startswith("stage1_"):
        if phase != "text_only" or modes != {"none"}:
            raise RuntimeError(f"{contract} requires text_only with only the none control mode")
        if float(args.control_cont_loss_weight) != 0.0 or float(
            args.control_contact_loss_weight
        ) != 0.0:
            raise RuntimeError(f"{contract} requires zero controlled-entry losses")
        if contract == "stage1_production" and max_steps != 200_000:
            raise RuntimeError("stage1_production requires max_steps=200000")
        if contract == "stage1_pilot" and not 0 < max_steps < 200_000:
            raise RuntimeError("stage1_pilot requires 0 < max_steps < 200000")
        return
    if phase != "control":
        raise RuntimeError(f"{contract} requires the control phase")
    complete_protocol = contract.startswith("stage2_kimodo_complete_")
    expected_modes = {
        "none",
        "root_sparse",
        "root_dense",
        "endpoints",
        "fullpose",
        "mixed",
    }
    if complete_protocol:
        expected_modes.add("contact")
    if modes != expected_modes:
        protocol_name = "complete Kimodo" if complete_protocol else "frozen v1"
        raise RuntimeError(f"{contract} requires the {protocol_name} control mode set")
    if float(args.control_cont_loss_weight) <= 0.0:
        raise RuntimeError(f"{contract} requires a positive continuous control loss")
    if complete_protocol:
        if float(args.control_contact_loss_weight) <= 0.0:
            raise RuntimeError(f"{contract} requires a positive contact-control loss")
        if not bool(args.control_include_endpoint_rotations):
            raise RuntimeError(f"{contract} requires endpoint rotation control")
        if str(args.endpoint_preset) != "kimodo_ee":
            raise RuntimeError(f"{contract} requires endpoint_preset=kimodo_ee")
        if float(args.control_dense_min_fraction) != 1.0:
            raise RuntimeError(
                f"{contract} requires full-sequence dense root paths"
            )
    elif float(args.control_contact_loss_weight) != 0.0:
        raise RuntimeError(f"{contract} requires zero contact-control loss for v1")
    if int(args.control_curriculum_start_step) != 200_000:
        raise RuntimeError(f"{contract} requires curriculum_start_step=200000")
    if contract in {"stage2_production", "stage2_kimodo_complete_production"} and max_steps != 400_000:
        raise RuntimeError(f"{contract} requires max_steps=400000")
    if contract == "stage2_kimodo_complete_production":
        if int(args.max_epochs) < 8_000:
            raise RuntimeError(
                "stage2_kimodo_complete_production requires max_epochs>=8000"
            )
        if int(args.save_every) != 50_000:
            raise RuntimeError(
                "stage2_kimodo_complete_production requires save_every=50000"
            )
    if contract in {"stage2_pilot", "stage2_kimodo_complete_pilot"} and not 200_000 < max_steps < 400_000:
        raise RuntimeError(f"{contract} requires 200000 < max_steps < 400000")


def build_arg_parser() -> argparse.ArgumentParser:
    # Config precedence depends on exact option identities, so accepting argparse
    # abbreviations would let a parsed CLI value be mistaken for an absent option.
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--config", default="configs/raw_flow_hy273.yaml")
    p.add_argument("--data_root", default="")
    p.add_argument("--text_root", default="")
    p.add_argument("--split", default="train")
    p.add_argument("--output_dir", default="checkpoints/t2m/hy273_raw_flow")
    p.add_argument("--name", default="hy273_raw_flow")
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--min_frames", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=4000)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=50000)
    p.add_argument(
        "--save_final",
        "--save-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write latest.pt when training exits; disable only for short smoke runs.",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_every", type=int, default=1)
    p.add_argument("--time_schedule", choices=["uniform", "logit_normal"], default="logit_normal")
    p.add_argument("--denoiser_p_mean", type=float, default=0.0)
    p.add_argument("--denoiser_p_std", type=float, default=1.0)
    p.add_argument("--prediction_type", choices=["velocity", "x0"], default="x0")
    p.add_argument(
        "--architecture",
        choices=["one_stage", "redenoise_kimodo_like"],
        default="one_stage",
    )
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--depth_double", type=int, default=6)
    p.add_argument("--depth_single", type=int, default=12)
    p.add_argument("--mlp_ratio", type=float, default=2.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--root_depth_double", type=int, default=3)
    p.add_argument("--root_depth_single", type=int, default=6)
    p.add_argument("--body_depth_double", type=int, default=3)
    p.add_argument("--body_depth_single", type=int, default=6)
    p.add_argument("--motion_stats_dir", default="")
    p.add_argument("--local_root_stats_dir", default="")
    p.add_argument("--stats_variance_eps", type=float, default=0.0)
    p.add_argument("--stats_manifest_sha256", default="")
    p.add_argument("--asset_manifest_path", default="")
    p.add_argument("--asset_manifest_sha256", default="")
    p.add_argument("--detach_root_bridge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--text_encoder", choices=["clip", "hy_cache", "hytext_cache", "qwen_clip_cache", "none"], default="clip")
    p.add_argument("--clip_path", default="checkpoints/clip/ViT-B-32.pt")
    p.add_argument("--clip_version", default="ViT-B/32")
    p.add_argument("--max_text_tokens", type=int, default=50)
    p.add_argument("--hytext_cache_dir", default="")
    p.add_argument("--hytext_ctxt_dim", type=int, default=4096)
    p.add_argument("--hytext_vtxt_dim", type=int, default=768)
    p.add_argument("--hytext_max_open_shards", type=int, default=8)
    p.add_argument("--hytext_allow_cache_miss", action="store_true")
    p.add_argument("--text_dropout_prob", type=float, default=0.1)
    p.add_argument("--random_first_heading", action="store_true")
    p.add_argument("--no_random_first_heading", action="store_true")
    p.add_argument("--root_origin_shift", action="store_true")
    p.add_argument("--no_root_origin_shift", action="store_true")
    p.add_argument("--control_modes", default="none,root,endpoints,fullpose,mixed")
    p.add_argument(
        "--training_phase",
        choices=["text_only", "control"],
        default="text_only",
    )
    p.add_argument("--control_curriculum_start_step", type=int, default=200000)
    p.add_argument("--control_curriculum_steps", type=int, default=200000)
    p.add_argument("--control_none_prob", type=float, default=0.10)
    p.add_argument("--control_mixed_prob", type=float, default=0.25)
    p.add_argument("--control_dense_min_fraction", type=float, default=0.25)
    p.add_argument(
        "--control_include_endpoint_rotations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--max_control_keyframes", type=int, default=8)
    p.add_argument("--endpoint_preset", choices=["kimodo_ee", "five_point"], default="kimodo_ee")
    p.add_argument(
        "--endpoint_subset_mode",
        choices=["all", "random_nonempty"],
        default="random_nonempty",
    )
    p.add_argument(
        "--endpoint_root_ref_mode",
        choices=["kimodo_hidden_root", "none"],
        default="kimodo_hidden_root",
    )
    p.add_argument("--flow_loss_weight", type=float, default=1.0)
    p.add_argument("--contact_loss_weight", type=float, default=0.1)
    p.add_argument("--control_cont_loss_weight", type=float, default=0.25)
    p.add_argument("--control_contact_loss_weight", type=float, default=0.05)
    p.add_argument("--clean_cont_loss_weight", type=float, default=0.0)
    p.add_argument("--clean_root_vel_loss_weight", type=float, default=0.0)
    p.add_argument("--clean_joint_vel_loss_weight", type=float, default=0.0)
    p.add_argument("--foot_lock_loss_weight", type=float, default=0.0)
    p.add_argument("--semantic_loss_fps", type=float, default=30.0)
    p.add_argument("--foot_lock_contact_threshold", type=float, default=0.5)
    p.add_argument("--root_heading_loss_weight", type=float, default=1.0)
    p.add_argument("--velocity_loss_weight", type=float, default=1.0)
    p.add_argument(
        "--representation_loss_mode",
        choices=["per_entry", "semantic_weighted"],
        default="per_entry",
    )
    p.add_argument("--representation_loss_scale", type=float, default=1.0)
    p.add_argument(
        "--representation_loss_space",
        choices=["auto", "x0", "velocity"],
        default="auto",
        help=(
            "Space used by the continuous representation loss. 'auto' preserves the "
            "legacy behavior: x0 loss for x0 prediction and velocity loss for velocity prediction."
        ),
    )
    p.add_argument(
        "--velocity_loss_t_eps",
        type=float,
        default=0.05,
        help="Clamp for 1-t when an x0 prediction is transformed into velocity loss space.",
    )
    p.add_argument("--fk_consistency_loss_weight", type=float, default=0.0)
    p.add_argument("--fk_consistency_scale_m", type=float, default=0.05)
    p.add_argument("--fk_consistency_warmup_steps", type=int, default=0)
    p.add_argument("--deterministic_trace", action="store_true")
    p.add_argument("--trace_seed", type=int, default=3407)
    p.add_argument("--trace_hash_steps", type=int, default=100)
    p.add_argument(
        "--resume_mode",
        choices=["strict", "loss_fork", "phase_transition"],
        default="strict",
    )
    p.add_argument("--expected_resume_step", type=int, default=-1)
    p.add_argument("--resume_epoch", type=int, default=-1)
    p.add_argument("--resume_step_in_epoch", type=int, default=-1)
    p.add_argument("--require_exact_resume_cursor", action="store_true")
    p.add_argument("--resume_sha256", default="")
    p.add_argument("--expected_initial_model_sha256", default="")
    p.add_argument("--source_manifest_sha256", default="")
    p.add_argument(
        "--execution_contract",
        choices=[
            "none",
            "stage1_production",
            "stage1_pilot",
            "stage2_production",
            "stage2_pilot",
            "stage2_kimodo_complete_production",
            "stage2_kimodo_complete_pilot",
        ],
        default="none",
    )
    p.add_argument("--self_conditioning", action="store_true")
    p.add_argument("--self_cond_train_prob", type=float, default=0.5)
    p.add_argument("--self_cond_mode", default="add_proj")
    p.add_argument("--self_cond_scale", type=float, default=1.0)
    p.set_defaults(resume_contract_version=2)
    return p


def explicit_cli_destinations(
    parser: argparse.ArgumentParser, argv: list[str]
) -> set[str]:
    destinations: set[str] = set()
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            destinations.add(action.dest)
    return destinations


def merge_config(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    explicit_cli: set[str] | None = None,
) -> argparse.Namespace:
    defaults = {
        "data_root": "data.data_root",
        "text_root": "data.text_root",
        "max_frames": "data.max_frames",
        "min_frames": "data.min_frames",
        "batch_size": "train.batch_size",
        "num_workers": "train.num_workers",
        "max_epochs": "train.max_epochs",
        "max_steps": "train.max_steps",
        "save_every": "train.save_every",
        "lr": "train.lr",
        "weight_decay": "train.weight_decay",
        "grad_clip": "train.grad_clip",
        "gradient_accumulation_steps": "train.gradient_accumulation_steps",
        "time_schedule": "train.time_schedule",
        "denoiser_p_mean": "train.denoiser_p_mean",
        "denoiser_p_std": "train.denoiser_p_std",
        "prediction_type": "model.prediction_type",
        "architecture": "model.architecture",
        "hidden_dim": "model.hidden_dim",
        "num_heads": "model.num_heads",
        "depth_double": "model.depth_double",
        "depth_single": "model.depth_single",
        "mlp_ratio": "model.mlp_ratio",
        "dropout": "model.dropout",
        "root_depth_double": "model.root_depth_double",
        "root_depth_single": "model.root_depth_single",
        "body_depth_double": "model.body_depth_double",
        "body_depth_single": "model.body_depth_single",
        "motion_stats_dir": "normalization.motion_stats_dir",
        "local_root_stats_dir": "normalization.local_root_stats_dir",
        "stats_variance_eps": "normalization.variance_eps",
        "stats_manifest_sha256": "normalization.manifest_sha256",
        "asset_manifest_path": "assets.manifest_path",
        "asset_manifest_sha256": "assets.manifest_sha256",
        "source_manifest_sha256": "assets.source_manifest_sha256",
        "expected_initial_model_sha256": "assets.expected_initial_model_sha256",
        "detach_root_bridge": "model.detach_root_bridge",
        "text_encoder": "text.encoder",
        "clip_path": "text.clip_path",
        "clip_version": "text.clip_version",
        "max_text_tokens": "text.max_text_tokens",
        "hytext_cache_dir": "text.hytext_cache_dir",
        "hytext_ctxt_dim": "text.hytext_ctxt_dim",
        "hytext_vtxt_dim": "text.hytext_vtxt_dim",
        "hytext_max_open_shards": "text.hytext_max_open_shards",
        "text_dropout_prob": "text.dropout_prob",
        "amp_dtype": "train.amp_dtype",
        "ema_decay": "ema.decay",
        "ema_every": "ema.every",
        "flow_loss_weight": "loss.flow",
        "contact_loss_weight": "loss.contact",
        "control_cont_loss_weight": "loss.control_cont",
        "control_contact_loss_weight": "loss.control_contact",
        "clean_cont_loss_weight": "loss.clean_cont",
        "clean_root_vel_loss_weight": "loss.clean_root_vel",
        "clean_joint_vel_loss_weight": "loss.clean_joint_vel",
        "foot_lock_loss_weight": "loss.foot_lock",
        "semantic_loss_fps": "loss.semantic_fps",
        "foot_lock_contact_threshold": "loss.foot_lock_contact_threshold",
        "root_heading_loss_weight": "loss.root_heading",
        "velocity_loss_weight": "loss.velocity",
        "representation_loss_mode": "loss.representation_mode",
        "representation_loss_scale": "loss.representation_scale",
        "representation_loss_space": "loss.representation_space",
        "velocity_loss_t_eps": "loss.velocity_t_eps",
        "fk_consistency_loss_weight": "loss.fk_consistency",
        "fk_consistency_scale_m": "loss.fk_consistency_scale_m",
        "fk_consistency_warmup_steps": "loss.fk_consistency_warmup_steps",
        "deterministic_trace": "train.deterministic_trace",
        "trace_seed": "train.trace_seed",
        "trace_hash_steps": "train.trace_hash_steps",
        "control_modes": "control.modes",
        "training_phase": "control.training_phase",
        "control_curriculum_start_step": "control.curriculum_start_step",
        "control_curriculum_steps": "control.curriculum_steps",
        "control_none_prob": "control.none_prob",
        "control_mixed_prob": "control.mixed_prob",
        "control_dense_min_fraction": "control.dense_min_fraction",
        "control_include_endpoint_rotations": "control.include_endpoint_rotations",
        "max_control_keyframes": "control.max_keyframes",
        "endpoint_preset": "control.endpoint_preset",
        "endpoint_subset_mode": "control.endpoint_subset_mode",
        "endpoint_root_ref_mode": "control.endpoint_root_ref_mode",
        "self_cond_train_prob": "self_conditioning.train_prob",
        "self_cond_mode": "self_conditioning.mode",
        "self_cond_scale": "self_conditioning.scale",
    }
    explicit_cli = explicit_cli or set()
    parser_defaults = vars(build_arg_parser().parse_args([]))
    for attr, path in defaults.items():
        if attr not in explicit_cli and getattr(args, attr) == parser_defaults[attr]:
            setattr(args, attr, flatten_get(cfg, path, getattr(args, attr)))
    if not args.self_conditioning:
        args.self_conditioning = bool(flatten_get(cfg, "self_conditioning.enabled", False))
    if not args.hytext_allow_cache_miss:
        args.hytext_allow_cache_miss = bool(flatten_get(cfg, "text.hytext_allow_cache_miss", False))
    if not args.amp:
        args.amp = bool(flatten_get(cfg, "train.amp", False))
    if not args.ema and not args.no_ema:
        args.ema = bool(flatten_get(cfg, "ema.enabled", False))
    if isinstance(args.control_modes, (list, tuple)):
        args.control_modes = ",".join(str(item) for item in args.control_modes)
    if not args.random_first_heading and not args.no_random_first_heading:
        args.random_first_heading = bool(flatten_get(cfg, "transform.random_first_heading", True))
    if not args.root_origin_shift and not args.no_root_origin_shift:
        args.root_origin_shift = bool(flatten_get(cfg, "transform.root_origin_shift", True))
    if args.no_random_first_heading:
        args.random_first_heading = False
    if args.no_root_origin_shift:
        args.root_origin_shift = False
    if args.no_ema:
        args.ema = False
    return args


def create_model(args: argparse.Namespace) -> torch.nn.Module:
    common = dict(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        text_encoder=args.text_encoder,
        clip_path=args.clip_path,
        clip_version=args.clip_version,
        max_text_tokens=args.max_text_tokens,
        hytext_cache_dir=getattr(args, "hytext_cache_dir", ""),
        hytext_ctxt_dim=getattr(args, "hytext_ctxt_dim", 4096),
        hytext_vtxt_dim=getattr(args, "hytext_vtxt_dim", 768),
        hytext_max_open_shards=getattr(args, "hytext_max_open_shards", 8),
        hytext_strict_cache=not bool(getattr(args, "hytext_allow_cache_miss", False)),
        self_conditioning=args.self_conditioning,
        self_cond_scale=args.self_cond_scale,
    )
    if str(getattr(args, "architecture", "one_stage")) == "one_stage":
        return HY273RawFlow(
            depth_double=args.depth_double,
            depth_single=args.depth_single,
            self_cond_mode=args.self_cond_mode,
            **common,
        )
    if args.architecture == "redenoise_kimodo_like":
        if str(args.prediction_type) != "x0":
            raise ValueError("redenoise_kimodo_like requires --prediction_type x0")
        from models.raw_motion.kimodo_like_flow_dit import HY273RedenoiseKimodoLike

        return HY273RedenoiseKimodoLike(
            root_depth_double=args.root_depth_double,
            root_depth_single=args.root_depth_single,
            body_depth_double=args.body_depth_double,
            body_depth_single=args.body_depth_single,
            motion_stats_dir=args.motion_stats_dir,
            local_root_stats_dir=args.local_root_stats_dir,
            fps=args.semantic_loss_fps,
            stats_variance_eps=args.stats_variance_eps,
            detach_root_bridge=args.detach_root_bridge,
            **common,
        )
    raise ValueError(f"Unknown architecture: {args.architecture}")


def continuous_loss_weights(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.ones(1, 1, CONT_DIM, device=device, dtype=dtype)
    weights[..., :ROOT_DIM] = float(args.root_heading_loss_weight)
    weights[..., VELOCITY_SLICE] = float(args.velocity_loss_weight)
    return weights


def weighted_mse_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(-1)
    weight = weight.to(device=pred.device, dtype=pred.dtype)
    denom = (mask_f * weight).sum().clamp_min(1.0)
    return ((pred - target).square() * mask_f * weight).sum() / denom


def weighted_smooth_l1_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(-1)
    weight = weight.to(device=pred.device, dtype=pred.dtype)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    denom = (mask_f * weight).sum().clamp_min(1.0)
    return (loss * mask_f * weight).sum() / denom


SEMANTIC_BLOCKS = {
    "root": (ROOT_SLICE, 10.0),
    "heading": (HEADING_SLICE, 2.0),
    "joint_pos": (JOINT_POS_SLICE, 10.0),
    "rot6d": (GLOBAL_ROT_SLICE, 10.0),
    "velocity": (VELOCITY_SLICE, 3.0),
}
SEMANTIC_WEIGHT_SUM = sum(weight for _, weight in SEMANTIC_BLOCKS.values())


def representation_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    raw_components = {
        name: mse_masked(pred[..., block_slice], target[..., block_slice], mask[..., block_slice])
        for name, (block_slice, _weight) in SEMANTIC_BLOCKS.items()
    }
    mode = str(getattr(args, "representation_loss_mode", "per_entry"))
    scale = float(getattr(args, "representation_loss_scale", 1.0))
    if mode == "per_entry":
        entry_weights = continuous_loss_weights(args, pred.device, pred.dtype)
        total = weighted_mse_masked(pred, target, mask, entry_weights)
        weighted_components = {
            name: raw.detach() * 0.0 for name, raw in raw_components.items()
        }
        return total, raw_components, weighted_components
    if mode != "semantic_weighted":
        raise ValueError(f"Unknown representation_loss_mode: {mode}")
    weighted_components = {
        name: raw_components[name] * (scale * weight / SEMANTIC_WEIGHT_SUM)
        for name, (_block_slice, weight) in SEMANTIC_BLOCKS.items()
    }
    total = sum(weighted_components.values())
    return total, raw_components, weighted_components


def predict_clean_cont(
    z_cont_imp: torch.Tensor,
    t: torch.Tensor,
    pred_cont: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "x0":
        return pred_cont
    if prediction_type == "velocity":
        return clean_from_velocity(z_cont_imp, t, pred_cont)
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


def prediction_velocity_cont(
    z_cont_imp: torch.Tensor,
    t: torch.Tensor,
    pred_cont: torch.Tensor,
    x0_hat_cont: torch.Tensor,
    prediction_type: str,
    velocity_t_eps: float = 1e-4,
) -> torch.Tensor:
    if prediction_type == "velocity":
        return pred_cont
    if prediction_type == "x0":
        if velocity_t_eps <= 0:
            raise ValueError(f"velocity_t_eps must be positive, got {velocity_t_eps}")
        t_view = t.view(-1, 1, 1).to(device=z_cont_imp.device, dtype=z_cont_imp.dtype)
        return (x0_hat_cont - z_cont_imp) / (1.0 - t_view).clamp_min(
            float(velocity_t_eps)
        )
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


def resolve_representation_loss_space(
    prediction_type: str,
    requested_space: str,
) -> str:
    if requested_space == "auto":
        return "x0" if prediction_type == "x0" else "velocity"
    if requested_space not in {"x0", "velocity"}:
        raise ValueError(f"Unknown representation_loss_space: {requested_space}")
    return requested_space


def representation_loss_pair(
    *,
    z_cont_imp: torch.Tensor,
    t: torch.Tensor,
    x0_hat_cont: torch.Tensor,
    x0_target_cont: torch.Tensor,
    v_pred_cont: torch.Tensor,
    v_target_cont: torch.Tensor,
    prediction_type: str,
    loss_space: str,
    velocity_t_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    resolved_space = resolve_representation_loss_space(prediction_type, loss_space)
    if resolved_space == "x0":
        return x0_hat_cont, x0_target_cont, resolved_space
    if prediction_type == "velocity":
        return v_pred_cont, v_target_cont, resolved_space
    if prediction_type != "x0":
        raise ValueError(f"Unknown prediction_type: {prediction_type}")
    if velocity_t_eps <= 0:
        raise ValueError(f"velocity_loss_t_eps must be positive, got {velocity_t_eps}")
    t_view = t.view(-1, 1, 1).to(device=z_cont_imp.device, dtype=z_cont_imp.dtype)
    denom = (1.0 - t_view).clamp_min(float(velocity_t_eps))
    return (
        (x0_hat_cont - z_cont_imp) / denom,
        (x0_target_cont - z_cont_imp) / denom,
        resolved_space,
    )


def entries_smooth_l1_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    mask_f = mask.to(device=pred.device, dtype=pred.dtype)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(-1)
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    denom = mask_f.expand_as(loss).sum().clamp_min(1.0)
    return (loss * mask_f).sum() / denom


def fk_position_consistency_loss(
    x0_hat_norm: torch.Tensor,
    observed_norm: torch.Tensor,
    motion_mask: torch.Tensor,
    valid: torch.Tensor,
    normalizer: HY273Normalizer,
    scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scale_m <= 0:
        raise ValueError(f"fk_consistency_scale_m must be positive, got {scale_m}")
    mask_f = motion_mask.to(device=x0_hat_norm.device, dtype=x0_hat_norm.dtype)
    x0_hat_clamped = x0_hat_norm * (1.0 - mask_f) + observed_norm * mask_f
    with torch.autocast(device_type=x0_hat_norm.device.type, enabled=False):
        x0_hat_un = normalizer.denormalize(x0_hat_clamped.float())
        joints_pos = reconstruct_global_joints_from_features(x0_hat_un)
        joints_fk = fk_positions_from_global_rot6d(x0_hat_un)
        residual_scaled = (joints_fk - joints_pos) / float(scale_m)
        element_loss = F.smooth_l1_loss(
            residual_scaled,
            torch.zeros_like(residual_scaled),
            reduction="none",
            beta=1.0,
        )
        valid_f = valid.to(device=element_loss.device, dtype=element_loss.dtype)[..., None, None]
        denom = valid_f.expand_as(element_loss).sum().clamp_min(1.0)
        loss = (element_loss * valid_f).sum() / denom
        distance_cm = (joints_fk - joints_pos).norm(dim=-1) * 100.0
        distance_denom = valid_f[..., 0].expand_as(distance_cm).sum().clamp_min(1.0)
        mean_distance_cm = (distance_cm * valid_f[..., 0]).sum() / distance_denom
    return loss, mean_distance_cm


def compute_clean_semantic_losses(
    x0_hat_un: torch.Tensor,
    x0_un: torch.Tensor,
    valid: torch.Tensor,
    fps: float,
    contact_threshold: float,
) -> dict[str, torch.Tensor]:
    zero = x0_hat_un.sum() * 0.0
    if x0_hat_un.shape[1] < 2:
        return {"clean_root_vel": zero, "clean_joint_vel": zero, "foot_lock": zero}

    valid_pair = valid[:, 1:] & valid[:, :-1]
    fps_f = float(fps)
    pred_root_vel = (x0_hat_un[:, 1:, ROOT_SLICE] - x0_hat_un[:, :-1, ROOT_SLICE]) * fps_f
    target_root_vel = (x0_un[:, 1:, ROOT_SLICE] - x0_un[:, :-1, ROOT_SLICE]) * fps_f
    clean_root_vel = entries_smooth_l1_masked(pred_root_vel, target_root_vel, valid_pair)

    pred_joints = reconstruct_global_joints_from_features(x0_hat_un)
    target_joints = reconstruct_global_joints_from_features(x0_un)
    pred_joint_vel = (pred_joints[:, 1:] - pred_joints[:, :-1]) * fps_f
    target_joint_vel = (target_joints[:, 1:] - target_joints[:, :-1]) * fps_f
    clean_joint_vel = entries_smooth_l1_masked(pred_joint_vel, target_joint_vel, valid_pair)

    contact_gt = x0_un[..., CONTACT_SLICE] > float(contact_threshold)
    contact_pair = contact_gt[:, 1:] & contact_gt[:, :-1] & valid_pair[..., None]
    pred_foot_vel = pred_joint_vel[:, :, list(CONTACT_JOINTS)]
    foot_lock = entries_smooth_l1_masked(pred_foot_vel, torch.zeros_like(pred_foot_vel), contact_pair)
    return {
        "clean_root_vel": clean_root_vel,
        "clean_joint_vel": clean_joint_vel,
        "foot_lock": foot_lock,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    ema_state: dict[str, torch.Tensor] | None = None,
    train_state: dict[str, Any] | None = None,
    normalizer_state: dict[str, torch.Tensor] | None = None,
) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "step": int(step),
    }
    if train_state is not None:
        payload["train_state"] = dict(train_state)
        payload["epoch"] = int(train_state["next_epoch"])
    if ema_state is not None:
        payload["ema"] = {key: value.detach().cpu() for key, value in ema_state.items()}
    if normalizer_state is not None:
        payload["normalizer"] = {
            key: value.detach().cpu().clone() for key, value in normalizer_state.items()
        }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def hash_model_state(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_fk_consistency_weight(
    base_weight: float, warmup_steps: int, optimizer_step: int
) -> tuple[float, float]:
    if warmup_steps <= 0:
        return float(base_weight), 1.0
    factor = min(max((int(optimizer_step) + 1) / float(warmup_steps), 0.0), 1.0)
    return float(base_weight) * factor, factor


RESUME_CONTRACT_FIELDS = (
    "resume_contract_version",
    "data_root",
    "text_root",
    "split",
    "seed",
    "max_frames",
    "min_frames",
    "batch_size",
    "lr",
    "weight_decay",
    "grad_clip",
    "amp",
    "amp_dtype",
    "ema",
    "ema_decay",
    "ema_every",
    "time_schedule",
    "denoiser_p_mean",
    "denoiser_p_std",
    "prediction_type",
    "architecture",
    "hidden_dim",
    "num_heads",
    "depth_double",
    "depth_single",
    "mlp_ratio",
    "dropout",
    "root_depth_double",
    "root_depth_single",
    "body_depth_double",
    "body_depth_single",
    "motion_stats_dir",
    "local_root_stats_dir",
    "stats_variance_eps",
    "stats_manifest_sha256",
    "asset_manifest_path",
    "asset_manifest_sha256",
    "detach_root_bridge",
    "text_encoder",
    "max_text_tokens",
    "hytext_cache_dir",
    "hytext_ctxt_dim",
    "hytext_vtxt_dim",
    "hytext_max_open_shards",
    "hytext_allow_cache_miss",
    "text_dropout_prob",
    "random_first_heading",
    "root_origin_shift",
    "control_modes",
    "training_phase",
    "control_curriculum_start_step",
    "control_curriculum_steps",
    "control_none_prob",
    "control_mixed_prob",
    "control_dense_min_fraction",
    "control_include_endpoint_rotations",
    "max_control_keyframes",
    "endpoint_preset",
    "endpoint_subset_mode",
    "endpoint_root_ref_mode",
    "self_conditioning",
    "self_cond_train_prob",
    "self_cond_mode",
    "self_cond_scale",
    "flow_loss_weight",
    "contact_loss_weight",
    "control_cont_loss_weight",
    "control_contact_loss_weight",
    "clean_cont_loss_weight",
    "clean_root_vel_loss_weight",
    "clean_joint_vel_loss_weight",
    "foot_lock_loss_weight",
    "semantic_loss_fps",
    "foot_lock_contact_threshold",
    "root_heading_loss_weight",
    "velocity_loss_weight",
    "representation_loss_space",
    "velocity_loss_t_eps",
    "source_manifest_sha256",
)

OPTIONAL_RESUME_CONTRACT_FIELDS = (
    "gradient_accumulation_steps",
    "representation_loss_mode",
    "representation_loss_scale",
    "fk_consistency_loss_weight",
    "fk_consistency_scale_m",
    "fk_consistency_warmup_steps",
    "deterministic_trace",
    "trace_seed",
)


def validate_resume_contract(
    args: argparse.Namespace,
    checkpoint_args: Any,
    path: str,
    allowed_mismatches: tuple[str, ...] = (),
    allowed_missing_fields: tuple[str, ...] = (),
) -> None:
    if not isinstance(checkpoint_args, dict):
        raise RuntimeError(f"Checkpoint is missing its resolved args contract: {path}")
    allowed_missing = set(allowed_missing_fields)
    missing = [
        field
        for field in RESUME_CONTRACT_FIELDS
        if field not in checkpoint_args and field not in allowed_missing
    ]
    fields_to_compare = RESUME_CONTRACT_FIELDS + tuple(
        field for field in OPTIONAL_RESUME_CONTRACT_FIELDS if field in checkpoint_args
    )
    allowed = set(allowed_mismatches)
    mismatches = [
        (field, checkpoint_args[field], getattr(args, field))
        for field in fields_to_compare
        if field not in allowed
        and field in checkpoint_args
        and checkpoint_args[field] != getattr(args, field)
    ]
    if missing or mismatches:
        details: list[str] = []
        if missing:
            details.append(f"missing fields={missing}")
        if mismatches:
            details.append(
                "mismatches="
                + ", ".join(
                    f"{field}: checkpoint={saved!r}, requested={requested!r}"
                    for field, saved, requested in mismatches
                )
            )
        raise RuntimeError(f"Checkpoint resume contract mismatch for {path}: {'; '.join(details)}")


def validate_ema_contract(
    model_state: dict[str, Any], ema_state: Any, path: str
) -> dict[str, Any]:
    if not isinstance(ema_state, dict):
        raise RuntimeError(f"EMA was requested but checkpoint has no EMA state: {path}")
    model_keys = set(model_state)
    ema_keys = set(ema_state)
    missing = sorted(model_keys - ema_keys)
    unexpected = sorted(ema_keys - model_keys)
    shape_mismatches = sorted(
        key
        for key in model_keys & ema_keys
        if torch.is_tensor(model_state[key])
        and torch.is_tensor(ema_state[key])
        and model_state[key].shape != ema_state[key].shape
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            f"Checkpoint EMA is incompatible for {path}: missing={missing}, "
            f"unexpected={unexpected}, shape_mismatches={shape_mismatches}"
        )
    return ema_state


def validate_normalizer_contract(
    normalizer: HY273Normalizer, checkpoint_state: Any, path: str
) -> None:
    if not isinstance(checkpoint_state, dict):
        raise RuntimeError(f"Checkpoint is missing its normalizer state: {path}")
    current = normalizer.state_dict()
    if set(current) != set(checkpoint_state):
        raise RuntimeError(
            f"Checkpoint normalizer keys mismatch for {path}: "
            f"checkpoint={sorted(checkpoint_state)}, current={sorted(current)}"
        )
    for key, value in current.items():
        saved = checkpoint_state[key]
        if not torch.is_tensor(saved) or not torch.equal(value.cpu(), saved.cpu()):
            raise RuntimeError(f"Checkpoint normalizer tensor mismatch for {path}: {key}")


@torch.no_grad()
def init_ema_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        key: value.detach().clone().float() if torch.is_floating_point(value) else value.detach().clone()
        for key, value in raw_model.state_dict().items()
    }


@torch.no_grad()
def update_ema_state(ema_state: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    raw_model = model.module if isinstance(model, DDP) else model
    for key, value in raw_model.state_dict().items():
        if key not in ema_state:
            ema_state[key] = value.detach().clone().float() if torch.is_floating_point(value) else value.detach().clone()
            continue
        target = ema_state[key]
        value = value.detach()
        if torch.is_floating_point(value):
            target.mul_(float(decay)).add_(value.to(device=target.device, dtype=target.dtype), alpha=1.0 - float(decay))
        else:
            target.copy_(value.to(device=target.device))


_MASK63 = (1 << 63) - 1


def _splitmix63(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & _MASK63


def trace_stream_seed(base_seed: int, rank: int, step: int, micro_step: int, stream: int) -> int:
    value = int(base_seed)
    value ^= int(rank) * 0xD2B74407B1CE6E93
    value ^= int(step) * 0xCA5A826395121157
    value ^= int(micro_step) * 0x9E3779B97F4A7C15
    value ^= int(stream) * 0x94D049BB133111EB
    return _splitmix63(value)


def make_trace_generator(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    step: int,
    micro_step: int,
    stream: int,
) -> torch.Generator | None:
    if not bool(getattr(args, "deterministic_trace", False)):
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(
        trace_stream_seed(
            int(getattr(args, "trace_seed", args.seed)),
            rank,
            step,
            micro_step,
            stream,
        )
    )
    return generator


def apply_deterministic_text_dropout(
    texts: list[str],
    probability: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[list[str], torch.Tensor | None, float]:
    dropped = torch.rand(len(texts), device=device, generator=generator) < float(probability)
    output = ["" if bool(dropped[i].item()) else text for i, text in enumerate(texts)]
    return output, dropped, 0.0


def training_trace_digest(
    batch: dict[str, Any],
    yaw_delta: torch.Tensor,
    timesteps: torch.Tensor,
    model_in: torch.Tensor,
    text_dropped: torch.Tensor | None,
) -> str:
    digest = hashlib.sha256()
    for key in ("dataset_indices", "crop_starts", "caption_indices", "lengths"):
        value = batch.get(key)
        if torch.is_tensor(value):
            digest.update(key.encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update("\n".join(str(item) for item in batch.get("motion_ids", [])).encode("utf-8"))
    for name, value in (("yaw", yaw_delta), ("t", timesteps), ("model_in", model_in)):
        digest.update(name.encode("ascii"))
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    if text_dropped is not None:
        digest.update(text_dropped.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def append_trace_record(
    out_dir: Path,
    rank: int,
    optimizer_step: int,
    micro_digests: list[str],
) -> None:
    trace_dir = out_dir / "logs"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"trace_rank{rank:02d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "optimizer_step": int(optimizer_step),
                    "micro_digests": list(micro_digests),
                },
                sort_keys=True,
            )
            + "\n"
        )


def make_train_state(
    *,
    next_epoch: int,
    next_step_in_epoch: int,
    optimizer_steps_per_epoch: int,
    world_size: int,
    batch_size_per_rank: int,
    gradient_accumulation_steps: int,
) -> dict[str, Any]:
    return {
        "format": "hy273_train_cursor_v1",
        "next_epoch": int(next_epoch),
        "next_step_in_epoch": int(next_step_in_epoch),
        "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
        "world_size": int(world_size),
        "batch_size_per_rank": int(batch_size_per_rank),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "effective_global_batch": int(
            world_size * batch_size_per_rank * gradient_accumulation_steps
        ),
    }


def resolve_resume_cursor(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    optimizer_steps_per_epoch: int,
    effective_global_batch: int,
) -> tuple[int, int]:
    state = checkpoint.get("train_state")
    if isinstance(state, dict):
        if state.get("format") != "hy273_train_cursor_v1":
            raise RuntimeError(f"Unsupported checkpoint train_state: {state.get('format')!r}")
        saved_steps = int(state.get("optimizer_steps_per_epoch", -1))
        saved_batch = int(state.get("effective_global_batch", -1))
        if saved_steps != int(optimizer_steps_per_epoch):
            raise RuntimeError(
                "Resume cursor is incompatible with this loader: "
                f"checkpoint optimizer_steps_per_epoch={saved_steps}, "
                f"requested={optimizer_steps_per_epoch}"
            )
        if saved_batch != int(effective_global_batch):
            raise RuntimeError(
                "Resume cursor is incompatible with this global batch: "
                f"checkpoint={saved_batch}, requested={effective_global_batch}"
            )
        return int(state["next_epoch"]), int(state["next_step_in_epoch"])

    explicit_epoch = int(getattr(args, "resume_epoch", -1))
    explicit_step = int(getattr(args, "resume_step_in_epoch", -1))
    if (explicit_epoch >= 0) != (explicit_step >= 0):
        raise RuntimeError("--resume_epoch and --resume_step_in_epoch must be provided together")
    if explicit_epoch >= 0:
        return explicit_epoch, explicit_step
    if bool(getattr(args, "require_exact_resume_cursor", False)):
        raise RuntimeError(
            "Checkpoint has no exact train_state cursor; provide explicit legacy resume cursor"
        )
    return int(checkpoint.get("epoch", 0)), 0


def average_metrics_across_ranks(
    metrics: dict[str, float], device: torch.device
) -> dict[str, float]:
    if not is_dist():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values.div_(float(dist.get_world_size()))
    return {key: float(value) for key, value in zip(keys, values.tolist())}


def compute_step_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    normalizer: HY273Normalizer,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    rank: int = 0,
    optimizer_step: int = 0,
    micro_step: int = 0,
) -> tuple[torch.Tensor, dict[str, float], str | None]:
    x0_un = batch["motion"].to(device=device, dtype=torch.float32)
    lengths = batch["lengths"].to(device=device)
    valid = batch["valid"].to(device=device)
    heading_generator = make_trace_generator(args, device, rank, optimizer_step, micro_step, stream=0)
    control_generator = make_trace_generator(args, device, rank, optimizer_step, micro_step, stream=1)
    timestep_generator = make_trace_generator(args, device, rank, optimizer_step, micro_step, stream=2)
    noise_generator = make_trace_generator(args, device, rank, optimizer_step, micro_step, stream=3)
    contact_generator = make_trace_generator(args, device, rank, optimizer_step, micro_step, stream=4)
    self_cond_generator = make_trace_generator(args, device, rank, optimizer_step, micro_step, stream=5)
    text_drop_generator = make_trace_generator(args, device, rank, optimizer_step, micro_step, stream=6)
    transform = apply_kimodo_training_transform(
        x0_un,
        random_heading=args.random_first_heading,
        root_shift=args.root_origin_shift,
        generator=heading_generator,
    )
    x0_un = transform.motion
    c_dir = transform.c_dir
    if str(getattr(args, "training_phase", "text_only")) == "control":
        curriculum_steps = max(1, int(args.control_curriculum_steps))
        # The first phase-2 update is 1/curriculum_steps and the final executed
        # update is exactly 1.0, so max_sparse_keyframes is reachable in-run.
        curriculum_progress = (
            int(optimizer_step) - int(args.control_curriculum_start_step) + 1
        ) / float(curriculum_steps)
        controls = build_kimodo_control_curriculum_batch(
            x0_un,
            lengths=lengths,
            progress=curriculum_progress,
            config=KimodoControlCurriculum(
                none_prob=float(args.control_none_prob),
                mixed_prob=float(args.control_mixed_prob),
                max_sparse_keyframes=int(args.max_control_keyframes),
                dense_min_fraction=float(args.control_dense_min_fraction),
                endpoint_preset=args.endpoint_preset,
                endpoint_subset_mode=args.endpoint_subset_mode,
                include_root_ref_for_endpoints=args.endpoint_root_ref_mode == "kimodo_hidden_root",
                include_endpoint_rotations=bool(args.control_include_endpoint_rotations),
                include_contact_pattern="contact"
                in {
                    item.strip()
                    for item in str(args.control_modes).split(",")
                    if item.strip()
                },
            ),
            generator=control_generator,
        )
    else:
        controls = build_synthetic_control_batch(
            x0_un,
            lengths=lengths,
            modes=("none",),
            endpoint_preset=args.endpoint_preset,
            endpoint_subset_mode=args.endpoint_subset_mode,
            max_keyframes=args.max_control_keyframes,
            include_root_ref_for_endpoints=args.endpoint_root_ref_mode == "kimodo_hidden_root",
            generator=control_generator,
        )
    obs_un = controls.observed_motion
    motion_mask = controls.motion_mask.to(device=device)
    x0 = normalizer.normalize(x0_un)
    obs = normalizer.normalize(obs_un)
    timesteps = sample_timesteps(
        x0.shape[0],
        device=device,
        schedule=args.time_schedule,
        p_mean=args.denoiser_p_mean,
        p_std=args.denoiser_p_std,
        generator=timestep_generator,
    ).to(dtype=x0.dtype)
    noise_cont = None
    contact_aux = None
    if bool(getattr(args, "deterministic_trace", False)):
        noise_cont = torch.randn(
            x0[..., :CONT_DIM].shape,
            device=device,
            dtype=x0.dtype,
            generator=noise_generator,
        )
        contact_aux = torch.rand(
            x0[..., CONTACT_SLICE].shape,
            device=device,
            dtype=x0.dtype,
            generator=contact_generator,
        )
    state = build_flow_state(
        x0,
        obs,
        motion_mask,
        timesteps,
        noise_cont=noise_cont,
        contact_aux=contact_aux,
    )
    model_in = state["model_in"]
    x_self_cond = None
    model_texts, text_dropped, model_text_drop_prob = apply_deterministic_text_dropout(
        list(batch["texts"]),
        float(args.text_dropout_prob),
        device,
        text_drop_generator,
    )

    use_sc = bool(args.self_conditioning) and float(args.self_cond_train_prob) > 0.0
    use_sc_this_step = False
    if use_sc:
        sc_draw = torch.rand((), device=device, generator=self_cond_generator)
        use_sc_this_step = bool(sc_draw < float(args.self_cond_train_prob))
    if use_sc_this_step:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                pred0 = model(
                    model_in,
                    t=timesteps,
                    c_dir=c_dir,
                    text=model_texts,
                    length_mask=valid,
                    x_self_cond=None,
                    text_drop_prob=model_text_drop_prob,
                )
            x0_hat0_cont = predict_clean_cont(
                state["z_cont_imp"],
                timesteps,
                pred0[..., :CONT_DIM],
                args.prediction_type,
            )
            x0_hat0 = torch.cat([x0_hat0_cont, torch.sigmoid(pred0[..., CONTACT_SLICE])], dim=-1)
            mask_f = motion_mask.to(dtype=x0.dtype)
            x_self_cond = (x0_hat0 * (1.0 - mask_f) + obs * mask_f).detach()

    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
        pred = model(
            model_in,
            t=timesteps,
            c_dir=c_dir,
            text=model_texts,
            length_mask=valid,
            x_self_cond=x_self_cond,
            text_drop_prob=model_text_drop_prob,
        )
        pred_cont = pred[..., :CONT_DIM]
        contact_logits = pred[..., CONTACT_SLICE]
        x0_hat_cont = predict_clean_cont(
            state["z_cont_imp"],
            timesteps,
            pred_cont,
            args.prediction_type,
        )
        v_pred_cont = prediction_velocity_cont(
            state["z_cont_imp"],
            timesteps,
            pred_cont,
            x0_hat_cont,
            args.prediction_type,
        )
        mask_cont = motion_mask[..., :CONT_DIM].bool()
        mask_contact = motion_mask[..., CONTACT_SLICE].bool()
        valid_cont_unmasked = valid[..., None] & (~mask_cont)
        valid_contact = valid[..., None].expand_as(contact_logits)
        cont_weights = continuous_loss_weights(args, v_pred_cont.device, v_pred_cont.dtype)
        primary_pred, primary_target, resolved_loss_space = representation_loss_pair(
            z_cont_imp=state["z_cont_imp"],
            t=timesteps,
            x0_hat_cont=x0_hat_cont,
            x0_target_cont=state["x0_cont"],
            v_pred_cont=v_pred_cont,
            v_target_cont=state["v_target_cont"],
            prediction_type=args.prediction_type,
            loss_space=str(getattr(args, "representation_loss_space", "auto")),
            velocity_t_eps=float(getattr(args, "velocity_loss_t_eps", 0.05)),
        )
        flow_loss, repr_raw, repr_weighted = representation_mse_loss(
            primary_pred,
            primary_target,
            valid_cont_unmasked,
            args,
        )
        contact_loss = bce_logits_masked(contact_logits, state["x0_contact"], valid_contact)
        control_cont = smooth_l1_masked(x0_hat_cont, obs[..., :CONT_DIM], valid[..., None] & mask_cont)
        control_contact = bce_logits_masked(contact_logits, obs[..., CONTACT_SLICE], valid[..., None] & mask_contact)
        clean_cont = (
            weighted_smooth_l1_masked(x0_hat_cont, state["x0_cont"], valid[..., None], cont_weights)
            if args.clean_cont_loss_weight > 0
            else flow_loss.detach() * 0.0
        )
        root_heading_primary = mse_masked(
            primary_pred[..., :ROOT_DIM],
            primary_target[..., :ROOT_DIM],
            valid_cont_unmasked[..., :ROOT_DIM],
        )
        velocity_channel_primary = mse_masked(
            primary_pred[..., VELOCITY_SLICE],
            primary_target[..., VELOCITY_SLICE],
            valid_cont_unmasked[..., VELOCITY_SLICE],
        )
        x0_hat = torch.cat([x0_hat_cont, torch.sigmoid(contact_logits)], dim=-1)
        x0_hat_un = normalizer.denormalize(x0_hat.float())
        semantic_losses = compute_clean_semantic_losses(
            x0_hat_un,
            x0_un.float(),
            valid,
            fps=float(args.semantic_loss_fps),
            contact_threshold=float(args.foot_lock_contact_threshold),
        )
        fk_consistency_base_weight = float(
            getattr(args, "fk_consistency_loss_weight", 0.0)
        )
        fk_consistency_weight, fk_consistency_warmup = effective_fk_consistency_weight(
            fk_consistency_base_weight,
            int(getattr(args, "fk_consistency_warmup_steps", 0)),
            optimizer_step,
        )
        if fk_consistency_weight > 0.0:
            fk_consistency, fk_consistency_cm = fk_position_consistency_loss(
                x0_hat,
                obs,
                motion_mask,
                valid,
                normalizer,
                scale_m=float(getattr(args, "fk_consistency_scale_m", 0.05)),
            )
        else:
            with torch.no_grad():
                fk_consistency, fk_consistency_cm = fk_position_consistency_loss(
                    x0_hat,
                    obs,
                    motion_mask,
                    valid,
                    normalizer,
                    scale_m=float(getattr(args, "fk_consistency_scale_m", 0.05)),
                )
        loss = (
            args.flow_loss_weight * flow_loss
            + args.contact_loss_weight * contact_loss
            + args.control_cont_loss_weight * control_cont
            + args.control_contact_loss_weight * control_contact
            + args.clean_cont_loss_weight * clean_cont
            + args.clean_root_vel_loss_weight * semantic_losses["clean_root_vel"]
            + args.clean_joint_vel_loss_weight * semantic_losses["clean_joint_vel"]
            + args.foot_lock_loss_weight * semantic_losses["foot_lock"]
            + fk_consistency_weight * fk_consistency
        )
    metrics = {
        "loss": float(loss.detach().float().item()),
        "flow": float(flow_loss.detach().float().item()),
        "contact": float(contact_loss.detach().float().item()),
        "control_cont": float(control_cont.detach().float().item()),
        "control_contact": float(control_contact.detach().float().item()),
        "clean_cont": float(clean_cont.detach().float().item()),
        "clean_root_vel": float(semantic_losses["clean_root_vel"].detach().float().item()),
        "clean_joint_vel": float(semantic_losses["clean_joint_vel"].detach().float().item()),
        "foot_lock": float(semantic_losses["foot_lock"].detach().float().item()),
        "root_heading_primary": float(root_heading_primary.detach().float().item()),
        "velocity_channel_primary": float(velocity_channel_primary.detach().float().item()),
        "fk_consistency": float(fk_consistency.detach().float().item()),
        "fk_consistency_cm": float(fk_consistency_cm.detach().float().item()),
        "fk_consistency_weighted": float(
            (fk_consistency_weight * fk_consistency).detach().float().item()
        ),
        "fk_consistency_weight": fk_consistency_weight,
        "fk_consistency_warmup": fk_consistency_warmup,
        "mask_frac": float(motion_mask.float().mean().detach().item()),
        "prediction_type_x0": float(args.prediction_type == "x0"),
        "loss_space_velocity": float(resolved_loss_space == "velocity"),
        "timestep_mean": float(timesteps.detach().float().mean().item()),
        "timestep_min": float(timesteps.detach().float().min().item()),
        "timestep_max": float(timesteps.detach().float().max().item()),
        "self_cond": float(x_self_cond is not None),
    }
    if resolved_loss_space == "velocity" and args.prediction_type == "x0":
        with torch.no_grad():
            velocity_x0_weight = (
                1.0
                / (1.0 - timesteps.detach().float())
                .clamp_min(float(args.velocity_loss_t_eps))
                .square()
            )
            metrics["velocity_x0_weight_mean"] = float(velocity_x0_weight.mean().item())
            metrics["velocity_x0_weight_max"] = float(velocity_x0_weight.max().item())
            metrics["velocity_x0_weight_scaled_mean"] = float(
                velocity_x0_weight.mean().item() * float(args.representation_loss_scale)
            )
            metrics["velocity_x0_weight_scaled_max"] = float(
                velocity_x0_weight.max().item() * float(args.representation_loss_scale)
            )
    for name, value in repr_raw.items():
        metrics[f"repr_{name}"] = float(value.detach().float().item())
        metrics[f"repr_{name}_weighted"] = float(
            repr_weighted[name].detach().float().item()
        )
    trace_digest = None
    trace_start = int(getattr(args, "expected_resume_step", -1))
    if trace_start < 0:
        trace_start = 0
    trace_offset = int(optimizer_step) - trace_start
    if (
        bool(getattr(args, "deterministic_trace", False))
        and 0 <= trace_offset < int(getattr(args, "trace_hash_steps", 0))
    ):
        trace_digest = training_trace_digest(
            batch,
            transform.yaw_delta,
            timesteps,
            model_in,
            text_dropped,
        )
    return loss, metrics, trace_digest


def main() -> None:
    parser = build_arg_parser()
    argv = sys.argv[1:]
    explicit_cli = explicit_cli_destinations(parser, argv)
    args = parser.parse_args(argv)
    cfg = load_yaml(args.config)
    args = merge_config(args, cfg, explicit_cli=explicit_cli)
    if not args.data_root:
        raise ValueError("--data_root is required")
    args.name = validate_run_name(args.name)
    validate_execution_contract(args)
    if int(args.gradient_accumulation_steps) < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    if args.representation_loss_mode == "semantic_weighted" and args.prediction_type != "x0":
        raise ValueError("semantic_weighted representation loss requires --prediction_type x0")
    if float(args.representation_loss_scale) <= 0:
        raise ValueError("--representation_loss_scale must be positive")
    if float(args.velocity_loss_t_eps) <= 0:
        raise ValueError("--velocity_loss_t_eps must be positive")
    if float(args.fk_consistency_loss_weight) < 0:
        raise ValueError("--fk_consistency_loss_weight must be non-negative")
    if int(args.fk_consistency_warmup_steps) < 0:
        raise ValueError("--fk_consistency_warmup_steps must be non-negative")
    configured_control_modes = {
        item.strip() for item in str(args.control_modes).split(",") if item.strip()
    }
    has_contact_control = "contact" in configured_control_modes
    if str(getattr(args, "training_phase", "text_only")) == "control":
        if has_contact_control and float(args.control_contact_loss_weight) <= 0.0:
            raise ValueError("contact control requires --control_contact_loss_weight > 0")
        if not has_contact_control and float(args.control_contact_loss_weight) != 0.0:
            raise ValueError(
                "--control_contact_loss_weight must be 0 when contact is not a control mode"
            )
    if args.resume_mode in {"loss_fork", "phase_transition"} and not args.resume:
        raise ValueError(f"--resume_mode {args.resume_mode} requires --resume")
    if args.resume_mode in {"loss_fork", "phase_transition"} and not str(args.resume_sha256).strip():
        raise ValueError(f"--resume_mode {args.resume_mode} requires --resume_sha256")
    if args.resume_sha256 and not args.resume:
        raise ValueError("--resume_sha256 requires --resume")
    if args.stats_manifest_sha256:
        stats_manifest = Path(args.motion_stats_dir).resolve().parent / "manifest.json"
        actual_stats_sha = sha256_file(stats_manifest)
        expected_stats_sha = str(args.stats_manifest_sha256).strip().lower()
        if actual_stats_sha != expected_stats_sha:
            raise RuntimeError(
                "Statistics manifest SHA256 mismatch: "
                f"expected={expected_stats_sha}, actual={actual_stats_sha}, path={stats_manifest}"
            )
    device, rank, world, local_rank = setup_distributed()
    if args.architecture == "redenoise_kimodo_like" and not args.asset_manifest_sha256:
        raise ValueError("redenoise_kimodo_like requires --asset_manifest_sha256")
    if args.asset_manifest_path:
        asset_result: list[dict[str, str] | None] = [None]
        if rank == 0:
            try:
                verified = verify_asset_manifest(
                    args.asset_manifest_path,
                    expected_manifest_sha256=args.asset_manifest_sha256,
                )
                asset_result[0] = {"error": "", "files": str(len(verified["files"]))}
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                asset_result[0] = {"error": str(exc), "files": "0"}
        if world > 1:
            dist.broadcast_object_list(asset_result, src=0)
        asset_status = asset_result[0]
        if not isinstance(asset_status, dict) or asset_status.get("error"):
            raise RuntimeError(
                "Training asset verification failed: "
                f"{None if asset_status is None else asset_status.get('error')}"
            )
        if rank == 0:
            print(f"[assets] verified_files={asset_status['files']}", flush=True)
    elif args.architecture == "redenoise_kimodo_like":
        raise ValueError("redenoise_kimodo_like requires --asset_manifest_path")
    if args.resume_sha256:
        resume_hash_result: list[dict[str, str] | None] = [None]
        if rank == 0:
            try:
                resume_hash_result[0] = {
                    "actual": sha256_file(args.resume),
                    "error": "",
                }
            except OSError as exc:
                resume_hash_result[0] = {"actual": "", "error": str(exc)}
        if world > 1:
            dist.broadcast_object_list(resume_hash_result, src=0)
        result = resume_hash_result[0]
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(
                f"Could not hash resume checkpoint {args.resume}: "
                f"{None if result is None else result.get('error')}"
            )
        actual_resume_sha = str(result["actual"])
        expected_resume_sha = str(args.resume_sha256).strip().lower()
        if actual_resume_sha != expected_resume_sha:
            raise RuntimeError(
                "Resume checkpoint SHA256 mismatch: "
                f"expected={expected_resume_sha}, actual={actual_resume_sha}"
            )
    seed_all(args.seed, rank)
    out_dir = Path(args.output_dir) / args.name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_resolved.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    dataset = Kimodo273TextDataset(
        args.data_root,
        split=args.split,
        text_root=args.text_root or None,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        random_crop=True,
        trace_seed=int(args.trace_seed) if args.deterministic_trace else None,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_kimodo273_text,
    )
    accumulation_steps = int(args.gradient_accumulation_steps)
    usable_micro_batches = (len(loader) // accumulation_steps) * accumulation_steps
    if usable_micro_batches <= 0:
        raise RuntimeError(
            f"DataLoader has {len(loader)} batches, fewer than accumulation={accumulation_steps}"
        )
    optimizer_steps_per_epoch = usable_micro_batches // accumulation_steps
    effective_global_batch = int(world * args.batch_size * accumulation_steps)
    normalizer = HY273Normalizer.from_data_root(
        args.data_root,
        stats_dir=args.motion_stats_dir or None,
        variance_eps=float(args.stats_variance_eps),
    ).to(device)
    model = create_model(args)
    initial_model_sha256 = ""
    if rank == 0 and not args.resume:
        initial_model_sha256 = hash_model_state(model)
        expected_initial_sha256 = str(args.expected_initial_model_sha256).strip()
        if expected_initial_sha256 and initial_model_sha256 != expected_initial_sha256:
            raise RuntimeError(
                "Initial model SHA256 mismatch: "
                f"expected={expected_initial_sha256}, actual={initial_model_sha256}"
            )
    model = model.to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    ema_state: dict[str, torch.Tensor] | None = None
    if args.resume:
        ckpt = torch.load(
            args.resume,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        allowed_mismatches: tuple[str, ...] = ()
        allowed_missing_fields: tuple[str, ...] = ()
        if args.resume_mode == "loss_fork":
            allowed_mismatches = (
                "gradient_accumulation_steps",
                "representation_loss_mode",
                "representation_loss_scale",
                "representation_loss_space",
                "velocity_loss_t_eps",
                "fk_consistency_loss_weight",
                "fk_consistency_scale_m",
                "fk_consistency_warmup_steps",
                "deterministic_trace",
                "trace_seed",
                "source_manifest_sha256",
            )
            allowed_missing_fields = (
                "resume_contract_version",
                "representation_loss_space",
                "velocity_loss_t_eps",
                "source_manifest_sha256",
            )
        elif args.resume_mode == "phase_transition":
            if str(getattr(args, "training_phase", "")) != "control":
                raise RuntimeError("phase_transition resume requires --training_phase control")
            checkpoint_step = int(ckpt.get("step", -1))
            if checkpoint_step != int(args.control_curriculum_start_step):
                raise RuntimeError(
                    "phase_transition checkpoint step must equal control_curriculum_start_step: "
                    f"checkpoint={checkpoint_step}, requested={args.control_curriculum_start_step}"
                )
            allowed_mismatches = (
                "training_phase",
                "control_modes",
                "control_cont_loss_weight",
                "control_contact_loss_weight",
                "control_dense_min_fraction",
                "control_include_endpoint_rotations",
                "endpoint_preset",
                "endpoint_subset_mode",
                "endpoint_root_ref_mode",
            )
        validate_resume_contract(
            args,
            ckpt.get("args"),
            args.resume,
            allowed_mismatches=allowed_mismatches,
            allowed_missing_fields=allowed_missing_fields,
        )
        validate_normalizer_contract(normalizer, ckpt.get("normalizer"), args.resume)
        raw_model = model.module if isinstance(model, DDP) else model
        try:
            raw_model.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint model is incompatible with the requested architecture: {args.resume}"
            ) from exc
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"Checkpoint optimizer is incompatible with the requested parameter groups: {args.resume}"
            ) from exc
        global_step = int(ckpt.get("step", 0))
        if int(args.expected_resume_step) >= 0 and global_step != int(args.expected_resume_step):
            raise RuntimeError(
                f"Expected resume step {args.expected_resume_step}, got {global_step}: {args.resume}"
            )
        if bool(args.ema):
            checkpoint_ema = validate_ema_contract(
                raw_model.state_dict(), ckpt.get("ema"), args.resume
            )
            ema_state = {
                key: value.to(device=device) if torch.is_tensor(value) else value
                for key, value in checkpoint_ema.items()
            }
        start_epoch, start_step_in_epoch = resolve_resume_cursor(
            args,
            ckpt,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            effective_global_batch=effective_global_batch,
        )
        del ckpt
        if bool(args.ema):
            del checkpoint_ema
        gc.collect()
    if not 0 <= start_step_in_epoch < optimizer_steps_per_epoch:
        raise RuntimeError(
            f"Invalid resume step_in_epoch={start_step_in_epoch}; "
            f"expected [0,{optimizer_steps_per_epoch})"
        )
    if bool(args.ema) and ema_state is None:
        ema_state = init_ema_state(model)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "fp16")
    if rank == 0 and args.deterministic_trace:
        trace_contract = {
            "format": "hy273_stateless_trace_v1",
            "trace_seed": int(args.trace_seed),
            "world_size": int(world),
            "microbatch_per_rank": int(args.batch_size),
            "gradient_accumulation_steps": accumulation_steps,
            "effective_global_batch": effective_global_batch,
            "expected_resume_step": int(args.expected_resume_step),
            "resume_epoch": int(start_epoch),
            "resume_step_in_epoch": int(start_step_in_epoch),
            "resume_sha256": str(args.resume_sha256),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "master_port": os.environ.get("MASTER_PORT", ""),
            "initial_model_sha256": initial_model_sha256,
            "source_manifest_sha256": str(args.source_manifest_sha256),
            "trace_hash_steps": int(args.trace_hash_steps),
            "streams": [
                "random_heading",
                "control",
                "timestep",
                "continuous_noise",
                "contact_aux",
                "self_conditioning",
                "text_dropout",
            ],
        }
        (out_dir / "trace_contract.json").write_text(
            json.dumps(trace_contract, indent=2, sort_keys=True)
        )

    current_train_state = make_train_state(
        next_epoch=start_epoch,
        next_step_in_epoch=start_step_in_epoch,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        world_size=world,
        batch_size_per_rank=args.batch_size,
        gradient_accumulation_steps=accumulation_steps,
    )
    try:
        for epoch in range(start_epoch, int(args.max_epochs)):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if args.deterministic_trace:
                dataset.set_trace_epoch(epoch)
            optimizer.zero_grad(set_to_none=True)
            group_metrics: dict[str, float] = {}
            micro_digests: list[str] = []
            first_batch_index = (
                start_step_in_epoch * accumulation_steps if epoch == start_epoch else 0
            )
            for batch_index, batch in enumerate(loader):
                if batch_index >= usable_micro_batches:
                    break
                if batch_index < first_batch_index:
                    continue
                micro_step = batch_index % accumulation_steps
                sync_gradients = micro_step == accumulation_steps - 1
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DDP) and not sync_gradients
                    else contextlib.nullcontext()
                )
                with sync_context:
                    loss, micro_metrics, trace_digest = compute_step_loss(
                        model,
                        batch,
                        normalizer,
                        args,
                        device,
                        amp_enabled=bool(args.amp and device.type == "cuda"),
                        amp_dtype=amp_dtype,
                        rank=rank,
                        optimizer_step=global_step,
                        micro_step=micro_step,
                    )
                    if not torch.isfinite(loss.detach()).all():
                        raise RuntimeError(
                            f"Non-finite loss at step={global_step} micro={micro_step}: {micro_metrics}"
                        )
                    backward_loss = loss / float(accumulation_steps)
                    if scaler.is_enabled():
                        scaler.scale(backward_loss).backward()
                    else:
                        backward_loss.backward()
                for key, value in micro_metrics.items():
                    group_metrics[key] = group_metrics.get(key, 0.0) + float(value)
                if trace_digest is not None:
                    micro_digests.append(trace_digest)
                if not sync_gradients:
                    continue

                grad_norm = 0.0
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
                        )
                    optimizer.step()
                metrics = {
                    key: value / float(accumulation_steps)
                    for key, value in group_metrics.items()
                }
                metrics["grad_norm_pre_clip"] = grad_norm
                metrics["grad_clip_active"] = float(
                    float(args.grad_clip) > 0.0 and grad_norm > float(args.grad_clip)
                )
                metrics["effective_global_batch"] = float(effective_global_batch)
                if micro_digests:
                    append_trace_record(out_dir, rank, global_step, micro_digests)
                if (
                    ema_state is not None
                    and int(args.ema_every) > 0
                    and global_step % int(args.ema_every) == 0
                ):
                    update_ema_state(ema_state, model, float(args.ema_decay))
                global_step += 1
                completed_step_in_epoch = batch_index // accumulation_steps + 1
                if completed_step_in_epoch >= optimizer_steps_per_epoch:
                    next_epoch = epoch + 1
                    next_step_in_epoch = 0
                else:
                    next_epoch = epoch
                    next_step_in_epoch = completed_step_in_epoch
                current_train_state = make_train_state(
                    next_epoch=next_epoch,
                    next_step_in_epoch=next_step_in_epoch,
                    optimizer_steps_per_epoch=optimizer_steps_per_epoch,
                    world_size=world,
                    batch_size_per_rank=args.batch_size,
                    gradient_accumulation_steps=accumulation_steps,
                )
                should_log = global_step == 1 or global_step % args.log_every == 0
                if should_log:
                    metrics = average_metrics_across_ranks(metrics, device)
                if rank == 0 and should_log:
                    msg = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                    print(f"[train] epoch={epoch} step={global_step} {msg}", flush=True)
                if rank == 0 and args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        out_dir / "model" / f"step_{global_step:08d}.pt",
                        model,
                        optimizer,
                        args,
                        epoch,
                        global_step,
                        ema_state=ema_state,
                        train_state=current_train_state,
                        normalizer_state=normalizer.state_dict(),
                    )
                    save_checkpoint(
                        out_dir / "model" / "latest.pt",
                        model,
                        optimizer,
                        args,
                        epoch,
                        global_step,
                        ema_state=ema_state,
                        train_state=current_train_state,
                        normalizer_state=normalizer.state_dict(),
                    )
                optimizer.zero_grad(set_to_none=True)
                group_metrics = {}
                micro_digests = []
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            start_step_in_epoch = 0
        if rank == 0 and bool(args.save_final):
            save_checkpoint(
                out_dir / "model" / "latest.pt",
                model,
                optimizer,
                args,
                int(current_train_state["next_epoch"]),
                global_step,
                ema_state=ema_state,
                train_state=current_train_state,
                normalizer_state=normalizer.state_dict(),
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
