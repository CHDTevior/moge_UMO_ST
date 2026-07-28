#!/usr/bin/env python3
"""Train a MotionMillion-style part-aware VQ-VAE on HumanML3D_272 features."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--dataset_name",
        choices=[
            "kit",
            "t2m",
            "motionmillion",
            "humanml3d_272",
            "motionfix",
            "motionfix_hml272",
            "motionfix207",
        ],
        default="humanml3d_272",
    )
    parser.add_argument("--data_root", type=Path, default=Path("dataset/HumanML3D_272"))
    parser.add_argument("--motion_dir", type=Path, default=None)
    parser.add_argument("--stats_dir", type=Path, default=None)
    parser.add_argument("--split_dir", type=Path, default=None)
    parser.add_argument("--train_split_file", type=Path, default=None)
    parser.add_argument("--val_split_file", type=Path, default=None)
    parser.add_argument(
        "--kv_root",
        type=Path,
        default=None,
        help="Optional external KV-Control checkout. Omit to use the vendored kvctrl package in this repo.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--partition_file",
        type=Path,
        default=Path("configs/humanml3d_272_skeleton_partition_pscf_nooverlap.json"),
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--window_size",
        type=int,
        default=0,
        help="0 selects the dataset default: MotionFix uses 64, other datasets use 96.",
    )
    parser.add_argument("--sampling_mode", choices=["all_windows", "random_crop"], default="all_windows")
    parser.add_argument("--min_motion_length", type=int, default=0)
    parser.add_argument("--max_motion_length", type=int, default=0)
    parser.add_argument("--index_path", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--motionfix_train_manifest", type=Path, default=None)
    parser.add_argument("--motionfix_val_manifest", type=Path, default=None)
    parser.add_argument("--motionfix_test_manifest", type=Path, default=None)
    parser.add_argument("--motionfix_repo", type=Path, default=Path("/mnt/afs/motionfix"))
    parser.add_argument("--motionfix_official_data_root", type=Path, default=Path("/mnt/afs/motionfix/data/motionfix-dataset"))
    parser.add_argument("--motionfix_eval_retrieval_batch_size", type=int, default=32)

    parser.add_argument("--total_iter", type=int, default=300000)
    parser.add_argument("--max_epoch", type=int, default=0, help="If >0, override total_iter as max_epoch * len(train_loader).")
    parser.add_argument("--warm_up_iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--milestones", nargs="+", type=int, default=[200000, 1000000, 1800000])
    parser.add_argument(
        "--scale_motionmillion_milestones",
        action="store_true",
        help="Scale the MotionMillion reference milestones to this run's resolved total_iter.",
    )
    parser.add_argument("--motionmillion_reference_total_iter", type=int, default=2385000)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="If >0, clip gradient norm before optimizer.step and log the unclipped norm.",
    )
    parser.add_argument("--commit", type=float, default=0.02)
    parser.add_argument("--loss_vel", type=float, default=0.5)
    parser.add_argument("--recons_loss", choices=["l1", "l1_smooth"], default="l1_smooth")

    parser.add_argument("--code_dim", type=int, default=128)
    parser.add_argument("--nb_code", type=int, default=1024)
    parser.add_argument("--mu", type=float, default=0.99)
    parser.add_argument("--down_t", type=int, default=2)
    parser.add_argument("--stride_t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation_growth_rate", type=int, default=3)
    parser.add_argument("--output_emb_width", type=int, default=128)
    parser.add_argument("--vq_act", choices=["relu", "silu", "gelu"], default="relu")
    parser.add_argument("--vq_norm", default=None)
    parser.add_argument("--quantizer", choices=["ema_reset", "orig", "ema", "reset"], default="ema_reset")
    parser.add_argument(
        "--ddp_codebook_sync",
        choices=["sum", "mean", "rank0"],
        default="sum",
        help=(
            "How EMA/reset quantizers synchronize codebook batch statistics under DDP. "
            "sum uses global batch stats; mean preserves per-rank EMA scale; rank0 emulates single-rank codebook updates."
        ),
    )

    parser.add_argument("--feat_bias", type=float, default=5.0)
    parser.add_argument(
        "--stat_bias_passes",
        type=int,
        default=2,
        help="Number of times to apply feat_bias to root/contact std. 2 matches the existing KIT VQ/evaluator metadata.",
    )
    parser.add_argument("--print_iter", type=int, default=200)
    parser.add_argument("--eval_iter", type=int, default=5000)
    parser.add_argument("--eval_every_epoch", action="store_true")
    parser.add_argument("--save_latest", type=int, default=500)
    parser.add_argument(
        "--mpjpe_eval_only",
        action="store_true",
        help="For MotionFix-style reconstruction, skip retrieval/FID evaluators and compute only reconstruction MPJPE.",
    )
    parser.add_argument(
        "--save_only_last_and_best_mpjpe",
        action="store_true",
        help="Only keep latest.tar and net_best_mpjpe.tar aliases; do not save val/FID/R@k best checkpoints.",
    )
    parser.add_argument(
        "--hml272_rebase_crop_start",
        action="store_true",
        help="For MotionFix HML272 crops, reset first-frame root velocity, heading delta, and local velocity.",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument(
        "--official_eval_every_epoch",
        type=int,
        default=0,
        help="MotionMillion only: run official 272-RPR reconstruction FID/top3 every N completed epochs.",
    )
    parser.add_argument("--official_eval_start_epoch", type=int, default=0)
    parser.add_argument("--official_eval_python", type=Path, default=Path("/root/miniconda3/envs/mogo/bin/python"))
    parser.add_argument("--official_eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--official_eval_batch_size", type=int, default=32)
    parser.add_argument("--official_eval_num_workers", type=int, default=4)
    parser.add_argument("--official_eval_max_batches", type=int, default=0)
    parser.add_argument("--hymotion_root", type=Path, default=Path("/mnt/afs/HY-Motion-1.0"))
    parser.add_argument(
        "--motionstreamer_evaluator_checkpoint",
        type=Path,
        default=Path("checkpoints/evaluators/motionstreamer/Evaluator_272/epoch=99_state_dict.pt"),
    )
    parser.add_argument(
        "--motionstreamer_distilbert_path",
        type=Path,
        default=Path("checkpoints/evaluators/distilbert-base-uncased"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume_budget_mode",
        choices=["auto", "epoch", "step"],
        default="auto",
        help=(
            "How to resolve --max_epoch when resuming with a different world size. "
            "auto/epoch preserve dataset-pass progress using the checkpoint's old iters/epoch; "
            "step preserves the current step budget."
        ),
    )
    parser.add_argument(
        "--resume_ema_code_count",
        type=float,
        default=1024.0,
        help=(
            "Fallback EMA history count for old ema_reset checkpoints that lack "
            "quantizer_runtime_state. Larger values preserve the loaded codebook "
            "instead of resetting rarely hit codes on the first resumed batches."
        ),
    )
    parser.add_argument(
        "--freeze_codebook_updates",
        action="store_true",
        help="Keep EMA/reset VQ codebooks fixed during training; useful when resuming old checkpoints without EMA state.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()
    if int(args.window_size) <= 0:
        args.window_size = 64 if args.dataset_name in {"motionfix", "motionfix_hml272", "motionfix207"} else 96
    return args


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, torch.device]:
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is None:
        if not torch.cuda.is_available() and args.gpu_id >= 0:
            raise RuntimeError("CUDA is not available in this process")
        device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return False, 0, 1, 0, device

    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA")
    local_rank = int(local_rank_env)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank, torch.device("cuda", local_rank)


def dist_barrier() -> None:
    if not is_dist():
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def cleanup_distributed() -> None:
    if is_dist():
        dist_barrier()
        dist.destroy_process_group()


def master_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_paths(args: argparse.Namespace, rank: int = 0) -> dict[str, Path]:
    paths = {
        "root": args.output_dir,
        "model": args.output_dir / "model",
        "meta": args.output_dir / "meta",
        "stats": args.output_dir / "stats",
        "logs": args.output_dir / "logs",
        "config": args.output_dir / "config",
    }
    if rank == 0:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    dist_barrier()
    return paths


def dataset_shape(dataset_name: str) -> tuple[int, int]:
    if dataset_name == "kit":
        return 251, 21
    if dataset_name in {"t2m", "motionmillion", "humanml3d_272", "motionfix", "motionfix_hml272"}:
        return 272, 22
    if dataset_name == "motionfix207":
        return 207, 22
    raise ValueError(f"Unknown dataset: {dataset_name}")


def install_numpy_pickle_compat() -> None:
    """Read NumPy 2.x object arrays from this NumPy 1.x training env."""
    try:
        import numpy.core as np_core
        import numpy.core.multiarray as np_multiarray
        import numpy.core.numeric as np_numeric
    except Exception:
        return
    sys.modules.setdefault("numpy._core", np_core)
    sys.modules.setdefault("numpy._core.multiarray", np_multiarray)
    sys.modules.setdefault("numpy._core.numeric", np_numeric)


def local_position_slice(dataset_name: str, joints_num: int) -> slice:
    if dataset_name == "motionfix207":
        return slice(141, 141 + joints_num * 3)
    if dataset_name in {"t2m", "motionmillion", "humanml3d_272", "motionfix", "motionfix_hml272"}:
        return slice(8, 8 + joints_num * 3)
    return slice(4, 4 + (joints_num - 1) * 3)


def resolve_motion_dir(args: argparse.Namespace) -> Path:
    if args.motion_dir is not None:
        return args.motion_dir
    if args.dataset_name == "motionfix":
        return args.data_root / "motionstreamer272_unified_v2_joint_vecs"
    if args.dataset_name == "motionfix_hml272":
        return args.data_root / "motionstreamer272_hml_joint_vecs"
    if args.dataset_name == "motionfix207":
        return args.data_root / "motionfix207_joint_vecs"
    if args.dataset_name == "motionmillion":
        return args.data_root / "motion_272rpr"
    return args.data_root / "new_joint_vecs"


def resolve_stats_dir(args: argparse.Namespace) -> Path:
    if args.stats_dir is not None:
        return args.stats_dir
    if args.dataset_name == "motionfix":
        return args.data_root / "stats" / "motionstreamer272_unified_v2_source_target_train"
    if args.dataset_name == "motionfix_hml272":
        return args.data_root / "stats" / "motionstreamer272_hml_source_target_train"
    if args.dataset_name == "motionfix207":
        return args.data_root / "stats" / "motionfix207_source_target_train"
    if args.dataset_name == "motionmillion":
        return args.data_root / "mean_std"
    return args.data_root


def resolve_split_file(args: argparse.Namespace, split: str) -> Path:
    override = args.train_split_file if split == "train" else args.val_split_file
    if override is not None:
        return override
    if args.split_dir is not None:
        return args.split_dir / f"{split}.txt"
    if args.dataset_name == "motionmillion":
        split_name = "train.txt" if split == "train" else "val.txt"
        canonical = args.data_root / "split" / "version1" / "t2m_60_300" / split_name
        if canonical.exists():
            return canonical
        return args.data_root / split_name
    return args.data_root / f"{split}.txt"


def resolve_motionfix_manifest(args: argparse.Namespace, split: str) -> Path:
    if split == "train" and args.motionfix_train_manifest is not None:
        return args.motionfix_train_manifest
    if split == "val" and args.motionfix_val_manifest is not None:
        return args.motionfix_val_manifest
    if split == "test" and args.motionfix_test_manifest is not None:
        return args.motionfix_test_manifest
    if args.dataset_name == "motionfix207":
        prefix = "motionfix_motionfix207"
    elif args.dataset_name == "motionfix_hml272":
        prefix = "motionfix_motionstreamer272_hml"
    else:
        prefix = "motionfix_motionstreamer272_unified_v2"
    return args.data_root / "manifests" / f"{prefix}_{split}.jsonl"


def load_motionfix_manifest_entries(
    manifest_path: Path,
    data_root: Path,
    *,
    paired: bool = False,
    feature_dim: int = 272,
) -> list[dict[str, Any]]:
    from data.t2m_dataset import _first_present, _load_motion_edit_manifest, _resolve_motion_path

    manifest = manifest_path.expanduser().resolve()
    manifest_dir = manifest.parent
    root_path = data_root.expanduser().resolve()
    raw_records = _load_motion_edit_manifest(manifest)
    entries: list[dict[str, Any]] = []
    for raw in raw_records:
        source_value = _first_present(raw, ("source_motion", "source_path", "source", "src_motion", "src_path", "src"))
        target_value = _first_present(raw, ("target_motion", "target_path", "target", "tgt_motion", "tgt_path", "tgt"))
        if source_value is None or target_value is None:
            continue
        sample_id = _first_present(raw, ("id", "keyid", "sample_id", "motion_id", "uid"))
        source_len_value = raw.get("source_len")
        target_len_value = raw.get("target_len")
        feature_dim_value = raw.get("feature_dim")
        if source_len_value is not None and target_len_value is not None and int(feature_dim_value or feature_dim) == int(feature_dim):
            source_path = Path(str(source_value)).expanduser()
            target_path = Path(str(target_value)).expanduser()
            if not source_path.is_absolute():
                source_path = root_path / source_path
            if not target_path.is_absolute():
                target_path = root_path / target_path
            source_len = int(source_len_value)
            target_len = int(target_len_value)
        else:
            source_path = _resolve_motion_path(source_value, root_path, manifest_dir)
            target_path = _resolve_motion_path(target_value, root_path, manifest_dir)
            source_shape = np.load(source_path, mmap_mode="r").shape
            target_shape = np.load(target_path, mmap_mode="r").shape
            if (
                len(source_shape) != 2
                or len(target_shape) != 2
                or source_shape[1] != int(feature_dim)
                or target_shape[1] != int(feature_dim)
            ):
                continue
            source_len = int(source_shape[0])
            target_len = int(target_shape[0])
        sample_id = str(sample_id or source_path.stem.replace("_source", "").replace("_target", ""))
        instruction = str(_first_present(raw, ("instruction", "edit_instruction", "edit", "text", "caption", "prompt")) or "")
        conversion_version = str(raw.get("conversion_version") or "motionfix_motionstreamer272")
        if paired:
            entries.append(
                {
                    "id": sample_id,
                    "instruction": instruction,
                    "source_path": source_path,
                    "target_path": target_path,
                    "source_len": source_len,
                    "target_len": target_len,
                    "conversion_version": conversion_version,
                }
            )
        else:
            entries.append(
                {
                    "id": sample_id,
                    "side": "source",
                    "path": source_path,
                    "length": source_len,
                    "conversion_version": conversion_version,
                }
            )
            entries.append(
                {
                    "id": sample_id,
                    "side": "target",
                    "path": target_path,
                    "length": target_len,
                    "conversion_version": conversion_version,
                }
            )
    if not entries:
        raise RuntimeError(f"No MotionFix 272 entries loaded from {manifest}")
    return entries


def compute_motionfix_train_stats(
    manifest_path: Path,
    data_root: Path,
    stats_dir: Path,
    feature_dim: int = 272,
    dataset_name: str = "motionfix",
) -> tuple[np.ndarray, np.ndarray]:
    entries = load_motionfix_manifest_entries(manifest_path, data_root, paired=False, feature_dim=feature_dim)
    total = 0
    sum_vec = np.zeros((int(feature_dim),), dtype=np.float64)
    sum_sq_vec = np.zeros((int(feature_dim),), dtype=np.float64)
    for entry in entries:
        motion = np.load(entry["path"], mmap_mode="r")
        if motion.ndim != 2 or motion.shape[1] != int(feature_dim):
            raise ValueError(f"Expected [T,{feature_dim}] MotionFix motion at {entry['path']}, got {motion.shape}")
        arr = np.asarray(motion, dtype=np.float64)
        sum_vec += arr.sum(axis=0)
        sum_sq_vec += np.square(arr).sum(axis=0)
        total += int(arr.shape[0])
    if total <= 0:
        raise RuntimeError(f"Cannot compute MotionFix stats from empty manifest: {manifest_path}")
    mean = (sum_vec / float(total)).astype(np.float32)
    var = np.maximum(sum_sq_vec / float(total) - np.square(sum_vec / float(total)), 1e-12)
    std = np.sqrt(var).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "Mean.npy", mean)
    np.save(stats_dir / "Std.npy", std)
    with (stats_dir / "stats_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(dataset_name),
                "manifest": str(manifest_path),
                "num_motions": len(entries),
                "num_frames": total,
                "feature_dim": int(feature_dim),
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    return mean, std


def _normalize_np(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = _normalize_np(a1)
    b2 = _normalize_np(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-2).astype(np.float32)


def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6).astype(np.float32)


def _motionfix_accumulate_heading_root(motion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover MotionFix unified_v2 absolute heading/root state in source-canonical coordinates."""
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 272:
        raise ValueError(f"Expected MotionFix unified_v2 motion [T,272], got {motion.shape}")
    frames = int(motion.shape[0])
    if frames <= 0:
        raise ValueError("Cannot accumulate heading/root for an empty MotionFix motion")

    heading_delta = rotation_6d_to_matrix_np(motion[:, 2:8])
    yaw_rel = np.empty((frames, 3, 3), dtype=np.float32)
    yaw_rel[0] = heading_delta[0]
    for frame in range(1, frames):
        yaw_rel[frame] = heading_delta[frame] @ yaw_rel[frame - 1]

    local_vel = np.zeros((frames, 3), dtype=np.float32)
    local_vel[:, 0] = motion[:, 0]
    local_vel[:, 2] = motion[:, 1]
    canonical_root = np.zeros((frames, 3), dtype=np.float32)
    canonical_root[:, 1] = motion[:, 9]
    canonical_root[0, 0] = local_vel[0, 0]
    canonical_root[0, 2] = local_vel[0, 2]
    for frame in range(1, frames):
        canonical_delta = yaw_rel[frame - 1].T @ local_vel[frame]
        canonical_root[frame, 0] = canonical_root[frame - 1, 0] + canonical_delta[0]
        canonical_root[frame, 2] = canonical_root[frame - 1, 2] + canonical_delta[2]
    return yaw_rel, canonical_root


def rebase_motionfix_272_window(motion: np.ndarray, start: int, window_size: int) -> np.ndarray:
    """Return a self-contained MotionFix unified_v2 window.

    In unified_v2, frame 0 stores absolute root/heading state relative to the
    pair source frame, while later frames store deltas. A raw middle crop would
    therefore make the first cropped frame look like a small delta to identity.
    This preserves the original global source-canonical state at the crop start.
    """
    motion = np.asarray(motion, dtype=np.float32)
    start = int(start)
    end = start + int(window_size)
    if start < 0 or end > int(motion.shape[0]):
        raise IndexError(f"Invalid MotionFix window start={start} end={end} length={motion.shape[0]}")
    window = np.array(motion[start:end], dtype=np.float32, copy=True)
    if window.shape[0] <= 0:
        raise RuntimeError("MotionFix crop produced an empty window")

    yaw_rel, canonical_root = _motionfix_accumulate_heading_root(motion[:end])
    window[0, 0] = canonical_root[start, 0]
    window[0, 1] = canonical_root[start, 2]
    window[0, 2:8] = matrix_to_rotation_6d_np(yaw_rel[start])
    window[0, 74:140] = 0.0
    return window


def rebase_motionfix207_window(motion: np.ndarray, start: int, window_size: int) -> np.ndarray:
    """Return a self-contained MotionFix 207D window.

    MotionFix 207D stores root translation and yaw as frame-to-frame deltas.
    A middle crop should start with zero translation delta and identity yaw delta,
    matching the full-sequence convention at frame 0.
    """
    motion = np.asarray(motion, dtype=np.float32)
    start = int(start)
    end = start + int(window_size)
    if start < 0 or end > int(motion.shape[0]):
        raise IndexError(f"Invalid MotionFix207 window start={start} end={end} length={motion.shape[0]}")
    window = np.array(motion[start:end], dtype=np.float32, copy=True)
    if window.shape[0] <= 0:
        raise RuntimeError("MotionFix207 crop produced an empty window")
    window[0, 0:3] = 0.0
    window[0, 9:15] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return window


def rebase_motionstreamer272_hml_window(motion: np.ndarray, start: int, window_size: int) -> np.ndarray:
    """Return a HML272 crop with first-frame delta channels reset."""
    motion = np.asarray(motion, dtype=np.float32)
    start = int(start)
    end = start + int(window_size)
    if start < 0 or end > int(motion.shape[0]):
        raise IndexError(f"Invalid HML272 window start={start} end={end} length={motion.shape[0]}")
    window = np.array(motion[start:end], dtype=np.float32, copy=True)
    if window.shape[0] <= 0:
        raise RuntimeError("HML272 crop produced an empty window")
    window[0, 0:2] = 0.0
    window[0, 2:8] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    window[0, 74:140] = 0.0
    return window


def apply_feat_bias(std: np.ndarray, joints_num: int, feat_bias: float, passes: int) -> np.ndarray:
    std = std.copy()
    for _ in range(passes):
        std[0:1] = std[0:1] / feat_bias
        std[1:3] = std[1:3] / feat_bias
        std[3:4] = std[3:4] / feat_bias
        std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
        std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = (
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] / 1.0
        )
        std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = (
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] / 1.0
        )
        std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] / feat_bias
        )
    return std


def load_stats(args: argparse.Namespace, paths: dict[str, Path], rank: int = 0) -> tuple[np.ndarray, np.ndarray]:
    dim_pose, joints_num = dataset_shape(args.dataset_name)
    stats_dir = resolve_stats_dir(args)
    if args.dataset_name in {"motionfix", "motionfix_hml272", "motionfix207"}:
        if rank == 0:
            mean_path = stats_dir / "Mean.npy"
            std_path = stats_dir / "Std.npy"
            if not mean_path.is_file() or not std_path.is_file():
                compute_motionfix_train_stats(
                    resolve_motionfix_manifest(args, "train"),
                    args.data_root,
                    stats_dir,
                    feature_dim=dim_pose,
                    dataset_name=args.dataset_name,
                )
        dist_barrier()
    mean = np.load(stats_dir / "Mean.npy").astype(np.float32)
    std_raw = np.load(stats_dir / "Std.npy").astype(np.float32)

    expected_dim = dim_pose
    if args.dataset_name == "kit":
        expected_dim = 4 + (joints_num - 1) * 9 + joints_num * 3 + 4
    if mean.shape[-1] != dim_pose or std_raw.shape[-1] != dim_pose or expected_dim != dim_pose:
        raise ValueError(
            f"Unexpected feature dim: mean={mean.shape}, std={std_raw.shape}, expected={dim_pose}"
        )

    stat_bias_passes = (
        0
        if args.dataset_name
        in {"motionmillion", "humanml3d_272", "t2m", "motionfix", "motionfix_hml272", "motionfix207"}
        else args.stat_bias_passes
    )
    std = apply_feat_bias(std_raw, joints_num, args.feat_bias, stat_bias_passes).astype(np.float32)
    if rank == 0:
        for out_dir in (paths["meta"], paths["stats"]):
            np.save(out_dir / "mean.npy", mean)
            np.save(out_dir / "std.npy", std)
    dist_barrier()
    return mean, std


class FixedWindowMotionDataset(Dataset):
    def __init__(
        self,
        motion_dir: Path,
        split_path: Path,
        window_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        sampling_mode: str = "all_windows",
        min_motion_length: int = 0,
        max_motion_length: int = 0,
        index_path: Path | None = None,
    ) -> None:
        self.motion_dir = motion_dir
        self.window_size = window_size
        self.mean = mean
        self.std = std
        self.sampling_mode = sampling_mode
        self.lazy = sampling_mode == "random_crop"
        self.data: list[np.ndarray] | None = [] if not self.lazy else None
        self.names: list[str] = []
        self.lengths: list[int] = []

        ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        min_len = max(int(min_motion_length), int(window_size))
        max_len = int(max_motion_length) if int(max_motion_length) > 0 else 0
        if index_path is not None and index_path.exists():
            install_numpy_pickle_compat()
            with np.load(index_path, allow_pickle=True) as data:
                ids = [str(name) for name in data["name_list"]]
                indexed_lengths = [int(x) for x in data["length_arr"]]
            if self.sampling_mode == "random_crop":
                self.names = ids
                self.lengths = indexed_lengths
                snippets = len(self.names)
                print(
                    f"[dataset:{split_path.name}] motions={len(self.names)} snippets={snippets} "
                    f"missing=0 short=0 long=0 sampling_mode={self.sampling_mode} index={index_path}"
                )
                if snippets <= 0:
                    raise RuntimeError(f"No snippets loaded for split {split_path}")
                self.cumsum = None
                return
        else:
            indexed_lengths = []
        missing = 0
        short = 0
        long = 0
        for row, name in enumerate(ids):
            motion_path = self.motion_dir / f"{name}.npy"
            if not motion_path.exists():
                missing += 1
                continue
            if indexed_lengths:
                seq_len = indexed_lengths[row]
            else:
                motion_ref = np.load(motion_path, mmap_mode="r")
                seq_len = int(motion_ref.shape[0])
                del motion_ref
            if seq_len < min_len:
                short += 1
                continue
            if max_len > 0 and seq_len > max_len:
                long += 1
                continue
            self.names.append(name)
            if self.sampling_mode == "random_crop":
                self.lengths.append(seq_len)
            else:
                self.lengths.append(seq_len - window_size)
                motion = np.load(motion_path).astype(np.float32)
                assert self.data is not None
                self.data.append(motion)

        self.cumsum = None if self.sampling_mode == "random_crop" else np.cumsum([0] + self.lengths)
        snippets = len(self.names) if self.cumsum is None else int(self.cumsum[-1])
        print(
            f"[dataset:{split_path.name}] motions={len(self.names)} snippets={snippets} "
            f"missing={missing} short={short} long={long} sampling_mode={self.sampling_mode}"
        )
        if snippets <= 0:
            raise RuntimeError(f"No snippets loaded for split {split_path}")

    def __len__(self) -> int:
        if self.sampling_mode == "random_crop":
            return len(self.names)
        return int(self.cumsum[-1])

    def __getitem__(self, item: int) -> torch.Tensor:
        if self.sampling_mode == "random_crop":
            motion_id = int(item)
            motion = np.load(self.motion_dir / f"{self.names[motion_id]}.npy", mmap_mode="r")
            max_offset = int(motion.shape[0]) - int(self.window_size)
            idx = 0 if max_offset <= 0 else random.randint(0, max_offset)
            motion = np.asarray(motion[idx:idx + self.window_size], dtype=np.float32)
        else:
            if item != 0:
                motion_id = int(np.searchsorted(self.cumsum, item) - 1)
                idx = int(item - self.cumsum[motion_id] - 1)
            else:
                motion_id = 0
                idx = 0
            assert self.data is not None
            motion = self.data[motion_id][idx:idx + self.window_size]
        motion = (motion - self.mean) / self.std
        return torch.from_numpy(motion.astype(np.float32))


class MotionFixWindowDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        window_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        sampling_mode: str = "random_crop",
        min_motion_length: int = 0,
        max_motion_length: int = 0,
        feature_dim: int = 272,
        hml272_rebase_crop_start: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.data_root = data_root.expanduser().resolve()
        self.window_size = int(window_size)
        self.mean = mean
        self.std = std
        self.feature_dim = int(feature_dim)
        self.hml272_rebase_crop_start = bool(hml272_rebase_crop_start)
        self.sampling_mode = sampling_mode
        self.lazy = sampling_mode == "random_crop"
        self.data: list[np.ndarray] | None = [] if not self.lazy else None
        self.entries: list[dict[str, Any]] = []
        self.lengths: list[int] = []

        min_len = max(int(min_motion_length), self.window_size)
        max_len = int(max_motion_length) if int(max_motion_length) > 0 else 0
        raw_entries = load_motionfix_manifest_entries(
            self.manifest_path,
            self.data_root,
            paired=False,
            feature_dim=self.feature_dim,
        )
        short = 0
        long = 0
        for entry in raw_entries:
            seq_len = int(entry["length"])
            if seq_len < min_len:
                short += 1
                continue
            if max_len > 0 and seq_len > max_len:
                long += 1
                continue
            self.entries.append(entry)
            if self.sampling_mode == "random_crop":
                self.lengths.append(seq_len)
            else:
                self.lengths.append(max(1, seq_len - self.window_size + 1))
                motion = np.load(entry["path"]).astype(np.float32)
                assert self.data is not None
                self.data.append(motion)
        self.cumsum = None if self.sampling_mode == "random_crop" else np.cumsum([0] + self.lengths)
        snippets = len(self.entries) if self.cumsum is None else int(self.cumsum[-1])
        print(
            f"[motionfix:{self.manifest_path.name}] motions={len(self.entries)} snippets={snippets} "
            f"short={short} long={long} sampling_mode={self.sampling_mode} window={self.window_size}"
        )
        if snippets <= 0:
            raise RuntimeError(f"No MotionFix snippets loaded for {self.manifest_path}")

    def __len__(self) -> int:
        if self.sampling_mode == "random_crop":
            return len(self.entries)
        return int(self.cumsum[-1])

    def __getitem__(self, item: int) -> torch.Tensor:
        if self.sampling_mode == "random_crop":
            entry = self.entries[int(item)]
            motion = np.load(entry["path"], mmap_mode="r")
            max_offset = int(motion.shape[0]) - self.window_size
            idx = 0 if max_offset <= 0 else random.randint(0, max_offset)
            if entry.get("conversion_version") == "motionfix_motionstreamer272_unified_v2":
                motion_np = rebase_motionfix_272_window(motion, idx, self.window_size)
            elif (
                self.hml272_rebase_crop_start
                and entry.get("conversion_version") == "motionfix_motionstreamer272_hml"
            ):
                motion_np = rebase_motionstreamer272_hml_window(motion, idx, self.window_size)
            elif entry.get("conversion_version") == "motionfix207":
                motion_np = rebase_motionfix207_window(motion, idx, self.window_size)
            else:
                motion_np = np.asarray(motion[idx:idx + self.window_size], dtype=np.float32)
        else:
            motion_id = int(np.searchsorted(self.cumsum, item, side="right") - 1)
            idx = int(item - self.cumsum[motion_id])
            assert self.data is not None
            entry = self.entries[motion_id]
            if entry.get("conversion_version") == "motionfix_motionstreamer272_unified_v2":
                motion_np = rebase_motionfix_272_window(self.data[motion_id], idx, self.window_size)
            elif (
                self.hml272_rebase_crop_start
                and entry.get("conversion_version") == "motionfix_motionstreamer272_hml"
            ):
                motion_np = rebase_motionstreamer272_hml_window(self.data[motion_id], idx, self.window_size)
            elif entry.get("conversion_version") == "motionfix207":
                motion_np = rebase_motionfix207_window(self.data[motion_id], idx, self.window_size)
            else:
                motion_np = np.asarray(self.data[motion_id][idx:idx + self.window_size], dtype=np.float32)
        motion_np = (motion_np - self.mean) / self.std
        return torch.from_numpy(motion_np.astype(np.float32))


class MotionFixReconEvalDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        unit_length: int = 4,
        max_motion_length: int = 300,
        max_samples: int = 0,
        feature_dim: int = 272,
    ) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.data_root = data_root.expanduser().resolve()
        self.unit_length = max(1, int(unit_length))
        self.max_motion_length = int(max_motion_length) if int(max_motion_length) > 0 else 300
        self.feature_dim = int(feature_dim)
        self.entries = load_motionfix_manifest_entries(
            self.manifest_path,
            self.data_root,
            paired=True,
            feature_dim=self.feature_dim,
        )
        if int(max_samples) > 0:
            self.entries = self.entries[: int(max_samples)]
        if not self.entries:
            raise RuntimeError(f"No MotionFix eval pairs loaded for {self.manifest_path}")
        print(f"[motionfix_eval:{self.manifest_path.name}] pairs={len(self.entries)}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, item: int) -> dict[str, Any]:
        entry = self.entries[int(item)]
        source = np.load(entry["source_path"]).astype(np.float32)
        target = np.load(entry["target_path"]).astype(np.float32)
        source_len = (min(int(source.shape[0]), self.max_motion_length) // self.unit_length) * self.unit_length
        target_len = (min(int(target.shape[0]), self.max_motion_length) // self.unit_length) * self.unit_length
        if source_len <= 0 or target_len <= 0:
            raise RuntimeError(f"MotionFix eval sample {entry['id']} became empty after unit alignment")
        common_len = min(source_len, target_len)
        return {
            "sample_id": str(entry["id"]),
            "instruction": str(entry["instruction"]),
            "source": source[:source_len],
            "target": target[:target_len],
            "source_length": int(source_len),
            "target_length": int(target_len),
            "length": int(common_len),
        }


def motionfix_recon_eval_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(max(int(item["source_length"]), int(item["target_length"])) for item in batch)
    feature_dim = int(batch[0]["source"].shape[-1])
    source = np.zeros((len(batch), max_len, feature_dim), dtype=np.float32)
    target = np.zeros((len(batch), max_len, feature_dim), dtype=np.float32)
    lengths = np.zeros((len(batch),), dtype=np.int64)
    source_lengths = np.zeros((len(batch),), dtype=np.int64)
    target_lengths = np.zeros((len(batch),), dtype=np.int64)
    sample_ids: list[str] = []
    instructions: list[str] = []
    for idx, item in enumerate(batch):
        length = int(item["length"])
        source_length = int(item["source_length"])
        target_length = int(item["target_length"])
        source[idx, :source_length] = item["source"]
        target[idx, :target_length] = item["target"]
        lengths[idx] = length
        source_lengths[idx] = source_length
        target_lengths[idx] = target_length
        sample_ids.append(str(item["sample_id"]))
        instructions.append(str(item["instruction"]))
    return {
        "source": torch.from_numpy(source),
        "target": torch.from_numpy(target),
        "length": torch.from_numpy(lengths),
        "source_length": torch.from_numpy(source_lengths),
        "target_length": torch.from_numpy(target_lengths),
        "sample_id": sample_ids,
        "instruction": instructions,
    }


def build_model(args: argparse.Namespace, device: torch.device):
    if args.kv_root is not None:
        sys.path.insert(0, str(args.kv_root))
    from kvctrl.models.vqvae import HumanVQVAE

    dim_pose, _ = dataset_shape(args.dataset_name)
    dataname = (
        "t2m"
        if args.dataset_name in {"humanml3d_272", "motionfix", "motionfix_hml272", "motionfix207"}
        else args.dataset_name
    )
    model_args = SimpleNamespace(
        dataname=dataname,
        input_dim=dim_pose,
        quantizer=args.quantizer,
        mu=args.mu,
        ddp_codebook_sync=args.ddp_codebook_sync,
        partition_file=str(args.partition_file.resolve()) if args.partition_file else None,
    )
    model = HumanVQVAE(
        model_args,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm,
    ).to(device)
    if getattr(args, "freeze_codebook_updates", False):
        for module in model.modules():
            if hasattr(module, "codebook"):
                module.freeze_codebook_updates = True
    return model


def reconstruction_loss(kind: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if kind == "l1":
        return F.l1_loss(pred, target)
    if kind == "l1_smooth":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(kind)


def compute_losses(args: argparse.Namespace, model, motions: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
    _, joints_num = dataset_shape(args.dataset_name)
    pred_motion, loss_commit, perplexity = model(motions)
    loss_rec = reconstruction_loss(args.recons_loss, pred_motion, motions)
    pos_slice = local_position_slice(args.dataset_name, joints_num)
    loss_explicit = reconstruction_loss(args.recons_loss, pred_motion[..., pos_slice], motions[..., pos_slice])
    loss = loss_rec + args.loss_vel * loss_explicit + args.commit * loss_commit
    return OrderedDict(
        loss=loss,
        loss_rec=loss_rec,
        loss_explicit=loss_explicit,
        loss_commit=loss_commit,
        perplexity=perplexity,
    )


def lr_at_step(args: argparse.Namespace, step: int) -> float:
    if args.warm_up_iter > 0 and step < args.warm_up_iter:
        return args.lr * float(step + 1) / float(args.warm_up_iter + 1)
    decays = sum(step >= milestone for milestone in args.milestones)
    return args.lr * (args.gamma ** decays)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def average_val_loss(args: argparse.Namespace, model, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    keys = ["loss", "loss_rec", "loss_explicit", "loss_commit", "perplexity"]
    totals: OrderedDict[str, float] = OrderedDict((key, 0.0) for key in keys)
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            motions = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            losses = compute_losses(args, model, motions)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
            count += 1
    model.train()
    packed = torch.tensor([totals[key] for key in keys] + [float(count)], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denom = max(float(packed[-1].item()), 1.0)
    return {key: float(packed[idx].item() / denom) for idx, key in enumerate(keys)}


def average_train_logs(logs: OrderedDict[str, float], log_count: int, device: torch.device) -> dict[str, float]:
    keys = ["loss", "loss_rec", "loss_explicit", "loss_commit", "perplexity", "lr", "grad_norm"]
    packed = torch.tensor([logs.get(key, 0.0) for key in keys] + [float(log_count)], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    denom = max(float(packed[-1].item()), 1.0)
    return {key: float(packed[idx].item() / denom) for idx, key in enumerate(keys)}


def assert_losses_finite(losses: OrderedDict[str, torch.Tensor], device: torch.device, rank: int, step: int) -> None:
    local_ok = True
    bad_names: list[str] = []
    for key, value in losses.items():
        if torch.is_tensor(value) and not bool(torch.isfinite(value.detach()).all().item()):
            local_ok = False
            bad_names.append(key)
    flag = torch.tensor([1.0 if local_ok else 0.0], device=device)
    if is_dist():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if flag.item() < 1.0:
        detail = ",".join(bad_names) if bad_names else "other-rank"
        master_print(rank, f"[error] non-finite loss at step={step} tensors={detail}")
        raise FloatingPointError(f"non-finite loss at step={step}: {detail}")


def check_grad_norm_finite(grad_norm: torch.Tensor, device: torch.device, rank: int, step: int) -> None:
    local_ok = bool(torch.isfinite(grad_norm.detach()).all().item())
    flag = torch.tensor([1.0 if local_ok else 0.0], device=device)
    if is_dist():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if flag.item() < 1.0:
        master_print(rank, f"[error] non-finite grad_norm at step={step} grad_norm={float(grad_norm.detach().cpu())}")
        raise FloatingPointError(f"non-finite grad norm at step={step}")


@torch.no_grad()
def evaluate_retrieval(model, eval_loader: DataLoader, eval_wrapper, device: torch.device) -> dict[str, float]:
    from utils.metrics import (
        calculate_R_precision,
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
        euclidean_distance_matrix,
    )

    model.eval()
    motion_annotation_list = []
    motion_pred_list = []
    r_precision_real = 0.0
    r_precision = 0.0
    matching_score_real = 0.0
    matching_score_pred = 0.0
    nb_sample = 0

    for batch in eval_loader:
        word_embeddings, pos_one_hots, _caption, sent_len, motion, m_length, _token = batch
        motion = motion.to(device=device, dtype=torch.float32, non_blocking=True)
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        pred_pose_eval, _loss_commit, _perplexity = model(motion)
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        et_np = et.detach().cpu().numpy()
        em_np = em.detach().cpu().numpy()
        et_pred_np = et_pred.detach().cpu().numpy()
        em_pred_np = em_pred.detach().cpu().numpy()
        r_precision_real += calculate_R_precision(et_np, em_np, top_k=3, sum_all=True)
        matching_score_real += euclidean_distance_matrix(et_np, em_np).trace()
        r_precision += calculate_R_precision(et_pred_np, em_pred_np, top_k=3, sum_all=True)
        matching_score_pred += euclidean_distance_matrix(et_pred_np, em_pred_np).trace()
        nb_sample += motion.shape[0]

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    r_precision_real = np.asarray(r_precision_real, dtype=np.float64) / nb_sample
    r_precision = np.asarray(r_precision, dtype=np.float64) / nb_sample
    metrics = {
        "fid": float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov)),
        "diversity_real": float(diversity_real),
        "diversity": float(diversity),
        "top1_real": float(r_precision_real[0]),
        "top2_real": float(r_precision_real[1]),
        "top3_real": float(r_precision_real[2]),
        "top1": float(r_precision[0]),
        "top2": float(r_precision[1]),
        "top3": float(r_precision[2]),
        "matching_score_real": float(matching_score_real / nb_sample),
        "matching_score": float(matching_score_pred / nb_sample),
    }
    model.train()
    return metrics


def pad_motion_batches(batches: list[np.ndarray], feature_dim: int = 272) -> np.ndarray:
    max_len = max(batch.shape[1] for batch in batches)
    total = sum(batch.shape[0] for batch in batches)
    out = np.zeros((total, max_len, feature_dim), dtype=np.float32)
    offset = 0
    for batch in batches:
        count, cur_len, _ = batch.shape
        out[offset: offset + count, :cur_len] = batch
        offset += count
    return out


def _to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _recover_motionstreamer_positions(hymotion_root: str):
    root = str(Path(hymotion_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from hymotion.eval.representation import recover_motionstreamer272_positions

    return recover_motionstreamer272_positions


def _geometry_recon_metrics(raw: np.ndarray, pred: np.ndarray, recover_fn) -> dict[str, float]:
    pos = _to_numpy(recover_fn(np.asarray(raw, dtype=np.float32))).astype(np.float32)
    pred_pos = _to_numpy(recover_fn(np.asarray(pred, dtype=np.float32))).astype(np.float32)
    length = min(int(pos.shape[0]), int(pred_pos.shape[0]))
    pos = pos[:length]
    pred_pos = pred_pos[:length]
    joint_l2 = np.linalg.norm(pos - pred_pos, axis=-1)
    root_l2 = np.linalg.norm(pos[:, 0] - pred_pos[:, 0], axis=-1)
    if length > 1:
        vel_l2 = np.linalg.norm(np.diff(pos, axis=0) - np.diff(pred_pos, axis=0), axis=-1)
    else:
        vel_l2 = np.zeros((0, 22), dtype=np.float32)
    return {
        "mpjpe_cm": float(joint_l2.mean() * 100.0),
        "root_cm": float(root_l2.mean() * 100.0),
        "vel_cm": float(vel_l2.mean() * 100.0) if vel_l2.size else 0.0,
    }


def recover_motionfix207_positions(motion: np.ndarray) -> np.ndarray:
    """Recover approximate global joints from MotionFix official 207D features.

    The 207D representation stores pelvis-local joints plus root/yaw deltas.
    Absolute initial global position/yaw is not represented, so the recovered
    trajectory is rooted at frame 0 with identity yaw. This is sufficient for
    comparing reconstruction against GT in the same representation.
    """
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 207:
        raise ValueError(f"Expected MotionFix207 motion [T,207], got {motion.shape}")
    frames = int(motion.shape[0])
    if frames <= 0:
        return np.zeros((0, 22, 3), dtype=np.float32)

    root_delta = motion[:, 0:3].astype(np.float32)
    orient_xy = rotation_6d_to_matrix_np(motion[:, 3:9])
    yaw_delta = rotation_6d_to_matrix_np(motion[:, 9:15])
    local_joints = motion[:, 141:207].reshape(frames, 22, 3).astype(np.float32)

    yaw = np.zeros((frames, 3, 3), dtype=np.float32)
    yaw[0] = np.eye(3, dtype=np.float32)
    for frame in range(1, frames):
        yaw[frame] = yaw[frame - 1] @ yaw_delta[frame]
    pelvis_orient = np.einsum("tij,tjk->tik", yaw, orient_xy).astype(np.float32)

    root = np.zeros((frames, 3), dtype=np.float32)
    for frame in range(1, frames):
        root[frame] = root[frame - 1] + pelvis_orient[frame - 1] @ root_delta[frame]

    joints = np.einsum("tdi,tji->tjd", yaw, local_joints).astype(np.float32)
    joints = joints + root[:, None, :]
    return joints


def _motionfix207_local_joint_metrics(raw: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    raw_local = np.asarray(raw, dtype=np.float32)[:, 141:207].reshape(-1, 22, 3)
    pred_local = np.asarray(pred, dtype=np.float32)[:, 141:207].reshape(-1, 22, 3)
    length = min(int(raw_local.shape[0]), int(pred_local.shape[0]))
    raw_local = raw_local[:length]
    pred_local = pred_local[:length]
    joint_l2 = np.linalg.norm(raw_local - pred_local, axis=-1)
    return {"local_mpjpe_cm": float(joint_l2.mean() * 100.0)}


def _motionstreamer272_local_metrics(raw: np.ndarray, pred: np.ndarray, recover_fn) -> dict[str, float]:
    raw = np.asarray(raw, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    length = min(int(raw.shape[0]), int(pred.shape[0]))
    raw = raw[:length]
    pred = pred[:length]
    raw_local = raw[:, 8:74].reshape(length, 22, 3)
    pred_local = pred[:, 8:74].reshape(length, 22, 3)
    local_l2 = np.linalg.norm(raw_local - pred_local, axis=-1)
    raw_local_vel = raw[:, 74:140].reshape(length, 22, 3)
    pred_local_vel = pred[:, 74:140].reshape(length, 22, 3)
    local_vel_l2 = np.linalg.norm(raw_local_vel - pred_local_vel, axis=-1)
    raw_pos = recover_fn(raw)
    pred_pos = recover_fn(pred)
    raw_centered = raw_pos - raw_pos[:, [0], :]
    pred_centered = pred_pos - pred_pos[:, [0], :]
    centered_l2 = np.linalg.norm(raw_centered - pred_centered, axis=-1)
    return {
        "local_pos_mpjpe_cm": float(local_l2.mean() * 100.0),
        "local_vel_mpjpe_cm": float(local_vel_l2.mean() * 100.0),
        "root_centered_mpjpe_cm": float(centered_l2.mean() * 100.0),
    }


@torch.no_grad()
def evaluate_motionfix_mpjpe_reconstruction(
    model,
    eval_loader: DataLoader,
    eval_bundle: dict,
    device: torch.device,
) -> dict[str, float]:
    if str(eval_bundle.get("backend")) != "motionfix_mpjpe_recon":
        raise ValueError(f"Unexpected eval bundle backend: {eval_bundle.get('backend')}")

    feature_dim = int(eval_bundle["feature_dim"])
    if feature_dim == 272:
        recover_fn = _recover_motionstreamer_positions(str(eval_bundle["hymotion_root"]))
    elif feature_dim == 207:
        recover_fn = recover_motionfix207_positions
    else:
        raise ValueError(f"Unsupported MotionFix MPJPE feature_dim={feature_dim}")

    model.eval()
    mean_t = torch.from_numpy(eval_bundle["mean"].astype(np.float32)).to(device).view(1, 1, -1)
    std_t = torch.from_numpy(eval_bundle["std"].astype(np.float32)).to(device).view(1, 1, -1)
    max_batches = int(eval_bundle.get("max_batches", 0) or 0)

    totals: dict[str, float] = {
        "source_mpjpe_cm": 0.0,
        "source_root_cm": 0.0,
        "source_vel_cm": 0.0,
        "target_mpjpe_cm": 0.0,
        "target_root_cm": 0.0,
        "target_vel_cm": 0.0,
    }
    if feature_dim == 207:
        totals["source_local_mpjpe_cm"] = 0.0
        totals["target_local_mpjpe_cm"] = 0.0
    if feature_dim == 272:
        totals["source_local_pos_mpjpe_cm"] = 0.0
        totals["target_local_pos_mpjpe_cm"] = 0.0
        totals["source_local_vel_mpjpe_cm"] = 0.0
        totals["target_local_vel_mpjpe_cm"] = 0.0
        totals["source_root_centered_mpjpe_cm"] = 0.0
        totals["target_root_centered_mpjpe_cm"] = 0.0
    count = 0

    for batch_id, batch in enumerate(eval_loader):
        if max_batches > 0 and batch_id >= max_batches:
            break
        source_raw = batch["source"].to(device=device, dtype=torch.float32, non_blocking=True)
        target_raw = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        source_lengths = batch.get("source_length", batch["length"]).detach().cpu().numpy().astype(np.int64)
        target_lengths = batch.get("target_length", batch["length"]).detach().cpu().numpy().astype(np.int64)

        source_norm = (source_raw - mean_t) / std_t
        target_norm = (target_raw - mean_t) / std_t
        source_pred_norm, _loss_commit, _perplexity = model(source_norm)
        target_pred_norm, _loss_commit, _perplexity = model(target_norm)
        source_pred = source_pred_norm * std_t + mean_t
        target_pred = target_pred_norm * std_t + mean_t

        source_raw_np = source_raw.detach().cpu().numpy().astype(np.float32)
        target_raw_np = target_raw.detach().cpu().numpy().astype(np.float32)
        source_pred_np = source_pred.detach().cpu().numpy().astype(np.float32)
        target_pred_np = target_pred.detach().cpu().numpy().astype(np.float32)

        for local_idx, (source_length_value, target_length_value) in enumerate(zip(source_lengths, target_lengths)):
            source_length = int(source_length_value)
            target_length = int(target_length_value)
            source_raw_i = source_raw_np[local_idx, :source_length]
            target_raw_i = target_raw_np[local_idx, :target_length]
            source_pred_i = source_pred_np[local_idx, :source_length]
            target_pred_i = target_pred_np[local_idx, :target_length]
            source_geom = _geometry_recon_metrics(source_raw_i, source_pred_i, recover_fn)
            target_geom = _geometry_recon_metrics(target_raw_i, target_pred_i, recover_fn)
            for key, value in source_geom.items():
                totals[f"source_{key}"] += float(value)
            for key, value in target_geom.items():
                totals[f"target_{key}"] += float(value)
            if feature_dim == 207:
                source_local = _motionfix207_local_joint_metrics(source_raw_i, source_pred_i)
                target_local = _motionfix207_local_joint_metrics(target_raw_i, target_pred_i)
                totals["source_local_mpjpe_cm"] += float(source_local["local_mpjpe_cm"])
                totals["target_local_mpjpe_cm"] += float(target_local["local_mpjpe_cm"])
            if feature_dim == 272:
                source_local = _motionstreamer272_local_metrics(source_raw_i, source_pred_i, recover_fn)
                target_local = _motionstreamer272_local_metrics(target_raw_i, target_pred_i, recover_fn)
                for key, value in source_local.items():
                    totals[f"source_{key}"] += float(value)
                for key, value in target_local.items():
                    totals[f"target_{key}"] += float(value)
            count += 1

    if count <= 0:
        raise RuntimeError("MotionFix MPJPE reconstruction eval collected no samples")

    metrics = {key: float(value / count) for key, value in totals.items()}
    metrics["mpjpe_cm"] = float(metrics["target_mpjpe_cm"])
    metrics["root_cm"] = float(metrics["target_root_cm"])
    metrics["vel_cm"] = float(metrics["target_vel_cm"])
    if feature_dim == 272:
        metrics["local_pos_mpjpe_cm"] = float(metrics["target_local_pos_mpjpe_cm"])
        metrics["local_vel_mpjpe_cm"] = float(metrics["target_local_vel_mpjpe_cm"])
        metrics["root_centered_mpjpe_cm"] = float(metrics["target_root_centered_mpjpe_cm"])
    metrics["nb_sample"] = float(count)
    metrics["eval_backend"] = "motionfix_mpjpe_recon"
    metrics["feature_dim"] = float(feature_dim)
    model.train()
    return metrics


@torch.no_grad()
def evaluate_motionstreamer272_reconstruction(
    model,
    eval_loader: DataLoader,
    eval_bundle: dict,
    device: torch.device,
) -> dict[str, float]:
    if str(eval_bundle.get("backend")) != "motionstreamer272_recon":
        raise ValueError(f"Unexpected eval bundle backend: {eval_bundle.get('backend')}")
    if getattr(eval_loader, "batch_size", None) != 32:
        raise ValueError(f"MotionStreamer272 FID/Top3 eval requires batch_size=32, got {eval_loader.batch_size}")

    if str(eval_bundle["hymotion_root"]) not in sys.path:
        sys.path.insert(0, str(eval_bundle["hymotion_root"]))
    from hymotion.eval.motionstreamer272 import evaluate_motionstreamer272

    model.eval()
    mean_t = torch.from_numpy(eval_bundle["mean"].astype(np.float32)).to(device).view(1, 1, -1)
    std_t = torch.from_numpy(eval_bundle["std"].astype(np.float32)).to(device).view(1, 1, -1)
    max_batches = int(eval_bundle.get("max_batches", 0) or 0)

    captions: list[str] = []
    lengths_list: list[np.ndarray] = []
    gt_motions: list[np.ndarray] = []
    pred_motions: list[np.ndarray] = []

    for batch_id, batch in enumerate(eval_loader):
        if max_batches > 0 and batch_id >= max_batches:
            break
        raw_motion = batch["motion"].to(device=device, dtype=torch.float32, non_blocking=True)
        lengths = batch["length"].to(device=device, dtype=torch.long, non_blocking=True)
        norm_motion = (raw_motion - mean_t) / std_t
        pred_norm, _loss_commit, _perplexity = model(norm_motion)
        pred_raw = pred_norm * std_t + mean_t

        padded_pred = torch.zeros_like(raw_motion)
        copy_len = min(raw_motion.shape[1], pred_raw.shape[1])
        padded_pred[:, :copy_len] = pred_raw[:, :copy_len]

        captions.extend(batch["caption"])
        lengths_list.append(lengths.detach().cpu().numpy().astype(np.int64))
        gt_motions.append(raw_motion.detach().cpu().numpy().astype(np.float32))
        pred_motions.append(padded_pred.detach().cpu().numpy().astype(np.float32))

    if not captions:
        raise RuntimeError("MotionStreamer272 reconstruction eval collected no samples")

    lengths_np = np.concatenate(lengths_list, axis=0)
    gt_np = pad_motion_batches(gt_motions, feature_dim=272)
    pred_np = pad_motion_batches(pred_motions, feature_dim=272)
    raw_metrics = evaluate_motionstreamer272(
        eval_bundle["evaluator"],
        texts=captions,
        gt_272=gt_np,
        gen_272=pred_np,
        lengths=lengths_np,
        batch_size=32,
        seed=1234,
    )
    r_precision = raw_metrics.get("r_precision", {})
    gt_r_precision = raw_metrics.get("gt_r_precision", {})
    metrics = {
        "fid": float(raw_metrics["fid"]),
        "top1": float(r_precision.get("top1", float("nan"))),
        "top2": float(r_precision.get("top2", float("nan"))),
        "top3": float(r_precision.get("top3", float("nan"))),
        "matching_score": float(raw_metrics.get("mm_dist", float("nan"))),
        "diversity": float(raw_metrics.get("diversity", float("nan"))),
        "top1_real": float(gt_r_precision.get("top1", float("nan"))),
        "top2_real": float(gt_r_precision.get("top2", float("nan"))),
        "top3_real": float(gt_r_precision.get("top3", float("nan"))),
        "matching_score_real": float(raw_metrics.get("gt_mm_dist", float("nan"))),
        "diversity_real": float(raw_metrics.get("gt_diversity", float("nan"))),
        "mpjpe_cm": float(raw_metrics.get("mpjpe_cm", float("nan"))),
        "foot_skating_ratio": float(raw_metrics.get("foot_skating_ratio", float("nan"))),
        "nb_sample": float(raw_metrics.get("samples", len(captions))),
    }
    model.train()
    return metrics


@torch.no_grad()
def evaluate_motionfix_official_reconstruction(
    model,
    eval_loader: DataLoader,
    eval_bundle: dict,
    device: torch.device,
) -> dict[str, float]:
    if str(eval_bundle.get("backend")) != "motionfix_official_recon":
        raise ValueError(f"Unexpected eval bundle backend: {eval_bundle.get('backend')}")

    from models.codeflow.eval_motionfix_official import (
        OfficialMotionFixDataset,
        _flatten_official_metrics,
        _mean_metric_dict,
        _metrics_from_sim_matrix,
        _official_generated_features,
        ensure_motionfix_imports,
        motionstreamer272_to_motionfix_pose,
        motionstreamer272_unified_v2_to_motionfix_pose,
    )

    model.eval()
    mean_t = torch.from_numpy(eval_bundle["mean"].astype(np.float32)).to(device).view(1, 1, -1)
    std_t = torch.from_numpy(eval_bundle["std"].astype(np.float32)).to(device).view(1, 1, -1)
    official_wrapper = eval_bundle["official_wrapper"]
    official_model = official_wrapper["model"]
    normalizer = official_wrapper["normalizer"]
    raw_records: dict[str, object] = official_wrapper["raw_records"]
    meta_by_id: dict[str, dict[str, Any]] = official_wrapper["meta_by_id"]
    rotation_corrections_value = official_wrapper.get("rotation_corrections")
    rotation_corrections = (
        np.asarray(rotation_corrections_value, dtype=np.float32)
        if rotation_corrections_value is not None
        else None
    )
    motionfix_repo: Path = official_wrapper["motionfix_repo"]
    inverse_mode = str(official_wrapper.get("motionfix_inverse_mode", ""))
    retrieval_batch_size = int(eval_bundle.get("retrieval_batch_size", 32))
    max_batches = int(eval_bundle.get("max_batches", 0) or 0)

    ensure_motionfix_imports(motionfix_repo)
    from src.tools.transforms3d import transform_body_pose
    from tmr_evaluator.motion2motion_retr import compute_sim_matrix, mat2name

    recover_fn = _recover_motionstreamer_positions(str(eval_bundle["hymotion_root"]))
    generated_poses: dict[str, np.ndarray] = {}
    ordered_keyids: list[str] = []
    geom_totals = OrderedDict(
        (
            ("source_mpjpe_cm", 0.0),
            ("source_root_cm", 0.0),
            ("source_vel_cm", 0.0),
            ("target_mpjpe_cm", 0.0),
            ("target_root_cm", 0.0),
            ("target_vel_cm", 0.0),
        )
    )
    geom_count = 0

    for batch_id, batch in enumerate(eval_loader):
        if max_batches > 0 and batch_id >= max_batches:
            break
        source_raw = batch["source"].to(device=device, dtype=torch.float32, non_blocking=True)
        target_raw = batch["target"].to(device=device, dtype=torch.float32, non_blocking=True)
        lengths = batch["length"].to(device=device, dtype=torch.long, non_blocking=True)
        sample_ids = [str(item) for item in batch["sample_id"]]

        source_norm = (source_raw - mean_t) / std_t
        target_norm = (target_raw - mean_t) / std_t
        source_pred_norm, _source_commit, _source_perplexity = model(source_norm)
        target_pred_norm, _target_commit, _target_perplexity = model(target_norm)
        source_pred_raw = source_pred_norm * std_t + mean_t
        target_pred_raw = target_pred_norm * std_t + mean_t

        source_raw_np = source_raw.detach().cpu().numpy().astype(np.float32)
        target_raw_np = target_raw.detach().cpu().numpy().astype(np.float32)
        source_pred_np = source_pred_raw.detach().cpu().numpy().astype(np.float32)
        target_pred_np = target_pred_raw.detach().cpu().numpy().astype(np.float32)
        lengths_np = lengths.detach().cpu().numpy().astype(np.int64)

        for local_idx, sample_id in enumerate(sample_ids):
            if sample_id not in raw_records:
                raise KeyError(f"MotionFix official raw split has no sample id {sample_id}")
            length = int(lengths_np[local_idx])
            source_raw_i = source_raw_np[local_idx, :length]
            target_raw_i = target_raw_np[local_idx, :length]
            source_pred_i = source_pred_np[local_idx, :length]
            target_pred_i = target_pred_np[local_idx, :length]

            source_geom = _geometry_recon_metrics(source_raw_i, source_pred_i, recover_fn)
            target_geom = _geometry_recon_metrics(target_raw_i, target_pred_i, recover_fn)
            for key, value in source_geom.items():
                geom_totals[f"source_{key}"] += float(value)
            for key, value in target_geom.items():
                geom_totals[f"target_{key}"] += float(value)
            geom_count += 1

            if inverse_mode == "motionstreamer272_unified_v2":
                if sample_id not in meta_by_id:
                    raise KeyError(f"MotionFix unified_v2 manifest/meta has no sample id {sample_id}")
                if rotation_corrections is None:
                    raise RuntimeError("MotionFix unified_v2 inverse requires rotation corrections")
                generated_poses[sample_id] = motionstreamer272_unified_v2_to_motionfix_pose(
                    target_pred_i,
                    meta_by_id[sample_id],
                    rotation_corrections,
                )
            elif inverse_mode in {"motionstreamer272_legacy", "motionstreamer272_hml"}:
                raw_item = raw_records[sample_id]
                generated_poses[sample_id] = motionstreamer272_to_motionfix_pose(
                    target_pred_i,
                    raw_item["motion_target"],
                    transform_body_pose,
                )
            else:
                raise RuntimeError(f"Unsupported MotionFix official inverse mode: {inverse_mode}")
            ordered_keyids.append(sample_id)

    if not ordered_keyids:
        raise RuntimeError("MotionFix official RVQ reconstruction eval collected no samples")

    gen_samples = _official_generated_features(generated_poses, normalizer, official_model.device, motionfix_repo)
    dataset = OfficialMotionFixDataset(raw_records, ordered_keyids, normalizer, motionfix_repo)
    full_result, _full_keys = compute_sim_matrix(
        official_model,
        dataset,
        np.asarray(dataset.keyids),
        gen_samples=gen_samples,
        batch_size=256,
        progress=False,
    )
    full_metrics = {
        mat2name[var]: _metrics_from_sim_matrix(sim_matrix)
        for var, sim_matrix in full_result.items()
    }

    batch_metrics_lists: dict[str, list[dict[str, float]]] = {"source_target": [], "target_generated": []}
    keyids_sorted = np.asarray(sorted(dataset.keyids))
    if len(keyids_sorted) >= retrieval_batch_size:
        rng = np.random.RandomState(0)
        idx = np.arange(len(keyids_sorted))
        rng.shuffle(idx)
        for start in range(0, (len(keyids_sorted) // retrieval_batch_size) * retrieval_batch_size, retrieval_batch_size):
            batch_keyids = keyids_sorted[idx[start: start + retrieval_batch_size]]
            batch_result, _batch_keys = compute_sim_matrix(
                official_model,
                dataset,
                batch_keyids,
                gen_samples=gen_samples,
                batch_size=256,
                progress=False,
            )
            for var, sim_matrix in batch_result.items():
                batch_metrics_lists[mat2name[var]].append(_metrics_from_sim_matrix(sim_matrix))
    batch_metrics = {key: _mean_metric_dict(value) for key, value in batch_metrics_lists.items()}
    metrics = _flatten_official_metrics(full_metrics, batch_metrics)
    metrics["motionfix_rtop3"] = float(metrics["rtop3"])
    metrics["motionfix_avgr"] = float(metrics["avgr"])
    metrics["nb_sample"] = float(len(ordered_keyids))
    metrics["eval_backend"] = "motionfix_official_recon"
    metrics["motionfix_official_split"] = str(official_wrapper.get("split", ""))
    metrics["motionfix_official_manifest_path"] = str(official_wrapper.get("manifest_path", ""))
    metrics["motionfix_official_inverse_mode"] = inverse_mode
    for key, value in geom_totals.items():
        metrics[key] = float(value / max(geom_count, 1))
    metrics["mpjpe_cm"] = float(metrics["target_mpjpe_cm"])
    metrics["root_cm"] = float(metrics["target_root_cm"])
    metrics["vel_cm"] = float(metrics["target_vel_cm"])
    model.train()
    return metrics


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def collect_quantizer_runtime_state(model) -> dict[str, dict]:
    state: dict[str, dict] = {}
    for name, module in unwrap_model(model).named_modules():
        if not hasattr(module, "codebook") or not hasattr(module, "init"):
            continue
        entry = {"init": bool(getattr(module, "init", False))}
        for attr in ("code_sum", "code_count"):
            value = getattr(module, attr, None)
            if torch.is_tensor(value):
                entry[attr] = value.detach().clone()
        state[name] = entry
    return state


def restore_quantizer_runtime_state(
    model,
    state: dict[str, dict] | None,
    device: torch.device,
    missing_code_count: float = 1024.0,
) -> None:
    state = state or {}
    missing_code_count = max(float(missing_code_count), 1.0)
    for name, module in model.named_modules():
        if not hasattr(module, "codebook") or not hasattr(module, "init"):
            continue
        entry = state.get(name, {})
        codebook = getattr(module, "codebook")

        if "code_sum" in entry:
            module.code_sum = entry["code_sum"].to(device=device, dtype=codebook.dtype)
        elif hasattr(module, "code_sum") and getattr(module, "code_sum", None) is None:
            module.code_sum = codebook.detach().clone() * missing_code_count

        if "code_count" in entry:
            module.code_count = entry["code_count"].to(device=device, dtype=codebook.dtype)
        elif hasattr(module, "code_count") and getattr(module, "code_count", None) is None:
            module.code_count = torch.full(
                (getattr(module, "nb_code"),),
                missing_code_count,
                device=device,
                dtype=codebook.dtype,
            )

        module.init = bool(entry.get("init", True))


def checkpoint_payload(
    args,
    model,
    optimizer,
    step: int,
    epoch: int,
    metrics=None,
    include_optimizer=False,
    train_state=None,
):
    payload = {
        "vq_model": unwrap_model(model).state_dict(),
        "step": step,
        "ep": epoch,
        "config": vars(args),
        "quantizer_runtime_state": collect_quantizer_runtime_state(model),
    }
    if train_state is not None:
        payload["train_state"] = train_state
    if metrics is not None:
        payload["metrics"] = metrics
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_checkpoint(
    path: Path,
    args,
    model,
    optimizer,
    step: int,
    epoch: int,
    metrics=None,
    include_optimizer=False,
    train_state=None,
):
    payload = checkpoint_payload(args, model, optimizer, step, epoch, metrics, include_optimizer, train_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


BEST_ALIAS_MAP = {
    "fid": "net_best_fid.tar",
    "top3": "net_best_top3.tar",
    "rtop3": "net_best_rtop3.tar",
    "avgr": "net_best_avgr.tar",
    "mpjpe_cm": "net_best_mpjpe.tar",
}


def best_alias_path(model_dir: Path, key: str) -> Path:
    return model_dir / BEST_ALIAS_MAP.get(key, f"net_best_{key}.tar")


def refresh_best_aliases(model_dir: Path, key: str, ranking: list[dict]) -> None:
    for idx, entry in enumerate(ranking, start=1):
        shutil.copy2(entry["path"], model_dir / f"best_{key}_rank{idx}.tar")
    if ranking:
        shutil.copy2(ranking[0]["path"], best_alias_path(model_dir, key))


def maybe_save_ranked(
    args,
    model,
    optimizer,
    model_dir: Path,
    rankings: dict[str, list[dict]],
    key: str,
    mode: str,
    value: float,
    step: int,
    epoch: int,
    metrics: dict[str, float],
) -> bool:
    current = rankings.setdefault(key, [])
    if len(current) >= args.topk:
        worst = current[-1]["value"]
        if mode == "min" and value >= worst:
            return False
        if mode == "max" and value <= worst:
            return False

    ckpt_dir = model_dir / f"best_{key}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        f"step_{step:09d}_{key}_{value:.6f}"
        f"_fid_{metrics.get('fid', math.nan):.6f}"
        f"_top3_{metrics.get('top3', math.nan):.6f}"
        f"_rtop3_{metrics.get('rtop3', math.nan):.6f}"
        f"_avgr_{metrics.get('avgr', math.nan):.6f}"
        f"_mpjpe_{metrics.get('mpjpe_cm', math.nan):.6f}.tar"
    )
    ckpt_path = ckpt_dir / ckpt_name
    save_checkpoint(ckpt_path, args, model, optimizer, step, epoch, metrics=metrics, include_optimizer=False)

    current.append({"step": step, "epoch": epoch, "value": value, "path": str(ckpt_path), "metrics": metrics})
    reverse = mode == "max"
    current.sort(key=lambda item: item["value"], reverse=reverse)
    del current[args.topk:]
    refresh_best_aliases(model_dir, key, current)
    return True


def maybe_save_best_alias(
    args,
    model,
    optimizer,
    model_dir: Path,
    rankings: dict[str, list[dict]],
    key: str,
    mode: str,
    value: float,
    step: int,
    epoch: int,
    metrics: dict[str, float],
    train_state=None,
) -> bool:
    value = float(value)
    if not math.isfinite(value):
        return False
    current = rankings.setdefault(key, [])
    if current:
        best_value = float(current[0]["value"])
        if mode == "min" and value >= best_value:
            return False
        if mode == "max" and value <= best_value:
            return False

    ckpt_path = best_alias_path(model_dir, key)
    save_checkpoint(
        ckpt_path,
        args,
        model,
        optimizer,
        step,
        epoch,
        metrics=metrics,
        include_optimizer=False,
        train_state=train_state,
    )
    rankings[key] = [{"step": step, "epoch": epoch, "value": value, "path": str(ckpt_path), "metrics": metrics}]
    return True


def load_existing_best_aliases(model_dir: Path, rankings: dict[str, list[dict]], keys: Iterable[str]) -> None:
    for key in keys:
        ckpt_path = best_alias_path(model_dir, key)
        if not ckpt_path.is_file():
            continue
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        except Exception:
            continue
        metrics = checkpoint.get("metrics", {}) or {}
        value = metrics.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        rankings[key] = [{
            "step": int(checkpoint.get("step", 0)),
            "epoch": int(checkpoint.get("ep", 0)),
            "value": float(value),
            "path": str(ckpt_path),
            "metrics": metrics,
        }]


def write_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def read_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_official_motionmillion_eval(
    args: argparse.Namespace,
    paths: dict[str, Path],
    model,
    optimizer,
    step: int,
    epoch: int,
    train_state: dict,
) -> dict[str, float]:
    eval_ckpt = paths["model"] / f"official_eval_step_{step:09d}.tar"
    eval_json = paths["logs"] / f"official_eval_step_{step:09d}_{args.official_eval_split}.json"
    eval_log = paths["logs"] / f"official_eval_step_{step:09d}_{args.official_eval_split}.log"
    save_checkpoint(
        eval_ckpt,
        args,
        model,
        optimizer,
        step,
        epoch,
        metrics=None,
        include_optimizer=False,
        train_state=train_state,
    )

    cmd = [
        str(args.official_eval_python),
        str(REPO_ROOT / "tools" / "eval_motionmillion_part_vq_recon.py"),
        "--checkpoint",
        str(eval_ckpt),
        "--partition_file",
        str(args.partition_file),
        "--gpu_id",
        "0",
        "--split",
        args.official_eval_split,
        "--batch_size",
        str(args.official_eval_batch_size),
        "--num_workers",
        str(args.official_eval_num_workers),
        "--max_batches",
        str(args.official_eval_max_batches),
        "--output_json",
        str(eval_json),
    ]
    start = time.time()
    with eval_log.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = "\n".join(eval_log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"Official MotionMillion eval failed with code {proc.returncode}. Log: {eval_log}\n{tail}")

    metrics = read_json(eval_json)
    if not isinstance(metrics, dict):
        raise RuntimeError(f"Official MotionMillion eval did not write metrics: {eval_json}")
    metrics["train_step"] = step
    metrics["train_epoch"] = epoch
    metrics["wall_seconds"] = time.time() - start
    metrics["result_json"] = str(eval_json)
    metrics["eval_log"] = str(eval_log)
    metrics["eval_checkpoint"] = str(eval_ckpt)
    try:
        eval_ckpt.unlink()
        metrics["eval_checkpoint_deleted"] = True
    except OSError:
        metrics["eval_checkpoint_deleted"] = False
    return metrics


def load_eval_components(args: argparse.Namespace, device: torch.device, mean: np.ndarray, std: np.ndarray):
    if args.dataset_name in {"motionfix", "motionfix_hml272", "motionfix207"} and (
        args.mpjpe_eval_only or args.dataset_name == "motionfix_hml272"
    ):
        dim_pose, _ = dataset_shape(args.dataset_name)
        eval_manifest = resolve_motionfix_manifest(args, args.official_eval_split)
        eval_dataset = MotionFixReconEvalDataset(
            eval_manifest,
            args.data_root,
            unit_length=4,
            max_motion_length=args.max_motion_length if args.max_motion_length > 0 else 300,
            max_samples=args.eval_max_samples,
            feature_dim=dim_pose,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.official_eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.official_eval_num_workers,
            pin_memory=True,
            collate_fn=motionfix_recon_eval_collate,
        )
        eval_bundle = {
            "backend": "motionfix_mpjpe_recon",
            "mean": mean,
            "std": std,
            "hymotion_root": str(args.hymotion_root),
            "max_batches": int(args.eval_max_batches),
            "feature_dim": dim_pose,
        }
        return eval_loader, eval_bundle

    if args.dataset_name == "motionfix":
        from models.codeflow.eval_motionfix_official import build_motionfix_official_eval_wrapper

        eval_manifest = resolve_motionfix_manifest(args, args.official_eval_split)
        eval_dataset = MotionFixReconEvalDataset(
            eval_manifest,
            args.data_root,
            unit_length=4,
            max_motion_length=args.max_motion_length if args.max_motion_length > 0 else 300,
            max_samples=args.eval_max_samples,
            feature_dim=272,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.official_eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.official_eval_num_workers,
            pin_memory=True,
            collate_fn=motionfix_recon_eval_collate,
        )
        eval_bundle = {
            "backend": "motionfix_official_recon",
            "official_wrapper": build_motionfix_official_eval_wrapper(
                motionfix_repo=str(args.motionfix_repo),
                motionfix_data_root=str(args.motionfix_official_data_root),
                manifest_path=str(eval_manifest),
                device=device,
            ),
            "mean": mean,
            "std": std,
            "hymotion_root": str(args.hymotion_root),
            "max_batches": int(args.eval_max_batches),
            "retrieval_batch_size": int(args.motionfix_eval_retrieval_batch_size),
        }
        return eval_loader, eval_bundle

    if args.dataset_name == "humanml3d_272":
        from models.codeflow.eval_motionstreamer272_t2m import (
            build_motionstreamer272_t2m_loader,
            load_motionstreamer272_evaluator,
        )

        eval_loader = build_motionstreamer272_t2m_loader(
            str(args.data_root),
            args.official_eval_split,
            args.eval_batch_size,
            args.official_eval_num_workers,
            unit_length=4,
            max_motion_length=args.max_motion_length if args.max_motion_length > 0 else 300,
            max_samples=args.eval_max_samples,
        )
        eval_bundle = {
            "backend": "motionstreamer272_recon",
            "evaluator": load_motionstreamer272_evaluator(
                hymotion_root=str(args.hymotion_root),
                checkpoint=str(args.motionstreamer_evaluator_checkpoint),
                distilbert_path=str(args.motionstreamer_distilbert_path),
                mean_path=str(resolve_stats_dir(args) / "Mean.npy"),
                std_path=str(resolve_stats_dir(args) / "Std.npy"),
                device=device,
            ),
            "mean": mean,
            "std": std,
            "hymotion_root": str(args.hymotion_root),
            "max_batches": int(args.eval_max_batches),
        }
        return eval_loader, eval_bundle

    from models.t2m_eval_wrapper import EvaluatorModelWrapper
    from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
    from utils.get_opt import get_opt

    eval_dataset = "t2m" if args.dataset_name == "humanml3d_272" else args.dataset_name
    opt_path = Path("checkpoints") / eval_dataset / "Comp_v6_KLD005" / "opt.txt"
    wrapper_opt = get_opt(str(opt_path), device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_loader, _ = get_dataset_motion_loader(str(opt_path), args.eval_batch_size, "val", device=device)
    return eval_loader, eval_wrapper


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def format_metric_value(key: str, value: object) -> str:
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{key}={float(value):.6f}"
    except Exception:
        pass
    return f"{key}={value}"


def infer_checkpoint_iters_per_epoch(checkpoint: dict) -> float | None:
    train_state = checkpoint.get("train_state") or {}
    if train_state.get("iters_per_epoch"):
        return float(train_state["iters_per_epoch"])
    config = checkpoint.get("config") or {}
    max_epoch = int(config.get("max_epoch") or 0)
    total_iter = int(config.get("total_iter") or 0)
    if max_epoch > 0 and total_iter > 0:
        return float(total_iter) / float(max_epoch)
    return None


def resolve_total_iter_after_resume(
    args: argparse.Namespace,
    current_iters_per_epoch: int,
    step: int,
    checkpoint: dict | None,
) -> tuple[int, dict[str, float | int | str | bool]]:
    if args.max_epoch <= 0:
        return int(args.total_iter), {
            "mode": "step",
            "target_total_iter": int(args.total_iter),
            "current_iters_per_epoch": int(current_iters_per_epoch),
        }

    default_total = int(args.max_epoch) * int(current_iters_per_epoch)
    summary: dict[str, float | int | str | bool] = {
        "mode": "fresh_max_epoch",
        "target_total_iter": default_total,
        "max_epoch": int(args.max_epoch),
        "current_iters_per_epoch": int(current_iters_per_epoch),
    }
    if not args.resume or checkpoint is None or step <= 0:
        return default_total, summary

    if args.resume_budget_mode == "step":
        summary["mode"] = "resume_step_budget"
        checkpoint_config = checkpoint.get("config") or {}
        target_total = int(checkpoint_config.get("total_iter") or args.total_iter)
        summary["target_total_iter"] = target_total
        summary["checkpoint_total_iter"] = int(checkpoint_config.get("total_iter") or 0)
        return target_total, summary

    old_iters_per_epoch = infer_checkpoint_iters_per_epoch(checkpoint)
    if old_iters_per_epoch is None or old_iters_per_epoch <= 0:
        if args.resume_budget_mode == "epoch":
            raise RuntimeError(
                "Cannot use --resume_budget_mode epoch because checkpoint has no recoverable iters/epoch metadata"
            )
        summary["mode"] = "resume_auto_fallback_step_budget"
        return default_total, summary

    progress_epoch = float(step) / float(old_iters_per_epoch)
    remaining_epochs = max(float(args.max_epoch) - progress_epoch, 0.0)
    target_total = int(step + math.ceil(remaining_epochs * float(current_iters_per_epoch)))
    summary.update(
        {
            "mode": "resume_epoch_budget",
            "target_total_iter": target_total,
            "old_iters_per_epoch": float(old_iters_per_epoch),
            "progress_epoch": progress_epoch,
            "remaining_epochs": remaining_epochs,
        }
    )
    return target_total, summary


def rebase_milestones_after_resume(args: argparse.Namespace, budget_summary: dict, step: int) -> None:
    if budget_summary.get("mode") != "resume_epoch_budget":
        return
    old_iters_per_epoch = float(budget_summary["old_iters_per_epoch"])
    current_iters_per_epoch = float(budget_summary["current_iters_per_epoch"])
    progress_epoch = float(budget_summary["progress_epoch"])
    original = [int(milestone) for milestone in args.milestones]
    rebased = []
    for milestone in original:
        milestone_epoch = float(milestone) / old_iters_per_epoch
        if milestone_epoch <= progress_epoch:
            rebased_step = int(step)
        else:
            rebased_step = int(step + math.ceil((milestone_epoch - progress_epoch) * current_iters_per_epoch))
        rebased.append(rebased_step)
    args.milestones = rebased
    budget_summary["original_milestones"] = original
    budget_summary["rebased_milestones"] = rebased


def main() -> None:
    args = parse_args()
    use_ddp, rank, world_size, local_rank, device = setup_distributed(args)
    set_seed(args.seed + rank)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    args.data_root = args.data_root.resolve()
    if args.motion_dir is not None:
        args.motion_dir = args.motion_dir.resolve()
    if args.stats_dir is not None:
        args.stats_dir = args.stats_dir.resolve()
    if args.split_dir is not None:
        args.split_dir = args.split_dir.resolve()
    if args.train_split_file is not None:
        args.train_split_file = args.train_split_file.resolve()
    if args.val_split_file is not None:
        args.val_split_file = args.val_split_file.resolve()
    if args.index_path is not None:
        args.index_path = args.index_path.resolve()
    if args.motionfix_train_manifest is not None:
        args.motionfix_train_manifest = args.motionfix_train_manifest.resolve()
    if args.motionfix_val_manifest is not None:
        args.motionfix_val_manifest = args.motionfix_val_manifest.resolve()
    if args.motionfix_test_manifest is not None:
        args.motionfix_test_manifest = args.motionfix_test_manifest.resolve()
    args.motionfix_repo = args.motionfix_repo.resolve()
    args.motionfix_official_data_root = args.motionfix_official_data_root.resolve()
    if args.kv_root is not None:
        args.kv_root = args.kv_root.resolve()
    args.hymotion_root = args.hymotion_root.resolve()
    args.motionstreamer_evaluator_checkpoint = args.motionstreamer_evaluator_checkpoint.resolve()
    args.motionstreamer_distilbert_path = args.motionstreamer_distilbert_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.partition_file = args.partition_file.resolve()
    paths = setup_paths(args, rank=rank)

    if not args.partition_file.exists():
        raise FileNotFoundError(args.partition_file)
    if rank == 0:
        shutil.copy2(args.partition_file, paths["config"] / args.partition_file.name)
        write_json(paths["config"] / "options.json", vars(args))
    dist_barrier()

    mean, std = load_stats(args, paths, rank=rank)
    if args.dataset_name in {"motionfix", "motionfix_hml272", "motionfix207"}:
        dim_pose, _ = dataset_shape(args.dataset_name)
        train_manifest = resolve_motionfix_manifest(args, "train").resolve()
        val_manifest = resolve_motionfix_manifest(args, "val").resolve()
        train_dataset = MotionFixWindowDataset(
            train_manifest,
            args.data_root,
            args.window_size,
            mean,
            std,
            sampling_mode=args.sampling_mode,
            min_motion_length=args.min_motion_length,
            max_motion_length=args.max_motion_length,
            feature_dim=dim_pose,
            hml272_rebase_crop_start=args.hml272_rebase_crop_start,
        )
        val_dataset = MotionFixWindowDataset(
            val_manifest,
            args.data_root,
            args.window_size,
            mean,
            std,
            sampling_mode=args.sampling_mode,
            min_motion_length=args.min_motion_length,
            max_motion_length=args.max_motion_length,
            feature_dim=dim_pose,
            hml272_rebase_crop_start=args.hml272_rebase_crop_start,
        )
    else:
        motion_dir = resolve_motion_dir(args).resolve()
        train_split_file = resolve_split_file(args, "train").resolve()
        val_split_file = resolve_split_file(args, "val").resolve()
        train_index_path = args.index_path
        val_index_path = None
        if train_index_path is None and args.dataset_name == "motionmillion":
            train_index_path = Path(str(train_split_file) + f".vq_w{args.window_size}.index.npz")
            val_index_path = Path(str(val_split_file) + f".vq_w{args.window_size}.index.npz")
        train_dataset = FixedWindowMotionDataset(
            motion_dir,
            train_split_file,
            args.window_size,
            mean,
            std,
            sampling_mode=args.sampling_mode,
            min_motion_length=args.min_motion_length,
            max_motion_length=args.max_motion_length,
            index_path=train_index_path,
        )
        val_dataset = FixedWindowMotionDataset(
            motion_dir,
            val_split_file,
            args.window_size,
            mean,
            std,
            sampling_mode=args.sampling_mode,
            min_motion_length=args.min_motion_length,
            max_motion_length=args.max_motion_length,
            index_path=val_index_path,
        )
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if use_ddp else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if use_ddp else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay
    )
    param_count = sum(param.numel() for param in model.parameters())
    master_print(rank, f"[ddp] enabled={use_ddp} world_size={world_size} rank={rank} local_rank={local_rank}")
    master_print(rank, f"[model] params={param_count / 1_000_000:.3f}M device={device}")
    master_print(rank, f"[stats] mean={mean.shape} std={std.shape} std_min={std.min():.6g} std_max={std.max():.6g}")
    if args.eval_every_epoch:
        args.eval_iter = len(train_loader)
        if rank == 0:
            write_json(paths["config"] / "options.json", vars(args))
        master_print(rank, f"[train] eval_every_epoch enabled eval_iter={args.eval_iter}")
    master_print(rank, f"[train] batches_per_epoch={len(train_loader)}")

    eval_loader = None
    eval_wrapper = None
    if args.eval_iter > 0:
        if rank == 0:
            eval_loader, eval_wrapper = load_eval_components(args, device, mean, std)
        dist_barrier()

    step = 0
    epoch = 0
    best_val = math.inf
    rankings = {"fid": [], "top3": [], "rtop3": [], "avgr": [], "mpjpe_cm": []}
    latest_path = paths["model"] / "latest.tar"
    best_val_path = paths["model"] / "net_best_val.tar"
    checkpoint = None
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["vq_model"])
        restore_quantizer_runtime_state(
            model,
            checkpoint.get("quantizer_runtime_state"),
            device,
            missing_code_count=args.resume_ema_code_count,
        )
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("ep", 0))
        master_print(rank, f"[resume] loaded {latest_path} step={step} epoch={epoch}")
        if best_val_path.exists():
            best_checkpoint = torch.load(best_val_path, map_location="cpu")
            val_loss = best_checkpoint.get("metrics", {}).get("val", {}).get("loss")
            if val_loss is not None and math.isfinite(float(val_loss)):
                best_val = float(val_loss)
                master_print(rank, f"[resume] best_val={best_val:.6f} from {best_val_path}")

    if args.dataset_name in {"motionfix", "motionfix_hml272", "motionfix207"} and rank == 0:
        mpjpe_only_aliases = args.save_only_last_and_best_mpjpe or args.dataset_name == "motionfix_hml272"
        alias_keys = ("mpjpe_cm",) if mpjpe_only_aliases else ("rtop3", "avgr", "mpjpe_cm")
        load_existing_best_aliases(paths["model"], rankings, alias_keys)

    best_official = read_json(paths["logs"] / "official_best_metrics.json") if rank == 0 else None
    best_official_fid = math.inf
    best_official_top3 = -math.inf
    if isinstance(best_official, dict):
        fid_value = best_official.get("best_fid", best_official.get("fid"))
        top3_value = best_official.get("best_top3", best_official.get("top3"))
        if fid_value is not None and math.isfinite(float(fid_value)):
            best_official_fid = float(fid_value)
        if top3_value is not None and math.isfinite(float(top3_value)):
            best_official_top3 = float(top3_value)
        master_print(
            rank,
            f"[official_eval] resume best_fid={best_official_fid:.6f} best_top3={best_official_top3:.6f}",
        )

    args.total_iter, budget_summary = resolve_total_iter_after_resume(
        args,
        current_iters_per_epoch=len(train_loader),
        step=step,
        checkpoint=checkpoint,
    )
    rebase_milestones_after_resume(args, budget_summary, step)
    if args.scale_motionmillion_milestones:
        reference_total = max(int(args.motionmillion_reference_total_iter), 1)
        reference_milestones = [int(milestone) for milestone in args.milestones]
        args.milestones = [
            max(1, int(round(float(args.total_iter) * float(milestone) / float(reference_total))))
            for milestone in reference_milestones
        ]
        budget_summary["reference_total_iter"] = reference_total
        budget_summary["reference_milestones"] = reference_milestones
        budget_summary["scaled_milestones"] = list(args.milestones)
    train_state = {
        "iters_per_epoch": len(train_loader),
        "world_size": world_size,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "resume_budget": budget_summary,
    }
    if rank == 0:
        write_json(paths["config"] / "options.json", vars(args))
        write_json(paths["logs"] / "train_state.json", train_state)
    master_print(rank, "[train] budget " + json.dumps(budget_summary, sort_keys=True))
    master_print(rank, f"[train] total_iter={args.total_iter}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    model.train()
    logs: OrderedDict[str, float] = OrderedDict()
    log_count = 0
    start_time = time.time()
    train_iter = iter(train_loader)
    start_step = step
    resume_progress_epoch = float(budget_summary.get("progress_epoch", float(epoch)))
    last_official_eval_epoch = -1
    while step < args.total_iter:
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step += 1
        lr = lr_at_step(args, step)
        set_optimizer_lr(optimizer, lr)

        motions = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        losses = compute_losses(args, model, motions)
        assert_losses_finite(losses, device, rank, step)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        grad_norm_value = 0.0
        if args.max_grad_norm and args.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(args.max_grad_norm), error_if_nonfinite=False
            )
            check_grad_norm_finite(grad_norm, device, rank, step)
            grad_norm_value = float(grad_norm.detach().cpu().item())
        optimizer.step()

        log_count += 1
        steps_since_start = step - start_step
        for key, value in losses.items():
            logs[key] = logs.get(key, 0.0) + float(value.detach().cpu().item())
        logs["lr"] = logs.get("lr", 0.0) + lr
        logs["grad_norm"] = logs.get("grad_norm", 0.0) + grad_norm_value

        if step % args.print_iter == 0:
            avg = average_train_logs(logs, log_count, device)
            elapsed = format_seconds(time.time() - start_time)
            msg = " ".join(f"{key}={value:.6f}" for key, value in avg.items())
            master_print(rank, f"[train] step={step} epoch={epoch} elapsed={elapsed} {msg}")
            logs.clear()
            log_count = 0

        if rank == 0 and (step % args.save_latest == 0 or step == args.total_iter):
            save_checkpoint(
                latest_path,
                args,
                model,
                optimizer,
                step,
                epoch,
                include_optimizer=True,
                train_state=train_state,
            )

        run_periodic_eval = (
            args.eval_iter > 0
            and steps_since_start > 0
            and steps_since_start % args.eval_iter == 0
        )
        if args.eval_iter > 0 and (run_periodic_eval or step == args.total_iter):
            val_metrics = average_val_loss(args, model, val_loader, device)
            completed_epoch = int(math.floor(resume_progress_epoch + steps_since_start / max(len(train_loader), 1)))
            master_print(
                rank,
                "[val] "
                + f"step={step} epoch={epoch} completed_epoch={completed_epoch} "
                + " ".join(f"{key}={value:.6f}" for key, value in val_metrics.items())
            )
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                if (
                    rank == 0
                    and args.dataset_name not in {"motionfix", "motionfix_hml272"}
                    and not args.save_only_last_and_best_mpjpe
                ):
                    save_checkpoint(
                        paths["model"] / "net_best_val.tar",
                        args,
                        model,
                        optimizer,
                        step,
                        epoch,
                        metrics={"val": val_metrics},
                        include_optimizer=False,
                        train_state=train_state,
                    )

            if args.eval_iter > 0:
                if is_dist():
                    dist_barrier()
                if rank == 0:
                    if eval_loader is None or eval_wrapper is None:
                        raise RuntimeError("Eval is mandatory, but eval components were not loaded on rank 0")
                    eval_backend = eval_wrapper.get("backend") if isinstance(eval_wrapper, dict) else ""
                    if eval_backend == "motionfix_mpjpe_recon":
                        eval_metrics = evaluate_motionfix_mpjpe_reconstruction(
                            unwrap_model(model), eval_loader, eval_wrapper, device
                        )
                    elif eval_backend == "motionstreamer272_recon":
                        eval_metrics = evaluate_motionstreamer272_reconstruction(
                            unwrap_model(model), eval_loader, eval_wrapper, device
                        )
                    elif eval_backend == "motionfix_official_recon":
                        eval_metrics = evaluate_motionfix_official_reconstruction(
                            unwrap_model(model), eval_loader, eval_wrapper, device
                        )
                    else:
                        eval_metrics = evaluate_retrieval(unwrap_model(model), eval_loader, eval_wrapper, device)
                    metrics = {"val": val_metrics, **eval_metrics}
                    master_print(
                        rank,
                        "[eval] "
                        + f"step={step} epoch={epoch} "
                        + " ".join(format_metric_value(key, value) for key, value in eval_metrics.items())
                    )
                    if args.save_only_last_and_best_mpjpe or eval_backend == "motionfix_mpjpe_recon":
                        mpjpe_hit = maybe_save_best_alias(
                            args, model, optimizer, paths["model"], rankings, "mpjpe_cm", "min",
                            eval_metrics["mpjpe_cm"], step, epoch, metrics, train_state=train_state
                        )
                        master_print(rank, f"[best] step={step} mpjpe_hit={mpjpe_hit}")
                    elif eval_backend == "motionfix_official_recon":
                        rtop3_hit = maybe_save_best_alias(
                            args, model, optimizer, paths["model"], rankings, "rtop3", "max",
                            eval_metrics["rtop3"], step, epoch, metrics, train_state=train_state
                        )
                        avgr_hit = maybe_save_best_alias(
                            args, model, optimizer, paths["model"], rankings, "avgr", "min",
                            eval_metrics["avgr"], step, epoch, metrics, train_state=train_state
                        )
                        mpjpe_hit = maybe_save_best_alias(
                            args, model, optimizer, paths["model"], rankings, "mpjpe_cm", "min",
                            eval_metrics["mpjpe_cm"], step, epoch, metrics, train_state=train_state
                        )
                        master_print(
                            rank,
                            f"[best] step={step} rtop3_hit={rtop3_hit} avgr_hit={avgr_hit} mpjpe_hit={mpjpe_hit}",
                        )
                    else:
                        fid_hit = maybe_save_ranked(
                            args, model, optimizer, paths["model"], rankings, "fid", "min",
                            eval_metrics["fid"], step, epoch, metrics
                        )
                        top3_hit = maybe_save_ranked(
                            args, model, optimizer, paths["model"], rankings, "top3", "max",
                            eval_metrics["top3"], step, epoch, metrics
                        )
                        master_print(rank, f"[best] step={step} fid_hit={fid_hit} top3_hit={top3_hit}")
                    write_json(paths["logs"] / "best_metrics.json", rankings)
                if is_dist():
                    dist_barrier()

            run_official = (
                args.dataset_name == "motionmillion"
                and args.official_eval_every_epoch > 0
                and completed_epoch >= int(args.official_eval_start_epoch)
                and completed_epoch > last_official_eval_epoch
                and completed_epoch % int(args.official_eval_every_epoch) == 0
            )
            if run_official:
                if is_dist():
                    dist_barrier()
                if rank == 0:
                    official_metrics = run_official_motionmillion_eval(
                        args, paths, model, optimizer, step, epoch, train_state
                    )
                    fid = float(official_metrics["fid"])
                    top3 = float(official_metrics["top3"])
                    official_payload = dict(official_metrics)
                    official_payload["completed_epoch"] = completed_epoch
                    append_path = paths["logs"] / "official_eval.jsonl"
                    with append_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(official_payload, sort_keys=True) + "\n")
                    master_print(
                        rank,
                        "[official_eval] "
                        f"step={step} completed_epoch={completed_epoch} fid={fid:.6f} "
                        f"top3={top3:.6f} top1={float(official_metrics['top1']):.6f} "
                        f"matching={float(official_metrics['matching_score']):.6f} "
                        f"json={official_metrics.get('result_json', '')}",
                    )
                    if fid < best_official_fid:
                        best_official_fid = fid
                        save_checkpoint(
                            paths["model"] / "net_best_fid.tar",
                            args,
                            model,
                            optimizer,
                            step,
                            epoch,
                            metrics={"official": official_metrics, "val": val_metrics},
                            include_optimizer=False,
                            train_state=train_state,
                        )
                    if top3 > best_official_top3:
                        best_official_top3 = top3
                        save_checkpoint(
                            paths["model"] / "net_best_top3.tar",
                            args,
                            model,
                            optimizer,
                            step,
                            epoch,
                            metrics={"official": official_metrics, "val": val_metrics},
                            include_optimizer=False,
                            train_state=train_state,
                        )
                    official_best = dict(official_metrics)
                    official_best.update(
                        {
                            "best_fid": best_official_fid,
                            "best_top3": best_official_top3,
                            "completed_epoch": completed_epoch,
                        }
                    )
                    write_json(paths["logs"] / "official_best_metrics.json", official_best)
                last_official_eval_epoch = completed_epoch
                if is_dist():
                    dist_barrier()

    if rank == 0:
        save_checkpoint(latest_path, args, model, optimizer, step, epoch, include_optimizer=True, train_state=train_state)
        print(f"[done] step={step} epoch={epoch} latest={latest_path}", flush=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
