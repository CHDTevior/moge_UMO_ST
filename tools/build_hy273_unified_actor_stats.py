#!/usr/bin/env python3
"""Build train-only normalization stats for T2M/Edit/Interaction targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.hy273_interaction_dataset import apply_shared_interaction_gauge
from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_slices import (
    BODY_DIM,
    CONTACT_SLICE,
    DIM_HY273,
    HEADING_SLICE,
    ROOT_DIM,
)
from tools.build_hy273_redenoise_stats import global_root_to_local, save_stats


STATS_FORMAT = "hy273_unified_actor_target_stats_v1"
YAW_TARGETS = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)


class WeightedMoments:
    def __init__(self, dim: int) -> None:
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros(dim, dtype=np.float64)
        self.weight = 0.0
        self.raw_frames = 0
        self.assets = 0

    def update(self, values: np.ndarray, weight: float) -> None:
        array = np.asarray(values, dtype=np.float64)
        scalar = float(weight)
        self.sum += array.sum(axis=0) * scalar
        self.sum_sq += np.square(array).sum(axis=0) * scalar
        self.weight += array.shape[0] * scalar
        self.raw_frames += int(array.shape[0])
        self.assets += 1

    def mean_e2(self) -> tuple[np.ndarray, np.ndarray]:
        if self.weight <= 0:
            raise RuntimeError("No frames accumulated for one stats domain")
        return self.sum / self.weight, self.sum_sq / self.weight


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_k273(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != DIM_HY273 or array.dtype != np.float32:
        raise ValueError(f"Invalid K273 stats asset {path}: {array.shape}/{array.dtype}")
    if array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError(f"Invalid K273 values under {path}")
    return torch.from_numpy(array.copy())


def _single_actor_versions(path: Path) -> list[torch.Tensor]:
    raw = _load_k273(path)
    shifted = root_origin_shift(raw)
    heading = shifted[0, HEADING_SLICE]
    current = torch.atan2(heading[1], heading[0])
    return [
        apply_yaw_rotation(
            shifted,
            torch.as_tensor(target, dtype=shifted.dtype) - current,
        )
        for target in YAW_TARGETS
    ]


def _interaction_versions(
    root: Path,
    row: dict[str, Any],
    *,
    max_frames: int,
    crop_samples: int,
) -> list[torch.Tensor]:
    frames = int(row["frames"])
    pair = torch.stack(
        [
            _load_k273((root / str(row["person1"])).resolve()),
            _load_k273((root / str(row["person2"])).resolve()),
        ],
        dim=0,
    )
    if pair.shape[1] != frames:
        raise ValueError(f"Interaction frame metadata mismatch for {row['clip_id']}")
    crop_length = min(frames, int(max_frames))
    crop_choices = max(1, frames - crop_length + 1)
    sample_count = min(crop_choices, max(1, int(crop_samples)))
    # Midpoints of equal-probability bins give a deterministic quadrature for
    # the loader's uniform crop-start distribution.
    starts = np.floor(
        (np.arange(sample_count, dtype=np.float64) + 0.5)
        * float(crop_choices)
        / float(sample_count)
    ).astype(np.int64)
    starts = np.unique(starts).tolist()
    versions = []
    for start in starts:
        crop = pair[:, int(start) : int(start) + crop_length]
        for target in YAW_TARGETS:
            augmented, _ = apply_shared_interaction_gauge(crop, float(target))
            versions.append(augmented)
    return versions


def _accumulate_single(
    full: WeightedMoments,
    local: WeightedMoments,
    path: Path,
    sample_weight: float,
    fps: float,
) -> None:
    versions = _single_actor_versions(path)
    for motion in versions:
        weight = float(sample_weight) / len(versions)
        full.update(motion.numpy(), weight)
        local.update(global_root_to_local(motion[:, :ROOT_DIM], fps).numpy(), weight)


def _accumulate_interaction(
    full: WeightedMoments,
    local: WeightedMoments,
    root: Path,
    row: dict[str, Any],
    fps: float,
    max_frames: int,
    crop_samples: int,
) -> None:
    versions = _interaction_versions(
        root,
        row,
        max_frames=max_frames,
        crop_samples=crop_samples,
    )
    for pair in versions:
        actor_weight = 1.0 / float(len(versions) * pair.shape[0])
        for actor in pair:
            full.update(actor.numpy(), actor_weight)
            local.update(
                global_root_to_local(actor[:, :ROOT_DIM], fps).numpy(),
                actor_weight,
            )


def _combine(
    domains: dict[str, WeightedMoments],
    weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = np.zeros_like(next(iter(domains.values())).sum)
    e2 = np.zeros_like(mean)
    report = {}
    for name, moments in domains.items():
        domain_mean, domain_e2 = moments.mean_e2()
        mix_weight = float(weights[name])
        mean += mix_weight * domain_mean
        e2 += mix_weight * domain_e2
        report[name] = {
            "mix_weight": mix_weight,
            "weighted_frame_mass": moments.weight,
            "raw_augmented_frames": moments.raw_frames,
            "asset_updates": moments.assets,
            "contact_prior": (
                domain_mean[CONTACT_SLICE].tolist()
                if domain_mean.shape[0] == DIM_HY273
                else None
            ),
        }
    variance = np.maximum(e2 - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multitask_train_manifest",
        default=(
            "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/"
            "manifests/hy273_multitask_v1/train.jsonl"
        ),
    )
    parser.add_argument(
        "--interaction_root",
        default="/mnt/afs/mogo_base/datasets/InteractionK273/combined",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/mnt/afs/mogo_base/datasets/HY273_unified_actor_v1/"
            "derived_stats/t2m30_edit35_interaction35"
        ),
    )
    parser.add_argument("--t2m_weight", type=float, default=0.30)
    parser.add_argument("--edit_weight", type=float, default=0.35)
    parser.add_argument("--interaction_weight", type=float, default=0.35)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min_interaction_frames", type=int, default=16)
    parser.add_argument("--max_interaction_frames", type=int, default=300)
    parser.add_argument("--interaction_crop_samples", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    weights = {
        "humanml3d_k273": float(args.t2m_weight),
        "motionfix_k273": float(args.edit_weight),
        "interaction_k273": float(args.interaction_weight),
    }
    if min(weights.values()) < 0.0 or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Domain stats weights must be non-negative and sum to one")
    multitask_manifest = Path(args.multitask_train_manifest).expanduser().resolve()
    interaction_root = Path(args.interaction_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    multitask_rows = _load_rows(multitask_manifest)
    hml_rows = [row for row in multitask_rows if row["dataset"] == "humanml3d_k273"]
    edit_rows = [row for row in multitask_rows if row["dataset"] == "motionfix_k273"]
    interaction_rows = [
        row
        for row in _load_rows(interaction_root / "manifest.jsonl")
        if row.get("split") == "train"
        and bool(row.get("has_text"))
        and row.get("texts")
        and int(row.get("frames", 0)) >= int(args.min_interaction_frames)
    ]
    if not hml_rows or not edit_rows or not interaction_rows:
        raise RuntimeError("All three train target domains are required")

    full_domains = {name: WeightedMoments(DIM_HY273) for name in weights}
    local_domains = {name: WeightedMoments(4) for name in weights}
    fps = float(args.fps)
    for index, row in enumerate(hml_rows, 1):
        path_weights: dict[str, float] = {}
        caption_weight = 1.0 / len(row["texts"])
        for text in row["texts"]:
            path = str(text["target_k273_asset"]["path"])
            path_weights[path] = path_weights.get(path, 0.0) + caption_weight
        for path, sample_weight in path_weights.items():
            _accumulate_single(
                full_domains["humanml3d_k273"],
                local_domains["humanml3d_k273"],
                Path(path),
                sample_weight,
                fps,
            )
        if index % 1000 == 0 or index == len(hml_rows):
            print(f"[stats] HML {index}/{len(hml_rows)}", flush=True)

    for index, row in enumerate(edit_rows, 1):
        _accumulate_single(
            full_domains["motionfix_k273"],
            local_domains["motionfix_k273"],
            Path(row["target_motion"]["k273_asset"]["path"]),
            1.0,
            fps,
        )
        if index % 500 == 0 or index == len(edit_rows):
            print(f"[stats] MotionFix {index}/{len(edit_rows)}", flush=True)

    for index, row in enumerate(interaction_rows, 1):
        _accumulate_interaction(
            full_domains["interaction_k273"],
            local_domains["interaction_k273"],
            interaction_root,
            row,
            fps,
            int(args.max_interaction_frames),
            int(args.interaction_crop_samples),
        )
        if index % 500 == 0 or index == len(interaction_rows):
            print(f"[stats] Interaction {index}/{len(interaction_rows)}", flush=True)

    full_mean, full_std, full_report = _combine(full_domains, weights)
    local_mean, local_std, local_report = _combine(local_domains, weights)
    planar_var = float(
        (
            local_std[1] ** 2
            + local_mean[1] ** 2
            + local_std[2] ** 2
            + local_mean[2] ** 2
        )
        / 2.0
    )
    local_mean[1:3] = 0.0
    local_std[1:3] = np.sqrt(planar_var)

    save_stats(output_dir / "full", full_mean, full_std)
    save_stats(output_dir / "global_root", full_mean[:ROOT_DIM], full_std[:ROOT_DIM])
    save_stats(output_dir / "body", full_mean[ROOT_DIM:], full_std[ROOT_DIM:])
    save_stats(output_dir / "local_root", local_mean, local_std)
    manifest = {
        "format": STATS_FORMAT,
        "contacts_normalized": True,
        "target_only": True,
        "source_in_stats": False,
        "fps": fps,
        "domain_task_weights": weights,
        "stage_b_task_mix": {"t2m": 0.30, "edit": 0.35, "interaction": 0.35},
        "hml_caption_weighting": "one motion row has unit mass across captions",
        "motionfix_stats_role": "target_only",
        "interaction_stats_role": "both target actors",
        "interaction_actor_swap_marginal": (
            "exact because both target actors enter the same domain moments"
        ),
        "interaction_text_rows_only": True,
        "interaction_min_frames": int(args.min_interaction_frames),
        "interaction_max_frames": int(args.max_interaction_frames),
        "interaction_crop_distribution": (
            "deterministic stratified quadrature over uniform crop starts"
        ),
        "interaction_crop_samples_per_long_pair": int(
            args.interaction_crop_samples
        ),
        "root_origin_shift": True,
        "yaw_quadrature_targets_rad": list(YAW_TARGETS),
        "full_domain_report": full_report,
        "local_domain_report": local_report,
        "row_counts": {
            "humanml3d_k273": len(hml_rows),
            "motionfix_k273": len(edit_rows),
            "interaction_k273": len(interaction_rows),
        },
        "dims": {
            "full": DIM_HY273,
            "global_root": ROOT_DIM,
            "body": BODY_DIM,
            "local_root": 4,
        },
        "sources": {
            "multitask_train_manifest": str(multitask_manifest),
            "interaction_manifest": str(interaction_root / "manifest.jsonl"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
