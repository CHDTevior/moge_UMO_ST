#!/usr/bin/env python3
"""Build frozen target-only 80/20 HY273 multitask normalization statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_slices import (
    BODY_DIM,
    CONTACT_SLICE,
    DIM_HY273,
    HEADING_SLICE,
    ROOT_DIM,
)
from tools.build_hy273_redenoise_stats import global_root_to_local, save_stats


LEGACY_FORMAT = "hy273_multitask_target_stats_v1"
UNIFIED_FORMAT = "hy273_multitask_target_stats_v2_unified273"
DOMAIN_WEIGHTS = {"humanml3d_k273": 0.8, "motionfix_k273": 0.2}
YAW_TARGETS = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WeightedMoments:
    def __init__(self, dim: int) -> None:
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sum_sq = np.zeros(dim, dtype=np.float64)
        self.weight = 0.0
        self.raw_frames = 0
        self.assets = 0

    def update(self, values: np.ndarray, weight: float) -> None:
        x = np.asarray(values, dtype=np.float64)
        scalar = float(weight)
        self.sum += x.sum(axis=0) * scalar
        self.sum_sq += np.square(x).sum(axis=0) * scalar
        self.weight += x.shape[0] * scalar
        self.raw_frames += int(x.shape[0])
        self.assets += 1

    def mean_e2(self) -> tuple[np.ndarray, np.ndarray]:
        if self.weight <= 0:
            raise RuntimeError("No weighted frames were accumulated")
        return self.sum / self.weight, self.sum_sq / self.weight


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def augmented_versions(path: Path) -> list[torch.Tensor]:
    raw = torch.from_numpy(np.load(path).astype(np.float32, copy=False))
    if raw.ndim != 2 or raw.shape[1] != DIM_HY273 or raw.shape[0] < 2:
        raise ValueError(f"Invalid stats asset {path}: {tuple(raw.shape)}")
    shifted = root_origin_shift(raw)
    heading = shifted[0, HEADING_SLICE]
    current = torch.atan2(heading[1], heading[0])
    return [
        apply_yaw_rotation(shifted, torch.as_tensor(target, dtype=shifted.dtype) - current)
        for target in YAW_TARGETS
    ]


def accumulate_asset(
    full: WeightedMoments,
    local: WeightedMoments,
    path: Path,
    sample_weight: float,
    fps: float,
) -> None:
    for augmented in augmented_versions(path):
        weight = float(sample_weight) / len(YAW_TARGETS)
        full.update(augmented.numpy(), weight)
        local.update(global_root_to_local(augmented[:, :ROOT_DIM], fps).numpy(), weight)


def combine_domains(
    moments: dict[str, WeightedMoments], weights: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = np.zeros_like(next(iter(moments.values())).sum)
    e2 = np.zeros_like(mean)
    report: dict[str, Any] = {}
    for name, domain in moments.items():
        domain_mean, domain_e2 = domain.mean_e2()
        weight = float(weights[name])
        mean += weight * domain_mean
        e2 += weight * domain_e2
        report[name] = {
            "mix_weight": weight,
            "weighted_frame_mass": domain.weight,
            "raw_augmented_frames": domain.raw_frames,
            "asset_updates": domain.assets,
            "mean_sha256": hashlib.sha256(domain_mean.tobytes()).hexdigest(),
            "e2_sha256": hashlib.sha256(domain_e2.tobytes()).hexdigest(),
            "contact_prior": domain_mean[CONTACT_SLICE].tolist()
            if domain_mean.shape[0] == DIM_HY273
            else None,
        }
    variance = np.maximum(e2 - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), report


def source_coverage(
    rows: list[dict[str, Any]], mean: np.ndarray, std: np.ndarray
) -> dict[str, Any]:
    absolute_values: list[np.ndarray] = []
    max_abs = np.zeros(DIM_HY273 - 4, dtype=np.float64)
    nonfinite = 0
    total = 0
    scale = np.sqrt(np.square(std[:269].astype(np.float64)) + 1e-5)
    for row in rows:
        source = Path(row["source_motion"]["k273_asset"]["path"])
        for augmented in augmented_versions(source):
            normalized = (
                augmented.numpy()[:, :269].astype(np.float64) - mean[:269]
            ) / scale
            nonfinite += int((~np.isfinite(normalized)).sum())
            max_abs = np.maximum(max_abs, np.nanmax(np.abs(normalized), axis=0))
            # Bounded deterministic sample for global quantiles.
            absolute_values.append(np.abs(normalized[:: max(1, normalized.shape[0] // 8)]).reshape(-1))
            total += normalized.size
    sampled = np.concatenate(absolute_values) if absolute_values else np.empty(0)
    return {
        "continuous_values": total,
        "nonfinite": nonfinite,
        "abs_z_quantiles": {
            str(q): float(np.quantile(sampled, q)) for q in (0.5, 0.9, 0.99, 0.999)
        },
        "channel_abs_z_max": max_abs.tolist(),
        "channels_abs_z_max_gt_10": int((max_abs > 10.0).sum()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_manifest",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/hy273_multitask_v1/train.jsonl",
    )
    parser.add_argument(
        "--accepted_captions",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/hy273_multitask_v1/accepted_captions.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/derived_stats/hy273_multitask_stats_v1",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--normalize_contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep learned target mean/std for contact instead of overriding them to 0/1.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.train_manifest).expanduser().resolve()
    captions_path = Path(args.accepted_captions).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(manifest_path)
    hml_rows = [row for row in rows if row["dataset"] == "humanml3d_k273"]
    motionfix_rows = [row for row in rows if row["dataset"] == "motionfix_k273"]
    if not hml_rows or not motionfix_rows:
        raise RuntimeError("Joint stats require both HML and MotionFix target rows")

    full_domains = {name: WeightedMoments(DIM_HY273) for name in DOMAIN_WEIGHTS}
    local_domains = {name: WeightedMoments(4) for name in DOMAIN_WEIGHTS}

    for index, row in enumerate(hml_rows, 1):
        path_weights: dict[str, float] = {}
        caption_weight = 1.0 / len(row["texts"])
        for text in row["texts"]:
            path = str(text["target_k273_asset"]["path"])
            path_weights[path] = path_weights.get(path, 0.0) + caption_weight
        for path, weight in path_weights.items():
            accumulate_asset(
                full_domains["humanml3d_k273"],
                local_domains["humanml3d_k273"],
                Path(path),
                weight,
                float(args.fps),
            )
        if index % 1000 == 0 or index == len(hml_rows):
            print(f"[stats] HML rows {index}/{len(hml_rows)}", flush=True)

    for index, row in enumerate(motionfix_rows, 1):
        accumulate_asset(
            full_domains["motionfix_k273"],
            local_domains["motionfix_k273"],
            Path(row["target_motion"]["k273_asset"]["path"]),
            1.0,
            float(args.fps),
        )
        if index % 500 == 0 or index == len(motionfix_rows):
            print(f"[stats] MotionFix targets {index}/{len(motionfix_rows)}", flush=True)

    full_mean, full_std, full_report = combine_domains(full_domains, DOMAIN_WEIGHTS)
    local_mean, local_std, local_report = combine_domains(local_domains, DOMAIN_WEIGHTS)
    combined_contact_prior = full_mean[CONTACT_SLICE].copy()
    combined_contact_std = full_std[CONTACT_SLICE].copy()
    if not args.normalize_contacts:
        full_mean[CONTACT_SLICE] = 0.0
        full_std[CONTACT_SLICE] = 1.0

    # Four equally spaced gauges make planar local-root velocity isotropic.
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
    coverage = source_coverage(motionfix_rows, full_mean, full_std)
    if coverage["nonfinite"]:
        raise RuntimeError(f"MotionFix source z-score coverage has non-finite values: {coverage}")

    manifest: dict[str, Any] = {
        "format": UNIFIED_FORMAT if args.normalize_contacts else LEGACY_FORMAT,
        "train_manifest": str(manifest_path),
        "train_manifest_sha256": sha256_file(manifest_path),
        "accepted_caption_table": str(captions_path),
        "accepted_caption_table_sha256": sha256_file(captions_path),
        "caption_span_policy": "hml_caption_span_30fps_round_half_up_v1",
        "frame_reduction": "sampled_window_valid_frame_weighted",
        "bucket_marginal_policy": "row_uniform_sortish_grouping_v1",
        "domain_task_weights": DOMAIN_WEIGHTS,
        "target_only": True,
        "source_in_stats": False,
        "root_origin_shift": True,
        "yaw_quadrature_targets_rad": list(YAW_TARGETS),
        "fps": float(args.fps),
        "min_frames": 16,
        "contacts_normalized": bool(args.normalize_contacts),
        "combined_contact_prior_before_override": combined_contact_prior.tolist(),
        "combined_contact_std_before_override": combined_contact_std.tolist(),
        "full_domain_report": full_report,
        "local_domain_report": local_report,
        "motionfix_source_zscore_coverage": coverage,
        "dims": {
            "full": DIM_HY273,
            "global_root": ROOT_DIM,
            "body": BODY_DIM,
            "local_root": 4,
        },
    }
    files = [
        output_dir / section / name
        for section in ("full", "global_root", "body", "local_root")
        for name in ("Mean.npy", "Std.npy")
    ]
    manifest["array_sha256"] = {
        str(path.relative_to(output_dir)): sha256_file(path) for path in files
    }
    manifest_path_out = output_dir / "manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
