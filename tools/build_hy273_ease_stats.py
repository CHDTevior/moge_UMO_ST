#!/usr/bin/env python3
"""Build yaw-isotropic HY273 Ease statistics from the HML train manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hy273_ease import (
    EASE_DIM,
    EASE_STATS_FORMAT,
    ease_from_k273,
)
from models.raw_motion.hy273_normalizer import root_origin_shift
from models.raw_motion.hy273_slices import DIM_HY273


def build_stats(manifest_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    weighted_sum = np.zeros(EASE_DIM, dtype=np.float64)
    weighted_square_sum = np.zeros(EASE_DIM, dtype=np.float64)
    total_weight = 0.0
    row_count = 0
    caption_count = 0
    cache: dict[str, np.ndarray] = {}

    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "humanml3d_k273":
                continue
            texts = row.get("texts")
            if not isinstance(texts, list) or not texts:
                raise ValueError(f"HML row has no captions at line {line_number}")
            row_count += 1
            weight = 1.0 / float(len(texts))
            for text_row in texts:
                asset = text_row["target_k273_asset"]
                path = str(Path(asset["path"]).expanduser().resolve())
                label = cache.get(path)
                if label is None:
                    motion = np.load(path).astype(np.float32, copy=False)
                    expected = (int(asset["frames"]), DIM_HY273)
                    if motion.shape != expected or not np.isfinite(motion).all():
                        raise ValueError(
                            f"Invalid K273 Ease source {path}: "
                            f"shape={motion.shape}, expected={expected}"
                        )
                    physical = root_origin_shift(
                        torch.from_numpy(motion.copy())
                    )
                    label = ease_from_k273(physical).double().cpu().numpy()
                    cache[path] = label
                weighted_sum += weight * label
                weighted_square_sum += weight * np.square(label)
                total_weight += weight
                caption_count += 1

    if row_count == 0 or total_weight <= 0:
        raise RuntimeError("No HumanML3D rows were found in the train manifest")
    if not np.isclose(total_weight, float(row_count), rtol=0.0, atol=1e-6):
        raise RuntimeError(
            f"Row-equal weighting drifted: total={total_weight}, rows={row_count}"
        )

    raw_mean = weighted_sum / total_weight
    raw_second = weighted_square_sum / total_weight
    mean = np.zeros(EASE_DIM, dtype=np.float64)
    std = np.zeros(EASE_DIM, dtype=np.float64)
    for offset in (0, 3):
        mean[offset + 1] = raw_mean[offset + 1]
        horizontal_scale = np.sqrt(
            max(
                (raw_second[offset] + raw_second[offset + 2]) / 2.0,
                1e-12,
            )
        )
        vertical_variance = max(
            raw_second[offset + 1] - raw_mean[offset + 1] ** 2,
            1e-12,
        )
        std[offset] = horizontal_scale
        std[offset + 2] = horizontal_scale
        std[offset + 1] = np.sqrt(vertical_variance)

    metadata = {
        "format": EASE_STATS_FORMAT,
        "feature_dim": EASE_DIM,
        "source_manifest": str(manifest_path),
        "dataset": "humanml3d_k273",
        "split": "train",
        "row_count": row_count,
        "caption_occurrences": caption_count,
        "unique_target_assets": len(cache),
        "effective_weight": total_weight,
        "weighting": "motion_row_equal_caption_uniform_within_row",
        "physical_label": (
            "global_joint_centroid_endpoint_exact_half_mean_residual_meters"
        ),
        "yaw_statistics": (
            "per_half_zero_xz_mean_shared_xz_rms_scale_vertical_mean_std"
        ),
        "raw_weighted_mean": raw_mean.tolist(),
        "raw_weighted_second_moment": raw_second.tolist(),
    }
    return mean.astype(np.float32), std.astype(np.float32), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.train_manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    mean, std, metadata = build_stats(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "Mean.npy", mean)
    np.save(output_dir / "Std.npy", std)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "mean": mean.tolist(),
                "std": std.tolist(),
                "row_count": metadata["row_count"],
                "unique_target_assets": metadata["unique_target_assets"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
