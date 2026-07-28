#!/usr/bin/env python3
"""Measure identifiable MotionFix instruction supervision before training."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_multitask_manifest_dataset import _transform_to_gauge
from models.raw_motion.hy273_unified_edit_losses import (
    build_source_target_discrepancy_mask,
)
from models.raw_motion.hy273_slices import (
    DIM_HY273,
    cont6d_to_matrix,
    reconstruct_global_joints_from_features,
    split_global_rot6d,
)


DEFAULT_MANIFEST = Path(
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/train.jsonl"
)
DEFAULT_SAME_SOURCE_GROUPS = Path(
    "outputs/hy273_multitask/diagnostics/r13_edit_objective_pilot_405k_20260722/"
    "tiny_overfit_candidate_groups.json"
)


def normalized_text(row: dict[str, Any]) -> str:
    return " ".join(str(row["texts"][0]["value"]).split()).casefold()


def asset(row: dict[str, Any], role: str) -> dict[str, Any]:
    return row[role]["k273_asset"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") == "motionfix_k273":
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No MotionFix rows in {path}")
    return rows


def load_gauge(path: str) -> torch.Tensor:
    values = np.load(path).astype(np.float32, copy=False)
    if values.ndim != 2 or values.shape[-1] != DIM_HY273:
        raise ValueError(f"Invalid K273 asset {path}: {values.shape}")
    motion, _ = _transform_to_gauge(torch.from_numpy(values.copy()), 0.0)
    return motion


def align_nearest(source: torch.Tensor, target_frames: int) -> torch.Tensor:
    if source.shape[0] == target_frames:
        return source
    if target_frames <= 1:
        return source[:1]
    positions = torch.arange(target_frames, dtype=torch.float64)
    indices = torch.floor(
        positions * float(source.shape[0] - 1) / float(target_frames - 1) + 0.5
    ).long()
    return source[indices]


def load_pair(paths: tuple[str, str]) -> tuple[torch.Tensor, torch.Tensor]:
    source = load_gauge(paths[0])
    target = load_gauge(paths[1])
    return align_nearest(source, target.shape[0]), target


def rotation_error_deg(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = cont6d_to_matrix(first).transpose(-1, -2) @ cont6d_to_matrix(second)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = (0.5 * torch.linalg.vector_norm(skew, dim=-1)).clamp(0.0, 1.0)
    return torch.rad2deg(torch.atan2(sine, cosine))


def summarize(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    quantiles = np.quantile(array, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "q00": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "q50": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q100": float(quantiles[6]),
    }


@torch.inference_mode()
def changed_metrics_batch(
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    position_threshold_m: float,
    rotation_threshold_deg: float,
    dilation_frames: int,
) -> list[dict[str, float | int | bool]]:
    max_frames = max(target.shape[0] for _, target in pairs)
    batch = len(pairs)
    source = torch.zeros(batch, max_frames, DIM_HY273, dtype=torch.float32)
    target = torch.zeros_like(source)
    valid = torch.zeros(batch, max_frames, dtype=torch.bool)
    lengths = []
    for index, (source_row, target_row) in enumerate(pairs):
        frames = int(target_row.shape[0])
        source[index, :frames] = source_row
        target[index, :frames] = target_row
        valid[index, :frames] = True
        lengths.append(frames)
    source = source.to(device)
    target = target.to(device)
    valid = valid.to(device)
    source_joints = reconstruct_global_joints_from_features(source)
    target_joints = reconstruct_global_joints_from_features(target)
    position_delta = torch.linalg.vector_norm(source_joints - target_joints, dim=-1)
    rotation_delta = rotation_error_deg(
        split_global_rot6d(source), split_global_rot6d(target)
    )
    position_changed = position_delta > float(position_threshold_m)
    rotation_changed = rotation_delta > float(rotation_threshold_deg)
    changed_seed = position_changed | rotation_changed
    changed_seed &= valid[..., None]
    position_changed &= valid[..., None]
    rotation_changed &= valid[..., None]
    if int(dilation_frames) > 0:
        changed = F.max_pool1d(
            changed_seed.transpose(1, 2).float(),
            kernel_size=2 * int(dilation_frames) + 1,
            stride=1,
            padding=int(dilation_frames),
        ).transpose(1, 2).bool()
        changed &= valid[..., None]
    else:
        changed = changed_seed

    discrepancy = build_source_target_discrepancy_mask(
        source_physical=source,
        source_lengths=torch.as_tensor(lengths, device=device),
        target_physical=target,
        target_valid=valid,
        hard_mask=torch.zeros_like(target, dtype=torch.bool),
        fraction=0.20,
    )

    output = []
    for index, frames in enumerate(lengths):
        mask = changed[index, :frames]
        seed = changed_seed[index, :frames]
        position_mask = position_changed[index, :frames]
        rotation_mask = rotation_changed[index, :frames]
        valid_entries = float(frames * mask.shape[-1])
        valid_coordinates = float(frames * discrepancy.mask.shape[-1])
        output.append(
            {
                "frames": frames,
                "changed_nonempty": bool(mask.any().item()),
                "changed_seed_joint_frame_fraction": float(
                    seed.sum().item() / valid_entries
                ),
                "position_changed_joint_frame_fraction": float(
                    position_mask.sum().item() / valid_entries
                ),
                "rotation_changed_joint_frame_fraction": float(
                    rotation_mask.sum().item() / valid_entries
                ),
                "changed_joint_frame_fraction": float(mask.sum().item() / valid_entries),
                "changed_frame_fraction": float(mask.any(dim=-1).float().mean().item()),
                "top20_discrepancy_coordinate_fraction": float(
                    discrepancy.mask[index, :frames].sum().item() / valid_coordinates
                ),
                "top20_root_time_fraction": float(
                    discrepancy.root_time_mask[index, :frames].float().mean().item()
                ),
                "top20_body_joint_time_fraction": float(
                    discrepancy.body_joint_time_mask[index, :frames].float().mean().item()
                ),
                "mean_position_delta_m": float(
                    position_delta[index, :frames].mean().item()
                ),
                "mean_rotation_delta_deg": float(
                    rotation_delta[index, :frames].mean().item()
                ),
            }
        )
    return output


def analyze_pairs(
    path_pairs: list[tuple[str, str]],
    *,
    workers: int,
    batch_size: int,
    device: torch.device,
    position_threshold_m: float,
    rotation_threshold_deg: float,
    dilation_frames: int,
) -> list[dict[str, float | int | bool]]:
    output: list[dict[str, float | int | bool]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for start in range(0, len(path_pairs), int(batch_size)):
            chunk = path_pairs[start : start + int(batch_size)]
            loaded = list(pool.map(load_pair, chunk))
            output.extend(
                changed_metrics_batch(
                    loaded,
                    device=device,
                    position_threshold_m=position_threshold_m,
                    rotation_threshold_deg=rotation_threshold_deg,
                    dilation_frames=dilation_frames,
                )
            )
            if (start // int(batch_size)) % 10 == 0:
                print(f"[coverage] {min(start + len(chunk), len(path_pairs))}/{len(path_pairs)}", flush=True)
    return output


def subset_summary(
    records: list[dict[str, float | int | bool]], indices: list[int]
) -> dict[str, Any]:
    selected = [records[index] for index in indices]
    total_joint_frames = sum(int(row["frames"]) * 22 for row in selected)
    changed_joint_frames = sum(
        float(row["changed_joint_frame_fraction"]) * int(row["frames"]) * 22
        for row in selected
    )
    return {
        "pairs": len(selected),
        "pairs_with_nonempty_changed_region": sum(
            bool(row["changed_nonempty"]) for row in selected
        ),
        "frames": sum(int(row["frames"]) for row in selected),
        "frame_weighted_changed_joint_fraction": (
            float(changed_joint_frames / total_joint_frames)
            if total_joint_frames
            else 0.0
        ),
        "changed_joint_frame_fraction": summarize(
            float(row["changed_joint_frame_fraction"]) for row in selected
        ),
        "position_changed_joint_frame_fraction": summarize(
            float(row["position_changed_joint_frame_fraction"]) for row in selected
        ),
        "rotation_changed_joint_frame_fraction": summarize(
            float(row["rotation_changed_joint_frame_fraction"]) for row in selected
        ),
        "changed_frame_fraction": summarize(
            float(row["changed_frame_fraction"]) for row in selected
        ),
        "top20_discrepancy_coordinate_fraction": summarize(
            float(row["top20_discrepancy_coordinate_fraction"]) for row in selected
        ),
        "top20_root_time_fraction": summarize(
            float(row["top20_root_time_fraction"]) for row in selected
        ),
        "top20_body_joint_time_fraction": summarize(
            float(row["top20_body_joint_time_fraction"]) for row in selected
        ),
        "mean_position_delta_m": summarize(
            float(row["mean_position_delta_m"]) for row in selected
        ),
        "mean_rotation_delta_deg": summarize(
            float(row["mean_rotation_delta_deg"]) for row in selected
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--same_source_groups", type=Path, default=DEFAULT_SAME_SOURCE_GROUPS
    )
    parser.add_argument("--minimum_target_pair_mse", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--changed_position_threshold_m", type=float, default=0.02)
    parser.add_argument("--changed_rotation_threshold_deg", type=float, default=5.0)
    parser.add_argument("--changed_temporal_dilation_frames", type=int, default=2)
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    rows = load_rows(manifest)
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[str(asset(row, "source_motion")["sha256"])].append(index)

    eligible_indices: set[int] = set()
    equal_length_sibling_indices: set[int] = set()
    eligible_sources = 0
    sibling_target_pairs: list[tuple[str, str]] = []
    sibling_group_rows: list[dict[str, Any]] = []
    for source_hash, indices in by_source.items():
        group = [rows[index] for index in indices]
        target_hashes = {str(asset(row, "target_motion")["sha256"]) for row in group}
        texts = {normalized_text(row) for row in group}
        if len(target_hashes) < 2 or len(texts) < 2:
            continue
        eligible_sources += 1
        eligible_indices.update(indices)
        equal_rows = []
        for index in indices:
            target_frames = int(asset(rows[index], "target_motion")["frames"])
            if any(
                other != index
                and int(asset(rows[other], "target_motion")["frames"]) == target_frames
                for other in indices
            ):
                equal_length_sibling_indices.add(index)
                equal_rows.append(index)
        pair_count = 0
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                left_target = asset(rows[left], "target_motion")
                right_target = asset(rows[right], "target_motion")
                if int(left_target["frames"]) != int(right_target["frames"]):
                    continue
                sibling_target_pairs.append((str(left_target["path"]), str(right_target["path"])))
                pair_count += 1
        sibling_group_rows.append(
            {
                "source_sha256": source_hash,
                "pairs": len(indices),
                "equal_length_eligible_pairs": len(equal_rows),
                "target_comparisons": pair_count,
            }
        )

    path_pairs = [
        (
            str(asset(row, "source_motion")["path"]),
            str(asset(row, "target_motion")["path"]),
        )
        for row in rows
    ]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    pair_metrics = analyze_pairs(
        path_pairs,
        workers=int(args.workers),
        batch_size=int(args.batch_size),
        device=device,
        position_threshold_m=float(args.changed_position_threshold_m),
        rotation_threshold_deg=float(args.changed_rotation_threshold_deg),
        dilation_frames=int(args.changed_temporal_dilation_frames),
    )
    sibling_metrics = analyze_pairs(
        sibling_target_pairs,
        workers=int(args.workers),
        batch_size=int(args.batch_size),
        device=device,
        position_threshold_m=float(args.changed_position_threshold_m),
        rotation_threshold_deg=float(args.changed_rotation_threshold_deg),
        dilation_frames=int(args.changed_temporal_dilation_frames),
    )

    all_indices = list(range(len(rows)))
    eligible_sorted = sorted(eligible_indices)
    equal_sibling_sorted = sorted(equal_length_sibling_indices)
    effective = [
        index for index in eligible_sorted if bool(pair_metrics[index]["changed_nonempty"])
    ]
    total_frames = sum(int(asset(row, "target_motion")["frames"]) for row in rows)
    eligible_frames = sum(
        int(asset(rows[index], "target_motion")["frames"]) for index in eligible_sorted
    )
    eligible_proxy_coordinates = sum(
        float(pair_metrics[index]["top20_discrepancy_coordinate_fraction"])
        * int(asset(rows[index], "target_motion")["frames"])
        for index in eligible_sorted
    )
    eligible_proxy_frame_fraction = eligible_proxy_coordinates / total_frames

    group_path = args.same_source_groups.expanduser().resolve()
    group_payload = json.loads(group_path.read_text(encoding="utf-8"))
    if not isinstance(group_payload, list):
        raise ValueError("--same_source_groups must contain a JSON list")
    threshold_rows: dict[str, dict[str, float | int]] = {}
    for threshold in (0.0, 0.01, 0.1, 1.0, 3.0):
        selected_groups = [
            row
            for row in group_payload
            if float(row.get("target_pair_mse", float("-inf"))) >= threshold
            and len({" ".join(str(text).split()).casefold() for text in row.get("texts", [])})
            == 2
        ]
        threshold_rows[str(threshold)] = {
            "groups": len(selected_groups),
            "eligible_rows": 2 * len(selected_groups),
            "pair_fraction": 2 * len(selected_groups) / len(rows),
        }
    strong_pair_ids = {
        str(pair_id)
        for group in group_payload
        if float(group.get("target_pair_mse", float("-inf")))
        >= float(args.minimum_target_pair_mse)
        and len(
            {" ".join(str(text).split()).casefold() for text in group.get("texts", [])}
        )
        == 2
        for pair_id in group["pair_ids"]
    }
    strong_indices = [
        index
        for index, row in enumerate(rows)
        if str(row["pair"]["official_pair_id"]) in strong_pair_ids
    ]
    strong_frames = sum(
        int(asset(rows[index], "target_motion")["frames"]) for index in strong_indices
    )
    strong_proxy_coordinates = sum(
        float(pair_metrics[index]["top20_discrepancy_coordinate_fraction"])
        * int(asset(rows[index], "target_motion")["frames"])
        for index in strong_indices
    )
    strong_proxy_frame_fraction = strong_proxy_coordinates / total_frames
    result = {
        "format": "hy273_edit_identifiable_supervision_coverage_v1",
        "manifest": str(manifest),
        "thresholds": {
            "changed_position_m": float(args.changed_position_threshold_m),
            "changed_rotation_deg": float(args.changed_rotation_threshold_deg),
            "temporal_dilation_frames": int(args.changed_temporal_dilation_frames),
            "gauge": "independent frame-0 root-origin shift then shared target heading 0",
            "unequal_length_alignment": "normalized-progress nearest frame; reported as coverage proxy",
        },
        "manifest_coverage": {
            "pairs": len(rows),
            "unique_sources": len(by_source),
            "eligible_sibling_sources": eligible_sources,
            "eligible_sibling_pairs": len(eligible_sorted),
            "eligible_sibling_pair_fraction": len(eligible_sorted) / len(rows),
            "eligible_sibling_frame_fraction": eligible_frames / total_frames,
            "equal_target_length_sibling_pairs": len(equal_sibling_sorted),
            "equal_target_length_sibling_pair_fraction": len(equal_sibling_sorted) / len(rows),
            "identifiable_changed_pairs": len(effective),
            "identifiable_changed_pair_fraction": len(effective) / len(rows),
            "eligible_top20_proxy_frame_coordinate_fraction_of_all_edit_data": (
                eligible_proxy_frame_fraction
            ),
            "eligible_top20_proxy_exposure_at_p_edit_0_30": (
                0.30 * eligible_proxy_frame_fraction
            ),
            "strong_sibling_minimum_target_pair_mse": float(
                args.minimum_target_pair_mse
            ),
            "strong_sibling_pairs": len(strong_indices),
            "strong_sibling_pair_fraction": len(strong_indices) / len(rows),
            "strong_sibling_frame_fraction": strong_frames / total_frames,
            "strong_sibling_top20_proxy_frame_coordinate_fraction_of_all_edit_data": (
                strong_proxy_frame_fraction
            ),
            "strong_sibling_top20_proxy_exposure_at_p_edit_0_30": (
                0.30 * strong_proxy_frame_fraction
            ),
        },
        "source_target_change": {
            "all_pairs": subset_summary(pair_metrics, all_indices),
            "sibling_eligible_pairs": subset_summary(pair_metrics, eligible_sorted),
            "equal_length_sibling_eligible_pairs": subset_summary(
                pair_metrics, equal_sibling_sorted
            ),
        },
        "sibling_target_separation": {
            "equal_length_target_comparisons": len(sibling_metrics),
            "comparisons_with_nonempty_changed_region": sum(
                bool(row["changed_nonempty"]) for row in sibling_metrics
            ),
            "changed_joint_frame_fraction": summarize(
                float(row["changed_joint_frame_fraction"]) for row in sibling_metrics
            ),
            "changed_frame_fraction": summarize(
                float(row["changed_frame_fraction"]) for row in sibling_metrics
            ),
            "fraction_at_least_1pct_changed_joints": float(
                np.mean(
                    [float(row["changed_joint_frame_fraction"]) >= 0.01 for row in sibling_metrics]
                )
            ),
            "fraction_at_least_5pct_changed_joints": float(
                np.mean(
                    [float(row["changed_joint_frame_fraction"]) >= 0.05 for row in sibling_metrics]
                )
            ),
            "fraction_at_least_10pct_changed_joints": float(
                np.mean(
                    [float(row["changed_joint_frame_fraction"]) >= 0.10 for row in sibling_metrics]
                )
            ),
        },
        "training_donor_filter": {
            "group_file": str(group_path),
            "minimum_target_pair_mse": float(args.minimum_target_pair_mse),
            "coverage_by_normalized_continuous_target_pair_mse": threshold_rows,
            "selected_pairs": subset_summary(pair_metrics, strong_indices),
            "target_pair_mse": summarize(
                float(row["target_pair_mse"])
                for row in group_payload
                if float(row.get("target_pair_mse", float("-inf")))
                >= float(args.minimum_target_pair_mse)
            ),
        },
        "sibling_groups": sibling_group_rows,
        "interpretation": {
            "identifiable_signal": "same source + distinct instruction/target + nonempty physical changed region",
            "not_claimed": "a discrepancy mask is not a semantic part label and cannot prove instruction correctness alone",
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["manifest_coverage"], indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
