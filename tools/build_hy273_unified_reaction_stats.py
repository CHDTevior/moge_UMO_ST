#!/usr/bin/env python3
"""Build target-only shared stats for T2M/Edit/Inter-X Reaction training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.hy273_reaction_dataset import apply_shared_reaction_gauge
from models.raw_motion.hy273_slices import BODY_DIM, DIM_HY273, ROOT_DIM
from tools.build_hy273_redenoise_stats import global_root_to_local, save_stats
from tools.build_hy273_unified_actor_stats import (
    WeightedMoments,
    YAW_TARGETS,
    _accumulate_single,
    _combine,
    _load_k273,
    _load_rows,
)


STATS_FORMAT = "hy273_unified_reaction_target_stats_v1"


def _accumulate_reactor(
    full: WeightedMoments,
    local: WeightedMoments,
    root: Path,
    row: dict,
    actor_index: int,
    fps: float,
) -> None:
    frames = int(row["frames"])
    people = (
        _load_k273((root / str(row["person1"])).resolve()),
        _load_k273((root / str(row["person2"])).resolve()),
    )
    if people[0].shape[0] != frames or people[1].shape[0] != frames:
        raise ValueError(f"Inter-X frame metadata mismatch for {row['clip_id']}")
    source = people[int(actor_index)]
    reactor = people[1 - int(actor_index)]
    for phi in YAW_TARGETS:
        _, target, _ = apply_shared_reaction_gauge(source, reactor, float(phi))
        weight = 1.0 / len(YAW_TARGETS)
        full.update(target.numpy(), weight)
        local.update(
            global_root_to_local(target[:, :ROOT_DIM], fps).numpy(),
            weight,
        )


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
        "--reaction_root",
        default="/mnt/afs/mogo_base/datasets/InteractionK273/interx",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/mnt/afs/mogo_base/datasets/HY273_unified_reaction_v1/"
            "derived_stats/t2m30_edit35_reaction35"
        ),
    )
    parser.add_argument("--t2m_weight", type=float, default=0.30)
    parser.add_argument("--edit_weight", type=float, default=0.35)
    parser.add_argument("--reaction_weight", type=float, default=0.35)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min_frames", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument(
        "--exclude_overlength",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    weights = {
        "humanml3d_k273": float(args.t2m_weight),
        "motionfix_k273": float(args.edit_weight),
        "interx_reactor_k273": float(args.reaction_weight),
    }
    if min(weights.values()) < 0.0 or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Stats weights must be non-negative and sum to one")
    manifest_path = Path(args.multitask_train_manifest).expanduser().resolve()
    reaction_root = Path(args.reaction_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(manifest_path)
    hml_rows = [row for row in rows if row["dataset"] == "humanml3d_k273"]
    edit_rows = [row for row in rows if row["dataset"] == "motionfix_k273"]
    reaction_rows = [
        row
        for row in _load_rows(reaction_root / "manifest.jsonl")
        if row.get("dataset") == "interx"
        and row.get("split") == "train"
        and bool(row.get("has_text"))
        and row.get("texts")
        and int(row.get("frames", 0)) >= int(args.min_frames)
        and (
            not bool(args.exclude_overlength)
            or int(row.get("frames", 0)) <= int(args.max_frames)
        )
    ]
    with (reaction_root / "interaction_order.pkl").open("rb") as handle:
        order = pickle.load(handle)
    if not hml_rows or not edit_rows or not reaction_rows:
        raise RuntimeError("T2M, Edit, and Reaction train domains are all required")

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
        if index % 2000 == 0 or index == len(hml_rows):
            print(f"[stats] HML {index}/{len(hml_rows)}", flush=True)

    for index, row in enumerate(edit_rows, 1):
        _accumulate_single(
            full_domains["motionfix_k273"],
            local_domains["motionfix_k273"],
            Path(row["target_motion"]["k273_asset"]["path"]),
            1.0,
            fps,
        )
        if index % 1000 == 0 or index == len(edit_rows):
            print(f"[stats] MotionFix {index}/{len(edit_rows)}", flush=True)

    for index, row in enumerate(reaction_rows, 1):
        clip_id = str(row["clip_id"])
        if clip_id not in order:
            raise KeyError(f"Missing actor/reactor order for {clip_id}")
        _accumulate_reactor(
            full_domains["interx_reactor_k273"],
            local_domains["interx_reactor_k273"],
            reaction_root,
            row,
            int(order[clip_id]),
            fps,
        )
        if index % 1000 == 0 or index == len(reaction_rows):
            print(f"[stats] Inter-X Reaction {index}/{len(reaction_rows)}", flush=True)

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

    report = {
        "format": STATS_FORMAT,
        "contacts_normalized": True,
        "target_only": True,
        "source_in_stats": False,
        "fps": fps,
        "domain_task_weights": weights,
        "stage_b_task_mix": {"t2m": 0.30, "edit": 0.35, "reaction": 0.35},
        "reaction_dataset": "Inter-X only",
        "reaction_target_role": "reactor selected by interaction_order.pkl",
        "reaction_source_role": "actor excluded from target moments",
        "reaction_gauge": "shared source-root-centered/source-heading yaw",
        "exclude_overlength": bool(args.exclude_overlength),
        "max_frames": int(args.max_frames),
        "full_domain_report": full_report,
        "local_domain_report": local_report,
        "row_counts": {
            "humanml3d_k273": len(hml_rows),
            "motionfix_k273": len(edit_rows),
            "interx_reactor_k273": len(reaction_rows),
        },
        "dims": {
            "full": DIM_HY273,
            "global_root": ROOT_DIM,
            "body": BODY_DIM,
            "local_root": 4,
        },
        "sources": {
            "multitask_train_manifest": str(manifest_path),
            "reaction_manifest": str(reaction_root / "manifest.jsonl"),
            "interaction_order": str(reaction_root / "interaction_order.pkl"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
