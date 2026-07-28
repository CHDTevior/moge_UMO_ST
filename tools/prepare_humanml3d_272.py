"""Prepare a stable local layout for MotionStreamer 272-dim HumanML3D."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


DEFAULT_SOURCE_ROOT = Path("/mnt/afs/MotionMillion/272-dim-HumanML3D")
DEFAULT_OUTPUT_ROOT = Path("dataset/HumanML3D_272")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{motion_id}\n" for motion_id in ids), encoding="utf-8")


def ensure_link(link_path: Path, target_path: Path, force: bool) -> None:
    if link_path.exists() or link_path.is_symlink():
        if not force:
            return
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    relative_target = os.path.relpath(target_path.resolve(), link_path.parent.resolve())
    link_path.symlink_to(relative_target)


def validate_source(root: Path) -> None:
    required = [
        root / "motion_data",
        root / "texts",
        root / "split" / "train.txt",
        root / "split" / "val.txt",
        root / "split" / "test.txt",
        root / "mean_std" / "Mean.npy",
        root / "mean_std" / "Std.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing 272 HumanML3D source paths: {missing}")
    mean = np.load(root / "mean_std" / "Mean.npy")
    std = np.load(root / "mean_std" / "Std.npy")
    if mean.shape != (272,) or std.shape != (272,):
        raise RuntimeError(f"Expected 272-dim stats, got mean={mean.shape}, std={std.shape}")


def build_layout(source_root: Path, output_root: Path, force: bool) -> dict[str, object]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser()
    validate_source(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ensure_link(output_root / "new_joint_vecs", source_root / "motion_data", force)
    ensure_link(output_root / "motion_data", source_root / "motion_data", force)
    ensure_link(output_root / "texts", source_root / "texts", force)
    ensure_link(output_root / "mean_std", source_root / "mean_std", force)
    ensure_link(output_root / "Mean.npy", source_root / "mean_std" / "Mean.npy", force)
    ensure_link(output_root / "Std.npy", source_root / "mean_std" / "Std.npy", force)

    split_raw = output_root / "split_raw"
    split_raw.mkdir(exist_ok=True)
    split_dir = output_root / "split"
    split_dir.mkdir(exist_ok=True)

    motion_ids = {path.stem for path in (source_root / "motion_data").glob("*.npy")}
    text_ids = {path.stem for path in (source_root / "texts").glob("*.txt")}
    summary: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root.resolve()),
        "feature_dim": 272,
        "fps": 30,
        "layout": "motionstreamer_humanml3d_272_for_mogeflow",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        raw_ids = read_ids(source_root / "split" / f"{split}.txt")
        usable_ids = [motion_id for motion_id in raw_ids if motion_id in motion_ids and motion_id in text_ids]
        missing_motion = [motion_id for motion_id in raw_ids if motion_id not in motion_ids]
        missing_text = [motion_id for motion_id in raw_ids if motion_id not in text_ids]
        write_ids(split_raw / f"{split}.txt", raw_ids)
        write_ids(output_root / f"{split}.txt", usable_ids)
        write_ids(split_dir / f"{split}.txt", usable_ids)
        summary["splits"][split] = {
            "raw_ids": len(raw_ids),
            "usable_ids": len(usable_ids),
            "missing_motion": len(missing_motion),
            "missing_text": len(missing_text),
            "first_missing_motion": missing_motion[:20],
            "first_missing_text": missing_text[:20],
        }

    sample_paths = sorted((source_root / "motion_data").glob("*.npy"))[:16]
    sample_shapes = []
    for path in sample_paths:
        arr = np.load(path, mmap_mode="r")
        sample_shapes.append({"file": path.name, "shape": list(arr.shape), "dtype": str(arr.dtype)})
    summary["sample_shapes"] = sample_shapes
    summary_path = output_root / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_layout(args.source_root, args.output_root, args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
