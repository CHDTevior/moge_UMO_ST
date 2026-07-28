#!/usr/bin/env python3
"""Hash every mutable input used by the HY273 HumanML3D training run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.asset_integrity import sha256_file, write_asset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--text_root", required=True)
    parser.add_argument("--stats_root", required=True)
    parser.add_argument("--hytext_cache_dir", required=True)
    parser.add_argument(
        "--skeleton_asset",
        default="external_repos/kimodo/kimodo/assets/skeletons/smplx22/joints.p",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    text_root = Path(args.text_root).expanduser().resolve()
    stats_root = Path(args.stats_root).expanduser().resolve()
    cache_root = Path(args.hytext_cache_dir).expanduser().resolve()
    split = data_root / "split_existing" / "train.txt"
    motion_ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]

    paths: list[Path] = [
        data_root / "manifest.jsonl",
        split,
        stats_root / "manifest.json",
        stats_root / "full" / "Mean.npy",
        stats_root / "full" / "Std.npy",
        stats_root / "local_root" / "Mean.npy",
        stats_root / "local_root" / "Std.npy",
        cache_root / "manifest.json",
        cache_root / "index.json",
        Path(args.skeleton_asset).expanduser().resolve(),
    ]
    for motion_id in motion_ids:
        paths.append(data_root / "motion_data" / f"{motion_id}.npy")
        paths.append(text_root / f"{motion_id}.txt")
    paths.extend(path for path in (cache_root / "shards").rglob("*") if path.is_file())

    manifest = write_asset_manifest(paths, args.output)
    output_path = Path(args.output).expanduser().resolve()
    print(
        {
            "output": str(output_path),
            "files": len(manifest["files"]),
            "sha256": sha256_file(output_path),
        }
    )


if __name__ == "__main__":
    main()
