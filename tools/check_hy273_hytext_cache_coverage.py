#!/usr/bin/env python
"""Check that all dataset captions are present in a HYText cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.kimodo273_datasets import parse_hml_text_line
from models.raw_motion.hytext_cache import hytext_key, normalize_text_key


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22")
    p.add_argument("--text_root", default="/mnt/afs/mogo_base/datasets/HumanML3D/texts")
    p.add_argument("--cache_dir", default="/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--output_json", default="")
    p.add_argument("--max_examples", type=int, default=20)
    return p


def read_captions(path: Path) -> list[str]:
    if not path.is_file():
        return [""]
    captions: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = parse_hml_text_line(line)
            if text:
                captions.append(text)
    return captions or [""]


def main() -> None:
    args = build_arg_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    text_root = Path(args.text_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    index_path = cache_dir / "index.json"
    manifest_path = cache_dir / "manifest.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"HYText cache index not found: {index_path}")
    index = json.loads(index_path.read_text())
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    missing: list[dict[str, str]] = []
    by_split: dict[str, dict[str, int]] = {}
    unique_keys: set[str] = set()
    duplicate_caption_hits = 0

    empty_key = hytext_key("")
    if empty_key not in index:
        missing.append({"split": "__empty__", "motion_id": "__empty__", "text": "", "key": empty_key})

    for split in splits:
        split_path = data_root / "split_existing" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        motion_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
        total = 0
        hit = 0
        miss = 0
        for motion_id in motion_ids:
            for caption in read_captions(text_root / f"{motion_id}.txt"):
                norm = normalize_text_key(caption)
                key = hytext_key(norm)
                total += 1
                if key in unique_keys:
                    duplicate_caption_hits += 1
                unique_keys.add(key)
                if key in index:
                    hit += 1
                else:
                    miss += 1
                    if len(missing) < int(args.max_examples):
                        missing.append({"split": split, "motion_id": motion_id, "text": norm, "key": key})
        by_split[split] = {"total": total, "hit": hit, "missing": miss}

    report = {
        "cache_dir": str(cache_dir),
        "format": manifest.get("format"),
        "max_length_llm": manifest.get("max_length_llm"),
        "ctxt_dim": manifest.get("ctxt_dim"),
        "vtxt_dim": manifest.get("vtxt_dim"),
        "index_size": len(index),
        "unique_dataset_caption_keys": len(unique_keys),
        "duplicate_caption_occurrences": duplicate_caption_hits,
        "splits": by_split,
        "empty_key_present": empty_key in index,
        "missing_examples": missing,
        "passed": not missing and all(row["missing"] == 0 for row in by_split.values()),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output_json:
        out = Path(args.output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
