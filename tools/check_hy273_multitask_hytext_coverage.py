#!/usr/bin/env python
"""Fail-closed coverage and row-shape audit for profile-aware HYText cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hytext_cache import HYTextMemmapCache, PROFILE_CACHE_FORMAT
from tools.build_hy273_hytext_profile_cache import PROFILE_PROMPTS, collect_required


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_dir",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/hy273_multitask_v1",
    )
    parser.add_argument(
        "--cache_dir",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/hytext_qwen3_clipL_profile_v2",
    )
    parser.add_argument("--output_json", default="")
    parser.add_argument("--max_examples", type=int, default=20)
    args = parser.parse_args()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache = HYTextMemmapCache(cache_dir, max_open_shards=64, strict=True)
    if cache.manifest.get("format") != PROFILE_CACHE_FORMAT:
        raise ValueError(f"Expected {PROFILE_CACHE_FORMAT}, got {cache.manifest.get('format')}")
    required, manifest_hashes = collect_required(manifest_dir)
    missing = []
    by_profile: Counter[str] = Counter()
    by_profile_hit: Counter[str] = Counter()
    bad_entries = []
    referenced_rows: dict[str, set[int]] = {}
    for profile, text in sorted(required):
        by_profile[profile] += 1
        key = cache._key(text, profile)
        entry = cache.index.get(key)
        if entry is None:
            if len(missing) < int(args.max_examples):
                missing.append({"profile": profile, "text": text, "key": key})
            continue
        by_profile_hit[profile] += 1
        if str(entry.get("profile")) != profile or str(entry.get("text", "")) != text:
            bad_entries.append({"key": key, "expected": [profile, text], "entry": entry})
            continue
        opened = cache._open_shard(str(entry["shard"]))
        row = int(entry["row"])
        referenced_rows.setdefault(str(entry["shard"]), set()).add(row)
        if not (
            0 <= row < opened["ctxt"].shape[0]
            and opened["ctxt"].shape[1:] == (128, 4096)
            and opened["vtxt"].shape[1:] == (1, 768)
        ):
            bad_entries.append({"key": key, "profile": profile, "row": row})
    numeric_rows_checked = 0
    for shard, rows in sorted(referenced_rows.items()):
        opened = cache._open_shard(shard)
        lengths = np.asarray(opened["ctxt_len"])
        if lengths.ndim != 1 or lengths.shape[0] != opened["ctxt"].shape[0]:
            bad_entries.append({"shard": shard, "reason": "ctxt_len_shape"})
            continue
        if (lengths < 0).any() or (lengths > 128).any():
            bad_entries.append({"shard": shard, "reason": "ctxt_len_range"})
        ordered = sorted(rows)
        sample_rows = sorted(
            {
                ordered[0],
                ordered[-1],
                ordered[len(ordered) // 2],
            }
        )
        for row in sample_rows:
            numeric_rows_checked += 1
            if not (
                np.isfinite(np.asarray(opened["ctxt"][row])).all()
                and np.isfinite(np.asarray(opened["vtxt"][row])).all()
            ):
                bad_entries.append(
                    {"shard": shard, "row": row, "reason": "nonfinite_sample"}
                )
    expected_hashes = cache.manifest.get("required_manifest_sha256", {})
    hash_match = expected_hashes == manifest_hashes
    empty_present = {
        profile: cache._key("", profile) in cache.index for profile in PROFILE_PROMPTS
    }
    report = {
        "format": cache.manifest.get("format"),
        "cache_dir": str(cache_dir),
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "cache_index_sha256": sha256_file(cache_dir / "index.json"),
        "required_total": len(required),
        "by_profile_required": dict(by_profile),
        "by_profile_hit": dict(by_profile_hit),
        "empty_present": empty_present,
        "missing_count": len(required) - sum(by_profile_hit.values()),
        "missing_examples": missing,
        "bad_entry_count": len(bad_entries),
        "bad_entry_examples": bad_entries[: int(args.max_examples)],
        "referenced_shards": len(referenced_rows),
        "numeric_rows_checked": numeric_rows_checked,
        "required_manifest_hash_match": hash_match,
        "required_manifest_sha256": manifest_hashes,
        "passed": (
            sum(by_profile_hit.values()) == len(required)
            and not bad_entries
            and all(empty_present.values())
            and hash_match
        ),
    }
    output = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(output)
    if args.output_json:
        path = Path(args.output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
