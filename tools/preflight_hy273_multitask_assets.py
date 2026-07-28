"""Full payload preflight for the frozen HY273 multitask asset bundle."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_slices import CONTACT_SLICE, DIM_HY273
from train_hy273_multitask import (
    canonical_sha,
    load_config,
    sha256_file,
    validate_assets,
    validate_frozen_contract,
)


FORMAT = "hy273_multitask_full_asset_preflight_v1"


def _asset_refs(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    refs = [("target", row["target_motion"]["k273_asset"])]
    if row.get("source_motion") is not None:
        refs.append(("source", row["source_motion"]["k273_asset"]))
    for text in row.get("texts", []):
        if "target_k273_asset" in text:
            refs.append(("caption_target", text["target_k273_asset"]))
    return refs


def collect_assets(manifest_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        row_count = 0
        ref_count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row_count += 1
                for role, ref in _asset_refs(row):
                    ref_count += 1
                    resolved = str(Path(ref["path"]).expanduser().resolve())
                    record = {
                        "path": resolved,
                        "sha256": str(ref["sha256"]),
                        "frames": int(ref["frames"]),
                        "feature_dim": int(ref["feature_dim"]),
                        "fps": float(ref["fps"]),
                    }
                    existing = assets.get(resolved)
                    if existing is not None and existing != record:
                        raise RuntimeError(
                            f"Conflicting manifest metadata for {resolved}: {existing}/{record}"
                        )
                    assets[resolved] = record
        split_counts[split] = {"rows": row_count, "asset_references": ref_count}
    return assets, split_counts


def verify_asset(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Training payload is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != record["sha256"]:
        raise RuntimeError(
            f"Training payload SHA mismatch: {path} expected={record['sha256']} actual={actual_sha}"
        )
    array = np.load(path, mmap_mode="r")
    expected_shape = (record["frames"], record["feature_dim"])
    if array.shape != expected_shape or array.shape[-1] != DIM_HY273:
        raise RuntimeError(f"Training payload shape mismatch: {path} {array.shape}/{expected_shape}")
    if array.dtype != np.float32:
        raise RuntimeError(f"Training payload dtype must be float32: {path} {array.dtype}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"Training payload contains non-finite values: {path}")
    contacts = array[:, CONTACT_SLICE]
    if not np.logical_or(contacts == 0.0, contacts == 1.0).all():
        raise RuntimeError(f"Training payload contacts are not exact binary 0/1: {path}")
    return {
        **record,
        "actual_sha256": actual_sha,
        "bytes": int(path.stat().st_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hy273_multitask_stage_a_t2m.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    validate_frozen_contract(config)
    asset_identity = validate_assets(config, include_full_preflight=False)
    manifest_dir = Path(config["data"]["manifest_dir"]).expanduser().resolve()
    assets, split_counts = collect_assets(manifest_dir)
    started = time.perf_counter()
    ordered = [assets[path] for path in sorted(assets)]
    verified: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        for index, record in enumerate(pool.map(verify_asset, ordered), 1):
            verified.append(record)
            if index % 1000 == 0 or index == len(ordered):
                print(
                    json.dumps(
                        {
                            "verified": index,
                            "total": len(ordered),
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    report = {
        "format": FORMAT,
        "status": "passed",
        "config_path": str(config_path),
        "asset_identity": asset_identity,
        "manifest_dir": str(manifest_dir),
        "split_counts": split_counts,
        "unique_k273_payloads": len(verified),
        "total_payload_bytes": sum(record["bytes"] for record in verified),
        "payload_records_sha256": canonical_sha(verified),
        "scanner_code_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_content_sha256"] = canonical_sha(report)
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **report}))


if __name__ == "__main__":
    main()
