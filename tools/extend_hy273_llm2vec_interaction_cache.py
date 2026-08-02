#!/usr/bin/env python3
"""Extend the existing K-Encoder cache with Interaction text rows only."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hy273_multitask_condition import INTERACTION_TEXT_PROFILE
from models.raw_motion.hytext_cache import (
    LLM2VEC_CACHE_FORMAT,
    hytext_profile_key,
    normalize_text_key,
)
from tools.cache_hy273_llm2vec_embeddings import (
    PROMPT_TEMPLATE_VERSION,
    _encode_worker,
    canonical_sha,
)
from tools.build_hy273_hytext_profile_cache import sha256_file


INTERACTION_PROFILE_DESCRIPTION = "raw two-person interaction caption"


def _link_or_copy_tree(source: Path, target: Path) -> None:
    def link_or_copy(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(source, target, copy_function=link_or_copy)


def _collect_interaction_rows(
    interaction_root: Path,
    *,
    min_frames: int,
) -> tuple[list[dict[str, str]], set[str]]:
    manifest_path = interaction_root / "manifest.jsonl"
    required: dict[str, set[str]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not bool(row.get("has_text"))
                or not row.get("texts")
                or int(row.get("frames", 0)) < int(min_frames)
            ):
                continue
            for text_index, text in enumerate(row["texts"]):
                normalized = normalize_text_key(str(text))
                source = (
                    f"interaction:{row['split']}:{row['clip_id']}:text{text_index}"
                )
                required.setdefault(normalized, set()).add(source)
    required.setdefault("", set()).add("interaction:cfg_empty")
    return (
        [
            {
                "text": text,
                "source": sorted(sources)[0],
            }
            for text, sources in sorted(required.items())
        ],
        set(required),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_cache",
        default=(
            "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/"
            "llm2vec_llama3_8b_profile_v1"
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
            "llm2vec_llama3_8b_profile_interaction_v1"
        ),
    )
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7",
    )
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--min_frames", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base_cache = Path(args.base_cache).expanduser().resolve()
    interaction_root = Path(args.interaction_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not (base_cache / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing base LLM2Vec cache: {base_cache}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output cache exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "shards").mkdir()

    base_manifest = json.loads((base_cache / "manifest.json").read_text())
    if base_manifest.get("format") != LLM2VEC_CACHE_FORMAT:
        raise ValueError("Base cache is not a Kimodo LLM2Vec cache")
    if base_manifest.get("prompt_template_version") != PROMPT_TEMPLATE_VERSION:
        raise ValueError("Base cache prompt template differs from this encoder")
    if int(base_manifest.get("encoding_batch_size", -1)) != 1:
        raise ValueError("Base cache was not encoded with Kimodo batch_size=1")
    encoder_identity = str(base_manifest["encoder_identity"])
    base_index: dict[str, dict[str, Any]] = json.loads(
        (base_cache / "index.json").read_text()
    )
    base_profile_rows = [
        json.loads(line)
        for line in (base_cache / "profile_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    base_shards = sorted((base_cache / "shards").glob("shard_*"))
    if not base_shards:
        raise RuntimeError("Base cache has no shards")
    for shard in base_shards:
        _link_or_copy_tree(shard, output_dir / "shards" / shard.name)

    collected, required_texts = _collect_interaction_rows(
        interaction_root,
        min_frames=int(args.min_frames),
    )
    rows = [
        {
            "key": hytext_profile_key(
                row["text"],
                INTERACTION_TEXT_PROFILE,
                encoder_identity=encoder_identity,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
            "profile": INTERACTION_TEXT_PROFILE,
            **row,
        }
        for row in collected
    ]
    rows = [row for row in rows if row["key"] not in base_index]
    if args.limit > 0:
        rows = rows[: int(args.limit)]
    devices = [value.strip() for value in str(args.devices).split(",") if value.strip()]
    if not devices:
        raise ValueError("At least one encoding device is required")
    start_shard = (
        max(int(path.name.rsplit("_", 1)[1]) for path in base_shards) + 1
    )
    shard_specs = [
        (start_shard + offset, start, rows[start : start + int(args.shard_size)])
        for offset, start in enumerate(range(0, len(rows), int(args.shard_size)))
    ]
    assignments = [[] for _ in devices]
    for index, shard_spec in enumerate(shard_specs):
        assignments[index % len(devices)].append(shard_spec)
    worker_kwargs = [
        {
            "worker_id": worker_id,
            "device": device,
            "shard_specs": assignments[worker_id],
            "output_dir": str(output_dir),
            "base_model": str(base_manifest["base_model"]),
            "supervised_model": str(base_manifest["supervised_model"]),
            "model_dtype": str(base_manifest["model_dtype"]),
            "batch_size": 1,
            "storage_dtype": str(base_manifest["storage_dtype"]),
            "total_rows": len(rows),
        }
        for worker_id, device in enumerate(devices)
        if assignments[worker_id]
    ]
    completed: set[int] = set()
    if len(worker_kwargs) == 1:
        completed.update(_encode_worker(**worker_kwargs[0]))
    elif worker_kwargs:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(worker_kwargs),
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(_encode_worker, **kwargs)
                for kwargs in worker_kwargs
            ]
            for future in as_completed(futures):
                completed.update(future.result())
    expected = {shard_id for shard_id, _, _ in shard_specs}
    if completed != expected:
        raise RuntimeError("Interaction cache workers did not complete every shard")

    index = dict(base_index)
    profile_rows = list(base_profile_rows)
    for shard_id, _, shard_rows in shard_specs:
        shard_name = f"shard_{shard_id:05d}"
        for row_index, row in enumerate(shard_rows):
            entry = {
                "shard": shard_name,
                "row": row_index,
                "text": row["text"],
                "profile": row["profile"],
            }
            index[row["key"]] = entry
            profile_rows.append(
                {
                    "key": row["key"],
                    **entry,
                    "source": row["source"],
                }
            )
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "profile_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(profile_rows, key=lambda item: item["key"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    profiles = dict(base_manifest["profiles"])
    profiles[INTERACTION_TEXT_PROFILE] = INTERACTION_PROFILE_DESCRIPTION
    prompt_hashes = dict(base_manifest["profile_prompt_sha256"])
    prompt_hashes[INTERACTION_TEXT_PROFILE] = hashlib.sha256(
        INTERACTION_PROFILE_DESCRIPTION.encode("utf-8")
    ).hexdigest()
    empty_keys = dict(base_manifest["profile_empty_keys"])
    empty_keys[INTERACTION_TEXT_PROFILE] = hytext_profile_key(
        "",
        INTERACTION_TEXT_PROFILE,
        encoder_identity=encoder_identity,
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
    )
    manifest_hashes = dict(base_manifest.get("required_manifest_sha256", {}))
    manifest_hashes["interaction_manifest.jsonl"] = sha256_file(
        interaction_root / "manifest.jsonl"
    )
    index_sha = sha256_file(output_dir / "index.json")
    profile_rows_sha = sha256_file(output_dir / "profile_rows.jsonl")
    manifest = {
        **base_manifest,
        "profiles": profiles,
        "profile_prompt_sha256": prompt_hashes,
        "profile_empty_keys": empty_keys,
        "required_manifest_sha256": manifest_hashes,
        "required_pair_count": len(index),
        "num_texts": len(index),
        "index_sha256": index_sha,
        "profile_rows_sha256": profile_rows_sha,
        "encoding_devices": devices,
        "base_cache": str(base_cache),
        "base_cache_rows": len(base_index),
        "interaction_required_unique_texts": len(required_texts),
        "interaction_encoded_rows": len(rows),
        "interaction_min_frames": int(args.min_frames),
    }
    manifest["profile_contract_sha256"] = canonical_sha(
        {
            "encoder_identity": encoder_identity,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "profile_prompt_sha256": prompt_hashes,
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    missing = []
    for text in required_texts:
        key = hytext_profile_key(
            text,
            INTERACTION_TEXT_PROFILE,
            encoder_identity=encoder_identity,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        if key not in index:
            missing.append(text)
    coverage = {
        "format": "hy273_unified_actor_text_cache_coverage_v1",
        "passed": not missing and not args.limit,
        "base_rows": len(base_index),
        "interaction_required": len(required_texts),
        "interaction_encoded": len(rows),
        "cache_rows": len(index),
        "missing": len(missing),
        "missing_examples": missing[:20],
        "cache_index_sha256": index_sha,
    }
    (output_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(index),
                "interaction_rows": len(rows),
                "coverage_passed": coverage["passed"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
