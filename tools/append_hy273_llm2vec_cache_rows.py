#!/usr/bin/env python
"""Append a small, explicit set of frozen LLM2Vec rows to paired caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hytext_cache import HYTextMemmapCache, hytext_profile_key
from models.raw_motion.llm2vec_context_cache import LLM2VecContextMemmapCache
from tools.cache_hy273_hytext_embeddings import write_shard
from tools.hy273_runtime_text_encoding import encode_missing_text_rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _next_shard_id(cache_dir: Path) -> int:
    ids = [
        int(path.name.rsplit("_", 1)[1])
        for path in (cache_dir / "shards").glob("shard_*")
    ]
    return max(ids, default=-1) + 1


def _load_texts(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, str) for row in payload):
        raise ValueError("--texts_json must contain a JSON list of strings")
    texts = []
    seen = set()
    for value in payload:
        text = " ".join(value.strip().split())
        if text and text not in seen:
            texts.append(text)
            seen.add(text)
    if not texts:
        raise ValueError("No non-empty text rows were provided")
    return texts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_cache", required=True)
    parser.add_argument("--context_cache", required=True)
    parser.add_argument("--texts_json", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--source", default="explicit_eval_cache_supplement")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    global_dir = Path(args.global_cache).expanduser().resolve()
    context_dir = Path(args.context_cache).expanduser().resolve()
    texts = _load_texts(Path(args.texts_json).expanduser().resolve())
    profiles = [str(args.profile)] * len(texts)

    global_cache = HYTextMemmapCache(global_dir, max_open_shards=1, strict=True)
    context_cache = LLM2VecContextMemmapCache(
        context_dir,
        embedding_dim=int(global_cache.manifest["ctxt_dim"]),
        max_open_shards=1,
        strict=True,
    )
    if str(args.profile) not in global_cache.manifest.get("profiles", {}):
        raise ValueError(f"Unknown profile {args.profile!r}")
    already_present = [
        global_cache.has_text(text, args.profile)
        and context_cache.has_text(text, args.profile)
        for text in texts
    ]
    if all(already_present):
        print(json.dumps({"appended": 0, "already_present": len(texts)}, indent=2))
        return
    if any(already_present):
        texts = [text for text, present in zip(texts, already_present) if not present]
        profiles = [str(args.profile)] * len(texts)

    rows = encode_missing_text_rows(
        global_cache,
        texts,
        profiles,
        torch.device(args.device),
        context_cache=context_cache,
    )
    if rows.count != len(texts):
        raise RuntimeError(f"Encoded {rows.count} rows, expected {len(texts)}")
    if any(value is None for value in (rows.vtxt, rows.ctxt, rows.lengths)):
        raise RuntimeError("Global LLM2Vec rows are incomplete")
    if rows.contextual is None or rows.contextual_lengths is None:
        raise RuntimeError("Contextual LLM2Vec rows are incomplete")

    global_manifest = dict(global_cache.manifest)
    encoder_identity = str(global_manifest["encoder_identity"])
    prompt_version = str(global_manifest["prompt_template_version"])
    cache_rows = [
        {
            "key": hytext_profile_key(
                text,
                args.profile,
                encoder_identity=encoder_identity,
                prompt_template_version=prompt_version,
            ),
            "text": text,
            "profile": str(args.profile),
            "source": str(args.source),
        }
        for text in texts
    ]

    global_shard_id = _next_shard_id(global_dir)
    global_shard = f"shard_{global_shard_id:05d}"
    write_shard(
        global_dir,
        global_shard_id,
        cache_rows,
        rows.vtxt,
        rows.ctxt,
        rows.lengths,
        str(global_manifest["storage_dtype"]),
    )
    global_index = dict(global_cache.index)
    profile_rows_path = global_dir / "profile_rows.jsonl"
    profile_rows = [
        json.loads(line)
        for line in profile_rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row_index, row in enumerate(cache_rows):
        entry = {
            "shard": global_shard,
            "row": row_index,
            "text": row["text"],
            "profile": row["profile"],
        }
        global_index[row["key"]] = entry
        profile_rows.append({"key": row["key"], **entry, "source": row["source"]})
    (global_dir / "index.json").write_text(
        json.dumps(global_index, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    with profile_rows_path.open("w", encoding="utf-8") as handle:
        for row in sorted(profile_rows, key=lambda value: value["key"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    global_manifest["num_texts"] = len(global_index)
    global_manifest["required_pair_count"] = len(global_index)
    global_manifest["index_sha256"] = _sha256_file(global_dir / "index.json")
    global_manifest["profile_rows_sha256"] = _sha256_file(profile_rows_path)
    global_manifest["supplemental_eval_rows"] = int(
        global_manifest.get("supplemental_eval_rows", 0)
    ) + len(cache_rows)
    global_manifest["supplemental_eval_source"] = str(args.source)
    (global_dir / "manifest.json").write_text(
        json.dumps(global_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    coverage_path = global_dir / "coverage_report.json"
    if coverage_path.is_file():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        coverage["cache_rows"] = len(global_index)
        coverage["cache_index_sha256"] = global_manifest["index_sha256"]
        coverage["supplemental_eval_rows"] = int(
            coverage.get("supplemental_eval_rows", 0)
        ) + len(cache_rows)
        coverage_path.write_text(
            json.dumps(coverage, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    context_manifest = dict(context_cache.manifest)
    context_shard_id = _next_shard_id(context_dir)
    context_shard = f"shard_{context_shard_id:05d}"
    context_shard_dir = context_dir / "shards" / context_shard
    context_shard_dir.mkdir(parents=True, exist_ok=False)
    lengths = rows.contextual_lengths.numpy().astype(np.int32, copy=False)
    packed = torch.cat(
        [rows.contextual[i, : int(length)] for i, length in enumerate(lengths)],
        dim=0,
    ).numpy().astype(np.float16, copy=False)
    np.save(context_shard_dir / "tokens.npy", packed, allow_pickle=False)
    np.save(context_shard_dir / "lengths.npy", lengths, allow_pickle=False)
    context_index = dict(context_cache.index)
    offset = 0
    for row, length in zip(cache_rows, lengths.tolist()):
        context_index[row["key"]] = {
            "shard": context_shard,
            "offset": offset,
            "length": int(length),
            "text": row["text"],
            "profile": row["profile"],
        }
        offset += int(length)
    (context_dir / "index.json").write_text(
        json.dumps(context_index, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    old_count = int(context_manifest["num_texts"])
    old_tokens = int(context_manifest["total_tokens"])
    context_manifest["num_texts"] = len(context_index)
    context_manifest["total_tokens"] = old_tokens + int(lengths.sum())
    context_manifest["mean_tokens"] = context_manifest["total_tokens"] / max(
        1, len(context_index)
    )
    context_manifest["max_tokens"] = max(
        int(context_manifest["max_tokens"]), int(lengths.max())
    )
    context_manifest["supplemental_eval_rows"] = int(
        context_manifest.get("supplemental_eval_rows", 0)
    ) + (len(context_index) - old_count)
    context_manifest["supplemental_eval_source"] = str(args.source)
    (context_dir / "manifest.json").write_text(
        json.dumps(context_manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "appended": len(cache_rows),
                "global_shard": global_shard,
                "context_shard": context_shard,
                "context_tokens": int(lengths.sum()),
                "global_rows": len(global_index),
                "context_rows": len(context_index),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
