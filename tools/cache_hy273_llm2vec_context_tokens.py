#!/usr/bin/env python3
"""Cache variable-length LLM2Vec contextual states in packed mmap shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
KIMODO_ROOT = REPO_ROOT / "external_repos" / "kimodo"
for path in (REPO_ROOT, KIMODO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.raw_motion.hytext_cache import LLM2VEC_CACHE_FORMAT
from models.raw_motion.llm2vec_context_cache import (
    LLM2VEC_CONTEXT_CACHE_FORMAT,
)
from tools.cache_hy273_llm2vec_embeddings import load_encoder


DEFAULT_GLOBAL_CACHE = (
    "/mnt/afs/mogo_base/datasets/HY273_unified_actor_v1/"
    "llm2vec_llama3_8b_profile_interaction_v1"
)
DEFAULT_OUTPUT_CACHE = (
    "/mnt/afs/mogo_base/datasets/HY273_unified_actor_v1/"
    "llm2vec_llama3_8b_context_profile_interaction_v1"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_cache_dir", default=DEFAULT_GLOBAL_CACHE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7",
    )
    parser.add_argument("--model_dtype", choices=("bf16", "fp32"), default="")
    parser.add_argument("--storage_dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--rows_per_shard", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _encode_context(
    encoder: Any,
    text: str,
    *,
    device: str,
) -> torch.Tensor:
    """Return exactly the token states used by LLM2Vec mean pooling."""

    if not str(text).strip():
        return torch.empty(0, 4096, dtype=torch.float32)
    converted = encoder._convert_to_str("", str(text))
    prepared = encoder.prepare_for_tokenization(converted)
    features = encoder.tokenize([prepared])
    embed_mask = features["embed_mask"].to(device=device, dtype=torch.bool)
    model_inputs = {
        key: value.to(device)
        for key, value in features.items()
        if key != "embed_mask"
    }
    with torch.inference_mode():
        output = encoder.model(
            **model_inputs,
            use_cache=False,
            return_dict=True,
        )
    states = output.last_hidden_state[0, embed_mask[0]]
    if states.ndim != 2 or states.shape[1] != 4096 or states.shape[0] <= 0:
        raise ValueError(
            f"Unexpected LLM2Vec contextual shape {tuple(states.shape)} "
            f"for text={text!r}"
        )
    states = states.detach().float().cpu()
    if not bool(torch.isfinite(states).all()):
        raise ValueError(f"Non-finite LLM2Vec contextual states for {text!r}")
    return states


def _encode_worker(
    *,
    worker_id: int,
    device: str,
    shard_specs: list[tuple[int, list[dict[str, Any]]]],
    output_dir: str,
    base_model: str,
    supervised_model: str,
    model_dtype: str,
    storage_dtype: str,
    total_rows: int,
) -> list[int]:
    encoder = load_encoder(
        base_model=base_model,
        supervised_model=supervised_model,
        model_dtype=model_dtype,
        device=device,
    )
    if str(getattr(encoder, "pooling_mode", "")) != "mean":
        raise ValueError("Context cache requires LLM2Vec pooling_mode='mean'")
    if not bool(getattr(encoder, "skip_instruction", False)):
        raise ValueError("Context cache requires LLM2Vec skip_instruction=True")
    numpy_dtype = np.float16 if storage_dtype == "fp16" else np.float32
    completed: list[int] = []
    processed_rows = 0
    for shard_id, rows in shard_specs:
        encoded = [
            _encode_context(encoder, str(row["text"]), device=device)
            for row in rows
        ]
        lengths = np.asarray(
            [int(tokens.shape[0]) for tokens in encoded],
            dtype=np.int32,
        )
        if encoded and int(lengths.sum()) > 0:
            packed = torch.cat(
                [tokens for tokens in encoded if tokens.shape[0] > 0],
                dim=0,
            ).numpy().astype(numpy_dtype, copy=False)
        else:
            packed = np.empty((0, 4096), dtype=numpy_dtype)
        if not np.isfinite(packed).all():
            raise ValueError(
                f"Contextual states overflowed {storage_dtype} in shard {shard_id}"
            )
        shard_dir = (
            Path(output_dir) / "shards" / f"shard_{shard_id:05d}"
        )
        shard_dir.mkdir(parents=True, exist_ok=False)
        np.save(shard_dir / "tokens.npy", packed, allow_pickle=False)
        np.save(shard_dir / "lengths.npy", lengths, allow_pickle=False)
        processed_rows += len(rows)
        completed.append(shard_id)
        print(
            f"[context-cache worker={worker_id} device={device}] "
            f"shard={shard_id:05d} local_rows={processed_rows} "
            f"tokens={int(lengths.sum())} total_rows={total_rows}",
            flush=True,
        )
    return completed


def _load_global_rows(
    global_cache: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(
        (global_cache / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("format") != LLM2VEC_CACHE_FORMAT:
        raise ValueError("The source cache is not the profile-aware LLM2Vec cache")
    if int(manifest.get("encoding_batch_size", -1)) != 1:
        raise ValueError("The source LLM2Vec cache was not encoded with batch_size=1")
    if str(manifest.get("model_dtype", "")).lower() != "bf16":
        raise ValueError("The source LLM2Vec cache must use bf16 model inference")
    if str(manifest.get("storage_dtype", "")).lower() != "fp16":
        raise ValueError("The source LLM2Vec cache must use fp16 storage")
    rows = [
        json.loads(line)
        for line in (global_cache / "profile_rows.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("num_texts", -1)):
        raise ValueError("Global cache profile_rows count differs from its manifest")
    if len({str(row["key"]) for row in rows}) != len(rows):
        raise ValueError("Global cache profile_rows contains duplicate keys")
    return manifest, rows


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.rows_per_shard <= 0 or args.limit < 0:
        raise ValueError("rows_per_shard must be positive and limit non-negative")
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("At least one CUDA device is required")
    global_cache = Path(args.global_cache_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest, rows = _load_global_rows(global_cache)
    if args.limit:
        rows = rows[: int(args.limit)]
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output cache exists; pass --overwrite: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_model_dtype = str(manifest["model_dtype"]).lower()
    model_dtype = str(args.model_dtype or source_model_dtype).lower()
    if model_dtype != source_model_dtype:
        raise ValueError(
            "Context model dtype must match the global cache exactly: "
            f"context={model_dtype}, global={source_model_dtype}"
        )
    if model_dtype != "bf16":
        raise ValueError("The full-text experiment requires bf16 LLM2Vec inference")
    if str(args.storage_dtype).lower() != "fp16":
        raise ValueError("The full-text experiment requires fp16 contextual storage")
    base_model = str(manifest["base_model"])
    supervised_model = str(manifest["supervised_model"])
    shard_specs = [
        (shard_id, rows[start : start + int(args.rows_per_shard)])
        for shard_id, start in enumerate(
            range(0, len(rows), int(args.rows_per_shard))
        )
    ]
    assignments: list[list[tuple[int, list[dict[str, Any]]]]] = [
        [] for _ in devices
    ]
    for index, shard in enumerate(shard_specs):
        assignments[index % len(devices)].append(shard)
    worker_kwargs = [
        {
            "worker_id": worker_id,
            "device": device,
            "shard_specs": assignments[worker_id],
            "output_dir": str(output_dir),
            "base_model": base_model,
            "supervised_model": supervised_model,
            "model_dtype": model_dtype,
            "storage_dtype": str(args.storage_dtype),
            "total_rows": len(rows),
        }
        for worker_id, device in enumerate(devices)
        if assignments[worker_id]
    ]
    if len(worker_kwargs) == 1:
        completed = set(_encode_worker(**worker_kwargs[0]))
    else:
        completed: set[int] = set()
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
    expected = {shard_id for shard_id, _ in shard_specs}
    if completed != expected:
        raise RuntimeError(
            f"Context cache workers completed {sorted(completed)}, "
            f"expected {sorted(expected)}"
        )

    index: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    max_tokens = 0
    for shard_id, shard_rows in shard_specs:
        shard_name = f"shard_{shard_id:05d}"
        lengths = np.load(
            output_dir / "shards" / shard_name / "lengths.npy",
            allow_pickle=False,
        )
        if lengths.shape != (len(shard_rows),):
            raise ValueError(f"Bad lengths shape in {shard_name}")
        offset = 0
        for row, length_value in zip(shard_rows, lengths.tolist()):
            length = int(length_value)
            index[str(row["key"])] = {
                "shard": shard_name,
                "offset": offset,
                "length": length,
                "text": str(row["text"]),
                "profile": str(row["profile"]),
            }
            offset += length
            total_tokens += length
            max_tokens = max(max_tokens, length)
        packed = np.load(
            output_dir / "shards" / shard_name / "tokens.npy",
            mmap_mode="r",
        )
        if packed.shape != (offset, 4096):
            raise ValueError(
                f"Packed shape {packed.shape} differs from index in {shard_name}"
            )

    (output_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    context_manifest = {
        "format": LLM2VEC_CONTEXT_CACHE_FORMAT,
        "global_cache_dir": str(global_cache),
        "encoder_identity": str(manifest["encoder_identity"]),
        "prompt_template_version": str(manifest["prompt_template_version"]),
        "base_model": base_model,
        "supervised_model": supervised_model,
        "model_dtype": model_dtype,
        "storage_dtype": str(args.storage_dtype),
        "pooling_mode": "mean",
        "skip_instruction": True,
        "encoding_batch_size": 1,
        "encoding_devices": devices,
        "embedding_dim": 4096,
        "num_texts": len(index),
        "total_tokens": total_tokens,
        "mean_tokens": total_tokens / max(1, len(index)),
        "max_tokens": max_tokens,
        "rows_per_shard": int(args.rows_per_shard),
        "complete": not bool(args.limit),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(
            context_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(context_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
