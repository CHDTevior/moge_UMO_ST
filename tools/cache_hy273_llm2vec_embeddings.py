#!/usr/bin/env python
"""Build the profile-aware Kimodo LLM2Vec sentence cache for HY273."""

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

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
KIMODO_ROOT = Path(
    os.environ.get("KIMODO_ROOT", REPO_ROOT / "external_repos" / "kimodo")
).expanduser().resolve()
for path in (REPO_ROOT, KIMODO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kimodo.model.llm2vec.llm2vec import LLM2Vec
from models.raw_motion.hytext_cache import (
    LLM2VEC_CACHE_FORMAT,
    hytext_profile_key,
)
from tools.build_hy273_hytext_profile_cache import (
    ABSOLUTE_PROFILE,
    RELATIVE_PROFILE,
    collect_required,
    sha256_file,
)
from tools.cache_hy273_hytext_embeddings import write_shard


PROMPT_TEMPLATE_VERSION = "kimodo_llm2vec_raw_text_v1"
PROFILE_DESCRIPTIONS = {
    ABSOLUTE_PROFILE: "raw absolute motion caption",
    RELATIVE_PROFILE: "raw relative motion-edit instruction",
}
DEFAULT_BASE_MODEL = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
DEFAULT_SUPERVISED_MODEL = (
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
)


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_local_model_path(value: str) -> str:
    """Prefer an already downloaded Hub snapshot without contacting the network."""

    path = Path(value).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if "/" not in value:
        return value
    cache_root = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_CACHE_DIR")
    )
    if not cache_root:
        return value
    organization, model = value.split("/", 1)
    repository = (
        Path(cache_root).expanduser()
        / f"models--{organization}--{model}"
    )
    main_ref = repository / "refs" / "main"
    if main_ref.is_file():
        snapshot = repository / "snapshots" / main_ref.read_text().strip()
        if snapshot.is_dir():
            return str(snapshot.resolve())
    snapshots = sorted((repository / "snapshots").glob("*"))
    if len(snapshots) == 1 and snapshots[0].is_dir():
        return str(snapshots[0].resolve())
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_dir",
        default=(
            "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/"
            "manifests/hy273_multitask_v1"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/"
            "llm2vec_llama3_8b_profile_v1"
        ),
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--supervised_model", default=DEFAULT_SUPERVISED_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model_dtype",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=(
            "LLM2Vec forward batch. Kimodo fixes this to 1 because changing the "
            "batch materially changes sentence embeddings."
        ),
    )
    parser.add_argument(
        "--devices",
        default="",
        help=(
            "Optional comma-separated devices for shard-parallel encoding, for "
            "example cuda:0,cuda:1. Each device still encodes with batch_size=1."
        ),
    )
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument(
        "--storage_dtype",
        choices=("fp16", "fp32"),
        default="fp16",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser


def load_encoder(
    *,
    base_model: str,
    supervised_model: str,
    model_dtype: str,
    device: str,
) -> LLM2Vec:
    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    cache_dir = (
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_CACHE_DIR")
    )
    encoder = LLM2Vec.from_pretrained(
        base_model_name_or_path=base_model,
        peft_model_name_or_path=supervised_model,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    encoder.eval()
    encoder.requires_grad_(False)
    encoder.to(device)
    return encoder


def encode_sentences_single_device(
    encoder: LLM2Vec,
    texts: list[str],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Run LLM2Vec without its implicit all-visible-GPU multiprocessing path."""

    if not texts:
        return torch.empty(0, 4096, dtype=torch.float32)
    converted = [encoder._convert_to_str("", text) for text in texts]
    length_order = np.argsort(
        [-encoder._text_length(text) for text in converted]
    )
    sorted_texts = [converted[index] for index in length_order]
    batches = []
    for start in range(0, len(sorted_texts), int(batch_size)):
        batches.append(
            encoder._encode(
                sorted_texts[start : start + int(batch_size)],
                device=device,
                convert_to_numpy=False,
            )
        )
    encoded = torch.cat(batches, dim=0)
    encoded = encoded[np.argsort(length_order)]
    return encoded.float().cpu()


def encode_rows(
    encoder: LLM2Vec,
    rows: list[dict[str, str]],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    output = torch.zeros(len(rows), 4096, dtype=torch.float32)
    nonempty = [
        index for index, row in enumerate(rows) if str(row["text"]).strip()
    ]
    if not nonempty:
        return output
    text = [rows[index]["text"] for index in nonempty]
    encoded = encode_sentences_single_device(
        encoder,
        text,
        batch_size=int(batch_size),
        device=device,
    )
    if encoded.shape != (len(nonempty), 4096):
        raise ValueError(
            f"Unexpected LLM2Vec output shape {tuple(encoded.shape)}"
        )
    output[torch.tensor(nonempty, dtype=torch.long)] = encoded.float().cpu()
    return output


def _encode_worker(
    *,
    worker_id: int,
    device: str,
    shard_specs: list[tuple[int, int, list[dict[str, str]]]],
    output_dir: str,
    base_model: str,
    supervised_model: str,
    model_dtype: str,
    batch_size: int,
    storage_dtype: str,
    total_rows: int,
) -> list[int]:
    encoder = load_encoder(
        base_model=base_model,
        supervised_model=supervised_model,
        model_dtype=model_dtype,
        device=device,
    )
    completed = []
    for shard_id, start, shard_rows in shard_specs:
        embedding = encode_rows(
            encoder,
            shard_rows,
            batch_size=batch_size,
            device=device,
        )
        ctxt = embedding[:, None, :]
        vtxt = torch.zeros(len(shard_rows), 1, 1, dtype=torch.float32)
        lengths = torch.ones(len(shard_rows), dtype=torch.long)
        write_shard(
            Path(output_dir),
            shard_id,
            shard_rows,
            vtxt,
            ctxt,
            lengths,
            storage_dtype,
        )
        completed.append(shard_id)
        print(
            f"[llm2vec-cache worker={worker_id} device={device}] "
            f"shard={shard_id:05d} rows={start + len(shard_rows)}/{total_rows}",
            flush=True,
        )
    return completed


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("batch_size and shard_size must be positive")
    if args.batch_size != 1:
        raise ValueError(
            "Kimodo-compatible LLM2Vec caches require --batch_size 1; "
            "larger batches materially change the sentence embeddings"
        )
    devices = [
        value.strip()
        for value in (args.devices.split(",") if args.devices else [args.device])
        if value.strip()
    ]
    if not devices:
        raise ValueError("At least one encoding device is required")
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output cache exists; pass --overwrite: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    required, manifest_hashes = collect_required(manifest_dir)
    pairs = sorted(required)
    if args.limit > 0:
        pairs = pairs[: int(args.limit)]
    base_model = resolve_local_model_path(str(args.base_model))
    supervised_model = resolve_local_model_path(str(args.supervised_model))
    encoder_identity = (
        "kimodo_llm2vec_meta-llama3-8b-mntp-supervised:"
        f"dtype={args.model_dtype}:batch_size=1"
    )
    rows = [
        {
            "key": hytext_profile_key(
                text,
                profile,
                encoder_identity=encoder_identity,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            ),
            "profile": profile,
            "text": text,
            "source": sorted(required[(profile, text)])[0],
        }
        for profile, text in pairs
    ]
    shard_specs = [
        (shard_id, start, rows[start : start + int(args.shard_size)])
        for shard_id, start in enumerate(
            range(0, len(rows), int(args.shard_size))
        )
    ]
    assignments = [[] for _ in devices]
    for shard_index, shard_spec in enumerate(shard_specs):
        assignments[shard_index % len(devices)].append(shard_spec)

    worker_kwargs = [
        {
            "worker_id": worker_id,
            "device": device,
            "shard_specs": assignments[worker_id],
            "output_dir": str(output_dir),
            "base_model": base_model,
            "supervised_model": supervised_model,
            "model_dtype": str(args.model_dtype),
            "batch_size": int(args.batch_size),
            "storage_dtype": str(args.storage_dtype),
            "total_rows": len(rows),
        }
        for worker_id, device in enumerate(devices)
        if assignments[worker_id]
    ]
    if len(worker_kwargs) == 1:
        completed_shards = set(_encode_worker(**worker_kwargs[0]))
    else:
        completed_shards: set[int] = set()
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
                completed_shards.update(future.result())
    expected_shards = {shard_id for shard_id, _, _ in shard_specs}
    if completed_shards != expected_shards:
        raise RuntimeError(
            "LLM2Vec shard workers did not complete the expected shard set: "
            f"completed={sorted(completed_shards)}, "
            f"expected={sorted(expected_shards)}"
        )

    index: dict[str, dict[str, Any]] = {}
    profile_rows: list[dict[str, Any]] = []
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

    index_text = json.dumps(
        index,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    (output_dir / "index.json").write_text(index_text, encoding="utf-8")
    profile_rows_path = output_dir / "profile_rows.jsonl"
    with profile_rows_path.open("w", encoding="utf-8") as handle:
        for row in sorted(profile_rows, key=lambda value: value["key"]):
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )

    index_sha = sha256_file(output_dir / "index.json")
    profile_rows_sha = sha256_file(profile_rows_path)
    profile_hashes = {
        profile: hashlib.sha256(description.encode("utf-8")).hexdigest()
        for profile, description in PROFILE_DESCRIPTIONS.items()
    }
    manifest = {
        "format": LLM2VEC_CACHE_FORMAT,
        "encoder_identity": encoder_identity,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "profiles": PROFILE_DESCRIPTIONS,
        "profile_prompt_sha256": profile_hashes,
        "profile_empty_keys": {
            profile: hytext_profile_key(
                "",
                profile,
                encoder_identity=encoder_identity,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            )
            for profile in PROFILE_DESCRIPTIONS
        },
        "base_model": base_model,
        "supervised_model": supervised_model,
        "model_dtype": str(args.model_dtype),
        "storage_dtype": str(args.storage_dtype),
        "encoding_batch_size": int(args.batch_size),
        "encoding_devices": list(devices),
        "max_length_llm": 1,
        "ctxt_dim": 4096,
        "vtxt_dim": 1,
        "required_manifest_sha256": manifest_hashes,
        "required_pair_count": len(required),
        "num_texts": len(index),
        "index_sha256": index_sha,
        "profile_rows_sha256": profile_rows_sha,
    }
    manifest["profile_contract_sha256"] = canonical_sha(
        {
            "encoder_identity": encoder_identity,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "profile_prompt_sha256": profile_hashes,
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    coverage = {
        "format": "hy273_multitask_text_cache_coverage_v1",
        "passed": len(index) == len(required) and not args.limit,
        "required_pairs": len(required),
        "cache_rows": len(index),
        "missing": max(0, len(required) - len(index)),
        "required_manifest_sha256": manifest_hashes,
        "cache_index_sha256": index_sha,
    }
    (output_dir / "coverage_report.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rows": len(index),
                "coverage_passed": coverage["passed"],
                "manifest_sha256": sha256_file(
                    output_dir / "manifest.json"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
