#!/usr/bin/env python
"""Migrate HYText shards to profile-aware keys and encode only missing rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.raw_motion.hytext_cache import (
    PROFILE_CACHE_FORMAT,
    hytext_key,
    hytext_profile_key,
    normalize_text_key,
)
from tools.cache_hy273_hytext_embeddings import (
    PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION,
    PROMPT_TEMPLATE_ENCODE_RELATIVE_EDIT,
    compute_crop_start,
    dtype_from_name,
    encode_batch,
    write_shard,
)


ABSOLUTE_PROFILE = "hytext_absolute_motion_v1"
RELATIVE_PROFILE = "hytext_relative_edit_v1"
PROMPT_TEMPLATE_VERSION = "hy273_hytext_profiles_v1"
PROFILE_PROMPTS = {
    ABSOLUTE_PROFILE: PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION,
    RELATIVE_PROFILE: PROMPT_TEMPLATE_ENCODE_RELATIVE_EDIT,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_cache",
        default="/mnt/afs/mogo_base/datasets/HumanML3D/hytext_qwen3_clipL_mlen128_unified_edit_v1",
    )
    parser.add_argument(
        "--manifest_dir",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/hy273_multitask_v1",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/hytext_qwen3_clipL_profile_v2",
    )
    parser.add_argument("--qwen_path", default="/mnt/afs/HY-Motion-1.0/ckpts/Qwen3-8B")
    parser.add_argument(
        "--clip_path", default="/mnt/afs/HY-Motion-1.0/ckpts/clip-vit-large-patch14"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--storage_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def collect_required(manifest_dir: Path) -> tuple[dict[tuple[str, str], set[str]], dict[str, str]]:
    required: dict[tuple[str, str], set[str]] = {}
    manifest_hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Unified manifest split is missing: {path}")
        manifest_hashes[path.name] = sha256_file(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                dataset = str(row["dataset"])
                for text in row["texts"]:
                    profile = str(text["encoding_profile"])
                    if profile not in PROFILE_PROMPTS:
                        raise ValueError(
                            f"Unsupported profile {profile!r} at {path}:{line_number}"
                        )
                    normalized = normalize_text_key(str(text["value"]))
                    required.setdefault((profile, normalized), set()).add(
                        f"{split}:{dataset}:{row['uid']}"
                    )
    for profile in PROFILE_PROMPTS:
        required.setdefault((profile, ""), set()).add("__profile_empty__")
    return required, manifest_hashes


def source_rows(source_cache: Path) -> list[dict[str, Any]]:
    source_index = json.loads((source_cache / "index.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for text_path in sorted((source_cache / "shards").glob("*/texts.jsonl")):
        shard = text_path.parent.name
        with text_path.open(encoding="utf-8") as handle:
            for row_number, line in enumerate(handle):
                row = json.loads(line)
                old_key = str(row["key"])
                indexed = source_index.get(old_key)
                if indexed is None:
                    raise RuntimeError(f"Source texts row has no index entry: {text_path}:{row_number + 1}")
                if str(indexed["shard"]) != shard or int(indexed["row"]) != row_number:
                    raise RuntimeError(f"Source row/index mismatch for key={old_key}")
                profile = str(row.get("profile") or ABSOLUTE_PROFILE)
                if profile not in PROFILE_PROMPTS:
                    raise ValueError(f"Unknown source profile {profile!r}")
                rows.append(
                    {
                        "old_key": old_key,
                        "profile": profile,
                        "text": normalize_text_key(str(row.get("text", ""))),
                        "source": str(row.get("source", "")),
                        "shard": shard,
                        "row": row_number,
                    }
                )
    if len(rows) != len(source_index):
        raise RuntimeError(
            f"Source cache text/index cardinality mismatch: rows={len(rows)} index={len(source_index)}"
        )
    return rows


def link_source_shards(source_cache: Path, output_dir: Path) -> None:
    target_root = output_dir / "shards"
    target_root.mkdir(parents=True, exist_ok=True)
    for source in sorted((source_cache / "shards").iterdir()):
        target = target_root / source.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Target shard link already exists: {target}")
        target.symlink_to(source.resolve(), target_is_directory=True)


def load_text_models(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = dtype_from_name(args.model_dtype, device)
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_path, padding_side="right", trust_remote_code=True
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen_model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval()
    qwen_model.requires_grad_(False).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_path)
    clip_model = CLIPTextModel.from_pretrained(
        args.clip_path, torch_dtype=dtype
    ).eval()
    clip_model.requires_grad_(False).to(device)
    return device, qwen_model, qwen_tokenizer, clip_model, clip_tokenizer


def main() -> None:
    args = build_arg_parser().parse_args()
    source_cache = Path(args.source_cache).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_manifest_path = source_cache / "manifest.json"
    source_index_path = source_cache / "index.json"
    if not source_manifest_path.is_file() or not source_index_path.is_file():
        raise FileNotFoundError(f"Invalid source HYText cache: {source_cache}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("format") != "hytext_memmap_v1":
        raise ValueError("Migration source must use hytext_memmap_v1")
    if int(source_manifest.get("max_length_llm", -1)) != 128:
        raise ValueError("R11 requires max_length_llm=128")

    required, manifest_hashes = collect_required(manifest_dir)
    rows = source_rows(source_cache)
    source_manifest_sha = sha256_file(source_manifest_path)
    source_index_sha = sha256_file(source_index_path)
    encoder_identity = (
        "hytext_qwen3-8b_clip-vit-l14_fp16:"
        f"source_manifest={source_manifest_sha}:source_index={source_index_sha}"
    )

    migrated_index: dict[str, dict[str, Any]] = {}
    available: dict[tuple[str, str], dict[str, Any]] = {}
    profile_rows: list[dict[str, Any]] = []
    for row in rows:
        pair = (row["profile"], row["text"])
        if pair in available:
            raise RuntimeError(f"Duplicate profile/text row in source cache: {pair}")
        key = hytext_profile_key(
            row["text"],
            row["profile"],
            encoder_identity=encoder_identity,
            prompt_template_version=PROMPT_TEMPLATE_VERSION,
        )
        entry = {
            "shard": row["shard"],
            "row": row["row"],
            "text": row["text"],
            "profile": row["profile"],
        }
        migrated_index[key] = entry
        available[pair] = entry
        profile_rows.append({"key": key, **entry, "source": row["source"]})

    missing = sorted(set(required) - set(available))
    report = {
        "required_profile_text_pairs": len(required),
        "source_rows": len(rows),
        "reused_required_rows": len(required) - len(missing),
        "missing_to_encode": len(missing),
        "missing_by_profile": dict(Counter(profile for profile, _ in missing)),
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output cache exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    link_source_shards(source_cache, output_dir)

    profile_crop_starts = {
        str(key): int(value)
        for key, value in source_manifest.get("profile_crop_starts", {}).items()
    }
    profile_crop_starts.setdefault(
        ABSOLUTE_PROFILE, int(source_manifest["crop_start"])
    )
    if missing:
        device, qwen_model, qwen_tokenizer, clip_model, clip_tokenizer = load_text_models(args)
        for profile in sorted(PROFILE_PROMPTS):
            profile_missing = [text for row_profile, text in missing if row_profile == profile]
            if not profile_missing:
                continue
            prompt = PROFILE_PROMPTS[profile]
            crop_start = compute_crop_start(qwen_tokenizer, prompt)
            expected_crop = profile_crop_starts.get(profile)
            if expected_crop is not None and int(expected_crop) != int(crop_start):
                raise RuntimeError(
                    f"Profile crop_start changed for {profile}: source={expected_crop}, current={crop_start}"
                )
            profile_crop_starts[profile] = int(crop_start)
            for shard_offset, start in enumerate(
                range(0, len(profile_missing), int(args.shard_size))
            ):
                texts = profile_missing[start : start + int(args.shard_size)]
                all_vtxt, all_ctxt, all_lengths = [], [], []
                for batch_start in range(0, len(texts), int(args.batch_size)):
                    batch = texts[batch_start : batch_start + int(args.batch_size)]
                    vtxt, ctxt, lengths = encode_batch(
                        batch,
                        qwen_model,
                        qwen_tokenizer,
                        clip_model,
                        clip_tokenizer,
                        device,
                        crop_start,
                        128,
                        prompt,
                    )
                    all_vtxt.append(vtxt)
                    all_ctxt.append(ctxt)
                    all_lengths.append(lengths)
                    print(
                        f"[profile-cache] {profile} encoded "
                        f"{min(batch_start + len(batch), len(texts))}/{len(texts)}",
                        flush=True,
                    )
                encoded_rows = [
                    {
                        "key": hytext_profile_key(
                            text,
                            profile,
                            encoder_identity=encoder_identity,
                            prompt_template_version=PROMPT_TEMPLATE_VERSION,
                        ),
                        "text": text,
                        "source": "__profile_cache_missing__",
                        "profile": profile,
                    }
                    for text in texts
                ]
                shard_id = 90_000 + list(sorted(PROFILE_PROMPTS)).index(profile) * 1_000 + shard_offset
                written = write_shard(
                    output_dir,
                    shard_id,
                    encoded_rows,
                    torch.cat(all_vtxt, dim=0),
                    torch.cat(all_ctxt, dim=0),
                    torch.cat(all_lengths, dim=0),
                    args.storage_dtype,
                )
                shard_name = f"shard_{shard_id:05d}"
                for row_idx, row in enumerate(encoded_rows):
                    entry = {
                        "shard": shard_name,
                        "row": row_idx,
                        "text": row["text"],
                        "profile": profile,
                    }
                    migrated_index[row["key"]] = entry
                    available[(profile, row["text"])] = entry
                    profile_rows.append({"key": row["key"], **entry, "source": row["source"]})

    still_missing = sorted(set(required) - set(available))
    if still_missing:
        raise RuntimeError(f"Internal profile cache coverage failure: {still_missing[:10]}")
    index_text = json.dumps(migrated_index, indent=2, ensure_ascii=False, sort_keys=True)
    (output_dir / "index.json").write_text(index_text, encoding="utf-8")
    with (output_dir / "profile_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(profile_rows, key=lambda item: item["key"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    prompt_hashes = {
        profile: hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        for profile, prompt in PROFILE_PROMPTS.items()
    }
    manifest = {
        "format": PROFILE_CACHE_FORMAT,
        "encoder_identity": encoder_identity,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "profiles": PROFILE_PROMPTS,
        "profile_prompt_sha256": prompt_hashes,
        "profile_crop_starts": profile_crop_starts,
        "profile_empty_keys": {
            profile: hytext_profile_key(
                "",
                profile,
                encoder_identity=encoder_identity,
                prompt_template_version=PROMPT_TEMPLATE_VERSION,
            )
            for profile in PROFILE_PROMPTS
        },
        "max_length_llm": 128,
        "ctxt_dim": 4096,
        "vtxt_dim": 768,
        "storage_dtype": args.storage_dtype,
        "qwen_path": str(Path(args.qwen_path).expanduser().resolve()),
        "clip_path": str(Path(args.clip_path).expanduser().resolve()),
        "source_cache": str(source_cache),
        "source_manifest_sha256": source_manifest_sha,
        "source_index_sha256": source_index_sha,
        "required_manifest_sha256": manifest_hashes,
        "required_pair_count": len(required),
        "num_texts": len(migrated_index),
        "num_reused_rows": len(rows),
        "num_encoded_rows": len(missing),
        "index_sha256": hashlib.sha256(index_text.encode("utf-8")).hexdigest(),
        "profile_rows_sha256": sha256_file(output_dir / "profile_rows.jsonl"),
    }
    manifest["profile_contract_sha256"] = canonical_sha(
        {
            "encoder_identity": encoder_identity,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "profile_prompt_sha256": prompt_hashes,
            "profile_crop_starts": profile_crop_starts,
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_rows": len(migrated_index),
                "encoded_rows": len(missing),
                "manifest_sha256": sha256_file(output_dir / "manifest.json"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

