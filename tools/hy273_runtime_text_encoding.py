#!/usr/bin/env python
"""Encode cache-miss HY273 prompts without mutating the training cache."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Iterable

import torch

from models.raw_motion.hytext_cache import (
    LLM2VEC_CACHE_FORMAT,
    PROFILE_CACHE_FORMAT,
    normalize_text_key,
)


@dataclass(frozen=True)
class RuntimeTextRows:
    texts: tuple[str, ...]
    profiles: tuple[str, ...]
    vtxt: torch.Tensor | None
    ctxt: torch.Tensor | None
    lengths: torch.Tensor | None
    contextual: torch.Tensor | None = None
    contextual_lengths: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return len(self.texts)


def _missing_pairs(
    cache,
    texts: Iterable[str],
    profiles: Iterable[str],
    context_cache=None,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for text, profile in zip(texts, profiles):
        pair = (normalize_text_key(str(text)), str(profile))
        global_present = cache.text_source(pair[0], pair[1]) is not None
        context_present = (
            context_cache is None
            or context_cache.has_text(pair[0], pair[1])
        )
        if not pair[0] or pair in seen or (global_present and context_present):
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _encode_hytext(
    cache,
    pairs: list[tuple[str, str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        CLIPTextModel,
        CLIPTokenizer,
    )
    from tools.cache_hy273_hytext_embeddings import compute_crop_start, encode_batch

    manifest = cache.manifest
    qwen_path = Path(str(manifest["qwen_path"])).expanduser().resolve()
    clip_path = Path(str(manifest["clip_path"])).expanduser().resolve()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        qwen_path,
        padding_side="right",
        trust_remote_code=True,
    )
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).eval()
    qwen_model.requires_grad_(False).to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
    clip_model = CLIPTextModel.from_pretrained(
        clip_path,
        torch_dtype=dtype,
    ).eval()
    clip_model.requires_grad_(False).to(device)

    by_profile: dict[str, list[tuple[int, str]]] = {}
    for index, (text, profile) in enumerate(pairs):
        by_profile.setdefault(profile, []).append((index, text))
    rows: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
    ] = [None] * len(pairs)
    for profile, indexed_texts in by_profile.items():
        prompt_template = str(manifest["profiles"][profile])
        crop_start = compute_crop_start(qwen_tokenizer, prompt_template)
        expected_crop = manifest.get("profile_crop_starts", {}).get(profile)
        if expected_crop is not None and int(expected_crop) != int(crop_start):
            raise RuntimeError(
                f"HYText prompt crop changed for {profile}: "
                f"expected {expected_crop}, got {crop_start}"
            )
        encoded = encode_batch(
            [text for _, text in indexed_texts],
            qwen_model,
            qwen_tokenizer,
            clip_model,
            clip_tokenizer,
            device,
            crop_start,
            int(manifest["max_length_llm"]),
            prompt_template,
        )
        for local_index, (output_index, _) in enumerate(indexed_texts):
            rows[output_index] = tuple(
                tensor[local_index] for tensor in encoded
            )
    del qwen_model, qwen_tokenizer, clip_model, clip_tokenizer
    if any(row is None for row in rows):
        raise RuntimeError("Internal HYText runtime encoding order mismatch")
    complete_rows = [row for row in rows if row is not None]
    return tuple(
        torch.stack([row[field] for row in complete_rows], dim=0)
        for field in range(3)
    )


def _encode_llm2vec(
    cache,
    pairs: list[tuple[str, str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from tools.cache_hy273_llm2vec_embeddings import (
        encode_sentences_single_device,
        load_encoder,
    )

    manifest = cache.manifest
    if int(manifest.get("encoding_batch_size", -1)) != 1:
        raise RuntimeError(
            "Runtime LLM2Vec encoding requires a batch_size=1 cache contract"
        )
    encoder = load_encoder(
        base_model=str(manifest["base_model"]),
        supervised_model=str(manifest["supervised_model"]),
        model_dtype=str(manifest["model_dtype"]),
        device=str(device),
    )
    sentence = encode_sentences_single_device(
        encoder,
        [text for text, _ in pairs],
        batch_size=1,
        device=str(device),
    )
    del encoder
    return (
        torch.zeros(len(pairs), 1, 1, dtype=torch.float32),
        sentence[:, None, :],
        torch.ones(len(pairs), dtype=torch.long),
    )


def _encode_llm2vec_with_context(
    cache,
    context_cache,
    pairs: list[tuple[str, str]],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    from tools.cache_hy273_llm2vec_context_tokens import _encode_context
    from tools.cache_hy273_llm2vec_embeddings import load_encoder

    manifest = cache.manifest
    context_manifest = context_cache.manifest
    for field in (
        "encoder_identity",
        "prompt_template_version",
        "model_dtype",
        "storage_dtype",
    ):
        if str(manifest.get(field, "")) != str(context_manifest.get(field, "")):
            raise RuntimeError(
                f"Global/context runtime encoder contract differs at {field}"
            )
    if int(manifest.get("encoding_batch_size", -1)) != 1:
        raise RuntimeError(
            "Runtime LLM2Vec encoding requires a batch_size=1 cache contract"
        )
    encoder = load_encoder(
        base_model=str(manifest["base_model"]),
        supervised_model=str(manifest["supervised_model"]),
        model_dtype=str(manifest["model_dtype"]),
        device=str(device),
    )
    if str(getattr(encoder, "pooling_mode", "")) != "mean":
        raise RuntimeError("Runtime full-text encoding requires mean pooling")
    if not bool(getattr(encoder, "skip_instruction", False)):
        raise RuntimeError(
            "Runtime full-text encoding requires skip_instruction=True"
        )
    contextual_rows = [
        _encode_context(encoder, text, device=str(device))
        for text, _ in pairs
    ]
    del encoder
    contextual_lengths = torch.tensor(
        [int(tokens.shape[0]) for tokens in contextual_rows],
        dtype=torch.long,
    )
    max_length = max(1, int(contextual_lengths.max().item()))
    embedding_dim = int(manifest["ctxt_dim"])
    contextual = torch.zeros(
        len(pairs),
        max_length,
        embedding_dim,
        dtype=torch.float32,
    )
    sentence_rows = []
    for row, tokens in enumerate(contextual_rows):
        length = int(tokens.shape[0])
        if tuple(tokens.shape[1:]) != (embedding_dim,):
            raise ValueError(
                f"Runtime contextual row has shape {tuple(tokens.shape)}"
            )
        contextual[row, :length] = tokens
        sentence_rows.append(tokens.mean(dim=0))
    sentence = torch.stack(sentence_rows, dim=0)
    return (
        torch.zeros(len(pairs), 1, 1, dtype=torch.float32),
        sentence[:, None, :],
        torch.ones(len(pairs), dtype=torch.long),
        contextual,
        contextual_lengths,
    )


def _validate_and_quantize_rows(
    cache,
    pairs: list[tuple[str, str]],
    vtxt: torch.Tensor,
    ctxt: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    manifest = cache.manifest
    count = len(pairs)
    expected_vtxt = (count, 1, int(manifest["vtxt_dim"]))
    expected_ctxt = (
        count,
        int(manifest["max_length_llm"]),
        int(manifest["ctxt_dim"]),
    )
    if tuple(vtxt.shape) != expected_vtxt:
        raise ValueError(
            f"Runtime vtxt shape {tuple(vtxt.shape)} != {expected_vtxt}"
        )
    if tuple(ctxt.shape) != expected_ctxt:
        raise ValueError(
            f"Runtime ctxt shape {tuple(ctxt.shape)} != {expected_ctxt}"
        )
    if tuple(lengths.shape) != (count,):
        raise ValueError(
            f"Runtime lengths shape {tuple(lengths.shape)} != {(count,)}"
        )
    length_rows = lengths.detach().long().cpu()
    if bool((length_rows < 0).any()) or bool(
        (length_rows > expected_ctxt[1]).any()
    ):
        raise ValueError("Runtime text lengths are outside the cache contract")
    if str(manifest.get("format", "")) == LLM2VEC_CACHE_FORMAT and not bool(
        torch.all(length_rows == 1)
    ):
        raise ValueError("Runtime LLM2Vec rows must all have length 1")
    return (
        cache.storage_round_trip(vtxt),
        cache.storage_round_trip(ctxt),
        length_rows,
    )


def encode_missing_text_rows(
    cache,
    texts: Iterable[str],
    profiles: Iterable[str],
    device: torch.device,
    *,
    context_cache=None,
) -> RuntimeTextRows:
    text_list = [str(text) for text in texts]
    profile_list = [str(profile) for profile in profiles]
    if len(text_list) != len(profile_list):
        raise ValueError("Runtime text/profile counts differ")
    pairs = _missing_pairs(
        cache,
        text_list,
        profile_list,
        context_cache=context_cache,
    )
    if not pairs:
        return RuntimeTextRows((), (), None, None, None)

    cache_format = str(cache.manifest.get("format", ""))
    print(
        f"[text] encoding {len(pairs)} cache-miss prompts "
        f"with {cache_format}",
        flush=True,
    )
    if cache_format == PROFILE_CACHE_FORMAT:
        vtxt, ctxt, lengths = _encode_hytext(cache, pairs, device)
        contextual = None
        contextual_lengths = None
    elif cache_format == LLM2VEC_CACHE_FORMAT:
        if context_cache is None:
            vtxt, ctxt, lengths = _encode_llm2vec(cache, pairs, device)
            contextual = None
            contextual_lengths = None
        else:
            (
                vtxt,
                ctxt,
                lengths,
                contextual,
                contextual_lengths,
            ) = _encode_llm2vec_with_context(
                cache,
                context_cache,
                pairs,
                device,
            )
    else:
        raise ValueError(
            f"Unsupported runtime text cache format {cache_format!r}"
        )
    vtxt, ctxt, lengths = _validate_and_quantize_rows(
        cache,
        pairs,
        vtxt,
        ctxt,
        lengths,
    )
    if contextual is not None:
        contextual = context_cache.storage_round_trip(contextual)
        contextual_lengths = contextual_lengths.detach().long().cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return RuntimeTextRows(
        texts=tuple(text for text, _ in pairs),
        profiles=tuple(profile for _, profile in pairs),
        vtxt=vtxt,
        ctxt=ctxt,
        lengths=lengths,
        contextual=contextual,
        contextual_lengths=contextual_lengths,
    )


def register_runtime_text_rows(
    cache,
    rows: RuntimeTextRows,
    *,
    context_cache=None,
) -> int:
    if not rows.count:
        return 0
    if rows.vtxt is None or rows.ctxt is None or rows.lengths is None:
        raise ValueError("Runtime text rows have incomplete tensors")
    cache.add_runtime_rows(
        rows.texts,
        rows.vtxt,
        rows.ctxt,
        rows.lengths,
        profiles=rows.profiles,
    )
    if rows.contextual is not None:
        if context_cache is None or rows.contextual_lengths is None:
            raise ValueError(
                "Runtime contextual rows require their contextual cache"
            )
        context_cache.add_runtime_rows(
            rows.texts,
            rows.contextual,
            rows.contextual_lengths,
            rows.profiles,
        )
    elif context_cache is not None:
        raise ValueError(
            "Full-text runtime registration is missing contextual rows"
        )
    return rows.count
