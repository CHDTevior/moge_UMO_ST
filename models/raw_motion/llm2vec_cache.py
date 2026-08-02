"""Cached LLM2Vec sentence embeddings for Kimodo-style text conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from .hytext_cache import HYTextMemmapCache, LLM2VEC_CACHE_FORMAT
from .llm2vec_context_cache import LLM2VecContextMemmapCache
from .text_condition import RawTextCondition


LLM2VEC_TOKEN_SEQUENCE_MODES = (
    "sentence",
    "sentence_plus_context",
    "context",
)


class CachedLLM2VecTextEncoder(nn.Module):
    """Read one 4096-D LLM2Vec sentence token per prompt.

    The cache keeps the existing profile-aware key contract so absolute motion
    captions and relative edit instructions remain distinct data records. Text
    dropout zeros the sentence feature, matching Kimodo's unconditional branch,
    instead of encoding an empty sentence.
    """

    supports_profiles = True

    def __init__(
        self,
        *,
        hidden_dim: int,
        cache_dir: str | Path,
        embedding_dim: int = 4096,
        max_open_shards: int = 64,
        strict_cache: bool = True,
        project_tokens: bool = True,
        context_cache_dir: str | Path = "",
        context_max_open_shards: int = 16,
        token_sequence_mode: str = "sentence",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.token_sequence_mode = str(token_sequence_mode).lower()
        if self.token_sequence_mode not in LLM2VEC_TOKEN_SEQUENCE_MODES:
            raise ValueError(
                "token_sequence_mode must be one of "
                f"{LLM2VEC_TOKEN_SEQUENCE_MODES}, got {token_sequence_mode!r}"
            )
        self.max_text_tokens = 1
        self.project_tokens = bool(project_tokens)
        self.cache = HYTextMemmapCache(
            cache_dir,
            max_open_shards=max_open_shards,
            strict=strict_cache,
        )
        self._validate_manifest()
        self.token_proj: nn.Module
        if self.project_tokens:
            self.token_proj = nn.Linear(self.embedding_dim, self.hidden_dim)
        else:
            self.token_proj = nn.Identity()
        self.context_cache: LLM2VecContextMemmapCache | None = None
        self.local_proj: nn.Module | None = None
        if str(context_cache_dir):
            if not self.project_tokens:
                raise ValueError(
                    "Contextual LLM2Vec tokens require the projected Flux path"
                )
            self.context_cache = LLM2VecContextMemmapCache(
                context_cache_dir,
                embedding_dim=self.embedding_dim,
                max_open_shards=context_max_open_shards,
                strict=strict_cache,
            )
            self._validate_context_manifest()
            # Do not perturb initialization of the existing global-token
            # backbone when the optional local-memory path is enabled.
            with torch.random.fork_rng(devices=[]):
                self.local_proj = nn.Sequential(
                    nn.LayerNorm(self.embedding_dim),
                    nn.Linear(self.embedding_dim, self.hidden_dim),
                )
            context_max = int(self.context_cache.manifest.get("max_tokens", 0))
            self.max_text_tokens = context_max + (
                1 if self.token_sequence_mode == "sentence_plus_context" else 0
            )
        elif self.token_sequence_mode != "sentence":
            raise ValueError(
                f"{self.token_sequence_mode} requires context_cache_dir"
            )

    def _validate_manifest(self) -> None:
        manifest = self.cache.manifest
        if manifest.get("format") != LLM2VEC_CACHE_FORMAT:
            raise ValueError(
                "CachedLLM2VecTextEncoder requires "
                f"{LLM2VEC_CACHE_FORMAT}, got {manifest.get('format')!r}"
            )
        if int(manifest.get("ctxt_dim", -1)) != self.embedding_dim:
            raise ValueError(
                "LLM2Vec embedding dimension mismatch: "
                f"cache={manifest.get('ctxt_dim')}, model={self.embedding_dim}"
            )
        if int(manifest.get("max_length_llm", -1)) != 1:
            raise ValueError("LLM2Vec cache must contain exactly one sentence token")
        if int(manifest.get("vtxt_dim", -1)) != 1:
            raise ValueError("LLM2Vec cache vtxt_dim must be the scalar placeholder 1")
        if int(manifest.get("encoding_batch_size", -1)) != 1:
            raise ValueError(
                "Kimodo-compatible LLM2Vec caches must be encoded with batch_size=1"
            )

    def _validate_context_manifest(self) -> None:
        if self.context_cache is None:
            return
        context = self.context_cache.manifest
        global_manifest = self.cache.manifest
        for field in (
            "encoder_identity",
            "prompt_template_version",
            "model_dtype",
            "storage_dtype",
        ):
            if str(context.get(field, "")) != str(global_manifest.get(field, "")):
                raise ValueError(
                    f"Global and contextual LLM2Vec caches differ at {field}"
                )
        if str(context.get("pooling_mode", "")) != "mean":
            raise ValueError("Contextual LLM2Vec cache must use mean pooling")
        if not bool(context.get("skip_instruction", False)):
            raise ValueError(
                "Contextual LLM2Vec cache must skip instruction wrapper tokens"
            )
        if int(context.get("num_texts", -1)) != len(self.cache.index):
            raise ValueError(
                "Global and contextual LLM2Vec caches contain different row counts"
            )

    def forward(
        self,
        texts: Iterable[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        drop_prob: float = 0.0,
        force_drop: bool = False,
        profiles: Iterable[str] | None = None,
    ) -> RawTextCondition:
        text_list = [str(text) for text in texts]
        profile_list = None if profiles is None else [str(profile) for profile in profiles]
        if profile_list is not None and len(profile_list) != len(text_list):
            raise ValueError(
                f"Expected {len(text_list)} text profiles, got {len(profile_list)}"
            )

        _, sentence, lengths = self.cache.lookup_rows(
            text_list,
            profiles=profile_list,
        )
        if sentence.shape[1:] != (1, self.embedding_dim):
            raise ValueError(
                "LLM2Vec cache row shape mismatch: "
                f"expected [B,1,{self.embedding_dim}], got {tuple(sentence.shape)}"
            )
        if not bool(torch.all(lengths == 1)):
            raise ValueError("LLM2Vec cache rows must all have length 1")

        sentence = sentence.to(device=device, dtype=dtype)
        dropped = torch.tensor(
            [not text.strip() for text in text_list],
            device=device,
            dtype=torch.bool,
        )
        if force_drop:
            dropped.fill_(True)
        elif drop_prob > 0.0 and text_list:
            dropped |= torch.rand(len(text_list), device=device) < float(drop_prob)
        sentence = torch.where(
            dropped[:, None, None],
            torch.zeros_like(sentence),
            sentence,
        )
        sentence_tokens = self.token_proj(sentence)
        sentence_padding = torch.zeros(
            len(text_list),
            1,
            device=device,
            dtype=torch.bool,
        )
        pooled = torch.zeros(
            len(text_list),
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        local_tokens = None
        local_padding = None
        if self.context_cache is not None:
            if profile_list is None:
                raise ValueError(
                    "Contextual LLM2Vec conditioning requires text profiles"
                )
            contextual, contextual_lengths = self.context_cache.lookup_rows(
                text_list,
                profile_list,
            )
            contextual = contextual.to(device=device, dtype=dtype)
            if self.local_proj is None:
                raise RuntimeError("Context cache is enabled without a projection")
            local_tokens = self.local_proj(contextual)
            positions = torch.arange(
                local_tokens.shape[1],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            local_padding = positions >= contextual_lengths.to(device)[:, None]
            local_padding |= dropped[:, None]
            local_tokens = torch.where(
                local_padding[:, :, None],
                torch.zeros_like(local_tokens),
                local_tokens,
            )
        tokens = sentence_tokens
        padding = sentence_padding
        output_local_tokens = local_tokens
        output_local_padding = local_padding
        if self.token_sequence_mode == "sentence_plus_context":
            if local_tokens is None or local_padding is None:
                raise RuntimeError("Contextual token sequence was not constructed")
            tokens = torch.cat([sentence_tokens, local_tokens], dim=1)
            padding = torch.cat([sentence_padding, local_padding], dim=1)
            output_local_tokens = None
            output_local_padding = None
        elif self.token_sequence_mode == "context":
            if local_tokens is None or local_padding is None:
                raise RuntimeError("Contextual token sequence was not constructed")
            tokens = local_tokens
            padding = local_padding
            output_local_tokens = None
            output_local_padding = None
        return RawTextCondition(
            tokens=tokens,
            pooled=pooled,
            padding_mask=padding,
            local_tokens=output_local_tokens,
            local_padding_mask=output_local_padding,
        )
