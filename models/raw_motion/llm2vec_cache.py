"""Cached LLM2Vec sentence embeddings for Kimodo-style text conditioning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from .hytext_cache import HYTextMemmapCache, LLM2VEC_CACHE_FORMAT
from .text_condition import RawTextCondition


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
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
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
        tokens = self.token_proj(sentence)
        padding = torch.zeros(
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
        return RawTextCondition(
            tokens=tokens,
            pooled=pooled,
            padding_mask=padding,
        )
