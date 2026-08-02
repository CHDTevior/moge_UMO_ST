"""Cached HY-Motion text embeddings for HY273 raw-flow training."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .text_condition import RawTextCondition


_SPACE_RE = re.compile(r"\s+")
PROFILE_CACHE_FORMAT = "hytext_memmap_profile_v2"
LLM2VEC_CACHE_FORMAT = "llm2vec_memmap_profile_v1"
PROFILE_AWARE_CACHE_FORMATS = {
    PROFILE_CACHE_FORMAT,
    LLM2VEC_CACHE_FORMAT,
}


def normalize_text_key(text: str) -> str:
    """Normalize only formatting so training captions still keep their wording."""
    return _SPACE_RE.sub(" ", str(text).strip())


def hytext_key(text: str) -> str:
    return hashlib.sha1(normalize_text_key(text).encode("utf-8")).hexdigest()


def hytext_profile_key(
    text: str,
    profile: str,
    *,
    encoder_identity: str,
    prompt_template_version: str,
) -> str:
    """Hash every semantic input that can change a cached text embedding."""

    payload = {
        "encoder_identity": str(encoder_identity),
        "normalized_text": normalize_text_key(text),
        "profile": str(profile),
        "prompt_template_version": str(prompt_template_version),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HYTextMemmapCache:
    """Lazy row-level reader for Qwen3 token and CLIP-L sentence embeddings."""

    def __init__(self, cache_dir: str | Path, max_open_shards: int = 8, strict: bool = True) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.strict = bool(strict)
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"HYText cache directory not found: {self.cache_dir}")
        index_path = self.cache_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"HYText cache index not found: {index_path}")
        self.index: dict[str, dict[str, object]] = json.loads(index_path.read_text())
        manifest_path = self.cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        fmt = self.manifest.get("format")
        if fmt is not None and fmt not in {
            "hytext_memmap_v1",
            *PROFILE_AWARE_CACHE_FORMATS,
        }:
            raise ValueError(f"Unsupported HYText cache format={fmt!r} under {self.cache_dir}")
        self.profile_aware = fmt in PROFILE_AWARE_CACHE_FORMATS
        self.encoder_identity = str(self.manifest.get("encoder_identity", ""))
        self.prompt_template_version = str(
            self.manifest.get("prompt_template_version", "")
        )
        if self.profile_aware and (
            not self.encoder_identity or not self.prompt_template_version
        ):
            raise ValueError(
                "Profile-aware HYText cache requires encoder_identity and "
                "prompt_template_version"
            )
        self.max_open_shards = max(1, int(max_open_shards))
        self._shards: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        # Demo/inference callers may register freshly encoded text rows here.
        # They deliberately stay process-local: the immutable training cache and
        # its audited manifest/index are never rewritten by interactive prompts.
        self._runtime_rows: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _open_shard(self, shard: str) -> dict[str, np.ndarray]:
        shard = str(shard)
        if shard in self._shards:
            opened = self._shards.pop(shard)
            self._shards[shard] = opened
            return opened
        shard_dir = self.cache_dir / "shards" / shard
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"HYText shard directory not found: {shard_dir}")
        opened = {
            "ctxt": np.load(shard_dir / "ctxt.npy", mmap_mode="r"),
            "vtxt": np.load(shard_dir / "vtxt.npy", mmap_mode="r"),
            "ctxt_len": np.load(shard_dir / "ctxt_len.npy", mmap_mode="r"),
        }
        self._shards[shard] = opened
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return opened

    def _key(self, text: str, profile: str | None) -> str:
        if not self.profile_aware:
            return hytext_key(text)
        if not profile:
            raise ValueError("Profile-aware HYText lookup requires one profile per text")
        return hytext_profile_key(
            text,
            profile,
            encoder_identity=self.encoder_identity,
            prompt_template_version=self.prompt_template_version,
        )

    def has_text(self, text: str, profile: str | None = None) -> bool:
        """Return whether an exact persistent or process-local row exists."""

        key = self._key(str(text), profile)
        return key in self.index or key in self._runtime_rows

    def text_source(self, text: str, profile: str | None = None) -> str | None:
        """Identify where an exact text row is stored."""

        key = self._key(str(text), profile)
        if key in self._runtime_rows:
            return "runtime_encoder"
        if key in self.index:
            return "persistent_cache"
        return None

    def storage_numpy_dtype(self) -> np.dtype:
        """Return the dtype used by persistent cache rows."""

        configured = self.manifest.get("storage_dtype")
        if configured is not None:
            aliases = {
                "fp16": np.dtype(np.float16),
                "float16": np.dtype(np.float16),
                "fp32": np.dtype(np.float32),
                "float32": np.dtype(np.float32),
            }
            dtype = aliases.get(str(configured).lower())
            if dtype is None:
                raise ValueError(
                    f"Unsupported HYText cache storage_dtype={configured!r}"
                )
            if self.index:
                first = next(iter(self.index.values()))
                opened = self._open_shard(str(first["shard"]))
                ctxt_dtype = np.dtype(opened["ctxt"].dtype)
                vtxt_dtype = np.dtype(opened["vtxt"].dtype)
                if ctxt_dtype != dtype or vtxt_dtype != dtype:
                    raise ValueError(
                        "HYText cache arrays differ from manifest storage_dtype"
                    )
            return dtype
        if self.index:
            first = next(iter(self.index.values()))
            opened = self._open_shard(str(first["shard"]))
            ctxt_dtype = np.dtype(opened["ctxt"].dtype)
            vtxt_dtype = np.dtype(opened["vtxt"].dtype)
            if ctxt_dtype != vtxt_dtype:
                raise ValueError(
                    "HYText cache ctxt/vtxt arrays use different storage dtypes"
                )
            if ctxt_dtype not in {np.dtype(np.float16), np.dtype(np.float32)}:
                raise ValueError(
                    f"Unsupported HYText cache array dtype={ctxt_dtype}"
                )
            return ctxt_dtype
        return np.dtype(np.float32)

    def storage_round_trip(self, tensor: torch.Tensor) -> torch.Tensor:
        """Match the float32-to-storage conversion used by cache builders."""

        if not torch.is_floating_point(tensor):
            raise TypeError("HYText cache values must be floating-point tensors")
        source = tensor.detach().float().cpu()
        if not bool(torch.isfinite(source).all()):
            raise ValueError("Runtime HYText rows contain non-finite values")
        array = source.numpy().astype(self.storage_numpy_dtype(), copy=True)
        if not bool(np.isfinite(array).all()):
            raise ValueError(
                "Runtime HYText rows overflowed the persistent storage dtype"
            )
        return torch.from_numpy(array)

    def add_runtime_rows(
        self,
        texts: Iterable[str],
        vtxt: torch.Tensor,
        ctxt: torch.Tensor,
        lengths: torch.Tensor,
        profiles: Iterable[str] | None = None,
    ) -> None:
        """Register in-memory HYText rows without mutating the training cache."""

        text_list = [normalize_text_key(str(text)) for text in texts]
        if profiles is None:
            profile_list: list[str | None] = [None] * len(text_list)
        else:
            profile_list = [str(profile) for profile in profiles]
            if len(profile_list) != len(text_list):
                raise ValueError(
                    f"Expected {len(text_list)} HYText profiles, got {len(profile_list)}"
                )
        if (
            vtxt.ndim != 3
            or vtxt.shape[0] != len(text_list)
            or vtxt.shape[1] != 1
        ):
            raise ValueError("Runtime HYText vtxt must have shape [B,1,D]")
        if ctxt.ndim != 3 or ctxt.shape[0] != len(text_list):
            raise ValueError("Runtime HYText ctxt must have shape [B,L,D]")
        if lengths.shape != (len(text_list),):
            raise ValueError("Runtime HYText lengths must have shape [B]")
        if int(vtxt.shape[-1]) != int(self.manifest.get("vtxt_dim", vtxt.shape[-1])):
            raise ValueError("Runtime HYText vtxt dimension differs from the cache contract")
        if int(ctxt.shape[-1]) != int(self.manifest.get("ctxt_dim", ctxt.shape[-1])):
            raise ValueError("Runtime HYText ctxt dimension differs from the cache contract")
        expected_length = self.manifest.get("max_length_llm")
        if expected_length is not None and int(ctxt.shape[1]) != int(expected_length):
            raise ValueError(
                "Runtime HYText token length differs from the cache contract"
            )
        length_values = lengths.detach().long().cpu()
        if bool((length_values < 0).any()) or bool(
            (length_values > int(ctxt.shape[1])).any()
        ):
            raise ValueError("Runtime HYText lengths are outside the token array")

        vtxt_rows = self.storage_round_trip(vtxt).numpy()
        ctxt_rows = self.storage_round_trip(ctxt).numpy()
        length_rows = length_values.numpy()
        for index, (text, profile) in enumerate(zip(text_list, profile_list)):
            key = self._key(text, profile)
            self._runtime_rows[key] = (
                np.asarray(vtxt_rows[index]).copy(),
                np.asarray(ctxt_rows[index]).copy(),
                int(length_rows[index].item()),
            )

    def lookup_rows(
        self,
        texts: Iterable[str],
        profiles: Iterable[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_list = [str(text) for text in texts]
        if profiles is None:
            profile_list: list[str | None] = [None] * len(text_list)
        else:
            profile_list = [str(profile) for profile in profiles]
            if len(profile_list) != len(text_list):
                raise ValueError(
                    f"Expected {len(text_list)} HYText profiles, got {len(profile_list)}"
                )
        entries: list[dict[str, object]] = []
        for text, profile in zip(text_list, profile_list):
            key = self._key(text, profile)
            if key in self._runtime_rows:
                entries.append({"runtime_key": key})
                continue
            entry = self.index.get(key)
            if entry is None:
                empty_entry = self.index.get(self._key("", profile))
                if self.strict or empty_entry is None:
                    raise KeyError(
                        f"HYText cache miss for key={key} text={normalize_text_key(str(text))!r} "
                        f"profile={profile!r} "
                        f"under {self.cache_dir}"
                    )
                entry = empty_entry
            entries.append(entry)
        if not entries:
            ctxt_dim = int(self.manifest.get("ctxt_dim", 4096))
            vtxt_dim = int(self.manifest.get("vtxt_dim", 768))
            max_len = int(self.manifest.get("max_length_llm", 0))
            return (
                torch.empty(0, 1, vtxt_dim),
                torch.empty(0, max_len, ctxt_dim),
                torch.empty(0, dtype=torch.long),
            )

        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for out_i, entry in enumerate(entries):
            runtime_key = entry.get("runtime_key")
            if runtime_key is None:
                groups[str(entry["shard"])].append((out_i, int(entry["row"])))

        ctxt_rows: list[np.ndarray | None] = [None] * len(entries)
        vtxt_rows: list[np.ndarray | None] = [None] * len(entries)
        len_rows: list[int] = [0] * len(entries)
        for out_i, entry in enumerate(entries):
            runtime_key = entry.get("runtime_key")
            if runtime_key is None:
                continue
            runtime_vtxt, runtime_ctxt, runtime_length = self._runtime_rows[
                str(runtime_key)
            ]
            vtxt_rows[out_i] = runtime_vtxt
            ctxt_rows[out_i] = runtime_ctxt
            len_rows[out_i] = runtime_length
        for shard, pairs in groups.items():
            opened = self._open_shard(shard)
            row_ids = np.array([row for _, row in pairs], dtype=np.int64)
            ctxt_batch = np.asarray(opened["ctxt"][row_ids]).copy()
            vtxt_batch = np.asarray(opened["vtxt"][row_ids]).copy()
            len_batch = np.asarray(opened["ctxt_len"][row_ids]).copy()
            for j, (out_i, _) in enumerate(pairs):
                ctxt_rows[out_i] = ctxt_batch[j]
                vtxt_rows[out_i] = vtxt_batch[j]
                len_rows[out_i] = int(len_batch[j].item())
        if any(row is None for row in ctxt_rows) or any(row is None for row in vtxt_rows):
            raise RuntimeError("Internal HYText cache grouping error: missing output rows")
        ctxt = torch.from_numpy(np.stack(ctxt_rows, axis=0))
        vtxt = torch.from_numpy(np.stack(vtxt_rows, axis=0))
        lengths = torch.tensor(len_rows, dtype=torch.long)
        return vtxt, ctxt, lengths


class CachedHYTextEncoder(nn.Module):
    """Projection bridge from cached HY-Motion HYText embeddings to RawTextCondition."""

    def __init__(
        self,
        hidden_dim: int,
        cache_dir: str | Path,
        max_text_tokens: int = 128,
        ctxt_dim: int = 4096,
        vtxt_dim: int = 768,
        max_open_shards: int = 8,
        strict_cache: bool = True,
    ) -> None:
        super().__init__()
        self.supports_profiles = True
        self.hidden_dim = int(hidden_dim)
        self.max_text_tokens = int(max_text_tokens)
        self.ctxt_dim = int(ctxt_dim)
        self.vtxt_dim = int(vtxt_dim)
        self.cache = HYTextMemmapCache(cache_dir, max_open_shards=max_open_shards, strict=strict_cache)
        self._validate_manifest()
        self.token_proj = nn.Linear(self.ctxt_dim, self.hidden_dim)
        self.pooled_proj = nn.Sequential(
            nn.Linear(self.vtxt_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _validate_manifest(self) -> None:
        manifest = self.cache.manifest
        cache_ctxt_dim = manifest.get("ctxt_dim")
        cache_vtxt_dim = manifest.get("vtxt_dim")
        cache_max_len = manifest.get("max_length_llm")
        if cache_ctxt_dim is not None and int(cache_ctxt_dim) != self.ctxt_dim:
            raise ValueError(f"HYText ctxt_dim mismatch: cache={cache_ctxt_dim}, model={self.ctxt_dim}")
        if cache_vtxt_dim is not None and int(cache_vtxt_dim) != self.vtxt_dim:
            raise ValueError(f"HYText vtxt_dim mismatch: cache={cache_vtxt_dim}, model={self.vtxt_dim}")
        if cache_max_len is not None and int(cache_max_len) < self.max_text_tokens:
            raise ValueError(
                f"HYText cache max_length_llm={cache_max_len} is shorter than model max_text_tokens={self.max_text_tokens}"
            )
        if self.cache.index:
            first = next(iter(self.cache.index.values()))
            opened = self.cache._open_shard(str(first["shard"]))
            ctxt = opened["ctxt"]
            vtxt = opened["vtxt"]
            if ctxt.ndim != 3 or int(ctxt.shape[-1]) != self.ctxt_dim:
                raise ValueError(f"HYText ctxt array shape {ctxt.shape} does not match ctxt_dim={self.ctxt_dim}")
            if vtxt.ndim != 3 or int(vtxt.shape[-1]) != self.vtxt_dim:
                raise ValueError(f"HYText vtxt array shape {vtxt.shape} does not match vtxt_dim={self.vtxt_dim}")

    def forward(
        self,
        texts: Iterable[str],
        device: torch.device,
        dtype: torch.dtype,
        drop_prob: float = 0.0,
        force_drop: bool = False,
        profiles: Iterable[str] | None = None,
    ) -> RawTextCondition:
        text_list = [str(t) for t in texts]
        profile_list = None if profiles is None else [str(p) for p in profiles]
        if profile_list is not None and len(profile_list) != len(text_list):
            raise ValueError(
                f"Expected {len(text_list)} text profiles, got {len(profile_list)}"
            )
        if force_drop:
            text_list = [""] * len(text_list)
        elif drop_prob > 0.0 and text_list:
            keep = torch.rand(len(text_list), device=device) >= float(drop_prob)
            text_list = [text if bool(keep[i].item()) else "" for i, text in enumerate(text_list)]

        vtxt, ctxt, lengths = self.cache.lookup_rows(text_list, profiles=profile_list)
        tokens = ctxt[:, : self.max_text_tokens]
        if tokens.shape[1] < self.max_text_tokens:
            pad = tokens.new_zeros(tokens.shape[0], self.max_text_tokens - tokens.shape[1], tokens.shape[2])
            tokens = torch.cat([tokens, pad], dim=1)
        pooled = vtxt[:, 0]
        lengths = lengths.clamp(min=0, max=self.max_text_tokens)
        arange = torch.arange(self.max_text_tokens).view(1, self.max_text_tokens)
        padding = arange >= lengths.view(-1, 1)
        if padding.all(dim=1).any():
            padding = padding.clone()
            padding[padding.all(dim=1), 0] = False

        tokens = self.token_proj(tokens.to(device=device, dtype=dtype))
        pooled = self.pooled_proj(pooled.to(device=device, dtype=dtype))
        padding = padding.to(device=device)
        return RawTextCondition(tokens=tokens, pooled=pooled, padding_mask=padding)
