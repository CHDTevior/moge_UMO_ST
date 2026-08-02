"""Ragged contextual-token cache for profile-aware LLM2Vec conditioning."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .hytext_cache import hytext_profile_key


LLM2VEC_CONTEXT_CACHE_FORMAT = "llm2vec_context_ragged_profile_v1"


class LLM2VecContextMemmapCache:
    """Read variable-length LLM2Vec token states from packed mmap shards."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        embedding_dim: int = 4096,
        max_open_shards: int = 16,
        strict: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.embedding_dim = int(embedding_dim)
        self.max_open_shards = max(1, int(max_open_shards))
        self.strict = bool(strict)
        manifest_path = self.cache_dir / "manifest.json"
        index_path = self.cache_dir / "index.json"
        if not manifest_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(
                f"Incomplete LLM2Vec contextual cache under {self.cache_dir}"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.index: dict[str, dict[str, object]] = json.loads(
            index_path.read_text(encoding="utf-8")
        )
        if self.manifest.get("format") != LLM2VEC_CONTEXT_CACHE_FORMAT:
            raise ValueError(
                "Unsupported LLM2Vec contextual cache format "
                f"{self.manifest.get('format')!r}"
            )
        if int(self.manifest.get("embedding_dim", -1)) != self.embedding_dim:
            raise ValueError(
                "LLM2Vec contextual embedding dimension mismatch: "
                f"cache={self.manifest.get('embedding_dim')} "
                f"model={self.embedding_dim}"
            )
        if int(self.manifest.get("encoding_batch_size", -1)) != 1:
            raise ValueError(
                "LLM2Vec contextual states must be encoded with batch_size=1"
            )
        if str(self.manifest.get("model_dtype", "")).lower() != "bf16":
            raise ValueError(
                "LLM2Vec contextual states must use bf16 encoder inference"
            )
        if str(self.manifest.get("storage_dtype", "")).lower() != "fp16":
            raise ValueError("LLM2Vec contextual states must use fp16 storage")
        if str(self.manifest.get("pooling_mode", "")) != "mean":
            raise ValueError("LLM2Vec contextual cache must use mean pooling")
        if not bool(self.manifest.get("skip_instruction", False)):
            raise ValueError(
                "LLM2Vec contextual cache must skip instruction wrapper tokens"
            )
        self.encoder_identity = str(
            self.manifest.get("encoder_identity", "")
        )
        self.prompt_template_version = str(
            self.manifest.get("prompt_template_version", "")
        )
        if not self.encoder_identity or not self.prompt_template_version:
            raise ValueError(
                "Contextual cache is missing its text-key contract"
            )
        self._shards: OrderedDict[str, np.ndarray] = OrderedDict()
        self._runtime_rows: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def _key(self, text: str, profile: str | None) -> str:
        if not profile:
            raise ValueError(
                "Profile-aware contextual lookup requires one profile per text"
            )
        return hytext_profile_key(
            text,
            profile,
            encoder_identity=self.encoder_identity,
            prompt_template_version=self.prompt_template_version,
        )

    def has_text(self, text: str, profile: str) -> bool:
        key = self._key(str(text), str(profile))
        return key in self._runtime_rows or key in self.index

    def storage_round_trip(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(tensor):
            raise TypeError("Contextual cache values must be floating-point tensors")
        source = tensor.detach().float().cpu()
        if not bool(torch.isfinite(source).all()):
            raise ValueError("Runtime contextual rows contain non-finite values")
        array = source.numpy().astype(np.float16, copy=True)
        if not bool(np.isfinite(array).all()):
            raise ValueError("Runtime contextual rows overflowed fp16 storage")
        return torch.from_numpy(array)

    def add_runtime_rows(
        self,
        texts: Iterable[str],
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        profiles: Iterable[str],
    ) -> None:
        """Register process-local ragged rows without mutating the disk cache."""

        text_list = [str(text) for text in texts]
        profile_list = [str(profile) for profile in profiles]
        if len(profile_list) != len(text_list):
            raise ValueError(
                f"Expected {len(text_list)} profiles, got {len(profile_list)}"
            )
        if (
            tokens.ndim != 3
            or tokens.shape[0] != len(text_list)
            or tokens.shape[2] != self.embedding_dim
        ):
            raise ValueError(
                "Runtime contextual tokens must have shape [B,L,embedding_dim]"
            )
        if lengths.shape != (len(text_list),):
            raise ValueError("Runtime contextual lengths must have shape [B]")
        length_rows = lengths.detach().long().cpu()
        if bool((length_rows < 0).any()) or bool(
            (length_rows > int(tokens.shape[1])).any()
        ):
            raise ValueError("Runtime contextual lengths are outside token storage")
        stored = self.storage_round_trip(tokens).numpy()
        for row, (text, profile) in enumerate(zip(text_list, profile_list)):
            length = int(length_rows[row].item())
            self._runtime_rows[self._key(text, profile)] = np.asarray(
                stored[row, :length]
            ).copy()

    def _open_shard(self, shard: str) -> np.ndarray:
        shard = str(shard)
        if shard in self._shards:
            opened = self._shards.pop(shard)
            self._shards[shard] = opened
            return opened
        path = self.cache_dir / "shards" / shard / "tokens.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing contextual token shard: {path}")
        opened = np.load(path, mmap_mode="r")
        if opened.ndim != 2 or opened.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Bad contextual token shard shape {opened.shape} at {path}"
            )
        if opened.dtype != np.dtype(np.float16):
            raise ValueError(
                f"Unsupported contextual token dtype {opened.dtype} at {path}"
            )
        self._shards[shard] = opened
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return opened

    def lookup_rows(
        self,
        texts: Iterable[str],
        profiles: Iterable[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_list = [str(text) for text in texts]
        profile_list = [str(profile) for profile in profiles]
        if len(profile_list) != len(text_list):
            raise ValueError(
                f"Expected {len(text_list)} profiles, got {len(profile_list)}"
            )

        entries: list[dict[str, object]] = []
        for text, profile in zip(text_list, profile_list):
            key = self._key(text, profile)
            runtime = self._runtime_rows.get(key)
            if runtime is not None:
                entries.append(
                    {
                        "runtime_key": key,
                        "length": int(runtime.shape[0]),
                    }
                )
                continue
            entry = self.index.get(key)
            if entry is None:
                empty_entry = self.index.get(self._key("", profile))
                if self.strict or empty_entry is None:
                    raise KeyError(
                        "LLM2Vec contextual cache miss for "
                        f"text={text!r} profile={profile!r}"
                    )
                entry = empty_entry
            entries.append(entry)

        lengths = torch.tensor(
            [int(entry["length"]) for entry in entries],
            dtype=torch.long,
        )
        if bool((lengths < 0).any()):
            raise ValueError("Contextual cache contains a negative token length")
        padded_length = max(1, int(lengths.max().item()) if len(entries) else 0)
        storage_dtype = np.dtype(np.float16)
        padded = np.zeros(
            (len(entries), padded_length, self.embedding_dim),
            dtype=storage_dtype,
        )

        groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for output_index, entry in enumerate(entries):
            length = int(entry["length"])
            runtime_key = entry.get("runtime_key")
            if runtime_key is not None:
                padded[output_index, :length] = self._runtime_rows[
                    str(runtime_key)
                ]
                continue
            if length:
                groups[str(entry["shard"])].append(
                    (output_index, int(entry["offset"]), length)
                )
        for shard, rows in groups.items():
            opened = self._open_shard(shard)
            for output_index, offset, length in rows:
                end = offset + length
                if offset < 0 or end > opened.shape[0]:
                    raise ValueError(
                        f"Contextual cache slice [{offset}:{end}] is outside "
                        f"{shard} with {opened.shape[0]} tokens"
                    )
                padded[output_index, :length] = np.asarray(
                    opened[offset:end]
                )
        return torch.from_numpy(padded), lengths
