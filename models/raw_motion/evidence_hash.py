"""Stable content hashes for tensors and model state used as evaluation evidence."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    if tensor.numel():
        byte_view = tensor.reshape(-1).view(torch.uint8)
        digest.update(memoryview(byte_view.numpy()))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    rows = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a tensor")
        rows.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "sha256": tensor_sha256(value),
            }
        )
    return canonical_sha256(rows)


def combined_tensor_sha256(values: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256(
        [
            {
                "name": name,
                "sha256": tensor_sha256(values[name]),
            }
            for name in sorted(values)
        ]
    )
