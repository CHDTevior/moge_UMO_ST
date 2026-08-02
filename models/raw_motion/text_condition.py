"""Shared text-condition container for raw motion denoisers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RawTextCondition:
    tokens: torch.Tensor
    pooled: torch.Tensor
    padding_mask: torch.Tensor
    local_tokens: torch.Tensor | None = None
    local_padding_mask: torch.Tensor | None = None
