"""Shared text-condition container for raw motion denoisers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RawTextCondition:
    tokens: torch.Tensor
    pooled: torch.Tensor
    padding_mask: torch.Tensor
