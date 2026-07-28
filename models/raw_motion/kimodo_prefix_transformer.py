"""Kimodo-style prefix Transformer for one stage of HY273 denoising."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


KIMODO_PREFIX_NUM_LAYERS = 16
KIMODO_PREFIX_NUM_REGISTER_TOKENS = 49
KIMODO_PREFIX_TEXT_DIM = 4096


class SinusoidalSequencePosition(nn.Module):
    def __init__(self, hidden_dim: int, max_length: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        exponent = torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        exponent = -math.log(10000.0) * exponent / float(hidden_dim)
        frequencies = torch.exp(exponent)
        table = torch.zeros(max_length, hidden_dim, dtype=torch.float32)
        table[:, 0::2] = torch.sin(position * frequencies)
        table[:, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.table.shape[1]:
            raise ValueError(
                f"Sequence length {tokens.shape[1]} exceeds positional limit "
                f"{self.table.shape[1]}"
            )
        return tokens + self.table[:, : tokens.shape[1]].to(
            device=tokens.device,
            dtype=tokens.dtype,
        )


class KimodoPrefixTransformer(nn.Module):
    """Full self-attention over condition-prefix and motion tokens.

    Prefix order follows Kimodo: one LLM2Vec sentence token, 49 zero register
    tokens, timestep, and first-heading direction. Only motion outputs are
    returned to the denoising heads.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_layers: int = KIMODO_PREFIX_NUM_LAYERS,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        text_dim: int = KIMODO_PREFIX_TEXT_DIM,
        num_register_tokens: int = KIMODO_PREFIX_NUM_REGISTER_TOKENS,
        max_sequence_length: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.text_dim = int(text_dim)
        self.num_register_tokens = int(num_register_tokens)
        self.text_proj = nn.Linear(self.text_dim, self.hidden_size)
        self.position = SinusoidalSequencePosition(
            self.hidden_size,
            max_length=max_sequence_length,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_heads),
            dim_feedforward=int(self.hidden_size * float(mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        cond: torch.Tensor,
        motion_valid: torch.Tensor,
        text_padding_mask: torch.Tensor,
        motion_pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        del text_padding_mask, motion_pos_ids
        if motion.ndim != 3 or motion.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected motion [B,T,{self.hidden_size}], got {tuple(motion.shape)}"
            )
        if text.ndim != 3 or text.shape[1:] != (1, self.text_dim):
            raise ValueError(
                f"Expected one raw LLM2Vec token [B,1,{self.text_dim}], "
                f"got {tuple(text.shape)}"
            )
        if cond.shape != (motion.shape[0], 2, self.hidden_size):
            raise ValueError(
                "Kimodo prefix condition must contain separate timestep and "
                f"heading tokens [B,2,H], got {tuple(cond.shape)}"
            )
        if motion_valid.shape != motion.shape[:2]:
            raise ValueError("motion_valid must have shape [B,T]")

        registers = text.new_zeros(
            text.shape[0],
            self.num_register_tokens,
            self.text_dim,
        )
        text_prefix = self.text_proj(torch.cat([text, registers], dim=1))
        prefix = torch.cat([text_prefix, cond], dim=1)
        prefix_valid = torch.ones(
            prefix.shape[:2],
            device=motion.device,
            dtype=torch.bool,
        )
        sequence = self.position(torch.cat([prefix, motion], dim=1))
        valid = torch.cat([prefix_valid, motion_valid], dim=1)
        output = self.encoder(
            sequence,
            src_key_padding_mask=~valid,
        )
        return output[:, prefix.shape[1] :]
