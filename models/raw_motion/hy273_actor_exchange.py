"""Actor-aware extensions for the HY273 frame/text DiT backbone."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.codeflow.dit_blocks import FrameMotionTextDiT


class BidirectionalActorExchange(nn.Module):
    """Symmetric cross-attention between two synchronized actor streams.

    Both directions share every parameter. The final projection is zero
    initialized so adding the interaction path does not perturb a single-actor
    checkpoint before interaction training starts.
    """

    def __init__(
        self,
        hidden_size: int,
        exchange_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if exchange_dim <= 0 or exchange_dim % num_heads:
            raise ValueError("exchange_dim must be positive and divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.exchange_dim = int(exchange_dim)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.down = nn.Linear(self.hidden_size, self.exchange_dim, bias=False)
        self.attn = nn.MultiheadAttention(
            self.exchange_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.up = nn.Linear(self.exchange_dim, self.hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def _zero_anchor(self, value: torch.Tensor) -> torch.Tensor:
        anchor = sum(
            (parameter.reshape(-1)[0] * 0.0 for parameter in self.parameters()),
            value.new_zeros(()),
        )
        return value + anchor.to(dtype=value.dtype)

    def forward(
        self,
        motion: torch.Tensor,
        motion_valid: torch.Tensor,
        *,
        scene_batch_size: int,
        actor_count: int,
    ) -> torch.Tensor:
        if motion.ndim != 3:
            raise ValueError("motion must have shape [B*A,T,H]")
        if motion_valid.shape != motion.shape[:2]:
            raise ValueError("motion_valid must match motion [B*A,T]")
        scene_batch_size = int(scene_batch_size)
        actor_count = int(actor_count)
        if scene_batch_size * actor_count != motion.shape[0]:
            raise ValueError("scene_batch_size * actor_count must equal flattened batch")
        if actor_count == 1:
            return self._zero_anchor(motion)
        if actor_count != 2:
            raise ValueError("HY273 interaction exchange currently supports one or two actors")

        frames, hidden = motion.shape[1:]
        actors = motion.reshape(scene_batch_size, actor_count, frames, hidden)
        valid = motion_valid.reshape(scene_batch_size, actor_count, frames)
        both_present = valid.any(dim=-1).all(dim=1)

        projected = self.down(self.norm(actors))
        queries = projected.reshape(scene_batch_size * actor_count, frames, -1)
        other = projected.flip(dims=(1,)).reshape(
            scene_batch_size * actor_count, frames, -1
        )
        other_valid = valid.flip(dims=(1,)).reshape(
            scene_batch_size * actor_count, frames
        )

        # MultiheadAttention cannot consume an all-padding key row. Such rows
        # are made numerically finite and then removed by the explicit gate.
        safe_other_valid = other_valid.clone()
        all_padding = ~safe_other_valid.any(dim=1)
        if bool(all_padding.any()):
            safe_other_valid[all_padding, 0] = True
            other = other.clone()
            other[all_padding, 0] = 0.0
        exchanged, _ = self.attn(
            queries,
            other,
            other,
            key_padding_mask=~safe_other_valid,
            need_weights=False,
        )
        update = self.up(exchanged).reshape(
            scene_batch_size, actor_count, frames, hidden
        )
        gate = (
            valid
            & both_present[:, None, None]
        ).unsqueeze(-1)
        output = actors + torch.where(gate, update, torch.zeros_like(update))
        return output.reshape_as(motion)


class ActorExchangeFrameMotionTextDiT(FrameMotionTextDiT):
    """FrameMotionTextDiT with actor exchange after every transformer block."""

    def __init__(
        self,
        *args,
        actor_exchange_dim: int = 256,
        actor_exchange_heads: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        hidden_size = int(kwargs.get("hidden_size", args[0] if args else 0))
        dropout = float(kwargs.get("dropout", 0.0))
        # Preserve the base DiT initialization stream. The zero-output actor
        # path is a Stage-B extension and must not perturb Stage-A body weights.
        with torch.random.fork_rng(devices=[]):
            self.double_actor_exchange = nn.ModuleList(
                [
                    BidirectionalActorExchange(
                        hidden_size,
                        exchange_dim=int(actor_exchange_dim),
                        num_heads=int(actor_exchange_heads),
                        dropout=dropout,
                    )
                    for _ in self.double_blocks
                ]
            )
            self.single_actor_exchange = nn.ModuleList(
                [
                    BidirectionalActorExchange(
                        hidden_size,
                        exchange_dim=int(actor_exchange_dim),
                        num_heads=int(actor_exchange_heads),
                        dropout=dropout,
                    )
                    for _ in self.single_blocks
                ]
            )

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        cond: torch.Tensor,
        motion_valid: torch.Tensor,
        text_padding_mask: torch.Tensor,
        motion_pos_ids: torch.Tensor,
        control_cond: Optional[torch.Tensor] = None,
        local_text: Optional[torch.Tensor] = None,
        local_text_padding_mask: Optional[torch.Tensor] = None,
        *,
        scene_batch_size: int | None = None,
        actor_count: int = 1,
    ) -> torch.Tensor:
        if scene_batch_size is None:
            scene_batch_size = int(motion.shape[0]) // int(actor_count)
        text_valid = ~text_padding_mask
        control_tokens = self._encode_control(control_cond, motion, motion_valid)
        layer_idx = 0
        for double_index, (block, exchange) in enumerate(zip(
            self.double_blocks, self.double_actor_exchange
        )):
            control_k = control_v = control_bias = None
            if control_tokens is not None:
                control_k, control_v = self._control_kv(control_tokens, layer_idx)
                control_bias = self.control_attn_bias[layer_idx]
            motion, text = block(
                motion,
                text,
                cond,
                motion_valid=motion_valid,
                text_valid=text_valid,
                pos_ids=motion_pos_ids,
                rope_axes_dims=self.rope_axes_dims,
                control_k=control_k,
                control_v=control_v,
                control_valid=motion_valid,
                control_pos=motion_pos_ids,
                control_attn_bias=control_bias,
            )
            motion = self._inject_local_text(
                motion,
                motion_valid,
                local_text,
                local_text_padding_mask,
                double_index,
            )
            motion = exchange(
                motion,
                motion_valid,
                scene_batch_size=int(scene_batch_size),
                actor_count=int(actor_count),
            )
            layer_idx += 1

        text_pos = torch.zeros(
            text.shape[0],
            text.shape[1],
            motion_pos_ids.shape[-1],
            device=motion_pos_ids.device,
            dtype=motion_pos_ids.dtype,
        )
        motion_tokens = motion.shape[1]
        x = torch.cat([motion, text], dim=1)
        valid = torch.cat([motion_valid, text_valid], dim=1)
        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
        for block, exchange in zip(
            self.single_blocks, self.single_actor_exchange
        ):
            control_k = control_v = control_bias = None
            if control_tokens is not None:
                control_k, control_v = self._control_kv(control_tokens, layer_idx)
                control_bias = self.control_attn_bias[layer_idx]
            x = block(
                x,
                cond,
                valid=valid,
                pos_ids=pos_ids,
                rope_axes_dims=self.rope_axes_dims,
                control_k=control_k,
                control_v=control_v,
                control_valid=motion_valid,
                control_pos=motion_pos_ids,
                control_attn_bias=control_bias,
                motion_token_count=motion_tokens,
            )
            x_motion = exchange(
                x[:, :motion_tokens],
                motion_valid,
                scene_batch_size=int(scene_batch_size),
                actor_count=int(actor_count),
            )
            x = torch.cat([x_motion, x[:, motion_tokens:]], dim=1)
            layer_idx += 1
        return x[:, :motion_tokens]
