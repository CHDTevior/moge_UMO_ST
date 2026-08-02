"""Interleaved Kimodo-like root/body rectified-flow denoiser for HY273."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .hy273_root_conditioning import KimodoRootConditioner
from .hy273_slices import BODY_DIM, DIM_HY273, LOCAL_ROOT_DIM, ROOT_DIM
from .raw_flow_dit import (
    FrameMotionTextDiT,
    FrozenCLIPTextEncoder,
    NullTextEncoder,
    TimestepEmbedder,
)
from .text_condition import RawTextCondition


TEXT_GLOBAL_CONDITIONING_MODES = (
    "pooled_adaln",
    "qwen_tokens_only",
    "llm2vec_tokens_only",
)
BACKBONE_TYPES = ("flux", "kimodo_prefix")


@dataclass
class KimodoLikeFlowOutput:
    prediction: torch.Tensor
    root_prediction_raw: torch.Tensor
    root_for_body: torch.Tensor
    local_root: torch.Tensor


class HY273RedenoiseKimodoLike(nn.Module):
    """Predict clean global root, then clean body conditioned on predicted local root."""

    source_dim = DIM_HY273
    mask_dim = DIM_HY273
    model_in_dim = DIM_HY273 * 2
    output_dim = DIM_HY273

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        root_depth_double: int = 3,
        root_depth_single: int = 6,
        body_depth_double: int = 3,
        body_depth_single: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_text_tokens: int = 128,
        text_encoder: str = "hy_cache",
        clip_path: str = "checkpoints/clip/ViT-B-32.pt",
        clip_version: str = "ViT-B/32",
        hytext_cache_dir: str = "",
        hytext_ctxt_dim: int = 4096,
        hytext_vtxt_dim: int = 768,
        hytext_max_open_shards: int = 64,
        hytext_strict_cache: bool = True,
        motion_stats_dir: str = "",
        local_root_stats_dir: str = "",
        fps: float = 30.0,
        stats_variance_eps: float = 1e-5,
        detach_root_bridge: bool = True,
        self_conditioning: bool = False,
        self_cond_scale: float = 1.0,
        zero_init_self_cond: bool = True,
        text_global_conditioning: str = "pooled_adaln",
        text_fusion_mode: str = "f00",
        backbone_type: str = "flux",
        actor_exchange: bool = False,
        actor_exchange_dim: int = 256,
        actor_exchange_heads: int = 4,
        llm2vec_context_cache_dir: str = "",
        llm2vec_context_max_open_shards: int = 16,
        local_text_cross_attention: bool = False,
        llm2vec_token_sequence_mode: str = "sentence",
    ) -> None:
        super().__init__()
        if not motion_stats_dir or not local_root_stats_dir:
            raise ValueError("redenoise_kimodo_like requires motion_stats_dir and local_root_stats_dir")
        if text_global_conditioning not in TEXT_GLOBAL_CONDITIONING_MODES:
            raise ValueError(
                "text_global_conditioning must be one of "
                f"{TEXT_GLOBAL_CONDITIONING_MODES}, got {text_global_conditioning!r}"
            )
        if backbone_type not in BACKBONE_TYPES:
            raise ValueError(
                f"backbone_type must be one of {BACKBONE_TYPES}, got {backbone_type!r}"
            )
        self.hidden_dim = int(hidden_dim)
        self.detach_root_bridge = bool(detach_root_bridge)
        self.self_conditioning = bool(self_conditioning)
        self.self_cond_scale = float(self_cond_scale)
        self.text_global_conditioning = str(text_global_conditioning)
        self.text_fusion_mode = str(text_fusion_mode).lower()
        self.backbone_type = str(backbone_type)
        self.actor_exchange = bool(actor_exchange)
        self.local_text_cross_attention = bool(local_text_cross_attention)
        if self.actor_exchange and self.backbone_type != "flux":
            raise ValueError("Actor exchange currently requires the flux backbone")
        if self.local_text_cross_attention and (
            self.backbone_type != "flux"
            or text_encoder not in {"llm2vec_cache", "kimodo_llm2vec_cache"}
            or not str(llm2vec_context_cache_dir)
        ):
            raise ValueError(
                "Local text cross-attention requires the Flux LLM2Vec path "
                "and a contextual cache"
            )

        self.root_input_proj = nn.Linear(self.model_in_dim, self.hidden_dim)
        self.body_input_proj = nn.Linear(LOCAL_ROOT_DIM + BODY_DIM + DIM_HY273, self.hidden_dim)
        self.root_output_proj = nn.Linear(self.hidden_dim, ROOT_DIM)
        self.body_output_proj = nn.Linear(self.hidden_dim, BODY_DIM)
        self.root_self_cond_proj = nn.Linear(ROOT_DIM, self.hidden_dim)
        self.body_self_cond_proj = nn.Linear(BODY_DIM, self.hidden_dim)
        if zero_init_self_cond:
            nn.init.zeros_(self.root_self_cond_proj.weight)
            nn.init.zeros_(self.root_self_cond_proj.bias)
            nn.init.zeros_(self.body_self_cond_proj.weight)
            nn.init.zeros_(self.body_self_cond_proj.bias)
        if not self.self_conditioning:
            for module in (self.root_self_cond_proj, self.body_self_cond_proj):
                for param in module.parameters():
                    param.requires_grad_(False)

        self.timestep_embed = TimestepEmbedder(self.hidden_dim)
        self.direction_embed = nn.Sequential(
            nn.Linear(2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if text_encoder == "clip":
            self.text_encoder = FrozenCLIPTextEncoder(
                hidden_dim=self.hidden_dim,
                clip_path=clip_path,
                clip_version=clip_version,
                max_text_tokens=max_text_tokens,
            )
        elif text_encoder in {"hy_cache", "hytext_cache", "qwen_clip_cache"}:
            from .hytext_cache import CachedHYTextEncoder

            self.text_encoder = CachedHYTextEncoder(
                hidden_dim=self.hidden_dim,
                cache_dir=hytext_cache_dir,
                max_text_tokens=max_text_tokens,
                ctxt_dim=hytext_ctxt_dim,
                vtxt_dim=hytext_vtxt_dim,
                max_open_shards=hytext_max_open_shards,
                strict_cache=hytext_strict_cache,
            )
        elif text_encoder in {"llm2vec_cache", "kimodo_llm2vec_cache"}:
            from .llm2vec_cache import CachedLLM2VecTextEncoder

            self.text_encoder = CachedLLM2VecTextEncoder(
                hidden_dim=self.hidden_dim,
                cache_dir=hytext_cache_dir,
                embedding_dim=hytext_ctxt_dim,
                max_open_shards=hytext_max_open_shards,
                strict_cache=hytext_strict_cache,
                project_tokens=self.backbone_type == "flux",
                context_cache_dir=llm2vec_context_cache_dir,
                context_max_open_shards=llm2vec_context_max_open_shards,
                token_sequence_mode=llm2vec_token_sequence_mode,
            )
        elif text_encoder in {"none", "null"}:
            self.text_encoder = NullTextEncoder(self.hidden_dim, max_text_tokens=max_text_tokens)
        else:
            raise ValueError(f"Unsupported text_encoder: {text_encoder}")
        if text_encoder in {"llm2vec_cache", "kimodo_llm2vec_cache"}:
            if self.text_global_conditioning != "llm2vec_tokens_only":
                raise ValueError(
                    "LLM2Vec experiments require text_global_conditioning="
                    "'llm2vec_tokens_only'"
                )
        elif self.text_global_conditioning == "llm2vec_tokens_only":
            raise ValueError(
                "llm2vec_tokens_only requires text_encoder='llm2vec_cache'"
            )

        if self.backbone_type == "flux":
            backbone_cls = FrameMotionTextDiT
            actor_kwargs = {}
            if self.actor_exchange:
                from .hy273_actor_exchange import ActorExchangeFrameMotionTextDiT

                backbone_cls = ActorExchangeFrameMotionTextDiT
                actor_kwargs = {
                    "actor_exchange_dim": int(actor_exchange_dim),
                    "actor_exchange_heads": int(actor_exchange_heads),
                }
            backbone_kwargs = {
                "hidden_size": self.hidden_dim,
                "num_heads": int(num_heads),
                "mlp_ratio": float(mlp_ratio),
                "dropout": float(dropout),
                "text_fusion_mode": self.text_fusion_mode,
                "local_text_cross_attention": self.local_text_cross_attention,
            }
            self.root_backbone = backbone_cls(
                depth_double=int(root_depth_double),
                depth_single=int(root_depth_single),
                **actor_kwargs,
                **backbone_kwargs,
            )
            self.body_backbone = backbone_cls(
                depth_double=int(body_depth_double),
                depth_single=int(body_depth_single),
                **actor_kwargs,
                **backbone_kwargs,
            )
        else:
            if text_encoder not in {"llm2vec_cache", "kimodo_llm2vec_cache"}:
                raise ValueError(
                    "kimodo_prefix backbone requires text_encoder='llm2vec_cache'"
                )
            from .kimodo_prefix_transformer import KimodoPrefixTransformer

            backbone_kwargs = {
                "hidden_size": self.hidden_dim,
                "num_heads": int(num_heads),
                "mlp_ratio": float(mlp_ratio),
                "dropout": float(dropout),
                "text_dim": int(hytext_ctxt_dim),
            }
            self.root_backbone = KimodoPrefixTransformer(**backbone_kwargs)
            self.body_backbone = KimodoPrefixTransformer(**backbone_kwargs)
        self.root_conditioner = KimodoRootConditioner(
            motion_stats_dir=motion_stats_dir,
            local_root_stats_dir=local_root_stats_dir,
            fps=fps,
            variance_eps=stats_variance_eps,
        )

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def _compose_global_condition(
        self,
        timestep: torch.Tensor,
        direction: torch.Tensor,
        text_pooled: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep_token = self.timestep_embed(timestep.float()).to(dtype=dtype)
        direction_token = self.direction_embed(direction)
        if self.backbone_type == "kimodo_prefix":
            return torch.stack([timestep_token, direction_token], dim=1)
        cond = timestep_token + direction_token
        scale = 1.0 if self.text_global_conditioning == "pooled_adaln" else 0.0
        # Keep the disabled projection in the autograd graph. The fixed-bucket
        # DDP synchronizer requires a dense (here exactly zero) gradient for
        # every trainable parameter, while the forward value remains text-free.
        return cond + text_pooled * scale

    def _text_condition(
        self,
        texts: Optional[Iterable[str]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text_drop_prob: float,
        force_drop_text: bool,
    ) -> RawTextCondition:
        text_list = [""] * batch_size if texts is None else list(texts)
        if len(text_list) != batch_size:
            raise ValueError(f"Expected {batch_size} texts, got {len(text_list)}")
        return self.text_encoder(
            text_list,
            device=device,
            dtype=dtype,
            drop_prob=text_drop_prob,
            force_drop=force_drop_text,
        )

    def forward(
        self,
        model_in: torch.Tensor,
        t: torch.Tensor,
        c_dir: Optional[torch.Tensor] = None,
        text: Optional[Iterable[str]] = None,
        length_mask: Optional[torch.Tensor] = None,
        x_self_cond: Optional[torch.Tensor] = None,
        text_drop_prob: float = 0.0,
        force_drop_text: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | KimodoLikeFlowOutput:
        if model_in.ndim != 3 or model_in.shape[-1] != self.model_in_dim:
            raise ValueError(f"Expected model_in [B,T,{self.model_in_dim}], got {tuple(model_in.shape)}")
        bsz, frames, _ = model_in.shape
        device, dtype = model_in.device, model_in.dtype
        t = t.expand(bsz) if t.ndim == 0 else t
        t = t.to(device=device, dtype=dtype).view(bsz)
        if c_dir is None:
            c_dir = torch.zeros(bsz, 2, device=device, dtype=dtype)
            c_dir[:, 0] = 1.0
        else:
            c_dir = c_dir.to(device=device, dtype=dtype).view(bsz, 2)
        if length_mask is None:
            length_mask = torch.ones(bsz, frames, device=device, dtype=torch.bool)
        else:
            length_mask = length_mask.to(device=device, dtype=torch.bool)
        lengths = length_mask.sum(dim=-1).clamp_min(1)

        text_cond = self._text_condition(
            text,
            bsz,
            device,
            dtype,
            text_drop_prob=float(text_drop_prob),
            force_drop_text=bool(force_drop_text),
        )
        cond = self._compose_global_condition(
            t,
            c_dir,
            text_cond.pooled,
            dtype=dtype,
        )
        pos = torch.arange(frames, device=device, dtype=torch.long).view(1, frames, 1).expand(bsz, frames, 1)

        state = model_in[..., :DIM_HY273]
        motion_mask = model_in[..., DIM_HY273:]
        root_hidden = self.root_input_proj(model_in)
        if self.self_conditioning and x_self_cond is not None:
            root_hidden = root_hidden + self.self_cond_scale * self.root_self_cond_proj(
                x_self_cond[..., :ROOT_DIM].to(device=device, dtype=dtype)
            )
        root_hidden = self.root_backbone(
            motion=root_hidden,
            text=text_cond.tokens,
            cond=cond,
            motion_valid=length_mask,
            text_padding_mask=text_cond.padding_mask,
            motion_pos_ids=pos,
        )
        root_prediction_raw = self.root_output_proj(root_hidden)

        # Match Kimodo's two-stage contract: the body stage consumes one coherent
        # root trajectory predicted by the root denoiser. Mixing sparse imputed
        # keyframes into that trajectory would create finite-difference spikes.
        root_for_body = root_prediction_raw
        bridge_root = root_for_body.detach() if self.training and self.detach_root_bridge else root_for_body
        local_root = self.root_conditioner(bridge_root, lengths)

        body_in = torch.cat([local_root, state[..., ROOT_DIM:], motion_mask], dim=-1)
        body_hidden = self.body_input_proj(body_in)
        if self.self_conditioning and x_self_cond is not None:
            body_hidden = body_hidden + self.self_cond_scale * self.body_self_cond_proj(
                x_self_cond[..., ROOT_DIM:].to(device=device, dtype=dtype)
            )
        body_hidden = self.body_backbone(
            motion=body_hidden,
            text=text_cond.tokens,
            cond=cond,
            motion_valid=length_mask,
            text_padding_mask=text_cond.padding_mask,
            motion_pos_ids=pos,
        )
        body_prediction = self.body_output_proj(body_hidden)
        prediction = torch.cat([root_prediction_raw, body_prediction], dim=-1)
        if return_details:
            return KimodoLikeFlowOutput(
                prediction=prediction,
                root_prediction_raw=root_prediction_raw,
                root_for_body=root_for_body,
                local_root=local_root,
            )
        return prediction
