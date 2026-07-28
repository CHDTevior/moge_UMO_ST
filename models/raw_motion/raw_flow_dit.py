"""Frame-level HY273 raw-space rectified-flow denoiser."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn

from .hy273_slices import DIM_HY273
from .text_condition import RawTextCondition


def _load_dit_blocks():
    path = Path(__file__).resolve().parents[1] / "codeflow" / "dit_blocks.py"
    spec = importlib.util.spec_from_file_location("_hy273_codeflow_dit_blocks", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DiT blocks from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dit_blocks = _load_dit_blocks()
FrameMotionTextDiT = _dit_blocks.FrameMotionTextDiT
TimestepEmbedder = _dit_blocks.TimestepEmbedder


class NullTextEncoder(nn.Module):
    def __init__(self, hidden_dim: int, max_text_tokens: int = 50) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_text_tokens = int(max_text_tokens)

    def forward(
        self,
        texts: Iterable[str],
        device: torch.device,
        dtype: torch.dtype,
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> RawTextCondition:
        texts = list(texts)
        bsz = len(texts)
        tokens = torch.zeros(bsz, self.max_text_tokens, self.hidden_dim, device=device, dtype=dtype)
        pooled = torch.zeros(bsz, self.hidden_dim, device=device, dtype=dtype)
        padding = torch.ones(bsz, self.max_text_tokens, device=device, dtype=torch.bool)
        padding[:, 0] = False
        return RawTextCondition(tokens=tokens, pooled=pooled, padding_mask=padding)


class FrozenCLIPTextEncoder(nn.Module):
    """Local CLIP text tower wrapper that avoids importing models.codeflow package init."""

    def __init__(
        self,
        hidden_dim: int,
        clip_path: str = "",
        clip_version: str = "ViT-B/32",
        max_text_tokens: int = 50,
    ) -> None:
        super().__init__()
        import clip

        resolved = clip_path if clip_path else clip_version
        model, _ = clip.load(resolved, device="cpu", jit=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.clip = clip
        self.clip_model = model
        self.max_text_tokens = int(max_text_tokens)
        width = int(model.ln_final.weight.shape[0])
        output_dim = int(model.text_projection.shape[1])
        self.token_proj = nn.Linear(width, hidden_dim)
        self.pooled_proj = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def device(self) -> torch.device:
        return next(self.clip_model.parameters()).device

    @torch.no_grad()
    def _encode_clip(
        self,
        texts: list[str],
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if force_drop:
            texts = [""] * len(texts)
        elif drop_prob > 0.0:
            keep = torch.rand(len(texts), device=self.device) >= float(drop_prob)
            texts = [text if bool(keep[i].item()) else "" for i, text in enumerate(texts)]
        text_tokens = self.clip.tokenize(texts, truncate=True).to(self.device)
        cm = self.clip_model
        x = cm.token_embedding(text_tokens).type(cm.dtype)
        x = x + cm.positional_embedding.type(cm.dtype)
        x = x.permute(1, 0, 2)
        x = cm.transformer(x)
        x = x.permute(1, 0, 2)
        x = cm.ln_final(x).type(cm.dtype)
        pooled = x[torch.arange(x.shape[0], device=x.device), text_tokens.argmax(dim=-1)] @ cm.text_projection
        padding = text_tokens == 0
        return x.float(), pooled.float(), padding

    def forward(
        self,
        texts: Iterable[str],
        device: torch.device,
        dtype: torch.dtype,
        drop_prob: float = 0.0,
        force_drop: bool = False,
    ) -> RawTextCondition:
        text_list = [str(t) for t in texts]
        tokens, pooled, padding = self._encode_clip(text_list, drop_prob=drop_prob, force_drop=force_drop)
        max_tokens = self.max_text_tokens
        tokens = tokens[:, :max_tokens]
        padding = padding[:, :max_tokens]
        if tokens.shape[1] < max_tokens:
            pad_len = max_tokens - tokens.shape[1]
            tokens = torch.cat([tokens, tokens.new_zeros(tokens.shape[0], pad_len, tokens.shape[2])], dim=1)
            padding = torch.cat([padding, torch.ones(padding.shape[0], pad_len, device=padding.device, dtype=torch.bool)], dim=1)
        tokens = self.token_proj(tokens.to(device=device, dtype=dtype))
        pooled = self.pooled_proj(pooled.to(device=device, dtype=dtype))
        padding = padding.to(device=device)
        if padding.all(dim=1).any():
            padding = padding.clone()
            padding[padding.all(dim=1), 0] = False
        return RawTextCondition(tokens=tokens, pooled=pooled, padding_mask=padding)


class HY273RawFlow(nn.Module):
    def __init__(
        self,
        source_dim: int = DIM_HY273,
        mask_dim: int = DIM_HY273,
        hidden_dim: int = 1024,
        output_dim: int = DIM_HY273,
        num_heads: int = 8,
        depth_double: int = 4,
        depth_single: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_text_tokens: int = 50,
        text_encoder: str = "clip",
        clip_path: str = "checkpoints/clip/ViT-B-32.pt",
        clip_version: str = "ViT-B/32",
        hytext_cache_dir: str = "",
        hytext_ctxt_dim: int = 4096,
        hytext_vtxt_dim: int = 768,
        hytext_max_open_shards: int = 8,
        hytext_strict_cache: bool = True,
        self_conditioning: bool = False,
        self_cond_mode: str = "add_proj",
        self_cond_scale: float = 1.0,
        zero_init_self_cond: bool = True,
    ) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.mask_dim = int(mask_dim)
        self.model_in_dim = self.source_dim + self.mask_dim
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.self_conditioning = bool(self_conditioning)
        self.self_cond_mode = str(self_cond_mode)
        self.self_cond_scale = float(self_cond_scale)

        if self.self_cond_mode not in {"add_proj", "concat"}:
            raise ValueError(f"Unknown self_cond_mode: {self.self_cond_mode}")
        input_dim = self.model_in_dim
        if self.self_cond_mode == "concat" and self.self_conditioning:
            input_dim += self.source_dim
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        self.self_cond_proj: Optional[nn.Linear]
        if self.self_cond_mode == "add_proj":
            self.self_cond_proj = nn.Linear(self.source_dim, self.hidden_dim)
            if zero_init_self_cond:
                nn.init.zeros_(self.self_cond_proj.weight)
                nn.init.zeros_(self.self_cond_proj.bias)
            if not self.self_conditioning:
                for param in self.self_cond_proj.parameters():
                    param.requires_grad_(False)
        else:
            self.self_cond_proj = None

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
        elif text_encoder in {"none", "null"}:
            self.text_encoder = NullTextEncoder(self.hidden_dim, max_text_tokens=max_text_tokens)
        else:
            raise ValueError(f"Unsupported text_encoder: {text_encoder}")
        self.backbone = FrameMotionTextDiT(
            hidden_size=self.hidden_dim,
            num_heads=int(num_heads),
            depth_double=int(depth_double),
            depth_single=int(depth_single),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)

    def trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def _text_condition(
        self,
        texts: Optional[Iterable[str]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text_drop_prob: float = 0.0,
        force_drop_text: bool = False,
    ) -> RawTextCondition:
        if texts is None:
            texts = [""] * batch_size
        text_list = list(texts)
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
    ) -> torch.Tensor:
        if model_in.ndim != 3 or model_in.shape[-1] != self.model_in_dim:
            raise ValueError(f"Expected model_in [B,T,{self.model_in_dim}], got {tuple(model_in.shape)}")
        bsz, frames, _ = model_in.shape
        device = model_in.device
        dtype = model_in.dtype
        if t.ndim == 0:
            t = t.expand(bsz)
        t = t.to(device=device, dtype=dtype).view(bsz)
        if c_dir is None:
            c_dir = torch.zeros(bsz, 2, device=device, dtype=dtype)
            c_dir[:, 0] = 1.0
        c_dir = c_dir.to(device=device, dtype=dtype).view(bsz, 2)
        if length_mask is None:
            length_mask = torch.ones(bsz, frames, device=device, dtype=torch.bool)
        else:
            length_mask = length_mask.to(device=device, dtype=torch.bool)

        if self.self_cond_mode == "concat":
            if self.self_conditioning:
                if x_self_cond is None:
                    x_self_cond = torch.zeros(bsz, frames, self.source_dim, device=device, dtype=dtype)
                model_in = torch.cat([model_in, x_self_cond.to(device=device, dtype=dtype)], dim=-1)
            motion = self.input_proj(model_in)
        else:
            motion = self.input_proj(model_in)
            if self.self_conditioning:
                if x_self_cond is None:
                    x_self_cond = torch.zeros(bsz, frames, self.source_dim, device=device, dtype=dtype)
                motion = motion + self.self_cond_scale * self.self_cond_proj(x_self_cond.to(device=device, dtype=dtype))

        text_cond = self._text_condition(
            text,
            bsz,
            device,
            dtype,
            text_drop_prob=text_drop_prob,
            force_drop_text=force_drop_text,
        )
        cond = self.timestep_embed(t.float()).to(dtype=dtype) + self.direction_embed(c_dir) + text_cond.pooled
        pos = torch.arange(frames, device=device, dtype=torch.long).view(1, frames, 1).expand(bsz, frames, 1)
        hidden = self.backbone(
            motion=motion,
            text=text_cond.tokens,
            cond=cond,
            motion_valid=length_mask,
            text_padding_mask=text_cond.padding_mask,
            motion_pos_ids=pos,
        )
        return self.output_proj(hidden)
