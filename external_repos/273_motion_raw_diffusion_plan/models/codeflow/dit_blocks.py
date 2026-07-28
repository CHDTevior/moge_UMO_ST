"""DiT/Flux-style transformer blocks for motion-code flow."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class AdaLNModulation(nn.Module):
    """AdaLN-Zero modulation: returns shift, scale and residual gate."""

    def __init__(self, hidden_size: int, num: int = 1) -> None:
        super().__init__()
        self.num = num
        self.linear = nn.Linear(hidden_size, hidden_size * 3 * num)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> List[ModulationOut]:
        chunks = self.linear(F.silu(cond)).chunk(self.num * 3, dim=-1)
        outs = []
        for idx in range(self.num):
            shift, scale, gate = chunks[idx * 3 : (idx + 1) * 3]
            outs.append(ModulationOut(shift[:, None], scale[:, None], gate[:, None]))
        return outs


def _rope_cos_sin(
    pos_ids: torch.Tensor,
    head_dim: int,
    axes_dims: List[int],
    theta: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if sum(axes_dims) != head_dim:
        raise ValueError(f"axes_dims sum {sum(axes_dims)} must equal head_dim {head_dim}")
    if pos_ids.shape[-1] != len(axes_dims):
        raise ValueError(f"pos_ids has {pos_ids.shape[-1]} axes, expected {len(axes_dims)}")
    cos_parts = []
    sin_parts = []
    for axis, axis_dim in enumerate(axes_dims):
        if axis_dim % 2 != 0:
            raise ValueError(f"RoPE axis dim must be even, got {axis_dim}")
        half = axis_dim // 2
        scale = torch.arange(0, half, dtype=torch.float32, device=pos_ids.device) / max(half, 1)
        omega = 1.0 / (theta ** scale)
        angles = pos_ids[..., axis].float()[..., None] * omega
        cos_parts.append(torch.cos(angles))
        sin_parts.append(torch.sin(angles))
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, N, D], cos/sin: [B, N, D/2]
    x_float = x.float()
    x_even = x_float[..., 0::2]
    x_odd = x_float[..., 1::2]
    cos = cos[:, None]
    sin = sin[:, None]
    out = torch.empty_like(x_float)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out.to(x.dtype)


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid: Optional[torch.Tensor],
    dropout_p: float,
    extra_key_count: int = 0,
    extra_attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # q/k/v: [B, H, N, D]
    attn_mask = None
    if key_valid is not None or (extra_key_count > 0 and extra_attn_bias is not None):
        key_len = k.shape[2]
        mask_value = -1.0e4 if q.dtype in (torch.float16, torch.bfloat16) else -1.0e9
        attn_mask = torch.zeros(
            q.shape[0], 1, 1, key_len,
            device=q.device,
            dtype=q.dtype,
        )
        if key_valid is not None:
            attn_mask = attn_mask.masked_fill(~key_valid[:, None, None], mask_value)
        if extra_key_count > 0 and extra_attn_bias is not None:
            attn_mask[..., -int(extra_key_count):] = (
                attn_mask[..., -int(extra_key_count):]
                + torch.as_tensor(extra_attn_bias, device=q.device, dtype=q.dtype)
            )

    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    probs = F.dropout(probs, p=dropout_p, training=dropout_p > 0)
    return torch.matmul(probs, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.kv = nn.Linear(hidden_size, hidden_size * 2)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        rope_axes_dims: Optional[List[int]] = None,
        extra_k: Optional[torch.Tensor] = None,
        extra_v: Optional[torch.Tensor] = None,
        extra_key_valid: Optional[torch.Tensor] = None,
        extra_pos: Optional[torch.Tensor] = None,
        extra_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = query.shape
        k_len = memory.shape[1]
        q = self.q(query).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(memory).chunk(2, dim=-1)
        k = k.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if query_pos is not None and memory_pos is not None and rope_axes_dims is not None:
            q_cos, q_sin = _rope_cos_sin(query_pos, self.head_dim, rope_axes_dims)
            k_cos, k_sin = _rope_cos_sin(memory_pos, self.head_dim, rope_axes_dims)
            q = _apply_rope(q, q_cos, q_sin)
            k = _apply_rope(k, k_cos, k_sin)

        extra_key_count = 0
        if extra_k is not None or extra_v is not None:
            if extra_k is None or extra_v is None:
                raise ValueError("extra_k and extra_v must be provided together")
            if extra_k.shape != extra_v.shape:
                raise ValueError(f"extra_k/extra_v shape mismatch: {tuple(extra_k.shape)} vs {tuple(extra_v.shape)}")
            if extra_k.ndim != 3 or extra_k.shape[0] != bsz or extra_k.shape[2] != self.hidden_size:
                raise ValueError(
                    f"Expected extra_k/extra_v [B,T,{self.hidden_size}], got {tuple(extra_k.shape)}"
                )
            extra_key_count = int(extra_k.shape[1])
            extra_k_heads = extra_k.view(bsz, extra_key_count, self.num_heads, self.head_dim).transpose(1, 2)
            extra_v_heads = extra_v.view(bsz, extra_key_count, self.num_heads, self.head_dim).transpose(1, 2)
            extra_k_heads = self.k_norm(extra_k_heads)
            if extra_pos is not None and rope_axes_dims is not None:
                extra_cos, extra_sin = _rope_cos_sin(extra_pos, self.head_dim, rope_axes_dims)
                extra_k_heads = _apply_rope(extra_k_heads, extra_cos, extra_sin)
            k = torch.cat([k, extra_k_heads], dim=2)
            v = torch.cat([v, extra_v_heads], dim=2)
            if extra_key_valid is None:
                extra_key_valid = torch.ones(bsz, extra_key_count, device=query.device, dtype=torch.bool)
            else:
                extra_key_valid = extra_key_valid.to(device=query.device, dtype=torch.bool)
                if extra_key_valid.shape != (bsz, extra_key_count):
                    raise ValueError(
                        f"Expected extra_key_valid shape {(bsz, extra_key_count)}, got {tuple(extra_key_valid.shape)}"
                    )
            if key_valid is None:
                key_valid = torch.ones(bsz, k_len, device=query.device, dtype=torch.bool)
            key_valid = torch.cat([key_valid.to(device=query.device, dtype=torch.bool), extra_key_valid], dim=1)

        out = _attention(
            q, k, v,
            key_valid=key_valid,
            dropout_p=self.dropout if self.training else 0.0,
            extra_key_count=extra_key_count,
            extra_attn_bias=extra_attn_bias,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.out(out)
        if query_valid is not None:
            out = out * query_valid[:, :, None].to(out.dtype)
        return out


class DoubleStreamBlock(nn.Module):
    """Joint text-motion attention with separate stream updates."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.motion_mod = AdaLNModulation(hidden_size, num=2)
        self.text_mod = AdaLNModulation(hidden_size, num=2)
        self.motion_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.text_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.joint_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.motion_ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.text_ffn_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.motion_ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))
        self.text_ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        cond: torch.Tensor,
        motion_valid: torch.Tensor,
        text_valid: torch.Tensor,
        pos_ids: torch.Tensor,
        rope_axes_dims: List[int],
        control_k: Optional[torch.Tensor] = None,
        control_v: Optional[torch.Tensor] = None,
        control_valid: Optional[torch.Tensor] = None,
        control_pos: Optional[torch.Tensor] = None,
        control_attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m_attn, m_ff = self.motion_mod(cond)
        t_attn, t_ff = self.text_mod(cond)

        motion_in = self.motion_norm(motion)
        motion_in = (1 + m_attn.scale) * motion_in + m_attn.shift
        text_in = self.text_norm(text)
        text_in = (1 + t_attn.scale) * text_in + t_attn.shift

        joint = torch.cat([motion_in, text_in], dim=1)
        joint_valid = torch.cat([motion_valid, text_valid], dim=1)
        text_pos = torch.zeros(
            text.shape[0], text.shape[1], pos_ids.shape[-1],
            device=pos_ids.device,
            dtype=pos_ids.dtype,
        )
        joint_pos = torch.cat([pos_ids, text_pos], dim=1)
        joint_out = self.joint_attn(
            joint,
            joint,
            key_valid=joint_valid,
            query_valid=joint_valid,
            query_pos=joint_pos,
            memory_pos=joint_pos,
            rope_axes_dims=rope_axes_dims,
            extra_k=control_k,
            extra_v=control_v,
            extra_key_valid=control_valid,
            extra_pos=control_pos,
            extra_attn_bias=control_attn_bias,
        )
        motion_out, text_out = joint_out.split([motion.shape[1], text.shape[1]], dim=1)
        motion = motion + m_attn.gate * motion_out
        text = text + t_attn.gate * text_out

        motion_ff = self.motion_ffn_norm(motion)
        motion_ff = (1 + m_ff.scale) * motion_ff + m_ff.shift
        motion = motion + m_ff.gate * self.motion_ffn(motion_ff)

        text_ff = self.text_ffn_norm(text)
        text_ff = (1 + t_ff.scale) * text_ff + t_ff.shift
        text = text + t_ff.gate * self.text_ffn(text_ff)
        return motion, text


class SingleStreamBlock(nn.Module):
    """Single-stream DiT block over concatenated motion and text tokens."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mod = AdaLNModulation(hidden_size, num=2)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        valid: torch.Tensor,
        pos_ids: torch.Tensor,
        rope_axes_dims: List[int],
        control_k: Optional[torch.Tensor] = None,
        control_v: Optional[torch.Tensor] = None,
        control_valid: Optional[torch.Tensor] = None,
        control_pos: Optional[torch.Tensor] = None,
        control_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_mod, ffn_mod = self.mod(cond)
        h = self.norm1(x)
        h = (1 + attn_mod.scale) * h + attn_mod.shift
        x = x + attn_mod.gate * self.attn(
            h,
            h,
            key_valid=valid,
            query_valid=valid,
            query_pos=pos_ids,
            memory_pos=pos_ids,
            rope_axes_dims=rope_axes_dims,
            extra_k=control_k,
            extra_v=control_v,
            extra_key_valid=control_valid,
            extra_pos=control_pos,
            extra_attn_bias=control_attn_bias,
        )
        h = self.norm2(x)
        h = (1 + ffn_mod.scale) * h + ffn_mod.shift
        x = x + ffn_mod.gate * self.ffn(h)
        return x


class FrameHolderCouplingBlock(nn.Module):
    """Per-frame holder-query coupling over the fixed body-part token slots."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float, num_parts: int) -> None:
        super().__init__()
        if num_parts <= 0:
            raise ValueError(f"num_parts must be positive, got {num_parts}")
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num_parts = int(num_parts)
        self.holder = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.normal_(self.holder, std=0.02)
        self.mod = AdaLNModulation(hidden_size, num=2)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(SwiGLU(hidden_size, mlp_hidden), nn.Dropout(dropout))

    def forward(self, motion: torch.Tensor, cond: torch.Tensor, motion_valid: torch.Tensor) -> torch.Tensor:
        bsz, token_count, hidden_size = motion.shape
        if token_count % self.num_parts != 0:
            raise RuntimeError(
                f"Motion token count {token_count} is not divisible by num_parts={self.num_parts}"
            )
        frame_count = token_count // self.num_parts
        parts = motion.reshape(bsz, frame_count, self.num_parts, hidden_size).reshape(
            bsz * frame_count,
            self.num_parts,
            hidden_size,
        )
        part_valid = motion_valid.reshape(bsz, frame_count, self.num_parts).reshape(
            bsz * frame_count,
            self.num_parts,
        )
        holder = self.holder.to(device=motion.device, dtype=motion.dtype).expand(bsz * frame_count, 1, hidden_size)
        seq = torch.cat([holder, parts], dim=1)

        holder_valid = torch.ones(part_valid.shape[0], 1, device=part_valid.device, dtype=torch.bool)
        valid = torch.cat([holder_valid, part_valid], dim=1)
        cond_frame = cond[:, None, :].expand(bsz, frame_count, hidden_size).reshape(bsz * frame_count, hidden_size)

        attn_mod, ffn_mod = self.mod(cond_frame)
        h = self.norm1(seq)
        h = (1 + attn_mod.scale) * h + attn_mod.shift
        seq = seq + attn_mod.gate * self.attn(h, h, key_valid=valid, query_valid=valid)

        h = self.norm2(seq)
        h = (1 + ffn_mod.scale) * h + ffn_mod.shift
        seq = seq + ffn_mod.gate * self.ffn(h)

        parts = seq[:, 1:].reshape(bsz, frame_count, self.num_parts, hidden_size).reshape(
            bsz,
            token_count,
            hidden_size,
        )
        return parts * motion_valid[:, :, None].to(parts.dtype)


class FrameHolderOutput(nn.Module):
    """Final holder-query head that emits all part latents for each frame."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        num_parts: int,
        output_size: int,
        holder_depth: int,
        holder_mlp_ratio: float,
    ) -> None:
        super().__init__()
        if holder_depth <= 0:
            raise ValueError(f"holder_depth must be positive, got {holder_depth}")
        self.num_parts = int(num_parts)
        self.output_size = int(output_size)
        self.holder = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.normal_(self.holder, std=0.02)
        holder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * holder_mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.mixer = nn.TransformerEncoder(holder_layer, num_layers=holder_depth)
        self.linear = FinalLayer(hidden_size, num_parts * output_size)

    def forward(self, motion: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        bsz, token_count, hidden_size = motion.shape
        if token_count % self.num_parts != 0:
            raise RuntimeError(
                f"Motion token count {token_count} is not divisible by num_parts={self.num_parts}"
            )
        frame_count = token_count // self.num_parts
        parts = motion.reshape(bsz, frame_count, self.num_parts, hidden_size)
        holder = self.holder.to(device=motion.device, dtype=motion.dtype).expand(bsz, frame_count, 1, hidden_size)
        seq = torch.cat([holder, parts], dim=2).reshape(
            bsz * frame_count,
            1 + self.num_parts,
            hidden_size,
        )
        seq = self.mixer(seq)
        holder_out = seq[:, :1]
        cond_frame = cond[:, None, :].expand(bsz, frame_count, hidden_size).reshape(bsz * frame_count, hidden_size)
        out = self.linear(holder_out, cond_frame)
        return out.reshape(bsz, frame_count, self.num_parts, self.output_size)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = (1 + scale[:, None]) * self.norm(x) + shift[:, None]
        return self.linear(x)


class MotionTextDiT(nn.Module):
    """A full text-motion DiT backbone with double and single stream blocks."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth_double: int,
        depth_single: int,
        output_size: int,
        num_parts: int = 6,
        holder_depth: int = 2,
        holder_mlp_ratio: float = 4.0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope_axes_dims: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if depth_double <= 0 or depth_single <= 0:
            raise ValueError("Both double-stream and single-stream depths must be positive")
        head_dim = hidden_size // num_heads
        if rope_axes_dims is None:
            rope_axes_dims = [head_dim // 2, head_dim - head_dim // 2]
        if sum(rope_axes_dims) != head_dim:
            raise ValueError(f"rope_axes_dims must sum to head_dim {head_dim}")
        self.num_parts = int(num_parts)
        self.rope_axes_dims = rope_axes_dims
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth_double)
        ])
        self.double_holder_couplings = nn.ModuleList([
            FrameHolderCouplingBlock(hidden_size, num_heads, mlp_ratio, dropout, num_parts=self.num_parts)
            for _ in range(depth_double)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth_single)
        ])
        self.single_holder_couplings = nn.ModuleList([
            FrameHolderCouplingBlock(hidden_size, num_heads, mlp_ratio, dropout, num_parts=self.num_parts)
            for _ in range(depth_single)
        ])
        self.holder_output = FrameHolderOutput(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            num_parts=self.num_parts,
            output_size=output_size,
            holder_depth=holder_depth,
            holder_mlp_ratio=holder_mlp_ratio,
        )

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        cond: torch.Tensor,
        motion_valid: torch.Tensor,
        text_padding_mask: torch.Tensor,
        motion_pos_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        text_valid = ~text_padding_mask
        for block, holder_coupling in zip(self.double_blocks, self.double_holder_couplings):
            motion, text = block(
                motion,
                text,
                cond,
                motion_valid=motion_valid,
                text_valid=text_valid,
                pos_ids=motion_pos_ids,
                rope_axes_dims=self.rope_axes_dims,
            )
            motion = holder_coupling(motion, cond, motion_valid)

        text_pos = torch.zeros(
            text.shape[0], text.shape[1], motion_pos_ids.shape[-1],
            device=motion_pos_ids.device,
            dtype=motion_pos_ids.dtype,
        )
        x = torch.cat([motion, text], dim=1)
        valid = torch.cat([motion_valid, text_valid], dim=1)
        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
        motion_token_count = motion.shape[1]
        for block, holder_coupling in zip(self.single_blocks, self.single_holder_couplings):
            x = block(x, cond, valid=valid, pos_ids=pos_ids, rope_axes_dims=self.rope_axes_dims)
            motion_x = holder_coupling(x[:, :motion_token_count], cond, motion_valid)
            x = torch.cat([motion_x, x[:, motion_token_count:]], dim=1)

        motion = x[:, :motion_token_count]
        if return_hidden:
            return motion
        return self.holder_output(motion, cond)


class FrameMotionTextDiT(nn.Module):
    """Text-conditioned DiT over one structured motion token per frame."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        depth_double: int,
        depth_single: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        rope_axes_dims: Optional[List[int]] = None,
        control_input_dim: int = 0,
        control_rank: int = 64,
        control_encoder_width: int = 512,
        control_attn_bias_init: float = -5.0,
    ) -> None:
        super().__init__()
        if depth_double <= 0 or depth_single <= 0:
            raise ValueError("Both double-stream and single-stream depths must be positive")
        head_dim = hidden_size // num_heads
        if rope_axes_dims is None:
            rope_axes_dims = [head_dim]
        if sum(rope_axes_dims) != head_dim:
            raise ValueError(f"rope_axes_dims must sum to head_dim {head_dim}")
        self.rope_axes_dims = rope_axes_dims
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth_double)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth_single)
        ])
        self.control_input_dim = int(control_input_dim)
        self.control_layer_count = int(depth_double) + int(depth_single)
        if self.control_input_dim > 0:
            if int(control_rank) <= 0:
                raise ValueError(f"control_rank must be positive, got {control_rank}")
            if int(control_encoder_width) <= 0:
                raise ValueError(f"control_encoder_width must be positive, got {control_encoder_width}")
            self.control_encoder = nn.Sequential(
                nn.Conv1d(self.control_input_dim, int(control_encoder_width), kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(int(control_encoder_width), hidden_size, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            )
            self.control_kv_down = nn.ModuleList([
                nn.Linear(hidden_size, int(control_rank), bias=False)
                for _ in range(self.control_layer_count)
            ])
            self.control_kv_up_k = nn.ModuleList([
                nn.Linear(int(control_rank), hidden_size, bias=False)
                for _ in range(self.control_layer_count)
            ])
            self.control_kv_up_v = nn.ModuleList([
                nn.Linear(int(control_rank), hidden_size, bias=False)
                for _ in range(self.control_layer_count)
            ])
            self.control_attn_bias = nn.ParameterList([
                nn.Parameter(torch.full((), float(control_attn_bias_init)))
                for _ in range(self.control_layer_count)
            ])
            for down in self.control_kv_down:
                nn.init.zeros_(down.weight)
        else:
            self.control_encoder = None
            self.control_kv_down = nn.ModuleList()
            self.control_kv_up_k = nn.ModuleList()
            self.control_kv_up_v = nn.ModuleList()
            self.control_attn_bias = nn.ParameterList()

    def _encode_control(
        self,
        control_cond: Optional[torch.Tensor],
        motion: torch.Tensor,
        motion_valid: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if control_cond is None:
            return None
        if self.control_encoder is None:
            raise RuntimeError("control_cond was provided but this FrameMotionTextDiT has no control adapter")
        if control_cond.ndim != 3 or control_cond.shape[0] != motion.shape[0] or control_cond.shape[-1] != self.control_input_dim:
            raise ValueError(
                f"Expected control_cond [B,F,{self.control_input_dim}], got {tuple(control_cond.shape)}"
            )
        control = control_cond.to(device=motion.device, dtype=motion.dtype).transpose(1, 2).contiguous()
        control_tokens = self.control_encoder(control).transpose(1, 2).contiguous()
        if control_tokens.shape[1] != motion.shape[1]:
            raise ValueError(
                "Control encoder output length must match motion token length: "
                f"control={control_tokens.shape[1]} motion={motion.shape[1]} "
                f"from control frame length={control_cond.shape[1]}"
            )
        return control_tokens * motion_valid[:, :, None].to(control_tokens.dtype)

    def _control_kv(self, control_tokens: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.control_kv_down[layer_idx](control_tokens)
        return self.control_kv_up_k[layer_idx](hidden), self.control_kv_up_v[layer_idx](hidden)

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        cond: torch.Tensor,
        motion_valid: torch.Tensor,
        text_padding_mask: torch.Tensor,
        motion_pos_ids: torch.Tensor,
        control_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_valid = ~text_padding_mask
        control_tokens = self._encode_control(control_cond, motion, motion_valid)
        layer_idx = 0
        for block in self.double_blocks:
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
            layer_idx += 1

        text_pos = torch.zeros(
            text.shape[0], text.shape[1], motion_pos_ids.shape[-1],
            device=motion_pos_ids.device,
            dtype=motion_pos_ids.dtype,
        )
        x = torch.cat([motion, text], dim=1)
        valid = torch.cat([motion_valid, text_valid], dim=1)
        pos_ids = torch.cat([motion_pos_ids, text_pos], dim=1)
        for block in self.single_blocks:
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
            )
            layer_idx += 1
        return x[:, : motion.shape[1]]
