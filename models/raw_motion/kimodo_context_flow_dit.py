"""Kimodo-like root/body denoiser with a typed, noise-free source context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn

from .hy273_ease import HY273EaseConditioner
from .hy273_multitask_condition import (
    NUM_SOURCE_ROLES,
    NUM_TARGET_OPS,
    NUM_TASKS,
    ConditionBatch,
    TaskId,
)
from .hy273_slices import (
    BODY_DIM,
    CONTACT_SLICE,
    DIM_HY273,
    LOCAL_ROOT_DIM,
    ROOT_DIM,
)
from .kimodo_like_flow_dit import HY273RedenoiseKimodoLike
from .text_condition import RawTextCondition


def build_source_token_block(
    target_hidden: torch.Tensor,
    source_hidden: torch.Tensor,
    target_valid: torch.Tensor,
    context_present: torch.Tensor,
    target_pos_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, slice]:
    """Build ``[source, SEP, target]`` without adding source into target tokens.

    Source and target frame tokens share temporal RoPE coordinates so aligned
    frames remain easy to associate. The zero separator has a unique coordinate
    and acquires context through the existing transformer; it adds no parameters
    when a trained additive checkpoint is migrated to this research treatment.
    """

    if target_hidden.ndim != 3 or source_hidden.shape != target_hidden.shape:
        raise ValueError(
            "target_hidden/source_hidden must share shape [B,T,H], got "
            f"{tuple(target_hidden.shape)} and {tuple(source_hidden.shape)}"
        )
    bsz, frames, hidden = target_hidden.shape
    if target_valid.shape != (bsz, frames):
        raise ValueError("target_valid must be [B,T]")
    if context_present.shape != (bsz,):
        raise ValueError("context_present must be [B]")
    if target_pos_ids.ndim != 3 or target_pos_ids.shape[:2] != (bsz, frames):
        raise ValueError("target_pos_ids must be [B,T,A]")

    source_valid = context_present[:, None] & target_valid
    separator = target_hidden.new_zeros((bsz, 1, hidden))
    separator_valid = context_present[:, None]
    separator_pos = target_pos_ids.new_zeros(
        (bsz, 1, target_pos_ids.shape[-1])
    )
    separator_pos[:, 0, 0] = target_valid.sum(dim=-1).to(
        dtype=target_pos_ids.dtype
    )
    tokens = torch.cat([source_hidden, separator, target_hidden], dim=1)
    valid = torch.cat([source_valid, separator_valid, target_valid], dim=1)
    pos_ids = torch.cat([target_pos_ids, separator_pos, target_pos_ids], dim=1)
    target_slice = slice(frames + 1, 2 * frames + 1)
    return tokens, valid, pos_ids, target_slice


def _load_motion_stats(
    stats_dir: str | Path, *, normalize_contacts: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    root = Path(stats_dir).expanduser().resolve()
    mean_path = root / "Mean.npy"
    std_path = root / "Std.npy"
    if not mean_path.is_file() or not std_path.is_file():
        raise FileNotFoundError(f"Missing HY273 Mean.npy/Std.npy under {root}")
    mean = torch.from_numpy(np.load(mean_path)).float().reshape(DIM_HY273)
    std = torch.from_numpy(np.load(std_path)).float().reshape(DIM_HY273)
    if not normalize_contacts:
        mean[CONTACT_SLICE] = 0.0
        std[CONTACT_SLICE] = 1.0
    return mean, std


def align_projected_source(
    source_hidden: torch.Tensor,
    source_valid: torch.Tensor,
    source_lengths: torch.Tensor,
    target_valid: torch.Tensor,
    target_lengths: torch.Tensor,
    target_to_source_time_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align projected source tokens to target queries without resampling K273.

    Invalid neighbors are excluded and the remaining interpolation weights are
    renormalized. Equal native lengths use a strict identity path.
    """

    if source_hidden.ndim != 4:
        raise ValueError(f"Expected source_hidden [B,K,Ts,H], got {source_hidden.shape}")
    bsz, slots, source_frames, hidden = source_hidden.shape
    target_frames = target_valid.shape[1]
    if source_valid.shape != (bsz, slots, source_frames):
        raise ValueError("source_valid shape does not match source_hidden")
    if source_lengths.shape != (bsz, slots):
        raise ValueError("source_lengths shape does not match source_hidden")
    if target_lengths.shape != (bsz,):
        raise ValueError("target_lengths must be [B]")
    if target_to_source_time_map is not None and target_to_source_time_map.shape != (
        bsz,
        slots,
        target_frames,
    ):
        raise ValueError("target_to_source_time_map must be [B,K,Tt]")

    aligned = source_hidden.new_zeros((bsz, slots, target_frames, hidden))
    aligned_valid = torch.zeros(
        bsz, slots, target_frames, device=source_hidden.device, dtype=torch.bool
    )
    clean_source = torch.where(
        source_valid[..., None], source_hidden, torch.zeros_like(source_hidden)
    )
    for batch_idx in range(bsz):
        target_len = int(target_lengths[batch_idx].item())
        if target_len <= 0:
            continue
        for slot_idx in range(slots):
            source_len = int(source_lengths[batch_idx, slot_idx].item())
            if source_len <= 0:
                continue
            if target_to_source_time_map is None and source_len == target_len:
                aligned[batch_idx, slot_idx, :target_len] = clean_source[
                    batch_idx, slot_idx, :source_len
                ]
                aligned_valid[batch_idx, slot_idx, :target_len] = source_valid[
                    batch_idx, slot_idx, :source_len
                ]
                continue

            if target_to_source_time_map is not None:
                coord = target_to_source_time_map[
                    batch_idx, slot_idx, :target_len
                ].to(device=source_hidden.device, dtype=torch.float32)
            elif source_len <= 1 or target_len <= 1:
                coord = torch.zeros(target_len, device=source_hidden.device)
            else:
                coord = torch.linspace(
                    0.0,
                    float(source_len - 1),
                    target_len,
                    device=source_hidden.device,
                    dtype=torch.float32,
                )
            coord = coord.clamp(0.0, float(max(source_len - 1, 0)))
            lower = coord.floor().long()
            upper = coord.ceil().long()
            alpha = (coord - lower.float()).to(dtype=source_hidden.dtype)
            lower_valid = source_valid[batch_idx, slot_idx, lower]
            upper_valid = source_valid[batch_idx, slot_idx, upper]
            lower_weight = (1.0 - alpha) * lower_valid.to(alpha.dtype)
            upper_weight = alpha * upper_valid.to(alpha.dtype)
            weight_sum = lower_weight + upper_weight
            valid = weight_sum > 0
            values = (
                clean_source[batch_idx, slot_idx, lower] * lower_weight[:, None]
                + clean_source[batch_idx, slot_idx, upper] * upper_weight[:, None]
            )
            values = values / weight_sum.clamp_min(1e-12)[:, None]
            aligned[batch_idx, slot_idx, :target_len] = torch.where(
                valid[:, None], values, torch.zeros_like(values)
            )
            aligned_valid[batch_idx, slot_idx, :target_len] = valid

    final_valid = aligned_valid & target_valid[:, None, :]
    aligned = torch.where(final_valid[..., None], aligned, torch.zeros_like(aligned))
    return aligned, final_valid


@dataclass
class SourceContextOutput:
    root: torch.Tensor
    body: torch.Tensor
    aligned_valid: torch.Tensor
    context_present: torch.Tensor


class HY273SourceContext(nn.Module):
    """Project, align, and aggregate K273 source slots in hidden space."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        motion_stats_dir: str | Path,
        max_frames: int = 300,
        variance_eps: float = 1e-5,
        normalize_contacts: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_frames = int(max_frames)
        self.variance_eps = float(variance_eps)
        self.normalize_contacts = bool(normalize_contacts)
        mean, std = _load_motion_stats(
            motion_stats_dir, normalize_contacts=self.normalize_contacts
        )
        self.register_buffer("source_mean", mean.view(1, 1, 1, DIM_HY273))
        self.register_buffer("source_std", std.view(1, 1, 1, DIM_HY273))

        self.root_source_proj = nn.Linear(DIM_HY273 * 2, self.hidden_dim)
        self.body_source_proj = nn.Linear(DIM_HY273 * 2, self.hidden_dim)
        self.task_embed = nn.Embedding(NUM_TASKS, self.hidden_dim)
        self.op_embed = nn.Embedding(NUM_TARGET_OPS, self.hidden_dim)
        self.role_embed = nn.Embedding(NUM_SOURCE_ROLES, self.hidden_dim)
        self.length_proj = nn.Linear(3, self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Exact-zero output keeps the no-source path identical and lets the first
        # source-present batch directly train every context parameter.
        for module in (
            self.root_source_proj,
            self.body_source_proj,
            self.length_proj,
        ):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.zeros_(self.task_embed.weight)
        nn.init.zeros_(self.op_embed.weight)
        nn.init.zeros_(self.role_embed.weight)

    def weight_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.root_source_proj.weight,
            self.body_source_proj.weight,
            self.length_proj.weight,
        )

    def bias_and_embedding_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.root_source_proj.bias,
            self.body_source_proj.bias,
            self.length_proj.bias,
            self.task_embed.weight,
            self.op_embed.weight,
            self.role_embed.weight,
        )

    def _normalize_source(self, source: torch.Tensor) -> torch.Tensor:
        mean = self.source_mean.to(device=source.device, dtype=source.dtype)
        std = self.source_std.to(device=source.device, dtype=source.dtype)
        if self.variance_eps > 0:
            std = torch.sqrt(std.square() + self.variance_eps)
        normalized = (source - mean) / std
        if not self.normalize_contacts:
            normalized[..., CONTACT_SLICE] = source[..., CONTACT_SLICE]
        return normalized

    def forward(
        self,
        condition: ConditionBatch,
        *,
        target_frames: int,
        dtype: torch.dtype,
    ) -> SourceContextOutput:
        if condition.target_frames != int(target_frames):
            raise ValueError(
                f"Condition Tt={condition.target_frames} differs from model T={target_frames}"
            )
        source = condition.source_motion.to(device=self.source_mean.device, dtype=torch.float32)
        value_mask = condition.source_value_mask.to(device=source.device)
        time_valid = condition.source_time_valid.to(device=source.device)
        present = condition.source_present.to(device=source.device)
        source_lengths = condition.source_native_lengths.to(device=source.device)
        target_valid = condition.target_valid.to(device=source.device)
        target_lengths = condition.requested_target_len.to(device=source.device)
        time_map = (
            None
            if condition.target_to_source_time_map is None
            else condition.target_to_source_time_map.to(device=source.device)
        )

        source_norm = self._normalize_source(source)
        source_norm = torch.where(value_mask, source_norm, torch.zeros_like(source_norm))
        source_input = torch.cat([source_norm, value_mask.to(source_norm.dtype)], dim=-1)
        source_input = torch.where(
            (present[..., None, None] & time_valid[..., None]),
            source_input,
            torch.zeros_like(source_input),
        )
        source_input = source_input.to(dtype=dtype)
        root_projected = self.root_source_proj(source_input)
        body_projected = self.body_source_proj(source_input)
        projected_valid = time_valid & value_mask.any(dim=-1) & present[..., None]

        aligned_root, aligned_valid = align_projected_source(
            root_projected,
            projected_valid,
            source_lengths,
            target_valid,
            target_lengths,
            time_map,
        )
        aligned_body, body_valid = align_projected_source(
            body_projected,
            projected_valid,
            source_lengths,
            target_valid,
            target_lengths,
            time_map,
        )
        if not torch.equal(aligned_valid, body_valid):
            raise RuntimeError("Root/body source alignment validity diverged")

        token_denom = aligned_valid.sum(dim=1).clamp_min(1).to(dtype=dtype)[..., None]
        token_root = aligned_root.sum(dim=1) / token_denom
        token_body = aligned_body.sum(dim=1) / token_denom

        requested = target_lengths[:, None].expand_as(source_lengths).float()
        length_features = torch.stack(
            [
                source_lengths.float() / float(self.max_frames),
                requested / float(self.max_frames),
                torch.log((requested + 1.0) / (source_lengths.float() + 1.0)),
            ],
            dim=-1,
        ).to(dtype=dtype)
        role = self.role_embed(condition.source_role_id.to(device=source.device))
        length = self.length_proj(length_features)
        present_f = present.to(dtype=dtype)[..., None]
        slot_denom = present_f.sum(dim=1).clamp_min(1.0)
        slot_meta = ((role + length) * present_f).sum(dim=1) / slot_denom

        task = self.task_embed(condition.task_id.to(device=source.device))
        op = self.op_embed(condition.target_op_id.to(device=source.device))
        source_present = present.any(dim=1)
        edit_task = condition.task_id.to(device=source.device) == int(TaskId.EDIT)
        # A source-free EDIT row is the text-only/task-unconditional CFG branch.
        # It must still receive the explicit EDIT task/op identity. GENERATE with
        # no source remains an exact-zero legacy context path.
        context_present = source_present | edit_task
        final_gate = context_present[:, None] & target_valid
        shared = slot_meta[:, None, :] + task[:, None, :] + op
        root = torch.where(
            final_gate[..., None], token_root + shared, torch.zeros_like(token_root)
        )
        body = torch.where(
            final_gate[..., None], token_body + shared, torch.zeros_like(token_body)
        )
        # With an all-absent source, temporal alignment legitimately skips every
        # source projection. Keep all context tensors in the autograd topology so
        # Stage A/B1 can distinguish audited exact-zero gradients from an
        # accidentally disconnected branch. Reading one scalar creates a dense
        # all-zero gradient tensor for each parameter without changing values.
        zero_anchor = sum(
            (
                parameter.reshape(-1)[0] * 0.0
                for parameter in (
                    *self.weight_parameters(),
                    *self.bias_and_embedding_parameters(),
                )
            ),
            root.new_zeros(()),
        ).to(dtype=root.dtype)
        root = root + zero_anchor
        body = body + zero_anchor
        return SourceContextOutput(
            root=root,
            body=body,
            aligned_valid=aligned_valid,
            context_present=context_present,
        )


@dataclass
class KimodoContextFlowOutput:
    prediction: torch.Tensor
    root_prediction_raw: torch.Tensor
    root_for_body: torch.Tensor
    local_root: torch.Tensor
    context_root: torch.Tensor
    context_body: torch.Tensor
    context_present: torch.Tensor
    text_tokens: torch.Tensor
    text_pooled: torch.Tensor
    text_padding_mask: torch.Tensor
    ease_bias: torch.Tensor


class HY273KimodoContextFlow(HY273RedenoiseKimodoLike):
    """The root/body backbone with configurable source-context fusion."""

    def __init__(
        self,
        *args,
        max_frames: int = 300,
        normalize_contacts: bool = False,
        source_fusion_mode: str = "additive",
        use_ease: bool = False,
        ease_stats_dir: str | Path = "",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_fusion_mode = str(source_fusion_mode)
        self.use_ease = bool(use_ease)
        if self.source_fusion_mode not in {"additive", "token_block"}:
            raise ValueError(
                "source_fusion_mode must be 'additive' or 'token_block', got "
                f"{self.source_fusion_mode!r}"
            )
        motion_stats_dir = kwargs.get("motion_stats_dir", "")
        stats_variance_eps = float(kwargs.get("stats_variance_eps", 1e-5))
        # Created only after every legacy module, so legacy initialization consumes
        # exactly the same RNG sequence as HY273RedenoiseKimodoLike.
        # The additive branch is exactly zero-initialized, so consuming a new RNG
        # stream here would only perturb downstream reproducibility. Preserve the
        # legacy constructor's post-init RNG state as part of M0.2 identity.
        with torch.random.fork_rng(devices=[]):
            self.source_context = HY273SourceContext(
                hidden_dim=self.hidden_dim,
                motion_stats_dir=motion_stats_dir,
                max_frames=max_frames,
                variance_eps=stats_variance_eps,
                normalize_contacts=normalize_contacts,
            )
            self.ease_conditioner = (
                HY273EaseConditioner(self.hidden_dim, ease_stats_dir)
                if self.use_ease
                else None
            )

    def context_weight_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.source_context.weight_parameters()

    def context_bias_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.source_context.bias_and_embedding_parameters()

    def ease_weight_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.ease_conditioner is None:
            return ()
        return self.ease_conditioner.weight_parameters()

    def ease_bias_parameters(self) -> tuple[nn.Parameter, ...]:
        if self.ease_conditioner is None:
            return ()
        return self.ease_conditioner.bias_parameters()

    def base_parameters(self) -> tuple[nn.Parameter, ...]:
        auxiliary_ids = {
            id(param)
            for param in (
                *self.context_weight_parameters(),
                *self.context_bias_parameters(),
                *self.ease_weight_parameters(),
                *self.ease_bias_parameters(),
            )
        }
        return tuple(
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in auxiliary_ids
        )

    def _fuse_source_and_run_backbone(
        self,
        *,
        backbone: nn.Module,
        target_hidden: torch.Tensor,
        context_hidden: torch.Tensor,
        context_present: torch.Tensor,
        use_source_token_block: bool,
        text_tokens: torch.Tensor,
        cond: torch.Tensor,
        target_valid: torch.Tensor,
        text_padding_mask: torch.Tensor,
        target_pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not use_source_token_block:
            return backbone(
                motion=target_hidden + context_hidden,
                text=text_tokens,
                cond=cond,
                motion_valid=target_valid,
                text_padding_mask=text_padding_mask,
                motion_pos_ids=target_pos_ids,
            )

        motion, motion_valid, motion_pos_ids, target_slice = (
            build_source_token_block(
                target_hidden,
                context_hidden,
                target_valid,
                context_present,
                target_pos_ids,
            )
        )
        hidden = backbone(
            motion=motion,
            text=text_tokens,
            cond=cond,
            motion_valid=motion_valid,
            text_padding_mask=text_padding_mask,
            motion_pos_ids=motion_pos_ids,
        )
        return hidden[:, target_slice]

    def _profiled_text_condition(
        self,
        texts: Optional[Iterable[str]],
        profiles: tuple[str, ...] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text_drop_prob: float,
        force_drop_text: bool,
    ) -> RawTextCondition:
        text_list = [""] * batch_size if texts is None else list(texts)
        if len(text_list) != batch_size:
            raise ValueError(f"Expected {batch_size} texts, got {len(text_list)}")
        kwargs = dict(
            device=device,
            dtype=dtype,
            drop_prob=float(text_drop_prob),
            force_drop=bool(force_drop_text),
        )
        if getattr(self.text_encoder, "supports_profiles", False):
            return self.text_encoder(text_list, profiles=profiles, **kwargs)
        return self.text_encoder(text_list, **kwargs)

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
        condition: ConditionBatch | None = None,
    ) -> torch.Tensor | KimodoContextFlowOutput:
        if model_in.ndim != 3 or model_in.shape[-1] != self.model_in_dim:
            raise ValueError(f"Expected model_in [B,T,{self.model_in_dim}], got {model_in.shape}")
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

        text_list = tuple([""] * bsz if text is None else list(text))
        if len(text_list) != bsz:
            raise ValueError(f"Expected {bsz} texts, got {len(text_list)}")
        profiles = None if condition is None else condition.text_encoding_profile
        text_cond = self._profiled_text_condition(
            text_list,
            profiles,
            bsz,
            device,
            dtype,
            text_drop_prob,
            force_drop_text,
        )
        cond = self._compose_global_condition(
            t,
            c_dir,
            text_cond.pooled,
            dtype=dtype,
        )
        pos = (
            torch.arange(frames, device=device, dtype=torch.long)
            .view(1, frames, 1)
            .expand(bsz, frames, 1)
        )

        if condition is None:
            context_root = model_in.new_zeros((bsz, frames, self.hidden_dim))
            context_body = model_in.new_zeros((bsz, frames, self.hidden_dim))
            context_present = torch.zeros(bsz, device=device, dtype=torch.bool)
        else:
            context_output = self.source_context(
                condition, target_frames=frames, dtype=dtype
            )
            context_root = context_output.root.to(device=device, dtype=dtype)
            context_body = context_output.body.to(device=device, dtype=dtype)
            context_present = context_output.context_present.to(device=device)
        use_source_token_block = self.source_fusion_mode == "token_block" and bool(
            context_present.any().item()
        )
        if self.use_ease:
            if condition is None:
                ease_physical = model_in.new_zeros((bsz, 6), dtype=torch.float32)
                ease_present = torch.zeros(bsz, device=device, dtype=torch.bool)
            else:
                ease_physical = condition.ease_physical.to(device=device)
                ease_present = condition.ease_present.to(device=device)
            if self.ease_conditioner is None:
                raise RuntimeError("Ease is enabled without an Ease conditioner")
            ease_bias = self.ease_conditioner(
                ease_physical,
                ease_present,
                dtype=dtype,
            ).unsqueeze(1)
        else:
            if condition is not None and bool(condition.ease_present.any()):
                raise ValueError("Condition carries Ease but the model has Ease disabled")
            ease_bias = model_in.new_zeros((bsz, 1, self.hidden_dim))

        state = model_in[..., :DIM_HY273]
        motion_mask = model_in[..., DIM_HY273:]
        root_hidden = self.root_input_proj(model_in) + ease_bias
        if self.self_conditioning and x_self_cond is not None:
            root_hidden = root_hidden + self.self_cond_scale * self.root_self_cond_proj(
                x_self_cond[..., :ROOT_DIM].to(device=device, dtype=dtype)
            )
        root_hidden = self._fuse_source_and_run_backbone(
            backbone=self.root_backbone,
            target_hidden=root_hidden,
            context_hidden=context_root,
            context_present=context_present,
            use_source_token_block=use_source_token_block,
            text_tokens=text_cond.tokens,
            cond=cond,
            target_valid=length_mask,
            text_padding_mask=text_cond.padding_mask,
            target_pos_ids=pos,
        )
        root_prediction_raw = self.root_output_proj(root_hidden)

        root_for_body = root_prediction_raw
        bridge_root = (
            root_for_body.detach()
            if self.training and self.detach_root_bridge
            else root_for_body
        )
        local_root = self.root_conditioner(bridge_root, lengths)

        body_in = torch.cat([local_root, state[..., ROOT_DIM:], motion_mask], dim=-1)
        body_hidden = self.body_input_proj(body_in) + ease_bias
        if self.self_conditioning and x_self_cond is not None:
            body_hidden = body_hidden + self.self_cond_scale * self.body_self_cond_proj(
                x_self_cond[..., ROOT_DIM:].to(device=device, dtype=dtype)
            )
        body_hidden = self._fuse_source_and_run_backbone(
            backbone=self.body_backbone,
            target_hidden=body_hidden,
            context_hidden=context_body,
            context_present=context_present,
            use_source_token_block=use_source_token_block,
            text_tokens=text_cond.tokens,
            cond=cond,
            target_valid=length_mask,
            text_padding_mask=text_cond.padding_mask,
            target_pos_ids=pos,
        )
        body_prediction = self.body_output_proj(body_hidden)
        prediction = torch.cat([root_prediction_raw, body_prediction], dim=-1)
        if return_details:
            return KimodoContextFlowOutput(
                prediction=prediction,
                root_prediction_raw=root_prediction_raw,
                root_for_body=root_for_body,
                local_root=local_root,
                context_root=context_root,
                context_body=context_body,
                context_present=context_present,
                text_tokens=text_cond.tokens,
                text_pooled=text_cond.pooled,
                text_padding_mask=text_cond.padding_mask,
                ease_bias=ease_bias,
            )
        return prediction
