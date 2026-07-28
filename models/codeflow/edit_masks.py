"""Mask helpers for instruction-based global motion editing."""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Sequence, Tuple

import torch

from .inpaint_protocols import PARTGRID_OP_EDIT, PARTGRID_OP_PRESERVE
from .motion_code_flow import lengths_to_mask


def dilate_time_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate an edit mask along time while keeping the part axis unchanged."""
    radius = int(radius)
    if radius <= 0:
        return mask
    out = mask.clone()
    for offset in range(1, radius + 1):
        out[:, offset:] |= mask[:, :-offset]
        out[:, :-offset] |= mask[:, offset:]
    return out


def _same_shape_ids(source_ids: torch.Tensor, target_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if source_ids.shape != target_ids.shape:
        raise ValueError(f"source_ids and target_ids must have the same shape, got {source_ids.shape} and {target_ids.shape}")
    if source_ids.ndim != 3:
        raise ValueError(f"edit masks expect code ids [B,T,P], got {tuple(source_ids.shape)}")
    return source_ids.long(), target_ids.long()


def build_code_edit_preserve_mask(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    mode: str = "code",
    embedding_threshold: float = 0.0,
    min_edit_frac: float = 0.0,
    min_edit_cells: int = 1,
    temporal_dilate: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a training mask from source-target differences.

    Returns preserve_mask, op_ids, and scalar tensor stats. The mask uses only
    frozen VQ ids/embeddings; it does not change the tokenizer.
    """
    mode = str(mode or "code").lower()
    if mode not in {"none", "code", "embedding", "code_or_embedding"}:
        raise ValueError(f"Unsupported global edit mask mode: {mode}")

    bsz, latent_len, num_parts, _ = target_embeddings.shape
    device = target_embeddings.device
    source_ids, target_ids = _same_shape_ids(source_ids.to(device), target_ids.to(device))
    token_lengths = token_lengths.to(device).long().clamp(min=1, max=latent_len)
    valid = lengths_to_mask(token_lengths, latent_len)
    valid_parts = valid[:, :, None].expand(bsz, latent_len, num_parts)

    if mode == "none":
        edit_mask = valid_parts.clone()
        dist = torch.zeros((bsz, latent_len, num_parts), device=device, dtype=target_embeddings.dtype)
    else:
        code_edit = source_ids != target_ids
        dist = (target_embeddings.float() - source_embeddings.float()).square().mean(dim=-1).sqrt()
        if mode == "code":
            edit_mask = code_edit
        elif mode == "embedding":
            threshold = float(embedding_threshold)
            edit_mask = dist > threshold
        else:
            threshold = float(embedding_threshold)
            edit_mask = code_edit | ((dist > threshold) if threshold > 0.0 else torch.zeros_like(code_edit))
        edit_mask = edit_mask & valid_parts

        min_frac = max(0.0, float(min_edit_frac))
        min_cells = max(0, int(min_edit_cells))
        if min_frac > 0.0 or min_cells > 0:
            edit_mask = edit_mask.clone()
            for batch_idx in range(bsz):
                valid_flat = valid_parts[batch_idx].reshape(-1)
                valid_count = int(valid_flat.sum().item())
                if valid_count <= 0:
                    continue
                required = max(min_cells, int(math.ceil(valid_count * min_frac)))
                required = min(required, valid_count)
                current = int(edit_mask[batch_idx].sum().item())
                if current >= required:
                    continue
                scores = dist[batch_idx].reshape(-1)
                valid_scores = scores.masked_fill(~valid_flat, float("-inf"))
                topk = torch.topk(valid_scores, k=required, largest=True).indices
                flat = edit_mask[batch_idx].reshape(-1)
                flat[topk] = True

        edit_mask = dilate_time_mask(edit_mask, int(temporal_dilate)) & valid_parts

    preserve_mask = valid_parts & ~edit_mask
    op_ids = torch.full((bsz, latent_len, num_parts), PARTGRID_OP_EDIT, device=device, dtype=torch.long)
    op_ids = torch.where(preserve_mask, torch.full_like(op_ids, PARTGRID_OP_PRESERVE), op_ids)

    valid_count = valid_parts.sum().float().clamp_min(1.0)
    edit_count = edit_mask.sum().float()
    preserve_count = preserve_mask.sum().float()
    stats = {
        "global_edit_generated_cell_frac": edit_count / valid_count,
        "global_edit_preserved_cell_frac": preserve_count / valid_count,
        "global_edit_mask_code_change_frac": ((source_ids != target_ids) & valid_parts).sum().float() / valid_count,
        "global_edit_mask_embed_dist": (dist * valid_parts.to(dist.dtype)).sum() / valid_count,
    }
    return preserve_mask, op_ids, stats


_TOKEN_RE = re.compile(r"[a-z0-9']+")

_GLOBAL_WORDS = {
    "whole", "entire", "body", "person", "motion", "action", "pose", "style", "faster", "slower",
    "speed", "pace", "trajectory", "path", "circle", "turn", "rotate", "rotation", "direction",
    "forward", "backward", "leftward", "rightward", "higher", "lower", "bigger", "smaller", "wider",
    "narrower", "jump", "walk", "run", "dance", "sit", "stand", "kneel", "crouch",
}
_START_WORDS = {"start", "begin", "beginning", "initial", "first", "early", "earlier"}
_END_WORDS = {"end", "ending", "final", "last", "late", "later"}
_LEFT_WORDS = {"left"}
_RIGHT_WORDS = {"right"}
_ARM_WORDS = {
    "arm", "arms", "hand", "hands", "elbow", "elbows", "wrist", "wrists", "shoulder", "shoulders",
    "throw", "throws", "throwing", "catch", "catches", "catching", "reach", "reaches", "reaching",
    "wave", "waves", "waving", "clap", "claps", "clapping", "punch", "punches", "punching",
}
_LEG_WORDS = {
    "leg", "legs", "foot", "feet", "knee", "knees", "ankle", "ankles", "toe", "toes",
    "kick", "kicks", "kicking",
}
_HEAD_WORDS = {"head", "neck", "face", "gaze", "look", "looking"}
_TORSO_WORDS = {"torso", "chest", "waist", "hip", "hips", "spine", "back"}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def _part_ids_for_tokens(tokens: set[str], num_parts: int) -> Sequence[int]:
    # The released HumanML3D part tokenizer uses six overlapping partitions.
    # These groups are intentionally broad; if a phrase is ambiguous, callers
    # should fall back to editing all parts rather than over-preserving.
    if num_parts < 6:
        return list(range(num_parts))
    parts = set()
    left = bool(tokens & _LEFT_WORDS)
    right = bool(tokens & _RIGHT_WORDS)
    if tokens & _ARM_WORDS:
        if left and not right:
            parts.update([3])
        elif right and not left:
            parts.update([5])
        else:
            parts.update([3, 5])
        parts.add(1)
    if tokens & _LEG_WORDS:
        if left and not right:
            parts.update([4])
        elif right and not left:
            parts.update([2])
        else:
            parts.update([2, 4])
        parts.add(0)
    if tokens & _HEAD_WORDS:
        parts.update([0, 1])
    if tokens & _TORSO_WORDS:
        parts.update([0, 1])
    return sorted(part for part in parts if 0 <= part < num_parts)


def build_instruction_edit_preserve_mask(
    texts: Iterable[str],
    token_lengths: torch.Tensor,
    latent_len: int,
    num_parts: int,
    device: torch.device,
    *,
    temporal_dilate: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a non-oracle inference mask from edit instructions only.

    If an instruction is not clearly local, all valid cells remain editable.
    """
    text_list = list(texts)
    bsz = len(text_list)
    token_lengths = token_lengths.to(device).long().clamp(min=1, max=int(latent_len))
    valid = lengths_to_mask(token_lengths, int(latent_len))
    valid_parts = valid[:, :, None].expand(bsz, int(latent_len), int(num_parts))
    edit_mask = torch.zeros_like(valid_parts, dtype=torch.bool)

    for batch_idx, text in enumerate(text_list):
        toks = _tokens(text)
        local_parts = list(_part_ids_for_tokens(toks, int(num_parts)))
        has_temporal = bool((toks & _START_WORDS) or (toks & _END_WORDS))
        is_global = (not local_parts and bool(toks & _GLOBAL_WORDS)) or (not local_parts and not has_temporal)
        valid_len = int(token_lengths[batch_idx].item())

        if is_global:
            edit_mask[batch_idx, :valid_len, :] = True
            continue

        if toks & _START_WORDS:
            end = max(1, int(math.ceil(valid_len * 0.35)))
            part_ids = local_parts or list(range(int(num_parts)))
            edit_mask[batch_idx, :end, part_ids] = True
        if toks & _END_WORDS:
            start = int(math.floor(valid_len * 0.65))
            part_ids = local_parts or list(range(int(num_parts)))
            edit_mask[batch_idx, start:valid_len, part_ids] = True
        if local_parts:
            edit_mask[batch_idx, :valid_len, local_parts] = True

        if not bool(edit_mask[batch_idx].any()):
            edit_mask[batch_idx, :valid_len, :] = True

    edit_mask = dilate_time_mask(edit_mask, int(temporal_dilate)) & valid_parts
    preserve_mask = valid_parts & ~edit_mask
    op_ids = torch.full_like(preserve_mask, PARTGRID_OP_EDIT, dtype=torch.long)
    op_ids = torch.where(preserve_mask, torch.full_like(op_ids, PARTGRID_OP_PRESERVE), op_ids)
    valid_count = valid_parts.sum().float().clamp_min(1.0)
    stats = {
        "global_edit_infer_generated_cell_frac": edit_mask.sum().float() / valid_count,
        "global_edit_infer_preserved_cell_frac": preserve_mask.sum().float() / valid_count,
    }
    return preserve_mask, op_ids, stats
