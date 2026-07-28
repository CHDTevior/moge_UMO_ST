"""Shared inpainting mask protocols for PartGrid CodeFlow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .motion_code_flow import lengths_to_mask


PARTGRID_OP_PRESERVE = 0
PARTGRID_OP_GENERATE = 1
PARTGRID_OP_EDIT = 2

PARTGRID_TASK_T2M = 0
PARTGRID_TASK_TEMPORAL = 1
PARTGRID_TASK_PARTGRID = 2
PARTGRID_TASK_PREDICTION = 3
PARTGRID_TASK_BACKCASTING = 4
PARTGRID_TASK_IN_BETWEENING = 5
PARTGRID_TASK_INFILLING = 6
PARTGRID_TASK_GLOBAL_EDIT = 7
PARTGRID_NUM_TASKS_WITHOUT_GLOBAL_EDIT = PARTGRID_TASK_INFILLING + 1
PARTGRID_NUM_TASKS_WITH_GLOBAL_EDIT = PARTGRID_TASK_GLOBAL_EDIT + 1

INPAINT_MODE_T2M = "t2m"
INPAINT_MODE_TEMPORAL = "temporal"
INPAINT_MODE_PARTGRID = "partgrid"
INPAINT_MODE_PREDICTION = "prediction"
INPAINT_MODE_BACKCASTING = "backcasting"
INPAINT_MODE_IN_BETWEEN = "in_between"
INPAINT_MODE_KEYFRAME = "keyframe"
EDIT_MODE_GLOBAL = "global_edit"

UMO_MODE_DISPLAY_NAMES = {
    INPAINT_MODE_PREDICTION: "prediction",
    INPAINT_MODE_BACKCASTING: "backcasting",
    INPAINT_MODE_IN_BETWEEN: "inbetweening",
    INPAINT_MODE_KEYFRAME: "infilling",
}

UMO_TEMPORAL_MODES = (
    INPAINT_MODE_PREDICTION,
    INPAINT_MODE_BACKCASTING,
    INPAINT_MODE_IN_BETWEEN,
    INPAINT_MODE_KEYFRAME,
)
INPAINT_EVAL_MODES = (
    INPAINT_MODE_TEMPORAL,
    INPAINT_MODE_PARTGRID,
    *UMO_TEMPORAL_MODES,
)
INPAINT_TRAIN_MODES = (
    INPAINT_MODE_T2M,
    INPAINT_MODE_TEMPORAL,
    INPAINT_MODE_PARTGRID,
    *UMO_TEMPORAL_MODES,
)

_MODE_ALIASES = {
    "t2m": INPAINT_MODE_T2M,
    "text_to_motion": INPAINT_MODE_T2M,
    "text-to-motion": INPAINT_MODE_T2M,
    "temporal": INPAINT_MODE_TEMPORAL,
    "span": INPAINT_MODE_TEMPORAL,
    "temporal_span": INPAINT_MODE_TEMPORAL,
    "partgrid": INPAINT_MODE_PARTGRID,
    "part_grid": INPAINT_MODE_PARTGRID,
    "part-grid": INPAINT_MODE_PARTGRID,
    "prediction": INPAINT_MODE_PREDICTION,
    "future": INPAINT_MODE_PREDICTION,
    "backcasting": INPAINT_MODE_BACKCASTING,
    "past": INPAINT_MODE_BACKCASTING,
    "inbetween": INPAINT_MODE_IN_BETWEEN,
    "in_between": INPAINT_MODE_IN_BETWEEN,
    "in-between": INPAINT_MODE_IN_BETWEEN,
    "inbetweening": INPAINT_MODE_IN_BETWEEN,
    "in_betweening": INPAINT_MODE_IN_BETWEEN,
    "in-betweening": INPAINT_MODE_IN_BETWEEN,
    "keyframe": INPAINT_MODE_KEYFRAME,
    "keyframes": INPAINT_MODE_KEYFRAME,
    "infilling": INPAINT_MODE_KEYFRAME,
    "infill": INPAINT_MODE_KEYFRAME,
    "keyframe_infilling": INPAINT_MODE_KEYFRAME,
    "keyframe-infilling": INPAINT_MODE_KEYFRAME,
}


@dataclass(frozen=True)
class InpaintMaskConfig:
    mask_protocol: str = "random"
    temporal_min_ratio: float = 0.20
    temporal_max_ratio: float = 0.60
    partgrid_min_frame_ratio: float = 0.20
    partgrid_max_frame_ratio: float = 0.60
    partgrid_min_parts: int = 1
    partgrid_max_parts: int = 3
    partgrid_regions: int = 1
    prediction_min_ratio: float = 0.65
    prediction_max_ratio: float = 0.85
    backcasting_min_ratio: float = 0.65
    backcasting_max_ratio: float = 0.85
    in_between_min_ratio: float = 0.65
    in_between_max_ratio: float = 0.85
    keyframe_min_preserve_ratio: float = 0.05
    keyframe_max_preserve_ratio: float = 0.15
    keyframe_include_endpoints: bool = False
    fixed_prediction_generate_ratio: float = 0.75
    fixed_backcasting_generate_ratio: float = 0.75
    fixed_in_between_generate_ratio: float = 0.50
    fixed_keyframe_count: int = 5
    fixed_keyframe_density: float = 0.0


def normalize_inpaint_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    key = key.replace(" ", "_")
    if key not in _MODE_ALIASES:
        valid = ", ".join(INPAINT_TRAIN_MODES)
        raise ValueError(f"Unsupported inpainting mode: {mode}. Valid modes: {valid}")
    return _MODE_ALIASES[key]


def normalize_mask_protocol(protocol: str) -> str:
    key = str(protocol or "random").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "random",
        "random": "random",
        "stochastic": "random",
        "fixed": "fixed_umo",
        "fixed_eval": "fixed_umo",
        "fixed_umo": "fixed_umo",
        "fixed_umo4": "fixed_umo",
        "umo_fixed": "fixed_umo",
    }
    if key not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported inpainting mask protocol: {protocol}. Valid protocols: {valid}")
    return aliases[key]


def task_id_for_mode(mode: str) -> int:
    mode = normalize_inpaint_mode(mode)
    if mode == INPAINT_MODE_T2M:
        return PARTGRID_TASK_T2M
    if mode == INPAINT_MODE_PARTGRID:
        return PARTGRID_TASK_PARTGRID
    if mode == INPAINT_MODE_PREDICTION:
        return PARTGRID_TASK_PREDICTION
    if mode == INPAINT_MODE_BACKCASTING:
        return PARTGRID_TASK_BACKCASTING
    if mode == INPAINT_MODE_IN_BETWEEN:
        return PARTGRID_TASK_IN_BETWEENING
    if mode == INPAINT_MODE_KEYFRAME:
        return PARTGRID_TASK_INFILLING
    return PARTGRID_TASK_TEMPORAL


def paper_name_for_mode(mode: str) -> str:
    mode = normalize_inpaint_mode(mode)
    return UMO_MODE_DISPLAY_NAMES.get(mode, mode)


def _rand_int(low: int, high: int, device: torch.device) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(int(low), int(high) + 1, (1,), device=device).item())


def _ratio_count(
    valid_len: int,
    min_ratio: float,
    max_ratio: float,
    device: torch.device,
    *,
    min_count: int = 1,
    max_count: int | None = None,
) -> int:
    valid_len = max(int(valid_len), 1)
    min_ratio = max(0.0, min(float(min_ratio), 1.0))
    max_ratio = max(min_ratio, min(float(max_ratio), 1.0))
    lo = max(int(min_count), int(math.ceil(valid_len * min_ratio)))
    hi = max(lo, int(math.ceil(valid_len * max_ratio)))
    if max_count is not None:
        hi = min(hi, int(max_count))
        lo = min(lo, hi)
    hi = min(hi, valid_len)
    return _rand_int(lo, hi, device)


def _rand_span(valid_len: int, min_ratio: float, max_ratio: float, device: torch.device) -> tuple[int, int]:
    span_len = _ratio_count(valid_len, min_ratio, max_ratio, device)
    start_max = max(int(valid_len) - span_len, 0)
    start = _rand_int(0, start_max, device)
    return start, start + span_len


def _choose_keyframes(valid_len: int, cfg: InpaintMaskConfig, device: torch.device) -> torch.Tensor:
    valid_len = max(int(valid_len), 1)
    max_preserve = valid_len - 1 if valid_len > 1 else 1
    preserve_count = _ratio_count(
        valid_len,
        cfg.keyframe_min_preserve_ratio,
        cfg.keyframe_max_preserve_ratio,
        device,
        min_count=1,
        max_count=max_preserve,
    )
    if bool(cfg.keyframe_include_endpoints) and valid_len >= 2 and preserve_count >= 2:
        endpoint_ids = torch.tensor([0, valid_len - 1], device=device, dtype=torch.long)
        extra_count = preserve_count - 2
        if extra_count <= 0 or valid_len <= 2:
            return endpoint_ids[:preserve_count]
        interior = torch.randperm(valid_len - 2, device=device, dtype=torch.long)[:extra_count] + 1
        return torch.cat([endpoint_ids, interior], dim=0)
    return torch.randperm(valid_len, device=device, dtype=torch.long)[:preserve_count]


def _fixed_count(
    valid_len: int,
    ratio: float,
    *,
    min_count: int = 1,
    max_count: int | None = None,
) -> int:
    valid_len = max(int(valid_len), 1)
    count = int(round(valid_len * max(0.0, min(float(ratio), 1.0))))
    count = max(int(min_count), count)
    if max_count is not None:
        count = min(count, int(max_count))
    return max(1, min(count, valid_len))


def _choose_fixed_keyframes(valid_len: int, cfg: InpaintMaskConfig, device: torch.device) -> torch.Tensor:
    valid_len = max(int(valid_len), 1)
    if int(cfg.fixed_keyframe_count) > 0:
        preserve_count = min(int(cfg.fixed_keyframe_count), valid_len)
    else:
        preserve_count = _fixed_count(
            valid_len,
            float(cfg.fixed_keyframe_density),
            min_count=1,
            max_count=valid_len,
        )
    if preserve_count <= 1:
        return torch.tensor([valid_len // 2], device=device, dtype=torch.long)

    ids = torch.linspace(0, valid_len - 1, preserve_count, device=device).round().long().unique(sorted=True)
    if ids.numel() < preserve_count:
        present = torch.zeros(valid_len, device=device, dtype=torch.bool)
        present[ids] = True
        fill = torch.arange(valid_len, device=device, dtype=torch.long)[~present][: preserve_count - ids.numel()]
        ids = torch.cat([ids, fill]).unique(sorted=True)
    return ids[:preserve_count]


def _as_mode_list(modes: str | Iterable[str], batch_size: int) -> list[str]:
    if isinstance(modes, str):
        return [normalize_inpaint_mode(modes)] * int(batch_size)
    out = [normalize_inpaint_mode(mode) for mode in modes]
    if len(out) != int(batch_size):
        raise ValueError(f"Expected {batch_size} inpainting modes, got {len(out)}")
    return out


def build_inpaint_preserve_mask(
    token_lengths: torch.Tensor,
    latent_len: int,
    num_parts: int,
    modes: str | Sequence[str],
    cfg: InpaintMaskConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = token_lengths.device
    bsz = int(token_lengths.shape[0])
    token_lengths = token_lengths.to(device=device, dtype=torch.long).clamp(min=1, max=int(latent_len))
    valid = lengths_to_mask(token_lengths, int(latent_len))
    preserve_mask = torch.zeros((bsz, int(latent_len), int(num_parts)), device=device, dtype=torch.bool)
    mode_list = _as_mode_list(modes, bsz)
    task_ids = torch.tensor([task_id_for_mode(mode) for mode in mode_list], device=device, dtype=torch.long)
    mask_protocol = normalize_mask_protocol(cfg.mask_protocol)

    for batch_idx, mode in enumerate(mode_list):
        valid_len = int(token_lengths[batch_idx].item())
        if mode == INPAINT_MODE_T2M:
            continue

        if mask_protocol == "fixed_umo" and mode in UMO_TEMPORAL_MODES:
            if mode == INPAINT_MODE_KEYFRAME:
                frame_ids = _choose_fixed_keyframes(valid_len, cfg, device)
                preserve_mask[batch_idx, frame_ids, :] = True
                continue

            preserve_mask[batch_idx, :valid_len, :] = True
            if mode == INPAINT_MODE_PREDICTION:
                span_len = _fixed_count(
                    valid_len,
                    cfg.fixed_prediction_generate_ratio,
                    max_count=valid_len - 1 if valid_len > 1 else 1,
                )
                preserve_mask[batch_idx, valid_len - span_len:valid_len, :] = False
            elif mode == INPAINT_MODE_BACKCASTING:
                span_len = _fixed_count(
                    valid_len,
                    cfg.fixed_backcasting_generate_ratio,
                    max_count=valid_len - 1 if valid_len > 1 else 1,
                )
                preserve_mask[batch_idx, :span_len, :] = False
            elif mode == INPAINT_MODE_IN_BETWEEN:
                if valid_len >= 3:
                    span_len = _fixed_count(
                        valid_len,
                        cfg.fixed_in_between_generate_ratio,
                        max_count=valid_len - 2,
                    )
                    start = max(1, (valid_len - span_len) // 2)
                    end = min(valid_len - 1, start + span_len)
                    start = max(1, end - span_len)
                    preserve_mask[batch_idx, start:end, :] = False
                else:
                    span_len = _fixed_count(valid_len, cfg.fixed_in_between_generate_ratio)
                    start = max(0, (valid_len - span_len) // 2)
                    preserve_mask[batch_idx, start:start + span_len, :] = False
            continue

        if mode == INPAINT_MODE_KEYFRAME:
            frame_ids = _choose_keyframes(valid_len, cfg, device)
            preserve_mask[batch_idx, frame_ids, :] = True
            continue

        preserve_mask[batch_idx, :valid_len, :] = True
        if mode == INPAINT_MODE_TEMPORAL:
            start, end = _rand_span(valid_len, cfg.temporal_min_ratio, cfg.temporal_max_ratio, device)
            preserve_mask[batch_idx, start:end, :] = False
        elif mode == INPAINT_MODE_PREDICTION:
            span_len = _ratio_count(valid_len, cfg.prediction_min_ratio, cfg.prediction_max_ratio, device)
            preserve_mask[batch_idx, valid_len - span_len:valid_len, :] = False
        elif mode == INPAINT_MODE_BACKCASTING:
            span_len = _ratio_count(valid_len, cfg.backcasting_min_ratio, cfg.backcasting_max_ratio, device)
            preserve_mask[batch_idx, :span_len, :] = False
        elif mode == INPAINT_MODE_IN_BETWEEN:
            if valid_len >= 3:
                span_len = _ratio_count(
                    valid_len,
                    cfg.in_between_min_ratio,
                    cfg.in_between_max_ratio,
                    device,
                    max_count=valid_len - 2,
                )
                start = _rand_int(1, valid_len - span_len - 1, device)
                preserve_mask[batch_idx, start:start + span_len, :] = False
            else:
                start, end = _rand_span(valid_len, cfg.in_between_min_ratio, cfg.in_between_max_ratio, device)
                preserve_mask[batch_idx, start:end, :] = False
        elif mode == INPAINT_MODE_PARTGRID:
            regions = max(1, int(cfg.partgrid_regions))
            min_parts = max(1, int(cfg.partgrid_min_parts))
            max_parts = min(max(min_parts, int(cfg.partgrid_max_parts)), int(num_parts))
            for _ in range(regions):
                start, end = _rand_span(
                    valid_len,
                    cfg.partgrid_min_frame_ratio,
                    cfg.partgrid_max_frame_ratio,
                    device,
                )
                part_count = _rand_int(min_parts, max_parts, device)
                part_ids = torch.randperm(int(num_parts), device=device, dtype=torch.long)[:part_count]
                preserve_mask[batch_idx, start:end, part_ids] = False
        else:
            raise ValueError(f"Unsupported inpainting mode after normalization: {mode}")

    preserve_mask = preserve_mask & valid[:, :, None]
    return preserve_mask, task_ids


def build_partgrid_op_ids(preserve_mask: torch.Tensor) -> torch.Tensor:
    op_ids = torch.full_like(preserve_mask, PARTGRID_OP_GENERATE, dtype=torch.long)
    return torch.where(
        preserve_mask.to(dtype=torch.bool),
        torch.full_like(op_ids, PARTGRID_OP_PRESERVE),
        op_ids,
    )
