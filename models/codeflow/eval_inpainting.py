"""Inpainting evaluation for PartGrid CodeFlow models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from utils.metrics import (
    calculate_mpjpe,
    calculate_pa_mpjpe,
    calculate_R_precision,
    calculate_activation_statistics,
    calculate_diversity,
    calculate_frechet_distance,
    calculate_matching_score,
)
from utils.motion_process import recover_from_ric

from .eval_t2m import eval_motion_to_vq_space, prepare_codeflow_motion_for_eval
from .inpaint_protocols import InpaintMaskConfig, build_inpaint_preserve_mask, paper_name_for_mode
from .motion_code_flow import lengths_to_mask


@dataclass
class CodeFlowInpaintEvalConfig:
    steps: int = 96
    cond_scale: float = 6.0
    terminal_mode: Optional[str] = None
    decode_mode: Optional[str] = None
    unit_length: int = 4
    max_batches: int = 0
    mode: str = "temporal"
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
    allow_small_eval: bool = False
    geometry_severe_quantile: float = 0.75
    joints_num: int = 22


def _mask_config_from_eval_cfg(cfg: CodeFlowInpaintEvalConfig) -> InpaintMaskConfig:
    return InpaintMaskConfig(
        mask_protocol=str(cfg.mask_protocol),
        temporal_min_ratio=float(cfg.temporal_min_ratio),
        temporal_max_ratio=float(cfg.temporal_max_ratio),
        partgrid_min_frame_ratio=float(cfg.partgrid_min_frame_ratio),
        partgrid_max_frame_ratio=float(cfg.partgrid_max_frame_ratio),
        partgrid_min_parts=int(cfg.partgrid_min_parts),
        partgrid_max_parts=int(cfg.partgrid_max_parts),
        partgrid_regions=int(cfg.partgrid_regions),
        prediction_min_ratio=float(cfg.prediction_min_ratio),
        prediction_max_ratio=float(cfg.prediction_max_ratio),
        backcasting_min_ratio=float(cfg.backcasting_min_ratio),
        backcasting_max_ratio=float(cfg.backcasting_max_ratio),
        in_between_min_ratio=float(cfg.in_between_min_ratio),
        in_between_max_ratio=float(cfg.in_between_max_ratio),
        keyframe_min_preserve_ratio=float(cfg.keyframe_min_preserve_ratio),
        keyframe_max_preserve_ratio=float(cfg.keyframe_max_preserve_ratio),
        keyframe_include_endpoints=bool(cfg.keyframe_include_endpoints),
        fixed_prediction_generate_ratio=float(cfg.fixed_prediction_generate_ratio),
        fixed_backcasting_generate_ratio=float(cfg.fixed_backcasting_generate_ratio),
        fixed_in_between_generate_ratio=float(cfg.fixed_in_between_generate_ratio),
        fixed_keyframe_count=int(cfg.fixed_keyframe_count),
        fixed_keyframe_density=float(cfg.fixed_keyframe_density),
    )


def build_eval_preserve_mask(
    token_lengths: torch.Tensor,
    latent_len: int,
    num_parts: int,
    cfg: CodeFlowInpaintEvalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    return build_inpaint_preserve_mask(
        token_lengths=token_lengths,
        latent_len=latent_len,
        num_parts=num_parts,
        modes=cfg.mode,
        cfg=_mask_config_from_eval_cfg(cfg),
    )


def _update_code_sums(
    sums: Dict[str, float],
    tokenizer,
    target_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    preserve_mask: torch.Tensor,
    token_lengths: torch.Tensor,
    cfg: CodeFlowInpaintEvalConfig,
) -> None:
    common_len = min(target_ids.shape[1], pred_ids.shape[1], preserve_mask.shape[1])
    target_ids = target_ids[:, :common_len].long()
    pred_ids = pred_ids[:, :common_len].long()
    preserve_mask = preserve_mask[:, :common_len].bool()
    token_lengths = token_lengths.to(target_ids.device).long().clamp(min=1, max=common_len)
    valid = lengths_to_mask(token_lengths, common_len)
    valid_parts = valid[:, :, None].expand_as(target_ids)
    preserve = preserve_mask & valid_parts
    generate = valid_parts & ~preserve
    code_dist, rank_pct = tokenizer.code_id_distances(target_ids, pred_ids)
    correct = pred_ids == target_ids
    severe = (rank_pct >= float(cfg.geometry_severe_quantile)) & ~correct

    for name, mask in (("generated", generate), ("preserved", preserve)):
        count = mask.sum().float().clamp_min(1.0)
        sums[f"{name}_parts"] += float(mask.sum().detach().cpu())
        sums[f"{name}_correct"] += float((correct & mask).sum().detach().cpu())
        sums[f"{name}_dist_sum"] += float((code_dist * mask.to(code_dist.dtype)).sum().detach().cpu())
        sums[f"{name}_rank_sum"] += float((rank_pct * mask.to(rank_pct.dtype)).sum().detach().cpu())
        sums[f"{name}_severe"] += float((severe & mask).sum().detach().cpu())
        sums[f"{name}_denom_debug"] += float(count.detach().cpu())


def _finalize_code_sums(sums: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = max(sums["generated_parts"] + sums["preserved_parts"], 1.0)
    for name in ("generated", "preserved"):
        count = max(sums[f"{name}_parts"], 1.0)
        out[f"{name}_code_acc"] = sums[f"{name}_correct"] / count
        out[f"{name}_geom_code_dist"] = sums[f"{name}_dist_sum"] / count
        out[f"{name}_geom_rank_pct"] = sums[f"{name}_rank_sum"] / count
        out[f"{name}_geom_severe_rate"] = sums[f"{name}_severe"] / count
        out[f"{name}_cell_frac"] = sums[f"{name}_parts"] / total
    out["generated_token_acc"] = out["generated_code_acc"]
    out["preserved_token_acc"] = out["preserved_code_acc"]
    return out


def _norm_to_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=dtype)
    if value.dim() == 1:
        value = value.view(1, 1, -1)
    return value


def _motion_from_vq_to_raw(
    decoded_motion: torch.Tensor,
    reference_motion: torch.Tensor,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
) -> torch.Tensor:
    bsz, seq_len, feat_dim = reference_motion.shape
    if decoded_motion.shape[0] != bsz or decoded_motion.shape[-1] != feat_dim:
        raise ValueError(
            f"Decoded motion shape {tuple(decoded_motion.shape)} is incompatible with "
            f"reference shape {tuple(reference_motion.shape)}"
        )
    padded = reference_motion.new_zeros((bsz, seq_len, feat_dim))
    copy_len = min(seq_len, decoded_motion.shape[1])
    padded[:, :copy_len] = decoded_motion[:, :copy_len].to(reference_motion.dtype)
    vq_mean = _norm_to_tensor(vq_mean, padded.device, padded.dtype)
    vq_std = _norm_to_tensor(vq_std, padded.device, padded.dtype)
    return padded * vq_std + vq_mean


def _motion_from_eval_to_raw(
    eval_motion: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
) -> torch.Tensor:
    eval_mean = _norm_to_tensor(eval_mean, eval_motion.device, eval_motion.dtype)
    eval_std = _norm_to_tensor(eval_std, eval_motion.device, eval_motion.dtype)
    return eval_motion * eval_std + eval_mean


def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _recover_motionstreamer272_positions(motion_272: torch.Tensor, num_joints: int = 22) -> torch.Tensor:
    if motion_272.ndim != 3 or motion_272.shape[-1] != 272:
        raise ValueError(f"Expected [B,T,272] MotionStreamer motion, got {tuple(motion_272.shape)}")
    bsz, seq_len, _feat_dim = motion_272.shape
    local_positions = motion_272[:, :, 8 : 8 + 3 * num_joints].reshape(bsz, seq_len, num_joints, 3)
    root_velocity = motion_272[:, :, :2]
    heading_delta = motion_272[:, :, 2:8]
    heading_delta_matrix = _rotation_6d_to_matrix(heading_delta)

    heading = torch.empty_like(heading_delta_matrix)
    running = heading_delta_matrix[:, 0]
    heading[:, 0] = running
    for frame in range(1, seq_len):
        running = heading_delta_matrix[:, frame] @ running
        heading[:, frame] = running
    inv_heading = heading.transpose(-1, -2)

    positions = torch.matmul(
        inv_heading[:, :, None].expand(-1, -1, num_joints, -1, -1),
        local_positions[..., None],
    ).squeeze(-1)

    velocity_xyz = motion_272.new_zeros((bsz, seq_len, 3))
    velocity_xyz[:, :, 0] = root_velocity[:, :, 0]
    velocity_xyz[:, :, 2] = root_velocity[:, :, 1]
    if seq_len > 1:
        velocity_xyz[:, 1:] = torch.matmul(inv_heading[:, :-1], velocity_xyz[:, 1:, :, None]).squeeze(-1)
    root_translation = torch.cumsum(velocity_xyz, dim=1)
    positions[:, :, :, 0] = positions[:, :, :, 0] + root_translation[:, :, None, 0]
    positions[:, :, :, 2] = positions[:, :, :, 2] + root_translation[:, :, None, 2]
    return positions


def _frame_joint_errors_cm(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    cfg: CodeFlowInpaintEvalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_raw.shape[-1] == 272 and target_raw.shape[-1] == 272:
        pred_joints = _recover_motionstreamer272_positions(pred_raw, int(cfg.joints_num))
        target_joints = _recover_motionstreamer272_positions(target_raw, int(cfg.joints_num))
    else:
        pred_joints = recover_from_ric(pred_raw, int(cfg.joints_num))
        target_joints = recover_from_ric(target_raw, int(cfg.joints_num))

    pred_centered = pred_joints - pred_joints[:, :, [0]]
    target_centered = target_joints - target_joints[:, :, [0]]
    flat_pred = pred_centered.reshape(-1, pred_centered.shape[-2], 3)
    flat_target = target_centered.reshape(-1, target_centered.shape[-2], 3)
    mpjpe = calculate_mpjpe(
        flat_target,
        flat_pred,
    ).view(target_joints.shape[0], target_joints.shape[1]) * 100.0
    pa_mpjpe = calculate_pa_mpjpe(
        flat_target,
        flat_pred,
    ).view(target_joints.shape[0], target_joints.shape[1]) * 100.0
    return mpjpe, pa_mpjpe


def _token_frame_mask(
    token_mask: torch.Tensor,
    seq_len: int,
    unit_length: int,
    frame_lengths: torch.Tensor,
) -> torch.Tensor:
    unit_length = max(int(unit_length), 1)
    repeated = token_mask.to(dtype=torch.bool).repeat_interleave(unit_length, dim=1)
    out = torch.zeros((token_mask.shape[0], int(seq_len)), device=token_mask.device, dtype=torch.bool)
    copy_len = min(int(seq_len), repeated.shape[1])
    out[:, :copy_len] = repeated[:, :copy_len]
    frame_ids = torch.arange(int(seq_len), device=token_mask.device).view(1, -1)
    valid = frame_ids < frame_lengths.to(device=token_mask.device, dtype=torch.long).view(-1, 1)
    return out & valid


def _frame_masks_from_preserve_tokens(
    preserve_mask: torch.Tensor,
    token_lengths: torch.Tensor,
    frame_lengths: torch.Tensor,
    seq_len: int,
    unit_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unit_length = max(int(unit_length), 1)
    token_lengths = token_lengths.to(device=preserve_mask.device, dtype=torch.long).clamp(
        min=1,
        max=preserve_mask.shape[1],
    )
    effective_lengths = torch.minimum(
        frame_lengths.to(device=preserve_mask.device, dtype=torch.long),
        token_lengths * unit_length,
    )
    valid_token_mask = (
        torch.arange(preserve_mask.shape[1], device=preserve_mask.device).view(1, -1)
        < token_lengths.view(-1, 1)
    )
    preserved_token_mask = preserve_mask.all(dim=-1) & valid_token_mask
    preserved_frame_mask = _token_frame_mask(preserved_token_mask, seq_len, unit_length, effective_lengths)
    valid_frame_mask = _token_frame_mask(valid_token_mask, seq_len, unit_length, effective_lengths)
    generated_frame_mask = valid_frame_mask & ~preserved_frame_mask
    return preserved_frame_mask, generated_frame_mask, valid_frame_mask


def _init_pose_sums() -> Dict[str, float]:
    return {
        "mpjpe_sum": 0.0,
        "mpjpe_frames": 0.0,
        "pa_mpjpe_sum": 0.0,
        "pa_mpjpe_frames": 0.0,
        "generated_mpjpe_sum": 0.0,
        "generated_mpjpe_frames": 0.0,
        "generated_pa_mpjpe_sum": 0.0,
        "generated_pa_mpjpe_frames": 0.0,
        "preserved_mpjpe_sum": 0.0,
        "preserved_mpjpe_frames": 0.0,
        "preserved_pa_mpjpe_sum": 0.0,
        "preserved_pa_mpjpe_frames": 0.0,
    }


def _update_pose_sums(
    sums: Dict[str, float],
    pred_motion_vq: torch.Tensor,
    target_motion_eval: torch.Tensor,
    preserve_mask: torch.Tensor,
    token_lengths: torch.Tensor,
    frame_lengths: torch.Tensor,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowInpaintEvalConfig,
) -> None:
    pred_raw = _motion_from_vq_to_raw(pred_motion_vq, target_motion_eval, vq_mean, vq_std)
    target_raw = _motion_from_eval_to_raw(target_motion_eval, eval_mean, eval_std)

    seq_len = target_motion_eval.shape[1]
    preserved_frame_mask, generated_frame_mask, valid_frame_mask = _frame_masks_from_preserve_tokens(
        preserve_mask=preserve_mask,
        token_lengths=token_lengths,
        frame_lengths=frame_lengths,
        seq_len=seq_len,
        unit_length=int(cfg.unit_length),
    )
    pred_raw = torch.where(preserved_frame_mask[:, :, None].to(pred_raw.device), target_raw, pred_raw)
    frame_errors, pa_frame_errors = _frame_joint_errors_cm(pred_raw, target_raw, cfg)

    for name, mask in (
        ("mpjpe", valid_frame_mask),
        ("generated_mpjpe", generated_frame_mask),
        ("preserved_mpjpe", preserved_frame_mask),
    ):
        count = float(mask.sum().detach().cpu())
        if count <= 0.0:
            continue
        sums[f"{name}_sum"] += float((frame_errors * mask.to(frame_errors.dtype)).sum().detach().cpu())
        sums[f"{name}_frames"] += count
        pa_name = name.replace("mpjpe", "pa_mpjpe")
        sums[f"{pa_name}_sum"] += float((pa_frame_errors * mask.to(pa_frame_errors.dtype)).sum().detach().cpu())
        sums[f"{pa_name}_frames"] += count


def _finalize_pose_sums(sums: Dict[str, float]) -> Dict[str, float]:
    mpjpe = sums["mpjpe_sum"] / max(sums["mpjpe_frames"], 1.0)
    pa_mpjpe = sums["pa_mpjpe_sum"] / max(sums["pa_mpjpe_frames"], 1.0)
    generated_mpjpe = sums["generated_mpjpe_sum"] / max(sums["generated_mpjpe_frames"], 1.0)
    generated_pa_mpjpe = sums["generated_pa_mpjpe_sum"] / max(sums["generated_pa_mpjpe_frames"], 1.0)
    preserved_mpjpe = sums["preserved_mpjpe_sum"] / max(sums["preserved_mpjpe_frames"], 1.0)
    preserved_pa_mpjpe = sums["preserved_pa_mpjpe_sum"] / max(sums["preserved_pa_mpjpe_frames"], 1.0)
    return {
        "mpjpe": mpjpe,
        "mpjpe_cm": mpjpe,
        "pa_mpjpe": pa_mpjpe,
        "pa_mpjpe_cm": pa_mpjpe,
        "p_mpjpe": preserved_mpjpe,
        "p_mpjpe_cm": preserved_mpjpe,
        "pmpjpe": preserved_mpjpe,
        "generated_mpjpe": generated_mpjpe,
        "generated_mpjpe_cm": generated_mpjpe,
        "generated_pa_mpjpe": generated_pa_mpjpe,
        "generated_pa_mpjpe_cm": generated_pa_mpjpe,
        "preserved_mpjpe": preserved_mpjpe,
        "preserved_mpjpe_cm": preserved_mpjpe,
        "preserved_pa_mpjpe": preserved_pa_mpjpe,
        "preserved_pa_mpjpe_cm": preserved_pa_mpjpe,
        "mpjpe_frames": int(sums["mpjpe_frames"]),
        "pa_mpjpe_frames": int(sums["pa_mpjpe_frames"]),
        "generated_mpjpe_frames": int(sums["generated_mpjpe_frames"]),
        "generated_pa_mpjpe_frames": int(sums["generated_pa_mpjpe_frames"]),
        "preserved_mpjpe_frames": int(sums["preserved_mpjpe_frames"]),
        "preserved_pa_mpjpe_frames": int(sums["preserved_pa_mpjpe_frames"]),
    }


def _finalize_eval_metrics(
    real_motion_embeddings: List[torch.Tensor],
    pred_motion_embeddings: List[torch.Tensor],
    r_precision_real_sum: np.ndarray,
    r_precision_sum: np.ndarray,
    matching_real_sum: float,
    matching_pred_sum: float,
    nb_sample: int,
    cfg: CodeFlowInpaintEvalConfig,
) -> Dict[str, float]:
    real_np = torch.cat(real_motion_embeddings, dim=0).cpu().numpy()
    pred_np = torch.cat(pred_motion_embeddings, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(real_np)
    mu, cov = calculate_activation_statistics(pred_np)
    if nb_sample > 300:
        diversity_times = 300
    elif nb_sample > 100:
        diversity_times = 100
    elif cfg.allow_small_eval and nb_sample > 1:
        diversity_times = nb_sample - 1
    else:
        raise RuntimeError(
            f"Inpainting eval needs more than 100 samples for diversity; got nb_sample={nb_sample}. "
            "Use more batches or allow_small_eval only for debugging."
        )
    return {
        "fid": float(calculate_frechet_distance(gt_mu, gt_cov, mu, cov)),
        "diversity_real": float(calculate_diversity(real_np, diversity_times)),
        "diversity": float(calculate_diversity(pred_np, diversity_times)),
        "r_precision_real": (r_precision_real_sum / float(nb_sample)).astype(float).tolist(),
        "r_precision": (r_precision_sum / float(nb_sample)).astype(float).tolist(),
        "top1": float(r_precision_sum[0] / float(nb_sample)),
        "top2": float(r_precision_sum[1] / float(nb_sample)),
        "top3": float(r_precision_sum[2] / float(nb_sample)),
        "matching_score_real": float(matching_real_sum / float(nb_sample)),
        "matching_score": float(matching_pred_sum / float(nb_sample)),
        "nb_sample": int(nb_sample),
    }


@torch.no_grad()
def evaluate_codeflow_inpainting(
    loader,
    model,
    eval_wrapper,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowInpaintEvalConfig,
    repeat_id: int = 0,
) -> Dict[str, object]:
    model.eval()
    real_motion_embeddings: List[torch.Tensor] = []
    pred_motion_embeddings: List[torch.Tensor] = []
    r_precision_real = np.zeros(3, dtype=np.float64)
    r_precision = np.zeros(3, dtype=np.float64)
    matching_score_real = 0.0
    matching_score_pred = 0.0
    nb_sample = 0
    code_sums = {
        "generated_parts": 0.0,
        "generated_correct": 0.0,
        "generated_dist_sum": 0.0,
        "generated_rank_sum": 0.0,
        "generated_severe": 0.0,
        "generated_denom_debug": 0.0,
        "preserved_parts": 0.0,
        "preserved_correct": 0.0,
        "preserved_dist_sum": 0.0,
        "preserved_rank_sum": 0.0,
        "preserved_severe": 0.0,
        "preserved_denom_debug": 0.0,
    }
    pose_sums = _init_pose_sums()

    for batch_id, batch in enumerate(loader):
        if cfg.max_batches > 0 and batch_id >= cfg.max_batches:
            break
        word_embeddings, pos_one_hots, captions, sent_len, pose, m_length, _tokens = batch
        pose = pose.to(model.device).float()
        m_length = m_length.to(model.device).long()
        bsz = pose.shape[0]
        gt_vq_motion = eval_motion_to_vq_space(pose, m_length, vq_mean, vq_std, eval_mean, eval_std)
        target_ids, target_embeddings = model.tokenizer.encode(gt_vq_motion)
        token_lengths = (m_length.to(model.device).long() // int(cfg.unit_length)).clamp(
            min=1,
            max=target_embeddings.shape[1],
        )
        preserve_mask, task_ids = build_eval_preserve_mask(
            token_lengths,
            target_embeddings.shape[1],
            target_embeddings.shape[2],
            cfg,
        )
        source_embeddings = torch.where(
            preserve_mask[:, :, :, None],
            target_embeddings,
            torch.zeros_like(target_embeddings),
        )
        pred_motion_vq, pred_ids = model.generate_inpaint_motion(
            captions,
            source_embeddings=source_embeddings,
            preserve_mask=preserve_mask,
            token_lengths=token_lengths,
            task_ids=task_ids,
            steps=int(cfg.steps),
            cond_scale=float(cfg.cond_scale),
            terminal_mode=cfg.terminal_mode,
            decode_mode=cfg.decode_mode,
        )
        pred_motion_eval = prepare_codeflow_motion_for_eval(
            decoded_motion=pred_motion_vq,
            reference_motion=pose,
            lengths=m_length,
            vq_mean=vq_mean,
            vq_std=vq_std,
            eval_mean=eval_mean,
            eval_std=eval_std,
        )
        preserved_frame_mask, _generated_frame_mask, _valid_frame_mask = _frame_masks_from_preserve_tokens(
            preserve_mask=preserve_mask,
            token_lengths=token_lengths,
            frame_lengths=m_length,
            seq_len=pose.shape[1],
            unit_length=int(cfg.unit_length),
        )
        pred_motion_eval = torch.where(
            preserved_frame_mask[:, :, None].to(device=pred_motion_eval.device),
            pose,
            pred_motion_eval,
        )

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings,
            pos_one_hots,
            sent_len,
            pred_motion_eval.clone(),
            m_length,
        )
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        real_motion_embeddings.append(em)
        pred_motion_embeddings.append(em_pred)

        et_cpu = et.cpu().numpy()
        em_cpu = em.cpu().numpy()
        et_pred_cpu = et_pred.cpu().numpy()
        em_pred_cpu = em_pred.cpu().numpy()
        r_precision_real += calculate_R_precision(et_cpu, em_cpu, top_k=3, sum_all=True)
        r_precision += calculate_R_precision(et_pred_cpu, em_pred_cpu, top_k=3, sum_all=True)
        matching_score_real += float(calculate_matching_score(et_cpu, em_cpu, sum_all=True))
        matching_score_pred += float(calculate_matching_score(et_pred_cpu, em_pred_cpu, sum_all=True))
        nb_sample += bsz

        _update_code_sums(code_sums, model.tokenizer, target_ids, pred_ids, preserve_mask, token_lengths, cfg)
        _update_pose_sums(
            pose_sums,
            pred_motion_vq,
            pose,
            preserve_mask,
            token_lengths,
            m_length,
            vq_mean,
            vq_std,
            eval_mean,
            eval_std,
            cfg,
        )

    if nb_sample == 0:
        raise RuntimeError("Inpainting eval loader produced zero samples")

    metrics = _finalize_eval_metrics(
        real_motion_embeddings,
        pred_motion_embeddings,
        r_precision_real,
        r_precision,
        matching_score_real,
        matching_score_pred,
        nb_sample,
        cfg,
    )
    metrics.update(_finalize_code_sums(code_sums))
    metrics.update(_finalize_pose_sums(pose_sums))
    print(
        f"--> \t CodeFlow Inpaint Eval Repeat {repeat_id} mode={cfg.mode}: "
        f"paper_task={paper_name_for_mode(cfg.mode)}, "
        f"FID. {metrics['fid']:.4f}, R_precision. {metrics['r_precision']}, "
        f"matching_score_pred. {metrics['matching_score']:.4f}, "
        f"MPJPE_cm. {metrics['mpjpe']:.4f}, PA-MPJPE_cm. {metrics['pa_mpjpe']:.4f}, "
        f"[P]-MPJPE_cm. {metrics['p_mpjpe']:.4f}, "
        f"generated_acc. {metrics['generated_code_acc']:.4f}, "
        f"preserved_acc. {metrics['preserved_code_acc']:.4f}"
    )
    return metrics


@torch.no_grad()
def evaluate_codeflow_inpainting_proxy(
    loader,
    model,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowInpaintEvalConfig,
    repeat_id: int = 0,
) -> Dict[str, object]:
    """Lightweight online inpainting eval: generated-region code metrics only."""
    model.eval()
    nb_sample = 0
    code_sums = {
        "generated_parts": 0.0,
        "generated_correct": 0.0,
        "generated_dist_sum": 0.0,
        "generated_rank_sum": 0.0,
        "generated_severe": 0.0,
        "generated_denom_debug": 0.0,
        "preserved_parts": 0.0,
        "preserved_correct": 0.0,
        "preserved_dist_sum": 0.0,
        "preserved_rank_sum": 0.0,
        "preserved_severe": 0.0,
        "preserved_denom_debug": 0.0,
    }
    pose_sums = _init_pose_sums()

    for batch_id, batch in enumerate(loader):
        if cfg.max_batches > 0 and batch_id >= cfg.max_batches:
            break
        _word_embeddings, _pos_one_hots, captions, _sent_len, pose, m_length, _tokens = batch
        pose = pose.to(model.device).float()
        m_length = m_length.to(model.device).long()
        bsz = pose.shape[0]
        gt_vq_motion = eval_motion_to_vq_space(pose, m_length, vq_mean, vq_std, eval_mean, eval_std)
        target_ids, target_embeddings = model.tokenizer.encode(gt_vq_motion)
        token_lengths = (m_length.to(model.device).long() // int(cfg.unit_length)).clamp(
            min=1,
            max=target_embeddings.shape[1],
        )
        preserve_mask, task_ids = build_eval_preserve_mask(
            token_lengths,
            target_embeddings.shape[1],
            target_embeddings.shape[2],
            cfg,
        )
        source_embeddings = torch.where(
            preserve_mask[:, :, :, None],
            target_embeddings,
            torch.zeros_like(target_embeddings),
        )
        pred_motion_vq, pred_ids = model.generate_inpaint_motion(
            captions,
            source_embeddings=source_embeddings,
            preserve_mask=preserve_mask,
            token_lengths=token_lengths,
            task_ids=task_ids,
            steps=int(cfg.steps),
            cond_scale=float(cfg.cond_scale),
            terminal_mode=cfg.terminal_mode,
            decode_mode=cfg.decode_mode,
        )
        _update_code_sums(code_sums, model.tokenizer, target_ids, pred_ids, preserve_mask, token_lengths, cfg)
        _update_pose_sums(
            pose_sums,
            pred_motion_vq,
            pose,
            preserve_mask,
            token_lengths,
            m_length,
            vq_mean,
            vq_std,
            eval_mean,
            eval_std,
            cfg,
        )
        nb_sample += bsz

    if nb_sample == 0:
        raise RuntimeError("Inpainting proxy eval loader produced zero samples")

    metrics = _finalize_code_sums(code_sums)
    metrics.update(_finalize_pose_sums(pose_sums))
    metrics.update({
        "nb_sample": int(nb_sample),
        "metric_set": "proxy",
    })
    print(
        f"--> \t CodeFlow Inpaint Proxy Repeat {repeat_id} mode={cfg.mode}: "
        f"paper_task={paper_name_for_mode(cfg.mode)}, "
        f"generated_acc. {metrics['generated_code_acc']:.4f}, "
        f"generated_dist. {metrics['generated_geom_code_dist']:.4f}, "
        f"generated_severe. {metrics['generated_geom_severe_rate']:.4f}, "
        f"MPJPE_cm. {metrics['mpjpe']:.4f}, PA-MPJPE_cm. {metrics['pa_mpjpe']:.4f}, "
        f"[P]-MPJPE_cm. {metrics['p_mpjpe']:.4f}, "
        f"preserved_acc. {metrics['preserved_code_acc']:.4f}"
    )
    return metrics
