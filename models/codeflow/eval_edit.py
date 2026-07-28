"""Global edit evaluation for PartGrid CodeFlow models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import (
    calculate_R_precision,
    calculate_matching_score,
)

from .eval_t2m import (
    _accumulate_code_metrics,
    _finalize_code_metrics,
    eval_motion_to_vq_space,
    prepare_codeflow_motion_for_eval,
)
from .edit_masks import build_instruction_edit_preserve_mask


@dataclass
class CodeFlowEditEvalConfig:
    steps: int = 96
    cond_scale: float = 6.0
    terminal_mode: Optional[str] = None
    decode_mode: Optional[str] = None
    unit_length: int = 4
    max_batches: int = 0
    allow_small_eval: bool = False
    include_code_metrics: bool = True
    geometry_severe_quantile: float = 0.75
    mask_mode: str = "instruction"
    mask_temporal_dilate: int = 0
    retrieval_backend: str = "motion_embedding"
    retrieval_batch_size: int = 32


def _retrieval_ranks(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    if query.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Motion retrieval expects 2D arrays, got {query.shape} and {target.shape}")
    if query.shape != target.shape:
        raise ValueError(f"Motion retrieval expects matched query/target shapes, got {query.shape} and {target.shape}")
    count = int(query.shape[0])
    if count <= 0:
        return np.zeros((0,), dtype=np.float64)
    query_f = query.astype(np.float64, copy=False)
    target_f = target.astype(np.float64, copy=False)
    dist = (
        np.sum(query_f * query_f, axis=1, keepdims=True)
        + np.sum(target_f * target_f, axis=1, keepdims=True).T
        - 2.0 * np.matmul(query_f, target_f.T)
    )
    order = np.argsort(dist, axis=1)
    gt = np.arange(count)
    ranks = np.empty((count,), dtype=np.float64)
    for idx in range(count):
        ranks[idx] = float(np.nonzero(order[idx] == gt[idx])[0][0] + 1)
    return ranks


def _metrics_from_ranks(ranks: np.ndarray, prefix: str) -> Dict[str, float]:
    if ranks.size == 0:
        return {
            f"{prefix}_r1": 0.0,
            f"{prefix}_r2": 0.0,
            f"{prefix}_r3": 0.0,
            f"{prefix}_avgr": float("nan"),
        }
    return {
        f"{prefix}_r1": float(np.mean(ranks <= 1)),
        f"{prefix}_r2": float(np.mean(ranks <= 2)),
        f"{prefix}_r3": float(np.mean(ranks <= 3)),
        f"{prefix}_avgr": float(np.mean(ranks)),
    }


def _calculate_motion_retrieval_metrics(
    query: np.ndarray,
    target: np.ndarray,
    *,
    prefix: str,
    batch_size: int = 32,
) -> Dict[str, float]:
    full_ranks = _retrieval_ranks(query, target)
    out = _metrics_from_ranks(full_ranks, f"{prefix}_full")

    batch_size = max(1, int(batch_size))
    batch_ranks = []
    for start in range(0, int(query.shape[0]), batch_size):
        end = min(start + batch_size, int(query.shape[0]))
        batch_ranks.append(_retrieval_ranks(query[start:end], target[start:end]))
    if batch_ranks:
        batch_ranks_np = np.concatenate(batch_ranks, axis=0)
    else:
        batch_ranks_np = np.zeros((0,), dtype=np.float64)
    out.update(_metrics_from_ranks(batch_ranks_np, f"{prefix}_batch"))
    return out


def _pad_motion_batches(arrays: List[np.ndarray], feature_dim: int = 272) -> np.ndarray:
    max_len = max(arr.shape[1] for arr in arrays)
    out = np.zeros((sum(arr.shape[0] for arr in arrays), max_len, feature_dim), dtype=np.float32)
    offset = 0
    for arr in arrays:
        count, cur_len, _ = arr.shape
        out[offset : offset + count, :cur_len] = arr
        offset += count
    return out


def _norm_to_tensor(value, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=device, dtype=dtype)
    if value.dim() == 1:
        value = value.view(1, 1, -1)
    return value


def _eval_motion_to_raw(eval_motion: torch.Tensor, eval_mean: torch.Tensor, eval_std: torch.Tensor) -> torch.Tensor:
    eval_mean = _norm_to_tensor(eval_mean, eval_motion.device, eval_motion.dtype)
    eval_std = _norm_to_tensor(eval_std, eval_motion.device, eval_motion.dtype)
    return eval_motion * eval_std + eval_mean


def _decoded_vq_motion_to_raw(
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


def _metrics_from_motionstreamer_embeddings(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    pred_emb: np.ndarray,
    nb_sample: int,
    cfg: CodeFlowEditEvalConfig,
) -> Dict[str, float]:
    source_target_l2_values = np.linalg.norm(source_emb - target_emb, axis=1)
    pred_target_l2_values = np.linalg.norm(pred_emb - target_emb, axis=1)
    pred_source_l2_values = np.linalg.norm(pred_emb - source_emb, axis=1)
    target_delta = target_emb - source_emb
    pred_delta = pred_emb - source_emb
    target_delta_norm = np.maximum(np.linalg.norm(target_delta, axis=1), 1e-8)
    pred_delta_norm = np.linalg.norm(pred_delta, axis=1)
    delta_cos = np.sum(pred_delta * target_delta, axis=1) / np.maximum(pred_delta_norm * target_delta_norm, 1e-8)

    retrieval_batch_size = int(getattr(cfg, "retrieval_batch_size", 32))
    generated_to_target = _calculate_motion_retrieval_metrics(
        pred_emb,
        target_emb,
        prefix="g2t",
        batch_size=retrieval_batch_size,
    )
    generated_to_source = _calculate_motion_retrieval_metrics(
        pred_emb,
        source_emb,
        prefix="g2s",
        batch_size=retrieval_batch_size,
    )
    source_to_target = _calculate_motion_retrieval_metrics(
        source_emb,
        target_emb,
        prefix="s2t",
        batch_size=retrieval_batch_size,
    )

    source_target_l2 = float(source_target_l2_values.mean())
    pred_target_l2 = float(pred_target_l2_values.mean())
    metrics = {
        "source_target_emb_l2": source_target_l2,
        "pred_target_emb_l2": pred_target_l2,
        "pred_source_emb_l2": float(pred_source_l2_values.mean()),
        "pred_target_minus_source_target_l2": pred_target_l2 - source_target_l2,
        "pred_target_over_source_target_l2": pred_target_l2 / max(source_target_l2, 1e-8),
        "edit_delta_emb_l2": float(np.linalg.norm(pred_delta - target_delta, axis=1).mean()),
        "edit_delta_emb_cos": float(delta_cos.mean()),
        "edit_delta_mag_ratio": float((pred_delta_norm / target_delta_norm).mean()),
        "m2m_pred_target_r1": generated_to_target["g2t_full_r1"],
        "m2m_pred_target_r2": generated_to_target["g2t_full_r2"],
        "m2m_pred_target_r3": generated_to_target["g2t_full_r3"],
        "m2m_source_target_r1": source_to_target["s2t_full_r1"],
        "m2m_source_target_r2": source_to_target["s2t_full_r2"],
        "m2m_source_target_r3": source_to_target["s2t_full_r3"],
        "nb_sample": int(nb_sample),
        "retrieval_batch_size": retrieval_batch_size,
        "eval_backend": "motionstreamer272",
    }
    metrics.update(generated_to_target)
    metrics.update(generated_to_source)
    metrics.update(source_to_target)
    return metrics


def _finalize_edit_metrics(
    source_motion_embeddings: List[torch.Tensor],
    target_motion_embeddings: List[torch.Tensor],
    pred_motion_embeddings: List[torch.Tensor],
    r_precision_target_sum: np.ndarray,
    r_precision_pred_sum: np.ndarray,
    matching_target_sum: float,
    matching_pred_sum: float,
    source_target_l2_sum: float,
    pred_target_l2_sum: float,
    pred_source_l2_sum: float,
    delta_l2_sum: float,
    delta_cos_sum: float,
    delta_mag_ratio_sum: float,
    nb_sample: int,
    cfg: CodeFlowEditEvalConfig,
) -> Dict[str, float]:
    source_np = torch.cat(source_motion_embeddings, dim=0).cpu().numpy()
    target_np = torch.cat(target_motion_embeddings, dim=0).cpu().numpy()
    pred_np = torch.cat(pred_motion_embeddings, dim=0).cpu().numpy()
    source_target_l2 = float(source_target_l2_sum / float(nb_sample))
    pred_target_l2 = float(pred_target_l2_sum / float(nb_sample))
    pred_source_l2 = float(pred_source_l2_sum / float(nb_sample))
    retrieval_batch_size = int(getattr(cfg, "retrieval_batch_size", 32))
    generated_to_target = _calculate_motion_retrieval_metrics(
        pred_np,
        target_np,
        prefix="g2t",
        batch_size=retrieval_batch_size,
    )
    generated_to_source = _calculate_motion_retrieval_metrics(
        pred_np,
        source_np,
        prefix="g2s",
        batch_size=retrieval_batch_size,
    )
    source_to_target = _calculate_motion_retrieval_metrics(
        source_np,
        target_np,
        prefix="s2t",
        batch_size=retrieval_batch_size,
    )
    metrics = {
        "source_target_emb_l2": source_target_l2,
        "pred_target_emb_l2": pred_target_l2,
        "pred_source_emb_l2": pred_source_l2,
        "pred_target_minus_source_target_l2": pred_target_l2 - source_target_l2,
        "pred_target_over_source_target_l2": pred_target_l2 / max(source_target_l2, 1e-8),
        "edit_delta_emb_l2": float(delta_l2_sum / float(nb_sample)),
        "edit_delta_emb_cos": float(delta_cos_sum / float(nb_sample)),
        "edit_delta_mag_ratio": float(delta_mag_ratio_sum / float(nb_sample)),
        "m2m_pred_target_r1": generated_to_target["g2t_full_r1"],
        "m2m_pred_target_r2": generated_to_target["g2t_full_r2"],
        "m2m_pred_target_r3": generated_to_target["g2t_full_r3"],
        "m2m_source_target_r1": source_to_target["s2t_full_r1"],
        "m2m_source_target_r2": source_to_target["s2t_full_r2"],
        "m2m_source_target_r3": source_to_target["s2t_full_r3"],
        "nb_sample": int(nb_sample),
        "retrieval_batch_size": retrieval_batch_size,
    }
    metrics.update(generated_to_target)
    metrics.update(generated_to_source)
    metrics.update(source_to_target)
    return metrics


@torch.no_grad()
def _evaluate_codeflow_global_edit_motionstreamer272(
    loader,
    model,
    evaluator,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowEditEvalConfig,
    repeat_id: int = 0,
) -> Dict[str, object]:
    model.eval()
    source_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    pred_batches: List[np.ndarray] = []
    lengths_batches: List[np.ndarray] = []
    nb_sample = 0
    code_sums = {
        "code_valid_parts": 0.0,
        "code_correct_parts": 0.0,
        "code_wrong_parts": 0.0,
        "geom_code_dist_sum": 0.0,
        "geom_rank_pct_sum": 0.0,
        "geom_wrong_code_dist_sum": 0.0,
        "geom_wrong_rank_pct_sum": 0.0,
        "geom_severe_parts": 0.0,
    }
    mask_generated_frac_sum = 0.0
    mask_preserved_frac_sum = 0.0
    mask_batches = 0

    for batch_id, batch in enumerate(loader):
        if cfg.max_batches > 0 and batch_id >= cfg.max_batches:
            break
        _word_embeddings, _pos_one_hots, instructions, _sent_len, source_pose, target_pose, m_length, _tokens, *extra = batch
        del extra
        source_pose = source_pose.to(model.device).float()
        target_pose = target_pose.to(model.device).float()
        m_length = m_length.to(model.device).long()
        bsz = target_pose.shape[0]

        source_vq_motion = eval_motion_to_vq_space(source_pose, m_length, vq_mean, vq_std, eval_mean, eval_std)
        target_vq_motion = eval_motion_to_vq_space(target_pose, m_length, vq_mean, vq_std, eval_mean, eval_std)
        target_ids, target_embeddings = model.tokenizer.encode(target_vq_motion)
        _source_ids, source_embeddings = model.tokenizer.encode(source_vq_motion)
        token_lengths = (m_length.to(model.device).long() // int(cfg.unit_length)).clamp(
            min=1,
            max=target_embeddings.shape[1],
        )

        preserve_mask = None
        if str(cfg.mask_mode) == "instruction":
            preserve_mask, _op_ids, mask_stats = build_instruction_edit_preserve_mask(
                instructions,
                token_lengths=token_lengths,
                latent_len=source_embeddings.shape[1],
                num_parts=source_embeddings.shape[2],
                device=model.device,
                temporal_dilate=int(cfg.mask_temporal_dilate),
            )
            mask_generated_frac_sum += float(mask_stats["global_edit_infer_generated_cell_frac"].detach().cpu())
            mask_preserved_frac_sum += float(mask_stats["global_edit_infer_preserved_cell_frac"].detach().cpu())
            mask_batches += 1
        elif str(cfg.mask_mode) not in {"none", ""}:
            raise ValueError(f"Unsupported global edit eval mask mode: {cfg.mask_mode}")

        pred_motion_vq, pred_ids = model.generate_global_edit_motion(
            instructions,
            source_embeddings=source_embeddings,
            token_lengths=token_lengths,
            steps=int(cfg.steps),
            cond_scale=float(cfg.cond_scale),
            terminal_mode=cfg.terminal_mode,
            decode_mode=cfg.decode_mode,
            preserve_mask=preserve_mask,
        )

        source_raw = _eval_motion_to_raw(source_pose, eval_mean, eval_std)
        target_raw = _eval_motion_to_raw(target_pose, eval_mean, eval_std)
        pred_raw = _decoded_vq_motion_to_raw(pred_motion_vq, target_pose, vq_mean, vq_std)
        frame_ids = torch.arange(target_raw.shape[1], device=model.device).view(1, -1, 1)
        valid = frame_ids < m_length.view(-1, 1, 1)
        source_raw = torch.where(valid, source_raw, torch.zeros_like(source_raw))
        target_raw = torch.where(valid, target_raw, torch.zeros_like(target_raw))
        pred_raw = torch.where(valid, pred_raw, torch.zeros_like(pred_raw))

        source_batches.append(source_raw.detach().cpu().numpy().astype(np.float32))
        target_batches.append(target_raw.detach().cpu().numpy().astype(np.float32))
        pred_batches.append(pred_raw.detach().cpu().numpy().astype(np.float32))
        lengths_batches.append(m_length.detach().cpu().numpy().astype(np.int64))
        nb_sample += bsz

        if cfg.include_code_metrics:
            _accumulate_code_metrics(code_sums, model.tokenizer, target_ids, pred_ids, token_lengths, cfg)

    if nb_sample == 0:
        raise RuntimeError("Global edit eval loader produced zero samples")

    lengths_np = np.concatenate(lengths_batches, axis=0)
    source_np = _pad_motion_batches(source_batches, feature_dim=272)
    target_np = _pad_motion_batches(target_batches, feature_dim=272)
    pred_np = _pad_motion_batches(pred_batches, feature_dim=272)
    batch_size = int(getattr(cfg, "retrieval_batch_size", 32))
    source_emb = evaluator.encode_motion(source_np, lengths_np, batch_size=batch_size)
    target_emb = evaluator.encode_motion(target_np, lengths_np, batch_size=batch_size)
    pred_emb = evaluator.encode_motion(pred_np, lengths_np, batch_size=batch_size)

    metrics = _metrics_from_motionstreamer_embeddings(source_emb, target_emb, pred_emb, nb_sample, cfg)
    if cfg.include_code_metrics:
        metrics.update(_finalize_code_metrics(code_sums))
    if mask_batches > 0:
        metrics["mask_generated_cell_frac"] = float(mask_generated_frac_sum / float(mask_batches))
        metrics["mask_preserved_cell_frac"] = float(mask_preserved_frac_sum / float(mask_batches))
    line = (
        f"--> \t CodeFlow Global Edit MotionStreamer272 Eval Repeat {repeat_id}: "
        f"g2t_batch_r1/r2/r3. {metrics['g2t_batch_r1']:.4f}/"
        f"{metrics['g2t_batch_r2']:.4f}/{metrics['g2t_batch_r3']:.4f}, "
        f"g2t_batch_avgr. {metrics['g2t_batch_avgr']:.2f}, "
        f"g2t_full_r1/r2/r3. {metrics['g2t_full_r1']:.4f}/"
        f"{metrics['g2t_full_r2']:.4f}/{metrics['g2t_full_r3']:.4f}, "
        f"g2t_full_avgr. {metrics['g2t_full_avgr']:.2f}"
    )
    print(line)
    return metrics


@torch.no_grad()
def evaluate_codeflow_global_edit(
    loader,
    model,
    eval_wrapper,
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    eval_mean: torch.Tensor,
    eval_std: torch.Tensor,
    cfg: CodeFlowEditEvalConfig,
    repeat_id: int = 0,
) -> Dict[str, object]:
    if not hasattr(model, "generate_global_edit_motion"):
        raise RuntimeError("Model does not implement generate_global_edit_motion")
    model.eval()
    if isinstance(eval_wrapper, dict) and eval_wrapper.get("backend") == "motionstreamer272":
        return _evaluate_codeflow_global_edit_motionstreamer272(
            loader=loader,
            model=model,
            evaluator=eval_wrapper["evaluator"],
            vq_mean=vq_mean,
            vq_std=vq_std,
            eval_mean=eval_mean,
            eval_std=eval_std,
            cfg=cfg,
            repeat_id=repeat_id,
        )
    if isinstance(eval_wrapper, dict) and eval_wrapper.get("backend") == "motionfix_official":
        from .eval_motionfix_official import evaluate_codeflow_global_edit_motionfix_official

        return evaluate_codeflow_global_edit_motionfix_official(
            loader=loader,
            model=model,
            official_wrapper=eval_wrapper,
            vq_mean=vq_mean,
            vq_std=vq_std,
            cfg=cfg,
            repeat_id=repeat_id,
        )

    retrieval_backend = str(getattr(cfg, "retrieval_backend", "none") or "none")
    if retrieval_backend in {"motionstreamer272", "motionfix_official"}:
        raise RuntimeError(f"{retrieval_backend} edit eval requires its evaluator component")
    if retrieval_backend not in {"none", "motion_embedding"}:
        raise ValueError(f"Unsupported global edit retrieval backend: {retrieval_backend}")

    source_motion_embeddings: List[torch.Tensor] = []
    target_motion_embeddings: List[torch.Tensor] = []
    pred_motion_embeddings: List[torch.Tensor] = []
    r_precision_target = np.zeros(3, dtype=np.float64)
    r_precision_pred = np.zeros(3, dtype=np.float64)
    matching_score_target = 0.0
    matching_score_pred = 0.0
    source_target_l2_sum = 0.0
    pred_target_l2_sum = 0.0
    pred_source_l2_sum = 0.0
    delta_l2_sum = 0.0
    delta_cos_sum = 0.0
    delta_mag_ratio_sum = 0.0
    nb_sample = 0
    code_sums = {
        "code_valid_parts": 0.0,
        "code_correct_parts": 0.0,
        "code_wrong_parts": 0.0,
        "geom_code_dist_sum": 0.0,
        "geom_rank_pct_sum": 0.0,
        "geom_wrong_code_dist_sum": 0.0,
        "geom_wrong_rank_pct_sum": 0.0,
        "geom_severe_parts": 0.0,
    }
    mask_generated_frac_sum = 0.0
    mask_preserved_frac_sum = 0.0
    mask_batches = 0

    for batch_id, batch in enumerate(loader):
        if cfg.max_batches > 0 and batch_id >= cfg.max_batches:
            break
        word_embeddings, pos_one_hots, instructions, sent_len, source_pose, target_pose, m_length, _tokens, *extra = batch
        sample_ids = [str(item) for item in extra[0]] if extra else []
        cap_order = torch.argsort(sent_len, descending=True)
        word_embeddings = word_embeddings[cap_order]
        pos_one_hots = pos_one_hots[cap_order]
        sent_len = sent_len[cap_order]
        source_pose = source_pose[cap_order]
        target_pose = target_pose[cap_order]
        m_length = m_length[cap_order]
        instructions = [instructions[idx] for idx in cap_order.tolist()]
        if sample_ids:
            sample_ids = [sample_ids[idx] for idx in cap_order.tolist()]
        source_pose = source_pose.to(model.device).float()
        target_pose = target_pose.to(model.device).float()
        m_length = m_length.to(model.device).long()
        bsz = target_pose.shape[0]

        source_vq_motion = eval_motion_to_vq_space(source_pose, m_length, vq_mean, vq_std, eval_mean, eval_std)
        target_vq_motion = eval_motion_to_vq_space(target_pose, m_length, vq_mean, vq_std, eval_mean, eval_std)
        target_ids, target_embeddings = model.tokenizer.encode(target_vq_motion)
        _source_ids, source_embeddings = model.tokenizer.encode(source_vq_motion)
        token_lengths = (m_length.to(model.device).long() // int(cfg.unit_length)).clamp(
            min=1,
            max=target_embeddings.shape[1],
        )

        preserve_mask = None
        if str(cfg.mask_mode) == "instruction":
            preserve_mask, _op_ids, mask_stats = build_instruction_edit_preserve_mask(
                instructions,
                token_lengths=token_lengths,
                latent_len=source_embeddings.shape[1],
                num_parts=source_embeddings.shape[2],
                device=model.device,
                temporal_dilate=int(cfg.mask_temporal_dilate),
            )
            mask_generated_frac_sum += float(mask_stats["global_edit_infer_generated_cell_frac"].detach().cpu())
            mask_preserved_frac_sum += float(mask_stats["global_edit_infer_preserved_cell_frac"].detach().cpu())
            mask_batches += 1
        elif str(cfg.mask_mode) not in {"none", ""}:
            raise ValueError(f"Unsupported global edit eval mask mode: {cfg.mask_mode}")

        pred_motion_vq, pred_ids = model.generate_global_edit_motion(
            instructions,
            source_embeddings=source_embeddings,
            token_lengths=token_lengths,
            steps=int(cfg.steps),
            cond_scale=float(cfg.cond_scale),
            terminal_mode=cfg.terminal_mode,
            decode_mode=cfg.decode_mode,
            preserve_mask=preserve_mask,
        )
        pred_motion_eval = prepare_codeflow_motion_for_eval(
            decoded_motion=pred_motion_vq,
            reference_motion=target_pose,
            lengths=m_length,
            vq_mean=vq_mean,
            vq_std=vq_std,
            eval_mean=eval_mean,
            eval_std=eval_std,
        )
        et_target, em_target = eval_wrapper.get_co_embeddings(
            word_embeddings,
            pos_one_hots,
            sent_len,
            target_pose,
            m_length,
        )
        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings,
            pos_one_hots,
            sent_len,
            pred_motion_eval.clone(),
            m_length,
        )
        em_source = eval_wrapper.get_motion_embeddings(source_pose, m_length)

        source_motion_embeddings.append(em_source)
        target_motion_embeddings.append(em_target)
        pred_motion_embeddings.append(em_pred)

        et_target_cpu = et_target.cpu().numpy()
        em_target_cpu = em_target.cpu().numpy()
        et_pred_cpu = et_pred.cpu().numpy()
        em_pred_cpu = em_pred.cpu().numpy()
        r_precision_target += calculate_R_precision(et_target_cpu, em_target_cpu, top_k=3, sum_all=True)
        r_precision_pred += calculate_R_precision(et_pred_cpu, em_pred_cpu, top_k=3, sum_all=True)
        matching_score_target += float(calculate_matching_score(et_target_cpu, em_target_cpu, sum_all=True))
        matching_score_pred += float(calculate_matching_score(et_pred_cpu, em_pred_cpu, sum_all=True))
        source_target_l2_sum += float((em_source - em_target).square().sum(dim=-1).sqrt().sum().detach().cpu())
        pred_target_l2_sum += float((em_pred - em_target).square().sum(dim=-1).sqrt().sum().detach().cpu())
        pred_source_l2_sum += float((em_pred - em_source).square().sum(dim=-1).sqrt().sum().detach().cpu())
        target_delta = em_target - em_source
        pred_delta = em_pred - em_source
        target_delta_norm = target_delta.square().sum(dim=-1).sqrt().clamp_min(1e-8)
        pred_delta_norm = pred_delta.square().sum(dim=-1).sqrt()
        delta_l2_sum += float((pred_delta - target_delta).square().sum(dim=-1).sqrt().sum().detach().cpu())
        delta_cos_sum += float(F.cosine_similarity(pred_delta, target_delta, dim=-1, eps=1e-8).sum().detach().cpu())
        delta_mag_ratio_sum += float((pred_delta_norm / target_delta_norm).sum().detach().cpu())
        nb_sample += bsz

        if cfg.include_code_metrics:
            _accumulate_code_metrics(code_sums, model.tokenizer, target_ids, pred_ids, token_lengths, cfg)

    if nb_sample == 0:
        raise RuntimeError("Global edit eval loader produced zero samples")

    metrics = _finalize_edit_metrics(
        source_motion_embeddings,
        target_motion_embeddings,
        pred_motion_embeddings,
        r_precision_target,
        r_precision_pred,
        matching_score_target,
        matching_score_pred,
        source_target_l2_sum,
        pred_target_l2_sum,
        pred_source_l2_sum,
        delta_l2_sum,
        delta_cos_sum,
        delta_mag_ratio_sum,
        nb_sample,
        cfg,
    )
    if cfg.include_code_metrics:
        metrics.update(_finalize_code_metrics(code_sums))
    if mask_batches > 0:
        metrics["mask_generated_cell_frac"] = float(mask_generated_frac_sum / float(mask_batches))
        metrics["mask_preserved_cell_frac"] = float(mask_preserved_frac_sum / float(mask_batches))
    line = (
        f"--> \t CodeFlow Global Edit Eval Repeat {repeat_id}: "
        f"g2t_batch_r1/r2/r3. {metrics['g2t_batch_r1']:.4f}/"
        f"{metrics['g2t_batch_r2']:.4f}/{metrics['g2t_batch_r3']:.4f}, "
        f"g2t_batch_avgr. {metrics['g2t_batch_avgr']:.2f}, "
        f"g2t_full_r1/r2/r3. {metrics['g2t_full_r1']:.4f}/"
        f"{metrics['g2t_full_r2']:.4f}/{metrics['g2t_full_r3']:.4f}, "
        f"g2t_full_avgr. {metrics['g2t_full_avgr']:.2f}"
    )
    print(line)
    return metrics
