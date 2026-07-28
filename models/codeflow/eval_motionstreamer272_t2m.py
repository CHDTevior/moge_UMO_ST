"""MotionStreamer-272 T2M evaluation for CodeFlow."""

from __future__ import annotations

import codecs as cs
import sys
from os.path import join as pjoin
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils import data

from .eval_t2m import CodeFlowEvalConfig, _accumulate_code_metrics, _finalize_code_metrics


def _read_full_caption(text_path: str) -> str | None:
    if not Path(text_path).is_file():
        return None
    fallback = None
    with cs.open(text_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("#")
            if not parts or not parts[0]:
                continue
            caption = parts[0]
            fallback = fallback or caption
            if len(parts) >= 4:
                try:
                    f_tag = float(parts[2])
                    to_tag = float(parts[3])
                except ValueError:
                    f_tag, to_tag = 0.0, 0.0
                if np.isnan(f_tag):
                    f_tag = 0.0
                if np.isnan(to_tag):
                    to_tag = 0.0
                if f_tag == 0.0 and to_tag == 0.0:
                    return caption
    return fallback


class MotionStreamer272T2MEvalDataset(data.Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "test",
        *,
        unit_length: int = 4,
        max_motion_length: int = 300,
        max_samples: int = 0,
    ) -> None:
        self.data_root = data_root
        self.motion_dir = pjoin(data_root, "new_joint_vecs")
        self.text_dir = pjoin(data_root, "texts")
        self.unit_length = int(unit_length)
        self.max_motion_length = int(max_motion_length)

        split_file = pjoin(data_root, f"{split}.txt")
        if not Path(split_file).is_file():
            split_file = pjoin(data_root, "split", f"{split}.txt")
        if not Path(split_file).is_file():
            raise FileNotFoundError(f"Cannot find MotionStreamer272 split file for {split}: {split_file}")

        with cs.open(split_file, "r", encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]

        samples: List[Dict[str, Any]] = []
        for name in names:
            motion_path = pjoin(self.motion_dir, f"{name}.npy")
            caption = _read_full_caption(pjoin(self.text_dir, f"{name}.txt"))
            if caption is None or not Path(motion_path).is_file():
                continue
            try:
                motion = np.load(motion_path)
            except Exception:
                continue
            if motion.ndim != 2 or motion.shape[-1] != 272:
                continue
            length = min(int(motion.shape[0]), self.max_motion_length)
            length = (length // self.unit_length) * self.unit_length
            if length < self.unit_length:
                continue
            samples.append(
                {
                    "name": name,
                    "caption": caption,
                    "motion": motion[:length].astype(np.float32),
                    "length": length,
                }
            )
            if max_samples > 0 and len(samples) >= max_samples:
                break
        if not samples:
            raise RuntimeError(f"No valid MotionStreamer272 eval samples from {split_file}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]


def collate_motionstreamer272_t2m_eval(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch.sort(key=lambda item: item["length"], reverse=True)
    max_len = max(int(item["length"]) for item in batch)
    motions = np.zeros((len(batch), max_len, 272), dtype=np.float32)
    lengths = np.zeros((len(batch),), dtype=np.int64)
    captions: List[str] = []
    names: List[str] = []
    for idx, item in enumerate(batch):
        length = int(item["length"])
        motions[idx, :length] = item["motion"]
        lengths[idx] = length
        captions.append(str(item["caption"]))
        names.append(str(item["name"]))
    return {
        "motion": torch.from_numpy(motions),
        "length": torch.from_numpy(lengths),
        "caption": captions,
        "name": names,
    }


def build_motionstreamer272_t2m_loader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    *,
    unit_length: int = 4,
    max_motion_length: int = 300,
    max_samples: int = 0,
) -> data.DataLoader:
    if int(batch_size) != 32:
        raise ValueError(f"MotionStreamer272 FID/Top3 eval requires batch_size=32, got {batch_size}")
    dataset = MotionStreamer272T2MEvalDataset(
        data_root,
        split,
        unit_length=unit_length,
        max_motion_length=max_motion_length,
        max_samples=max_samples,
    )
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_motionstreamer272_t2m_eval,
        drop_last=False,
    )


def load_motionstreamer272_evaluator(
    *,
    hymotion_root: str,
    checkpoint: str,
    distilbert_path: str,
    mean_path: str,
    std_path: str,
    device: torch.device,
):
    if hymotion_root and hymotion_root not in sys.path:
        sys.path.insert(0, hymotion_root)
    from hymotion.eval.motionstreamer272 import MotionStreamer272Evaluator

    return MotionStreamer272Evaluator.from_checkpoint(
        checkpoint=checkpoint,
        distilbert_path=distilbert_path,
        mean_path=mean_path,
        std_path=std_path,
        device=device,
    )


def _pad_to_max(arrays: List[np.ndarray], feature_dim: int = 272) -> np.ndarray:
    max_len = max(arr.shape[1] for arr in arrays)
    out = np.zeros((sum(arr.shape[0] for arr in arrays), max_len, feature_dim), dtype=np.float32)
    offset = 0
    for arr in arrays:
        count, cur_len, _ = arr.shape
        out[offset : offset + count, :cur_len] = arr
        offset += count
    return out


def _batch_retrieval_metrics(text_embeddings: np.ndarray, motion_embeddings: np.ndarray, batch_size: int):
    from hymotion.eval.metrics import calculate_matching_score, calculate_r_precision

    r_sum = np.zeros((3,), dtype=np.float64)
    matching_sum = 0.0
    sample_count = 0
    for start in range(0, text_embeddings.shape[0], batch_size):
        text_batch = text_embeddings[start : start + batch_size]
        motion_batch = motion_embeddings[start : start + batch_size]
        if len(text_batch) < 2:
            continue
        top_k = min(3, len(text_batch))
        r = calculate_r_precision(text_batch, motion_batch, top_k=top_k, sum_all=True).astype(np.float64)
        if top_k < 3:
            r = np.pad(r, (0, 3 - top_k), constant_values=0.0)
        r_sum += r
        matching_sum += float(calculate_matching_score(text_batch, motion_batch, sum_all=True))
        sample_count += len(text_batch)
    if sample_count == 0:
        return np.full((3,), np.nan), float("nan")
    return r_sum / sample_count, matching_sum / sample_count


@torch.no_grad()
def evaluate_codeflow_t2m_motionstreamer272(
    *,
    loader,
    model,
    evaluator,
    vq_mean: np.ndarray,
    vq_std: np.ndarray,
    cfg: CodeFlowEvalConfig,
    retrieval_batch_size: int = 32,
    repeat_id: int = 0,
) -> Dict[str, object]:
    if int(retrieval_batch_size) != 32:
        raise ValueError(
            f"MotionStreamer272 retrieval metrics require retrieval_batch_size=32, got {retrieval_batch_size}"
        )
    if getattr(loader, "batch_size", None) != 32:
        raise ValueError(f"MotionStreamer272 FID/Top3 eval loader requires batch_size=32, got {loader.batch_size}")
    from hymotion.eval.metrics import (
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
    )

    del repeat_id
    model.eval()
    device = model.device
    mean_t = torch.from_numpy(vq_mean.astype(np.float32)).to(device).view(1, 1, -1)
    std_t = torch.from_numpy(vq_std.astype(np.float32)).to(device).view(1, 1, -1)

    captions: List[str] = []
    lengths_list: List[np.ndarray] = []
    gt_motions: List[np.ndarray] = []
    pred_motions: List[np.ndarray] = []
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

    for batch_id, batch in enumerate(loader):
        if cfg.max_batches > 0 and batch_id >= cfg.max_batches:
            break
        raw_motion = batch["motion"].to(device=device, dtype=torch.float32)
        lengths = batch["length"].to(device=device, dtype=torch.long)
        token_lengths = (lengths // int(cfg.unit_length)).clamp(min=1)
        pred_norm, pred_ids = model.generate_motion(
            batch["caption"],
            token_lengths=token_lengths,
            steps=int(cfg.steps),
            cond_scale=float(cfg.cond_scale),
            terminal_mode=cfg.terminal_mode,
            decode_mode=cfg.decode_mode,
        )
        pred_raw = pred_norm * std_t + mean_t
        padded_pred = torch.zeros_like(raw_motion)
        copy_len = min(raw_motion.shape[1], pred_raw.shape[1])
        padded_pred[:, :copy_len] = pred_raw[:, :copy_len]

        if cfg.include_code_metrics and str(cfg.metric_set) != "fid_top3":
            target_norm = (raw_motion - mean_t) / std_t
            target_ids, _target_embeddings = model.tokenizer.encode(target_norm)
            target_token_lengths = (lengths // int(cfg.unit_length)).clamp(min=1, max=target_ids.shape[1])
            _accumulate_code_metrics(
                code_sums,
                model.tokenizer,
                target_ids,
                pred_ids,
                target_token_lengths,
                cfg,
            )

        captions.extend(batch["caption"])
        lengths_list.append(lengths.detach().cpu().numpy().astype(np.int64))
        gt_motions.append(raw_motion.detach().cpu().numpy())
        pred_motions.append(padded_pred.detach().cpu().numpy())

    if not captions:
        raise RuntimeError("MotionStreamer272 eval collected no samples")

    lengths_np = np.concatenate(lengths_list, axis=0)
    gt_np = _pad_to_max(gt_motions)
    pred_np = _pad_to_max(pred_motions)

    text_emb = evaluator.encode_text(captions, batch_size=retrieval_batch_size)
    gt_emb = evaluator.encode_motion(gt_np, lengths_np, batch_size=retrieval_batch_size)
    pred_emb = evaluator.encode_motion(pred_np, lengths_np, batch_size=retrieval_batch_size)

    pred_r, pred_matching = _batch_retrieval_metrics(text_emb, pred_emb, retrieval_batch_size)
    gt_mu, gt_cov = calculate_activation_statistics(gt_emb)
    pred_mu, pred_cov = calculate_activation_statistics(pred_emb)

    metrics: Dict[str, object] = {
        "fid": float(calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)),
        "top3": float(pred_r[2]),
    }
    if str(cfg.metric_set) != "fid_top3":
        gt_r, gt_matching = _batch_retrieval_metrics(text_emb, gt_emb, retrieval_batch_size)
        rng = np.random.default_rng(1234)
        metrics.update({
            "top1": float(pred_r[0]),
            "top2": float(pred_r[1]),
            "matching_score": float(pred_matching),
            "diversity": float(calculate_diversity(pred_emb, min(300, max(1, pred_emb.shape[0] - 1)), rng=rng)),
            "gt_top1": float(gt_r[0]),
            "gt_top2": float(gt_r[1]),
            "gt_top3": float(gt_r[2]),
            "gt_matching_score": float(gt_matching),
            "gt_diversity": float(calculate_diversity(gt_emb, min(300, max(1, gt_emb.shape[0] - 1)), rng=rng)),
            "nb_sample": int(len(captions)),
            "metric_set": str(cfg.metric_set),
            "eval_backend": "motionstreamer272",
            "embedding_shapes": {
                "text": list(text_emb.shape),
                "motion_real": list(gt_emb.shape),
                "motion_pred": list(pred_emb.shape),
            },
        })
    if cfg.include_code_metrics and str(cfg.metric_set) != "fid_top3":
        metrics.update(_finalize_code_metrics(code_sums))
    return metrics
