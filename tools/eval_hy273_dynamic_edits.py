#!/usr/bin/env python3
"""Research evaluation for HY273 speed, repetition, and timing edits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_multitask_manifest_dataset import _transform_to_gauge
from models.raw_motion.hy273_slices import (
    CONTACT_JOINTS,
    CONTACT_SLICE,
    CONT_DIM,
    DIM_HY273,
    reconstruct_global_joints_from_features,
)
from sample_hy273_multitask import (
    make_edit_condition,
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from tools.render_hy273_edit_same_source_comparison import (
    joints as render_joints,
    render,
)
from train_hy273_multitask import create_model_from_checkpoint


DEFAULT_MANIFEST = Path(
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/test.jsonl"
)
FORMAT = "hy273_dynamic_edit_eval_v1"
WRIST_JOINTS = (20, 21)
TOE_JOINTS = (10, 11)
CATEGORY_PATTERNS = {
    "faster": re.compile(
        r"\b(faster|quicker|speed(?:s|ed|ing)?\s+up|more\s+quickly|"
        r"more\s+rapidly|increase(?:s|d)?\s+(?:the\s+)?(?:speed|pace))\b",
        re.IGNORECASE,
    ),
    "slower": re.compile(
        r"\b(slower|slow(?:s|ed|ing)?\s+down|more\s+slowly|"
        r"decrease(?:s|d)?\s+(?:the\s+)?(?:speed|pace))\b",
        re.IGNORECASE,
    ),
    "repeat": re.compile(
        r"\b(repeat(?:s|ed|ing)?|again|twice|two\s+more\s+times|"
        r"three\s+times|four\s+times|multiple\s+times|more\s+times|"
        r"one\s+more\s+time|another\s+time)\b",
        re.IGNORECASE,
    ),
    "timing": re.compile(
        r"\b(earlier|later|sooner|delay(?:ed|ing)?|wait(?:s|ed|ing)?|"
        r"immediately)\b",
        re.IGNORECASE,
    ),
}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip() or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid checkpoint: {value}")
    return label.strip(), path


def parse_csv(value: str) -> tuple[str, ...]:
    rows = tuple(token.strip() for token in value.split(",") if token.strip())
    if not rows:
        raise ValueError("expected at least one comma-separated value")
    return rows


def instruction(row: dict[str, Any]) -> str:
    texts = row.get("texts")
    if not isinstance(texts, list) or len(texts) != 1:
        raise ValueError(f"MotionFix row has an invalid text contract: {row.get('uid')}")
    text = str(texts[0].get("value", "")).strip()
    if not text:
        raise ValueError(f"MotionFix row has an empty instruction: {row.get('uid')}")
    return text


def load_rows(manifest: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") != "motionfix_k273":
                continue
            pair_id = str(row["pair"]["official_pair_id"])
            if pair_id in rows:
                raise RuntimeError(f"duplicate MotionFix pair id: {pair_id}")
            rows[pair_id] = row
    if not rows:
        raise RuntimeError(f"no MotionFix rows in {manifest}")
    return rows


def categories_for_text(text: str) -> tuple[str, ...]:
    return tuple(
        category
        for category, pattern in CATEGORY_PATTERNS.items()
        if pattern.search(text)
    )


def stable_order_key(seed: int, category: str, pair_id: str) -> bytes:
    return hashlib.blake2b(
        f"{seed}:{category}:{pair_id}".encode(), digest_size=16
    ).digest()


def stable_noise_seed(seed: int, pair_id: str) -> int:
    value = hashlib.blake2b(
        f"{seed}:noise:{pair_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(value, "little") & ((1 << 63) - 1)


def build_selection(
    rows: dict[str, dict[str, Any]],
    *,
    limit_per_category: int,
    max_frames: int,
    seed: int,
) -> dict[str, Any]:
    eligible: dict[str, list[str]] = {key: [] for key in CATEGORY_PATTERNS}
    categories_by_pair: dict[str, tuple[str, ...]] = {}
    for pair_id, row in rows.items():
        source = row["source_motion"]["k273_asset"]
        target = row["target_motion"]["k273_asset"]
        source_frames = int(source["frames"])
        target_frames = int(target["frames"])
        if (
            source_frames != target_frames
            or target_frames < 2
            or target_frames > int(max_frames)
        ):
            continue
        categories = categories_for_text(instruction(row))
        if not categories:
            continue
        categories_by_pair[pair_id] = categories
        for category in categories:
            eligible[category].append(pair_id)

    selected_by_category: dict[str, list[str]] = {}
    for category, pair_ids in eligible.items():
        pair_ids = sorted(
            pair_ids, key=lambda value: stable_order_key(seed, category, value)
        )
        if category == "faster" and "000038" in pair_ids:
            pair_ids.remove("000038")
            pair_ids.insert(0, "000038")
        selected_by_category[category] = pair_ids[: int(limit_per_category)]
    if any(len(values) < int(limit_per_category) for values in selected_by_category.values()):
        counts = {key: len(value) for key, value in selected_by_category.items()}
        raise RuntimeError(f"dynamic edit category is undersized: {counts}")

    selected_pairs = sorted(
        {pair_id for values in selected_by_category.values() for pair_id in values}
    )
    return {
        "format": FORMAT,
        "manifest": str(DEFAULT_MANIFEST),
        "selection_seed": int(seed),
        "max_frames": int(max_frames),
        "limit_per_category": int(limit_per_category),
        "eligible_counts": {key: len(value) for key, value in eligible.items()},
        "selected_by_category": selected_by_category,
        "selected_pairs": [
            {
                "pair_id": pair_id,
                "categories": list(categories_by_pair[pair_id]),
                "instruction": instruction(rows[pair_id]),
                "frames": int(
                    rows[pair_id]["target_motion"]["k273_asset"]["frames"]
                ),
            }
            for pair_id in selected_pairs
        ],
    }


def save_selection(args: argparse.Namespace) -> None:
    manifest = args.manifest.expanduser().resolve()
    rows = load_rows(manifest)
    payload = build_selection(
        rows,
        limit_per_category=args.limit_per_category,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    payload["manifest"] = str(manifest)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "selection.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selection": str(path),
                "unique_pairs": len(payload["selected_pairs"]),
                "eligible_counts": payload["eligible_counts"],
            },
            ensure_ascii=False,
        )
    )


def load_motion(ref: dict[str, Any]) -> torch.Tensor:
    values = np.load(ref["path"]).astype(np.float32, copy=False)
    expected = (int(ref["frames"]), DIM_HY273)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"invalid K273 asset {ref['path']}: {values.shape}")
    motion, _ = _transform_to_gauge(torch.from_numpy(values.copy()), 0.0)
    return motion.float()


def adaptive_peak_indices(signal: np.ndarray) -> np.ndarray:
    if signal.size < 3:
        return np.empty(0, dtype=np.int64)
    smooth = gaussian_filter1d(signal.astype(np.float64), 1.0)
    spread = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    prominence = max(0.03, 0.12 * spread)
    height = max(0.08, float(np.percentile(smooth, 60)))
    peaks, _ = find_peaks(
        smooth, distance=4, prominence=prominence, height=height
    )
    return peaks.astype(np.int64, copy=False)


def weighted_time_quantile(
    signal: np.ndarray, quantile: float
) -> float:
    if signal.size == 0:
        return 0.0
    weights = np.maximum(signal.astype(np.float64), 0.0)
    total = float(weights.sum())
    if total <= 1e-8:
        return 0.0
    index = int(np.searchsorted(np.cumsum(weights), quantile * total))
    return float(min(index, len(weights) - 1) / max(len(weights) - 1, 1))


def motion_descriptors(motion: np.ndarray, fps: float = 30.0) -> dict[str, Any]:
    value = torch.from_numpy(motion.astype(np.float32, copy=False))
    joints = reconstruct_global_joints_from_features(value).cpu().numpy()
    velocity = np.diff(joints, axis=0) * float(fps)
    speed = np.linalg.norm(velocity, axis=-1)
    root_speed = np.linalg.norm(np.diff(motion[:, (0, 2)], axis=0), axis=-1) * fps
    duration_seconds = float(len(motion) / fps)

    per_joint_mean = speed.mean(axis=0)
    per_joint_p95 = np.percentile(speed, 95, axis=0)
    per_joint_max = speed.max(axis=0)
    per_joint_peak_count = np.asarray(
        [len(adaptive_peak_indices(speed[:, joint])) for joint in range(speed.shape[1])],
        dtype=np.float64,
    )
    per_joint_center = np.asarray(
        [
            weighted_time_quantile(speed[:, joint], 0.50)
            for joint in range(speed.shape[1])
        ],
        dtype=np.float64,
    )
    per_joint_onset = np.asarray(
        [
            weighted_time_quantile(speed[:, joint], 0.10)
            for joint in range(speed.shape[1])
        ],
        dtype=np.float64,
    )
    per_joint_offset = np.asarray(
        [
            weighted_time_quantile(speed[:, joint], 0.90)
            for joint in range(speed.shape[1])
        ],
        dtype=np.float64,
    )

    # This exactly reproduces the earlier pair-000038 diagnostic definitions.
    strong_foot_signal = np.maximum(speed[:, TOE_JOINTS[0]], speed[:, TOE_JOINTS[1]])
    strong_foot_peaks = adaptive_peak_indices(strong_foot_signal)
    return {
        "mean_joint_speed_mps": float(speed.mean()),
        "mean_foot_speed_mps": float(speed[:, list(CONTACT_JOINTS)].mean()),
        "root_speed_mean_mps": float(root_speed.mean()),
        "strong_foot_peak_count": int(len(strong_foot_peaks)),
        "strong_foot_peak_frequency_hz": float(
            len(strong_foot_peaks) / max(duration_seconds, 1e-8)
        ),
        "wrist_peak_speed_mps": float(speed[:, list(WRIST_JOINTS)].max()),
        "contact_occupancy": float((motion[:, CONTACT_SLICE] > 0.5).mean()),
        "per_joint_mean_speed_mps": per_joint_mean.tolist(),
        "per_joint_p95_speed_mps": per_joint_p95.tolist(),
        "per_joint_max_speed_mps": per_joint_max.tolist(),
        "per_joint_peak_count": per_joint_peak_count.tolist(),
        "per_joint_activity_center": per_joint_center.tolist(),
        "per_joint_activity_onset": per_joint_onset.tolist(),
        "per_joint_activity_offset": per_joint_offset.tolist(),
    }


def delta_alignment(
    source: Iterable[float],
    target: Iterable[float],
    prediction: Iterable[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    source_array = np.asarray(tuple(source), dtype=np.float64)
    target_array = np.asarray(tuple(target), dtype=np.float64)
    prediction_array = np.asarray(tuple(prediction), dtype=np.float64)
    delta = target_array - source_array
    active = np.abs(delta) >= float(threshold)
    if not bool(active.any()):
        return {
            "active_dimensions": 0,
            "direction_accuracy": None,
            "mean_progress": None,
            "target_relative_error": None,
        }
    prediction_delta = prediction_array - source_array
    progress = prediction_delta[active] / delta[active]
    return {
        "active_dimensions": int(active.sum()),
        "direction_accuracy": float(
            np.mean(np.sign(prediction_delta[active]) == np.sign(delta[active]))
        ),
        "mean_progress": float(np.mean(np.clip(progress, -2.0, 3.0))),
        "target_relative_error": float(
            np.mean(
                np.abs(prediction_array[active] - target_array[active])
                / np.abs(delta[active])
            )
        ),
    }


def prediction_metrics(
    source: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    normalizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    source_desc = motion_descriptors(source)
    target_desc = motion_descriptors(target)
    prediction_desc = motion_descriptors(prediction)
    source_tensor = torch.from_numpy(source).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    prediction_tensor = torch.from_numpy(prediction).to(device)
    with torch.no_grad():
        source_norm = normalizer.normalize(source_tensor)
        target_norm = normalizer.normalize(target_tensor)
        prediction_norm = normalizer.normalize(prediction_tensor)
        source_joints = reconstruct_global_joints_from_features(source_tensor)
        target_joints = reconstruct_global_joints_from_features(target_tensor)
        prediction_joints = reconstruct_global_joints_from_features(prediction_tensor)
        source_velocity = (source_joints[1:] - source_joints[:-1]) * 30.0
        target_velocity = (target_joints[1:] - target_joints[:-1]) * 30.0
        prediction_velocity = (
            prediction_joints[1:] - prediction_joints[:-1]
        ) * 30.0
        target_speed = torch.linalg.vector_norm(target_velocity, dim=-1)
        prediction_speed = torch.linalg.vector_norm(prediction_velocity, dim=-1)
    return {
        "continuous_target_mse": float(
            (prediction_norm[..., :CONT_DIM] - target_norm[..., :CONT_DIM])
            .square()
            .mean()
            .item()
        ),
        "continuous_source_mse": float(
            (prediction_norm[..., :CONT_DIM] - source_norm[..., :CONT_DIM])
            .square()
            .mean()
            .item()
        ),
        "global_joint_target_error_m": float(
            torch.linalg.vector_norm(prediction_joints - target_joints, dim=-1)
            .mean()
            .item()
        ),
        "velocity_target_mae_mps": float(
            (prediction_velocity - target_velocity).abs().mean().item()
        ),
        "speed_target_mae_mps": float(
            (prediction_speed - target_speed).abs().mean().item()
        ),
        "source_target_velocity_mae_mps": float(
            (source_velocity - target_velocity).abs().mean().item()
        ),
        "mean_speed_alignment": delta_alignment(
            source_desc["per_joint_mean_speed_mps"],
            target_desc["per_joint_mean_speed_mps"],
            prediction_desc["per_joint_mean_speed_mps"],
            threshold=0.05,
        ),
        "peak_count_alignment": delta_alignment(
            source_desc["per_joint_peak_count"],
            target_desc["per_joint_peak_count"],
            prediction_desc["per_joint_peak_count"],
            threshold=1.0,
        ),
        "activity_center_alignment": delta_alignment(
            source_desc["per_joint_activity_center"],
            target_desc["per_joint_activity_center"],
            prediction_desc["per_joint_activity_center"],
            threshold=0.05,
        ),
        "activity_onset_alignment": delta_alignment(
            source_desc["per_joint_activity_onset"],
            target_desc["per_joint_activity_onset"],
            prediction_desc["per_joint_activity_onset"],
            threshold=0.05,
        ),
        "activity_offset_alignment": delta_alignment(
            source_desc["per_joint_activity_offset"],
            target_desc["per_joint_activity_offset"],
            prediction_desc["per_joint_activity_offset"],
            threshold=0.05,
        ),
        "source": source_desc,
        "target": target_desc,
        "prediction": prediction_desc,
    }


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def run_worker(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in run mode")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0,num_shards)")
    label, checkpoint_path = args.checkpoint
    output_dir = args.output_dir.expanduser().resolve()
    selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    manifest = Path(selection["manifest"]).expanduser().resolve()
    rows = load_rows(manifest)
    pair_ids = [
        item["pair_id"] for item in selection["selected_pairs"]
    ]
    pair_ids = [
        pair_id
        for index, pair_id in enumerate(pair_ids)
        if index % int(args.num_shards) == int(args.shard_id)
    ]
    categories = {
        item["pair_id"]: tuple(item["categories"])
        for item in selection["selected_pairs"]
    }

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    normalizer = normalizer_from_checkpoint(checkpoint, device)
    if not bool(normalizer.normalize_contacts):
        raise RuntimeError("dynamic Edit evaluation requires unified 273D flow")
    model = create_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[args.weight_source], strict=True)
    checkpoint_step = int(checkpoint["next_global_step"])
    del checkpoint
    model = model.to(device).eval()

    motion_dir = output_dir / "motion_outputs" / label
    motion_dir.mkdir(parents=True, exist_ok=True)
    record_dir = output_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / (
        f"{label}_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.jsonl"
    )
    if record_path.exists() and not args.overwrite:
        raise FileExistsError(record_path)

    records: list[dict[str, Any]] = []
    for chunk_index, chunk_ids in enumerate(batched(pair_ids, args.batch_size)):
        source_rows = [load_motion(rows[pair_id]["source_motion"]["k273_asset"]) for pair_id in chunk_ids]
        target_rows = [load_motion(rows[pair_id]["target_motion"]["k273_asset"]) for pair_id in chunk_ids]
        if any(source.shape != target.shape for source, target in zip(source_rows, target_rows)):
            raise RuntimeError("selected dynamic Edit pair is not exactly equal-length")
        lengths = torch.tensor([len(value) for value in target_rows], dtype=torch.long)
        frames = int(lengths.max().item())
        source_batch = torch.zeros(len(chunk_ids), frames, DIM_HY273)
        target_batch = torch.zeros_like(source_batch)
        noise_batch = torch.zeros_like(source_batch)
        for index, (pair_id, source, target) in enumerate(
            zip(chunk_ids, source_rows, target_rows)
        ):
            length = len(target)
            source_batch[index, :length] = source
            target_batch[index, :length] = target
            generator = torch.Generator(device="cpu").manual_seed(
                stable_noise_seed(args.seed, pair_id)
            )
            noise_batch[index, :length] = torch.randn(
                length, DIM_HY273, generator=generator
            )
        source_batch = source_batch.to(device)
        lengths_device = lengths.to(device)
        gauge = torch.zeros(len(chunk_ids), 2, device=device)
        gauge[:, 0] = 1.0
        condition = make_edit_condition(
            source_batch,
            target_lengths=lengths_device,
            source_lengths=lengths_device,
            target_frames=frames,
            frame_gauge_dir=gauge,
        )
        observed = torch.zeros(
            len(chunk_ids), frames, DIM_HY273, device=device
        )
        hard_mask = torch.zeros_like(observed, dtype=torch.bool)
        sampled = sample_hy273_multitask_ode(
            model,
            normalizer,
            condition,
            [instruction(rows[pair_id]) for pair_id in chunk_ids],
            observed,
            hard_mask,
            num_steps=args.ode_steps,
            source_cfg_scale=args.source_cfg_scale,
            edit_cfg_scale=args.edit_cfg_scale,
            cfg_apply_contacts=True,
            initial_unified_noise=noise_batch.to(device),
        )
        predictions = sampled.raw_motion.float().cpu().numpy()
        for index, pair_id in enumerate(chunk_ids):
            length = int(lengths[index].item())
            source = source_rows[index].numpy()
            target = target_rows[index].numpy()
            prediction = predictions[index, :length]
            prediction_path = motion_dir / f"pair_{pair_id}.npy"
            np.save(prediction_path, prediction)
            metrics = prediction_metrics(
                source,
                target,
                prediction,
                normalizer=normalizer,
                device=device,
            )
            records.append(
                {
                    "format": FORMAT,
                    "label": label,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step,
                    "weight_source": args.weight_source,
                    "pair_id": pair_id,
                    "instruction": instruction(rows[pair_id]),
                    "categories": list(categories[pair_id]),
                    "frames": length,
                    "noise_seed": stable_noise_seed(args.seed, pair_id),
                    "ode_steps": int(args.ode_steps),
                    "source_cfg_scale": float(args.source_cfg_scale),
                    "edit_cfg_scale": float(args.edit_cfg_scale),
                    "prediction_path": str(prediction_path),
                    "source_path": str(
                        rows[pair_id]["source_motion"]["k273_asset"]["path"]
                    ),
                    "target_path": str(
                        rows[pair_id]["target_motion"]["k273_asset"]["path"]
                    ),
                    "metrics": metrics,
                }
            )
        print(
            f"[{label}] shard={args.shard_id}/{args.num_shards} "
            f"batch={chunk_index + 1} completed={min((chunk_index + 1) * args.batch_size, len(pair_ids))}/{len(pair_ids)}",
            flush=True,
        )

    record_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "label": label,
                "record_path": str(record_path),
                "pairs": len(records),
            }
        )
    )


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return None if not finite else float(np.mean(finite))


def aggregate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    selection = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    labels = parse_csv(args.labels)
    records = []
    for path in sorted((output_dir / "records").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (str(record["label"]), str(record["pair_id"]))
        if key in by_key:
            raise RuntimeError(f"duplicate dynamic Edit result: {key}")
        by_key[key] = record
    selected_pairs = [item["pair_id"] for item in selection["selected_pairs"]]
    missing = [
        (label, pair_id)
        for label in labels
        for pair_id in selected_pairs
        if (label, pair_id) not in by_key
    ]
    if missing:
        raise RuntimeError(f"missing dynamic Edit outputs: {missing[:20]}")

    scalar_paths = (
        "continuous_target_mse",
        "global_joint_target_error_m",
        "velocity_target_mae_mps",
        "speed_target_mae_mps",
    )
    alignment_paths = (
        "mean_speed_alignment",
        "peak_count_alignment",
        "activity_center_alignment",
        "activity_onset_alignment",
        "activity_offset_alignment",
    )
    category_summary: dict[str, Any] = {}
    for category, category_pairs in selection["selected_by_category"].items():
        category_summary[category] = {}
        for label in labels:
            subset = [by_key[(label, pair_id)] for pair_id in category_pairs]
            metrics = {
                key: mean_or_none(record["metrics"][key] for record in subset)
                for key in scalar_paths
            }
            for alignment in alignment_paths:
                for key in (
                    "direction_accuracy",
                    "mean_progress",
                    "target_relative_error",
                ):
                    metrics[f"{alignment}/{key}"] = mean_or_none(
                        record["metrics"][alignment][key] for record in subset
                    )
            category_summary[category][label] = {
                "pairs": len(subset),
                "metrics": metrics,
            }

    known_pair = {
        label: by_key[(label, "000038")]
        for label in labels
        if (label, "000038") in by_key
    }
    known_pair_summary = {
        "instruction": next(iter(known_pair.values()))["instruction"],
        "source": next(iter(known_pair.values()))["metrics"]["source"],
        "target": next(iter(known_pair.values()))["metrics"]["target"],
        "predictions": {
            label: record["metrics"]["prediction"]
            for label, record in known_pair.items()
        },
    }

    render_ids: list[str] = []
    for category, pair_ids in selection["selected_by_category"].items():
        for pair_id in pair_ids[: int(args.render_per_category)]:
            if pair_id not in render_ids:
                render_ids.append(pair_id)
    if "000038" in selected_pairs:
        render_ids = ["000038", *[value for value in render_ids if value != "000038"]]
    row_by_pair = {
        item["pair_id"]: item for item in selection["selected_pairs"]
    }
    visual_dir = output_dir / "visuals"
    for visual_index, pair_id in enumerate(render_ids):
        reference = by_key[(labels[0], pair_id)]
        source = load_motion(
            {"path": reference["source_path"], "frames": reference["frames"]}
        ).numpy()
        target = load_motion(
            {"path": reference["target_path"], "frames": reference["frames"]}
        ).numpy()
        panels = [
            ("source", render_joints(source), "#6b7280"),
            ("target", render_joints(target), "#059669"),
        ]
        for label in labels:
            record = by_key[(label, pair_id)]
            prediction = np.load(record["prediction_path"])
            foot_speed = record["metrics"]["prediction"]["mean_foot_speed_mps"]
            panels.append(
                (
                    f"{label}\nfoot={foot_speed:.3f} m/s",
                    render_joints(prediction),
                    "#2563eb",
                )
            )
        render(
            panels,
            title=(
                f"pair {pair_id} | {row_by_pair[pair_id]['instruction']} | "
                f"{','.join(row_by_pair[pair_id]['categories'])}"
            ),
            path=visual_dir / f"{visual_index:02d}_pair_{pair_id}.gif",
            fps=30,
            stride=args.render_stride,
        )

    summary = {
        "format": FORMAT,
        "status": "completed",
        "labels": list(labels),
        "selection": selection,
        "category_summary": category_summary,
        "pair_000038": known_pair_summary,
        "visual_pair_ids": render_ids,
        "record_count": len(records),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "records": len(records),
                "gifs": len(render_ids),
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "aggregate"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=parse_checkpoint)
    parser.add_argument("--weight_source", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--limit_per_category", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--ode_steps", type=int, default=32)
    parser.add_argument("--source_cfg_scale", type=float, default=2.0)
    parser.add_argument("--edit_cfg_scale", type=float, default=2.0)
    parser.add_argument(
        "--labels", default="parent400k,positive450k,temporal450k"
    )
    parser.add_argument("--render_per_category", type=int, default=3)
    parser.add_argument("--render_stride", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.batch_size,
        args.limit_per_category,
        args.max_frames,
        args.ode_steps,
        args.num_shards,
    ) <= 0:
        raise ValueError("batch/selection/frame/ODE/shard counts must be positive")
    if args.mode == "prepare":
        save_selection(args)
    elif args.mode == "run":
        run_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
