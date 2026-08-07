#!/usr/bin/env python3
"""Matched UID-cluster comparison for two HY273 Reaction evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_reaction_metrics import reaction_fixed_role_metrics
from models.raw_motion.hy273_slices import DIM_HY273


COUNT_FIELDS = ("tp", "fp", "fn", "target_positive", "target_negative")
THRESHOLDS_CM = (10, 20, 30)
CAUSAL_VARIANTS = ("source_only", "shuffled_text", "unrelated_source", "empty")
REQUIRED_VARIANTS = ("source_text", *CAUSAL_VARIANTS)
EXPECTED_FORMAT = "hy273_fixed_role_reaction_eval_v2"
EXPECTED_DATASET = "Inter-X K273"
EXPECTED_ASSIGNMENT = "fixed_source_actor_to_target_reactor_no_swap"
DEFAULT_CHECKPOINT_STEP = 150_000
EXPECTED_SPLIT_COUNTS = {"val": 522, "test": 1_579}
EXPECTED_SAMPLING = {
    "num_steps": 32,
    "source_cfg_scale": 2.0,
    "text_cfg_scale": 2.0,
    "seed": 20260801,
    "initial_noise_policy": "deterministic_per_uid_and_caption",
    "matched_initial_noise_across_variants": True,
}
MEAN_METRICS = (
    "reactor_fk_mpjpe_cm",
    "reactor_root_error_cm",
    "fk_relation_distance_mae_cm",
    "position_relation_distance_mae_cm",
    "partner_facing_error_deg",
    "relative_root_radius_error_cm",
    "relative_root_bearing_error_deg",
    "relative_heading_error_deg",
    "frame0_relative_root_error_cm",
    "initial_15f_relative_root_error_cm",
    "frame0_relative_heading_error_deg",
    "initial_15f_relative_heading_error_deg",
    "first_close_timing_error_s_20cm",
    "first_close_too_early_s_20cm",
    "first_close_too_late_s_20cm",
    "reactor_fk_jerk_error_mps3",
    "reactor_prediction_fk_jerk_mps3",
)
LAYOUT_PHASE_MEAN_METRICS = (
    "frame0_relative_root_error_cm",
    "initial_15f_relative_root_error_cm",
    "frame0_relative_heading_error_deg",
    "initial_15f_relative_heading_error_deg",
    "first_close_timing_error_s_20cm",
    "first_close_too_early_s_20cm",
    "first_close_too_late_s_20cm",
)
LAYOUT_PHASE_POOLED_METRICS = {
    "precontact_relative_root_error_cm": (
        "precontact_relative_root_error_sum_cm",
        "precontact_valid_frames_20cm",
    ),
    "precontact_relative_heading_error_deg": (
        "precontact_relative_heading_error_sum_deg",
        "precontact_valid_frames_20cm",
    ),
    "precontact_false_close_rate_20cm": (
        "precontact_false_close_frames_20cm",
        "precontact_valid_frames_20cm",
    ),
}
CONTACT_LIFECYCLE_POOLED_METRICS = {
    "fk_contact_vector_error_cm_15cm": (
        "fk_contact_vector_error_sum_cm_15cm",
        "fk_contact_vector_target_pairs_15cm",
    ),
}
PREDICTION_REPORT_CONSISTENCY_METRICS = (
    "reactor_fk_mpjpe_cm",
    "reactor_root_error_cm",
    "fk_relation_distance_mae_cm",
)
TASK_EXPOSURE_KEYS = {
    "realized_hml": "t2m",
    "realized_edit": "edit",
    "realized_interaction": "reaction",
}
TASK_STREAM_IDS = {
    "realized_hml": 0,
    "realized_edit": 1,
    "realized_interaction": 2,
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("per_sample"), dict):
        raise ValueError(f"Evaluation has no per_sample variants: {path}")
    return payload


def _rows_by_uid(payload: dict[str, Any], variant: str) -> dict[str, list[dict[str, Any]]]:
    rows = payload["per_sample"].get(variant)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing non-empty per_sample variant {variant!r}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("variant") != variant:
            raise ValueError(
                f"Row variant mismatch: requested {variant!r}, got {row.get('variant')!r}"
            )
        uid = str(row["uid"])
        grouped.setdefault(uid, []).append(row)
    for uid_rows in grouped.values():
        uid_rows.sort(
            key=lambda row: (
                int(row["caption_index"]),
                int(row["dataset_index"]),
            )
        )
    return grouped


def _load_prediction_motion(path: Path, length: int) -> torch.Tensor:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != DIM_HY273 or array.shape[0] < length:
        raise ValueError(
            f"Prediction tensor {path} cannot provide [{length},{DIM_HY273}]: "
            f"{array.shape}"
        )
    selected = np.asarray(array[:length], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError(f"Prediction tensor contains non-finite values: {path}")
    return torch.from_numpy(selected.copy())


def _recompute_metrics_from_predictions(
    payload: dict[str, Any],
    prediction_dir: Path,
) -> dict[str, Any]:
    """Replace row metrics using saved samples and the current metric implementation."""

    prediction_dir = prediction_dir.expanduser().resolve(strict=True)
    grouped = {
        variant: _rows_by_uid(payload, variant) for variant in REQUIRED_VARIANTS
    }
    uids = sorted(grouped["source_text"])
    for variant in REQUIRED_VARIANTS:
        if set(grouped[variant]) != set(uids):
            raise ValueError(f"Saved-report UID set differs for variant {variant!r}")

    for uid in uids:
        rows = [grouped[variant][uid] for variant in REQUIRED_VARIANTS]
        if any(len(variant_rows) != 1 for variant_rows in rows):
            raise ValueError(
                "Prediction metric recomputation expects one uid-balanced caption per UID"
            )
        report_rows = [variant_rows[0] for variant_rows in rows]
        length = int(report_rows[0]["length"])
        case_dir = prediction_dir / uid
        metadata_path = case_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            str(metadata.get("uid")) != uid
            or int(metadata.get("length", -1)) != length
            or int(metadata.get("caption_index", -1))
            != int(report_rows[0]["caption_index"])
        ):
            raise ValueError(f"Prediction metadata does not match report row for {uid}")

        source = _load_prediction_motion(case_dir / "source.npy", length)
        target = _load_prediction_motion(case_dir / "target.npy", length)
        predictions = torch.stack(
            [
                _load_prediction_motion(case_dir / f"{variant}.npy", length)
                for variant in REQUIRED_VARIANTS
            ],
            dim=0,
        )
        count = len(REQUIRED_VARIANTS)
        metrics = reaction_fixed_role_metrics(
            source.unsqueeze(0).expand(count, -1, -1),
            predictions,
            target.unsqueeze(0).expand(count, -1, -1),
            lengths=torch.full((count,), length, dtype=torch.long),
            fps=30.0,
        )["per_sample"]
        for report_row, metric_row in zip(report_rows, metrics):
            for metric in PREDICTION_REPORT_CONSISTENCY_METRICS:
                if metric not in report_row:
                    raise ValueError(
                        f"Report row {uid} variant={report_row['variant']!r} "
                        f"is missing required consistency metric {metric}"
                    )
                try:
                    reported = float(report_row[metric])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Report row {uid} variant={report_row['variant']!r} "
                        f"has non-numeric consistency metric {metric}="
                        f"{report_row[metric]!r}"
                    ) from error
                if not math.isfinite(reported):
                    raise ValueError(
                        f"Report row {uid} variant={report_row['variant']!r} "
                        f"has non-finite consistency metric {metric}={reported}"
                    )
                recomputed = float(metric_row[metric])
                if not math.isclose(
                    reported,
                    recomputed,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-4,
                ):
                    raise ValueError(
                        f"Saved predictions do not match report row {uid} "
                        f"variant={report_row['variant']!r}: {metric} "
                        f"reported={reported}, recomputed={recomputed}"
                    )
            report_row.update(
                {
                    key: value
                    for key, value in metric_row.items()
                    if key not in {"index", "length", "assignment"}
                }
            )

    payload["matched_metric_recomputation"] = {
        "prediction_dir": str(prediction_dir),
        "metric_function": "reaction_fixed_role_metrics",
        "fps": 30.0,
        "uids": len(uids),
        "variants": list(REQUIRED_VARIANTS),
        "report_consistency_metrics": list(
            PREDICTION_REPORT_CONSISTENCY_METRICS
        ),
    }
    return payload


def _validate_prediction_input_identity(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_dir: Path,
    candidate_dir: Path,
) -> dict[str, Any]:
    """Require both matched arms to use identical physical inputs and captions."""

    baseline_dir = baseline_dir.expanduser().resolve(strict=True)
    candidate_dir = candidate_dir.expanduser().resolve(strict=True)
    baseline_rows = _rows_by_uid(baseline, "source_text")
    candidate_rows = _rows_by_uid(candidate, "source_text")
    if set(baseline_rows) != set(candidate_rows):
        raise ValueError("Prediction input identity requires matched UID sets")

    metadata_fields = (
        "uid",
        "text",
        "caption_index",
        "negative_donor_uid",
        "negative_donor_text",
        "actor_person_index",
        "length",
    )
    tensor_names = ("source.npy", "target.npy")
    for uid in sorted(baseline_rows):
        left_case = baseline_dir / uid
        right_case = candidate_dir / uid
        with (left_case / "metadata.json").open("r", encoding="utf-8") as handle:
            left_metadata = json.load(handle)
        with (right_case / "metadata.json").open("r", encoding="utf-8") as handle:
            right_metadata = json.load(handle)
        for field in metadata_fields:
            if left_metadata.get(field) != right_metadata.get(field):
                raise ValueError(
                    f"Matched evaluation metadata differs for UID {uid}, "
                    f"field {field!r}: {left_metadata.get(field)!r} != "
                    f"{right_metadata.get(field)!r}"
                )
        for tensor_name in tensor_names:
            left = np.load(left_case / tensor_name, mmap_mode="r", allow_pickle=False)
            right = np.load(right_case / tensor_name, mmap_mode="r", allow_pickle=False)
            if left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(
                    f"Matched {tensor_name} schema differs for UID {uid}: "
                    f"{left.shape}/{left.dtype} != {right.shape}/{right.dtype}"
                )
            if not np.array_equal(left, right):
                raise ValueError(
                    f"Matched evaluation {tensor_name} content differs for UID {uid}"
                )

    return {
        "uids": len(baseline_rows),
        "metadata_fields": list(metadata_fields),
        "physical_tensors": list(tensor_names),
        "comparison": "exact_array_and_metadata_equality",
    }


def _validate_single_protocol(
    payload: dict[str, Any],
    label: str,
    *,
    expected_checkpoint_step: int,
    expected_split: str,
) -> None:
    expected_count = EXPECTED_SPLIT_COUNTS[expected_split]
    expected_scalars = {
        "format": EXPECTED_FORMAT,
        "dataset": EXPECTED_DATASET,
        "split": expected_split,
        "caption_policy": "uid_balanced",
        "weight_source": "ema",
        "assignment_rule": EXPECTED_ASSIGNMENT,
        "checkpoint_next_global_step": expected_checkpoint_step,
    }
    for field, expected in expected_scalars.items():
        actual = payload.get(field)
        if actual != expected:
            raise ValueError(
                f"{label} violates fixed protocol for {field}: {actual!r} != {expected!r}"
            )

    sampling = payload.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError(f"{label} has no sampling metadata")
    for field, expected in EXPECTED_SAMPLING.items():
        actual = sampling.get(field)
        if actual != expected:
            raise ValueError(
                f"{label} violates fixed sampling for {field}: {actual!r} != {expected!r}"
            )

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{label} has no selection metadata")
    expected_selection = {
        "count": expected_count,
        "dataset_count_after_filters": expected_count,
        "start_index": 0,
    }
    for field, expected in expected_selection.items():
        actual = selection.get(field)
        if actual != expected:
            raise ValueError(
                f"{label} is not the complete fixed {expected_split} selection for {field}: "
                f"{actual!r} != {expected!r}"
            )

    variants = payload.get("per_sample")
    if not isinstance(variants, dict):
        raise ValueError(f"{label} has no per_sample mapping")
    for variant in REQUIRED_VARIANTS:
        rows = variants.get(variant)
        if not isinstance(rows, list) or len(rows) != expected_count:
            raise ValueError(
                f"{label} variant {variant!r} is incomplete: "
                f"{len(rows) if isinstance(rows, list) else None} != {expected_count}"
            )

    causal = payload.get("protocols", {}).get("causal_ablations")
    if causal != ["empty", "shuffled_text", "unrelated_source"]:
        raise ValueError(f"{label} has unexpected causal ablations: {causal!r}")
    donor = payload.get("negative_donor_protocol")
    if not isinstance(donor, dict) or donor.get("scope") != "complete_filtered_split":
        raise ValueError(f"{label} lacks the complete negative-donor protocol")


def _validate_protocol(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_checkpoint_step: int,
    candidate_checkpoint_step: int,
    expected_split: str,
) -> None:
    _validate_single_protocol(
        baseline,
        "baseline",
        expected_checkpoint_step=baseline_checkpoint_step,
        expected_split=expected_split,
    )
    _validate_single_protocol(
        candidate,
        "candidate",
        expected_checkpoint_step=candidate_checkpoint_step,
        expected_split=expected_split,
    )
    scalar_fields = ("split", "caption_policy", "weight_source", "assignment_rule")
    for field in scalar_fields:
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"Protocol mismatch for {field}: {baseline.get(field)!r} != {candidate.get(field)!r}")
    sampling_fields = (
        "num_steps",
        "source_cfg_scale",
        "text_cfg_scale",
        "seed",
    )
    for field in sampling_fields:
        left = baseline.get("sampling", {}).get(field)
        right = candidate.get("sampling", {}).get(field)
        if left != right:
            raise ValueError(f"Sampling mismatch for {field}: {left!r} != {right!r}")
    if baseline.get("negative_donor_protocol") != candidate.get("negative_donor_protocol"):
        raise ValueError("Negative-donor protocols differ between evaluation arms")


def _config_differences(
    left: Any,
    right: Any,
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[tuple[tuple[str, ...], Any, Any]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append((path + (str(key),), left.get(key), right.get(key)))
            else:
                differences.extend(
                    _config_differences(left[key], right[key], path + (str(key),))
                )
        return differences
    if left != right:
        return [(path, left, right)]
    return []


def _expected_schedule_exposures(
    segments: list[dict[str, Any]],
    checkpoint_step: int,
) -> dict[str, int]:
    expected = {key: 0 for key in TASK_EXPOSURE_KEYS}
    previous_end = 0
    covered = 0
    paired_key: str | None = None
    for segment in segments:
        start = int(segment["start"])
        end = int(segment["end"])
        if start != previous_end or end <= start:
            raise ValueError("Task schedule is not contiguous and non-empty")
        current_paired_key = "reaction" if "reaction" in segment else "interaction"
        if paired_key is None:
            paired_key = current_paired_key
        elif paired_key != current_paired_key:
            raise ValueError("Task schedule changes paired-task key")
        weights = {
            "realized_hml": int(segment["t2m"]),
            "realized_edit": int(segment["edit"]),
            "realized_interaction": int(segment[current_paired_key]),
        }
        if min(weights.values()) < 0 or sum(weights.values()) != 100:
            raise ValueError("Task schedule weights must be non-negative and sum to 100")
        active = max(0, min(end, checkpoint_step) - start)
        if active:
            covered += active
            for state_key, weight in weights.items():
                numerator = active * weight
                if numerator % 100:
                    raise ValueError(
                        "Checkpoint boundary does not have an exact weighted task count"
                    )
                expected[state_key] += numerator // 100
        previous_end = end
    if covered != checkpoint_step:
        raise ValueError(
            f"Task schedule covers {covered} updates at checkpoint {checkpoint_step}"
        )
    return expected


def _validate_scheduler_at_step(
    contract: dict[str, Any],
    checkpoint_step: int,
) -> dict[str, int]:
    batcher = contract["batcher"]
    scheduler = batcher.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("Checkpoint lacks scheduler state")
    config_segments = list(contract["config"]["schedule"]["segments"])
    if scheduler.get("segments") != config_segments:
        raise ValueError("Checkpoint scheduler segments differ from embedded config")
    state = scheduler.get("state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint lacks scheduler counters")
    if int(state.get("next_step", -1)) != checkpoint_step:
        raise ValueError("Checkpoint scheduler step does not match checkpoint")
    if any(
        int(state.get(key, -1)) != 0
        for key in ("debt_hml", "debt_edit", "debt_interaction")
    ):
        raise ValueError("Checkpoint scheduler has non-zero task debt")

    expected = _expected_schedule_exposures(config_segments, checkpoint_step)
    actual = {key: int(state.get(key, -1)) for key in TASK_EXPOSURE_KEYS}
    if actual != expected:
        raise ValueError(
            f"Checkpoint has wrong absolute task exposures: {actual} != {expected}"
        )

    world_size = int(batcher.get("world_size", -1))
    local_batch_sizes = batcher.get("local_batch_sizes")
    ordinals = batcher.get("next_global_sample_ordinal")
    cursors = batcher.get("cursors")
    if (
        world_size <= 0
        or not isinstance(local_batch_sizes, dict)
        or not isinstance(ordinals, dict)
        or not isinstance(cursors, dict)
    ):
        raise ValueError("Checkpoint lacks complete stream-cursor metadata")
    for state_key, stream_id in TASK_STREAM_IDS.items():
        stream_key = str(stream_id)
        global_batch = int(local_batch_sizes.get(stream_key, -1)) * world_size
        cursor = cursors.get(stream_key)
        if global_batch <= 0 or not isinstance(cursor, dict):
            raise ValueError(f"Checkpoint lacks cursor for stream {stream_id}")
        if int(cursor.get("global_batch_size", -1)) != global_batch:
            raise ValueError(f"Cursor batch size is wrong for stream {stream_id}")
        expected_ordinal = expected[state_key] * global_batch
        if int(ordinals.get(stream_key, -1)) != expected_ordinal:
            raise ValueError(
                f"Stream {stream_id} sample ordinal is not aligned with task exposure"
            )
    return expected


def _replay_cursor_state(
    saved: dict[str, Any],
    *,
    row_bucket_keys: tuple[tuple[int, int], ...],
    manifest_sha256: str,
    run_seed: int,
    stream: int,
    updates: int,
) -> dict[str, Any]:
    from data.hy273_multitask_scheduler import DeterministicStreamCursor
    from models.raw_motion.hy273_multitask_condition import TrainStream

    cursor = DeterministicStreamCursor(
        row_bucket_keys=row_bucket_keys,
        manifest_sha256=manifest_sha256,
        run_seed=run_seed,
        stream=TrainStream(stream),
        global_batch_size=int(saved["global_batch_size"]),
        sort_window_batches=int(saved["sort_window_batches"]),
    )
    cursor.load_state_dict(saved)
    for _ in range(int(updates)):
        cursor.next_global_batch()
    return cursor.state_dict()


def _validate_stream_continuity(
    left: dict[str, Any],
    right: dict[str, Any],
    exposure_delta: dict[str, int],
) -> dict[str, Any]:
    """Replay only cursor bookkeeping from 200K to prove exact data continuation."""

    from data.hy273_multitask_manifest_dataset import HY273MultitaskManifestDataset
    from data.hy273_reaction_dataset import HY273ReactionDataset
    from models.raw_motion.hy273_multitask_condition import TrainStream

    left_batcher = left["batcher"]
    right_batcher = right["batcher"]
    datasets = {
        TrainStream.HML_MIXED: HY273MultitaskManifestDataset(
            left_batcher["multitask_manifest"],
            TrainStream.HML_MIXED,
            verify_payload_hash=False,
        ),
        TrainStream.MOTION_EDIT: HY273MultitaskManifestDataset(
            left_batcher["multitask_manifest"],
            TrainStream.MOTION_EDIT,
            verify_payload_hash=False,
        ),
        TrainStream.REACTION: HY273ReactionDataset(
            left_batcher["interaction_root"],
            split="train",
            exclude_overlength=bool(
                left_batcher["interaction_exclude_overlength"]
            ),
        ),
    }
    streams = {
        "realized_hml": TrainStream.HML_MIXED,
        "realized_edit": TrainStream.MOTION_EDIT,
        "realized_interaction": TrainStream.REACTION,
    }
    for state_key, stream in streams.items():
        stream_key = str(int(stream))
        saved = left_batcher["cursors"][stream_key]
        expected_cursor = _replay_cursor_state(
            saved,
            row_bucket_keys=datasets[stream].bucket_keys,
            manifest_sha256=datasets[stream].manifest_sha256,
            run_seed=int(left_batcher["run_seed"]),
            stream=int(stream),
            updates=int(exposure_delta[state_key]),
        )
        if expected_cursor != right_batcher["cursors"].get(stream_key):
            raise ValueError(
                f"Stream {int(stream)} cursor does not exactly continue from baseline"
            )
    return {
        "replayed_task_updates": dict(exposure_delta),
        "streams": [0, 1, 2],
        "exact_cursor_state_match": True,
    }


def _load_checkpoint_contract(
    payload: dict[str, Any],
    label: str,
    *,
    expected_checkpoint_step: int,
) -> dict[str, Any]:
    try:
        import torch
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError as error:
        raise RuntimeError("PyTorch is required to validate checkpoint contracts") from error

    checkpoint_value = payload.get("checkpoint")
    if not isinstance(checkpoint_value, str):
        raise ValueError(f"{label} has no checkpoint path")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve(strict=True)
    expected_name = f"step_{expected_checkpoint_step:08d}.pt"
    if checkpoint_path.name != expected_name:
        raise ValueError(
            f"{label} is not the fixed {expected_checkpoint_step} checkpoint: "
            f"{checkpoint_path}"
        )
    with FakeTensorMode():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    if checkpoint.get("format") != "hy273_unified_actor_checkpoint_v1":
        raise ValueError(f"{label} has unexpected checkpoint format")
    if checkpoint.get("next_global_step") != expected_checkpoint_step:
        raise ValueError(
            f"{label} checkpoint is not at {expected_checkpoint_step}"
        )
    if payload.get("checkpoint_next_global_step") != checkpoint.get("next_global_step"):
        raise ValueError(f"{label} evaluation/checkpoint step metadata disagree")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{label} checkpoint has no embedded config")
    ema_every = int(config.get("training", {}).get("ema_every", 0))
    if ema_every <= 0:
        raise ValueError(f"{label} checkpoint has an invalid EMA interval")
    expected_ema_updates = (
        expected_checkpoint_step + ema_every - 1
    ) // ema_every
    ema_update_count = int(checkpoint.get("ema_update_count", -1))
    if ema_update_count != expected_ema_updates:
        raise ValueError(
            f"{label} EMA update count is wrong: "
            f"{ema_update_count} != {expected_ema_updates}"
        )
    batcher = checkpoint.get("batcher")
    if not isinstance(batcher, dict):
        raise ValueError(f"{label} checkpoint has no batcher state")
    return {
        "path": str(checkpoint_path),
        "run_name": checkpoint.get("run_name"),
        "config_path": checkpoint.get("config_path"),
        "config": config,
        "rng_contract": checkpoint.get("rng_contract"),
        "ema_update_count": ema_update_count,
        "batcher": batcher,
    }


def _validate_training_contract(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    mode: str,
    baseline_checkpoint_step: int,
    candidate_checkpoint_step: int,
) -> dict[str, Any]:
    left = _load_checkpoint_contract(
        baseline,
        "baseline",
        expected_checkpoint_step=baseline_checkpoint_step,
    )
    right = _load_checkpoint_contract(
        candidate,
        "candidate",
        expected_checkpoint_step=candidate_checkpoint_step,
    )
    differences = _config_differences(left["config"], right["config"])
    stream_continuity: dict[str, Any] | None = None
    if mode != "same_run_dose_extension" and (
        baseline_checkpoint_step != candidate_checkpoint_step
    ):
        raise ValueError(f"{mode} requires same-step checkpoints")
    if mode == "p_only_ablation":
        expected_differences = [
            (("reaction_loss", "close_joint_vector"), 0.0, 0.01),
        ]
    elif mode == "reaction_v3_adaptive":
        baseline_close_vector = left["config"].get("reaction_loss", {}).get(
            "close_joint_vector"
        )
        if baseline_close_vector not in (0.0, 0.01):
            raise ValueError(
                "Reaction-v3 comparison expects a v2 or v2-P-only baseline, "
                f"got close_joint_vector={baseline_close_vector!r}"
            )
        expected_differences = [
            (("reaction_loss", "adaptive_distance_beta_m"), None, 0.05),
            (("reaction_loss", "adaptive_distance_eps_m"), None, 0.1),
            (
                ("reaction_loss", "close_joint_vector"),
                baseline_close_vector,
                0.00191,
            ),
            (("reaction_loss", "fine_min_flow_t"), 0.55, 0.2),
            (("reaction_loss", "joint_distance"), 0.01, 0.0273),
            (
                ("reaction_loss", "joint_distance_mode"),
                None,
                "adaptive_gt_inverse",
            ),
        ]
    elif mode == "reaction_v4_layout":
        expected_differences = [
            (("reaction_loss", "heading_beta"), None, 0.1),
            (("reaction_loss", "layout_contact_threshold_m"), None, 0.2),
            (("reaction_loss", "layout_initial_frames"), None, 15),
            (("reaction_loss", "layout_initial_multiplier"), None, 3.0),
            (("reaction_loss", "layout_precontact_multiplier"), None, 2.0),
            (("reaction_loss", "relative_heading"), 0.0, 0.0217),
            (("reaction_loss", "relative_root"), 0.0, 0.0195),
        ]
    elif mode == "reaction_v5_1_full_contact":
        expected_differences = [
            (("reaction_loss", "fk_contact_map_negative"), None, 0.005),
            (("reaction_loss", "fk_contact_map_positive"), None, 0.001),
            (("reaction_loss", "fk_contact_temperature_m"), None, 0.02),
            (("reaction_loss", "fk_contact_threshold_m"), None, 0.15),
            (("reaction_loss", "fk_contact_transition"), None, 0.003),
            (("reaction_loss", "fk_contact_transition_beta"), None, 0.1),
            (("reaction_loss", "fk_contact_vector"), None, 0.002),
            (("reaction_loss", "fk_contact_vector_scale_m"), None, 0.05),
        ]
    elif mode == "same_run_dose_extension":
        if candidate_checkpoint_step <= baseline_checkpoint_step:
            raise ValueError("Dose comparison requires candidate step > baseline step")
        if left["run_name"] != right["run_name"]:
            raise ValueError("Dose comparison checkpoints belong to different runs")
        expected_paths = {
            ("schedule", "segments"),
            ("training", "max_global_step"),
        }
        if {path for path, _, _ in differences} != expected_paths:
            formatted = [
                {"path": ".".join(path), "baseline": old, "candidate": new}
                for path, old, new in differences
            ]
            raise ValueError(
                "Dose checkpoints differ outside the same-mix horizon extension: "
                f"{formatted}"
            )
        left_segments = list(left["config"]["schedule"]["segments"])
        right_segments = list(right["config"]["schedule"]["segments"])
        if len(left_segments) != len(right_segments) or not left_segments:
            raise ValueError("Dose checkpoint schedules have different structures")
        left_last = dict(left_segments[-1])
        right_last = dict(right_segments[-1])
        left_end = int(left_last.pop("end"))
        right_end = int(right_last.pop("end"))
        if not (
            left_segments[:-1] == right_segments[:-1]
            and left_last == right_last
            and left_end == baseline_checkpoint_step
            and right_end == candidate_checkpoint_step
            and int(left["config"]["training"]["max_global_step"])
            == baseline_checkpoint_step
            and int(right["config"]["training"]["max_global_step"])
            == candidate_checkpoint_step
        ):
            raise ValueError("Dose checkpoints are not an exact final-segment extension")

        static_batcher_keys = (
            "format",
            "multitask_manifest",
            "interaction_root",
            "run_seed",
            "world_size",
            "interaction_exclude_overlength",
            "paired_task",
            "local_batch_sizes",
            "manifest_hashes",
        )
        left_static = {key: left["batcher"].get(key) for key in static_batcher_keys}
        right_static = {key: right["batcher"].get(key) for key in static_batcher_keys}
        if left_static != right_static:
            raise ValueError("Dose checkpoints use different data/batcher contracts")
        left_exposures = _validate_scheduler_at_step(
            left, baseline_checkpoint_step
        )
        right_exposures = _validate_scheduler_at_step(
            right, candidate_checkpoint_step
        )
        dose_steps = candidate_checkpoint_step - baseline_checkpoint_step
        total_weight = sum(
            int(right_last[name]) for name in TASK_EXPOSURE_KEYS.values()
        )
        exposure_delta: dict[str, int] = {}
        for state_key, schedule_key in TASK_EXPOSURE_KEYS.items():
            numerator = dose_steps * int(right_last[schedule_key])
            if numerator % total_weight:
                raise ValueError("Dose interval is not divisible by the task mixture")
            expected_delta = numerator // total_weight
            actual_delta = right_exposures[state_key] - left_exposures[state_key]
            if actual_delta != expected_delta:
                raise ValueError(
                    f"Dose task exposure mismatch for {state_key}: "
                    f"{actual_delta} != {expected_delta}"
                )
            exposure_delta[state_key] = actual_delta
        stream_continuity = _validate_stream_continuity(
            left,
            right,
            exposure_delta,
        )
        expected_differences = differences
    else:
        raise ValueError(f"Unsupported training-contract mode: {mode!r}")

    if differences != expected_differences:
        formatted = [
            {"path": ".".join(path), "baseline": old, "candidate": new}
            for path, old, new in differences
        ]
        raise ValueError(
            f"Checkpoint configs do not implement {mode!r}: "
            f"{formatted}"
        )
    if left["rng_contract"] != right["rng_contract"]:
        raise ValueError("Checkpoint RNG contracts differ")
    if mode != "same_run_dose_extension" and left["batcher"] != right["batcher"]:
        raise ValueError("Final deterministic batcher states differ between arms")
    result = {
        "baseline_checkpoint": left["path"],
        "candidate_checkpoint": right["path"],
        "baseline_run_name": left["run_name"],
        "candidate_run_name": right["run_name"],
        "baseline_config_path": left["config_path"],
        "candidate_config_path": right["config_path"],
        "mode": mode,
        "config_differences": [
            {
                "path": ".".join(path),
                "baseline": old,
                "candidate": new,
            }
            for path, old, new in differences
        ],
        "rng_contract": left["rng_contract"],
        "ema_update_counts": {
            "baseline": left["ema_update_count"],
            "candidate": right["ema_update_count"],
        },
        "deterministic_batcher_state_matched": (
            mode != "same_run_dose_extension"
        ),
        "parent_lineage_note": (
            "Both checkpoints carry one run name and exact replayed data-stream "
            "continuity."
            if mode == "same_run_dose_extension"
            else (
                "Both legacy runs were launched from the protocol-locked 100K "
                "parent; the child checkpoint format does not embed its resume path."
            )
        ),
    }
    if mode == "same_run_dose_extension":
        result.update(
            {
                "estimand": "same_run_additional_training_dose",
                "baseline_checkpoint_step": baseline_checkpoint_step,
                "candidate_checkpoint_step": candidate_checkpoint_step,
                "additional_global_steps": (
                    candidate_checkpoint_step - baseline_checkpoint_step
                ),
                "additional_task_updates": exposure_delta,
                "same_data_and_batcher_static_contract": True,
                "data_stream_continuity": stream_continuity,
                "causal_scope": (
                    "Effect of additional same-recipe training; not an isolated "
                    "full-contact mechanism effect."
                ),
            }
        )
    return result


def _validate_matched_rows(
    baseline: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
) -> list[str]:
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "UID sets differ: "
            f"missing_candidate={missing_candidate[:5]}, missing_baseline={missing_baseline[:5]}"
        )
    matched_fields = (
        "uid",
        "dataset_index",
        "caption_index",
        "length",
        "length_bucket",
        "action_category",
        "actor_person_index",
        "assignment",
        "negative_donor_uid",
        "negative_donor_action_category",
    )
    for uid in sorted(baseline):
        left_rows = baseline[uid]
        right_rows = candidate[uid]
        if len(left_rows) != len(right_rows):
            raise ValueError(f"Row count differs for UID {uid}")
        for left, right in zip(left_rows, right_rows):
            for field in matched_fields:
                if left.get(field) != right.get(field):
                    raise ValueError(
                        f"Matched field {field!r} differs for UID {uid}: "
                        f"{left.get(field)!r} != {right.get(field)!r}"
                    )
    return sorted(baseline)


def _cluster_counts(
    grouped: dict[str, list[dict[str, Any]]],
    uids: Iterable[str],
    prefix: str,
    threshold_cm: int,
    *,
    event_name: str = "close",
    pairs_per_frame: int = 1,
    frame_offset: int = 0,
    event_channels: int = 1,
) -> np.ndarray:
    if pairs_per_frame < 1 or event_channels < 1 or frame_offset < 0:
        raise ValueError("Event denominator parameters must be positive")
    rows = []
    stem = f"{prefix}{event_name}_{threshold_cm}cm_"
    for uid in uids:
        values = np.zeros(len(COUNT_FIELDS), dtype=np.float64)
        for row in grouped[uid]:
            local = []
            for field in COUNT_FIELDS:
                key = stem + field
                try:
                    value = float(row[key])
                except KeyError as error:
                    raise ValueError(f"Missing count field {key!r} for UID {uid}") from error
                if not math.isfinite(value) or value < 0 or value != round(value):
                    raise ValueError(f"Invalid integer count {key}={value!r} for UID {uid}")
                local.append(value)
            tp, fp, fn, target_positive, target_negative = local
            if tp + fn != target_positive:
                raise ValueError(f"TP+FN does not equal target_positive for UID {uid}")
            if fp > target_negative:
                raise ValueError(f"FP exceeds target_negative for UID {uid}")
            expected_denominator = (
                max(int(row["length"]) - frame_offset, 0)
                * pairs_per_frame
                * event_channels
            )
            if target_positive + target_negative != expected_denominator:
                raise ValueError(
                    f"GT {event_name} denominator does not match the event map "
                    f"for UID {uid}: {target_positive + target_negative} != "
                    f"{expected_denominator}"
                )
            values += np.asarray(local, dtype=np.float64)
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def _count_metrics(counts: np.ndarray) -> np.ndarray:
    tp, fp, fn, target_positive, target_negative = np.moveaxis(counts, -1, 0)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=denominator > 0,
        )

    return np.stack(
        (
            ratio(tp, tp + fp),
            ratio(tp, tp + fn),
            ratio(2.0 * tp, 2.0 * tp + fp + fn),
            ratio(fp, target_negative),
            ratio(fn, target_positive),
        ),
        axis=-1,
    )


def _summary(
    values: np.ndarray,
    point: np.ndarray,
    names: tuple[str, str, str, str, str],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        finite = values[np.isfinite(values[:, index]), index]
        if not math.isfinite(float(point[index])):
            raise ValueError(f"Pooled {name} is undefined")
        if len(finite) < max(100, int(0.95 * len(values))):
            raise ValueError(
                f"Too few defined bootstrap replicates for {name}: {len(finite)}/{len(values)}"
            )
        output[name] = {
            "delta": float(point[index]),
            "ci_low": float(np.quantile(finite, 0.025)),
            "ci_high": float(np.quantile(finite, 0.975)),
            "valid_resamples": int(len(finite)),
        }
    return output


def _paired_count_comparison(
    baseline_counts: np.ndarray,
    candidate_counts: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
    chunk_size: int = 512,
    false_positive_name: str = "false_close",
    false_negative_name: str = "missed_close",
) -> dict[str, Any]:
    if baseline_counts.shape != candidate_counts.shape:
        raise ValueError("Count arrays are not matched")
    if baseline_counts.ndim != 2 or baseline_counts.shape[1] != len(COUNT_FIELDS):
        raise ValueError("Count arrays have an invalid shape")
    if not np.array_equal(baseline_counts[:, 3:], candidate_counts[:, 3:]):
        raise ValueError("Matched arms have different target-positive/negative denominators")
    point_baseline = _count_metrics(baseline_counts.sum(axis=0))
    point_candidate = _count_metrics(candidate_counts.sum(axis=0))
    metric_names = (
        "precision",
        "recall",
        "f1",
        false_positive_name,
        false_negative_name,
    )
    bootstrap = np.empty((resamples, 5), dtype=np.float64)
    cluster_count = baseline_counts.shape[0]
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        left = baseline_counts[indices].sum(axis=1)
        right = candidate_counts[indices].sum(axis=1)
        bootstrap[start:stop] = _count_metrics(right) - _count_metrics(left)
    return {
        "aggregate_counts": {
            "baseline": dict(zip(COUNT_FIELDS, map(int, baseline_counts.sum(axis=0)))),
            "candidate": dict(zip(COUNT_FIELDS, map(int, candidate_counts.sum(axis=0)))),
        },
        "baseline": dict(
            zip(metric_names, map(float, point_baseline))
        ),
        "candidate": dict(
            zip(metric_names, map(float, point_candidate))
        ),
        "candidate_minus_baseline": _summary(
            bootstrap,
            point_candidate - point_baseline,
            metric_names,
        ),
    }


def _cluster_means(
    grouped: dict[str, list[dict[str, Any]]],
    uids: Iterable[str],
    metric: str,
) -> np.ndarray:
    values = []
    for uid in uids:
        local = np.asarray([float(row[metric]) for row in grouped[uid]], dtype=np.float64)
        if not np.isfinite(local).all():
            raise ValueError(f"Non-finite {metric} for UID {uid}")
        values.append(float(local.mean()))
    return np.asarray(values, dtype=np.float64)


def _paired_mean_comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, float]:
    delta = candidate - baseline
    indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "baseline": float(baseline.mean()),
        "candidate": float(candidate.mean()),
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def _cluster_ratio_components(
    grouped: dict[str, list[dict[str, Any]]],
    uids: Iterable[str],
    numerator_field: str,
    denominator_field: str,
) -> np.ndarray:
    output = []
    for uid in uids:
        numerator = 0.0
        denominator = 0.0
        for row in grouped[uid]:
            local_numerator = float(row[numerator_field])
            local_denominator = float(row[denominator_field])
            if (
                not math.isfinite(local_numerator)
                or not math.isfinite(local_denominator)
                or local_numerator < 0.0
                or local_denominator < 0.0
            ):
                raise ValueError(
                    f"Invalid pooled metric components for UID {uid}: "
                    f"{numerator_field}={local_numerator}, "
                    f"{denominator_field}={local_denominator}"
                )
            numerator += local_numerator
            denominator += local_denominator
        output.append((numerator, denominator))
    return np.asarray(output, dtype=np.float64)


def _paired_ratio_comparison(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.ndim != 2 or baseline.shape[1] != 2:
        raise ValueError("Pooled ratio components are not matched [N,2] arrays")
    if not np.array_equal(baseline[:, 1], candidate[:, 1]):
        raise ValueError("Matched pooled metrics have different GT-derived denominators")

    def ratio(values: np.ndarray) -> np.ndarray:
        return np.divide(
            values[..., 0],
            values[..., 1],
            out=np.full(values.shape[:-1], np.nan, dtype=np.float64),
            where=values[..., 1] > 0,
        )

    point_baseline = float(ratio(baseline.sum(axis=0)))
    point_candidate = float(ratio(candidate.sum(axis=0)))
    if not math.isfinite(point_baseline) or not math.isfinite(point_candidate):
        raise ValueError("Pooled ratio is undefined for the complete matched selection")
    cluster_count = baseline.shape[0]
    indices = rng.integers(0, cluster_count, size=(resamples, cluster_count))
    bootstrap = ratio(candidate[indices].sum(axis=1)) - ratio(
        baseline[indices].sum(axis=1)
    )
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    if len(bootstrap) < max(100, int(0.95 * resamples)):
        raise ValueError(
            f"Too few defined pooled-ratio bootstrap replicates: "
            f"{len(bootstrap)}/{resamples}"
        )
    return {
        "baseline": point_baseline,
        "candidate": point_candidate,
        "candidate_minus_baseline": point_candidate - point_baseline,
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "lower_is_better": True,
        "valid_resamples": int(len(bootstrap)),
        "aggregate_components": {
            "baseline_numerator": float(baseline[:, 0].sum()),
            "candidate_numerator": float(candidate[:, 0].sum()),
            "shared_denominator": float(baseline[:, 1].sum()),
        },
    }


def _causal_advantages(
    candidate: dict[str, Any],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, Any]:
    correct = _rows_by_uid(candidate, "source_text")
    output: dict[str, Any] = {}
    for variant in CAUSAL_VARIANTS:
        comparator = _rows_by_uid(candidate, variant)
        uids = _validate_matched_rows(correct, comparator)
        mean_metrics = {}
        for metric in ("reactor_fk_mpjpe_cm", "fk_relation_distance_mae_cm"):
            correct_values = _cluster_means(correct, uids, metric)
            comparator_values = _cluster_means(comparator, uids, metric)
            # Positive means the correct source+text branch has lower error.
            comparison = _paired_mean_comparison(
                correct_values,
                comparator_values,
                rng=rng,
                resamples=resamples,
            )
            mean_metrics[metric] = {
                "advantage": comparison["delta"],
                "ci_low": comparison["ci_low"],
                "ci_high": comparison["ci_high"],
            }
        correct_counts = _cluster_counts(correct, uids, "fk_", 20)
        comparator_counts = _cluster_counts(comparator, uids, "fk_", 20)
        count_comparison = _paired_count_comparison(
            comparator_counts,
            correct_counts,
            rng=rng,
            resamples=resamples,
        )
        output[variant] = {
            **mean_metrics,
            "fk_close_20cm_f1": count_comparison["candidate_minus_baseline"]["f1"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline_label", default="baseline")
    parser.add_argument("--candidate_label", default="candidate")
    parser.add_argument("--baseline_predictions", type=Path)
    parser.add_argument("--candidate_predictions", type=Path)
    parser.add_argument("--variant", choices=("source_text",), default="source_text")
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--expected_checkpoint_step",
        type=int,
        default=DEFAULT_CHECKPOINT_STEP,
    )
    parser.add_argument("--baseline_checkpoint_step", type=int)
    parser.add_argument("--candidate_checkpoint_step", type=int)
    parser.add_argument(
        "--expected_split",
        choices=tuple(EXPECTED_SPLIT_COUNTS),
        default="val",
    )
    parser.add_argument(
        "--training_contract",
        choices=(
            "p_only_ablation",
            "reaction_v3_adaptive",
            "reaction_v4_layout",
            "reaction_v5_1_full_contact",
            "same_run_dose_extension",
        ),
        default="p_only_ablation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1000:
        parser.error("bootstrap_resamples must be at least 1000 for a 95% interval")
    baseline_checkpoint_step = (
        args.baseline_checkpoint_step
        if args.baseline_checkpoint_step is not None
        else args.expected_checkpoint_step
    )
    candidate_checkpoint_step = (
        args.candidate_checkpoint_step
        if args.candidate_checkpoint_step is not None
        else args.expected_checkpoint_step
    )
    if baseline_checkpoint_step <= 0 or candidate_checkpoint_step <= 0:
        parser.error("checkpoint steps must be positive")

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    _validate_protocol(
        baseline,
        candidate,
        baseline_checkpoint_step=baseline_checkpoint_step,
        candidate_checkpoint_step=candidate_checkpoint_step,
        expected_split=args.expected_split,
    )
    prediction_args = (args.baseline_predictions, args.candidate_predictions)
    if (prediction_args[0] is None) != (prediction_args[1] is None):
        parser.error(
            "--baseline_predictions and --candidate_predictions must be provided together"
        )
    if (
        args.training_contract
        in {
            "reaction_v4_layout",
            "reaction_v5_1_full_contact",
            "same_run_dose_extension",
        }
        and prediction_args[0] is None
    ):
        parser.error(
            f"{args.training_contract} comparison requires both prediction directories so "
            "new phase metrics are recomputed under one implementation"
        )
    prediction_input_identity: dict[str, Any] | None = None
    if prediction_args[0] is not None and prediction_args[1] is not None:
        prediction_input_identity = _validate_prediction_input_identity(
            baseline,
            candidate,
            prediction_args[0],
            prediction_args[1],
        )
        baseline = _recompute_metrics_from_predictions(baseline, prediction_args[0])
        candidate = _recompute_metrics_from_predictions(candidate, prediction_args[1])
    training_contract = _validate_training_contract(
        baseline,
        candidate,
        mode=args.training_contract,
        baseline_checkpoint_step=baseline_checkpoint_step,
        candidate_checkpoint_step=candidate_checkpoint_step,
    )
    baseline_rows = _rows_by_uid(baseline, args.variant)
    candidate_rows = _rows_by_uid(candidate, args.variant)
    uids = _validate_matched_rows(baseline_rows, candidate_rows)
    rng = np.random.default_rng(args.seed)

    close_metrics: dict[str, Any] = {}
    for name, prefix in (("position", ""), ("fk", "fk_")):
        close_metrics[name] = {}
        for threshold_cm in THRESHOLDS_CM:
            left = _cluster_counts(baseline_rows, uids, prefix, threshold_cm)
            right = _cluster_counts(candidate_rows, uids, prefix, threshold_cm)
            close_metrics[name][f"{threshold_cm}cm"] = _paired_count_comparison(
                left,
                right,
                rng=rng,
                resamples=args.bootstrap_resamples,
            )

    pair_contact_metrics = _paired_count_comparison(
        _cluster_counts(
            baseline_rows,
            uids,
            "fk_pair_",
            15,
            pairs_per_frame=22 * 22,
        ),
        _cluster_counts(
            candidate_rows,
            uids,
            "fk_pair_",
            15,
            pairs_per_frame=22 * 22,
        ),
        rng=rng,
        resamples=args.bootstrap_resamples,
    )
    pair_transition_metrics = _paired_count_comparison(
        _cluster_counts(
            baseline_rows,
            uids,
            "fk_pair_",
            15,
            event_name="transition",
            pairs_per_frame=22 * 22,
            frame_offset=1,
            event_channels=2,
        ),
        _cluster_counts(
            candidate_rows,
            uids,
            "fk_pair_",
            15,
            event_name="transition",
            pairs_per_frame=22 * 22,
            frame_offset=1,
            event_channels=2,
        ),
        rng=rng,
        resamples=args.bootstrap_resamples,
        false_positive_name="false_transition",
        false_negative_name="missed_transition",
    )

    mean_metrics = {}
    for metric in MEAN_METRICS:
        left = _cluster_means(baseline_rows, uids, metric)
        right = _cluster_means(candidate_rows, uids, metric)
        mean_metrics[metric] = _paired_mean_comparison(
            left,
            right,
            rng=rng,
            resamples=args.bootstrap_resamples,
        )

    layout_phase_pooled_metrics = {}
    for metric, (numerator_field, denominator_field) in (
        LAYOUT_PHASE_POOLED_METRICS.items()
    ):
        left = _cluster_ratio_components(
            baseline_rows,
            uids,
            numerator_field,
            denominator_field,
        )
        right = _cluster_ratio_components(
            candidate_rows,
            uids,
            numerator_field,
            denominator_field,
        )
        layout_phase_pooled_metrics[metric] = _paired_ratio_comparison(
            left,
            right,
            rng=rng,
            resamples=args.bootstrap_resamples,
        )

    contact_lifecycle_pooled_metrics = {}
    for metric, (numerator_field, denominator_field) in (
        CONTACT_LIFECYCLE_POOLED_METRICS.items()
    ):
        left = _cluster_ratio_components(
            baseline_rows,
            uids,
            numerator_field,
            denominator_field,
        )
        right = _cluster_ratio_components(
            candidate_rows,
            uids,
            numerator_field,
            denominator_field,
        )
        contact_lifecycle_pooled_metrics[metric] = _paired_ratio_comparison(
            left,
            right,
            rng=rng,
            resamples=args.bootstrap_resamples,
        )

    result = {
        "format": "hy273_reaction_matched_uid_cluster_comparison_v2",
        "baseline": {"label": args.baseline_label, "path": str(args.baseline.resolve())},
        "candidate": {"label": args.candidate_label, "path": str(args.candidate.resolve())},
        "training_contract": training_contract,
        "variant": args.variant,
        "split": baseline["split"],
        "checkpoint_next_global_step": (
            candidate_checkpoint_step
            if baseline_checkpoint_step == candidate_checkpoint_step
            else None
        ),
        "checkpoint_steps": {
            "baseline": baseline_checkpoint_step,
            "candidate": candidate_checkpoint_step,
        },
        "uid_clusters": len(uids),
        "rows": sum(len(baseline_rows[uid]) for uid in uids),
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "unit": "uid_cluster",
            "interval": "paired_percentile_95",
            "numpy_version": np.__version__,
        },
        "close_metrics": close_metrics,
        "contact_lifecycle_metrics": {
            "fk_pair_close_15cm": pair_contact_metrics,
            "fk_pair_transition_15cm": pair_transition_metrics,
            "pooled_over_gt_contact_pairs": contact_lifecycle_pooled_metrics,
        },
        "mean_metrics": mean_metrics,
        "layout_phase_metrics": {
            "per_clip_means": {
                metric: mean_metrics[metric] for metric in LAYOUT_PHASE_MEAN_METRICS
            },
            "pooled_over_valid_precontact_frames": layout_phase_pooled_metrics,
        },
        "metric_recomputation": {
            "baseline": baseline.get("matched_metric_recomputation"),
            "candidate": candidate.get("matched_metric_recomputation"),
        },
        "prediction_input_identity": prediction_input_identity,
        "candidate_causal_advantage": _causal_advantages(
            candidate,
            rng=rng,
            resamples=args.bootstrap_resamples,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "uids": len(uids)}))


if __name__ == "__main__":
    main()
