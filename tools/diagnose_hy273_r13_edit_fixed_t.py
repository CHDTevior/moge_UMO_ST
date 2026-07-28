#!/usr/bin/env python3
"""Probe R13 Edit clean predictions at fixed rectified-flow timesteps.

This is a research diagnostic. It deliberately bypasses the ODE sampler so a
bad low-t denoising field can be separated from CFG and integration effects.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_multitask_condition import ConditionBatch
from models.raw_motion.hy273_unified_edit_losses import (
    build_source_target_discrepancy_mask,
)
from models.raw_motion.hy273_slices import (
    CONTACT_JOINTS,
    CONTACT_SLICE,
    DIM_HY273,
    reconstruct_global_joints_from_features,
)
from sample_hy273_multitask import (
    make_edit_condition,
    normalizer_from_checkpoint,
)
from train_hy273_multitask import create_model_from_checkpoint, repeat_condition_batch


DEFAULT_PAIR_IDS = (
    # Large target-motion increases.
    "000038",
    "000472",
    "002143",
    "002173",
    # Large target-motion decreases / static targets.
    "002024",
    "003165",
    "003485",
    "003896",
)
DEFAULT_TIMESTEPS = (0.0, 0.05, 0.1, 0.3, 0.6, 0.9)
SOURCE_MODES = ("correct", "donor", "absent")
TEXT_MODES = (
    "correct",
    "same_direction_donor",
    "opposite_direction_donor",
    "empty",
)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(",") if token.strip())


def concat_condition_batches(rows: list[ConditionBatch]) -> ConditionBatch:
    if not rows:
        raise ValueError("Cannot concatenate an empty condition list")
    values: dict[str, Any] = {}
    for field_name in rows[0].__dataclass_fields__:
        field_rows = [getattr(row, field_name) for row in rows]
        first = field_rows[0]
        if torch.is_tensor(first):
            values[field_name] = torch.cat(field_rows, dim=0)
        elif first is None:
            if any(value is not None for value in field_rows):
                raise ValueError(f"Mixed optional condition field {field_name}")
            values[field_name] = None
        elif field_name == "text_encoding_profile":
            values[field_name] = sum((tuple(value) for value in field_rows), ())
        else:
            raise TypeError(f"Unsupported ConditionBatch field {field_name}")
    output = replace(rows[0], **values)
    output.validate(v1_strict=False)
    return output


def load_rows(manifest: Path, pair_ids: Iterable[str]) -> list[dict[str, Any]]:
    requested = tuple(pair_ids)
    by_id: dict[str, dict[str, Any]] = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") != "motionfix_k273":
                continue
            pair_id = str(row["pair"]["official_pair_id"])
            if pair_id in requested:
                by_id[pair_id] = row
    missing = [pair_id for pair_id in requested if pair_id not in by_id]
    if missing:
        raise ValueError(f"Pair ids missing from {manifest}: {missing}")
    return [by_id[pair_id] for pair_id in requested]


def load_k273(ref: dict[str, Any]) -> torch.Tensor:
    value = np.load(ref["path"])
    expected = (int(ref["frames"]), DIM_HY273)
    if value.shape != expected or not np.isfinite(value).all():
        raise ValueError(f"Invalid K273 asset {ref['path']}: {value.shape}")
    return torch.from_numpy(value.astype(np.float32, copy=False)).clone()


def to_gauge(motion: torch.Tensor, phi: float = 0.0) -> torch.Tensor:
    shifted = root_origin_shift(motion)
    heading = shifted[0, 3:5]
    current = torch.atan2(heading[1], heading[0])
    return apply_yaw_rotation(shifted, torch.as_tensor(phi) - current)


def pad_motion(
    rows: list[dict[str, Any]], *, phi: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sources = [
        to_gauge(load_k273(row["source_motion"]["k273_asset"]), phi=phi)
        for row in rows
    ]
    targets = [
        to_gauge(load_k273(row["target_motion"]["k273_asset"]), phi=phi)
        for row in rows
    ]
    source_lengths = torch.tensor([value.shape[0] for value in sources], dtype=torch.long)
    target_lengths = torch.tensor([value.shape[0] for value in targets], dtype=torch.long)
    source_batch = torch.zeros(
        len(rows), int(source_lengths.max().item()), DIM_HY273
    )
    target_batch = torch.zeros(
        len(rows), int(target_lengths.max().item()), DIM_HY273
    )
    for index, (source, target) in enumerate(zip(sources, targets)):
        source_batch[index, : source.shape[0]] = source
        target_batch[index, : target.shape[0]] = target
    return source_batch, target_batch, source_lengths, target_lengths


def pad_sources(
    rows: list[dict[str, Any]], *, phi: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    sources = [
        to_gauge(load_k273(row["source_motion"]["k273_asset"]), phi=phi)
        for row in rows
    ]
    lengths = torch.tensor([value.shape[0] for value in sources], dtype=torch.long)
    batch = torch.zeros(len(rows), int(lengths.max().item()), DIM_HY273)
    for index, source in enumerate(sources):
        batch[index, : source.shape[0]] = source
    return batch, lengths


def motion_descriptors(motion: torch.Tensor, length: int) -> dict[str, float]:
    value = motion[:length].float()
    joints = reconstruct_global_joints_from_features(value)
    if length > 1:
        joint_velocity = (joints[1:] - joints[:-1]) * 30.0
        joint_speed = torch.linalg.vector_norm(joint_velocity, dim=-1)
        root_path = torch.linalg.vector_norm(
            value[1:, (0, 2)] - value[:-1, (0, 2)], dim=-1
        ).sum()
        mean_joint_speed = joint_speed.mean()
        mean_foot_speed = joint_speed[:, list(CONTACT_JOINTS)].mean()
    else:
        root_path = value.new_zeros(())
        mean_joint_speed = value.new_zeros(())
        mean_foot_speed = value.new_zeros(())
    contacts = value[..., CONTACT_SLICE]
    contact_binary = contacts > 0.5
    return {
        "mean_joint_speed_mps": float(mean_joint_speed.item()),
        "mean_foot_speed_mps": float(mean_foot_speed.item()),
        "root_path_m": float(root_path.item()),
        "contact_mean_physical": float(contacts.mean().item()),
        "contact_occupancy": float(contact_binary.float().mean().item()),
        "all_four_contact_ratio": float(contact_binary.all(dim=-1).float().mean().item()),
    }


def build_directional_donor_map(
    pair_ids: tuple[str, ...], direction_by_id: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Build a batch-order-independent map for source and directional text donors."""

    if len(pair_ids) != len(set(pair_ids)) or len(pair_ids) < 4:
        raise ValueError("Directional counterfactuals require at least four unique pairs")
    unknown = set(pair_ids) - set(direction_by_id)
    if unknown:
        raise ValueError(f"Missing direction labels for pairs: {sorted(unknown)}")
    groups = {
        direction: [pair_id for pair_id in pair_ids if direction_by_id[pair_id] == direction]
        for direction in ("increase", "decrease")
    }
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError(
            "Directional counterfactuals require at least two increase and two decrease pairs"
        )
    opposite = {"increase": "decrease", "decrease": "increase"}
    output: dict[str, dict[str, str]] = {}
    for global_index, pair_id in enumerate(pair_ids):
        direction = direction_by_id[pair_id]
        own_group = groups[direction]
        own_index = own_group.index(pair_id)
        other_group = groups[opposite[direction]]
        source_donor_id = pair_ids[(global_index + 1) % len(pair_ids)]
        output[pair_id] = {
            "direction": direction,
            "source_donor_id": source_donor_id,
            "same_direction_text_donor_id": own_group[(own_index + 1) % len(own_group)],
            "opposite_direction_text_donor_id": other_group[own_index % len(other_group)],
        }
    return output


def infer_directional_donor_map(
    rows: list[dict[str, Any]], *, minimum_speed_delta: float = 1e-6
) -> dict[str, dict[str, str]]:
    direction_by_id: dict[str, str] = {}
    pair_ids = tuple(str(row["pair"]["official_pair_id"]) for row in rows)
    for row, pair_id in zip(rows, pair_ids):
        source = load_k273(row["source_motion"]["k273_asset"])
        target = load_k273(row["target_motion"]["k273_asset"])
        source_speed = motion_descriptors(source, source.shape[0])["mean_joint_speed_mps"]
        target_speed = motion_descriptors(target, target.shape[0])["mean_joint_speed_mps"]
        delta = target_speed - source_speed
        if abs(delta) <= float(minimum_speed_delta):
            raise ValueError(
                f"Pair {pair_id} has ambiguous speed direction delta={delta:.8g}"
            )
        direction_by_id[pair_id] = "increase" if delta > 0.0 else "decrease"
    return build_directional_donor_map(pair_ids, direction_by_id)


def yaw_pair_normalized_noise(
    base_noise_norm: torch.Tensor,
    normalizer: Any,
    phi: float,
) -> torch.Tensor:
    """Rotate one normalized base noise through physical HY273 space."""

    if float(phi) == 0.0:
        return base_noise_norm
    physical = normalizer.denormalize(base_noise_norm)
    angle = torch.as_tensor(phi, device=physical.device, dtype=physical.dtype)
    return normalizer.normalize(apply_yaw_rotation(physical, angle))


def target_errors(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    prediction_physical: torch.Tensor,
    target_physical: torch.Tensor,
    length: int,
    discrepancy_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    pred_norm = prediction_norm[:length].float()
    tgt_norm = target_norm[:length].float()
    pred = prediction_physical[:length].float()
    target = target_physical[:length].float()
    pred_joints = reconstruct_global_joints_from_features(pred)
    target_joints = reconstruct_global_joints_from_features(target)
    contact_pred = pred[..., CONTACT_SLICE] > 0.5
    contact_target = target[..., CONTACT_SLICE] > 0.5
    metrics = {
        "normalized_mse": float((pred_norm - tgt_norm).square().mean().item()),
        "continuous_normalized_mse": float(
            (pred_norm[..., :269] - tgt_norm[..., :269]).square().mean().item()
        ),
        "global_joint_error_m": float(
            torch.linalg.vector_norm(pred_joints - target_joints, dim=-1).mean().item()
        ),
        "contact_physical_mae": float(
            (pred[..., CONTACT_SLICE] - target[..., CONTACT_SLICE]).abs().mean().item()
        ),
        "contact_accuracy": float((contact_pred == contact_target).float().mean().item()),
    }
    if discrepancy_mask is not None:
        selected = discrepancy_mask[:length].to(device=pred_norm.device, dtype=torch.bool)
        if selected.shape != pred_norm[..., :269].shape:
            raise ValueError("discrepancy_mask must match the valid continuous target span")
        unselected = ~selected
        if not bool(selected.any()) or not bool(unselected.any()):
            raise ValueError("Selected-region diagnostics require non-empty mask and complement")
        squared = (pred_norm[..., :269] - tgt_norm[..., :269]).square()
        metrics.update(
            {
                "selected_continuous_normalized_mse": float(
                    squared.masked_select(selected).mean().item()
                ),
                "unselected_continuous_normalized_mse": float(
                    squared.masked_select(unselected).mean().item()
                ),
                "selected_continuous_fraction": float(selected.float().mean().item()),
            }
        )
    return metrics


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float, float, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (
            str(record["checkpoint"]),
            int(record["checkpoint_step"]),
            float(record["yaw_degrees"]),
            float(record["t"]),
            str(record["branch"]),
        )
        grouped.setdefault(key, []).append(record)
    output = []
    metric_names = tuple(records[0]["metrics"])
    for (checkpoint, step, yaw_degrees, timestep, branch), rows in sorted(
        grouped.items()
    ):
        output.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_step": step,
                "yaw_degrees": yaw_degrees,
                "t": timestep,
                "branch": branch,
                "samples": len(rows),
                "metrics_mean": {
                    name: float(np.mean([row["metrics"][name] for row in rows]))
                    for name in metric_names
                },
            }
        )
    return output


@torch.inference_mode()
def probe_checkpoint(
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    timesteps: tuple[float, ...],
    *,
    weight_source: str,
    device: torch.device,
    batch_size: int,
    seed: int,
    yaw_degrees: tuple[float, ...],
    donor_map: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    step = int(checkpoint["next_global_step"])
    model = create_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[weight_source], strict=True)
    normalizer = normalizer_from_checkpoint(checkpoint, device)
    del checkpoint
    model = model.to(device).eval()

    records: list[dict[str, Any]] = []
    context_rms_rows: list[float] = []
    row_by_id = {
        str(row["pair"]["official_pair_id"]): row
        for row in rows
    }
    for yaw_degree in yaw_degrees:
        phi = math.radians(float(yaw_degree))
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            if len(batch_rows) < 2:
                raise ValueError(
                    "Every fixed-t counterfactual batch needs at least two rows"
                )
            source, target, source_lengths, lengths = pad_motion(
                batch_rows, phi=phi
            )
            batch_pair_ids = [
                str(row["pair"]["official_pair_id"]) for row in batch_rows
            ]
            source_donor_rows = [
                row_by_id[donor_map[pair_id]["source_donor_id"]]
                for pair_id in batch_pair_ids
            ]
            donor_source, donor_source_lengths = pad_sources(
                source_donor_rows, phi=phi
            )
            source = source.to(device)
            target = target.to(device)
            donor_source = donor_source.to(device)
            lengths = lengths.to(device)
            source_lengths = source_lengths.to(device)
            frames = target.shape[1]
            valid = torch.arange(frames, device=device)[None] < lengths[:, None]
            discrepancy = build_source_target_discrepancy_mask(
                source_physical=source,
                source_lengths=source_lengths,
                target_physical=target,
                target_valid=valid,
                hard_mask=torch.zeros_like(target, dtype=torch.bool),
            )
            gauge = torch.empty(len(batch_rows), 2, device=device)
            gauge[:, 0] = math.cos(phi)
            gauge[:, 1] = math.sin(phi)

            correct_source = make_edit_condition(
                source,
                source_lengths=source_lengths,
                target_lengths=lengths,
                target_frames=frames,
                frame_gauge_dir=gauge,
            )
            donor_source_condition = make_edit_condition(
                donor_source,
                source_lengths=donor_source_lengths.to(device),
                target_lengths=lengths,
                target_frames=frames,
                frame_gauge_dir=gauge,
            )
            absent_source = replace(
                correct_source,
                source_motion=torch.zeros_like(correct_source.source_motion),
                source_present=torch.zeros_like(correct_source.source_present),
                source_time_valid=torch.zeros_like(correct_source.source_time_valid),
                source_value_mask=torch.zeros_like(correct_source.source_value_mask),
                source_role_id=torch.zeros_like(correct_source.source_role_id),
                source_native_lengths=torch.zeros_like(
                    correct_source.source_native_lengths
                ),
            )
            absent_source.validate(v1_strict=False)
            source_conditions = {
                "correct": correct_source,
                "donor": donor_source_condition,
                "absent": absent_source,
            }
            repeated_condition = concat_condition_batches(
                [
                    repeat_condition_batch(
                        source_conditions[source_mode], len(TEXT_MODES), v1_strict=False
                    )
                    for source_mode in SOURCE_MODES
                ]
            )

            target_norm = normalizer.normalize(target)
            generator = torch.Generator(device=device).manual_seed(
                int(seed) + batch_start * 1009
            )
            base_noise = torch.randn(
                target_norm.shape,
                device=device,
                dtype=target_norm.dtype,
                generator=generator,
            )
            noise = yaw_pair_normalized_noise(base_noise, normalizer, phi)
            texts = [str(row["texts"][0]["value"]) for row in batch_rows]
            same_direction_texts = [
                str(
                    row_by_id[
                        donor_map[pair_id]["same_direction_text_donor_id"]
                    ]["texts"][0]["value"]
                )
                for pair_id in batch_pair_ids
            ]
            opposite_direction_texts = [
                str(
                    row_by_id[
                        donor_map[pair_id]["opposite_direction_text_donor_id"]
                    ]["texts"][0]["value"]
                )
                for pair_id in batch_pair_ids
            ]
            text_rows = {
                "correct": texts,
                "same_direction_donor": same_direction_texts,
                "opposite_direction_donor": opposite_direction_texts,
                "empty": [""] * len(texts),
            }
            branch_names = tuple(
                f"source_{source_mode}/text_{text_mode}"
                for source_mode in SOURCE_MODES
                for text_mode in TEXT_MODES
            )
            branch_texts = [
                text
                for _source_mode in SOURCE_MODES
                for text_mode in TEXT_MODES
                for text in text_rows[text_mode]
            ]
            branch_count = len(branch_names)

            for timestep in timesteps:
                t = torch.full(
                    (len(batch_rows),),
                    float(timestep),
                    device=device,
                    dtype=target_norm.dtype,
                )
                z = t[:, None, None] * target_norm + (1.0 - t[:, None, None]) * noise
                model_in = torch.cat([z, torch.zeros_like(z)], dim=-1)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    details = model(
                        torch.cat([model_in] * branch_count, dim=0),
                        t=torch.cat([t] * branch_count, dim=0),
                        c_dir=torch.cat([gauge] * branch_count, dim=0),
                        text=branch_texts,
                        length_mask=torch.cat([valid] * branch_count, dim=0),
                        x_self_cond=None,
                        text_drop_prob=0.0,
                        condition=repeated_condition,
                        return_details=True,
                    )
                prediction_rows = details.prediction.float().chunk(
                    branch_count, dim=0
                )
                predictions = dict(zip(branch_names, prediction_rows))
                context_rms_rows.append(
                    float(details.context_body.float().square().mean().sqrt().item())
                )
                reference_source_only = predictions[
                    "source_correct/text_empty"
                ]
                reference_correct = predictions[
                    "source_correct/text_correct"
                ]
                for branch, prediction_norm in predictions.items():
                    source_mode, text_mode = (
                        token.removeprefix(prefix)
                        for token, prefix in zip(
                            branch.split("/"), ("source_", "text_")
                        )
                    )
                    prediction = normalizer.denormalize(prediction_norm)
                    for index, row in enumerate(batch_rows):
                        length = int(lengths[index].item())
                        descriptors = motion_descriptors(prediction[index], length)
                        errors = target_errors(
                            prediction_norm[index],
                            target_norm[index],
                            prediction[index],
                            target[index],
                            length,
                            discrepancy.mask[index],
                        )
                        source_only_residual = (
                            prediction_norm[index, :length]
                            - reference_source_only[index, :length]
                        ).float()
                        correct_residual = (
                            prediction_norm[index, :length]
                            - reference_correct[index, :length]
                        ).float()
                        records.append(
                            {
                                "checkpoint_step": step,
                                "checkpoint": str(checkpoint_path),
                                "pair_id": str(row["pair"]["official_pair_id"]),
                                "instruction": str(row["texts"][0]["value"]),
                                "evaluated_text": text_rows[text_mode][index],
                                "edit_direction": donor_map[
                                    batch_pair_ids[index]
                                ]["direction"],
                                "source_donor_id": (
                                    donor_map[batch_pair_ids[index]]["source_donor_id"]
                                    if source_mode == "donor"
                                    else None
                                ),
                                "text_donor_id": (
                                    donor_map[batch_pair_ids[index]][
                                        {
                                            "same_direction_donor": "same_direction_text_donor_id",
                                            "opposite_direction_donor": "opposite_direction_text_donor_id",
                                        }[text_mode]
                                    ]
                                    if text_mode
                                    in {"same_direction_donor", "opposite_direction_donor"}
                                    else None
                                ),
                                "yaw_degrees": float(yaw_degree),
                                "t": float(timestep),
                                "branch": branch,
                                "source_condition": source_mode,
                                "text_condition": text_mode,
                                "metrics": {
                                    **errors,
                                    **descriptors,
                                    "rms_from_source_only_norm": float(
                                        source_only_residual.square().mean().sqrt().item()
                                    ),
                                    "rms_from_correct_norm": float(
                                        correct_residual.square().mean().sqrt().item()
                                    ),
                                },
                            }
                        )

    model_info = {
        "checkpoint_step": step,
        "weight_source": weight_source,
        "context_body_rms": float(np.mean(context_rms_rows)),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    torch.cuda.empty_cache()
    return records, model_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--manifest",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
        "hy273_multitask_v1/test.jsonl",
    )
    parser.add_argument("--pair_ids", default=",".join(DEFAULT_PAIR_IDS))
    parser.add_argument(
        "--timesteps", default=",".join(str(value) for value in DEFAULT_TIMESTEPS)
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--yaw_degrees",
        default="0,90",
        help="Comma-separated shared source/target gauges for yaw-equivariance checks.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weight_source", choices=("ema", "model"), default="ema")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pair_ids = parse_csv(args.pair_ids)
    timesteps = tuple(float(value) for value in parse_csv(args.timesteps))
    yaw_degrees = tuple(float(value) for value in parse_csv(args.yaw_degrees))
    if any(not 0.0 <= value <= 1.0 for value in timesteps):
        raise ValueError("All timesteps must be in [0,1]")
    if not yaw_degrees:
        raise ValueError("At least one yaw gauge is required")
    rows = load_rows(Path(args.manifest), pair_ids)
    donor_map = infer_directional_donor_map(rows)
    device = torch.device(args.device)

    all_records: list[dict[str, Any]] = []
    model_info: list[dict[str, Any]] = []
    for value in args.checkpoint:
        records, info = probe_checkpoint(
            Path(value).expanduser().resolve(),
            rows,
            timesteps,
            weight_source=str(args.weight_source),
            device=device,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            yaw_degrees=yaw_degrees,
            donor_map=donor_map,
        )
        all_records.extend(records)
        model_info.append(info)

    source_target = []
    for row in rows:
        source = to_gauge(load_k273(row["source_motion"]["k273_asset"]))
        target = to_gauge(load_k273(row["target_motion"]["k273_asset"]))
        source_target.append(
            {
                "pair_id": str(row["pair"]["official_pair_id"]),
                "instruction": str(row["texts"][0]["value"]),
                "source": motion_descriptors(source, source.shape[0]),
                "target": motion_descriptors(target, target.shape[0]),
            }
        )

    payload = {
        "format": "hy273_r13_edit_fixed_t_probe_v3",
        "pair_ids": list(pair_ids),
        "timesteps": list(timesteps),
        "seed": int(args.seed),
        "weight_source": str(args.weight_source),
        "yaw_degrees": list(yaw_degrees),
        "noise_pairing": "denormalize_yaw_rotate_normalize_v1",
        "donor_map": donor_map,
        "models": model_info,
        "source_target_descriptors": source_target,
        "aggregate": aggregate(all_records),
        "records": all_records,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "records": len(all_records)}, indent=2))


if __name__ == "__main__":
    main()
