#!/usr/bin/env python
"""Fixed-noise causal sweep for HY273 Ease-in/out conditioning."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_ease import (  # noqa: E402
    HY273EaseNormalizer,
    ease_from_centroid_trajectory,
    ease_from_k273,
)
from models.raw_motion.hy273_kimodo_benchmark import (  # noqa: E402
    KIMODO_CONTROL_SUBTYPES,
    CompiledKimodoConstraint,
    compile_kimodo_constraint,
    evaluate_kimodo_constraint_case,
)
from models.raw_motion.hy273_multitask_condition import (  # noqa: E402
    ABSOLUTE_TEXT_PROFILE,
    CapabilityId,
    make_absent_condition,
)
from models.raw_motion.hy273_normalizer import (  # noqa: E402
    apply_kimodo_training_transform,
)
from models.raw_motion.hy273_slices import (  # noqa: E402
    DIM_HY273,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)
from sample_hy273_multitask import (  # noqa: E402
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from tools.hy273_runtime_text_encoding import (  # noqa: E402
    encode_missing_text_rows,
    register_runtime_text_rows,
)
from train_hy273_multitask import (  # noqa: E402
    create_model_from_checkpoint,
    validate_frozen_contract,
)


SWEEP_FORMAT = "hy273_ease_fixed_noise_sweep_v1"
CONTROL_METRIC_KEYS = (
    "constraint_root2d_err",
    "constraint_root2d_acc",
    "constraint_fullbody_keyframe",
    "constraint_end_effector",
    "constraint_end_effector_rotation_deg",
)


def _average_ranks(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Rank values must be a finite non-empty vector")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(
    x: Iterable[float], y: Iterable[float]
) -> float | None:
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    if x_rank.shape != y_rank.shape or x_rank.size < 2:
        raise ValueError("Spearman inputs must have the same length >= 2")
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    if denominator == 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def _response_slope(x: Iterable[float], y: Iterable[float]) -> float:
    x_value = np.asarray(list(x), dtype=np.float64)
    y_value = np.asarray(list(y), dtype=np.float64)
    x_centered = x_value - x_value.mean()
    denominator = float(np.dot(x_centered, x_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_value - y_value.mean()) / denominator)


def build_sweep_requests(
    normalizer: HY273EaseNormalizer,
    target_physical: torch.Tensor,
    scales: Iterable[float],
) -> tuple[list[dict[str, object]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build absent plus independent normalized-space E_in/E_out sweeps."""

    scales = [float(value) for value in scales]
    if len(scales) < 3 or any(not np.isfinite(value) for value in scales):
        raise ValueError("Ease sweep needs at least three finite scales")
    if scales != sorted(set(scales)) or scales[0] < 0.0:
        raise ValueError("Ease scales must be unique, sorted, and non-negative")
    if 0.0 not in scales:
        raise ValueError("Ease scales must include the zero baseline")

    target_physical = torch.as_tensor(target_physical, dtype=torch.float32).reshape(6)
    target_normalized = normalizer.normalize(target_physical)
    for start in (0, 3):
        if float(target_normalized[start : start + 3].norm()) < 1e-6:
            raise ValueError(
                "Reference Ease is too close to the stats mean for a directional sweep"
            )

    rows: list[dict[str, object]] = [
        {"name": "ease_absent", "axis": "absent", "scale": None}
    ]
    absent_physical = torch.zeros(6, dtype=torch.float32)
    requested_normalized = [normalizer.normalize(absent_physical)]
    requested_physical = [absent_physical]
    present = [False]
    for axis, start in (("in", 0), ("out", 3)):
        for scale in scales:
            normalized = torch.zeros(6, dtype=torch.float32)
            normalized[start : start + 3] = (
                target_normalized[start : start + 3] * scale
            )
            rows.append(
                {
                    "name": f"ease_{axis}_scale_{scale:g}",
                    "axis": axis,
                    "scale": scale,
                }
            )
            requested_normalized.append(normalized)
            requested_physical.append(normalizer.denormalize(normalized))
            present.append(True)
    return (
        rows,
        torch.stack(requested_physical),
        torch.tensor(present, dtype=torch.bool),
        torch.stack(requested_normalized),
    )


def _load_reference(path: Path) -> torch.Tensor:
    value = torch.from_numpy(np.load(path)).float()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[-1] != DIM_HY273:
        raise ValueError(
            f"Reference must be [T,{DIM_HY273}], got {tuple(value.shape)}"
        )
    if value.shape[0] < 6 or value.shape[0] > 300:
        raise ValueError("Reference length must be in [6,300]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Reference contains non-finite values")
    return value


def _control_summary(
    samples: torch.Tensor,
    target: torch.Tensor,
    constraint: CompiledKimodoConstraint | None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    if constraint is None:
        return [{} for _ in range(samples.shape[0])], {}
    per_sample = [
        evaluate_kimodo_constraint_case(sample, target, constraint)
        for sample in samples
    ]
    summary = {
        key: float(np.mean([row[key] for row in per_sample]))
        for key in CONTROL_METRIC_KEYS
        if all(key in row for row in per_sample)
    }
    return per_sample, summary


def _axis_summary(
    axis: str,
    rows: list[dict[str, object]],
    target_normalized: torch.Tensor,
    requested_normalized: torch.Tensor,
    generated_normalized: torch.Tensor,
) -> dict[str, object]:
    indices = [index for index, row in enumerate(rows) if row["axis"] == axis]
    start = 0 if axis == "in" else 3
    other_start = 3 - start
    direction = target_normalized[start : start + 3]
    unit = direction / direction.norm()
    scales = [float(rows[index]["scale"]) for index in indices]
    projections = [
        float(torch.dot(generated_normalized[index, start : start + 3], unit))
        for index in indices
    ]
    zero_index = indices[scales.index(0.0)]
    zero_other = generated_normalized[zero_index, other_start : other_start + 3]
    active_mae = [
        float(
            (
                generated_normalized[index, start : start + 3]
                - requested_normalized[index, start : start + 3]
            )
            .abs()
            .mean()
        )
        for index in indices
    ]
    cross_half_drift = [
        float(
            (
                generated_normalized[index, other_start : other_start + 3]
                - zero_other
            ).norm()
        )
        for index in indices
    ]
    return {
        "scales": scales,
        "generated_direction_projection": projections,
        "spearman_scale_vs_projection": spearman_correlation(scales, projections),
        "linear_response_slope": _response_slope(scales, projections),
        "active_half_normalized_mae": active_mae,
        "active_half_normalized_mae_mean": float(np.mean(active_mae)),
        "cross_half_normalized_drift": cross_half_drift,
        "cross_half_normalized_drift_mean": float(np.mean(cross_half_drift)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_npy", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weight_source", choices=["ema", "model"], default="ema")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--text_cfg_scale", type=float, default=2.0)
    parser.add_argument("--control_cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(0.0, 0.5, 1.0, 1.5, 2.0),
    )
    parser.add_argument(
        "--control_subtype",
        choices=("none", *KIMODO_CONTROL_SUBTYPES),
        default="none",
    )
    parser.add_argument("--control_seed", type=int, default=3407)
    parser.add_argument("--max_sparse_keyframes", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint has no resolved multitask config")
    validate_frozen_contract(config)

    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint[args.weight_source], strict=True)
    model.eval()
    if not bool(getattr(model, "use_ease", False)):
        raise RuntimeError("Checkpoint model does not enable Ease conditioning")
    ease_conditioner = getattr(model, "ease_conditioner", None)
    if ease_conditioner is None:
        raise RuntimeError("Checkpoint model has no Ease conditioner")
    ease_normalizer = ease_conditioner.normalizer
    motion_normalizer = normalizer_from_checkpoint(checkpoint, device)
    if not motion_normalizer.normalize_contacts:
        raise RuntimeError("Ease sweep requires the unified 273D flow protocol")

    reference_path = Path(args.reference_npy).expanduser().resolve()
    reference = _load_reference(reference_path)
    transformed = apply_kimodo_training_transform(
        reference.unsqueeze(0),
        random_heading=False,
        root_shift=True,
    )
    target = transformed.motion[0].contiguous()
    frame_gauge_dir = transformed.c_dir[0].contiguous()
    target_ease_physical = ease_from_k273(target)
    target_fk_joints = fk_positions_from_global_rot6d(target)
    target_ease_fk_physical = ease_from_centroid_trajectory(
        target_fk_joints.mean(dim=-2)
    )
    (
        rows,
        requested_physical,
        ease_present,
        requested_normalized,
    ) = build_sweep_requests(
        ease_normalizer,
        target_ease_physical,
        args.scales,
    )

    count = len(rows)
    frames = int(target.shape[0])
    lengths = torch.full((count,), frames, dtype=torch.long)
    capability = (
        CapabilityId.T2M
        if args.control_subtype == "none"
        else CapabilityId.KIMODO_CONTROL
    )
    condition = make_absent_condition(
        batch_size=count,
        target_frames=frames,
        target_lengths=lengths,
        capability=capability,
    )
    condition = replace(
        condition,
        frame_gauge_dir=frame_gauge_dir[None].expand(count, 2).clone(),
        ease_physical=requested_physical,
        ease_present=ease_present,
    )
    condition.validate()

    constraint = None
    if args.control_subtype == "none":
        observed = torch.zeros(count, frames, DIM_HY273)
        hard_mask = torch.zeros_like(observed, dtype=torch.bool)
    else:
        constraint = compile_kimodo_constraint(
            target,
            args.control_subtype,
            seed=int(args.control_seed),
            max_sparse_keyframes=int(args.max_sparse_keyframes),
        )
        observed = constraint.observed_motion[None].expand(count, -1, -1).clone()
        hard_mask = constraint.motion_mask[None].expand(count, -1, -1).clone()

    texts = [str(args.text)] * count
    runtime_rows = encode_missing_text_rows(
        model.text_encoder.cache,
        texts,
        [ABSOLUTE_TEXT_PROFILE] * count,
        device,
    )
    register_runtime_text_rows(model.text_encoder.cache, runtime_rows)
    noise_generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    shared_noise = torch.randn(
        1,
        frames,
        DIM_HY273,
        generator=noise_generator,
        dtype=torch.float32,
    )
    initial_noise = shared_noise.expand(count, -1, -1).clone()

    sampled = sample_hy273_multitask_ode(
        model,
        motion_normalizer,
        condition,
        texts,
        observed,
        hard_mask,
        num_steps=int(args.num_steps),
        text_cfg_scale=float(args.text_cfg_scale),
        control_cfg_scale=float(args.control_cfg_scale),
        initial_unified_noise=initial_noise,
    )
    samples = sampled.raw_motion.detach().cpu().float()
    generated_physical = ease_from_k273(samples)
    generated_normalized = ease_normalizer.normalize(generated_physical)
    generated_position_joints = reconstruct_global_joints_from_features(samples)
    generated_fk_joints = fk_positions_from_global_rot6d(samples)
    generated_fk_physical = torch.stack(
        [
            ease_from_centroid_trajectory(joints.mean(dim=-2))
            for joints in generated_fk_joints
        ]
    )
    generated_fk_normalized = ease_normalizer.normalize(generated_fk_physical)
    position_fk_ease_mae = (
        generated_normalized - generated_fk_normalized
    ).abs().mean(dim=-1)
    position_fk_joint_consistency_cm = (
        (generated_position_joints - generated_fk_joints)
        .norm(dim=-1)
        .mean(dim=(1, 2))
        * 100.0
    )
    target_normalized = ease_normalizer.normalize(target_ease_physical)
    target_fk_normalized = ease_normalizer.normalize(target_ease_fk_physical)
    control_per_sample, control_aggregate = _control_summary(
        samples,
        target,
        constraint,
    )

    for index, row in enumerate(rows):
        row.update(
            {
                "requested_ease_physical": requested_physical[index].tolist(),
                "requested_ease_normalized": requested_normalized[index].tolist(),
                "requested_ease_defined": bool(ease_present[index]),
                "generated_ease_physical": generated_physical[index].tolist(),
                "generated_ease_normalized": generated_normalized[index].tolist(),
                "generated_ease_fk_physical": generated_fk_physical[
                    index
                ].tolist(),
                "generated_ease_fk_normalized": generated_fk_normalized[
                    index
                ].tolist(),
                "position_fk_ease_normalized_mae": float(
                    position_fk_ease_mae[index]
                ),
                "position_fk_joint_consistency_cm": float(
                    position_fk_joint_consistency_cm[index]
                ),
                "normalized_mae": (
                    float(
                        (
                            generated_normalized[index]
                            - requested_normalized[index]
                        )
                        .abs()
                        .mean()
                    )
                    if bool(ease_present[index])
                    else None
                ),
                "control_metrics": control_per_sample[index],
            }
        )

    summary = {
        "ease_in": _axis_summary(
            "in",
            rows,
            target_normalized,
            requested_normalized,
            generated_normalized,
        ),
        "ease_out": _axis_summary(
            "out",
            rows,
            target_normalized,
            requested_normalized,
            generated_normalized,
        ),
        "control_metrics_mean_across_sweep": control_aggregate,
        "fk_diagnostic": {
            "target_position_fk_ease_normalized_mae": float(
                (target_normalized - target_fk_normalized).abs().mean()
            ),
            "generated_position_fk_ease_normalized_mae_mean": float(
                position_fk_ease_mae.mean()
            ),
            "generated_position_fk_ease_normalized_mae_max": float(
                position_fk_ease_mae.max()
            ),
            "generated_position_fk_joint_consistency_cm_mean": float(
                position_fk_joint_consistency_cm.mean()
            ),
            "generated_position_fk_joint_consistency_cm_max": float(
                position_fk_joint_consistency_cm.max()
            ),
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "samples_raw.npy", samples.numpy())
    np.save(
        output_dir / "samples_exact_clamped.npy",
        sampled.exact_clamped_motion.detach().cpu().numpy(),
    )
    np.save(output_dir / "reference_transformed.npy", target.numpy())
    np.save(output_dir / "observed.npy", observed.numpy())
    np.save(output_dir / "mask.npy", hard_mask.numpy())
    np.save(output_dir / "initial_noise_shared.npy", shared_noise.numpy())
    np.save(output_dir / "requested_ease_physical.npy", requested_physical.numpy())
    np.save(output_dir / "requested_ease_normalized.npy", requested_normalized.numpy())
    np.save(output_dir / "generated_ease_physical.npy", generated_physical.numpy())
    np.save(output_dir / "generated_ease_normalized.npy", generated_normalized.numpy())
    np.save(
        output_dir / "generated_ease_fk_physical.npy",
        generated_fk_physical.numpy(),
    )
    np.save(
        output_dir / "generated_ease_fk_normalized.npy",
        generated_fk_normalized.numpy(),
    )
    payload = {
        "format": SWEEP_FORMAT,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("next_global_step", -1)),
        "weight_source": args.weight_source,
        "reference_npy": str(reference_path),
        "text": str(args.text),
        "length": frames,
        "seed": int(args.seed),
        "num_steps": int(args.num_steps),
        "text_cfg_scale": float(args.text_cfg_scale),
        "control_cfg_scale": float(args.control_cfg_scale),
        "control_subtype": args.control_subtype,
        "control_seed": int(args.control_seed),
        "runtime_encoded_text_rows": int(runtime_rows.count),
        "target_ease_physical": target_ease_physical.tolist(),
        "target_ease_normalized": target_normalized.tolist(),
        "target_ease_fk_physical": target_ease_fk_physical.tolist(),
        "target_ease_fk_normalized": target_fk_normalized.tolist(),
        "generated_ease_primary_space": "position_channel_centroid",
        "generated_ease_fk_role": "visible_motion_consistency_diagnostic",
        "sweep_definition": (
            "stats-mean baseline; vary one normalized Ease half along the "
            "reference direction while holding the other half at stats mean"
        ),
        "rows": rows,
        "summary": summary,
        "sampling_protocol": sampled.protocol,
    }
    (output_dir / "ease_sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_step": payload["checkpoint_step"],
                "ease_in_spearman": summary["ease_in"][
                    "spearman_scale_vs_projection"
                ],
                "ease_out_spearman": summary["ease_out"][
                    "spearman_scale_vs_projection"
                ],
                "control_metrics": control_aggregate,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
