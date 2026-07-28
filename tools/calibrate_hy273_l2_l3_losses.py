#!/usr/bin/env python3
"""Calibrate L2/L3 loss scales against a fixed HY273 checkpoint trace."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text
from models.raw_motion.flow_schedule import bce_logits_masked, build_flow_state, sample_timesteps
from models.raw_motion.hy273_normalizer import (
    HY273Normalizer,
    apply_kimodo_training_transform,
)
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM
from train_hy273_raw_flow import (
    apply_deterministic_text_dropout,
    build_arg_parser,
    compute_clean_semantic_losses,
    create_model,
    fk_position_consistency_loss,
    hash_model_state,
    load_yaml,
    make_trace_generator,
    merge_config,
    representation_loss_pair,
    representation_mse_loss,
    seed_all,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--scratch_config")
    parser.add_argument("--scratch_seed", type=int, default=3407)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--text_root", default="")
    parser.add_argument("--hytext_cache_dir", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--batches_per_bin", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--dataset_seed", type=int, default=3407)
    parser.add_argument("--trace_seed", type=int, default=3407)
    parser.add_argument("--trace_epoch", type=int, default=-1)
    parser.add_argument("--t_bins", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--lambda_candidates", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--target_ratio", type=float, default=0.075)
    parser.add_argument("--min_ratio", type=float, default=0.05)
    parser.add_argument("--max_ratio", type=float, default=0.10)
    parser.add_argument("--max_bin_ratio", type=float, default=0.15)
    parser.add_argument(
        "--selection_mode",
        choices=("aggregate_and_max_bin", "max_bin_only"),
        default="aggregate_and_max_bin",
        help=(
            "Use the aggregate target band plus the per-bin ceiling, or treat a "
            "fixed-timestep run as a ceiling-only audit."
        ),
    )
    parser.add_argument("--consistency_scale_m", type=float, default=0.05)
    parser.add_argument(
        "--calibration_loss_space",
        choices=["x0", "velocity"],
        default="x0",
    )
    parser.add_argument("--velocity_loss_t_eps", type=float, default=0.05)
    parser.add_argument("--sample_training_timesteps", action="store_true")
    parser.add_argument("--representation_scale_override", type=float, default=0.0)
    return parser.parse_args()


def _float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float")
    return values


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt().item())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_f = left.detach().float().reshape(-1)
    right_f = right.detach().float().reshape(-1)
    denom = left_f.norm() * right_f.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float(torch.dot(left_f, right_f).div(denom).item())


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(statistics.fmean(ordered)),
        "median": float(statistics.median(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _aggregate_rms(rows: list[dict[str, Any]], key: str) -> float:
    total_elements = sum(int(row["gradient_elements"]) for row in rows)
    if total_elements <= 0:
        return math.nan
    squared_sum = sum(
        float(row[key]) ** 2 * int(row["gradient_elements"]) for row in rows
    )
    return math.sqrt(squared_sum / float(total_elements))


def _namespace_from_checkpoint(
    checkpoint_args: dict[str, Any], cli: argparse.Namespace
) -> argparse.Namespace:
    resolved = dict(checkpoint_args)
    if cli.data_root:
        resolved["data_root"] = cli.data_root
    if cli.text_root:
        resolved["text_root"] = cli.text_root
    if cli.hytext_cache_dir:
        resolved["hytext_cache_dir"] = cli.hytext_cache_dir
    resolved.update(
        {
            "deterministic_trace": True,
            "trace_seed": int(cli.trace_seed),
            "representation_loss_mode": "per_entry",
            "representation_loss_scale": 1.0,
            "fk_consistency_loss_weight": 0.0,
            "fk_consistency_scale_m": float(cli.consistency_scale_m),
        }
    )
    return argparse.Namespace(**resolved)


def _namespace_from_scratch(cli: argparse.Namespace) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(["--config", str(cli.scratch_config)])
    args = merge_config(args, load_yaml(cli.scratch_config), explicit_cli={"config"})
    if cli.data_root:
        args.data_root = cli.data_root
    if cli.text_root:
        args.text_root = cli.text_root
    if cli.hytext_cache_dir:
        args.hytext_cache_dir = cli.hytext_cache_dir
    args.seed = int(cli.scratch_seed)
    args.deterministic_trace = True
    args.trace_seed = int(cli.trace_seed)
    args.representation_loss_mode = "per_entry"
    args.representation_loss_scale = 1.0
    args.fk_consistency_loss_weight = 0.0
    args.fk_consistency_scale_m = float(cli.consistency_scale_m)
    return args


def main() -> None:
    cli = parse_args()
    device = torch.device(cli.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Calibration requires an available CUDA device")
    if cli.batch_size < 1 or cli.batches_per_bin < 1:
        raise ValueError("batch_size and batches_per_bin must be positive")
    if cli.velocity_loss_t_eps <= 0:
        raise ValueError("velocity_loss_t_eps must be positive")
    if cli.representation_scale_override < 0:
        raise ValueError("representation_scale_override must be non-negative")

    checkpoint = None
    checkpoint_path: Path | None = None
    if cli.checkpoint:
        checkpoint_path = Path(cli.checkpoint).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        checkpoint_args = checkpoint.get("args")
        if not isinstance(checkpoint_args, dict):
            raise RuntimeError(f"Checkpoint has no args contract: {checkpoint_path}")
        args = _namespace_from_checkpoint(checkpoint_args, cli)
        checkpoint_step = int(checkpoint.get("step", 0))
        default_trace_epoch = int(checkpoint.get("epoch", 0))
        weights_source = "raw_model"
    else:
        args = _namespace_from_scratch(cli)
        checkpoint_step = 0
        default_trace_epoch = 0
        weights_source = "scratch_seeded_initialization"
        seed_all(int(args.seed), rank=0)
    trace_epoch = int(cli.trace_epoch if cli.trace_epoch >= 0 else default_trace_epoch)

    dataset = Kimodo273TextDataset(
        args.data_root,
        split="train",
        text_root=args.text_root or None,
        max_frames=int(cli.max_frames),
        min_frames=int(args.min_frames),
        random_crop=True,
        trace_seed=int(cli.dataset_seed),
    )
    dataset.set_trace_epoch(trace_epoch)
    sample_count = int(cli.batch_size) * int(cli.batches_per_bin)
    index_generator = torch.Generator(device="cpu")
    index_generator.manual_seed(int(cli.dataset_seed))
    indices = torch.randperm(len(dataset), generator=index_generator)[:sample_count].tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(cli.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_kimodo273_text,
    )
    model = create_model(args)
    initial_model_sha256 = hash_model_state(model) if checkpoint is None else ""
    model = model.to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
    model.eval()
    model.requires_grad_(False)
    batches = list(loader)
    if len(batches) != int(cli.batches_per_bin):
        raise RuntimeError(f"Expected {cli.batches_per_bin} batches, got {len(batches)}")
    normalizer = HY273Normalizer.from_data_root(args.data_root).to(device)
    t_bins = _float_list(cli.t_bins)
    lambda_candidates = _float_list(cli.lambda_candidates)
    if not cli.sample_training_timesteps and any(not 0.0 < value < 1.0 for value in t_bins):
        raise ValueError(f"All t bins must be in (0,1), got {t_bins}")

    rows: list[dict[str, Any]] = []
    if cli.sample_training_timesteps:
        work_items = [
            (None, batch_index, batch)
            for batch_index, batch in enumerate(batches)
        ]
    else:
        work_items = [
            (timestep_value, batch_index, batch)
            for timestep_value in t_bins
            for batch_index, batch in enumerate(batches)
        ]
    for work_index, (timestep_value, batch_index, batch) in enumerate(work_items):
            trace_step = checkpoint_step + work_index
            x0_un = batch["motion"].to(device=device, dtype=torch.float32, non_blocking=True)
            valid = batch["valid"].to(device=device, non_blocking=True)
            heading_generator = make_trace_generator(
                args, device, rank=0, step=trace_step, micro_step=0, stream=0
            )
            noise_generator = make_trace_generator(
                args, device, rank=0, step=trace_step, micro_step=0, stream=3
            )
            timestep_generator = make_trace_generator(
                args, device, rank=0, step=trace_step, micro_step=0, stream=2
            )
            contact_generator = make_trace_generator(
                args, device, rank=0, step=trace_step, micro_step=0, stream=4
            )
            text_generator = make_trace_generator(
                args, device, rank=0, step=trace_step, micro_step=0, stream=6
            )
            transform = apply_kimodo_training_transform(
                x0_un,
                random_heading=bool(args.random_first_heading),
                root_shift=bool(args.root_origin_shift),
                generator=heading_generator,
            )
            x0 = normalizer.normalize(transform.motion)
            observed = torch.zeros_like(x0)
            motion_mask = torch.zeros_like(x0, dtype=torch.bool)
            if timestep_value is None:
                timesteps = sample_timesteps(
                    x0.shape[0],
                    device=device,
                    schedule=str(args.time_schedule),
                    p_mean=float(args.denoiser_p_mean),
                    p_std=float(args.denoiser_p_std),
                    generator=timestep_generator,
                ).to(dtype=x0.dtype)
            else:
                timesteps = torch.full(
                    (x0.shape[0],),
                    float(timestep_value),
                    device=device,
                    dtype=x0.dtype,
                )
            noise_cont = torch.randn(
                x0[..., :CONT_DIM].shape,
                device=device,
                dtype=x0.dtype,
                generator=noise_generator,
            )
            contact_aux = torch.rand(
                x0[..., CONTACT_SLICE].shape,
                device=device,
                dtype=x0.dtype,
                generator=contact_generator,
            )
            state = build_flow_state(
                x0,
                observed,
                motion_mask,
                timesteps,
                noise_cont=noise_cont,
                contact_aux=contact_aux,
            )
            texts, _drop_bits, _ = apply_deterministic_text_dropout(
                list(batch["texts"]),
                float(args.text_dropout_prob),
                device,
                text_generator,
            )
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=bool(args.amp),
                dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16,
            ):
                model_output = model(
                    state["model_in"],
                    t=timesteps,
                    c_dir=transform.c_dir,
                    text=texts,
                    length_mask=valid,
                    x_self_cond=None,
                    text_drop_prob=0.0,
                )
            output = model_output.detach().float().requires_grad_(True)
            output_cont = output[..., :CONT_DIM]
            valid_cont = valid[..., None].expand_as(output_cont)

            args.representation_loss_mode = "per_entry"
            l0, _, _ = representation_mse_loss(
                output_cont, state["x0_cont"].float(), valid_cont, args
            )
            candidate_pred, candidate_target, _ = representation_loss_pair(
                z_cont_imp=state["z_cont_imp"].float(),
                t=timesteps,
                x0_hat_cont=output_cont,
                x0_target_cont=state["x0_cont"].float(),
                v_pred_cont=output_cont,
                v_target_cont=state["v_target_cont"].float(),
                prediction_type="x0",
                loss_space=str(cli.calibration_loss_space),
                velocity_t_eps=float(cli.velocity_loss_t_eps),
            )
            args.representation_loss_mode = "semantic_weighted"
            l2_unit, _, _ = representation_mse_loss(
                candidate_pred, candidate_target, valid_cont, args
            )
            x0_hat = torch.cat(
                [output_cont, torch.sigmoid(output[..., CONTACT_SLICE])], dim=-1
            )
            consistency, consistency_cm = fk_position_consistency_loss(
                x0_hat,
                observed,
                motion_mask,
                valid,
                normalizer,
                scale_m=float(cli.consistency_scale_m),
            )
            contact_loss = bce_logits_masked(
                output[..., CONTACT_SLICE],
                state["x0_contact"].float(),
                valid[..., None].expand_as(output[..., CONTACT_SLICE]),
            )
            x0_hat_un = normalizer.denormalize(x0_hat)
            semantic_losses = compute_clean_semantic_losses(
                x0_hat_un,
                transform.motion.float(),
                valid,
                fps=float(args.semantic_loss_fps),
                contact_threshold=float(args.foot_lock_contact_threshold),
            )
            common_objective = (
                float(args.contact_loss_weight) * contact_loss
                + float(args.clean_root_vel_loss_weight)
                * semantic_losses["clean_root_vel"]
                + float(args.clean_joint_vel_loss_weight)
                * semantic_losses["clean_joint_vel"]
                + float(args.foot_lock_loss_weight) * semantic_losses["foot_lock"]
            )
            grad_l0 = torch.autograd.grad(l0, output, retain_graph=True)[0]
            grad_l2 = torch.autograd.grad(l2_unit, output, retain_graph=True)[0]
            grad_common = torch.autograd.grad(
                common_objective, output, retain_graph=True
            )[0]
            grad_consistency = torch.autograd.grad(consistency, output)[0]
            rms_l0 = _rms(grad_l0)
            rms_l2 = _rms(grad_l2)
            rms_consistency = _rms(grad_consistency)
            rows.append(
                {
                    "t": (
                        float(timesteps.mean().item())
                        if timestep_value is None
                        else float(timestep_value)
                    ),
                    "t_min": float(timesteps.min().item()),
                    "t_max": float(timesteps.max().item()),
                    "batch_index": int(batch_index),
                    "dataset_indices": [int(value) for value in batch["dataset_indices"].tolist()],
                    "l0": float(l0.detach().item()),
                    "l2_unit": float(l2_unit.detach().item()),
                    "consistency": float(consistency.detach().item()),
                    "consistency_cm": float(consistency_cm.detach().item()),
                    "grad_rms_l0": rms_l0,
                    "grad_rms_l2_unit": rms_l2,
                    "grad_rms_common": _rms(grad_common),
                    "grad_rms_consistency": rms_consistency,
                    "grad_mean_dot_l2_common": float(
                        (grad_l2.detach().float() * grad_common.detach().float()).mean().item()
                    ),
                    "gradient_elements": int(output.numel()),
                    "alpha_sample": rms_l0 / max(rms_l2, 1e-30),
                    "cosine_l2_consistency": _cosine(grad_l2, grad_consistency),
                }
            )

    global_rms_l0 = _aggregate_rms(rows, "grad_rms_l0")
    global_rms_l2_unit = _aggregate_rms(rows, "grad_rms_l2_unit")
    global_rms_consistency = _aggregate_rms(rows, "grad_rms_consistency")
    computed_alpha = global_rms_l0 / max(global_rms_l2_unit, 1e-30)
    alpha = (
        float(cli.representation_scale_override)
        if float(cli.representation_scale_override) > 0
        else computed_alpha
    )
    for row in rows:
        full_rms_squared = (
            (float(args.flow_loss_weight) * alpha) ** 2
            * float(row["grad_rms_l2_unit"]) ** 2
            + float(row["grad_rms_common"]) ** 2
            + 2.0
            * float(args.flow_loss_weight)
            * alpha
            * float(row["grad_mean_dot_l2_common"])
        )
        row["grad_rms_l2_full_objective"] = math.sqrt(max(full_rms_squared, 0.0))
        row["unit_consistency_ratio"] = float(
            row["grad_rms_consistency"]
            / max(row["grad_rms_l2_full_objective"], 1e-30)
        )

    global_rms_l2_full = _aggregate_rms(rows, "grad_rms_l2_full_objective")

    by_t: dict[str, dict[str, Any]] = {}
    bucket_items: list[tuple[str, list[dict[str, Any]]]]
    if cli.sample_training_timesteps:
        bucket_items = [("training_distribution", rows)]
    else:
        bucket_items = [
            (str(timestep_value), [row for row in rows if row["t"] == timestep_value])
            for timestep_value in t_bins
        ]
    for bucket_name, bin_rows in bucket_items:
        bin_rms_l0 = _aggregate_rms(bin_rows, "grad_rms_l0")
        bin_rms_l2 = _aggregate_rms(bin_rows, "grad_rms_l2_unit")
        bin_rms_consistency = _aggregate_rms(bin_rows, "grad_rms_consistency")
        bin_rms_l2_full = _aggregate_rms(bin_rows, "grad_rms_l2_full_objective")
        by_t[bucket_name] = {
            "count": len(bin_rows),
            "gradient_elements": sum(int(row["gradient_elements"]) for row in bin_rows),
            "aggregate_grad_rms_l0": bin_rms_l0,
            "aggregate_grad_rms_l2_unit": bin_rms_l2,
            "aggregate_grad_rms_consistency": bin_rms_consistency,
            "aggregate_grad_rms_l2_full_objective": bin_rms_l2_full,
            "aggregate_alpha": bin_rms_l0 / max(bin_rms_l2, 1e-30),
            "aggregate_unit_consistency_ratio": bin_rms_consistency
            / max(bin_rms_l2_full, 1e-30),
            "alpha_sample": _summary([row["alpha_sample"] for row in bin_rows]),
            "unit_consistency_ratio": _summary(
                [row["unit_consistency_ratio"] for row in bin_rows]
            ),
            "cosine_l2_consistency": _summary(
                [row["cosine_l2_consistency"] for row in bin_rows]
            ),
            "consistency_cm": _summary([row["consistency_cm"] for row in bin_rows]),
        }

    candidates: list[dict[str, Any]] = []
    for value in lambda_candidates:
        ratios = [value * row["unit_consistency_ratio"] for row in rows]
        bin_ratios = {
            key: value * float(summary["aggregate_unit_consistency_ratio"])
            for key, summary in by_t.items()
        }
        aggregate_ratio = value * global_rms_consistency / max(global_rms_l2_full, 1e-30)
        max_observed_bin_ratio = max(bin_ratios.values())
        aggregate_passed = float(cli.min_ratio) <= aggregate_ratio <= float(cli.max_ratio)
        max_bin_passed = max_observed_bin_ratio <= float(cli.max_bin_ratio)
        passed = max_bin_passed and (
            aggregate_passed or cli.selection_mode == "max_bin_only"
        )
        candidates.append(
            {
                "lambda": float(value),
                "aggregate_ratio": aggregate_ratio,
                "per_batch_ratio": _summary(ratios),
                "bin_aggregate_ratios": bin_ratios,
                "aggregate_passed": bool(aggregate_passed),
                "max_bin_passed": bool(max_bin_passed),
                "passed": bool(passed),
                "target_distance": abs(aggregate_ratio - float(cli.target_ratio)),
            }
        )
    valid_candidates = [item for item in candidates if item["passed"]]
    selected = (
        min(valid_candidates, key=lambda item: item["target_distance"])
        if valid_candidates
        else None
    )
    result = {
        "format": "hy273_l2_l3_output_gradient_calibration_v2",
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_weights": weights_source,
        "scratch_config": None if cli.scratch_config is None else str(Path(cli.scratch_config).resolve()),
        "scratch_seed": int(cli.scratch_seed),
        "initial_model_sha256": initial_model_sha256,
        "trace_epoch": trace_epoch,
        "dataset_seed": int(cli.dataset_seed),
        "trace_seed": int(cli.trace_seed),
        "sample_count": sample_count,
        "t_bins": t_bins,
        "sample_training_timesteps": bool(cli.sample_training_timesteps),
        "selection_mode": str(cli.selection_mode),
        "selection_thresholds": {
            "target_ratio": float(cli.target_ratio),
            "min_ratio": float(cli.min_ratio),
            "max_ratio": float(cli.max_ratio),
            "max_bin_ratio": float(cli.max_bin_ratio),
        },
        "time_schedule": str(args.time_schedule),
        "denoiser_p_mean": float(args.denoiser_p_mean),
        "denoiser_p_std": float(args.denoiser_p_std),
        "consistency_scale_m": float(cli.consistency_scale_m),
        "calibration_loss_space": str(cli.calibration_loss_space),
        "velocity_loss_t_eps": float(cli.velocity_loss_t_eps),
        "representation_scale_alpha": alpha,
        "representation_scale_alpha_computed": computed_alpha,
        "representation_scale_override": float(cli.representation_scale_override),
        "aggregate_grad_rms_l0": global_rms_l0,
        "aggregate_grad_rms_l2_unit": global_rms_l2_unit,
        "aggregate_grad_rms_l2_full_objective": global_rms_l2_full,
        "aggregate_grad_rms_consistency": global_rms_consistency,
        "lambda_denominator": "full_l2_common_training_objective",
        "alpha_samples": _summary([row["alpha_sample"] for row in rows]),
        "by_t": by_t,
        "lambda_candidates": candidates,
        "selected_lambda": None if selected is None else float(selected["lambda"]),
        "passed": selected is not None,
        "rows": rows,
    }
    output_path = Path(cli.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("representation_scale_alpha", "selected_lambda", "passed")}, indent=2))
    print(f"[calibration] report={output_path}")
    if selected is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
