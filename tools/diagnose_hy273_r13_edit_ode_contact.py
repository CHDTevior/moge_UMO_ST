#!/usr/bin/env python3
"""Measure how contact-state feedback changes an R13 Edit ODE rollout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.flow_schedule import clean_x0_euler_step, make_ode_grid
from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_slices import CONTACT_JOINTS, CONTACT_SLICE, DIM_HY273
from models.raw_motion.hy273_slices import reconstruct_global_joints_from_features
from sample_hy273_multitask import make_edit_condition, normalizer_from_checkpoint
from train_hy273_multitask import create_model_from_checkpoint


CONTACT_INPUT_MODES = (
    "normal",
    "source_teacher",
    "target_teacher",
    "physical_zero_teacher",
)


def load_row(manifest: Path, pair_id: str) -> dict[str, Any]:
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("dataset") == "motionfix_k273"
                and str(row["pair"]["official_pair_id"]) == pair_id
            ):
                return row
    raise ValueError(f"MotionFix pair {pair_id} is absent from {manifest}")


def load_motion(ref: dict[str, Any]) -> torch.Tensor:
    value = np.load(ref["path"])
    if value.shape != (int(ref["frames"]), DIM_HY273):
        raise ValueError(f"Invalid K273 asset: {ref['path']}")
    return torch.from_numpy(value.astype(np.float32, copy=False)).clone()


def to_gauge(motion: torch.Tensor) -> torch.Tensor:
    shifted = root_origin_shift(motion)
    heading = shifted[0, 3:5]
    delta = -torch.atan2(heading[1], heading[0])
    return apply_yaw_rotation(shifted, delta)


def descriptors(motion: torch.Tensor) -> dict[str, float]:
    joints = reconstruct_global_joints_from_features(motion.float())
    velocity = (joints[:, 1:] - joints[:, :-1]) * 30.0
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    contacts = motion[..., CONTACT_SLICE]
    binary = contacts > 0.5
    return {
        "joint_speed_mps": float(speed.mean().item()),
        "foot_speed_mps": float(speed[..., list(CONTACT_JOINTS)].mean().item()),
        "contact_occupancy": float(binary.float().mean().item()),
        "all_four_contact_ratio": float(binary.all(dim=-1).float().mean().item()),
    }


@torch.inference_mode()
def rollout(
    *,
    model: torch.nn.Module,
    normalizer: Any,
    source: torch.Tensor,
    target: torch.Tensor,
    text: str,
    seeds: tuple[int, ...],
    num_steps: int,
    velocity_t_eps: float,
    mode: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if mode not in CONTACT_INPUT_MODES:
        raise ValueError(f"Unknown contact input mode {mode}")
    bsz, frames = len(seeds), target.shape[0]
    source_batch = source[None].expand(bsz, -1, -1).to(device)
    target_batch = target[None].expand(bsz, -1, -1).to(device)
    lengths = torch.full((bsz,), frames, device=device, dtype=torch.long)
    gauge = torch.zeros(bsz, 2, device=device)
    gauge[:, 0] = 1.0
    condition = make_edit_condition(
        source_batch,
        target_lengths=lengths,
        target_frames=frames,
        frame_gauge_dir=gauge,
    )
    state_rows = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        state_rows.append(
            torch.randn(
                frames,
                DIM_HY273,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )
    state = torch.stack(state_rows)
    initial_contact_noise = state[..., CONTACT_SLICE].clone()
    source_norm = normalizer.normalize(source_batch)
    target_norm = normalizer.normalize(target_batch)
    physical_zero = target_batch.clone()
    physical_zero[..., CONTACT_SLICE] = 0.0
    zero_contact_norm = normalizer.normalize(physical_zero)[..., CONTACT_SLICE]
    grid = make_ode_grid(num_steps, device=device)
    trace: list[dict[str, Any]] = []
    last_prediction = state
    for step_index in range(num_steps):
        timestep = grid[step_index]
        dt = grid[step_index + 1] - timestep
        model_state = state.clone()
        # Keep interventions on the same forward-flow marginal used in training.
        # Feeding clean contacts directly at t=0 would be an out-of-distribution
        # input and would confound contact feedback with a timestep mismatch.
        if mode != "normal":
            clean_contact = {
                "source_teacher": source_norm[..., CONTACT_SLICE],
                "target_teacher": target_norm[..., CONTACT_SLICE],
                "physical_zero_teacher": zero_contact_norm,
            }[mode]
            model_state[..., CONTACT_SLICE] = (
                (1.0 - timestep) * initial_contact_noise
                + timestep * clean_contact
            )
        model_in = torch.cat([model_state, torch.zeros_like(model_state)], dim=-1)
        t = timestep.expand(bsz)
        prediction = model(
            model_in,
            t=t,
            c_dir=gauge,
            text=[text] * bsz,
            length_mask=torch.ones(bsz, frames, device=device, dtype=torch.bool),
            x_self_cond=None,
            text_drop_prob=0.0,
            condition=condition,
        )
        prediction = prediction.float()
        state, _ = clean_x0_euler_step(
            state,
            prediction,
            timestep=t,
            dt=dt,
            velocity_t_eps=velocity_t_eps,
        )
        last_prediction = prediction
        clean = normalizer.denormalize(prediction)
        trace.append(
            {
                "step_index": step_index,
                "t": float(timestep.item()),
                "mode": mode,
                "clean_mean": {
                    key: float(np.mean([descriptors(row[None])[key] for row in clean]))
                    for key in descriptors(clean[:1]).keys()
                },
            }
        )
    final = normalizer.denormalize(state)
    final_clean = normalizer.denormalize(last_prediction)
    records = []
    for index, seed in enumerate(seeds):
        records.append(
            {
                "seed": int(seed),
                "mode": mode,
                "final_state": descriptors(final[index : index + 1]),
                "final_clean": descriptors(final_clean[index : index + 1]),
            }
        )
    return records, trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--manifest",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
        "hy273_multitask_v1/test.jsonl",
    )
    parser.add_argument("--pair_id", default="000038")
    parser.add_argument("--seeds", default="1011117445,1011117446,1011117447,1011117448")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument(
        "--velocity_t_eps",
        type=float,
        default=1e-4,
        help="ODE x0-to-velocity denominator floor; matches the formal sampler default.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.velocity_t_eps <= 0.0:
        raise ValueError("--velocity_t_eps must be positive")

    row = load_row(Path(args.manifest), args.pair_id)
    source = to_gauge(load_motion(row["source_motion"]["k273_asset"]))
    target = to_gauge(load_motion(row["target_motion"]["k273_asset"]))
    if source.shape != target.shape:
        raise ValueError("ODE contact diagnostic currently requires equal lengths")
    text = str(row["texts"][0]["value"])
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    device = torch.device(args.device)

    records: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for checkpoint_value in args.checkpoint:
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=False
        )
        step = int(checkpoint["next_global_step"])
        model = create_model_from_checkpoint(checkpoint)
        model.load_state_dict(checkpoint["ema"], strict=True)
        normalizer = normalizer_from_checkpoint(checkpoint, device)
        del checkpoint
        model = model.to(device).eval()
        for mode in CONTACT_INPUT_MODES:
            mode_records, mode_trace = rollout(
                model=model,
                normalizer=normalizer,
                source=source,
                target=target,
                text=text,
                seeds=seeds,
                num_steps=int(args.num_steps),
                velocity_t_eps=float(args.velocity_t_eps),
                mode=mode,
                device=device,
            )
            checkpoint_id = str(checkpoint_path)
            records.extend(
                {
                    "checkpoint": checkpoint_id,
                    "checkpoint_step": step,
                    **item,
                }
                for item in mode_records
            )
            traces.extend(
                {
                    "checkpoint": checkpoint_id,
                    "checkpoint_step": step,
                    **item,
                }
                for item in mode_trace
            )
        model_rows.append({"checkpoint_step": step, "checkpoint": str(checkpoint_path)})
        del model
        torch.cuda.empty_cache()

    payload = {
        "format": "hy273_r13_edit_ode_contact_feedback_probe_v4",
        "pair_id": args.pair_id,
        "instruction": text,
        "num_steps": int(args.num_steps),
        "velocity_t_eps": float(args.velocity_t_eps),
        "models": model_rows,
        "source": descriptors(source[None]),
        "target": descriptors(target[None]),
        "records": records,
        "trace": traces,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
