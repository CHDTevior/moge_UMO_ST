#!/usr/bin/env python
"""Generate the fixed HY273 T2M visual set and K273 physical diagnostics."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_kimodo_benchmark import kimodo_motion_quality_metrics
from models.raw_motion.hy273_multitask_condition import CapabilityId, make_absent_condition
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)
from sample_hy273_multitask import (
    create_model_from_checkpoint,
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from train_hy273_multitask import (
    FROZEN_STAGE_CONTRACTS,
    sha256_file,
    validate_frozen_contract,
)
from train_hy273_unified_actor import (
    CHECKPOINT_FORMAT as UNIFIED_ACTOR_CHECKPOINT_FORMAT,
    validate_config as validate_unified_actor_config,
)


PROMPTS = (
    "a person is walking in place at a slow pace.",
    "the person swings a golf club.",
    "the man runs back wards",
    "a man walks forward, then squats to pick something up with both hands, stands back up, and resumes walking.",
    "a person waves with his right hand.",
    "a person walks towards the camera.",
    "person is bent down trying to pick up stuff, left arm is moved to the back, picks up more stuff and touches back again while bending down",
    "a man walks forward, then turns around and walks back before facing back and standing still.",
    "man walks forward while upper body is leaning slightly to the left and steps are unbalanced and slow.",
    "a man using both hands to lift something off ground and places it back on ground in a slightly different position",
    "the person walks backwards in a straight line",
    "man walks forward moving hands and neck.",
    "a person walks forward to the left, picks something up and walks back and then shakes what is in the hand.",
    "a man crouches down while quickly walking forward and then stands up straight.",
    "a person walks forward casually with a swagger to their hips.",
    "a person walks up stairs turns left and walks back down stairs.",
)

LENGTHS = (300, 94, 97, 201, 124, 171, 225, 300, 300, 138, 153, 262, 300, 60, 217, 288)


def _validate_checkpoint_config(config: dict, checkpoint_format: object) -> str:
    """Validate old R12 checkpoints after schedule_version became explicit."""

    if checkpoint_format == UNIFIED_ACTOR_CHECKPOINT_FORMAT:
        validate_unified_actor_config(config)
        return "unified_actor_v1"
    try:
        validate_frozen_contract(config)
        return "exact"
    except ValueError:
        stage = config.get("stage")
        if not isinstance(stage, dict) or "schedule_version" in stage:
            raise
        stage_name = str(stage.get("name", ""))
        if stage_name not in FROZEN_STAGE_CONTRACTS:
            raise
        compatible = deepcopy(config)
        compatible["stage"]["schedule_version"] = FROZEN_STAGE_CONTRACTS[stage_name][3]
        validate_frozen_contract(compatible)
        return "legacy_implicit_schedule_version"


def _jerk_mps3(joints: torch.Tensor, fps: float = 30.0) -> float:
    if joints.shape[0] < 4:
        return 0.0
    jerk = torch.diff(joints.float(), n=3, dim=0) * float(fps) ** 3
    return float(jerk.norm(dim=-1).mean().item())


def _quality(motion: torch.Tensor) -> dict[str, float]:
    fk_joints = fk_positions_from_global_rot6d(motion.float())
    represented_joints = reconstruct_global_joints_from_features(motion.float())
    metrics = kimodo_motion_quality_metrics(
        fk_joints,
        motion[..., CONTACT_SLICE] > 0.5,
    )
    metrics["fk_jerk_mps3"] = _jerk_mps3(fk_joints)
    metrics["position_channel_jerk_mps3"] = _jerk_mps3(represented_joints)
    return {name: float(value) for name, value in metrics.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weight_source", choices=["ema", "model"], default="ema")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_samples", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count = min(max(int(args.max_samples), 1), len(PROMPTS))
    texts = list(PROMPTS[:count])
    lengths = torch.tensor(LENGTHS[:count], dtype=torch.long)
    frames = int(lengths.max().item())
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint has no resolved multitask config")
    config_compatibility = _validate_checkpoint_config(
        config, checkpoint.get("format")
    )
    checkpoint_step = int(checkpoint.get("next_global_step", -1))
    if checkpoint_step < 0:
        raise RuntimeError("Checkpoint has no valid next_global_step")
    code_identity = checkpoint.get("code_identity", {})
    asset_identity = checkpoint.get("asset_identity", {})
    model = create_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint[args.weight_source], strict=True)
    model.eval()
    normalizer = normalizer_from_checkpoint(checkpoint, device)

    condition = make_absent_condition(
        batch_size=count,
        target_frames=frames,
        target_lengths=lengths,
        capability=CapabilityId.T2M,
    )
    observed = torch.zeros(count, frames, DIM_HY273)
    mask = torch.zeros_like(observed, dtype=torch.bool)
    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    sampled = sample_hy273_multitask_ode(
        model,
        normalizer,
        condition,
        texts,
        observed,
        mask,
        num_steps=int(args.num_steps),
        text_cfg_scale=float(args.cfg_scale),
        contact_init="random",
        contact_feedback="blend",
        cfg_apply_contacts=None if normalizer.normalize_contacts else False,
        generator=generator,
    )
    samples = sampled.raw_motion.cpu().float()
    per_sample = [
        {
            "index": index,
            "text": texts[index],
            "length": int(lengths[index].item()),
            "metrics": _quality(samples[index, : int(lengths[index].item())]),
        }
        for index in range(count)
    ]
    metric_names = sorted(per_sample[0]["metrics"])
    aggregate = {
        name: float(np.mean([row["metrics"][name] for row in per_sample]))
        for name in metric_names
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "samples_raw.npy", samples.numpy())
    np.save(output_dir / "samples_exact_clamped.npy", sampled.exact_clamped_motion.cpu().numpy())
    np.save(output_dir / "observed.npy", observed.numpy())
    np.save(output_dir / "mask.npy", mask.numpy())
    np.save(output_dir / "lengths.npy", lengths.numpy())
    metadata = {
        **sampled.protocol,
        "evaluator_path": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_next_global_step": checkpoint_step,
        "checkpoint_format": checkpoint.get("format"),
        "run_name": checkpoint.get("run_name"),
        "run_uuid": checkpoint.get("run_uuid"),
        "config_sha256": checkpoint.get("config_sha256"),
        "base_contract_sha256": checkpoint.get("base_contract_sha256"),
        "code_identity_sha256": code_identity.get("identity_sha256"),
        "asset_identity_sha256": asset_identity.get("identity_sha256"),
        "weight_source": args.weight_source,
        "seed": int(args.seed),
        "texts": texts,
        "lengths": lengths.tolist(),
        "fixed_visual_protocol": "hy273_multitask_t2m_visual16_v1",
        "config_compatibility": config_compatibility,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "quality.json").write_text(
        json.dumps(
            {"aggregate": aggregate, "per_sample": per_sample},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "samples": count, "quality": aggregate},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
