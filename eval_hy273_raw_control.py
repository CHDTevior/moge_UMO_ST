"""Lightweight control evaluation for HY273 raw-flow checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch

from data.kimodo273_datasets import Kimodo273TextDataset, collate_kimodo273_text
from models.raw_motion.hy273_constraints import (
    KimodoControlCurriculum,
    build_kimodo_control_curriculum_batch,
)
from models.raw_motion.hy273_normalizer import apply_kimodo_training_transform
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    CONTACT_JOINTS,
    GLOBAL_ROT_SLICE,
    JOINT_POS_SLICE,
    NUM_JOINTS,
    cont6d_to_matrix,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)
from sample_hy273_raw import (
    ODESampleOutput,
    checkpoint_normalizer,
    checkpoint_weight_state,
    apply_checkpoint_path_override,
    resolve_endpoint_protocol,
    sample_ode,
    verify_checkpoint_assets,
)
from train_hy273_raw_flow import create_model


def _masked_mean_per_sample(
    value: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != mask.shape:
        raise ValueError(
            f"Value/mask shape mismatch: {tuple(value.shape)} vs {tuple(mask.shape)}"
        )
    flat_value = value.reshape(value.shape[0], -1)
    flat_mask = mask.bool().reshape(mask.shape[0], -1)
    counts = flat_mask.sum(dim=-1)
    means = (flat_value * flat_mask).sum(dim=-1) / counts.clamp_min(1)
    return means, counts > 0


def masked_l2_per_sample(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    value = (pred - target).square().sum(dim=-1).sqrt()
    mask = mask.squeeze(-1).bool()
    return _masked_mean_per_sample(value, mask)


def masked_l2(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values, valid = masked_l2_per_sample(pred, target, mask)
    return values[valid].mean() if valid.any() else values.new_tensor(0.0)


def masked_rotation_error_deg_per_sample(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_matrix = cont6d_to_matrix(pred)
    target_matrix = cont6d_to_matrix(target)
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0, 1.0
    )
    return _masked_mean_per_sample(torch.rad2deg(torch.acos(cosine)), mask.bool())


def masked_rotation_error_deg(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    values, valid = masked_rotation_error_deg_per_sample(pred, target, mask)
    return values[valid].mean() if valid.any() else values.new_tensor(0.0)


def masked_contact_metrics_per_sample(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = mask.bool()
    abs_error, valid = _masked_mean_per_sample((pred - target).abs(), mask)
    accuracy, accuracy_valid = _masked_mean_per_sample(
        ((pred > 0.5) == (target > 0.5)).float(), mask
    )
    if not torch.equal(valid, accuracy_valid):
        raise AssertionError("Contact metric validity masks diverged")
    return abs_error, accuracy, valid


def masked_contact_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    abs_error, accuracy, valid = masked_contact_metrics_per_sample(
        pred, target, mask
    )
    if not valid.any():
        zero = pred.new_tensor(0.0)
        return zero, zero
    return abs_error[valid].mean(), accuracy[valid].mean()


def foot_skate_from_joints_per_sample(
    joints: torch.Tensor,
    contacts: torch.Tensor,
    lengths: torch.Tensor,
    fps: float = 30.0,
) -> torch.Tensor:
    """Kimodo-style mean contacting-foot speed in meters/second per sequence."""
    feet = joints[:, :, list(CONTACT_JOINTS)]
    speed = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1) * float(fps)
    contact_now = contacts[:, :-1].bool()
    valid = torch.arange(joints.shape[1] - 1, device=joints.device)[None, :] < (
        lengths[:, None] - 1
    )
    mask = contact_now & valid[..., None]
    counts = mask.sum(dim=(1, 2))
    return (speed * mask).sum(dim=(1, 2)) / counts.clamp_min(1)


def foot_skate_metric(
    features: torch.Tensor, lengths: torch.Tensor, fps: float = 30.0
) -> torch.Tensor:
    joints = fk_positions_from_global_rot6d(features)
    contacts = features[..., CONTACT_SLICE] > 0.5
    return foot_skate_from_joints_per_sample(joints, contacts, lengths, fps).mean()


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        name: str,
        values: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        values = values.detach().float().reshape(-1)
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        else:
            valid = valid.detach().bool().reshape(-1)
        if values.shape != valid.shape:
            raise ValueError(
                f"Metric values/valid shape mismatch for {name}: "
                f"{tuple(values.shape)} vs {tuple(valid.shape)}"
            )
        self.sums[name] = self.sums.get(name, 0.0) + float(values[valid].sum())
        self.counts[name] = self.counts.get(name, 0) + int(valid.sum())

    def means(self) -> dict[str, float | None]:
        return {
            name: self.sums[name] / self.counts[name]
            if self.counts[name] > 0
            else None
            for name in sorted(self.sums)
        }


def phase2_distribution_conditioned_on_control(
    none_prob: float, mixed_prob: float
) -> float:
    controlled_mass = 1.0 - float(none_prob)
    if controlled_mass <= 0:
        raise ValueError("Checkpoint control distribution has no controlled examples")
    if not 0 <= float(mixed_prob) <= controlled_mass:
        raise ValueError("Invalid Phase-2 none/mixed probabilities")
    return float(mixed_prob) / controlled_mass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="")
    p.add_argument("--text_root", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=32)
    p.add_argument("--output", default="")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--cfg_scale", type=float, default=2.0)
    p.add_argument("--control_cfg_scale", type=float, default=2.0)
    p.add_argument("--curriculum_progress", type=float, default=1.0)
    p.add_argument("--weight_source", choices=["ema", "model", "auto"], default="ema")
    p.add_argument(
        "--text_encoder",
        choices=["clip", "hy_cache", "hytext_cache", "qwen_clip_cache", "none"],
        default="",
    )
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = argparse.Namespace(**ckpt.get("args", {}))
    apply_checkpoint_path_override(train_args, "data_root", args.data_root)
    apply_checkpoint_path_override(train_args, "text_root", args.text_root)
    if args.text_encoder:
        train_args.text_encoder = args.text_encoder
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    verify_checkpoint_assets(train_args)
    model = create_model(train_args).to(device)
    state_dict, weight_source = checkpoint_weight_state(
        ckpt, args.weight_source, args.checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    normalizer = checkpoint_normalizer(ckpt, train_args, device, args.checkpoint)
    endpoint_protocol = resolve_endpoint_protocol(train_args)
    train_none_prob = float(getattr(train_args, "control_none_prob", 0.10))
    train_mixed_prob = float(getattr(train_args, "control_mixed_prob", 0.25))
    controlled_mixed_prob = phase2_distribution_conditioned_on_control(
        train_none_prob, train_mixed_prob
    )
    dataset = Kimodo273TextDataset(
        train_args.data_root,
        split=args.split,
        text_root=train_args.text_root or None,
        max_frames=train_args.max_frames,
        random_crop=False,
        deterministic_text=True,
    )
    accumulator = MetricAccumulator()
    batch_count = 0
    sample_count = 0
    mode_counts: Counter[str] = Counter()
    for start in range(0, min(args.max_samples, len(dataset)), args.batch_size):
        samples = [dataset[i] for i in range(start, min(start + args.batch_size, args.max_samples, len(dataset)))]
        batch = collate_kimodo273_text(samples)
        gt = batch["motion"].to(device)
        lengths = batch["lengths"].to(device)
        transform = apply_kimodo_training_transform(gt, random_heading=False, root_shift=True)
        gt = transform.motion
        controls = build_kimodo_control_curriculum_batch(
            gt,
            lengths,
            progress=float(args.curriculum_progress),
            config=KimodoControlCurriculum(
                none_prob=0.0,
                mixed_prob=controlled_mixed_prob,
                max_sparse_keyframes=int(endpoint_protocol["max_control_keyframes"]),
                dense_min_fraction=float(
                    getattr(train_args, "control_dense_min_fraction", 0.25)
                ),
                endpoint_preset=str(endpoint_protocol["endpoint_preset"]),
                endpoint_subset_mode=str(endpoint_protocol["endpoint_subset_mode"]),
                include_root_ref_for_endpoints=bool(
                    endpoint_protocol["include_root_ref_for_endpoints"]
                ),
                include_endpoint_rotations=bool(
                    endpoint_protocol["include_endpoint_rotations"]
                ),
                include_contact_pattern=bool(
                    endpoint_protocol["include_contact_pattern"]
                ),
            ),
            generator=torch.Generator(device=device).manual_seed(int(args.seed) + start),
            track_pattern_masks=True,
        )
        mode_counts.update(controls.mode_ids)
        sampled = sample_ode(
            model,
            normalizer,
            lengths,
            batch["texts"],
            controls.observed_motion,
            controls.motion_mask,
            transform.c_dir,
            num_steps=args.num_steps,
            self_conditioning=bool(getattr(train_args, "self_conditioning", False)),
            cfg_scale=float(args.cfg_scale),
            control_cfg_scale=float(args.control_cfg_scale),
            cfg_apply_contacts=bool(endpoint_protocol["include_contact_pattern"]),
            prediction_type=str(getattr(train_args, "prediction_type", "x0")),
            return_details=True,
        )
        assert isinstance(sampled, ODESampleOutput)
        if controls.pattern_masks is None:
            raise AssertionError("Evaluation requires per-pattern control masks")
        zero_mask = torch.zeros_like(controls.motion_mask)

        def pattern_mask(name: str) -> torch.Tensor:
            return controls.pattern_masks.get(name, zero_mask)

        gt_feat_joints = reconstruct_global_joints_from_features(gt)
        endpoint_mask = (
            pattern_mask("endpoints")[..., JOINT_POS_SLICE]
            .reshape(gt.shape[0], gt.shape[1], NUM_JOINTS, 3)
            .any(dim=-1)
        )
        endpoint_rot_mask = (
            pattern_mask("endpoints")[..., GLOBAL_ROT_SLICE]
            .reshape(gt.shape[0], gt.shape[1], NUM_JOINTS, 6)
            .any(dim=-1)
        )
        fullbody_mask = (
            pattern_mask("fullpose")[..., JOINT_POS_SLICE]
            .reshape(gt.shape[0], gt.shape[1], NUM_JOINTS, 3)
            .any(dim=-1)
        )
        contact_control_mask = pattern_mask("contact")[..., CONTACT_SLICE]
        gt_rot6d = gt[..., GLOBAL_ROT_SLICE].reshape(
            gt.shape[0], gt.shape[1], NUM_JOINTS, 6
        )
        root_sparse_mask = pattern_mask("root_sparse")[..., [0, 2]].any(dim=-1)
        root_dense_mask = pattern_mask("root_dense")[..., [0, 2]].any(dim=-1)
        root_control_mask = root_sparse_mask | root_dense_mask
        valid_frame = torch.arange(gt.shape[1], device=device)[None, :] < lengths[:, None]
        valid_joint = valid_frame[..., None].expand(
            -1, -1, NUM_JOINTS
        )
        valid_contact = valid_frame[..., None].expand(-1, -1, 4)
        for prefix, pred in (
            ("raw", sampled.raw_motion),
            ("exact", sampled.exact_clamped_motion),
        ):
            pred_feat_joints = reconstruct_global_joints_from_features(pred)
            pred_fk_joints = fk_positions_from_global_rot6d(pred)
            values, valid = masked_l2_per_sample(
                pred_feat_joints, gt_feat_joints, endpoint_mask
            )
            accumulator.add(f"{prefix}_endpoint_err_feature_m", values, valid)
            values, valid = masked_l2_per_sample(
                pred_fk_joints, gt_feat_joints, endpoint_mask
            )
            accumulator.add(f"{prefix}_endpoint_err_fk_m", values, valid)
            values, valid = masked_l2_per_sample(
                pred_feat_joints, gt_feat_joints, fullbody_mask
            )
            accumulator.add(f"{prefix}_fullbody_err_feature_m", values, valid)
            values, valid = masked_l2_per_sample(
                pred_fk_joints, gt_feat_joints, fullbody_mask
            )
            accumulator.add(f"{prefix}_fullbody_err_fk_m", values, valid)
            pred_rot6d = pred[..., GLOBAL_ROT_SLICE].reshape(
                gt.shape[0], gt.shape[1], NUM_JOINTS, 6
            )
            values, valid = masked_rotation_error_deg_per_sample(
                pred_rot6d, gt_rot6d, endpoint_rot_mask
            )
            accumulator.add(f"{prefix}_endpoint_rot_err_deg", values, valid)
            values, valid = masked_l2_per_sample(
                pred_fk_joints, pred_feat_joints, valid_joint
            )
            accumulator.add(f"{prefix}_fk_consistency_err_m", values, valid)
            for root_name, root_mask in (
                ("root_sparse_xz_err_m", root_sparse_mask),
                ("root_dense_xz_err_m", root_dense_mask),
                ("root_xz_err_m", root_control_mask),
            ):
                values, valid = masked_l2_per_sample(
                    pred[..., [0, 2]], gt[..., [0, 2]], root_mask
                )
                accumulator.add(f"{prefix}_{root_name}", values, valid)
            contact_abs, contact_acc, contact_valid = (
                masked_contact_metrics_per_sample(
                    pred[..., CONTACT_SLICE],
                    gt[..., CONTACT_SLICE],
                    contact_control_mask,
                )
            )
            accumulator.add(
                f"{prefix}_controlled_contact_abs_error",
                contact_abs,
                contact_valid,
            )
            accumulator.add(
                f"{prefix}_controlled_contact_accuracy",
                contact_acc,
                contact_valid,
            )
            _, all_contact_acc, all_contact_valid = masked_contact_metrics_per_sample(
                pred[..., CONTACT_SLICE],
                gt[..., CONTACT_SLICE],
                valid_contact,
            )
            accumulator.add(
                f"{prefix}_all_contact_accuracy",
                all_contact_acc,
                all_contact_valid,
            )
            foot_skate_mps = foot_skate_from_joints_per_sample(
                pred_fk_joints,
                pred[..., CONTACT_SLICE] > 0.5,
                lengths,
                fps=30.0,
            )
            accumulator.add(f"{prefix}_foot_skate_mps", foot_skate_mps)
            accumulator.add(f"{prefix}_foot_skate_cmps", foot_skate_mps * 100.0)
        batch_count += 1
        sample_count += int(gt.shape[0])
    metrics = accumulator.means()
    metrics.update(
        {
            "checkpoint": args.checkpoint,
            "num_steps": args.num_steps,
            "seed": int(args.seed),
            "cfg_scale": float(args.cfg_scale),
            "control_cfg_scale": float(args.control_cfg_scale),
            "weight_source": weight_source,
            "batches": batch_count,
            "samples": sample_count,
            "metric_sample_counts": dict(sorted(accumulator.counts.items())),
            "endpoint_protocol": endpoint_protocol,
            "control_mode_counts": dict(sorted(mode_counts.items())),
            "distribution": {
                "protocol": "phase2_conditioned_on_controlled",
                "training_none_prob": train_none_prob,
                "training_mixed_prob": train_mixed_prob,
                "evaluation_none_prob": 0.0,
                "evaluation_mixed_prob": controlled_mixed_prob,
            },
        }
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
