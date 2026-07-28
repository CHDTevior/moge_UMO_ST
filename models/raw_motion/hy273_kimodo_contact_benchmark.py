"""Sparse-contact extension of the frozen HumanML3D Kimodo benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn.functional as F

from .hy273_kimodo_benchmark import (
    CompiledKimodoConstraint,
    compile_kimodo_constraint,
    evaluate_kimodo_constraint_case,
    load_smplx22_metric_joints,
)
from .hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    NUM_JOINTS,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)


V5_CONTACT_PROTOCOL = "hy273_hml3d_kimodo_constraints_v5_contact"
V5_CONTACT_SUBTYPES = (
    "contact_only_sparse",
    "root_sparse_contact",
    "root_dense_contact",
    "endpoints_contact",
    "fullpose_contact",
    "mixed_contact",
)
V5_CONTACT_BASE_SUBTYPE = {
    "contact_only_sparse": None,
    "root_sparse_contact": "waypoint_2dposrot",
    "root_dense_contact": "path_2dposrot",
    "endpoints_contact": "hands_feet_posrot",
    "fullpose_contact": "random",
    "mixed_contact": "root_ee_hands_feet_posrot_fullbody",
}


@dataclass
class CompiledKimodoContactConstraint:
    observed_motion: torch.Tensor
    motion_mask: torch.Tensor
    contact_metric_mask: torch.Tensor
    base: CompiledKimodoConstraint
    components: dict[str, list[int]]


def _empty_base(motion: torch.Tensor) -> CompiledKimodoConstraint:
    frames = motion.shape[0]
    return CompiledKimodoConstraint(
        observed_motion=torch.zeros_like(motion),
        motion_mask=torch.zeros_like(motion, dtype=torch.bool),
        root_metric_frames=torch.zeros(frames, dtype=torch.bool),
        fullbody_metric_frames=torch.zeros(frames, dtype=torch.bool),
        endpoint_position_metric_mask=torch.zeros(
            frames, NUM_JOINTS, dtype=torch.bool
        ),
        endpoint_rotation_metric_mask=torch.zeros(
            frames, NUM_JOINTS, dtype=torch.bool
        ),
        components={},
    )


def _contact_frames(length: int, *, seed: int, max_keyframes: int) -> torch.Tensor:
    if length < 2 or max_keyframes < 1:
        raise ValueError("Contact constraints require length>=2 and max_keyframes>=1")
    payload = f"{V5_CONTACT_PROTOCOL}:{seed}:contact-frames".encode("utf-8")
    contact_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed(contact_seed)
    count = min(length, max(2, min(int(max_keyframes), (length + 7) // 8)))
    random_frames = torch.randperm(length, generator=generator)[:count]
    return torch.unique(
        torch.cat([random_frames, torch.tensor([0, length - 1])]), sorted=True
    )


def compile_kimodo_contact_constraint(
    motion: torch.Tensor,
    subtype: str,
    *,
    seed: int,
    max_sparse_keyframes: int = 20,
) -> CompiledKimodoContactConstraint:
    if motion.ndim != 2 or motion.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected motion [T,{DIM_HY273}]")
    if subtype not in V5_CONTACT_BASE_SUBTYPE:
        raise ValueError(f"Unknown v5 contact subtype: {subtype}")
    base_name = V5_CONTACT_BASE_SUBTYPE[subtype]
    base = (
        _empty_base(motion)
        if base_name is None
        else compile_kimodo_constraint(
            motion,
            base_name,
            seed=seed,
            max_sparse_keyframes=max_sparse_keyframes,
        )
    )
    observed = base.observed_motion.clone()
    motion_mask = base.motion_mask.clone()
    contact_mask = torch.zeros(
        motion.shape[0], CONTACT_SLICE.stop - CONTACT_SLICE.start, dtype=torch.bool
    )
    frames = _contact_frames(
        motion.shape[0], seed=seed, max_keyframes=max_sparse_keyframes
    )
    contact_mask[frames] = True
    expanded_contact_mask = torch.zeros_like(motion_mask)
    expanded_contact_mask[..., CONTACT_SLICE] = contact_mask
    observed[expanded_contact_mask] = motion[expanded_contact_mask]
    motion_mask |= expanded_contact_mask
    components = dict(base.components)
    components["sparse_foot_contacts"] = frames.tolist()
    if not contact_mask.any() or not motion_mask.any():
        raise AssertionError("v5 contact compiler emitted an empty mask")
    if not torch.equal(observed[motion_mask], motion[motion_mask]):
        raise AssertionError("v5 observed values differ from the target motion")
    if bool(torch.count_nonzero(observed[~motion_mask])):
        raise AssertionError("v5 unobserved values must remain exact zero")
    return CompiledKimodoContactConstraint(
        observed_motion=observed,
        motion_mask=motion_mask,
        contact_metric_mask=contact_mask,
        base=base,
        components=components,
    )


def controlled_contact_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Evaluate only explicitly controlled contact entries."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("Controlled contact tensors must have identical [T,4] shapes")
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("Controlled contact mask must be non-empty bool")
    expected = target[mask].float()
    if bool(((expected != 0.0) & (expected != 1.0)).any()):
        raise ValueError("Controlled contact target must be exact binary 0/1")
    probability = prediction[mask].float().clamp(0.0, 1.0)
    binary = probability >= 0.5
    expected_binary = expected.bool()
    tp = (binary & expected_binary).sum().float()
    fp = (binary & ~expected_binary).sum().float()
    fn = (~binary & expected_binary).sum().float()
    f1_denom = 2.0 * tp + fp + fn
    f1 = torch.where(f1_denom > 0, 2.0 * tp / f1_denom, torch.ones_like(tp))
    return {
        "controlled_contact_bce": float(
            F.binary_cross_entropy(probability.clamp(1e-6, 1.0 - 1e-6), expected).item()
        ),
        "controlled_contact_brier": float((probability - expected).square().mean().item()),
        "controlled_contact_accuracy": float((binary == expected_binary).float().mean().item()),
        "controlled_contact_f1": float(f1.item()),
        "controlled_contact_exact_equality": float((probability == expected).float().mean().item()),
        "controlled_contact_entries": int(mask.sum().item()),
        "controlled_contact_positive_entries": int(expected_binary.sum().item()),
    }


def evaluate_kimodo_contact_case(
    predicted_features: torch.Tensor,
    target_features: torch.Tensor,
    constraint: CompiledKimodoContactConstraint,
    *,
    fps: float = 30.0,
) -> dict[str, float]:
    metrics = evaluate_kimodo_constraint_case(
        predicted_features,
        target_features,
        constraint.base,
        fps=fps,
    )
    metrics.update(
        controlled_contact_metrics(
            predicted_features[..., CONTACT_SLICE],
            target_features[..., CONTACT_SLICE],
            constraint.contact_metric_mask,
        )
    )
    joints_from_position = reconstruct_global_joints_from_features(predicted_features)
    neutral_joints = load_smplx22_metric_joints(
        device=predicted_features.device,
        dtype=predicted_features.dtype,
    )
    joints_from_rotation = fk_positions_from_global_rot6d(
        predicted_features, neutral_joints=neutral_joints
    )
    metrics["fk_position_rotation_consistency_cm"] = float(
        (joints_from_position - joints_from_rotation).norm(dim=-1).mean().item() * 100.0
    )
    return metrics
