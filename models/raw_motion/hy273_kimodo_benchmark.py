"""Kimodo constrained-benchmark adaptation for HumanML3D HY273 motions.

The public Kimodo benchmark defines thirteen constrained-generation leaf types.
HumanML3D does not ship Kimodo's test-suite metadata, so this module preserves
the public category and metric semantics while deterministically deriving each
constraint from a held-out HumanML3D motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Literal, Sequence

import torch

from .hy273_slices import (
    CONTACT_JOINTS,
    CONTACT_SLICE,
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    KIMODO_EE_GROUPS,
    KIMODO_EE_ROT_GROUPS,
    NUM_JOINTS,
    ROOT_SLICE,
    cont6d_to_matrix,
    fk_positions_from_global_rot6d,
    global_rot_slice_for,
    joint_pos_slice_for,
    load_smplx22_neutral_joints,
)


KIMODO_CONTROL_SUBTYPES = (
    "path_2dpos",
    "path_2dposrot",
    "waypoint_2dpos",
    "waypoint_2dposrot",
    "inbetweening",
    "random",
    "feet_posrot",
    "hands_posrot",
    "hands_feet_posrot",
    "root_ee_hands_feet_posrot_fullbody",
    "root_ee_hands_posrot",
    "root_ee_hands_posrot_fullbody",
    "root_path_fullbody",
)

SUBTYPE_TO_FAMILY = {
    "path_2dpos": "root",
    "path_2dposrot": "root",
    "waypoint_2dpos": "root",
    "waypoint_2dposrot": "root",
    "inbetweening": "fullbody",
    "random": "fullbody",
    "feet_posrot": "end-effectors",
    "hands_posrot": "end-effectors",
    "hands_feet_posrot": "end-effectors",
    "root_ee_hands_feet_posrot_fullbody": "mixture",
    "root_ee_hands_posrot": "mixture",
    "root_ee_hands_posrot_fullbody": "mixture",
    "root_path_fullbody": "mixture",
}

TEXT_REGIMES = ("withtext", "notext")
ROOT_THRESHOLD_M = 0.10
FPS = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMPLX22_METRIC_JOINTS_PATH = (
    PROJECT_ROOT
    / "external_repos"
    / "kimodo"
    / "kimodo"
    / "assets"
    / "skeletons"
    / "smplx22"
    / "joints.p"
).resolve()


def load_smplx22_metric_joints(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load the evaluation skeleton from the project-root asset, never the CWD."""

    if not SMPLX22_METRIC_JOINTS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing SMPL-X22 metric skeleton: {SMPLX22_METRIC_JOINTS_PATH}"
        )
    return load_smplx22_neutral_joints(
        path=SMPLX22_METRIC_JOINTS_PATH,
        device=device,
        dtype=dtype,
    )

TextRegime = Literal["withtext", "notext"]
AssignmentMode = Literal["balanced_partition", "cartesian"]


@dataclass(frozen=True)
class KimodoBenchmarkCase:
    dataset_index: int
    subtype: str
    family: str
    text_regime: TextRegime
    sample_seed: int

    @property
    def key(self) -> str:
        return (
            f"index_{self.dataset_index:05d}__{self.subtype}__"
            f"{self.text_regime}"
        )


@dataclass
class CompiledKimodoConstraint:
    observed_motion: torch.Tensor
    motion_mask: torch.Tensor
    root_metric_frames: torch.Tensor
    fullbody_metric_frames: torch.Tensor
    endpoint_position_metric_mask: torch.Tensor
    endpoint_rotation_metric_mask: torch.Tensor
    components: dict[str, list[int]]


def stable_case_seed(base_seed: int, dataset_index: int, subtype: str) -> int:
    """Return a stable positive seed shared by with-text/no-text paired cases."""
    payload = f"hy273-kimodo-v1:{base_seed}:{dataset_index}:{subtype}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def build_kimodo_case_plan(
    num_items: int,
    *,
    seed: int = 3407,
    subtypes: Sequence[str] = KIMODO_CONTROL_SUBTYPES,
    text_regimes: Sequence[TextRegime] = TEXT_REGIMES,
    assignment: AssignmentMode = "balanced_partition",
    cases_per_subtype: int = 0,
) -> list[KimodoBenchmarkCase]:
    """Build a deterministic benchmark plan.

    ``balanced_partition`` uses every dataset item once across the thirteen
    subtypes, then mirrors the plan for with-text and no-text evaluation. This
    follows the public benchmark's disjoint leaf-folder organization while
    covering the complete HumanML3D test split. ``cartesian`` is available for
    the much more expensive every-motion-by-every-subtype diagnostic.
    """
    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if not subtypes:
        raise ValueError("At least one subtype is required")
    unknown = sorted(set(subtypes) - set(KIMODO_CONTROL_SUBTYPES))
    if unknown:
        raise ValueError(f"Unknown Kimodo subtypes: {unknown}")
    invalid_regimes = sorted(set(text_regimes) - set(TEXT_REGIMES))
    if invalid_regimes:
        raise ValueError(f"Unknown text regimes: {invalid_regimes}")
    if cases_per_subtype < 0:
        raise ValueError("cases_per_subtype must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(num_items, generator=generator).tolist()
    assignments: list[tuple[int, str]] = []
    if assignment == "balanced_partition":
        assignments = [
            (dataset_index, str(subtypes[position % len(subtypes)]))
            for position, dataset_index in enumerate(permutation)
        ]
    elif assignment == "cartesian":
        assignments = [
            (dataset_index, str(subtype))
            for subtype in subtypes
            for dataset_index in permutation
        ]
    else:
        raise ValueError(f"Unknown assignment mode: {assignment}")

    if cases_per_subtype:
        kept: list[tuple[int, str]] = []
        counts = {str(subtype): 0 for subtype in subtypes}
        for dataset_index, subtype in assignments:
            if counts[subtype] >= cases_per_subtype:
                continue
            kept.append((dataset_index, subtype))
            counts[subtype] += 1
        assignments = kept

    plan = [
        KimodoBenchmarkCase(
            dataset_index=dataset_index,
            subtype=subtype,
            family=SUBTYPE_TO_FAMILY[subtype],
            text_regime=regime,
            sample_seed=stable_case_seed(seed, dataset_index, subtype),
        )
        for dataset_index, subtype in assignments
        for regime in text_regimes
    ]
    return plan


def shard_kimodo_case_plan(
    plan: Sequence[KimodoBenchmarkCase],
    *,
    shard_id: int,
    num_shards: int,
) -> list[KimodoBenchmarkCase]:
    """Shard by motion/subtype pair so text regimes stay on the same device."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Expected 0 <= shard_id < num_shards, got {shard_id}/{num_shards}")
    pair_indices: dict[tuple[int, str, int], int] = {}
    shard_cases: list[KimodoBenchmarkCase] = []
    for case in plan:
        pair_key = (case.dataset_index, case.subtype, case.sample_seed)
        pair_index = pair_indices.setdefault(pair_key, len(pair_indices))
        if pair_index % num_shards == shard_id:
            shard_cases.append(case)
    return shard_cases


def _low_biased_keyframe_count(
    length: int,
    max_sparse_keyframes: int,
    generator: torch.Generator,
) -> int:
    max_count = max(1, min(int(length), int(max_sparse_keyframes)))
    if max_count == 1:
        return 1
    draw = float(torch.rand((), generator=generator).item())
    return 1 + min(int(math.floor(draw * draw * max_count)), max_count - 1)


def _select_sparse_frames(
    length: int,
    max_sparse_keyframes: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Match the Stage-2 low-count, spread-over-time keyframe sampler."""
    count = _low_biased_keyframe_count(length, max_sparse_keyframes, generator)
    if count == 1:
        return torch.randint(0, length, (1,), generator=generator)
    base = torch.linspace(0, length - 1, count)
    jitter = torch.randint(-1, 2, (count,), generator=generator)
    return (base.round().long() + jitter).clamp(0, length - 1).unique()


def _copy_indices(
    observed: torch.Tensor,
    motion_mask: torch.Tensor,
    source: torch.Tensor,
    frames: torch.Tensor,
    indices: Iterable[int],
) -> None:
    indices_tensor = torch.as_tensor(list(indices), dtype=torch.long)
    if frames.numel() == 0 or indices_tensor.numel() == 0:
        return
    observed[frames[:, None], indices_tensor[None, :]] = source[
        frames[:, None], indices_tensor[None, :]
    ]
    motion_mask[frames[:, None], indices_tensor[None, :]] = True


def _endpoint_joint_ids(which: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # Logical groups are left foot, right foot, left hand, right hand.
    if which == "feet":
        selected = (0, 1)
    elif which == "hands":
        selected = (2, 3)
    elif which == "hands_feet":
        selected = (0, 1, 2, 3)
    else:
        raise ValueError(f"Unknown endpoint selection: {which}")
    position_joints = tuple(
        joint
        for group_index in selected
        for joint in KIMODO_EE_GROUPS[group_index]
    )
    rotation_joints = tuple(
        joint
        for group_index in selected
        for joint in KIMODO_EE_ROT_GROUPS[group_index]
    )
    return position_joints, rotation_joints


def compile_kimodo_constraint(
    motion: torch.Tensor,
    subtype: str,
    *,
    seed: int,
    max_sparse_keyframes: int = 20,
) -> CompiledKimodoConstraint:
    """Compile one official leaf type into same-space HY273 observation/mask tensors."""
    if motion.ndim != 2 or motion.shape[-1] != DIM_HY273:
        raise ValueError(f"Expected motion [T,{DIM_HY273}], got {tuple(motion.shape)}")
    if subtype not in SUBTYPE_TO_FAMILY:
        raise ValueError(f"Unknown Kimodo subtype: {subtype}")
    if max_sparse_keyframes < 1:
        raise ValueError("max_sparse_keyframes must be positive")
    length = int(motion.shape[0])
    if length < 2:
        raise ValueError("Kimodo benchmark constraints require at least two frames")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    observed = torch.zeros_like(motion)
    model_mask = torch.zeros_like(motion, dtype=torch.bool)
    root_metric_frames = torch.zeros(length, dtype=torch.bool)
    fullbody_metric_frames = torch.zeros(length, dtype=torch.bool)
    endpoint_position_mask = torch.zeros(length, NUM_JOINTS, dtype=torch.bool)
    endpoint_rotation_mask = torch.zeros(length, NUM_JOINTS, dtype=torch.bool)
    components: dict[str, list[int]] = {}

    root_xz = (ROOT_SLICE.start, ROOT_SLICE.start + 2)
    heading = tuple(range(HEADING_SLICE.start, HEADING_SLICE.stop))
    hidden_root_reference = tuple(range(ROOT_SLICE.start, HEADING_SLICE.stop))
    fullbody_positions = tuple(range(JOINT_POS_SLICE.start, JOINT_POS_SLICE.stop))

    def add_root(name: str, frames: torch.Tensor, include_heading: bool) -> None:
        indices = root_xz + (heading if include_heading else ())
        _copy_indices(observed, model_mask, motion, frames, indices)
        root_metric_frames[frames] = True
        components[name] = frames.tolist()

    def add_fullbody(name: str, frames: torch.Tensor) -> None:
        _copy_indices(
            observed,
            model_mask,
            motion,
            frames,
            hidden_root_reference + fullbody_positions,
        )
        fullbody_metric_frames[frames] = True
        components[name] = frames.tolist()

    def add_endpoints(name: str, frames: torch.Tensor, which: str) -> None:
        position_joints, rotation_joints = _endpoint_joint_ids(which)
        indices = (
            hidden_root_reference
            + tuple(joint_pos_slice_for(position_joints))
            + tuple(global_rot_slice_for(rotation_joints))
        )
        _copy_indices(observed, model_mask, motion, frames, indices)
        endpoint_position_mask[
            frames[:, None], torch.as_tensor(position_joints)[None, :]
        ] = True
        endpoint_rotation_mask[
            frames[:, None], torch.as_tensor(rotation_joints)[None, :]
        ] = True
        components[name] = frames.tolist()

    def sparse() -> torch.Tensor:
        return _select_sparse_frames(length, max_sparse_keyframes, generator)

    all_frames = torch.arange(length, dtype=torch.long)
    if subtype == "path_2dpos":
        add_root("root_path_2dpos", all_frames, include_heading=False)
    elif subtype == "path_2dposrot":
        add_root("root_path_2dposrot", all_frames, include_heading=True)
    elif subtype == "waypoint_2dpos":
        add_root("root_waypoint_2dpos", sparse(), include_heading=False)
    elif subtype == "waypoint_2dposrot":
        add_root("root_waypoint_2dposrot", sparse(), include_heading=True)
    elif subtype == "inbetweening":
        add_fullbody(
            "fullbody_inbetweening",
            torch.tensor([0, length - 1], dtype=torch.long).unique(),
        )
    elif subtype == "random":
        add_fullbody("fullbody_random", sparse())
    elif subtype == "feet_posrot":
        add_endpoints("endpoints_feet_posrot", sparse(), "feet")
    elif subtype == "hands_posrot":
        add_endpoints("endpoints_hands_posrot", sparse(), "hands")
    elif subtype == "hands_feet_posrot":
        add_endpoints("endpoints_hands_feet_posrot", sparse(), "hands_feet")
    elif subtype == "root_ee_hands_feet_posrot_fullbody":
        add_root("root_path_2dposrot", all_frames, include_heading=True)
        add_endpoints("endpoints_hands_feet_posrot", sparse(), "hands_feet")
        add_fullbody("fullbody_random", sparse())
    elif subtype == "root_ee_hands_posrot":
        add_root("root_waypoint_2dposrot", sparse(), include_heading=True)
        add_endpoints("endpoints_hands_posrot", sparse(), "hands")
    elif subtype == "root_ee_hands_posrot_fullbody":
        add_root("root_waypoint_2dposrot", sparse(), include_heading=True)
        add_endpoints("endpoints_hands_posrot", sparse(), "hands")
        add_fullbody("fullbody_random", sparse())
    elif subtype == "root_path_fullbody":
        add_root("root_path_2dposrot", all_frames, include_heading=True)
        add_fullbody("fullbody_random", sparse())
    else:
        raise AssertionError(f"Unhandled Kimodo subtype: {subtype}")

    if not model_mask.any():
        raise AssertionError(f"Compiled empty constraint for {subtype}")
    if not torch.equal(observed[model_mask], motion[model_mask]):
        raise AssertionError("Observed constraint values diverged from the source motion")
    if torch.count_nonzero(observed[~model_mask]):
        raise AssertionError("Unobserved HY273 entries must remain zero")
    return CompiledKimodoConstraint(
        observed_motion=observed,
        motion_mask=model_mask,
        root_metric_frames=root_metric_frames,
        fullbody_metric_frames=fullbody_metric_frames,
        endpoint_position_metric_mask=endpoint_position_mask,
        endpoint_rotation_metric_mask=endpoint_rotation_mask,
        components=components,
    )


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return numerator / (denominator + 1e-6)


def kimodo_motion_quality_metrics(
    posed_joints: torch.Tensor,
    foot_contacts: torch.Tensor,
    *,
    fps: float = FPS,
) -> dict[str, float]:
    """Compute the public Kimodo foot-skate/contact metrics for one motion."""
    if posed_joints.ndim != 3 or posed_joints.shape[-2:] != (NUM_JOINTS, 3):
        raise ValueError(
            f"Expected posed_joints [T,{NUM_JOINTS},3], got {tuple(posed_joints.shape)}"
        )
    if foot_contacts.shape != (posed_joints.shape[0], 4):
        raise ValueError(
            f"Expected foot_contacts [T,4], got {tuple(foot_contacts.shape)}"
        )
    if posed_joints.shape[0] < 2:
        raise ValueError("Motion-quality metrics require at least two frames")

    feet = posed_joints[:, list(CONTACT_JOINTS)]
    toes = feet[:, [1, 3]]
    foot_velocity = (feet[1:] - feet[:-1]).norm(dim=-1) * float(fps)
    toe_velocity = (toes[1:] - toes[:-1]).norm(dim=-1) * float(fps)

    toe_on_floor_current = toes[:-1, :, 1] < 0.05
    height_mean = _safe_ratio(
        (toe_velocity * toe_on_floor_current).sum(),
        toe_on_floor_current.sum(),
    )

    predicted_contact = foot_contacts[:-1].bool()
    contact_velocity = foot_velocity * predicted_contact
    contact_mean = _safe_ratio(contact_velocity.sum(), predicted_contact.sum())
    contact_max = contact_velocity.amax()

    toe_on_floor_both = (toes[:-1, :, 1] < 0.05) & (toes[1:, :, 1] < 0.05)
    toe_skate = (toe_velocity * toe_on_floor_both) > 0.2
    skate_ratio = _safe_ratio(toe_skate.sum(), toe_on_floor_both.sum())

    velocity = torch.empty_like(posed_joints)
    velocity[:-1] = (posed_joints[1:] - posed_joints[:-1]) * float(fps)
    velocity[-1] = velocity[-2]
    heuristic_contacts = (
        velocity[:, list(CONTACT_JOINTS)].norm(dim=-1) < 0.15
    ) & (feet[..., 1] < 0.10)
    incorrect = torch.logical_xor(heuristic_contacts[:-1], foot_contacts[:-1].bool())
    contact_consistency = 1.0 - incorrect.float().sum() / float(4 * (posed_joints.shape[0] - 1))

    return {
        "foot_skate_from_height": float(height_mean.item()),
        "foot_skate_from_pred_contacts": float(contact_mean.item()),
        "foot_skate_max_vel": float(contact_max.item()),
        "foot_skate_ratio": float(skate_ratio.item()),
        "foot_contact_consistency": float(contact_consistency.item()),
    }


def _rotation_error_deg(
    pred_rot6d: torch.Tensor,
    target_rot6d: torch.Tensor,
) -> torch.Tensor:
    pred_matrix = cont6d_to_matrix(pred_rot6d)
    target_matrix = cont6d_to_matrix(target_rot6d)
    relative = pred_matrix.transpose(-1, -2) @ target_matrix
    cosine = (
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def evaluate_kimodo_constraint_case(
    predicted_features: torch.Tensor,
    target_features: torch.Tensor,
    constraint: CompiledKimodoConstraint,
    *,
    fps: float = FPS,
) -> dict[str, float]:
    """Run one generated or GT pass with Kimodo's rotation-decoded skeleton."""
    if predicted_features.shape != target_features.shape:
        raise ValueError(
            "Predicted/target feature shapes differ: "
            f"{tuple(predicted_features.shape)} vs {tuple(target_features.shape)}"
        )
    if predicted_features.ndim != 2 or predicted_features.shape[-1] != DIM_HY273:
        raise ValueError(
            f"Expected features [T,{DIM_HY273}], got {tuple(predicted_features.shape)}"
        )

    neutral_joints = load_smplx22_metric_joints(
        device=predicted_features.device,
        dtype=predicted_features.dtype,
    )
    predicted_joints = fk_positions_from_global_rot6d(
        predicted_features, neutral_joints=neutral_joints
    )
    target_joints = fk_positions_from_global_rot6d(
        target_features, neutral_joints=neutral_joints
    )
    metrics = kimodo_motion_quality_metrics(
        predicted_joints,
        predicted_features[..., CONTACT_SLICE] > 0.5,
        fps=fps,
    )

    if constraint.root_metric_frames.any():
        frames = constraint.root_metric_frames
        pred_root_xz = predicted_joints[frames, 0][:, [0, 2]]
        target_smooth_root_xz = target_features[frames][:, [0, 2]]
        root_error = (pred_root_xz - target_smooth_root_xz).norm(dim=-1)
        metrics["constraint_root2d_err"] = float(root_error.mean().item())
        metrics["constraint_root2d_acc"] = float(
            (root_error <= ROOT_THRESHOLD_M).float().mean().item()
        )

    if constraint.fullbody_metric_frames.any():
        frames = constraint.fullbody_metric_frames
        error = (predicted_joints[frames] - target_joints[frames]).norm(dim=-1)
        metrics["constraint_fullbody_keyframe"] = float(error.mean().item())

    if constraint.endpoint_position_metric_mask.any():
        mask = constraint.endpoint_position_metric_mask
        error = (predicted_joints - target_joints).norm(dim=-1)
        metrics["constraint_end_effector"] = float(error[mask].mean().item())

    if constraint.endpoint_rotation_metric_mask.any():
        pred_rot = predicted_features[..., GLOBAL_ROT_SLICE].reshape(
            predicted_features.shape[0], NUM_JOINTS, 6
        )
        target_rot = target_features[..., GLOBAL_ROT_SLICE].reshape(
            target_features.shape[0], NUM_JOINTS, 6
        )
        rotation_error = _rotation_error_deg(pred_rot, target_rot)
        metrics["constraint_end_effector_rotation_deg"] = float(
            rotation_error[constraint.endpoint_rotation_metric_mask].mean().item()
        )
    return metrics


def aggregate_case_records(records: Sequence[dict]) -> dict:
    """Aggregate per-case JSON records using Kimodo's per-motion averaging."""
    valid_records = [record for record in records if record.get("status") == "ok"]
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in valid_records:
        keys = (
            (record["text_regime"], "subtype", record["subtype"]),
            (record["text_regime"], "family", record["family"]),
            (record["text_regime"], "all", "all"),
        )
        for key in keys:
            groups.setdefault(key, []).append(record)

    def aggregate_metrics(group_records: Sequence[dict], pass_name: str) -> dict:
        metric_values: dict[str, list[float]] = {}
        for record in group_records:
            for name, value in record["metrics"][pass_name].items():
                if value is None or not math.isfinite(float(value)):
                    continue
                metric_values.setdefault(name, []).append(float(value))
        output: dict[str, float | int] = {}
        for name, values in sorted(metric_values.items()):
            tensor = torch.tensor(values, dtype=torch.float64)
            output[name] = float(tensor.mean().item())
            output[f"{name}__count"] = len(values)
            if name == "constraint_root2d_err":
                output["constraint_root2d_err_p95"] = float(
                    torch.quantile(tensor, 0.95).item()
                )
        return output

    rows = []
    for (regime, level, name), group_records in sorted(groups.items()):
        rows.append(
            {
                "text_regime": regime,
                "level": level,
                "name": name,
                "num_cases": len(group_records),
                "generated_raw": aggregate_metrics(group_records, "generated_raw"),
                "ground_truth": aggregate_metrics(group_records, "ground_truth"),
                "diagnostic_exact_clamp": aggregate_metrics(
                    group_records, "diagnostic_exact_clamp"
                ),
            }
        )
    return {
        "num_records": len(records),
        "num_success": len(valid_records),
        "num_failed": len(records) - len(valid_records),
        "rows": rows,
    }
