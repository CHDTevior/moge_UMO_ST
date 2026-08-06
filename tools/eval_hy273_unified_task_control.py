#!/usr/bin/env python3
"""Research benchmark for orthogonal T2M, Edit, and Reaction control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_multitask_manifest_dataset import (
    HY273MultitaskManifestDataset,
    collate_hy273_multitask,
)
from data.hy273_multitask_scheduler import EditConditionPattern, SamplePlan
from data.hy273_reaction_dataset import (
    HY273ReactionDataset,
    ReactionConditionPattern,
    collate_hy273_reaction,
)
from data.kimodo273_datasets import Kimodo273TextDataset
from models.raw_motion.hy273_kimodo_benchmark import (
    KIMODO_CONTROL_SUBTYPES,
    SUBTYPE_TO_FAMILY,
    CompiledKimodoConstraint,
    evaluate_kimodo_constraint_case,
    compile_kimodo_constraint,
)
from models.raw_motion.hy273_kimodo_contact_benchmark import (
    V5_CONTACT_BASE_SUBTYPE,
    V5_CONTACT_SUBTYPES,
    CompiledKimodoContactConstraint,
    compile_kimodo_contact_constraint,
    evaluate_kimodo_contact_case,
)
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    ConditionBatch,
    TrainStream,
    make_absent_condition,
)
from models.raw_motion.hy273_normalizer import apply_kimodo_training_transform
from models.raw_motion.hy273_reaction_metrics import reaction_fixed_role_metrics
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
from train_hy273_unified_actor import CHECKPOINT_FORMAT, validate_config


PROTOCOL = "hy273_unified_orthogonal_control_gate_v1"
TASKS = ("t2m", "edit", "reaction")
CONTROL_SUBTYPES = (*KIMODO_CONTROL_SUBTYPES, *V5_CONTACT_SUBTYPES)
DEFAULT_HML_ROOT = (
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22"
)
DEFAULT_HML_TEXT_ROOT = "/mnt/afs/mogo_base/datasets/HumanML3D/texts"
DEFAULT_MANIFEST_ROOT = Path(
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1"
)
DEFAULT_REACTION_ROOT = "/mnt/afs/mogo_base/datasets/InteractionK273/interx"

HIGHER_IS_BETTER = {
    "constraint_root2d_acc",
    "controlled_contact_accuracy",
    "controlled_contact_f1",
    "controlled_contact_exact_equality",
}
CONTROL_ADHERENCE_METRICS = {
    "constraint_root2d_err",
    "constraint_root2d_acc",
    "constraint_fullbody_keyframe",
    "constraint_end_effector",
    "constraint_end_effector_rotation_deg",
    "controlled_contact_bce",
    "controlled_contact_brier",
    "controlled_contact_accuracy",
    "controlled_contact_f1",
    "controlled_contact_exact_equality",
}


@dataclass(frozen=True)
class ControlCase:
    dataset_index: int
    subtype: str
    sample_seed: int


@dataclass
class PreparedCase:
    uid: str
    text: str
    target: torch.Tensor
    source: torch.Tensor | None
    condition: ConditionBatch
    constraint: CompiledKimodoConstraint | CompiledKimodoContactConstraint


def _stable_u64(seed: int, task: str, index: int, subtype: str, name: str) -> int:
    payload = f"{PROTOCOL}:{seed}:{task}:{index}:{subtype}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _case_plan(
    num_items: int,
    *,
    task: str,
    seed: int,
    cases_per_subtype: int,
) -> list[ControlCase]:
    if num_items <= 0:
        raise ValueError("Control benchmark dataset is empty")
    if cases_per_subtype < 0:
        raise ValueError("cases_per_subtype must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(num_items, generator=generator).tolist()
    assignments = [
        (index, CONTROL_SUBTYPES[position % len(CONTROL_SUBTYPES)])
        for position, index in enumerate(permutation)
    ]
    if cases_per_subtype:
        counts = {name: 0 for name in CONTROL_SUBTYPES}
        kept: list[tuple[int, str]] = []
        for index, subtype in assignments:
            if counts[subtype] >= cases_per_subtype:
                continue
            counts[subtype] += 1
            kept.append((index, subtype))
        assignments = kept
    return [
        ControlCase(
            dataset_index=index,
            subtype=subtype,
            sample_seed=_stable_u64(
                seed, task, index, subtype, "sample"
            )
            % (2**31 - 1),
        )
        for index, subtype in assignments
    ]


def _compile_constraint(
    target: torch.Tensor,
    case: ControlCase,
    *,
    max_sparse_keyframes: int,
) -> CompiledKimodoConstraint | CompiledKimodoContactConstraint:
    if case.subtype in V5_CONTACT_SUBTYPES:
        return compile_kimodo_contact_constraint(
            target,
            case.subtype,
            seed=case.sample_seed,
            max_sparse_keyframes=max_sparse_keyframes,
        )
    return compile_kimodo_constraint(
        target,
        case.subtype,
        seed=case.sample_seed,
        max_sparse_keyframes=max_sparse_keyframes,
    )


def _prepare_t2m(
    dataset: Kimodo273TextDataset,
    case: ControlCase,
    *,
    max_sparse_keyframes: int,
) -> PreparedCase:
    item = dataset[case.dataset_index]
    transformed = apply_kimodo_training_transform(
        item["motion"].float().unsqueeze(0),
        random_heading=False,
        root_shift=True,
    )
    target = transformed.motion[0].contiguous()
    condition = make_absent_condition(
        batch_size=1,
        target_frames=target.shape[0],
        target_lengths=torch.tensor([target.shape[0]]),
        capability=CapabilityId.KIMODO_CONTROL,
    )
    condition = replace(
        condition,
        frame_gauge_dir=transformed.c_dir.reshape(1, 2).float(),
    )
    condition.validate()
    return PreparedCase(
        uid=str(item["motion_id"]),
        text=str(item["text"]),
        target=target,
        source=None,
        condition=condition,
        constraint=_compile_constraint(
            target, case, max_sparse_keyframes=max_sparse_keyframes
        ),
    )


def _prepare_edit(
    dataset: HY273MultitaskManifestDataset,
    case: ControlCase,
    *,
    seed: int,
    max_sparse_keyframes: int,
) -> PreparedCase:
    uid = dataset.uid(case.dataset_index)
    plan = SamplePlan(
        global_step=350_000,
        global_sample_ordinal=case.dataset_index,
        train_stream_id=TrainStream.MOTION_EDIT,
        capability_id=CapabilityId.MOTION_EDIT_CONTROL,
        row_index=case.dataset_index,
        uid=uid,
        caption_index=None,
        yaw_u64=_stable_u64(seed, "edit", case.dataset_index, case.subtype, "yaw"),
        control_u64=_stable_u64(
            seed, "edit", case.dataset_index, case.subtype, "control"
        ),
        text_drop=False,
        edit_pattern=EditConditionPattern.SOURCE_TEXT,
        control_present=True,
        ease_present=False,
    )
    batch = collate_hy273_multitask([dataset.materialize(plan)])
    target = batch["target_motion"][0, : int(batch["condition"].requested_target_len[0])]
    source_length = int(batch["condition"].source_native_lengths[0, 0])
    source = batch["condition"].source_motion[0, 0, :source_length].contiguous()
    return PreparedCase(
        uid=uid,
        text=str(batch["texts"][0]),
        target=target.contiguous(),
        source=source,
        condition=batch["condition"],
        constraint=_compile_constraint(
            target, case, max_sparse_keyframes=max_sparse_keyframes
        ),
    )


def _prepare_reaction(
    dataset: HY273ReactionDataset,
    case: ControlCase,
    *,
    seed: int,
    max_sparse_keyframes: int,
) -> PreparedCase:
    plan = dataset.build_plan(
        row_index=case.dataset_index,
        global_step=350_000,
        global_sample_ordinal=case.dataset_index,
        run_seed=seed,
        orthogonal_control_probability=1.0,
    )
    plan = replace(
        plan,
        condition_pattern=ReactionConditionPattern.SOURCE_AND_TEXT,
        control_present=True,
    )
    batch = collate_hy273_reaction([dataset.materialize(plan)])
    length = int(batch["condition"].requested_target_len[0])
    target = batch["target_motion"][0, :length].contiguous()
    source = batch["condition"].source_motion[0, 0, :length].contiguous()
    return PreparedCase(
        uid=plan.uid,
        text=str(batch["texts"][0]),
        target=target,
        source=source,
        condition=batch["condition"],
        constraint=_compile_constraint(
            target, case, max_sparse_keyframes=max_sparse_keyframes
        ),
    )


def _evaluate_constraint(
    prediction: torch.Tensor,
    prepared: PreparedCase,
) -> dict[str, float | int]:
    if isinstance(prepared.constraint, CompiledKimodoContactConstraint):
        return evaluate_kimodo_contact_case(
            prediction, prepared.target, prepared.constraint
        )
    return evaluate_kimodo_constraint_case(
        prediction, prepared.target, prepared.constraint
    )


def _target_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred_pos = reconstruct_global_joints_from_features(prediction)
    target_pos = reconstruct_global_joints_from_features(target)
    pred_fk = fk_positions_from_global_rot6d(prediction)
    target_fk = fk_positions_from_global_rot6d(target)
    metrics = {
        "target_position_mpjpe_cm": float(
            (pred_pos - target_pos).norm(dim=-1).mean().item() * 100.0
        ),
        "target_fk_mpjpe_cm": float(
            (pred_fk - target_fk).norm(dim=-1).mean().item() * 100.0
        ),
        "contact_accuracy": float(
            (
                (prediction[..., CONTACT_SLICE] >= 0.5)
                == (target[..., CONTACT_SLICE] >= 0.5)
            )
            .float()
            .mean()
            .item()
        ),
    }
    if prediction.shape[0] >= 4:
        pred_jerk = torch.diff(pred_fk, n=3, dim=0) * 30.0**3
        target_jerk = torch.diff(target_fk, n=3, dim=0) * 30.0**3
        metrics["target_fk_jerk_error_mps3"] = float(
            (pred_jerk - target_jerk).norm(dim=-1).mean().item()
        )
    return metrics


def _task_metrics(
    task: str,
    prediction: torch.Tensor,
    prepared: PreparedCase,
) -> dict[str, float]:
    if task == "t2m":
        metrics = _evaluate_constraint(prediction, prepared)
        return {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith("foot_")
            or key == "fk_position_rotation_consistency_cm"
        }
    if task == "edit":
        return _target_metrics(prediction, prepared.target)
    if prepared.source is None:
        raise RuntimeError("Reaction benchmark case has no observed actor")
    return reaction_fixed_role_metrics(
        prepared.source,
        prediction,
        prepared.target,
    )["aggregate"]


def _load_runtime(
    checkpoint_path: Path,
    *,
    weight_source: str,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("Control gate requires a unified-actor checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Unified-actor checkpoint has no config")
    validate_config(config)
    if str(config.get("data", {}).get("paired_task")) != "reaction":
        raise RuntimeError("Control gate requires the fixed-role Reaction model")
    model = create_model_from_checkpoint(checkpoint).to(device)
    state = checkpoint.get(weight_source)
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Checkpoint has no {weight_source!r} state")
    model.load_state_dict(state, strict=True)
    model.eval()
    normalizer = normalizer_from_checkpoint(checkpoint, device)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "next_global_step": int(checkpoint.get("next_global_step", -1)),
        "weight_source": weight_source,
        "run_name": str(checkpoint.get("run_name", "")),
    }
    del state, checkpoint
    return model, normalizer, metadata


def _sample_pair(
    *,
    model: torch.nn.Module,
    normalizer: Any,
    prepared: PreparedCase,
    case: ControlCase,
    args: argparse.Namespace,
) -> tuple[Any, Any]:
    constraint = prepared.constraint
    noise_generator = torch.Generator(device="cpu").manual_seed(case.sample_seed)
    initial_noise = torch.randn(
        1,
        prepared.target.shape[0],
        DIM_HY273,
        generator=noise_generator,
        dtype=torch.float32,
    )
    common = {
        "model": model,
        "normalizer": normalizer,
        "condition": prepared.condition,
        "texts": [prepared.text],
        "observed_physical": constraint.observed_motion.unsqueeze(0),
        "hard_mask": constraint.motion_mask.unsqueeze(0),
        "num_steps": args.num_steps,
        "text_cfg_scale": args.text_cfg_scale,
        "source_cfg_scale": args.source_cfg_scale,
        "edit_cfg_scale": args.edit_cfg_scale,
        "initial_unified_noise": initial_noise,
    }
    controlled = sample_hy273_multitask_ode(
        **common,
        control_cfg_scale=args.control_cfg_scale,
    )
    control_zero = sample_hy273_multitask_ode(
        **common,
        control_cfg_scale=0.0,
    )
    return controlled, control_zero


def _numeric_mean(rows: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: float(np.mean([float(row[key]) for row in rows if key in row]))
        for key in keys
    }


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(f"{record['task']}/all", []).append(record)
        groups.setdefault(
            f"{record['task']}/{record['subtype']}", []
        ).append(record)
    summary: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        controlled = _numeric_mean([row["controlled"] for row in rows])
        control_zero = _numeric_mean([row["control_zero"] for row in rows])
        common = sorted(set(controlled) & set(control_zero))
        improvement = {
            key: (
                controlled[key] - control_zero[key]
                if key in HIGHER_IS_BETTER
                else control_zero[key] - controlled[key]
            )
            for key in common
            if key in CONTROL_ADHERENCE_METRICS
        }
        summary[name] = {
            "cases": len(rows),
            "controlled": controlled,
            "control_zero": control_zero,
            "positive_means_control_helped": improvement,
            "controlled_task_metrics": _numeric_mean(
                [row["controlled_task_metrics"] for row in rows]
            ),
            "control_zero_task_metrics": _numeric_mean(
                [row["control_zero_task_metrics"] for row in rows]
            ),
        }
    return summary


def _dataset(task: str, args: argparse.Namespace) -> Any:
    if task == "t2m":
        return Kimodo273TextDataset(
            args.hml_root,
            split=args.split,
            text_root=args.hml_text_root,
            max_frames=args.max_frames,
            min_frames=2,
            random_crop=False,
            exclude_fallback_short_clips=False,
            deterministic_text=True,
            caption_policy="first_full_motion",
        )
    if task == "edit":
        manifest = (
            Path(args.multitask_manifest).expanduser().resolve()
            if args.multitask_manifest
            else DEFAULT_MANIFEST_ROOT / f"{args.split}.jsonl"
        )
        return HY273MultitaskManifestDataset(
            manifest,
            TrainStream.MOTION_EDIT,
            verify_payload_hash=False,
        )
    return HY273ReactionDataset(
        args.reaction_root,
        split=args.split,
        min_frames=16,
        max_frames=args.max_frames,
        exclude_overlength=True,
        exclude_known_test_anomalies=True,
    )


def _family(subtype: str) -> str:
    if subtype in SUBTYPE_TO_FAMILY:
        return SUBTYPE_TO_FAMILY[subtype]
    base = V5_CONTACT_BASE_SUBTYPE[subtype]
    return "contact" if base is None else f"contact+{SUBTYPE_TO_FAMILY[base]}"


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Control evaluation requires CUDA")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0,num_shards)")
    task = str(args.task)
    device = torch.device(args.device)
    model, normalizer, checkpoint_meta = _load_runtime(
        Path(args.checkpoint).expanduser().resolve(),
        weight_source=args.weight_source,
        device=device,
    )
    dataset = _dataset(task, args)
    plan = _case_plan(
        len(dataset),
        task=task,
        seed=args.seed,
        cases_per_subtype=args.cases_per_subtype,
    )
    shard_plan = [
        case
        for position, case in enumerate(plan)
        if position % args.num_shards == args.shard_id
    ]
    records: list[dict[str, Any]] = []
    for position, case in enumerate(shard_plan, 1):
        if task == "t2m":
            prepared = _prepare_t2m(
                dataset,
                case,
                max_sparse_keyframes=args.max_sparse_keyframes,
            )
        elif task == "edit":
            prepared = _prepare_edit(
                dataset,
                case,
                seed=args.seed,
                max_sparse_keyframes=args.max_sparse_keyframes,
            )
        else:
            prepared = _prepare_reaction(
                dataset,
                case,
                seed=args.seed,
                max_sparse_keyframes=args.max_sparse_keyframes,
            )
        controlled, control_zero = _sample_pair(
            model=model,
            normalizer=normalizer,
            prepared=prepared,
            case=case,
            args=args,
        )
        controlled_raw = controlled.raw_motion[0].detach().cpu()
        control_zero_raw = control_zero.raw_motion[0].detach().cpu()
        exact = controlled.exact_clamped_motion[0].detach().cpu()
        records.append(
            {
                "task": task,
                "dataset_index": case.dataset_index,
                "uid": prepared.uid,
                "subtype": case.subtype,
                "family": _family(case.subtype),
                "sample_seed": case.sample_seed,
                "length": int(prepared.target.shape[0]),
                "text": prepared.text,
                "mask_fraction": float(
                    prepared.constraint.motion_mask.float().mean().item()
                ),
                "controlled": _evaluate_constraint(controlled_raw, prepared),
                "control_zero": _evaluate_constraint(control_zero_raw, prepared),
                "diagnostic_exact_clamp": _evaluate_constraint(exact, prepared),
                "controlled_task_metrics": _task_metrics(
                    task, controlled_raw, prepared
                ),
                "control_zero_task_metrics": _task_metrics(
                    task, control_zero_raw, prepared
                ),
            }
        )
        print(
            f"[{task}] shard={args.shard_id} case={position}/{len(shard_plan)} "
            f"uid={prepared.uid} subtype={case.subtype}",
            flush=True,
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task}_shard_{args.shard_id:02d}.json"
    payload = {
        "protocol": PROTOCOL,
        "checkpoint": checkpoint_meta,
        "task": task,
        "split": args.split,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "cfg": {
            "text": args.text_cfg_scale,
            "source": args.source_cfg_scale,
            "edit": args.edit_cfg_scale,
            "control": args.control_cfg_scale,
            "paired_baseline_control_cfg": 0.0,
        },
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "cases_per_subtype": args.cases_per_subtype,
        "records": records,
        "summary": _summarize(records),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "cases": len(records)}))


def aggregate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    paths = sorted(output_dir.glob(f"{args.task}_shard_*.json"))
    if not paths:
        raise FileNotFoundError(f"No shard outputs for task={args.task}")
    records: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL or payload.get("task") != args.task:
            raise RuntimeError(f"Unexpected control shard payload: {path}")
        current = {
            key: payload[key]
            for key in ("checkpoint", "task", "split", "seed", "num_steps", "cfg")
        }
        if metadata is None:
            metadata = current
        elif current != metadata:
            raise RuntimeError("Control shard protocols differ")
        records.extend(payload["records"])
    records.sort(key=lambda row: (row["dataset_index"], row["subtype"]))
    result = {
        "protocol": PROTOCOL,
        **(metadata or {}),
        "shards": [str(path) for path in paths],
        "records": records,
        "summary": _summarize(records),
    }
    output_path = output_dir / f"{args.task}_summary.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "cases": len(records)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--weight_source", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--cases_per_subtype", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--text_cfg_scale", type=float, default=2.0)
    parser.add_argument("--source_cfg_scale", type=float, default=2.0)
    parser.add_argument("--edit_cfg_scale", type=float, default=2.0)
    parser.add_argument("--control_cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--max_sparse_keyframes", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--hml_root", default=DEFAULT_HML_ROOT)
    parser.add_argument("--hml_text_root", default=DEFAULT_HML_TEXT_ROOT)
    parser.add_argument("--multitask_manifest", default="")
    parser.add_argument("--reaction_root", default=DEFAULT_REACTION_ROOT)
    parser.add_argument("--aggregate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --aggregate is used")
        run(args)


if __name__ == "__main__":
    main()
