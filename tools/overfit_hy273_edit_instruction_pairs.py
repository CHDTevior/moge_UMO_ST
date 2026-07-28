#!/usr/bin/env python3
"""Tiny-overfit test for instruction use in R13 Motion Editing.

Each training group contains one source motion and two different
instruction/target pairs. The two rows share source context, flow noise,
timestep, gauge, and length, so text is the only target-selecting input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.flow_schedule import build_unified_273_flow_state
from models.raw_motion.hy273_multitask_condition import RELATIVE_EDIT_TEXT_PROFILE
from models.raw_motion.hy273_multitask_losses import (
    HY273MultitaskLossWeights,
    compute_hy273_unified_flow_loss,
)
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM, DIM_HY273
from models.raw_motion.hy273_unified_edit_losses import (
    UnifiedEditLossWeights,
    compute_unified_edit_loss,
)
from sample_hy273_multitask import make_edit_condition, normalizer_from_checkpoint
from tools.diagnose_hy273_r13_edit_fixed_t import load_k273, load_rows, to_gauge
from train_hy273_multitask import (
    create_model,
    optimizer_groups,
    repeat_condition_batch,
    validate_frozen_contract,
)


DEFAULT_PARENT = (
    ROOT
    / "outputs/hy273_multitask/"
    "hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/"
    "model/step_00400000.pt"
)
DEFAULT_MANIFEST = Path(
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/train.jsonl"
)
DEFAULT_CANDIDATES = (
    ROOT
    / "outputs/hy273_multitask/diagnostics/"
    "r13_edit_objective_pilot_405k_20260722/tiny_overfit_candidate_groups.json"
)
MEMORIZATION_FIDELITY_FRACTION = 0.10
MEMORIZATION_CONTACT_FIDELITY_FRACTION = 0.25


@dataclass(frozen=True)
class CandidateGroup:
    group_index: int
    source_sha256: str
    source_base_motion_id: str
    pair_ids: tuple[str, str]
    texts: tuple[str, str]
    frames: int
    target_pair_mse: float


@dataclass
class MaterializedGroup:
    candidate: CandidateGroup
    source: torch.Tensor
    targets: tuple[torch.Tensor, torch.Tensor]
    texts: tuple[str, str]
    noise: torch.Tensor


def parse_int_csv(value: str) -> tuple[int, ...]:
    rows = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not rows:
        raise ValueError("Expected at least one integer")
    return rows


def parse_optional_int_csv(value: str) -> tuple[int, ...]:
    if not value.strip() or value.strip().casefold() in {"none", "off"}:
        return ()
    return parse_int_csv(value)


def select_candidate_groups(
    payload: Sequence[dict[str, Any]],
    *,
    count: int,
    max_frames: int,
    minimum_target_pair_mse: float,
) -> list[CandidateGroup]:
    if count <= 0:
        raise ValueError("count must be positive")
    selected: list[CandidateGroup] = []
    seen_pairs: set[str] = set()
    seen_sources: set[str] = set()
    for source_index, row in enumerate(payload):
        pair_ids = tuple(str(value) for value in row.get("pair_ids", ()))
        texts = tuple(str(value) for value in row.get("texts", ()))
        normalized_texts = tuple(" ".join(value.split()).casefold() for value in texts)
        frames = int(row.get("frames", 0))
        pair_mse = float(row.get("target_pair_mse", float("nan")))
        source_sha = str(row.get("source_sha256", ""))
        if (
            len(pair_ids) != 2
            or len(set(pair_ids)) != 2
            or len(texts) != 2
            or len(set(normalized_texts)) != 2
            or not all(text.strip() for text in texts)
            or frames <= 0
            or frames > int(max_frames)
            or not math.isfinite(pair_mse)
            or pair_mse < float(minimum_target_pair_mse)
            or not source_sha
        ):
            continue
        if seen_pairs.intersection(pair_ids) or source_sha in seen_sources:
            continue
        selected.append(
            CandidateGroup(
                group_index=len(selected),
                source_sha256=source_sha,
                source_base_motion_id=str(row.get("source_base_motion_id", "")),
                pair_ids=(pair_ids[0], pair_ids[1]),
                texts=(texts[0], texts[1]),
                frames=frames,
                target_pair_mse=pair_mse,
            )
        )
        seen_pairs.update(pair_ids)
        seen_sources.add(source_sha)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Only selected {len(selected)} valid groups, requested {count}"
        )
    return selected


def _manifest_text(row: dict[str, Any]) -> str:
    texts = row.get("texts")
    if not isinstance(texts, list) or len(texts) != 1:
        raise ValueError("MotionFix tiny-overfit rows require exactly one instruction")
    value = str(texts[0].get("value", ""))
    if not value.strip():
        raise ValueError("MotionFix instruction is empty")
    return value


def materialize_groups(
    candidates: Sequence[CandidateGroup],
    *,
    manifest_path: Path,
    noise_seed: int,
) -> list[MaterializedGroup]:
    flat_pair_ids = [pair_id for group in candidates for pair_id in group.pair_ids]
    rows = load_rows(manifest_path, flat_pair_ids)
    by_id = {str(row["pair"]["official_pair_id"]): row for row in rows}
    output: list[MaterializedGroup] = []
    for group in candidates:
        row_a, row_b = (by_id[pair_id] for pair_id in group.pair_ids)
        source_a = to_gauge(load_k273(row_a["source_motion"]["k273_asset"]), 0.0)
        source_b = to_gauge(load_k273(row_b["source_motion"]["k273_asset"]), 0.0)
        if source_a.shape != source_b.shape or not torch.equal(source_a, source_b):
            maximum = (
                float((source_a - source_b).abs().max().item())
                if source_a.shape == source_b.shape
                else float("inf")
            )
            raise RuntimeError(
                f"Group {group.pair_ids} does not have an identical paired source; "
                f"max_abs_delta={maximum}"
            )
        targets = tuple(
            to_gauge(load_k273(row["target_motion"]["k273_asset"]), 0.0)
            for row in (row_a, row_b)
        )
        if targets[0].shape != targets[1].shape:
            raise RuntimeError(f"Target lengths differ for group {group.pair_ids}")
        if targets[0].shape[0] != group.frames:
            raise RuntimeError(
                f"Candidate frame count drift for {group.pair_ids}: "
                f"{targets[0].shape[0]} != {group.frames}"
            )
        manifest_texts = (_manifest_text(row_a), _manifest_text(row_b))
        if manifest_texts != group.texts:
            raise RuntimeError(
                f"Candidate instruction drift for {group.pair_ids}: "
                f"{manifest_texts!r} != {group.texts!r}"
            )
        generator = torch.Generator(device="cpu").manual_seed(
            int(noise_seed) + 104729 * int(group.group_index)
        )
        noise = torch.randn(
            group.frames, DIM_HY273, generator=generator, dtype=torch.float32
        )
        output.append(
            MaterializedGroup(
                candidate=group,
                source=source_a,
                targets=(targets[0], targets[1]),
                texts=manifest_texts,
                noise=noise,
            )
        )
    return output


def collate_groups(
    groups: Sequence[MaterializedGroup],
    group_indices: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    selected = [groups[int(index)] for index in group_indices]
    if not selected:
        raise ValueError("Cannot collate an empty group batch")
    batch_size = 2 * len(selected)
    max_target = max(group.targets[0].shape[0] for group in selected)
    max_source = max(group.source.shape[0] for group in selected)
    source = torch.zeros(batch_size, max_source, DIM_HY273, dtype=torch.float32)
    target = torch.zeros(batch_size, max_target, DIM_HY273, dtype=torch.float32)
    noise = torch.zeros_like(target)
    source_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    texts: list[str] = []
    pair_ids: list[str] = []
    group_ids: list[int] = []
    for local_group, group in enumerate(selected):
        start = 2 * local_group
        source_length = group.source.shape[0]
        target_length = group.targets[0].shape[0]
        for branch in range(2):
            row = start + branch
            source[row, :source_length] = group.source
            target[row, :target_length] = group.targets[branch]
            noise[row, :target_length] = group.noise
            source_lengths[row] = source_length
            target_lengths[row] = target_length
            texts.append(group.texts[branch])
            pair_ids.append(group.candidate.pair_ids[branch])
            group_ids.append(group.candidate.group_index)
    if not all(
        torch.equal(noise[index], noise[index + 1])
        and torch.equal(source[index], source[index + 1])
        for index in range(0, batch_size, 2)
    ):
        raise RuntimeError("Paired source/noise invariant failed before device transfer")
    source = source.to(device)
    target = target.to(device)
    noise = noise.to(device)
    source_lengths = source_lengths.to(device)
    target_lengths = target_lengths.to(device)
    gauge = torch.zeros(batch_size, 2, device=device, dtype=torch.float32)
    gauge[:, 0] = 1.0
    condition = make_edit_condition(
        source,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        target_frames=max_target,
        frame_gauge_dir=gauge,
    )
    valid = condition.target_valid
    return {
        "source": source,
        "target": target,
        "noise": noise,
        "source_lengths": source_lengths,
        "target_lengths": target_lengths,
        "condition": condition,
        "valid": valid,
        "texts": texts,
        "swapped_texts": [
            texts[index + 1] if index % 2 == 0 else texts[index - 1]
            for index in range(batch_size)
        ],
        "pair_ids": pair_ids,
        "group_ids": group_ids,
    }


def _space_assignment_metrics(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> dict[str, float]:
    distances = torch.stack(
        [
            (pred_a - target_a).square().mean(),
            (pred_a - target_b).square().mean(),
            (pred_b - target_a).square().mean(),
            (pred_b - target_b).square().mean(),
        ]
    )
    d_aa, d_ab, d_ba, d_bb = (float(value.item()) for value in distances)
    target_pair_mse = float((target_a - target_b).square().mean().item())
    midpoint_mse = 0.25 * target_pair_mse
    correct_mse = 0.5 * (d_aa + d_bb)
    swapped_mse = 0.5 * (d_ab + d_ba)
    fidelity_ratio = (
        correct_mse / midpoint_mse
        if midpoint_mse > 1e-12
        else (0.0 if correct_mse <= 1e-12 else float("inf"))
    )
    return {
        "pred_a_target_a": d_aa,
        "pred_a_target_b": d_ab,
        "pred_b_target_a": d_ba,
        "pred_b_target_b": d_bb,
        "correct_mse": correct_mse,
        "swapped_mse": swapped_mse,
        "instruction_margin": swapped_mse - correct_mse,
        "assignment_correct": float(d_aa + d_bb < d_ab + d_ba),
        "row_assignment_accuracy": 0.5
        * (float(d_aa < d_ab) + float(d_bb < d_ba)),
        "output_pair_rms": float((pred_a - pred_b).square().mean().sqrt().item()),
        "target_pair_rms": math.sqrt(max(target_pair_mse, 0.0)),
        "target_pair_mse": target_pair_mse,
        "text_blind_midpoint_mse": midpoint_mse,
        "fidelity_ratio_vs_midpoint": fidelity_ratio,
    }


def _gate_group(record: dict[str, Any]) -> bool:
    full = record["spaces"]["full_273"]
    continuous = record["spaces"]["continuous_269"]
    contact = record["spaces"]["contact_4"]
    contact_separates = float(contact["target_pair_mse"]) > 1e-12
    return bool(
        full["assignment_correct"] == 1.0
        and full["row_assignment_accuracy"] == 1.0
        and continuous["assignment_correct"] == 1.0
        and continuous["row_assignment_accuracy"] == 1.0
        and full["fidelity_ratio_vs_midpoint"]
        <= MEMORIZATION_FIDELITY_FRACTION
        and continuous["fidelity_ratio_vs_midpoint"]
        <= MEMORIZATION_FIDELITY_FRACTION
        and (
            not contact_separates
            or (
                contact["assignment_correct"] == 1.0
                and contact["row_assignment_accuracy"] == 1.0
                and contact["fidelity_ratio_vs_midpoint"]
                <= MEMORIZATION_CONTACT_FIDELITY_FRACTION
            )
        )
    )


def aggregate_assignment_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate empty assignment records")
    spaces: dict[str, dict[str, float]] = {}
    scalar_fields = (
        "correct_mse",
        "swapped_mse",
        "instruction_margin",
        "assignment_correct",
        "row_assignment_accuracy",
        "output_pair_rms",
        "target_pair_rms",
        "target_pair_mse",
        "text_blind_midpoint_mse",
        "fidelity_ratio_vs_midpoint",
    )
    for space_name in ("full_273", "continuous_269", "contact_4"):
        spaces[space_name] = {
            key: float(
                sum(float(row["spaces"][space_name][key]) for row in records)
                / len(records)
            )
            for key in scalar_fields
        }
    memorized_count = sum(bool(row["memorized"]) for row in records)
    return {
        "spaces": spaces,
        "memorized_groups": int(memorized_count),
        "total_groups": int(len(records)),
        "memorized_group_fraction": float(memorized_count / len(records)),
        "memorized": bool(memorized_count == len(records)),
        "gate": {
            "full_and_continuous_fidelity_fraction_vs_text_blind_midpoint": (
                MEMORIZATION_FIDELITY_FRACTION
            ),
            "contact_fidelity_fraction_vs_text_blind_midpoint": (
                MEMORIZATION_CONTACT_FIDELITY_FRACTION
            ),
            "requires_all_group_and_row_assignments": True,
            "requires_all_groups": True,
        },
    }


def paired_assignment_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_lengths: torch.Tensor,
    group_ids: Sequence[int],
    pair_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if prediction.shape != target.shape or prediction.shape[-1] != DIM_HY273:
        raise ValueError("prediction and target must match [B,T,273]")
    if prediction.shape[0] % 2:
        raise ValueError("Paired assignment requires an even batch")
    records: list[dict[str, Any]] = []
    for start in range(0, prediction.shape[0], 2):
        if int(group_ids[start]) != int(group_ids[start + 1]):
            raise ValueError("Adjacent rows do not belong to the same group")
        length_a = int(target_lengths[start].item())
        length_b = int(target_lengths[start + 1].item())
        if length_a != length_b:
            raise ValueError("Paired targets must have the same length")
        length = length_a
        pred_a = prediction[start, :length].float()
        pred_b = prediction[start + 1, :length].float()
        target_a = target[start, :length].float()
        target_b = target[start + 1, :length].float()
        spaces = {
            "full_273": _space_assignment_metrics(
                pred_a, pred_b, target_a, target_b
            ),
            "continuous_269": _space_assignment_metrics(
                pred_a[..., :CONT_DIM],
                pred_b[..., :CONT_DIM],
                target_a[..., :CONT_DIM],
                target_b[..., :CONT_DIM],
            ),
            "contact_4": _space_assignment_metrics(
                pred_a[..., CONTACT_SLICE],
                pred_b[..., CONTACT_SLICE],
                target_a[..., CONTACT_SLICE],
                target_b[..., CONTACT_SLICE],
            ),
        }
        record = {
            "group_id": int(group_ids[start]),
            "pair_ids": [str(pair_ids[start]), str(pair_ids[start + 1])],
            "frames": length,
            "spaces": spaces,
        }
        record["memorized"] = _gate_group(record)
        records.append(record)
    return records, aggregate_assignment_records(records)


def _batch_group_indices(
    step: int, *, num_groups: int, groups_per_batch: int
) -> tuple[int, ...]:
    if num_groups <= 0 or groups_per_batch <= 0 or groups_per_batch > num_groups:
        raise ValueError("Invalid group batch dimensions")
    start = ((int(step) - 1) * groups_per_batch) % num_groups
    return tuple((start + offset) % num_groups for offset in range(groups_per_batch))


def _autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


@torch.inference_mode()
def model_visible_text_diagnostics(
    model: torch.nn.Module,
    groups: Sequence[MaterializedGroup],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    texts = [text for group in groups for text in group.texts]
    profiles = [RELATIVE_EDIT_TEXT_PROFILE] * len(texts)
    was_training = model.training
    model.eval()
    encoded = model.text_encoder(
        texts,
        device=device,
        dtype=torch.float32,
        drop_prob=0.0,
        force_drop=False,
        profiles=profiles,
    )
    rows = []
    for index, group in enumerate(groups):
        first = 2 * index
        token_delta = float(
            (encoded.tokens[first] - encoded.tokens[first + 1]).abs().max().item()
        )
        pooled_delta = float(
            (encoded.pooled[first] - encoded.pooled[first + 1]).abs().max().item()
        )
        if token_delta == 0.0 and pooled_delta == 0.0:
            raise RuntimeError(
                f"Model-visible text encodings are identical for {group.candidate.pair_ids}"
            )
        rows.append(
            {
                "group_id": group.candidate.group_index,
                "pair_ids": list(group.candidate.pair_ids),
                "token_max_abs_delta": token_delta,
                "pooled_max_abs_delta": pooled_delta,
            }
        )
    model.train(was_training)
    return rows


def materialized_target_diagnostics(
    groups: Sequence[MaterializedGroup],
    normalizer: Any,
    *,
    minimum_continuous_pair_mse: float,
) -> list[dict[str, Any]]:
    rows = []
    device = normalizer.mean.device
    for group in groups:
        target_a = normalizer.normalize(group.targets[0].to(device)).float()
        target_b = normalizer.normalize(group.targets[1].to(device)).float()
        full_mse = float((target_a - target_b).square().mean().item())
        continuous_mse = float(
            (target_a[..., :CONT_DIM] - target_b[..., :CONT_DIM])
            .square()
            .mean()
            .item()
        )
        contact_mse = float(
            (target_a[..., CONTACT_SLICE] - target_b[..., CONTACT_SLICE])
            .square()
            .mean()
            .item()
        )
        if continuous_mse < float(minimum_continuous_pair_mse):
            raise RuntimeError(
                f"Materialized targets are insufficiently separated for "
                f"{group.candidate.pair_ids}: {continuous_mse}"
            )
        rows.append(
            {
                "group_id": group.candidate.group_index,
                "pair_ids": list(group.candidate.pair_ids),
                "candidate_target_pair_mse": group.candidate.target_pair_mse,
                "normalized_full_273_pair_mse": full_mse,
                "normalized_continuous_269_pair_mse": continuous_mse,
                "normalized_contact_4_pair_mse": contact_mse,
            }
        )
    return rows


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    groups: Sequence[MaterializedGroup],
    normalizer: Any,
    *,
    timestep: float,
    groups_per_batch: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    invariant_max = 0.0
    for start in range(0, len(groups), groups_per_batch):
        indices = tuple(range(start, min(start + groups_per_batch, len(groups))))
        batch = collate_groups(groups, indices, device=device)
        target_norm = normalizer.normalize(batch["target"])
        t = torch.full(
            (target_norm.shape[0],),
            float(timestep),
            device=device,
            dtype=target_norm.dtype,
        )
        zeros = torch.zeros_like(target_norm)
        state = build_unified_273_flow_state(
            target_norm,
            zeros,
            torch.zeros_like(target_norm, dtype=torch.bool),
            t,
            noise=batch["noise"],
        )
        for pair_start in range(0, target_norm.shape[0], 2):
            invariant_max = max(
                invariant_max,
                float(
                    (
                        state["model_in"][pair_start]
                        - state["model_in"][pair_start + 1]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
        with _autocast_context(device, precision):
            prediction = model(
                state["model_in"],
                t=t,
                c_dir=batch["condition"].frame_gauge_dir,
                text=batch["texts"],
                length_mask=batch["valid"],
                x_self_cond=None,
                text_drop_prob=0.0,
                condition=batch["condition"],
            )
        rows, _ = paired_assignment_metrics(
            prediction,
            target_norm,
            batch["target_lengths"],
            batch["group_ids"],
            batch["pair_ids"],
        )
        records.extend(rows)
    if invariant_max != 0.0:
        raise RuntimeError(f"Paired model inputs differ by {invariant_max}")
    aggregate = aggregate_assignment_records(records)
    aggregate["paired_model_input_max_abs_delta"] = invariant_max
    return {"aggregate": aggregate, "groups": records}


def _jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def save_model_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    parent_checkpoint: Path,
    parent_step: int,
    train_steps: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    selected_groups: Sequence[CandidateGroup],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "hy273_edit_instruction_tiny_overfit_v1",
            "parent_checkpoint": str(parent_checkpoint),
            "parent_step": int(parent_step),
            "tiny_overfit_steps": int(train_steps),
            "config": config,
            "args": _jsonable_args(args),
            "selected_groups": [group.__dict__ for group in selected_groups],
            "model": model.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate_groups", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_groups", type=int, default=16)
    parser.add_argument("--groups_per_batch", type=int, default=4)
    parser.add_argument("--eval_groups_per_batch", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=150)
    parser.add_argument("--minimum_target_pair_mse", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_steps", default="0,100,500,1000,2000")
    parser.add_argument("--save_model_steps", default="2000")
    parser.add_argument("--timestep", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=20260722)
    parser.add_argument("--train_seed", type=int, default=20260723)
    parser.add_argument("--base_lr", type=float, default=2e-5)
    parser.add_argument("--context_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.steps <= 0 or args.log_every <= 0:
        raise ValueError("steps and log_every must be positive")
    if float(args.timestep) != 0.0:
        raise ValueError(
            "This text-identifiability experiment requires exact t=0 so paired "
            "model inputs contain no target leakage"
        )
    eval_steps = tuple(sorted(set(parse_int_csv(args.eval_steps))))
    save_steps = set(parse_optional_int_csv(args.save_model_steps))
    if eval_steps[0] != 0 or eval_steps[-1] != args.steps:
        raise ValueError("eval_steps must start at 0 and end exactly at --steps")
    if any(step <= 0 or step > args.steps for step in save_steps):
        raise ValueError("save_model_steps must be in [1, steps]")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    candidate_path = args.candidate_groups.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(int(args.train_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.train_seed))

    with candidate_path.open(encoding="utf-8") as handle:
        candidate_payload = json.load(handle)
    if not isinstance(candidate_payload, list):
        raise ValueError("candidate_groups must contain a JSON list")
    selected = select_candidate_groups(
        candidate_payload,
        count=int(args.num_groups),
        max_frames=int(args.max_frames),
        minimum_target_pair_mse=float(args.minimum_target_pair_mse),
    )
    groups = materialize_groups(
        selected,
        manifest_path=manifest_path,
        noise_seed=int(args.noise_seed),
    )

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    config = checkpoint["config"]
    parent_step = int(checkpoint["next_global_step"])
    if parent_step != 400_000:
        raise RuntimeError(
            f"Tiny-overfit estimand requires the shared raw 400K parent, got {parent_step}"
        )
    if config.get("contract", {}).get("name") != "hy273_multitask_r13_unified273_v1":
        raise RuntimeError("Tiny-overfit estimand requires the R13 unified-273 contract")
    if (
        config.get("flow", {}).get("contact_protocol")
        != "unified_273_clean_flow_v1"
    ):
        raise RuntimeError("Tiny-overfit estimand requires unified 273 contact flow")
    loss_weights: HY273MultitaskLossWeights = validate_frozen_contract(config)
    normalizer = normalizer_from_checkpoint(checkpoint, device)
    model = create_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = model.to(device).train()

    target_diagnostics = materialized_target_diagnostics(
        groups,
        normalizer,
        minimum_continuous_pair_mse=float(args.minimum_target_pair_mse),
    )
    text_diagnostics = model_visible_text_diagnostics(model, groups, device=device)

    group_defs, _ = optimizer_groups(model, parent_step)
    for group in group_defs:
        name = str(group["group_name"])
        group["lr"] = (
            float(args.base_lr) if name == "G0_existing" else float(args.context_lr)
        )
        group["weight_decay"] = (
            0.0 if name == "G2_context_bias" else float(args.weight_decay)
        )
    optimizer = torch.optim.AdamW(group_defs, betas=(0.9, 0.999), eps=1e-8)
    edit_weights = UnifiedEditLossWeights(
        target_x0_scale=0.05,
        hard_x0_scale=0.02,
        hard_fraction=0.20,
        instruction_rank_scale=0.50,
        instruction_relative_margin=0.10,
    )

    run_manifest = {
        "format": "hy273_edit_instruction_tiny_overfit_run_v1",
        "args": _jsonable_args(args),
        "checkpoint": str(checkpoint_path),
        "parent_step": parent_step,
        "initial_weight_source": "model",
        "manifest": str(manifest_path),
        "candidate_groups": str(candidate_path),
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "selected_groups": [group.__dict__ for group in selected],
        "materialized_target_diagnostics": target_diagnostics,
        "model_visible_text_diagnostics": text_diagnostics,
        "decision_scope": {
            "success_supports": (
                "existing architecture and conditioning path can memorize the "
                "fixed exact-t=0 paired instruction mapping"
            ),
            "failure_supports": (
                "this fixed 2K optimization protocol did not reach memorization; "
                "it does not by itself prove architectural incapacity"
            ),
            "gate": {
                "fidelity_fraction": MEMORIZATION_FIDELITY_FRACTION,
                "contact_fidelity_fraction": (
                    MEMORIZATION_CONTACT_FIDELITY_FRACTION
                ),
            },
        },
        "loss": {
            "primary": "existing_r13_unified273_velocity_mse_plus_kinematic_terms",
            "edit_auxiliary": edit_weights.__dict__,
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, ensure_ascii=False)

    evaluations: list[dict[str, Any]] = []
    metrics_path = output_dir / "train_metrics.jsonl"
    eval_path = output_dir / "evaluations.json"

    def run_evaluation(step: int) -> None:
        result = evaluate(
            model,
            groups,
            normalizer,
            timestep=float(args.timestep),
            groups_per_batch=int(args.eval_groups_per_batch),
            device=device,
            precision=str(args.precision),
        )
        row = {"step": int(step), **result}
        evaluations.append(row)
        with eval_path.open("w", encoding="utf-8") as handle:
            json.dump(evaluations, handle, indent=2, ensure_ascii=False)
        aggregate = result["aggregate"]
        full = aggregate["spaces"]["full_273"]
        print(
            "[eval] "
            f"step={step} full_correct={full['correct_mse']:.6f} "
            f"full_swapped={full['swapped_mse']:.6f} "
            f"fidelity={full['fidelity_ratio_vs_midpoint']:.4f} "
            f"memorized={aggregate['memorized_groups']}/{aggregate['total_groups']} "
            f"pass={aggregate['memorized']}",
            flush=True,
        )
        model.train()

    run_evaluation(0)
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        step_started = time.perf_counter()
        indices = _batch_group_indices(
            step,
            num_groups=len(groups),
            groups_per_batch=int(args.groups_per_batch),
        )
        batch = collate_groups(groups, indices, device=device)
        x0_norm = normalizer.normalize(batch["target"])
        t = torch.full(
            (x0_norm.shape[0],),
            float(args.timestep),
            device=device,
            dtype=x0_norm.dtype,
        )
        observed = torch.zeros_like(x0_norm)
        hard_mask = torch.zeros_like(x0_norm, dtype=torch.bool)
        state = build_unified_273_flow_state(
            x0_norm,
            observed,
            hard_mask,
            t,
            noise=batch["noise"],
        )
        repeated_condition = repeat_condition_batch(batch["condition"], 2)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, str(args.precision)):
            paired_prediction = model(
                torch.cat([state["model_in"], state["model_in"]], dim=0),
                t=torch.cat([t, t], dim=0),
                c_dir=torch.cat(
                    [batch["condition"].frame_gauge_dir] * 2, dim=0
                ),
                text=[*batch["texts"], *batch["swapped_texts"]],
                length_mask=torch.cat([batch["valid"]] * 2, dim=0),
                x_self_cond=None,
                text_drop_prob=0.0,
                condition=repeated_condition,
            )
            prediction, swapped_prediction = paired_prediction.chunk(2, dim=0)
            primary = compute_hy273_unified_flow_loss(
                x0_hat_norm=prediction,
                z_imputed=state["z_imp"],
                x0_target_norm=x0_norm,
                x0_target_physical=batch["target"],
                hard_observed_norm=observed,
                hard_mask=hard_mask,
                target_valid=batch["valid"],
                timesteps=t,
                normalizer=normalizer,
                global_step=parent_step + step,
                weights=loss_weights,
                representation_loss_space="velocity_mse",
                contact_loss_space="velocity_mse",
            )
            auxiliary = compute_unified_edit_loss(
                correct_x0_hat_cont=prediction[..., :CONT_DIM],
                shuffled_x0_hat_cont=swapped_prediction[..., :CONT_DIM],
                x0_target_norm=x0_norm,
                target_valid=batch["valid"],
                hard_mask=hard_mask,
                weights=edit_weights,
            )
            loss = primary.total + auxiliary.total
        if not bool(torch.isfinite(loss.detach())):
            raise RuntimeError(f"Non-finite tiny-overfit loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(args.gradient_clip)
        )
        optimizer.step()

        if step == 1 or step % int(args.log_every) == 0:
            metric = {
                "step": step,
                "loss": float(loss.detach().item()),
                "primary": float(primary.total.detach().item()),
                "edit_auxiliary": float(auxiliary.total.detach().item()),
                "instruction_rank_raw": float(
                    auxiliary.instruction_rank_raw.detach().item()
                ),
                "instruction_gap": float(auxiliary.instruction_gap.detach().item()),
                "instruction_rank_active_fraction": float(
                    auxiliary.instruction_rank_active_fraction.detach().item()
                ),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().item()),
                "step_seconds": time.perf_counter() - step_started,
                "elapsed_seconds": time.perf_counter() - started,
                "group_indices": list(indices),
                "max_memory_gib": (
                    float(torch.cuda.max_memory_allocated(device) / 2**30)
                    if device.type == "cuda"
                    else 0.0
                ),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
            print(
                "[train] "
                f"step={step}/{args.steps} loss={metric['loss']:.6f} "
                f"primary={metric['primary']:.6f} "
                f"rank={metric['instruction_rank_raw']:.6f} "
                f"gap={metric['instruction_gap']:.6f} "
                f"grad={metric['grad_norm']:.4f} "
                f"sec={metric['step_seconds']:.3f} "
                f"mem={metric['max_memory_gib']:.1f}GiB",
                flush=True,
            )
        if step in eval_steps:
            run_evaluation(step)
        if step in save_steps:
            snapshot = output_dir / "model" / f"tiny_overfit_step_{step:06d}.pt"
            save_model_snapshot(
                snapshot,
                model=model,
                parent_checkpoint=checkpoint_path,
                parent_step=parent_step,
                train_steps=step,
                config=config,
                args=args,
                selected_groups=selected,
            )
            print(f"[checkpoint] {snapshot}", flush=True)

    final = {
        "format": "hy273_edit_instruction_tiny_overfit_result_v1",
        "run_manifest": run_manifest,
        "evaluations": evaluations,
        "protocol_steps": int(args.steps),
        "memorized": bool(evaluations[-1]["aggregate"]["memorized"]),
        "decision": (
            "capacity_demonstrated_at_fixed_exact_t0_mapping"
            if evaluations[-1]["aggregate"]["memorized"]
            else "not_memorized_within_this_fixed_optimization_protocol"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (output_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, ensure_ascii=False)
    print(f"[done] {output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
