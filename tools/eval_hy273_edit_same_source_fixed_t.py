#!/usr/bin/env python3
"""Evaluate instruction selection on held-out exact-same-source MotionFix pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.flow_schedule import build_unified_273_flow_state
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM, DIM_HY273
from sample_hy273_multitask import (
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from tools.diagnose_hy273_r13_edit_fixed_t import load_k273, to_gauge
from tools.overfit_hy273_edit_instruction_pairs import (
    CandidateGroup,
    MaterializedGroup,
    collate_groups,
    paired_assignment_metrics,
)
from train_hy273_multitask import create_model_from_checkpoint, repeat_condition_batch


DEFAULT_MANIFEST = Path(
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/test.jsonl"
)
DEFAULT_TRAIN_MANIFEST = DEFAULT_MANIFEST.with_name("train.jsonl")
SPACES = ("full_273", "continuous_269", "contact_4")
COMPARISON_METRICS = (
    "correct_instruction_mse",
    "instruction_margin",
    "text_effect_rms",
    "assignment_advantage",
    "empty_instruction_mse",
    "correct_vs_empty_mse_gap",
    "correct_vs_empty_effect_rms",
)


def parse_csv(value: str) -> tuple[str, ...]:
    rows = tuple(token.strip() for token in value.split(",") if token.strip())
    if not rows:
        raise ValueError("Expected at least one comma-separated value")
    return rows


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
    else:
        raw_path = value
        label = Path(raw_path).stem
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise ValueError(f"Invalid checkpoint specification: {value!r}")
    return label, path


def parse_system_expectation(value: str) -> tuple[str, int, str]:
    if "=" not in value:
        raise ValueError("System expectation must be LABEL=STEP,TREATMENT")
    label, raw = value.split("=", 1)
    parts = [token.strip() for token in raw.split(",", 1)]
    if not label.strip() or len(parts) != 2:
        raise ValueError("System expectation must be LABEL=STEP,TREATMENT")
    treatment = "" if parts[1].casefold() in {"none", "parent", "null"} else parts[1]
    return label.strip(), int(parts[0]), treatment


def parse_direct_comparison(value: str) -> tuple[str, str]:
    parts = parse_csv(value)
    if len(parts) != 2 or parts[0] == parts[1]:
        raise ValueError("Direct comparison must be LEFT_LABEL,RIGHT_LABEL")
    return parts[0], parts[1]


def instruction(row: dict[str, Any]) -> str:
    texts = row.get("texts")
    if not isinstance(texts, list) or len(texts) != 1:
        raise ValueError(f"Expected one MotionFix instruction for {row.get('uid')}")
    value = str(texts[0].get("value", ""))
    if not value.strip():
        raise ValueError(f"Empty MotionFix instruction for {row.get('uid')}")
    return value


def load_motionfix_rows(manifest: Path) -> list[dict[str, Any]]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") == "motionfix_k273":
                rows.append(row)
    return rows


def load_same_source_rows(manifest: Path) -> list[list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_motionfix_rows(manifest):
        source = row["source_motion"]["k273_asset"]
        target = row["target_motion"]["k273_asset"]
        if int(source["frames"]) != int(target["frames"]):
            continue
        by_source[str(source["sha256"])].append(row)
    return [rows for rows in by_source.values() if len(rows) >= 2]


def build_group_provenance(
    groups: list[MaterializedGroup],
    *,
    test_manifest: Path,
    train_manifest: Path,
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[str, ...]]]]:
    test_by_pair = {
        str(row["pair"]["official_pair_id"]): row
        for row in load_motionfix_rows(test_manifest)
    }
    train_rows = load_motionfix_rows(train_manifest)
    train_pair_ids = {
        str(row["pair"]["official_pair_id"]) for row in train_rows
    }
    train_target_hashes = {
        str(row["target_motion"]["k273_asset"]["sha256"])
        for row in train_rows
    }
    train_motion_hashes = {
        str(row[role]["k273_asset"]["sha256"])
        for row in train_rows
        for role in ("source_motion", "target_motion")
    }

    records = []
    all_pairs: set[tuple[str, ...]] = set()
    target_disjoint: set[tuple[str, ...]] = set()
    motion_disjoint: set[tuple[str, ...]] = set()
    for group in groups:
        pair_ids = tuple(str(value) for value in group.candidate.pair_ids)
        rows = [test_by_pair[pair_id] for pair_id in pair_ids]
        if any(pair_id in train_pair_ids for pair_id in pair_ids):
            raise RuntimeError(f"Train/test MotionFix pair-id overlap: {pair_ids}")
        source_hashes = {
            str(row["source_motion"]["k273_asset"]["sha256"]) for row in rows
        }
        target_hashes = tuple(
            str(row["target_motion"]["k273_asset"]["sha256"]) for row in rows
        )
        if source_hashes != {group.candidate.source_sha256}:
            raise RuntimeError(f"Selected source provenance mismatch: {pair_ids}")
        target_seen_as_target = tuple(
            value in train_target_hashes for value in target_hashes
        )
        target_seen_as_motion = tuple(
            value in train_motion_hashes for value in target_hashes
        )
        source_seen_as_motion = group.candidate.source_sha256 in train_motion_hashes
        pair_key = tuple(sorted(pair_ids))
        all_pairs.add(pair_key)
        if not any(target_seen_as_target):
            target_disjoint.add(pair_key)
        if not source_seen_as_motion and not any(target_seen_as_motion):
            motion_disjoint.add(pair_key)
        records.append(
            {
                "pair_ids": list(pair_ids),
                "source_sha256": group.candidate.source_sha256,
                "target_sha256": list(target_hashes),
                "source_seen_in_train_motion_assets": bool(source_seen_as_motion),
                "target_seen_as_train_target": list(target_seen_as_target),
                "target_seen_in_train_motion_assets": list(target_seen_as_motion),
                "target_disjoint": not any(target_seen_as_target),
                "motion_disjoint": (
                    not source_seen_as_motion and not any(target_seen_as_motion)
                ),
            }
        )
    target_disjoint_asset_nonoverlap: set[tuple[str, ...]] = set()
    used_motion_assets: set[str] = set()
    for record in sorted(records, key=lambda value: tuple(value["pair_ids"])):
        pair_key = tuple(sorted(record["pair_ids"]))
        assets = {record["source_sha256"], *record["target_sha256"]}
        if (
            bool(record["target_disjoint"])
            and not (assets & used_motion_assets)
        ):
            target_disjoint_asset_nonoverlap.add(pair_key)
            used_motion_assets.update(assets)
    if not target_disjoint_asset_nonoverlap:
        raise RuntimeError("No asset-independent target-disjoint groups remain")

    return records, {
        "pair_level_all": all_pairs,
        "target_disjoint": target_disjoint,
        "target_disjoint_asset_nonoverlap": target_disjoint_asset_nonoverlap,
        "motion_disjoint": motion_disjoint,
    }


def materialize_groups(
    row_groups: Iterable[list[dict[str, Any]]],
    normalizer: Any,
    *,
    minimum_target_pair_mse: float,
    max_frames: int,
    noise_seed: int,
) -> list[MaterializedGroup]:
    output: list[MaterializedGroup] = []
    for rows in row_groups:
        candidates: list[tuple[float, dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor]] = []
        for row_a, row_b in combinations(rows, 2):
            target_a = to_gauge(load_k273(row_a["target_motion"]["k273_asset"]), 0.0)
            target_b = to_gauge(load_k273(row_b["target_motion"]["k273_asset"]), 0.0)
            if (
                target_a.shape != target_b.shape
                or target_a.shape[0] > int(max_frames)
                or " ".join(instruction(row_a).split()).casefold()
                == " ".join(instruction(row_b).split()).casefold()
            ):
                continue
            norm_a = normalizer.normalize(target_a.to(normalizer.mean.device)).float()
            norm_b = normalizer.normalize(target_b.to(normalizer.mean.device)).float()
            pair_mse = float(
                (norm_a[..., :CONT_DIM] - norm_b[..., :CONT_DIM])
                .square()
                .mean()
                .item()
            )
            candidates.append((pair_mse, row_a, row_b, target_a, target_b))
        if not candidates:
            continue
        pair_mse, row_a, row_b, target_a, target_b = max(
            candidates, key=lambda value: value[0]
        )
        if pair_mse < float(minimum_target_pair_mse):
            continue
        source_a = to_gauge(load_k273(row_a["source_motion"]["k273_asset"]), 0.0)
        source_b = to_gauge(load_k273(row_b["source_motion"]["k273_asset"]), 0.0)
        if source_a.shape != source_b.shape or not torch.equal(source_a, source_b):
            raise RuntimeError(
                f"Manifest same-source group is not tensor-identical: "
                f"{row_a['uid']} / {row_b['uid']}"
            )
        group_index = len(output)
        generator = torch.Generator(device="cpu").manual_seed(
            int(noise_seed) + 104729 * group_index
        )
        noise = torch.randn(
            target_a.shape, generator=generator, dtype=torch.float32
        )
        pair_ids = (
            str(row_a["pair"]["official_pair_id"]),
            str(row_b["pair"]["official_pair_id"]),
        )
        texts = (instruction(row_a), instruction(row_b))
        candidate = CandidateGroup(
            group_index=group_index,
            source_sha256=str(row_a["source_motion"]["k273_asset"]["sha256"]),
            source_base_motion_id=str(row_a["source_motion"].get("base_motion_id", "")),
            pair_ids=pair_ids,
            texts=texts,
            frames=int(target_a.shape[0]),
            target_pair_mse=pair_mse,
        )
        output.append(
            MaterializedGroup(
                candidate=candidate,
                source=source_a,
                targets=(target_a, target_b),
                texts=texts,
                noise=noise,
            )
        )
    if not output:
        raise RuntimeError("No held-out same-source groups passed the separation filter")
    return output


def _masked_mse_per_row(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    block: slice,
) -> torch.Tensor:
    values = (prediction[..., block].float() - target[..., block].float()).square()
    mask = valid[..., None].expand_as(values)
    return (values * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)


def aggregate_group_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation subset")
    output = {}
    for space in SPACES:
        output[space] = {
            key: float(np.mean([record["spaces"][space][key] for record in records]))
            for key in (
                "correct_instruction_mse",
                "swapped_instruction_mse",
                "instruction_margin",
                "text_effect_rms",
                "correct_assignment",
                "swapped_assignment",
                "assignment_advantage",
                "empty_instruction_mse",
                "correct_vs_empty_mse_gap",
                "correct_vs_empty_effect_rms",
            )
        }
    return output


def evaluate_model(
    model: torch.nn.Module,
    normalizer: Any,
    groups: list[MaterializedGroup],
    *,
    timesteps: tuple[float, ...],
    groups_per_batch: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    by_timestep: dict[str, Any] = {}
    autocast_enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for timestep in timesteps:
        group_records: list[dict[str, Any]] = []
        repeated_input_error = 0.0
        pair_input_error = 0.0
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
            state = build_unified_273_flow_state(
                target_norm,
                torch.zeros_like(target_norm),
                torch.zeros_like(target_norm, dtype=torch.bool),
                t,
                noise=batch["noise"],
            )
            repeated_input_error = max(
                repeated_input_error,
                float(
                    (
                        torch.cat([state["model_in"], state["model_in"]], dim=0)[: target_norm.shape[0]]
                        - torch.cat([state["model_in"], state["model_in"]], dim=0)[target_norm.shape[0] :]
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
            for row in range(0, target_norm.shape[0], 2):
                pair_input_error = max(
                    pair_input_error,
                    float(
                        (state["model_in"][row] - state["model_in"][row + 1])
                        .abs()
                        .max()
                        .item()
                    ),
                )
            condition = repeat_condition_batch(batch["condition"], 3)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                prediction = model(
                    torch.cat([state["model_in"]] * 3, dim=0),
                    t=torch.cat([t] * 3, dim=0),
                    c_dir=torch.cat([batch["condition"].frame_gauge_dir] * 3, dim=0),
                    text=[
                        *batch["texts"],
                        *batch["swapped_texts"],
                        *([""] * len(batch["texts"])),
                    ],
                    length_mask=torch.cat([batch["valid"]] * 3, dim=0),
                    x_self_cond=None,
                    text_drop_prob=0.0,
                    condition=condition,
                )
            correct, swapped, empty = prediction.chunk(3, dim=0)
            correct_rows, _ = paired_assignment_metrics(
                correct,
                target_norm,
                batch["target_lengths"],
                batch["group_ids"],
                batch["pair_ids"],
            )
            swapped_rows, _ = paired_assignment_metrics(
                swapped,
                target_norm,
                batch["target_lengths"],
                batch["group_ids"],
                batch["pair_ids"],
            )
            blocks = {
                "full_273": slice(0, DIM_HY273),
                "continuous_269": slice(0, CONT_DIM),
                "contact_4": CONTACT_SLICE,
            }
            row_metrics = {}
            for space, block in blocks.items():
                correct_mse = _masked_mse_per_row(
                    correct, target_norm, batch["valid"], block
                )
                swapped_mse = _masked_mse_per_row(
                    swapped, target_norm, batch["valid"], block
                )
                text_effect = _masked_mse_per_row(
                    correct, swapped, batch["valid"], block
                ).sqrt()
                empty_mse = _masked_mse_per_row(
                    empty, target_norm, batch["valid"], block
                )
                correct_vs_empty_effect = _masked_mse_per_row(
                    correct, empty, batch["valid"], block
                ).sqrt()
                row_metrics[space] = (
                    correct_mse,
                    swapped_mse,
                    text_effect,
                    empty_mse,
                    correct_vs_empty_effect,
                )
            for local_index, (correct_record, swapped_record) in enumerate(
                zip(correct_rows, swapped_rows)
            ):
                first = 2 * local_index
                record = {
                    "group_id": int(correct_record["group_id"]),
                    "pair_ids": correct_record["pair_ids"],
                    "frames": int(correct_record["frames"]),
                    "spaces": {},
                }
                for space in SPACES:
                    (
                        correct_mse,
                        swapped_mse,
                        text_effect,
                        empty_mse,
                        correct_vs_empty_effect,
                    ) = row_metrics[space]
                    own_correct = float(correct_mse[first : first + 2].mean().item())
                    own_swapped = float(swapped_mse[first : first + 2].mean().item())
                    own_empty = float(empty_mse[first : first + 2].mean().item())
                    record["spaces"][space] = {
                        "correct_instruction_mse": own_correct,
                        "swapped_instruction_mse": own_swapped,
                        "instruction_margin": own_swapped - own_correct,
                        "text_effect_rms": float(
                            text_effect[first : first + 2].mean().item()
                        ),
                        "correct_assignment": float(
                            correct_record["spaces"][space]["assignment_correct"]
                        ),
                        "swapped_assignment": float(
                            swapped_record["spaces"][space]["assignment_correct"]
                        ),
                        "assignment_advantage": float(
                            correct_record["spaces"][space]["assignment_correct"]
                            - swapped_record["spaces"][space]["assignment_correct"]
                        ),
                        "empty_instruction_mse": own_empty,
                        "correct_vs_empty_mse_gap": own_empty - own_correct,
                        "correct_vs_empty_effect_rms": float(
                            correct_vs_empty_effect[first : first + 2].mean().item()
                        ),
                    }
                group_records.append(record)
        if repeated_input_error != 0.0:
            raise RuntimeError(
                f"Correct/swapped branches received different model inputs: "
                f"{repeated_input_error}"
            )
        if float(timestep) == 0.0 and pair_input_error != 0.0:
            raise RuntimeError(
                f"Exact t=0 sibling inputs contain target information: {pair_input_error}"
            )
        by_timestep[str(float(timestep))] = {
            "aggregate": aggregate_group_records(group_records),
            "paired_model_input_max_abs_delta": pair_input_error,
            "correct_vs_swapped_input_max_abs_delta": repeated_input_error,
            "estimand": (
                "text_only_same_source_instruction_selection"
                if float(timestep) == 0.0
                else "target_state_conditioned_text_ablation_supportive_only"
            ),
            "groups": group_records,
        }
    return by_timestep


@torch.inference_mode()
def evaluate_model_ode(
    model: torch.nn.Module,
    normalizer: Any,
    groups: list[MaterializedGroup],
    *,
    groups_per_batch: int,
    num_steps: int,
    source_cfg_scale: float,
    edit_cfg_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    """Run matched-noise Edit ODE rollouts for correct, sibling, and empty text."""

    if not bool(normalizer.normalize_contacts):
        raise RuntimeError("The R13 ODE probe requires unified 273D flow")
    model.eval()
    group_records: list[dict[str, Any]] = []
    for start in range(0, len(groups), groups_per_batch):
        indices = tuple(range(start, min(start + groups_per_batch, len(groups))))
        batch = collate_groups(groups, indices, device=device)
        observed = torch.zeros_like(batch["target"])
        hard_mask = torch.zeros_like(observed, dtype=torch.bool)
        branch_outputs: dict[str, torch.Tensor] = {}
        for branch_name, texts in (
            ("correct", batch["texts"]),
            ("sibling", batch["swapped_texts"]),
            ("empty", [""] * len(batch["texts"])),
        ):
            sampled = sample_hy273_multitask_ode(
                model,
                normalizer,
                batch["condition"],
                texts,
                observed,
                hard_mask,
                num_steps=int(num_steps),
                source_cfg_scale=float(source_cfg_scale),
                edit_cfg_scale=float(edit_cfg_scale),
                initial_unified_noise=batch["noise"],
            )
            branch_outputs[branch_name] = sampled.raw_motion

        target_norm = normalizer.normalize(batch["target"])
        output_norm = {
            name: normalizer.normalize(value)
            for name, value in branch_outputs.items()
        }
        correct_rows, _ = paired_assignment_metrics(
            output_norm["correct"],
            target_norm,
            batch["target_lengths"],
            batch["group_ids"],
            batch["pair_ids"],
        )
        sibling_rows, _ = paired_assignment_metrics(
            output_norm["sibling"],
            target_norm,
            batch["target_lengths"],
            batch["group_ids"],
            batch["pair_ids"],
        )
        blocks = {
            "full_273": slice(0, DIM_HY273),
            "continuous_269": slice(0, CONT_DIM),
            "contact_4": CONTACT_SLICE,
        }
        row_metrics: dict[str, tuple[torch.Tensor, ...]] = {}
        for space, block in blocks.items():
            correct_mse = _masked_mse_per_row(
                output_norm["correct"], target_norm, batch["valid"], block
            )
            sibling_mse = _masked_mse_per_row(
                output_norm["sibling"], target_norm, batch["valid"], block
            )
            empty_mse = _masked_mse_per_row(
                output_norm["empty"], target_norm, batch["valid"], block
            )
            correct_vs_sibling_effect = _masked_mse_per_row(
                output_norm["correct"],
                output_norm["sibling"],
                batch["valid"],
                block,
            ).sqrt()
            correct_vs_empty_effect = _masked_mse_per_row(
                output_norm["correct"],
                output_norm["empty"],
                batch["valid"],
                block,
            ).sqrt()
            row_metrics[space] = (
                correct_mse,
                sibling_mse,
                correct_vs_sibling_effect,
                empty_mse,
                correct_vs_empty_effect,
            )

        for local_index, (correct_record, sibling_record) in enumerate(
            zip(correct_rows, sibling_rows)
        ):
            first = 2 * local_index
            record = {
                "group_id": int(correct_record["group_id"]),
                "pair_ids": correct_record["pair_ids"],
                "frames": int(correct_record["frames"]),
                "spaces": {},
            }
            for space in SPACES:
                (
                    correct_mse,
                    sibling_mse,
                    sibling_effect,
                    empty_mse,
                    empty_effect,
                ) = row_metrics[space]
                own_correct = float(correct_mse[first : first + 2].mean().item())
                own_sibling = float(sibling_mse[first : first + 2].mean().item())
                own_empty = float(empty_mse[first : first + 2].mean().item())
                record["spaces"][space] = {
                    "correct_instruction_mse": own_correct,
                    "swapped_instruction_mse": own_sibling,
                    "instruction_margin": own_sibling - own_correct,
                    "text_effect_rms": float(
                        sibling_effect[first : first + 2].mean().item()
                    ),
                    "correct_assignment": float(
                        correct_record["spaces"][space]["assignment_correct"]
                    ),
                    "swapped_assignment": float(
                        sibling_record["spaces"][space]["assignment_correct"]
                    ),
                    "assignment_advantage": float(
                        correct_record["spaces"][space]["assignment_correct"]
                        - sibling_record["spaces"][space]["assignment_correct"]
                    ),
                    "empty_instruction_mse": own_empty,
                    "correct_vs_empty_mse_gap": own_empty - own_correct,
                    "correct_vs_empty_effect_rms": float(
                        empty_effect[first : first + 2].mean().item()
                    ),
                }
            group_records.append(record)

    return {
        "aggregate": aggregate_group_records(group_records),
        "groups": group_records,
        "protocol": {
            "ode_steps": int(num_steps),
            "source_cfg_scale": float(source_cfg_scale),
            "edit_cfg_scale": float(edit_cfg_scale),
            "initial_noise": "same provided unified 273D tensor across text branches",
            "output": "raw_motion",
            "text_branches": ["correct", "same_source_sibling", "empty"],
        },
    }


def paired_bootstrap_comparisons(
    systems: dict[str, dict[str, Any]],
    *,
    baseline_label: str,
    subsets: dict[str, set[tuple[str, ...]]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    rng = np.random.default_rng(int(seed))
    for label, result in systems.items():
        if label == baseline_label:
            continue
        output[label] = {}
        for subset_name, allowed_pairs in subsets.items():
            if not allowed_pairs:
                continue
            output[label][subset_name] = {}
            for timestep, candidate_step in result["timesteps"].items():
                output[label][subset_name][timestep] = _paired_bootstrap_step(
                    systems[baseline_label]["timesteps"][timestep],
                    candidate_step,
                    allowed_pairs=allowed_pairs,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                    left_name="baseline",
                    right_name="candidate",
                )
            if "ode" in systems[baseline_label] and "ode" in result:
                output[label][subset_name]["ode"] = _paired_bootstrap_step(
                    systems[baseline_label]["ode"],
                    result["ode"],
                    allowed_pairs=allowed_pairs,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                    left_name="baseline",
                    right_name="candidate",
                )
    return output


def _better_direction(metric: str) -> str:
    if metric == "correct_instruction_mse":
        return "negative"
    if metric in {
        "text_effect_rms",
        "correct_vs_empty_effect_rms",
        "empty_instruction_mse",
    }:
        return "diagnostic_two_sided"
    return "positive"


def _paired_bootstrap_step(
    left_step: dict[str, Any],
    right_step: dict[str, Any],
    *,
    allowed_pairs: set[tuple[str, ...]],
    bootstrap_samples: int,
    rng: np.random.Generator,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left_by_pair = {
        tuple(sorted(row["pair_ids"])): row for row in left_step["groups"]
    }
    right_by_pair = {
        tuple(sorted(row["pair_ids"])): row for row in right_step["groups"]
    }
    pair_ids = sorted(allowed_pairs & set(left_by_pair) & set(right_by_pair))
    if pair_ids != sorted(allowed_pairs):
        raise RuntimeError("Comparison systems do not share every requested group")
    output: dict[str, Any] = {}
    for space in SPACES:
        metric_rows = {}
        for metric in COMPARISON_METRICS:
            left_values = np.asarray(
                [left_by_pair[pair]["spaces"][space][metric] for pair in pair_ids],
                dtype=np.float64,
            )
            right_values = np.asarray(
                [right_by_pair[pair]["spaces"][space][metric] for pair in pair_ids],
                dtype=np.float64,
            )
            delta = right_values - left_values
            draws = rng.integers(
                0, len(delta), size=(bootstrap_samples, len(delta))
            )
            boot = delta[draws].mean(axis=1)
            metric_rows[metric] = {
                "groups": len(delta),
                f"{left_name}_mean": float(left_values.mean()),
                f"{right_name}_mean": float(right_values.mean()),
                f"mean_delta_{right_name}_minus_{left_name}": float(delta.mean()),
                "paired_bootstrap_ci95": [
                    float(np.quantile(boot, 0.025)),
                    float(np.quantile(boot, 0.975)),
                ],
                "better_direction_for_delta": _better_direction(metric),
                "better_direction": _better_direction(metric),
            }
        output[space] = metric_rows
    return output


def paired_bootstrap_direct_comparison(
    systems: dict[str, dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
    subsets: dict[str, set[tuple[str, ...]]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Directly bootstrap right-minus-left, rather than comparing marginal CIs."""

    if left_label == right_label or left_label not in systems or right_label not in systems:
        raise ValueError("Direct comparison requires two distinct known systems")
    rng = np.random.default_rng(int(seed))
    output: dict[str, Any] = {
        "left_label": left_label,
        "right_label": right_label,
        "delta_definition": f"{right_label}_minus_{left_label}",
        "subsets": {},
    }
    for subset_name, allowed_pairs in subsets.items():
        if not allowed_pairs:
            continue
        subset_output: dict[str, Any] = {"timesteps": {}}
        for timestep, right_step in systems[right_label]["timesteps"].items():
            subset_output["timesteps"][timestep] = _paired_bootstrap_step(
                systems[left_label]["timesteps"][timestep],
                right_step,
                allowed_pairs=allowed_pairs,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
                left_name=left_label,
                right_name=right_label,
            )
        if "ode" in systems[left_label] and "ode" in systems[right_label]:
            subset_output["ode"] = _paired_bootstrap_step(
                systems[left_label]["ode"],
                systems[right_label]["ode"],
                allowed_pairs=allowed_pairs,
                bootstrap_samples=bootstrap_samples,
                rng=rng,
                left_name=left_label,
                right_name=right_label,
            )
        output["subsets"][subset_name] = subset_output
    return output


def validate_checkpoint_systems(
    checkpoints: list[tuple[str, Path]],
    *,
    expectations: dict[str, tuple[int, str]],
    weight_source: str,
    allow_matched_continuations: bool = False,
) -> list[dict[str, Any]]:
    if len(checkpoints) < 2:
        raise ValueError("Comparison requires at least two systems")
    labels = {label for label, _ in checkpoints}
    if labels != set(expectations):
        raise ValueError("Checkpoint labels and system expectations must match exactly")
    metadata = []
    for label, path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        config = checkpoint.get("config", {})
        runtime = checkpoint.get("runtime_identity", {})
        treatment = runtime.get("research_overrides", {}).get(
            "research_treatment", {}
        )
        expected_step, expected_treatment = expectations[label]
        if int(checkpoint.get("next_global_step", -1)) != int(expected_step):
            raise RuntimeError(f"{label} is not step {expected_step}")
        if config.get("contract", {}).get("name") != "hy273_multitask_r13_unified273_v1":
            raise RuntimeError(f"{label} is not an R13 unified-273 checkpoint")
        if (
            config.get("flow", {}).get("contact_protocol")
            != "unified_273_clean_flow_v1"
        ):
            raise RuntimeError(f"{label} uses the wrong contact-flow protocol")
        if weight_source not in checkpoint:
            raise RuntimeError(f"{label} has no {weight_source} weights")
        actual_treatment = str(treatment.get("name", ""))
        if actual_treatment != expected_treatment:
            raise RuntimeError(
                f"{label} treatment is {actual_treatment!r}, expected "
                f"{expected_treatment!r}"
            )
        metadata.append(
            {
                "label": label,
                "path": str(path),
                "step": int(checkpoint["next_global_step"]),
                "treatment": actual_treatment,
                "parent": runtime.get("immediate_resume_parent", {}).get(
                    "checkpoint"
                ),
            }
        )
        del checkpoint
    trained = [row for row in metadata if row["treatment"]]
    parent_values = {row["parent"] for row in trained}
    if len(trained) > 1 and (None in parent_values or len(parent_values) != 1):
        if not allow_matched_continuations or None in parent_values:
            raise RuntimeError("Controlled trained systems do not share one resume parent")
        common_fork_parents: set[str] = set()
        for row in trained:
            parent_path = Path(str(row["parent"])).expanduser().resolve()
            child_step = int(row["step"])
            lineage_steps: list[int] = []
            seen_paths: set[Path] = set()
            while True:
                if parent_path in seen_paths:
                    raise RuntimeError(
                        f"{row['label']} has a cycle in its resume lineage"
                    )
                seen_paths.add(parent_path)
                if not parent_path.is_file():
                    raise RuntimeError(
                        f"{row['label']} continuation parent is missing: "
                        f"{parent_path}"
                    )
                parent_checkpoint = torch.load(
                    parent_path, map_location="cpu", mmap=True, weights_only=False
                )
                parent_step = int(parent_checkpoint.get("next_global_step", -1))
                if parent_step < 0 or parent_step >= child_step:
                    raise RuntimeError(
                        f"{row['label']} has invalid continuation step "
                        f"{parent_step} -> {child_step}"
                    )
                if (
                    parent_checkpoint.get("config", {})
                    .get("contract", {})
                    .get("name")
                    != "hy273_multitask_r13_unified273_v1"
                ):
                    raise RuntimeError(
                        f"{row['label']} lineage leaves the R13 unified-273 contract"
                    )
                if (
                    parent_checkpoint.get("config", {})
                    .get("flow", {})
                    .get("contact_protocol")
                    != "unified_273_clean_flow_v1"
                ):
                    raise RuntimeError(
                        f"{row['label']} lineage changes the contact-flow protocol"
                    )
                parent_runtime = parent_checkpoint.get("runtime_identity", {})
                parent_treatment = parent_runtime.get(
                    "research_overrides", {}
                ).get("research_treatment", {})
                parent_treatment_name = str(parent_treatment.get("name", ""))
                lineage_steps.append(parent_step)
                next_parent = parent_runtime.get(
                    "immediate_resume_parent", {}
                ).get("checkpoint")
                del parent_checkpoint

                if not parent_treatment_name:
                    common_fork_parents.add(str(parent_path))
                    row["continuation_parent_step"] = lineage_steps[0]
                    row["common_fork_parent"] = str(parent_path)
                    row["lineage_steps"] = lineage_steps
                    break
                if parent_treatment_name != row["treatment"]:
                    raise RuntimeError(
                        f"{row['label']} changed treatment across continuation: "
                        f"{parent_treatment_name!r} -> {row['treatment']!r}"
                    )
                if not next_parent:
                    raise RuntimeError(
                        f"{row['label']} treatment lineage has no untreated fork"
                    )
                child_step = parent_step
                parent_path = Path(str(next_parent)).expanduser().resolve()
        if len(common_fork_parents) != 1:
            raise RuntimeError(
                "Controlled continuations do not share one matched fork lineage"
            )
    parent_systems = [row for row in metadata if not row["treatment"]]
    if parent_systems and trained:
        parent_paths = {str(Path(row["path"]).resolve()) for row in parent_systems}
        for row in trained:
            lineage_parent_paths = {
                str(Path(value).resolve())
                for value in (row.get("parent"), row.get("common_fork_parent"))
                if value
            }
            if not (lineage_parent_paths & parent_paths):
                raise RuntimeError(
                    f"{row['label']} was not forked from one of the compared parent systems"
                )
    return metadata


def validate_checkpoint_pair(
    checkpoints: list[tuple[str, Path]],
    *,
    expected_step: int,
    weight_source: str,
    candidate_treatment: str = "same_source_contrast",
) -> list[dict[str, Any]]:
    if len(checkpoints) != 2:
        raise ValueError("This controlled comparison requires baseline and treatment")
    expectations = {
        checkpoints[0][0]: (int(expected_step), "baseline"),
        checkpoints[1][0]: (int(expected_step), str(candidate_treatment)),
    }
    return validate_checkpoint_systems(
        checkpoints,
        expectations=expectations,
        weight_source=weight_source,
        allow_matched_continuations=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train_manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--weight_source", choices=("model", "ema"), default="model")
    parser.add_argument(
        "--system_expectation",
        action="append",
        default=[],
        help="Expected checkpoint metadata as LABEL=STEP,TREATMENT; use none for parent.",
    )
    parser.add_argument(
        "--allow_matched_continuations",
        action="store_true",
        help=(
            "Allow treatment-preserving continuation checkpoints with distinct "
            "immediate parents when those parents share one matched fork parent."
        ),
    )
    parser.add_argument(
        "--candidate_treatment",
        default="same_source_contrast",
        help="Expected research treatment name stored in the second checkpoint.",
    )
    parser.add_argument("--expected_step", type=int, default=405000)
    parser.add_argument("--timesteps", default="0,0.05,0.1")
    parser.add_argument("--minimum_target_pair_mse", type=float, default=0.1)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--groups_per_batch", type=int, default=4)
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=0,
        help="Run matched-noise Edit ODE when positive; use 32 for the pilot gate.",
    )
    parser.add_argument("--ode_groups_per_batch", type=int, default=1)
    parser.add_argument("--source_cfg_scale", type=float, default=1.0)
    parser.add_argument("--edit_cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--direct_comparison",
        action="append",
        default=[],
        help="Direct paired bootstrap as LEFT_LABEL,RIGHT_LABEL.",
    )
    parser.add_argument("--noise_seed", type=int, default=20260722)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = [parse_checkpoint(value) for value in args.checkpoint]
    if len({label for label, _ in checkpoints}) != len(checkpoints):
        raise ValueError("Checkpoint labels must be unique")
    if args.system_expectation:
        parsed_expectations = [
            parse_system_expectation(value) for value in args.system_expectation
        ]
        if len({label for label, _, _ in parsed_expectations}) != len(
            parsed_expectations
        ):
            raise ValueError("System expectation labels must be unique")
        checkpoint_metadata = validate_checkpoint_systems(
            checkpoints,
            expectations={
                label: (step, treatment)
                for label, step, treatment in parsed_expectations
            },
            weight_source=str(args.weight_source),
            allow_matched_continuations=bool(args.allow_matched_continuations),
        )
    else:
        checkpoint_metadata = validate_checkpoint_pair(
            checkpoints,
            expected_step=int(args.expected_step),
            weight_source=str(args.weight_source),
            candidate_treatment=str(args.candidate_treatment),
        )
    timesteps = tuple(float(value) for value in parse_csv(args.timesteps))
    if any(not 0.0 <= value < 1.0 for value in timesteps):
        raise ValueError("Timesteps must lie in [0,1)")
    if (
        args.bootstrap_samples <= 0
        or args.groups_per_batch <= 0
        or args.ode_groups_per_batch <= 0
        or args.ode_steps < 0
    ):
        raise ValueError("Batch and bootstrap sizes must be positive")
    if 0.0 not in timesteps:
        raise ValueError("The primary exact t=0 endpoint is required")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    first_checkpoint = torch.load(
        checkpoints[0][1], map_location="cpu", mmap=True, weights_only=False
    )
    reference_normalizer = normalizer_from_checkpoint(first_checkpoint, device)
    manifest = args.manifest.expanduser().resolve()
    train_manifest = args.train_manifest.expanduser().resolve()
    row_groups = load_same_source_rows(manifest)
    groups = materialize_groups(
        row_groups,
        reference_normalizer,
        minimum_target_pair_mse=float(args.minimum_target_pair_mse),
        max_frames=int(args.max_frames),
        noise_seed=int(args.noise_seed),
    )
    group_provenance, subsets = build_group_provenance(
        groups,
        test_manifest=manifest,
        train_manifest=train_manifest,
    )
    if not subsets["target_disjoint"]:
        raise RuntimeError("No target-disjoint held-out groups remain")
    del first_checkpoint

    systems: dict[str, dict[str, Any]] = {}
    for label, path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        normalizer = normalizer_from_checkpoint(checkpoint, device)
        torch.testing.assert_close(normalizer.mean, reference_normalizer.mean)
        torch.testing.assert_close(normalizer.std, reference_normalizer.std)
        model = create_model_from_checkpoint(checkpoint)
        model.load_state_dict(checkpoint[args.weight_source], strict=True)
        model = model.to(device).eval()
        systems[label] = {
            "checkpoint": str(path),
            "checkpoint_step": int(checkpoint["next_global_step"]),
            "weight_source": str(args.weight_source),
            "timesteps": evaluate_model(
                model,
                normalizer,
                groups,
                timesteps=timesteps,
                groups_per_batch=int(args.groups_per_batch),
                device=device,
                precision=str(args.precision),
            ),
        }
        if int(args.ode_steps) > 0:
            systems[label]["ode"] = evaluate_model_ode(
                model,
                normalizer,
                groups,
                groups_per_batch=int(args.ode_groups_per_batch),
                num_steps=int(args.ode_steps),
                source_cfg_scale=float(args.source_cfg_scale),
                edit_cfg_scale=float(args.edit_cfg_scale),
                device=device,
            )
        del model, checkpoint, normalizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for system in systems.values():
        for timestep_result in system["timesteps"].values():
            rows_by_pair = {
                tuple(sorted(row["pair_ids"])): row
                for row in timestep_result["groups"]
            }
            timestep_result["subset_aggregates"] = {
                name: aggregate_group_records(
                    [rows_by_pair[pair] for pair in sorted(pair_ids)]
                )
                for name, pair_ids in subsets.items()
                if pair_ids
            }
        if "ode" in system:
            rows_by_pair = {
                tuple(sorted(row["pair_ids"])): row
                for row in system["ode"]["groups"]
            }
            system["ode"]["subset_aggregates"] = {
                name: aggregate_group_records(
                    [rows_by_pair[pair] for pair in sorted(pair_ids)]
                )
                for name, pair_ids in subsets.items()
                if pair_ids
            }

    direct_comparisons = {}
    for index, value in enumerate(args.direct_comparison):
        left_label, right_label = parse_direct_comparison(value)
        key = f"{right_label}_minus_{left_label}"
        if key in direct_comparisons:
            raise ValueError(f"Duplicate direct comparison: {key}")
        direct_comparisons[key] = paired_bootstrap_direct_comparison(
            systems,
            left_label=left_label,
            right_label=right_label,
            subsets=subsets,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.noise_seed) + 10_000 + index,
        )

    result = {
        "format": "hy273_edit_same_source_fixed_t_probe_v1",
        "manifest": str(manifest),
        "train_manifest": str(train_manifest),
        "weight_source": str(args.weight_source),
        "candidate_treatment": str(args.candidate_treatment),
        "primary_endpoint": (
            "t=0 text-only same-source instruction selection and correct-vs-empty gap"
        ),
        "supportive_endpoints": (
            "t>0 target-state-conditioned text ablations; not pure instruction assignment"
        ),
        "ode_endpoint": (
            None
            if int(args.ode_steps) == 0
            else {
                "steps": int(args.ode_steps),
                "source_cfg_scale": float(args.source_cfg_scale),
                "edit_cfg_scale": float(args.edit_cfg_scale),
                "weight_source": str(args.weight_source),
                "matched_noise": True,
            }
        ),
        "checkpoint_system_validation": checkpoint_metadata,
        "minimum_target_pair_mse": float(args.minimum_target_pair_mse),
        "groups": len(groups),
        "evaluation_subsets": {
            name: {
                "groups": len(pair_ids),
                "pair_ids": [list(pair) for pair in sorted(pair_ids)],
            }
            for name, pair_ids in subsets.items()
        },
        "group_provenance": group_provenance,
        "candidate_groups": [
            {
                "pair_ids": list(group.candidate.pair_ids),
                "texts": list(group.texts),
                "frames": group.candidate.frames,
                "source_sha256": group.candidate.source_sha256,
                "target_pair_mse": group.candidate.target_pair_mse,
            }
            for group in groups
        ],
        "systems": systems,
        "comparisons": paired_bootstrap_comparisons(
            systems,
            baseline_label=checkpoints[0][0],
            subsets=subsets,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.noise_seed) + 1,
        ),
        "direct_comparisons": direct_comparisons,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "[subsets] "
        + " ".join(f"{name}={len(pair_ids)}" for name, pair_ids in subsets.items()),
        flush=True,
    )
    for label, system in systems.items():
        for timestep, row in system["timesteps"].items():
            full = row["aggregate"]["full_273"]
            print(
                f"[{label}] t={timestep} groups={len(groups)} "
                f"correct={full['correct_instruction_mse']:.6f} "
                f"swapped={full['swapped_instruction_mse']:.6f} "
                f"margin={full['instruction_margin']:.6f} "
                f"empty_gap={full['correct_vs_empty_mse_gap']:.6f} "
                f"assignment={full['correct_assignment']:.3f}",
                flush=True,
            )
        if "ode" in system:
            full = system["ode"]["aggregate"]["full_273"]
            print(
                f"[{label}] ode{args.ode_steps} "
                f"correct={full['correct_instruction_mse']:.6f} "
                f"sibling={full['swapped_instruction_mse']:.6f} "
                f"empty_gap={full['correct_vs_empty_mse_gap']:.6f} "
                f"assignment={full['correct_assignment']:.3f}",
                flush=True,
            )
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
