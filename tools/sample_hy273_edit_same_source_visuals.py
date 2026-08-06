#!/usr/bin/env python3
"""Save matched-noise Edit ODE branches for selected same-source pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_slices import CONT_DIM
from sample_hy273_multitask import (
    create_model_from_checkpoint,
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from tools.eval_hy273_edit_same_source_fixed_t import (
    DEFAULT_MANIFEST,
    load_same_source_rows,
    materialize_groups,
)
from tools.overfit_hy273_edit_instruction_pairs import collate_groups


DEFAULT_GROUPS = (
    "003742,003746",
    "006470,006474",
    "006783,006787",
    "002924,002927",
    "003779,004587",
)


def parse_group(value: str) -> tuple[str, str]:
    pair_ids = tuple(token.strip() for token in value.split(",") if token.strip())
    if len(pair_ids) != 2 or pair_ids[0] == pair_ids[1]:
        raise argparse.ArgumentTypeError("group must be PAIR_A,PAIR_B")
    return tuple(sorted(pair_ids))  # type: ignore[return-value]


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    block: slice,
) -> list[float]:
    rows: list[float] = []
    for index, length_value in enumerate(lengths):
        length = int(length_value.item())
        rows.append(
            float(
                (
                    prediction[index, :length, block].float()
                    - target[index, :length, block].float()
                )
                .square()
                .mean()
                .item()
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--weight_source", choices=("model", "ema"), default="model")
    parser.add_argument("--group", action="append", type=parse_group, default=[])
    parser.add_argument("--minimum_target_pair_mse", type=float, default=0.1)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--noise_seed", type=int, default=20260722)
    parser.add_argument("--ode_steps", type=int, default=32)
    parser.add_argument("--source_cfg_scale", type=float, default=1.0)
    parser.add_argument("--edit_cfg_scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    requested = args.group or [parse_group(value) for value in DEFAULT_GROUPS]
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate --group values")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    normalizer = normalizer_from_checkpoint(checkpoint, device)
    if not bool(normalizer.normalize_contacts):
        raise RuntimeError("This visual protocol requires unified 273D flow")
    groups = materialize_groups(
        load_same_source_rows(args.manifest.expanduser().resolve()),
        normalizer,
        minimum_target_pair_mse=float(args.minimum_target_pair_mse),
        max_frames=int(args.max_frames),
        noise_seed=int(args.noise_seed),
    )
    by_pair = {
        tuple(sorted(group.candidate.pair_ids)): group for group in groups
    }
    missing = [pair for pair in requested if pair not in by_pair]
    if missing:
        raise RuntimeError(f"Requested groups are absent after materialization: {missing}")
    selected = [by_pair[pair] for pair in requested]
    batch = collate_groups(selected, range(len(selected)), device=device)

    model = create_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[str(args.weight_source)], strict=True)
    model = model.to(device).eval()
    observed = torch.zeros_like(batch["target"])
    hard_mask = torch.zeros_like(observed, dtype=torch.bool)
    branch_texts = {
        "correct": batch["texts"],
        "sibling": batch["swapped_texts"],
        "empty": [""] * len(batch["texts"]),
    }
    outputs: dict[str, torch.Tensor] = {}
    protocols: dict[str, Any] = {}
    with torch.inference_mode():
        for branch, texts in branch_texts.items():
            sampled = sample_hy273_multitask_ode(
                model,
                normalizer,
                batch["condition"],
                texts,
                observed,
                hard_mask,
                num_steps=int(args.ode_steps),
                source_cfg_scale=float(args.source_cfg_scale),
                edit_cfg_scale=float(args.edit_cfg_scale),
                initial_unified_noise=batch["noise"],
            )
            outputs[branch] = sampled.raw_motion.float().cpu()
            protocols[branch] = sampled.protocol

    source = batch["source"].float().cpu()
    target = batch["target"].float().cpu()
    lengths = batch["target_lengths"].cpu()
    source_lengths = batch["source_lengths"].cpu()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "source.npy", source.numpy())
    np.save(output_dir / "target.npy", target.numpy())
    np.save(output_dir / "lengths.npy", lengths.numpy())
    np.save(output_dir / "source_lengths.npy", source_lengths.numpy())
    for branch, values in outputs.items():
        np.save(output_dir / f"{branch}.npy", values.numpy())

    target_norm = normalizer.normalize(target.to(device)).cpu()
    source_norm = normalizer.normalize(source.to(device)).cpu()
    output_norm = {
        name: normalizer.normalize(value.to(device)).cpu()
        for name, value in outputs.items()
    }
    target_mse = {
        name: masked_mse(value, target_norm, lengths, slice(0, CONT_DIM))
        for name, value in output_norm.items()
    }
    source_mse = {
        name: masked_mse(value, source_norm, lengths, slice(0, CONT_DIM))
        for name, value in output_norm.items()
    }
    rows = []
    for row_index, pair_id in enumerate(batch["pair_ids"]):
        rows.append(
            {
                "row_index": row_index,
                "group_index": row_index // 2,
                "pair_id": str(pair_id),
                "instruction": str(batch["texts"][row_index]),
                "sibling_instruction": str(batch["swapped_texts"][row_index]),
                "frames": int(lengths[row_index].item()),
                "continuous_target_mse": {
                    name: values[row_index] for name, values in target_mse.items()
                },
                "continuous_source_mse": {
                    name: values[row_index] for name, values in source_mse.items()
                },
            }
        )
    metadata = {
        "format": "hy273_r13_same_source_visual_samples_v1",
        "label": str(args.label),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["next_global_step"]),
        "weight_source": str(args.weight_source),
        "manifest": str(args.manifest.expanduser().resolve()),
        "noise_seed": int(args.noise_seed),
        "ode_steps": int(args.ode_steps),
        "source_cfg_scale": float(args.source_cfg_scale),
        "edit_cfg_scale": float(args.edit_cfg_scale),
        "groups": [list(pair) for pair in requested],
        "rows": rows,
        "protocols": protocols,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}))


if __name__ == "__main__":
    main()
