#!/usr/bin/env python
"""Matched-noise T2M panel for text paraphrase routing."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.hy273_multitask_condition import (
    ABSOLUTE_TEXT_PROFILE,
    CapabilityId,
    make_absent_condition,
)
from models.raw_motion.hy273_slices import DIM_HY273, reconstruct_global_joints_from_features
from sample_hy273_multitask import (
    create_model_from_checkpoint,
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from tools.hy273_runtime_text_encoding import (
    RuntimeTextRows,
    encode_missing_text_rows,
    register_runtime_text_rows,
)


PROMPTS = (
    ("canonical", "a person is break dancing."),
    ("paraphrase_gerund", "a person is breaking dance."),
    ("paraphrase_verb", "a person breakdances."),
    ("empty", ""),
    ("counterfactual_walk", "a person is walking slowly."),
)


def _encode_missing_texts(
    text_encoder,
    texts: list[str],
    device: torch.device,
) -> RuntimeTextRows:
    context_cache = getattr(text_encoder, "context_cache", None)
    return encode_missing_text_rows(
        text_encoder.cache,
        texts,
        [ABSOLUTE_TEXT_PROFILE] * len(texts),
        device,
        context_cache=context_cache,
    )


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--checkpoint must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--checkpoint must use non-empty LABEL=PATH")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"Checkpoint does not exist: {resolved}")
    return label.strip(), resolved


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return seeds


def _motion_distances(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_joints = reconstruct_global_joints_from_features(left.float())
    right_joints = reconstruct_global_joints_from_features(right.float())
    full = torch.linalg.vector_norm(left_joints - right_joints, dim=-1).mean()
    root = torch.linalg.vector_norm(
        left_joints[:, 0] - right_joints[:, 0], dim=-1
    ).mean()
    left_local = left_joints - left_joints[:, :1]
    right_local = right_joints - right_joints[:, :1]
    local = torch.linalg.vector_norm(left_local - right_local, dim=-1).mean()
    feature_rmse = (left.float() - right.float()).square().mean().sqrt()
    return {
        "joint_mpjpe_m": float(full.item()),
        "root_path_l2_m": float(root.item()),
        "root_relative_mpjpe_m": float(local.item()),
        "feature_rmse": float(feature_rmse.item()),
    }


def _seed_metrics(samples: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    pairs = {
        "canonical_vs_empty": ("canonical", "empty"),
        "canonical_vs_counterfactual": ("canonical", "counterfactual_walk"),
        "gerund_vs_canonical": ("paraphrase_gerund", "canonical"),
        "gerund_vs_empty": ("paraphrase_gerund", "empty"),
        "verb_vs_canonical": ("paraphrase_verb", "canonical"),
        "verb_vs_empty": ("paraphrase_verb", "empty"),
    }
    result = {
        name: _motion_distances(samples[left], samples[right])
        for name, (left, right) in pairs.items()
    }
    for variant, prefix in (
        ("paraphrase_gerund", "gerund"),
        ("paraphrase_verb", "verb"),
    ):
        to_canonical = result[f"{prefix}_vs_canonical"]
        to_empty = result[f"{prefix}_vs_empty"]
        result[f"{prefix}_routing"] = {
            metric: float(to_empty[metric] - to_canonical[metric])
            for metric in to_canonical
        }
    return result


def _aggregate(seed_rows: list[dict[str, dict[str, float]]]) -> dict:
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for comparison in seed_rows[0]:
        aggregate[comparison] = {}
        for metric in seed_rows[0][comparison]:
            values = np.asarray(
                [row[comparison][metric] for row in seed_rows], dtype=np.float64
            )
            aggregate[comparison][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_checkpoint,
        required=True,
        help="Repeat LABEL=PATH for each model.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weight_source", choices=("ema", "raw"), default="ema")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--target_length", type=int, default=150)
    parser.add_argument("--seeds", type=_parse_seeds, default=(3407, 12345, 20260725))
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    length = int(args.target_length)
    condition = make_absent_condition(
        batch_size=1,
        target_frames=length,
        target_lengths=torch.tensor([length]),
        capability=CapabilityId.T2M,
    )
    observed = torch.zeros(1, length, DIM_HY273)
    mask = torch.zeros_like(observed, dtype=torch.bool)
    all_results: dict[str, dict] = {}
    runtime_rows_by_contract: dict[tuple[str, ...], RuntimeTextRows] = {}

    for checkpoint_label, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=False
        )
        model = create_model_from_checkpoint(checkpoint).to(device)
        weight_key = "ema" if args.weight_source == "ema" else "model"
        model.load_state_dict(checkpoint[weight_key], strict=True)
        model.eval()
        normalizer = normalizer_from_checkpoint(checkpoint, device)
        text_cache = model.text_encoder.cache
        context_cache = getattr(model.text_encoder, "context_cache", None)
        context_manifest = (
            {} if context_cache is None else context_cache.manifest
        )
        runtime_contract = (
            str(text_cache.manifest.get("format", "")),
            str(text_cache.manifest.get("encoder_identity", "")),
            str(text_cache.manifest.get("prompt_template_version", "")),
            str(context_manifest.get("format", "")),
            str(context_manifest.get("encoder_identity", "")),
            str(context_manifest.get("prompt_template_version", "")),
        )
        if runtime_contract not in runtime_rows_by_contract:
            runtime_rows_by_contract[runtime_contract] = _encode_missing_texts(
                model.text_encoder,
                [text for _, text in PROMPTS],
                device,
            )
        runtime_rows = runtime_rows_by_contract[runtime_contract]
        if runtime_rows.count:
            register_runtime_text_rows(
                text_cache,
                runtime_rows,
                context_cache=context_cache,
            )

        ordered_samples: list[np.ndarray] = []
        ordered_texts: list[str] = []
        ordered_labels: list[str] = []
        ordered_seeds: list[int] = []
        seed_rows: list[dict[str, dict[str, float]]] = []
        protocol = None
        with torch.inference_mode():
            for seed in args.seeds:
                samples_by_prompt: dict[str, torch.Tensor] = {}
                for prompt_label, text in PROMPTS:
                    generator = torch.Generator(device=device).manual_seed(int(seed))
                    sampled = sample_hy273_multitask_ode(
                        model,
                        normalizer,
                        condition,
                        [text],
                        observed,
                        mask,
                        num_steps=int(args.num_steps),
                        text_cfg_scale=float(args.cfg_scale),
                        contact_init="random",
                        contact_feedback="blend",
                        cfg_apply_contacts=(
                            None if normalizer.normalize_contacts else False
                        ),
                        generator=generator,
                    )
                    motion = sampled.raw_motion[0].detach().cpu().float()
                    if not torch.isfinite(motion).all():
                        raise RuntimeError(
                            f"Non-finite sample for {checkpoint_label}/{seed}/{prompt_label}"
                        )
                    samples_by_prompt[prompt_label] = motion
                    ordered_samples.append(motion.numpy())
                    ordered_texts.append(text)
                    ordered_labels.append(prompt_label)
                    ordered_seeds.append(int(seed))
                    protocol = sampled.protocol
                seed_rows.append(_seed_metrics(samples_by_prompt))

        model_dir = output_root / checkpoint_label
        model_dir.mkdir(parents=True, exist_ok=True)
        samples_array = np.stack(ordered_samples)
        np.save(model_dir / "samples_raw.npy", samples_array)
        np.save(model_dir / "observed.npy", np.zeros_like(samples_array))
        np.save(model_dir / "mask.npy", np.zeros_like(samples_array, dtype=bool))
        np.save(
            model_dir / "lengths.npy",
            np.full((len(samples_array),), length, dtype=np.int64),
        )
        metadata = {
            **(protocol or {}),
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint.get("next_global_step", -1)),
            "weight_source": args.weight_source,
            "texts": ordered_texts,
            "prompt_labels": ordered_labels,
            "seeds": ordered_seeds,
            "matched_noise_per_seed": True,
        }
        (model_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": int(checkpoint.get("next_global_step", -1)),
            "per_seed": [
                {"seed": int(seed), "comparisons": row}
                for seed, row in zip(args.seeds, seed_rows)
            ],
            "aggregate": _aggregate(seed_rows),
        }
        (model_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        all_results[checkpoint_label] = result
        del model, normalizer, checkpoint
        torch.cuda.empty_cache()

    summary = {
        "protocol": {
            "prompts": dict(PROMPTS),
            "seeds": list(args.seeds),
            "target_length": length,
            "num_steps": int(args.num_steps),
            "text_cfg_scale": float(args.cfg_scale),
            "weight_source": args.weight_source,
            "matched_noise_per_seed": True,
        },
        "models": all_results,
    }
    summary_path = output_root / "comparison.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(summary_path), "models": list(all_results)}))


if __name__ == "__main__":
    main()
