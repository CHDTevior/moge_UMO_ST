#!/usr/bin/env python3
"""Probe where T2M text sensitivity is lost under matched motion/noise inputs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_multitask_manifest_dataset import _transform_to_gauge
from models.raw_motion.flow_schedule import build_unified_273_flow_state
from models.raw_motion.hy273_multitask_condition import (
    ABSOLUTE_TEXT_PROFILE,
    CapabilityId,
    make_absent_condition,
)
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_SLICE,
    VELOCITY_SLICE,
)
from models.raw_motion.hytext_cache import LLM2VEC_CACHE_FORMAT
from models.raw_motion.text_condition import RawTextCondition
from sample_hy273_multitask import normalizer_from_checkpoint
from tools.eval_hy273_text_paraphrase_panel import _encode_missing_texts
from train_hy273_multitask import create_model_from_checkpoint


PROMPTS = (
    ("canonical", "a person is break dancing."),
    ("gerund", "a person is breaking dance."),
    ("verb", "a person breakdances."),
    ("empty", ""),
    ("counterfactual", "a person is walking slowly."),
)

GROUPS = {
    "root_xyz": ROOT_SLICE,
    "heading": HEADING_SLICE,
    "joint_position": JOINT_POS_SLICE,
    "global_rot6d": GLOBAL_ROT_SLICE,
    "joint_velocity": VELOCITY_SLICE,
    "contact": CONTACT_SLICE,
}


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--checkpoint must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"Invalid checkpoint {value!r}")
    return label, path


def masked_rms(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    selected = value.masked_select(mask.expand_as(value))
    return selected.square().mean().sqrt()


def per_row_temporal_centered_rms(
    value: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Measure frame-to-frame variation without mixing batch or feature axes."""
    if value.ndim != 3 or valid.shape != value.shape[:2]:
        raise ValueError(
            "Expected value [B,T,D] and matching valid [B,T], got "
            f"{tuple(value.shape)} and {tuple(valid.shape)}"
        )
    value = value.float()
    mask = valid.to(device=value.device, dtype=value.dtype).unsqueeze(-1)
    frame_count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    temporal_mean = (value * mask).sum(dim=1, keepdim=True) / frame_count
    centered = (value - temporal_mean) * mask
    denominator = (
        valid.sum(dim=1).to(device=value.device, dtype=value.dtype)
        * value.shape[-1]
    ).clamp_min(1.0)
    return (centered.square().sum(dim=(1, 2)) / denominator).sqrt()


def per_row_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    feature_slice: slice,
) -> torch.Tensor:
    error = (prediction[..., feature_slice] - target[..., feature_slice]).square()
    mask = valid[..., None].expand_as(error)
    return (error * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1)


def cosine_matrix(values: torch.Tensor) -> list[list[float]]:
    normalized = F.normalize(values.float(), dim=-1)
    return (normalized @ normalized.transpose(0, 1)).cpu().tolist()


def text_geometry(cache, model, device: torch.device) -> dict[str, Any]:
    texts = [text for _, text in PROMPTS]
    profiles = [ABSOLUTE_TEXT_PROFILE] * len(texts)
    vtxt, ctxt, lengths = cache.lookup_rows(texts, profiles=profiles)
    max_tokens = int(model.text_encoder.max_text_tokens)
    ctxt = ctxt[:, :max_tokens].float()
    lengths = lengths.clamp(min=1, max=max_tokens)
    valid = torch.arange(max_tokens)[None, :] < lengths[:, None]
    raw_token_mean = (ctxt * valid[..., None]).sum(dim=1) / lengths[:, None]
    with torch.inference_mode():
        condition = model.text_encoder(
            texts,
            profiles=profiles,
            device=device,
            dtype=torch.float32,
        )
    projected_token_mean = (
        condition.tokens.float()
        * (~condition.padding_mask).to(condition.tokens.dtype)[..., None]
    ).sum(dim=1) / lengths.to(device)[:, None]
    result = {
        "cache_format": str(cache.manifest.get("format", "")),
        "labels": [label for label, _ in PROMPTS],
        "lengths": lengths.tolist(),
    }
    if cache.manifest.get("format") == LLM2VEC_CACHE_FORMAT:
        result.update(
            {
                "raw_llm2vec_sentence_cosine": cosine_matrix(raw_token_mean),
                "projected_llm2vec_sentence_cosine": cosine_matrix(
                    projected_token_mean.cpu()
                ),
                "pooled_global_conditioning": False,
            }
        )
    else:
        result.update(
            {
                "raw_qwen_mean_cosine": cosine_matrix(raw_token_mean),
                "raw_clip_pooled_cosine": cosine_matrix(vtxt[:, 0].float()),
                "projected_qwen_mean_cosine": cosine_matrix(
                    projected_token_mean.cpu()
                ),
                "projected_clip_pooled_cosine": cosine_matrix(
                    condition.pooled.cpu()
                ),
                "pooled_global_conditioning": True,
            }
        )
    return result


def load_target(path: Path, max_frames: int) -> torch.Tensor:
    motion = torch.from_numpy(np.load(path).astype(np.float32))
    motion, _ = _transform_to_gauge(motion, 0.0)
    return motion[:max_frames]


def diagnose_checkpoint(
    label: str,
    path: Path,
    *,
    runtime_rows: tuple[Any, ...],
    target: torch.Tensor,
    timesteps: tuple[float, ...],
    seed: int,
    weight_source: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    model = create_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[weight_source], strict=True)
    normalizer = normalizer_from_checkpoint(checkpoint, device)
    checkpoint_step = int(checkpoint["next_global_step"])
    del checkpoint

    runtime_texts, runtime_vtxt, runtime_ctxt, runtime_lengths = runtime_rows
    if runtime_texts:
        model.text_encoder.cache.add_runtime_rows(
            runtime_texts,
            runtime_vtxt,
            runtime_ctxt,
            runtime_lengths,
            profiles=[ABSOLUTE_TEXT_PROFILE] * len(runtime_texts),
        )
    model = model.to(device).eval()
    model.requires_grad_(False)
    geometry = text_geometry(model.text_encoder.cache, model, device)

    labels = [label for label, _ in PROMPTS]
    texts = [text for _, text in PROMPTS]
    batch_size = len(texts)
    frames = int(target.shape[0])
    target = target.to(device)
    target_norm = normalizer.normalize(target[None]).expand(batch_size, -1, -1).clone()
    valid = torch.ones(batch_size, frames, dtype=torch.bool, device=device)
    condition = make_absent_condition(
        batch_size=batch_size,
        target_frames=frames,
        target_lengths=torch.full((batch_size,), frames),
        capability=CapabilityId.T2M,
    ).to(device)
    gauge = torch.zeros(batch_size, 2, device=device)
    gauge[:, 0] = 1.0
    generator = torch.Generator(device=device).manual_seed(seed)
    shared_noise = torch.randn(
        (1, frames, target_norm.shape[-1]),
        generator=generator,
        device=device,
        dtype=target_norm.dtype,
    ).expand_as(target_norm)

    captures: dict[str, torch.Tensor] = {}

    def leaf_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor):
            leaf = output.detach().requires_grad_(True)
            captures[name] = leaf
            return leaf

        return hook

    def observe_hook(name: str):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            captures[name] = output.detach()

        return hook

    hooks = [
        model.text_encoder.token_proj.register_forward_hook(
            leaf_hook("text_tokens")
        ),
        model.root_input_proj.register_forward_hook(
            observe_hook("root_input_projection")
        ),
        model.root_backbone.register_forward_hook(
            observe_hook("root_backbone_output")
        ),
        model.root_output_proj.register_forward_hook(
            observe_hook("root_prediction")
        ),
        model.body_input_proj.register_forward_hook(
            observe_hook("body_input_projection")
        ),
        model.body_backbone.register_forward_hook(
            observe_hook("body_backbone_output")
        ),
        model.body_output_proj.register_forward_hook(
            observe_hook("body_prediction")
        ),
    ]
    for backbone_name, backbone in (
        ("root", model.root_backbone),
        ("body", model.body_backbone),
    ):
        encoder = getattr(backbone, "encoder", None)
        layers = getattr(encoder, "layers", ())
        for layer_index, layer in enumerate(layers):
            hooks.append(
                layer.register_forward_hook(
                    observe_hook(
                        f"{backbone_name}_encoder_layer_{layer_index:02d}"
                    )
                )
            )
    pooled_proj = getattr(model.text_encoder, "pooled_proj", None)
    if pooled_proj is not None:
        hooks.append(
            pooled_proj.register_forward_hook(leaf_hook("text_pooled"))
        )
    records: list[dict[str, Any]] = []
    try:
        for timestep in timesteps:
            captures.clear()
            t = torch.full(
                (batch_size,), timestep, device=device, dtype=target_norm.dtype
            )
            state = build_unified_273_flow_state(
                target_norm,
                torch.zeros_like(target_norm),
                torch.zeros_like(target_norm, dtype=torch.bool),
                t,
                noise=shared_noise,
            )
            model_in = state["model_in"].detach().requires_grad_(True)
            prediction = model(
                model_in,
                t=t,
                c_dir=gauge,
                text=texts,
                length_mask=valid,
                text_drop_prob=0.0,
                condition=condition,
            )
            all_mse = per_row_mse(prediction, target_norm, valid, slice(0, 273))
            continuous_mse = per_row_mse(
                prediction, target_norm, valid, slice(0, 269)
            )
            temporal_variation: dict[str, torch.Tensor] = {}
            for activation_name, activation in captures.items():
                if activation_name.startswith("text_"):
                    continue
                motion_activation = activation
                if motion_activation.shape[1] != frames:
                    if motion_activation.shape[1] < frames:
                        raise RuntimeError(
                            f"Activation {activation_name} is shorter than the "
                            f"motion: {tuple(motion_activation.shape)}"
                        )
                    motion_activation = motion_activation[:, -frames:]
                temporal_variation[activation_name] = (
                    per_row_temporal_centered_rms(
                        motion_activation,
                        valid,
                    )
                )
            temporal_variation["final_prediction"] = (
                per_row_temporal_centered_rms(prediction, valid)
            )
            all_mse.sum().backward()

            empty_index = labels.index("empty")
            canonical_index = labels.index("canonical")
            prompt_rows = {}
            for row_index, prompt_label in enumerate(labels):
                group_mse = {
                    name: float(
                        per_row_mse(prediction, target_norm, valid, feature_slice)[
                            row_index
                        ].item()
                    )
                    for name, feature_slice in GROUPS.items()
                }
                prompt_rows[prompt_label] = {
                    "all_mse": float(all_mse[row_index].item()),
                    "continuous_mse": float(continuous_mse[row_index].item()),
                    "empty_minus_prompt_mse": float(
                        all_mse[empty_index].item() - all_mse[row_index].item()
                    ),
                    "prediction_vs_empty_rms": float(
                        masked_rms(
                            prediction[row_index] - prediction[empty_index],
                            valid[row_index],
                        ).item()
                    ),
                    "prediction_vs_canonical_rms": float(
                        masked_rms(
                            prediction[row_index] - prediction[canonical_index],
                            valid[row_index],
                        ).item()
                    ),
                    "group_mse": group_mse,
                    "activation_temporal_centered_rms": {
                        name: float(values[row_index].item())
                        for name, values in temporal_variation.items()
                    },
                    "gradient_rms": {
                        "text_tokens": float(
                            captures["text_tokens"].grad[row_index]
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                        "text_pooled": (
                            float(
                                captures["text_pooled"].grad[row_index]
                                .float()
                                .square()
                                .mean()
                                .sqrt()
                                .item()
                            )
                            if "text_pooled" in captures
                            and captures["text_pooled"].grad is not None
                            else 0.0
                        ),
                        "model_input": float(
                            model_in.grad[row_index]
                            .float()
                            .square()
                            .mean()
                            .sqrt()
                            .item()
                        ),
                    },
                }
            records.append({"t": timestep, "prompts": prompt_rows})
            model.zero_grad(set_to_none=True)
    finally:
        for hook in hooks:
            hook.remove()

    component_ablation: dict[str, list[dict[str, Any]]] = {}
    has_pooled_condition = pooled_proj is not None
    branch_labels = (
        ("full", "tokens_only", "pooled_only", "empty")
        if has_pooled_condition
        else ("full", "empty")
    )
    original_profiled_text_condition = model._profiled_text_condition
    try:
        for prompt_label, prompt_text in PROMPTS:
            if prompt_label == "empty":
                continue
            with torch.inference_mode():
                encoded = model.text_encoder(
                    [prompt_text, ""],
                    profiles=[ABSOLUTE_TEXT_PROFILE, ABSOLUTE_TEXT_PROFILE],
                    device=device,
                    dtype=torch.float32,
                )
            correct_tokens = encoded.tokens[0]
            empty_tokens = encoded.tokens[1]
            correct_pooled = encoded.pooled[0]
            empty_pooled = encoded.pooled[1]
            correct_padding = encoded.padding_mask[0]
            empty_padding = encoded.padding_mask[1]
            if has_pooled_condition:
                injected = RawTextCondition(
                    tokens=torch.stack(
                        [
                            correct_tokens,
                            correct_tokens,
                            empty_tokens,
                            empty_tokens,
                        ]
                    ),
                    pooled=torch.stack(
                        [
                            correct_pooled,
                            empty_pooled,
                            correct_pooled,
                            empty_pooled,
                        ]
                    ),
                    padding_mask=torch.stack(
                        [
                            correct_padding,
                            correct_padding,
                            empty_padding,
                            empty_padding,
                        ]
                    ),
                )
            else:
                injected = RawTextCondition(
                    tokens=torch.stack([correct_tokens, empty_tokens]),
                    pooled=torch.stack([correct_pooled, empty_pooled]),
                    padding_mask=torch.stack(
                        [correct_padding, empty_padding]
                    ),
                )

            def injected_condition(
                _self,
                _texts,
                _profiles,
                batch_size,
                _device,
                _dtype,
                _text_drop_prob,
                _force_drop_text,
            ):
                if batch_size != len(branch_labels):
                    raise RuntimeError("Unexpected component-ablation batch size")
                return injected

            model._profiled_text_condition = types.MethodType(
                injected_condition, model
            )
            ablation_target = target_norm[:1].expand(len(branch_labels), -1, -1)
            ablation_valid = valid[:1].expand(len(branch_labels), -1)
            ablation_noise = shared_noise[:1].expand_as(ablation_target)
            ablation_gauge = gauge[:1].expand(len(branch_labels), -1)
            ablation_condition = make_absent_condition(
                batch_size=len(branch_labels),
                target_frames=frames,
                target_lengths=torch.full((len(branch_labels),), frames),
                capability=CapabilityId.T2M,
            ).to(device)
            prompt_records = []
            with torch.inference_mode():
                for timestep in timesteps:
                    t = torch.full(
                        (len(branch_labels),),
                        timestep,
                        device=device,
                        dtype=ablation_target.dtype,
                    )
                    state = build_unified_273_flow_state(
                        ablation_target,
                        torch.zeros_like(ablation_target),
                        torch.zeros_like(ablation_target, dtype=torch.bool),
                        t,
                        noise=ablation_noise,
                    )
                    prediction = model(
                        state["model_in"],
                        t=t,
                        c_dir=ablation_gauge,
                        text=[prompt_text] * len(branch_labels),
                        length_mask=ablation_valid,
                        text_drop_prob=0.0,
                        condition=ablation_condition,
                    )
                    losses = per_row_mse(
                        prediction,
                        ablation_target,
                        ablation_valid,
                        slice(0, 273),
                    )
                    empty_prediction = prediction[branch_labels.index("empty")]
                    branches = {}
                    for index, branch in enumerate(branch_labels):
                        branches[branch] = {
                            "all_mse": float(losses[index].item()),
                            "empty_minus_branch_mse": float(
                                losses[-1].item() - losses[index].item()
                            ),
                            "prediction_vs_empty_rms": float(
                                masked_rms(
                                    prediction[index] - empty_prediction,
                                    ablation_valid[index],
                                ).item()
                            ),
                        }
                    prompt_records.append({"t": timestep, "branches": branches})
            component_ablation[prompt_label] = prompt_records
    finally:
        model._profiled_text_condition = original_profiled_text_condition
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "label": label,
        "checkpoint": str(path),
        "checkpoint_step": checkpoint_step,
        "weight_source": weight_source,
        "text_geometry": geometry,
        "fixed_t": records,
        "component_ablation": component_ablation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target_motion",
        default="/mnt/afs/mogo_base/datasets/HumanML3D/"
        "kimodo273_from_hy201_smplx22/motion_data/001288.npy",
    )
    parser.add_argument("--max_frames", type=int, default=193)
    parser.add_argument(
        "--timesteps", default="0.02,0.05,0.1,0.2,0.5,0.8"
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--weight_source", choices=("ema", "model"), default="ema"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    timesteps = tuple(float(value) for value in args.timesteps.split(","))
    target = load_target(Path(args.target_motion), int(args.max_frames))
    runtime_rows_by_contract: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    models = []
    for label, path in args.checkpoint:
        checkpoint = torch.load(
            path, map_location="cpu", mmap=True, weights_only=False
        )
        probe_model = create_model_from_checkpoint(checkpoint)
        cache = probe_model.text_encoder.cache
        contract = (
            str(cache.manifest.get("format", "")),
            str(cache.manifest.get("encoder_identity", "")),
            str(cache.manifest.get("prompt_template_version", "")),
        )
        if contract not in runtime_rows_by_contract:
            runtime_rows_by_contract[contract] = _encode_missing_texts(
                cache,
                [text for _, text in PROMPTS],
                device,
            )
        del probe_model, checkpoint
        gc.collect()
        models.append(
            diagnose_checkpoint(
            label,
            path,
            runtime_rows=runtime_rows_by_contract[contract],
            target=target,
            timesteps=timesteps,
            seed=int(args.seed),
            weight_source=str(args.weight_source),
            device=device,
            )
        )
    payload = {
        "format": "hy273_t2m_text_path_probe_v2",
        "prompts": dict(PROMPTS),
        "target_motion": str(Path(args.target_motion).resolve()),
        "target_frames": int(target.shape[0]),
        "timesteps": list(timesteps),
        "seed": int(args.seed),
        "models": models,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "models": len(models)}, indent=2))


if __name__ == "__main__":
    main()
