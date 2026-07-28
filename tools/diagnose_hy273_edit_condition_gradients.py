#!/usr/bin/env python3
"""Measure Edit target-loss gradients at the actual text/source injection tensors."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.flow_schedule import build_unified_273_flow_state
from models.raw_motion.hy273_slices import CONT_DIM, DIM_HY273
from sample_hy273_multitask import make_edit_condition, normalizer_from_checkpoint
from tools.diagnose_hy273_r13_edit_fixed_t import load_rows, pad_motion
from train_hy273_multitask import create_model_from_checkpoint


DEFAULT_PAIR_IDS = (
    "000038",
    "000472",
    "002143",
    "002173",
    "002024",
    "003165",
    "003485",
    "003896",
)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(",") if token.strip())


def tensor_stats(value: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    detached = value.detach().float()
    if valid is not None:
        valid = valid.to(device=detached.device, dtype=torch.bool)
        while valid.ndim < detached.ndim:
            valid = valid.unsqueeze(-1)
        detached = detached[valid.expand_as(detached)]
    if detached.numel() == 0:
        return {"rms": 0.0, "abs_mean": 0.0, "l2": 0.0, "elements": 0}
    return {
        "rms": float(detached.square().mean().sqrt().item()),
        "abs_mean": float(detached.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(detached).item()),
        "elements": int(detached.numel()),
    }


def gradient_stats(
    value: torch.Tensor, valid: torch.Tensor | None = None
) -> dict[str, float]:
    if value.grad is None:
        raise RuntimeError("Injection tensor did not receive a gradient")
    return tensor_stats(value.grad, valid)


def diagnose_checkpoint(
    checkpoint_path: Path,
    *,
    weight_source: str,
    rows: list[dict[str, Any]],
    timesteps: tuple[float, ...],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    model = create_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint[weight_source], strict=True)
    normalizer = normalizer_from_checkpoint(checkpoint, device)
    checkpoint_step = int(checkpoint["next_global_step"])
    del checkpoint
    model = model.to(device).eval()
    model.requires_grad_(False)

    source, target, source_lengths, target_lengths = pad_motion(rows, phi=0.0)
    source = source.to(device)
    target = target.to(device)
    source_lengths = source_lengths.to(device)
    target_lengths = target_lengths.to(device)
    frames = int(target.shape[1])
    valid = torch.arange(frames, device=device)[None] < target_lengths[:, None]
    gauge = torch.zeros(len(rows), 2, device=device)
    gauge[:, 0] = 1.0
    condition = make_edit_condition(
        source,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        target_frames=frames,
        frame_gauge_dir=gauge,
    )
    texts = [str(row["texts"][0]["value"]) for row in rows]
    target_norm = normalizer.normalize(target)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    noise = torch.randn(
        target_norm.shape,
        device=device,
        dtype=target_norm.dtype,
        generator=generator,
    )

    captures: dict[str, torch.Tensor] = {}

    def tensor_leaf_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor):
            leaf = output.detach().requires_grad_(True)
            captures[name] = leaf
            return leaf

        return hook

    def source_leaf_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any):
        root = output.root.detach().requires_grad_(True)
        body = output.body.detach().requires_grad_(True)
        captures["source_root"] = root
        captures["source_body"] = body
        return replace(output, root=root, body=body)

    hooks = (
        model.text_encoder.token_proj.register_forward_hook(
            tensor_leaf_hook("text_tokens")
        ),
        model.text_encoder.pooled_proj.register_forward_hook(
            tensor_leaf_hook("text_pooled")
        ),
        model.source_context.register_forward_hook(source_leaf_hook),
    )
    records: list[dict[str, Any]] = []
    try:
        for timestep in timesteps:
            captures.clear()
            t = torch.full(
                (len(rows),), float(timestep), device=device, dtype=target_norm.dtype
            )
            state = build_unified_273_flow_state(
                target_norm,
                torch.zeros_like(target_norm),
                torch.zeros_like(target_norm, dtype=torch.bool),
                t,
                noise=noise,
            )
            prediction = model(
                state["model_in"],
                t=t,
                c_dir=gauge,
                text=texts,
                length_mask=valid,
                x_self_cond=None,
                text_drop_prob=0.0,
                condition=condition,
            )
            error = (prediction[..., :CONT_DIM] - target_norm[..., :CONT_DIM]).square()
            error_mask = valid[..., None].expand_as(error)
            clean_mse = error[error_mask].mean()
            clean_mse.backward()

            token_grad = captures["text_tokens"].grad
            if token_grad is None:
                raise RuntimeError("Text tokens did not receive gradients")
            token_valid = token_grad.detach().abs().sum(dim=-1) > 0
            injection_valid = {
                "text_tokens": token_valid,
                "text_pooled": None,
                "source_root": valid,
                "source_body": valid,
            }
            injection_rows = {}
            for name, value in captures.items():
                mask = injection_valid[name]
                injection_rows[name] = {
                    "activation": tensor_stats(value, mask),
                    "gradient": gradient_stats(value, mask),
                }
            records.append(
                {
                    "t": float(timestep),
                    "clean_continuous_mse": float(clean_mse.detach().item()),
                    "velocity_loss_scale": float(
                        1.0 / max(1.0 - float(timestep), 0.05) ** 2
                    ),
                    "injections": injection_rows,
                }
            )
    finally:
        for hook in hooks:
            hook.remove()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weight_source": weight_source,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--manifest",
        default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
        "hy273_multitask_v1/test.jsonl",
    )
    parser.add_argument("--pair_ids", default=",".join(DEFAULT_PAIR_IDS))
    parser.add_argument("--timesteps", default="0.05,0.1,0.6")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--weight_source", choices=("ema", "model"), default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pair_ids = parse_csv(args.pair_ids)
    timesteps = tuple(float(value) for value in parse_csv(args.timesteps))
    rows = load_rows(Path(args.manifest), pair_ids)
    device = torch.device(args.device)
    models = [
        diagnose_checkpoint(
            Path(value).expanduser().resolve(),
            weight_source=str(args.weight_source),
            rows=rows,
            timesteps=timesteps,
            seed=int(args.seed),
            device=device,
        )
        for value in args.checkpoint
    ]
    payload = {
        "format": "hy273_edit_condition_gradient_probe_v1",
        "pair_ids": list(pair_ids),
        "timesteps": list(timesteps),
        "seed": int(args.seed),
        "loss_for_gradient": "clean_continuous_normalized_mse",
        "models": models,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "models": len(models)}, indent=2))


if __name__ == "__main__":
    main()
