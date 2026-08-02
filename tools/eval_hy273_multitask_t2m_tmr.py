#!/usr/bin/env python3
"""Evaluate unified HY273 checkpoints with the native HY273 TMR benchmark.

This keeps the evaluator data gauge and retrieval implementation from
``/mnt/afs/unified_kimodo/eval_hy273_t2m_tmr.py`` while adapting only the
generator loading and T2M sampling path to ``hy273_unified_actor_checkpoint_v1``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import current-project modules before loading the reference evaluator module.
# The reference file imports modules with the same top-level names.
import sample_hy273_raw  # noqa: F401,E402
import train_hy273_raw_flow  # noqa: F401,E402
import utils.metrics  # noqa: F401,E402
from models.raw_motion.hy273_multitask_condition import (  # noqa: E402
    CapabilityId,
    make_absent_condition,
)
from sample_hy273_multitask import (  # noqa: E402
    UNIFIED_ACTOR_CHECKPOINT_FORMAT,
    create_model_from_checkpoint,
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from utils.metrics import (  # noqa: E402
    calculate_activation_statistics,
    calculate_frechet_distance,
)

# The reference evaluator imports this legacy generator-only validator at module
# import time. It is not used by this adapter, whose checkpoint contract is
# validated below.
if not hasattr(sample_hy273_raw, "validate_unified_contact_flow_checkpoint_args"):
    sample_hy273_raw.validate_unified_contact_flow_checkpoint_args = (  # type: ignore[attr-defined]
        lambda *_args, **_kwargs: None
    )


REFERENCE_EVALUATOR = Path("/mnt/afs/unified_kimodo/eval_hy273_t2m_tmr.py")
DEFAULT_EVALUATOR_CHECKPOINT = Path(
    "/mnt/afs/unified_kimodo/checkpoints/evaluators/hy273_tmr/"
    "hy273_tmr_kimodo_gauge_b32_l256_l6_4gpu_20260714_235737/model/"
    "best-r03-024.ckpt"
)
DEFAULT_EVALUATOR_DATA = Path(
    "/mnt/afs/unified_kimodo/evaluator_data/humanml3d_kimodo273_tmr"
)
DEFAULT_TEXT_EMBEDDINGS = Path(
    "/mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272/"
    "tmr_evaluator/tmr_text_embeddings/"
    "token_distilbert-base-uncased_sent_sentence-transformers_all-mpnet-base-v2_t96_s96"
)


def load_source_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def distributed_context() -> tuple[torch.device, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return (
            torch.device("cuda", local_rank),
            dist.get_rank(),
            dist.get_world_size(),
            local_rank,
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Native HY273 TMR generation evaluation requires CUDA")
    return torch.device("cuda", 0), 0, 1, 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--evaluator_checkpoint", type=Path, default=DEFAULT_EVALUATOR_CHECKPOINT
    )
    parser.add_argument("--evaluator_data_root", type=Path, default=DEFAULT_EVALUATOR_DATA)
    parser.add_argument("--text_embedding_dir", type=Path, default=DEFAULT_TEXT_EMBEDDINGS)
    parser.add_argument("--umo_root", type=Path, default=Path("/mnt/afs/mogeflow-umo"))
    parser.add_argument("--reference_evaluator", type=Path, default=REFERENCE_EVALUATOR)
    parser.add_argument("--weight_source", choices=("ema", "model"), default="ema")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--text_cfg_scale", type=float, default=3.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--retrieval_pool_size", type=int, default=32)
    parser.add_argument("--threshold_selfsim", type=float, default=0.95)
    parser.add_argument("--min_motion_length", type=int, default=40)
    parser.add_argument("--max_motion_length", type=int, default=300)
    parser.add_argument("--expected_samples", type=int, default=1332)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def validate_checkpoint(checkpoint: dict[str, Any], path: Path) -> dict[str, Any]:
    if checkpoint.get("format") != UNIFIED_ACTOR_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"{path} is not a {UNIFIED_ACTOR_CHECKPOINT_FORMAT} checkpoint"
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{path} has no embedded config")
    schedule = config.get("schedule", {}).get("segments", [])
    next_step = int(checkpoint.get("next_global_step", -1))
    matching = [
        row
        for row in schedule
        if int(row.get("start", -1)) <= max(0, next_step - 1) < int(row.get("end", -1))
    ]
    if not matching:
        raise RuntimeError(f"Checkpoint step {next_step} is outside its schedule")
    return config


def load_generator_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> list[str]:
    """Load archived actor checkpoints across the Reaction role-table extension."""

    state = dict(state)
    migrations: list[str] = []
    key = "source_context.role_embed.weight"
    target = model.state_dict()
    if key in state and key in target and state[key].shape != target[key].shape:
        old = state[key]
        new = target[key]
        if (
            old.ndim == 2
            and new.ndim == 2
            and old.shape[1] == new.shape[1]
            and old.shape[0] == 4
            and new.shape[0] == 6
        ):
            expanded = torch.zeros_like(new)
            expanded[: old.shape[0]].copy_(old)
            state[key] = expanded
            migrations.append("source_role_embedding_4_to_6_zero_extend")
    model.load_state_dict(state, strict=True)
    return migrations


def main() -> None:
    args = build_parser().parse_args()
    device, rank, world, local_rank = distributed_context()
    try:
        rank_seed = args.seed + rank * 100003
        random.seed(rank_seed)
        np.random.seed(rank_seed % (2**32 - 1))
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed_all(rank_seed)
        torch.set_float32_matmul_precision("high")

        checkpoint_path = args.checkpoint.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        evaluator_path = args.evaluator_checkpoint.expanduser().resolve()
        data_root = args.evaluator_data_root.expanduser().resolve()
        text_root = args.text_embedding_dir.expanduser().resolve()
        umo_root = args.umo_root.expanduser().resolve()
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier(device_ids=[local_rank])

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", mmap=True, weights_only=False
        )
        config = validate_checkpoint(checkpoint, checkpoint_path)
        model = create_model_from_checkpoint(checkpoint)
        state = checkpoint.get(args.weight_source)
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint has no {args.weight_source!r} state")
        checkpoint_migrations = load_generator_state(model, state)
        model.to(device).eval()
        normalizer = normalizer_from_checkpoint(checkpoint, device)
        checkpoint_step = int(checkpoint.get("next_global_step", 0))
        run_name = str(checkpoint.get("run_name", checkpoint_path.parent.parent.name))
        del state

        reference = load_source_module(
            f"_hy273_native_reference_rank{rank}",
            args.reference_evaluator.expanduser().resolve(),
        )
        tmr_module = load_source_module(
            f"_hy273_native_tmr_rank{rank}", umo_root / "models/vimogen_tmr_modules.py"
        )
        text_module = load_source_module(
            f"_hy273_native_text_rank{rank}",
            umo_root / "tools/vimogen_tmr_text_embeddings.py",
        )
        evaluator_checkpoint = torch.load(
            evaluator_path, map_location="cpu", mmap=True, weights_only=False
        )
        hparams = dict(evaluator_checkpoint.get("hyper_parameters", {}))
        if int(hparams.get("motion_dim", -1)) != 273 or int(
            hparams.get("strip_last_dims", -1)
        ) != 4:
            raise RuntimeError(f"Not a native HY273 evaluator: {hparams}")
        evaluator = tmr_module.ViMoGenTMR(**hparams)
        evaluator.load_state_dict(evaluator_checkpoint["state_dict"], strict=True)
        evaluator.to(device).eval()
        text_store = text_module.PrecomputedTMRTextEmbeddings(text_root)
        del evaluator_checkpoint

        dataset = reference.HY273T2MEvalDataset(
            data_root,
            text_store,
            min_motion_length=args.min_motion_length,
            max_motion_length=args.max_motion_length,
            max_samples=args.max_samples,
        )
        if args.max_samples <= 0 and len(dataset) != args.expected_samples:
            raise RuntimeError(
                f"HY273 evaluator sample count changed: {len(dataset)} "
                f"!= {args.expected_samples}"
            )
        indices = list(range(rank, len(dataset), world))
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=reference.collate_eval,
        )
        evaluator_mean = torch.from_numpy(dataset.mean).to(device).view(1, 1, 273)
        evaluator_std = torch.from_numpy(dataset.std).to(device).view(1, 1, 273)

        record_indices: list[np.ndarray] = []
        generated_latents: list[np.ndarray] = []
        ground_truth_latents: list[np.ndarray] = []
        text_latents: list[np.ndarray] = []
        sentence_embeddings: list[np.ndarray] = []
        sample_ids: list[str] = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                batch_seed = args.seed + rank * 100003 + batch_index
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)
                target_raw = batch["raw_motion"].to(device=device, dtype=torch.float32)
                lengths = batch["lengths"].to(device=device)
                condition = make_absent_condition(
                    batch_size=int(target_raw.shape[0]),
                    target_frames=int(target_raw.shape[1]),
                    target_lengths=lengths,
                    device=device,
                    capability=CapabilityId.T2M,
                )
                condition = replace(
                    condition,
                    frame_gauge_dir=batch["c_dir"].to(device=device, dtype=torch.float32),
                )
                condition.validate()
                observed = torch.zeros_like(target_raw)
                hard_mask = torch.zeros_like(target_raw, dtype=torch.bool)
                sampled = sample_hy273_multitask_ode(
                    model,
                    normalizer,
                    condition,
                    list(batch["texts"]),
                    observed,
                    hard_mask,
                    num_steps=args.num_steps,
                    text_cfg_scale=args.text_cfg_scale,
                )
                generated_eval = (
                    sampled.raw_motion.float() - evaluator_mean
                ) / evaluator_std
                ground_truth_eval = batch["evaluator_motion"].to(
                    device=device, dtype=torch.float32
                )
                mask = batch["mask"].to(device=device, dtype=torch.bool)
                text_x_dict = {
                    key: value.to(device=device)
                    for key, value in batch["text_x_dict"].items()
                }
                t_latent, _ = evaluator._encode_text_x_dict(
                    text_x_dict, sample_mean=True
                )
                gt_latent, _ = evaluator._encode_motion_x_dict(
                    {"x": ground_truth_eval[..., :269], "mask": mask, "length": lengths},
                    sample_mean=True,
                )
                gen_latent, _ = evaluator._encode_motion_x_dict(
                    {"x": generated_eval[..., :269], "mask": mask, "length": lengths},
                    sample_mean=True,
                )
                record_indices.append(batch["record_indices"].numpy())
                generated_latents.append(gen_latent.float().cpu().numpy())
                ground_truth_latents.append(gt_latent.float().cpu().numpy())
                text_latents.append(t_latent.float().cpu().numpy())
                sentence_embeddings.append(batch["sent_emb"].float().numpy())
                sample_ids.extend(batch["sample_ids"])

        local_payload = {
            "indices": np.concatenate(record_indices, axis=0),
            "generated": np.concatenate(generated_latents, axis=0),
            "ground_truth": np.concatenate(ground_truth_latents, axis=0),
            "text": np.concatenate(text_latents, axis=0),
            "sent": np.concatenate(sentence_embeddings, axis=0),
            "sample_ids": sample_ids,
        }
        gathered: list[dict[str, Any] | None] | None = (
            [None] * world if rank == 0 else None
        )
        if world > 1:
            dist.gather_object(local_payload, gathered, dst=0)
        else:
            gathered = [local_payload]

        if rank == 0:
            assert gathered is not None and all(item is not None for item in gathered)
            items = [item for item in gathered if item is not None]
            order = np.argsort(np.concatenate([item["indices"] for item in items]))
            merged = {
                key: np.concatenate([item[key] for item in items], axis=0)[order]
                for key in ("generated", "ground_truth", "text", "sent")
            }
            merged_indices = np.concatenate([item["indices"] for item in items])[order]
            if not np.array_equal(merged_indices, np.arange(len(dataset))):
                raise RuntimeError("Distributed evaluator records are incomplete")

            generated_mu, generated_cov = calculate_activation_statistics(
                merged["generated"]
            )
            ground_truth_mu, ground_truth_cov = calculate_activation_statistics(
                merged["ground_truth"]
            )
            fid = float(
                calculate_frechet_distance(
                    ground_truth_mu, ground_truth_cov, generated_mu, generated_cov
                )
            )
            generated_pool = reference.pooled_retrieval(
                merged["text"],
                merged["generated"],
                merged["sent"],
                pool_size=args.retrieval_pool_size,
                threshold_selfsim=args.threshold_selfsim,
            )
            ground_truth_pool = reference.pooled_retrieval(
                merged["text"],
                merged["ground_truth"],
                merged["sent"],
                pool_size=args.retrieval_pool_size,
                threshold_selfsim=args.threshold_selfsim,
            )
            generated_global = reference.global_retrieval(
                merged["text"], merged["generated"], merged["sent"], args.threshold_selfsim
            )
            ground_truth_global = reference.global_retrieval(
                merged["text"],
                merged["ground_truth"],
                merged["sent"],
                args.threshold_selfsim,
            )
            paired_cosine = float(
                np.mean(
                    np.sum(
                        reference.normalize_rows(merged["text"])
                        * reference.normalize_rows(merged["generated"]),
                        axis=1,
                    )
                )
            )
            payload = {
                "protocol": "hy273_multitask_native_tmr_generation_v1",
                "checkpoint": str(checkpoint_path),
                "checkpoint_step": checkpoint_step,
                "run_name": run_name,
                "weight_source": args.weight_source,
                "checkpoint_migrations": checkpoint_migrations,
                "model_text_config": config.get("model", {}),
                "text_encoder_config": config.get("text", {}),
                "evaluator_checkpoint": str(evaluator_path),
                "evaluator_data_root": str(data_root),
                "samples": int(len(dataset)),
                "sampling": {
                    "num_steps": int(args.num_steps),
                    "text_cfg_scale": float(args.text_cfg_scale),
                    "seed": int(args.seed),
                    "world_size": int(world),
                    "batch_size_per_rank": int(args.batch_size),
                },
                "metrics": {
                    "fid": max(0.0, fid),
                    "paired_text_motion_cosine": paired_cosine,
                    "generated_pool32": generated_pool,
                    "ground_truth_pool32": ground_truth_pool,
                    "generated_global": generated_global,
                    "ground_truth_global": ground_truth_global,
                },
            }
            if not math.isfinite(fid) or not math.isfinite(paired_cosine):
                raise RuntimeError(f"Non-finite evaluator metrics: {payload['metrics']}")
            atomic_write_json(output_dir / "summary.json", payload)
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if world > 1:
            dist.barrier(device_ids=[local_rank])
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
