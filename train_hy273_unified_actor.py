#!/usr/bin/env python3
"""Research trainer for unified T2M, Edit, and paired-motion HY273 tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

from data.hy273_multitask_scheduler import sample_key_u64
from data.hy273_unified_actor_batcher import HY273UnifiedActorStepBatcher
from models.raw_motion.flow_schedule import sample_timesteps
from models.raw_motion.hy273_interaction_losses import (
    HY273InteractionLossWeights,
    compute_hy273_interaction_loss,
)
from models.raw_motion.hy273_multitask_condition import TaskId, TrainStream
from models.raw_motion.hy273_multitask_losses import (
    HY273MultitaskLossWeights,
    RatioLossTerm,
    compute_hy273_unified_flow_loss,
)
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import CONT_DIM, DIM_HY273
from models.raw_motion.hy273_unified_actor_flow import HY273UnifiedActorFlow
from models.raw_motion.hy273_unified_reaction_flow import HY273UnifiedReactionFlow
from models.raw_motion.hy273_unified_edit_losses import (
    UnifiedEditLossWeights,
    compute_unified_edit_loss,
)


CHECKPOINT_FORMAT = "hy273_unified_actor_checkpoint_v1"
RNG_CONTRACT = "stateless_sample_key_per_scene_v1"
FULLTEXT_STAGE_A_CONTRACT = "fulltext_stage_a"
FULLTEXT_STAGE_B_CONTRACT = "fulltext_stage_b"
FULLTEXT_STAGE_B_CONTINUE_CONTRACT = "fulltext_stage_b_continue"
FULLTEXT_PHASE_CONTRACTS = (
    "",
    FULLTEXT_STAGE_A_CONTRACT,
    FULLTEXT_STAGE_B_CONTRACT,
    FULLTEXT_STAGE_B_CONTINUE_CONTRACT,
)


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> tuple[torch.device, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        return device, dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 1


def cleanup_distributed() -> None:
    if _distributed():
        dist.destroy_process_group()


def cfg(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"Missing config field {dotted}")
        value = value[key]
    return value


def cfg_optional(
    config: Mapping[str, Any],
    dotted: str,
    default: Any,
) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {resolved}")
    return payload, resolved


def validate_config(config: Mapping[str, Any]) -> None:
    paired_task = str(cfg_optional(config, "data.paired_task", "interaction"))
    segments = list(cfg(config, "schedule.segments"))
    paired_key = "reaction" if paired_task == "reaction" else "interaction"
    max_global_step = int(cfg(config, "training.max_global_step"))
    if max_global_step not in {200_000, 250_000, 300_000, 350_000}:
        raise ValueError(
            "Unified actor v1 supports 200K plus 50K same-mix continuations "
            "through 350K"
        )
    expected_segments = [
        {"start": 0, "end": 100000, "t2m": 100, "edit": 0, paired_key: 0},
        {
            "start": 100000,
            "end": max_global_step,
            "t2m": 30,
            "edit": 35,
            paired_key: 35,
        },
    ]
    if segments != expected_segments:
        raise ValueError(
            "Unified actor v1 requires 100K T2M followed by the frozen 30/35/35 mix"
        )
    if str(cfg(config, "flow.prediction_type")) != "x0":
        raise ValueError("Unified actor v1 predicts normalized clean x0")
    if str(cfg(config, "flow.loss_space")) != "velocity_mse":
        raise ValueError("Unified actor v1 keeps the K-Encoder velocity-MSE objective")
    if str(cfg(config, "text.encoder")) != "llm2vec_cache":
        raise ValueError("Unified actor v1 requires the LLM2Vec K-Encoder cache")
    if str(cfg(config, "model.text_global_conditioning")) != "llm2vec_tokens_only":
        raise ValueError("LLM2Vec text must not enter the pooled AdaLN path")
    expected_source_fusion = "token_block" if paired_task == "reaction" else "additive"
    if str(cfg(config, "model.source_fusion_mode")) != expected_source_fusion:
        raise ValueError(
            f"{paired_task} requires source_fusion_mode={expected_source_fusion}"
        )
    if float(cfg(config, "model.dropout")) != 0.0:
        raise ValueError("Stateless resume requires model.dropout=0")
    if float(cfg(config, "training.stage_a_adaptation_lr")) != 0.0:
        raise ValueError("Stage-A capability-extension parameters must remain frozen")
    if float(cfg(config, "training.stage_b_adaptation_lr")) <= 0.0:
        raise ValueError("Stage-B capability-extension parameters need a positive LR")
    if not bool(cfg(config, "normalization.normalize_contacts")):
        raise ValueError("Unified 273D clean flow requires normalized contacts")
    if int(cfg(config, "training.batch_size_t2m_edit_per_rank")) <= 0:
        raise ValueError("Single-actor batch size must be positive")
    if int(
        cfg_optional(
            config,
            "training.batch_size_edit_per_rank",
            cfg(config, "training.batch_size_t2m_edit_per_rank"),
        )
    ) <= 0:
        raise ValueError("Edit batch size must be positive")
    paired_batch_field = (
        "training.batch_size_reaction_per_rank"
        if paired_task == "reaction"
        else "training.batch_size_interaction_per_rank"
    )
    if int(cfg(config, paired_batch_field)) <= 0:
        raise ValueError(f"{paired_task} batch size must be positive")
    local_text = bool(
        cfg_optional(config, "model.local_text_cross_attention", False)
    )
    context_cache = str(
        cfg_optional(config, "text.context_cache_dir", "")
    )
    token_sequence_mode = str(
        cfg_optional(config, "model.text_token_sequence", "sentence")
    )
    if paired_task == "reaction":
        if not context_cache or token_sequence_mode != "sentence_plus_context":
            raise ValueError(
                "Reaction requires sentence_plus_context with a contextual cache"
            )
        if local_text:
            raise ValueError(
                "Reaction full tokens belong to the main MMDiT stream, not side cross-attention"
            )
    elif local_text != bool(context_cache):
        raise ValueError(
            "local_text_cross_attention and text.context_cache_dir must be enabled together"
        )
    if context_cache and not bool(
        cfg_optional(config, "data.interaction_exclude_overlength", False)
    ):
        raise ValueError(
            "The first full-text experiment excludes overlength paired-task "
            "clips to avoid caption/crop semantic mismatch"
        )


def validate_fulltext_phase_contract(
    phase_contract: str,
    *,
    has_resume: bool,
    run_dir_exists: bool,
    declared_stop_step: int,
    global_step: int | None = None,
) -> None:
    """Keep the scratch and multitask curriculum boundaries unambiguous."""

    phase_contract = str(phase_contract)
    if phase_contract not in FULLTEXT_PHASE_CONTRACTS:
        raise ValueError(f"Unknown phase contract: {phase_contract!r}")
    if not phase_contract:
        return
    if phase_contract == FULLTEXT_STAGE_A_CONTRACT:
        if int(declared_stop_step) != 100_000:
            raise ValueError("Full-text Stage A must declare stop_step=100000")
        if has_resume:
            if not run_dir_exists:
                raise FileNotFoundError(
                    "Full-text Stage A resume requires its existing run directory"
                )
            if global_step is not None and not 0 < int(global_step) < 100_000:
                raise ValueError(
                    "Full-text Stage A resume step must be in (0,100000)"
                )
        else:
            if run_dir_exists:
                raise FileExistsError(
                    "Full-text Stage A scratch start requires a fresh run directory"
                )
            if global_step is not None and int(global_step) != 0:
                raise ValueError("Full-text Stage A scratch start requires global_step=0")
        return

    if not has_resume:
        raise ValueError("Full-text Stage B phases require a checkpoint")
    if not run_dir_exists:
        raise FileNotFoundError(
            "Full-text Stage B requires the existing Stage-A run directory"
        )
    if phase_contract == FULLTEXT_STAGE_B_CONTINUE_CONTRACT:
        stop_step = int(declared_stop_step)
        if stop_step not in {250_000, 300_000, 350_000}:
            raise ValueError(
                "Full-text Stage-B continuation must stop at a 50K boundary "
                "in {250000,300000,350000}"
            )
        start_step = stop_step - 50_000
        if global_step is not None and not start_step <= int(global_step) < stop_step:
            raise ValueError(
                "Full-text Stage-B continuation resume step must be in "
                f"[{start_step},{stop_step})"
            )
        return
    if int(declared_stop_step) != 200_000:
        raise ValueError("Full-text Stage B must declare stop_step=200000")
    if global_step is not None and not 100_000 <= int(global_step) < 200_000:
        raise ValueError(
            "Full-text Stage B resume step must be in [100000,200000)"
        )


def create_model(
    config: Mapping[str, Any],
) -> HY273UnifiedActorFlow | HY273UnifiedReactionFlow:
    stats_root = Path(str(cfg(config, "data.stats_root"))).expanduser().resolve()
    paired_task = str(cfg_optional(config, "data.paired_task", "interaction"))
    model_cls = (
        HY273UnifiedReactionFlow
        if paired_task == "reaction"
        else HY273UnifiedActorFlow
    )
    return model_cls(
        hidden_dim=int(cfg(config, "model.hidden_dim")),
        num_heads=int(cfg(config, "model.num_heads")),
        root_depth_double=int(cfg(config, "model.root_depth_double")),
        root_depth_single=int(cfg(config, "model.root_depth_single")),
        body_depth_double=int(cfg(config, "model.body_depth_double")),
        body_depth_single=int(cfg(config, "model.body_depth_single")),
        mlp_ratio=float(cfg(config, "model.mlp_ratio")),
        dropout=float(cfg(config, "model.dropout")),
        max_text_tokens=1,
        text_encoder="llm2vec_cache",
        hytext_cache_dir=str(cfg(config, "text.cache_dir")),
        hytext_ctxt_dim=int(cfg(config, "text.embedding_dim")),
        hytext_vtxt_dim=1,
        hytext_max_open_shards=int(cfg(config, "text.max_open_shards")),
        hytext_strict_cache=bool(cfg(config, "text.strict_cache")),
        motion_stats_dir=str(stats_root / "full"),
        local_root_stats_dir=str(stats_root / "local_root"),
        fps=float(cfg(config, "loss.fps")),
        stats_variance_eps=float(cfg(config, "normalization.variance_eps")),
        normalize_contacts=True,
        detach_root_bridge=bool(cfg(config, "model.detach_root_bridge")),
        self_conditioning=False,
        text_global_conditioning=str(
            cfg(config, "model.text_global_conditioning")
        ),
        text_fusion_mode=str(cfg(config, "model.text_fusion_mode")),
        backbone_type="flux",
        max_frames=int(cfg(config, "data.max_target_frames")),
        source_fusion_mode=str(cfg(config, "model.source_fusion_mode")),
        use_ease=False,
        actor_exchange_dim=int(cfg(config, "model.actor_exchange_dim")),
        actor_exchange_heads=int(cfg(config, "model.actor_exchange_heads")),
        llm2vec_context_cache_dir=str(
            cfg_optional(config, "text.context_cache_dir", "")
        ),
        llm2vec_context_max_open_shards=int(
            cfg_optional(config, "text.context_max_open_shards", 16)
        ),
        local_text_cross_attention=bool(
            cfg_optional(config, "model.local_text_cross_attention", False)
        ),
        llm2vec_token_sequence_mode=str(
            cfg_optional(config, "model.text_token_sequence", "sentence")
        ),
    )


def create_normalizer(config: Mapping[str, Any]) -> HY273Normalizer:
    stats_root = Path(str(cfg(config, "data.stats_root"))).expanduser().resolve()
    return HY273Normalizer.from_data_root(
        stats_root,
        stats_dir=stats_root / "full",
        variance_eps=float(cfg(config, "normalization.variance_eps")),
        normalize_contacts=True,
    )


def _is_adaptation_parameter(name: str) -> bool:
    return (
        name.startswith("source_context.")
        or "actor_exchange" in name
        or name.startswith("task_condition_embed.")
    )


def create_optimizer(
    model: HY273UnifiedActorFlow,
    config: Mapping[str, Any],
    *,
    step: int,
) -> torch.optim.AdamW:
    base: list[torch.nn.Parameter] = []
    adaptation_decay: list[torch.nn.Parameter] = []
    adaptation_no_decay: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"Duplicate optimizer parameter: {name}")
        seen.add(id(parameter))
        if not _is_adaptation_parameter(name):
            base.append(parameter)
        elif (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "embed" in name
            or "norm" in name
        ):
            adaptation_no_decay.append(parameter)
        else:
            adaptation_decay.append(parameter)
    if not base or not adaptation_decay or not adaptation_no_decay:
        raise RuntimeError("Optimizer groups are incomplete")
    base_lr = (
        float(cfg(config, "training.stage_a_base_lr"))
        if int(step) < 100_000
        else float(cfg(config, "training.stage_b_base_lr"))
    )
    adaptation_lr = (
        float(cfg(config, "training.stage_a_adaptation_lr"))
        if int(step) < 100_000
        else float(cfg(config, "training.stage_b_adaptation_lr"))
    )
    groups = [
        {
            "params": base,
            "group_name": "base",
            "lr": base_lr,
            "weight_decay": float(cfg(config, "training.weight_decay")),
        },
        {
            "params": adaptation_decay,
            "group_name": "adaptation_decay",
            "lr": adaptation_lr,
            "weight_decay": float(cfg(config, "training.weight_decay")),
        },
        {
            "params": adaptation_no_decay,
            "group_name": "adaptation_no_decay",
            "lr": adaptation_lr,
            "weight_decay": 0.0,
        },
    ]
    covered = {id(parameter) for group in groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if covered != expected:
        raise RuntimeError("Optimizer groups do not cover the complete trainable model")
    return torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in cfg(config, "training.adam_betas")),
        eps=float(cfg(config, "training.adam_eps")),
    )


def apply_learning_rates(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    *,
    step: int,
) -> None:
    base_lr = (
        float(cfg(config, "training.stage_a_base_lr"))
        if int(step) < 100_000
        else float(cfg(config, "training.stage_b_base_lr"))
    )
    adaptation_lr = (
        float(cfg(config, "training.stage_a_adaptation_lr"))
        if int(step) < 100_000
        else float(cfg(config, "training.stage_b_adaptation_lr"))
    )
    for group in optimizer.param_groups:
        name = str(group["group_name"])
        group["lr"] = base_lr if name == "base" else adaptation_lr


def make_loss_weights(config: Mapping[str, Any]) -> HY273MultitaskLossWeights:
    return HY273MultitaskLossWeights(
        representation_scale=float(cfg(config, "loss.representation_scale")),
        contact=float(cfg(config, "loss.contact")),
        clean_root_velocity=float(cfg(config, "loss.clean_root_velocity")),
        clean_joint_velocity=float(cfg(config, "loss.clean_joint_velocity")),
        foot_lock=float(cfg(config, "loss.foot_lock")),
        fk_consistency=float(cfg(config, "loss.fk_consistency")),
        control_continuous=0.0,
        control_contact=0.0,
        velocity_t_eps=float(cfg(config, "loss.velocity_t_eps")),
        fk_warmup_steps=int(cfg(config, "loss.fk_warmup_steps")),
        fk_scale_m=float(cfg(config, "loss.fk_scale_m")),
        fps=float(cfg(config, "loss.fps")),
        contact_threshold=float(cfg(config, "loss.contact_threshold")),
    )


def make_edit_weights(config: Mapping[str, Any]) -> UnifiedEditLossWeights:
    return UnifiedEditLossWeights(
        target_x0_scale=float(cfg(config, "edit_loss.target_x0")),
        hard_x0_scale=float(cfg(config, "edit_loss.hard_x0")),
        hard_fraction=float(cfg(config, "edit_loss.hard_fraction")),
        instruction_rank_scale=0.0,
        instruction_relative_margin=0.0,
    )


def make_interaction_weights(
    config: Mapping[str, Any],
) -> HY273InteractionLossWeights:
    section = cfg(
        config,
        "reaction_loss"
        if str(cfg_optional(config, "data.paired_task", "interaction")) == "reaction"
        else "interaction_loss",
    )
    return HY273InteractionLossWeights(
        relative_root=float(section["relative_root"]),
        relative_heading=float(section["relative_heading"]),
        joint_distance=float(section["joint_distance"]),
        close_joint_vector=float(section["close_joint_vector"]),
        root_scale_m=float(section["root_scale_m"]),
        distance_scale_m=float(section["distance_scale_m"]),
        close_vector_scale_m=float(section["close_vector_scale_m"]),
        distance_gt_threshold_m=float(section["distance_gt_threshold_m"]),
        close_gt_threshold_m=float(section["close_gt_threshold_m"]),
        min_flow_t=float(section["min_flow_t"]),
    )


def _flow_seed(
    *,
    plan: Any,
    manifest_sha256: str,
    run_seed: int,
    stream: TrainStream,
    paired_task: str,
    random_stream_id: str,
) -> int:
    task = (
        TaskId.GENERATE
        if stream == TrainStream.HML_MIXED
        else TaskId.EDIT
        if stream == TrainStream.MOTION_EDIT
        else TaskId.REACTION
        if str(paired_task) == "reaction"
        else TaskId.INTERACTION
    )
    return sample_key_u64(
        manifest_sha256=manifest_sha256,
        run_seed=int(run_seed),
        global_sample_ordinal=int(plan.global_sample_ordinal),
        train_stream_id=int(stream),
        task_id=int(task),
        uid=str(plan.uid),
        random_stream_id=random_stream_id,
    )


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def build_flow_batch(
    *,
    batch: Mapping[str, Any],
    stream: TrainStream,
    normalizer: HY273Normalizer,
    manifest_sha256: str,
    run_seed: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    target = batch["target_motion"].to(device=device, dtype=torch.float32)
    if target.ndim == 3:
        target = target.unsqueeze(1)
        actor_valid = batch["condition"].target_valid.to(device=device).unsqueeze(1)
    elif target.ndim == 4:
        actor_valid = batch["target_valid"].to(device=device)
    else:
        raise ValueError("Target motion must have shape [B,T,D] or [B,A,T,D]")
    if target.shape[1] not in {1, 2} or target.shape[-1] != DIM_HY273:
        raise ValueError("Unified target actor shape is invalid")
    batch_size, actor_count, frames = target.shape[:3]
    x0_norm = normalizer.normalize(
        target.reshape(batch_size * actor_count, frames, DIM_HY273)
    ).reshape_as(target)

    timesteps = []
    noises = []
    for plan in batch["plans"]:
        timesteps.append(
            sample_timesteps(
                1,
                device=device,
                schedule=str(cfg(config, "flow.timestep_schedule")),
                p_mean=float(cfg(config, "flow.timestep_mean")),
                p_std=float(cfg(config, "flow.timestep_std")),
                generator=_generator(
                    device,
                    _flow_seed(
                        plan=plan,
                        manifest_sha256=manifest_sha256,
                        run_seed=run_seed,
                        stream=stream,
                        paired_task=str(
                            cfg_optional(config, "data.paired_task", "interaction")
                        ),
                        random_stream_id="flow_t",
                    ),
                ),
            )
        )
        noises.append(
            torch.randn(
                actor_count,
                frames,
                DIM_HY273,
                device=device,
                dtype=torch.float32,
                generator=_generator(
                    device,
                    _flow_seed(
                        plan=plan,
                        manifest_sha256=manifest_sha256,
                        run_seed=run_seed,
                        stream=stream,
                        paired_task=str(
                            cfg_optional(config, "data.paired_task", "interaction")
                        ),
                        random_stream_id="unified_actor_noise",
                    ),
                ),
            )
        )
    timestep = torch.cat(timesteps)
    noise = torch.stack(noises)
    t_view = timestep.view(batch_size, 1, 1, 1)
    z = t_view * x0_norm + (1.0 - t_view) * noise
    hard_mask = torch.zeros_like(x0_norm, dtype=torch.bool)
    observed_norm = torch.zeros_like(x0_norm)
    model_in = torch.cat([z, hard_mask.to(dtype=z.dtype)], dim=-1)
    return {
        "target_physical": target,
        "x0_norm": x0_norm,
        "timestep": timestep,
        "noise": noise,
        "z_imputed": z,
        "hard_mask": hard_mask,
        "observed_norm": observed_norm,
        "model_in": model_in,
        "actor_valid": actor_valid,
        "actor_count": actor_count,
    }


def globalize_ratio_terms(
    terms: Mapping[str, RatioLossTerm],
    *,
    world_size: int,
) -> None:
    ordered = list(terms.values())
    if not ordered:
        return
    denominators = torch.stack(
        [term.denominator.detach().float() for term in ordered]
    )
    if _distributed():
        dist.all_reduce(denominators, op=dist.ReduceOp.SUM)
    for term, denominator in zip(ordered, denominators):
        term.backward_denominator = denominator.to(
            device=term.numerator.device,
            dtype=term.numerator.dtype,
        )
        term.backward_numerator_scale = float(world_size)


def sum_weighted_terms(
    terms: Mapping[str, RatioLossTerm], reference: torch.Tensor
) -> torch.Tensor:
    return sum((term.weighted for term in terms.values()), reference.sum() * 0.0)


def _masked_gradient_rms(
    gradient: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if tuple(valid.shape) != tuple(gradient.shape[:-1]):
        raise ValueError(
            "Gradient valid mask must match every non-feature axis: "
            f"gradient={tuple(gradient.shape)}, valid={tuple(valid.shape)}"
        )
    mask = valid.unsqueeze(-1).expand_as(gradient).to(dtype=gradient.dtype)
    numerator = (gradient.float().square() * mask.float()).sum()
    denominator = mask.float().sum()
    if _distributed():
        dist.all_reduce(numerator, op=dist.ReduceOp.SUM)
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return torch.sqrt(numerator / denominator.clamp_min(1.0))


class MetricWindow:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.scalars: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.ratio_numerators: dict[str, torch.Tensor] = {}
        self.ratio_denominators: dict[str, torch.Tensor] = {}
        self.local_actor_frames = torch.zeros((), device=device, dtype=torch.float64)
        self.local_scenes = torch.zeros((), device=device, dtype=torch.float64)
        self.steps = 0

    def add_scalar(self, name: str, value: torch.Tensor | float) -> None:
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32).detach()
        self.scalars[name] = self.scalars.get(name, tensor.new_zeros(())) + tensor
        self.counts[name] = self.counts.get(name, tensor.new_zeros(())) + 1.0

    def add_terms(self, prefix: str, terms: Mapping[str, RatioLossTerm]) -> None:
        for name, term in terms.items():
            key = f"{prefix}/{name}"
            numerator = term.numerator.detach().float()
            denominator = term.denominator.detach().float()
            self.ratio_numerators[key] = self.ratio_numerators.get(
                key, numerator.new_zeros(())
            ) + numerator
            self.ratio_denominators[key] = self.ratio_denominators.get(
                key, denominator.new_zeros(())
            ) + denominator

    def add_batch(self, actor_valid: torch.Tensor) -> None:
        self.local_actor_frames += actor_valid.sum().detach().to(dtype=torch.float64)
        self.local_scenes += float(actor_valid.shape[0])
        self.steps += 1

    def flush(self, elapsed: float, world_size: int) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name in sorted(self.scalars):
            values = torch.stack([self.scalars[name], self.counts[name]])
            if _distributed():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
            metrics[name] = float((values[0] / values[1].clamp_min(1.0)).item())
        for name in sorted(self.ratio_numerators):
            values = torch.stack(
                [self.ratio_numerators[name], self.ratio_denominators[name]]
            )
            if _distributed():
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
            metrics[f"{name}/raw"] = float(
                (values[0] / values[1].clamp_min(1.0)).item()
            )
        counts = torch.stack([self.local_actor_frames, self.local_scenes])
        if _distributed():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        elapsed_tensor = torch.tensor(elapsed, device=self.device, dtype=torch.float64)
        if _distributed():
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        elapsed_value = max(float(elapsed_tensor.item()), 1e-9)
        metrics["throughput/actor_frames_per_second"] = float(
            counts[0].item() / elapsed_value
        )
        metrics["throughput/scenes_per_second"] = float(
            counts[1].item() / elapsed_value
        )
        metrics["time/step_seconds"] = elapsed_value / max(self.steps, 1)
        metrics["schedule/world_size"] = float(world_size)
        return metrics


def initialize_ema(model: HY273UnifiedActorFlow) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def update_ema(
    ema: dict[str, torch.Tensor],
    model: HY273UnifiedActorFlow,
    decay: float,
) -> None:
    for name, value in model.state_dict().items():
        target = ema[name]
        if value.is_floating_point():
            target.mul_(float(decay)).add_(value.detach(), alpha=1.0 - float(decay))
        else:
            target.copy_(value.detach())


def save_checkpoint(
    path: Path,
    *,
    model: HY273UnifiedActorFlow,
    normalizer: HY273Normalizer,
    optimizer: torch.optim.Optimizer,
    ema: Mapping[str, torch.Tensor],
    batcher: HY273UnifiedActorStepBatcher,
    config: Mapping[str, Any],
    config_path: Path,
    run_name: str,
    next_global_step: int,
    ema_update_count: int,
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "run_name": run_name,
        "next_global_step": int(next_global_step),
        "model": model.state_dict(),
        "ema": dict(ema),
        "optimizer": optimizer.state_dict(),
        "batcher": batcher.state_dict(),
        "config": dict(config),
        "config_path": str(config_path),
        "ema_update_count": int(ema_update_count),
        "rng_contract": RNG_CONTRACT,
        "normalizer": normalizer.state_dict(),
        "normalization": {
            "stats_root": str(cfg(config, "data.stats_root")),
            "normalize_contacts": True,
            "variance_eps": float(cfg(config, "normalization.variance_eps")),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def update_latest_link(checkpoint: Path, latest: Path) -> None:
    if latest.exists() and os.path.samefile(checkpoint, latest):
        return
    temporary = latest.with_name(f"{latest.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        os.link(checkpoint, temporary)
    except OSError:
        import shutil

        shutil.copy2(checkpoint, temporary)
    os.replace(temporary, latest)


def validate_resume_config(
    checkpoint_config: object,
    current_config: Mapping[str, Any],
    *,
    allow_same_mix_extension_at_step: int | None = None,
) -> None:
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("Checkpoint is missing its resolved training config")
    checkpoint_dict = dict(checkpoint_config)
    current_dict = dict(current_config)
    if checkpoint_dict == current_dict:
        return

    extension_matches = False
    if allow_same_mix_extension_at_step is not None:
        boundary = int(allow_same_mix_extension_at_step)
        checkpoint_segments = list(cfg(checkpoint_dict, "schedule.segments"))
        current_segments = list(cfg(current_dict, "schedule.segments"))
        if len(checkpoint_segments) == len(current_segments) and checkpoint_segments:
            checkpoint_last = dict(checkpoint_segments[-1])
            current_last = dict(current_segments[-1])
            checkpoint_end = int(checkpoint_last.pop("end", -1))
            current_end = int(current_last.pop("end", -1))
            schedule_extension = (
                checkpoint_segments[:-1] == current_segments[:-1]
                and checkpoint_last == current_last
                and checkpoint_end == boundary
                and current_end > boundary
            )
            checkpoint_max = int(cfg(checkpoint_dict, "training.max_global_step"))
            current_max = int(cfg(current_dict, "training.max_global_step"))
            normalized_checkpoint = {
                **checkpoint_dict,
                "schedule": {
                    **dict(checkpoint_dict["schedule"]),
                    "segments": current_segments,
                },
                "training": {
                    **dict(checkpoint_dict["training"]),
                    "max_global_step": current_max,
                },
            }
            extension_matches = (
                schedule_extension
                and checkpoint_max == boundary
                and current_max == current_end
                and normalized_checkpoint == current_dict
            )
    if not extension_matches:
        raise ValueError(
            "Resolved training config changed across unified-actor resume"
        )


def validate_resume_run_name(
    checkpoint_run_name: object,
    expected_run_name: str,
) -> None:
    if str(checkpoint_run_name or "") != str(expected_run_name):
        raise ValueError(
            "Resume checkpoint belongs to a different run: "
            f"checkpoint={checkpoint_run_name!r}, "
            f"requested={expected_run_name!r}"
        )


def load_checkpoint(
    path: Path,
    *,
    expected_run_name: str,
    model: HY273UnifiedActorFlow,
    optimizer: torch.optim.Optimizer,
    batcher: HY273UnifiedActorStepBatcher,
    normalizer: HY273Normalizer,
    config: Mapping[str, Any],
    device: torch.device,
    allow_same_mix_extension: bool = False,
) -> tuple[int, dict[str, torch.Tensor], int]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported unified actor checkpoint: {path}")
    validate_resume_run_name(checkpoint.get("run_name"), expected_run_name)
    if checkpoint.get("rng_contract") != RNG_CONTRACT:
        raise ValueError("Checkpoint uses a different stochastic training contract")
    checkpoint_step = int(checkpoint["next_global_step"])
    schedule_extension_step = checkpoint_step if allow_same_mix_extension else None
    validate_resume_config(
        checkpoint.get("config"),
        config,
        allow_same_mix_extension_at_step=schedule_extension_step,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    batcher.load_state_dict(
        checkpoint["batcher"],
        allow_segment_extension_at_step=schedule_extension_step,
    )
    saved_normalizer = checkpoint.get("normalizer")
    if not isinstance(saved_normalizer, Mapping):
        raise ValueError("Checkpoint is missing the unified normalizer state")
    current_normalizer = normalizer.state_dict()
    for name in ("mean", "std", "variance_eps"):
        saved = torch.as_tensor(saved_normalizer[name]).cpu()
        current = torch.as_tensor(current_normalizer[name]).cpu()
        if not torch.equal(saved, current):
            raise ValueError(f"Normalizer changed across resume at {name}")
    step = checkpoint_step
    if batcher.scheduler.state.next_step != step:
        raise RuntimeError("Batcher resume step differs from the checkpoint step")
    ema = {
        name: value.to(device=device)
        for name, value in checkpoint["ema"].items()
    }
    expected = model.state_dict()
    if set(ema) != set(expected):
        raise ValueError("EMA tensor names differ from the current model")
    for name, value in expected.items():
        if ema[name].shape != value.shape:
            raise ValueError(f"EMA tensor shape mismatch at {name}")
    return step, ema, int(checkpoint.get("ema_update_count", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/hy273_unified_actor_v1.yaml",
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--stop_step", type=int, default=0)
    parser.add_argument("--max_updates", type=int, default=0)
    parser.add_argument("--materialize_workers", type=int, default=0)
    parser.add_argument(
        "--phase_contract",
        choices=FULLTEXT_PHASE_CONTRACTS,
        default="",
    )
    parser.add_argument("--save_final", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    validate_config(config)
    paired_task = str(cfg_optional(config, "data.paired_task", "interaction"))
    device, rank, world_size = setup_distributed()
    batcher: HY273UnifiedActorStepBatcher | None = None
    try:
        if device.type != "cuda":
            raise RuntimeError("Unified actor training requires CUDA")
        expected_world = int(cfg(config, "training.world_size"))
        if world_size != expected_world:
            raise ValueError(f"Expected world_size={expected_world}, got {world_size}")
        seed = int(cfg(config, "training.seed"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        output_root = Path(
            args.output_dir or str(cfg(config, "training.output_dir"))
        ).expanduser().resolve()
        run_dir = output_root / args.name
        model_dir = run_dir / "model"
        run_dir_existed_flag = torch.tensor(
            [int(run_dir.exists()) if rank == 0 else 0],
            device=device,
            dtype=torch.int32,
        )
        if _distributed():
            dist.broadcast(run_dir_existed_flag, src=0)
        run_dir_existed = bool(run_dir_existed_flag.item())
        declared_stop_step = (
            int(args.stop_step)
            if args.stop_step > 0
            else int(cfg(config, "training.max_global_step"))
        )
        validate_fulltext_phase_contract(
            args.phase_contract,
            has_resume=bool(args.resume),
            run_dir_exists=run_dir_existed,
            declared_stop_step=declared_stop_step,
        )
        if args.phase_contract:
            paired_task = str(
                cfg_optional(config, "data.paired_task", "interaction")
            )
            if paired_task == "reaction":
                if str(
                    cfg_optional(config, "model.text_token_sequence", "sentence")
                ) != "sentence_plus_context":
                    raise ValueError(
                        "Reaction full-text phases require the main contextual token stream"
                    )
            elif not bool(
                cfg_optional(config, "model.local_text_cross_attention", False)
            ):
                raise ValueError(
                    "Legacy full-text phases require local text cross-attention"
                )
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)
        if _distributed():
            dist.barrier()

        model = create_model(config).to(device)
        normalizer = create_normalizer(config).to(device)
        optimizer = create_optimizer(model, config, step=0)
        workers = (
            int(args.materialize_workers)
            if args.materialize_workers > 0
            else int(cfg(config, "data.materialize_workers"))
        )
        batcher = HY273UnifiedActorStepBatcher(
            multitask_manifest=str(cfg(config, "data.multitask_train_manifest")),
            interaction_root=str(
                cfg(
                    config,
                    "data.reaction_root"
                    if str(cfg_optional(config, "data.paired_task", "interaction"))
                    == "reaction"
                    else "data.interaction_root",
                )
            ),
            run_seed=seed,
            world_size=world_size,
            rank=rank,
            batch_size_t2m_edit=int(
                cfg(config, "training.batch_size_t2m_edit_per_rank")
            ),
            batch_size_interaction=int(
                cfg(
                    config,
                    "training.batch_size_reaction_per_rank"
                    if str(cfg_optional(config, "data.paired_task", "interaction"))
                    == "reaction"
                    else "training.batch_size_interaction_per_rank",
                )
            ),
            batch_size_edit=int(
                cfg_optional(
                    config,
                    "training.batch_size_edit_per_rank",
                    cfg(config, "training.batch_size_t2m_edit_per_rank"),
                )
            ),
            task_segments=list(cfg(config, "schedule.segments")),
            materialize_workers=workers,
            sort_window_batches=int(cfg(config, "data.sort_window_batches")),
            verify_payload_hash=False,
            interaction_exclude_overlength=bool(
                cfg_optional(
                    config,
                    "data.interaction_exclude_overlength",
                    False,
                )
            ),
            paired_task=str(
                cfg_optional(config, "data.paired_task", "interaction")
            ),
        )
        ema = initialize_ema(model)
        ema_update_count = 0
        global_step = 0
        if args.resume:
            global_step, ema, ema_update_count = load_checkpoint(
                Path(args.resume).expanduser().resolve(),
                expected_run_name=args.name,
                model=model,
                optimizer=optimizer,
                batcher=batcher,
                normalizer=normalizer,
                config=config,
                device=device,
                allow_same_mix_extension=(
                    args.phase_contract == FULLTEXT_STAGE_B_CONTINUE_CONTRACT
                ),
            )
        validate_fulltext_phase_contract(
            args.phase_contract,
            has_resume=bool(args.resume),
            run_dir_exists=run_dir_existed,
            declared_stop_step=declared_stop_step,
            global_step=global_step,
        )
        if rank == 0:
            (run_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
        if _distributed():
            dist.barrier()
        apply_learning_rates(optimizer, config, step=global_step)

        ddp_model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=float(cfg(config, "training.ddp_bucket_cap_mb")),
        )
        loss_weights = make_loss_weights(config)
        edit_weights = make_edit_weights(config)
        interaction_weights = make_interaction_weights(config)
        max_global_step = int(cfg(config, "training.max_global_step"))
        stop_step = declared_stop_step
        if not global_step < stop_step <= max_global_step:
            raise ValueError(
                f"Invalid training interval [{global_step}, {stop_step})"
            )
        if args.max_updates > 0:
            stop_step = min(stop_step, global_step + int(args.max_updates))

        precision = str(cfg(config, "training.precision"))
        if precision != "bf16" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Unified actor v1 requires CUDA bf16")
        log_every = int(cfg(config, "training.log_every"))
        latest_every = int(cfg(config, "training.latest_every"))
        archive_every = int(cfg(config, "training.archive_every"))
        ema_every = int(cfg(config, "training.ema_every"))
        ema_decay = float(cfg(config, "training.ema_decay"))
        gradient_clip = float(cfg(config, "training.gradient_clip"))
        seen_streams: set[TrainStream] = set()
        metrics_window = MetricWindow(device)
        torch.cuda.synchronize(device)
        window_start = time.perf_counter()

        if rank == 0:
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            trainable_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            print(
                json.dumps(
                    {
                        "event": "train_start",
                        "run": args.name,
                        "start_step": global_step,
                        "stop_step": stop_step,
                        "world_size": world_size,
                        "parameters": parameter_count,
                        "trainable_parameters": trainable_count,
                        "task_schedule": cfg(config, "schedule.segments"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        while global_step < stop_step:
            apply_learning_rates(optimizer, config, step=global_step)
            batch_start = time.perf_counter()
            batch, stream, trace = batcher.next_batch(global_step)
            data_seconds = time.perf_counter() - batch_start
            manifest_sha = batcher.manifest_hashes[int(stream)]
            flow = build_flow_batch(
                batch=batch,
                stream=stream,
                normalizer=normalizer,
                manifest_sha256=manifest_sha,
                run_seed=seed,
                config=config,
                device=device,
            )
            actors = int(flow["actor_count"])
            condition = (
                batch["condition"].to(device) if actors == 1 else None
            )
            task_id = (
                condition.task_id
                if condition is not None
                else batch["task_id"].to(device)
            )
            texts = list(batch["texts"])
            text_profiles = (
                condition.text_encoding_profile
                if condition is not None
                else tuple(batch["text_profiles"])
            )
            c_dir = (
                condition.frame_gauge_dir
                if condition is not None
                else batch["frame_gauge_dir"].to(device)
            )
            model_in = (
                flow["model_in"][:, 0]
                if actors == 1
                else flow["model_in"]
            )
            length_mask = (
                flow["actor_valid"][:, 0]
                if actors == 1
                else flow["actor_valid"]
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=True,
            ):
                prediction = ddp_model(
                    model_in,
                    flow["timestep"],
                    c_dir=c_dir,
                    text=texts,
                    length_mask=length_mask,
                    condition=condition,
                    task_id=task_id,
                    text_profiles=text_profiles,
                )
            prediction_actors = (
                prediction.unsqueeze(1) if actors == 1 else prediction
            )
            flat_prediction = prediction_actors.reshape(
                prediction_actors.shape[0] * actors,
                prediction_actors.shape[2],
                DIM_HY273,
            )
            flat_target_norm = flow["x0_norm"].reshape_as(flat_prediction)
            flat_target_physical = flow["target_physical"].reshape_as(flat_prediction)
            flat_z = flow["z_imputed"].reshape_as(flat_prediction)
            flat_observed = flow["observed_norm"].reshape_as(flat_prediction)
            flat_hard = flow["hard_mask"].reshape_as(flat_prediction)
            flat_valid = flow["actor_valid"].reshape(
                flat_prediction.shape[0], flat_prediction.shape[1]
            )
            flat_timestep = flow["timestep"].repeat_interleave(actors)

            with torch.autocast(device_type="cuda", enabled=False):
                base_bundle = compute_hy273_unified_flow_loss(
                    x0_hat_norm=flat_prediction.float(),
                    z_imputed=flat_z.float(),
                    x0_target_norm=flat_target_norm.float(),
                    x0_target_physical=flat_target_physical.float(),
                    hard_observed_norm=flat_observed.float(),
                    hard_mask=flat_hard,
                    target_valid=flat_valid,
                    timesteps=flat_timestep.float(),
                    normalizer=normalizer,
                    global_step=global_step,
                    weights=loss_weights,
                    representation_loss_space="velocity_mse",
                    contact_loss_space="velocity_mse",
                )
                globalize_ratio_terms(base_bundle.terms, world_size=world_size)
                base_bundle.total = sum_weighted_terms(
                    base_bundle.terms, flat_prediction
                )
                total_loss = base_bundle.total

                edit_bundle = None
                if stream == TrainStream.MOTION_EDIT:
                    edit_bundle = compute_unified_edit_loss(
                        correct_x0_hat_cont=prediction[..., :CONT_DIM].float(),
                        shuffled_x0_hat_cont=None,
                        x0_target_norm=flow["x0_norm"][:, 0].float(),
                        target_valid=flow["actor_valid"][:, 0],
                        hard_mask=flow["hard_mask"][:, 0],
                        weights=edit_weights,
                        auxiliary_reduction="global_element_ratio",
                    )
                    globalize_ratio_terms(
                        edit_bundle.terms, world_size=world_size
                    )
                    edit_bundle.refresh_ratio_terms(edit_weights)
                    total_loss = total_loss + edit_bundle.total

                interaction_bundle = None
                base_output_grad_rms = None
                relation_output_grad_rms = None
                if stream == TrainStream.REACTION:
                    if paired_task == "reaction":
                        if actors != 1 or condition is None:
                            raise RuntimeError(
                                "Reaction must use one target actor and a ConditionBatch"
                            )
                        prediction_physical = normalizer.denormalize(
                            prediction.float()
                        )
                        source = prediction_physical.new_zeros(
                            prediction_physical.shape
                        )
                        source_valid = torch.zeros_like(
                            flow["actor_valid"][:, 0]
                        )
                        source_frames = min(
                            source.shape[1], condition.source_motion.shape[2]
                        )
                        source[:, :source_frames] = condition.source_motion[
                            :, 0, :source_frames
                        ].to(device=device, dtype=source.dtype)
                        source_valid[:, :source_frames] = (
                            condition.source_time_valid[:, 0, :source_frames]
                            & condition.source_present[:, 0, None]
                        )
                        prediction_pair = torch.stack(
                            [source, prediction_physical], dim=1
                        )
                        target_pair = torch.stack(
                            [source, flow["target_physical"][:, 0]], dim=1
                        )
                        pair_valid = torch.stack(
                            [source_valid, flow["actor_valid"][:, 0]], dim=1
                        )
                        interaction_bundle = compute_hy273_interaction_loss(
                            prediction_physical=prediction_pair,
                            target_physical=target_pair,
                            actor_valid=pair_valid,
                            timesteps=flow["timestep"],
                            weights=interaction_weights,
                        )
                    else:
                        prediction_physical = normalizer.denormalize(
                            prediction.float().reshape(
                                prediction.shape[0] * 2,
                                prediction.shape[2],
                                DIM_HY273,
                            )
                        ).reshape_as(flow["target_physical"])
                        interaction_bundle = compute_hy273_interaction_loss(
                            prediction_physical=prediction_physical,
                            target_physical=flow["target_physical"],
                            actor_valid=flow["actor_valid"],
                            timesteps=flow["timestep"],
                            weights=interaction_weights,
                        )
                    globalize_ratio_terms(
                        interaction_bundle.terms, world_size=world_size
                    )
                    interaction_bundle.total = sum_weighted_terms(
                        interaction_bundle.terms, prediction
                    )
                    if 100_000 <= global_step < 100_500:
                        base_output_grad = torch.autograd.grad(
                            base_bundle.total,
                            prediction,
                            retain_graph=True,
                        )[0]
                        relation_output_grad = torch.autograd.grad(
                            interaction_bundle.total,
                            prediction,
                            retain_graph=True,
                        )[0]
                        base_output_grad_rms = _masked_gradient_rms(
                            base_output_grad, length_mask
                        )
                        relation_output_grad_rms = _masked_gradient_rms(
                            relation_output_grad, length_mask
                        )
                    total_loss = total_loss + interaction_bundle.total

            if not bool(torch.isfinite(total_loss.detach())):
                raise RuntimeError(f"Non-finite loss at step {global_step}")
            total_loss.backward()
            local_gates = [
                parameter
                for name, parameter in model.named_parameters()
                if "local_text_gates." in name
            ]
            if local_gates:
                metrics_window.add_scalar(
                    "text/local_gate_abs_mean",
                    torch.stack(
                        [torch.tanh(gate.detach()).abs() for gate in local_gates]
                    ).mean(),
                )
                gate_grads = [
                    gate.grad.detach().abs()
                    for gate in local_gates
                    if gate.grad is not None
                ]
                if gate_grads:
                    metrics_window.add_scalar(
                        "text/local_gate_grad_abs_mean",
                        torch.stack(gate_grads).mean(),
                    )
            if stream not in seen_streams:
                missing = [
                    name
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                if missing:
                    raise RuntimeError(
                        f"Unused trainable parameters for {stream.name}: {missing[:20]}"
                    )
                seen_streams.add(stream)
            if global_step < 100_000:
                # Stage-A is the pure K-Encoder T2M base. New capability
                # parameters retain both their initialization and empty Adam
                # state until Stage-B starts.
                for name, parameter in model.named_parameters():
                    if _is_adaptation_parameter(name):
                        parameter.grad = None
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip
            )
            if not bool(torch.isfinite(torch.as_tensor(grad_norm)).all()):
                raise RuntimeError(f"Non-finite gradient norm at step {global_step}")
            optimizer.step()
            if global_step % ema_every == 0:
                update_ema(ema, model, ema_decay)
                ema_update_count += 1

            stream_name = stream.name.lower()
            metrics_window.add_scalar("loss/total", total_loss)
            metrics_window.add_scalar(f"loss/{stream_name}/total", total_loss)
            metrics_window.add_scalar("grad/preclip_norm", grad_norm)
            metrics_window.add_scalar("time/data_seconds", data_seconds)
            metrics_window.add_terms(
                f"loss/{stream_name}/base", base_bundle.terms
            )
            if edit_bundle is not None:
                metrics_window.add_scalar(
                    "loss/motion_edit/target_x0_raw",
                    edit_bundle.target_x0_raw,
                )
                metrics_window.add_scalar(
                    "loss/motion_edit/hard_x0_raw",
                    edit_bundle.hard_x0_raw,
                )
                metrics_window.add_scalar(
                    "loss/motion_edit/target_x0_weighted",
                    edit_bundle.target_x0_weighted,
                )
                metrics_window.add_scalar(
                    "loss/motion_edit/hard_x0_weighted",
                    edit_bundle.hard_x0_weighted,
                )
            if interaction_bundle is not None:
                relation_namespace = (
                    "reaction" if paired_task == "reaction" else "interaction"
                )
                metrics_window.add_terms(
                    f"loss/{relation_namespace}/relation",
                    interaction_bundle.terms,
                )
                metrics_window.add_scalar(
                    f"{relation_namespace}/distance_mask_fraction",
                    interaction_bundle.distance_mask_fraction,
                )
                metrics_window.add_scalar(
                    f"{relation_namespace}/close_mask_fraction",
                    interaction_bundle.close_mask_fraction,
                )
                metrics_window.add_scalar(
                    f"{relation_namespace}/relation_active_scene_fraction",
                    interaction_bundle.active_scene_fraction,
                )
                if (
                    base_output_grad_rms is not None
                    and relation_output_grad_rms is not None
                ):
                    metrics_window.add_scalar(
                        f"{relation_namespace}/base_output_grad_rms",
                        base_output_grad_rms,
                    )
                    metrics_window.add_scalar(
                        f"{relation_namespace}/relation_output_grad_rms",
                        relation_output_grad_rms,
                    )
                    metrics_window.add_scalar(
                        f"{relation_namespace}/relation_to_base_output_grad_ratio",
                        relation_output_grad_rms
                        / base_output_grad_rms.clamp_min(1e-12),
                    )
            metrics_window.add_batch(flow["actor_valid"])
            global_step += 1

            if global_step % log_every == 0 or global_step == stop_step:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - window_start
                metrics = metrics_window.flush(elapsed, world_size)
                metrics["train/next_global_step"] = float(global_step)
                metrics["lr/base"] = float(optimizer.param_groups[0]["lr"])
                metrics["lr/adaptation"] = float(optimizer.param_groups[1]["lr"])
                metrics["gpu/max_memory_gib"] = float(
                    torch.cuda.max_memory_allocated(device) / (1024**3)
                )
                torch.cuda.reset_peak_memory_stats(device)
                if rank == 0:
                    record = {
                        "step": global_step,
                        "last_stream": stream_name,
                        "last_trace": trace,
                        "metrics": metrics,
                    }
                    with (run_dir / "metrics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(
                            json.dumps(record, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                    print(
                        f"[train] step={global_step} stream={stream_name} "
                        f"loss={metrics['loss/total']:.6f} "
                        f"step_s={metrics['time/step_seconds']:.3f} "
                        f"actor_fps={metrics['throughput/actor_frames_per_second']:.1f} "
                        f"grad={metrics['grad/preclip_norm']:.3f} "
                        f"mem_gib={metrics['gpu/max_memory_gib']:.2f}",
                        flush=True,
                    )
                metrics_window = MetricWindow(device)
                torch.cuda.synchronize(device)
                window_start = time.perf_counter()

            save_latest = latest_every > 0 and global_step % latest_every == 0
            save_archive = archive_every > 0 and global_step % archive_every == 0
            save_stage_end = global_step == stop_step and bool(args.save_final)
            if save_latest or save_archive or save_stage_end:
                if _distributed():
                    dist.barrier()
                if rank == 0:
                    if save_archive:
                        destination = model_dir / f"step_{global_step:08d}.pt"
                    else:
                        destination = model_dir / "latest.pt"
                    save_checkpoint(
                        destination,
                        model=model,
                        normalizer=normalizer,
                        optimizer=optimizer,
                        ema=ema,
                        batcher=batcher,
                        config=config,
                        config_path=config_path,
                        run_name=args.name,
                        next_global_step=global_step,
                        ema_update_count=ema_update_count,
                    )
                    if destination.name != "latest.pt":
                        update_latest_link(destination, model_dir / "latest.pt")
                    print(f"[checkpoint] {destination}", flush=True)
                if _distributed():
                    dist.barrier()

        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "stage_complete",
                        "run": args.name,
                        "next_global_step": global_step,
                        "ema_update_count": ema_update_count,
                        "task_state": batcher.scheduler.state_dict(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if batcher is not None:
            batcher.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
