#!/usr/bin/env python
"""Train versioned HY273 T2M/control/edit models with exact stage replay."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import uuid
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common.fixed_bucket_ddp import FixedBucketGradientSynchronizer
from models.codeflow.dit_blocks import TEXT_FUSION_MODES
from data.hy273_multitask_batcher import HY273MultitaskStepBatcher
from data.hy273_multitask_scheduler import (
    BUCKET_PLAN_VERSION,
    DECOMPOSED_CFG_EDIT_SCHEDULE_VERSIONS,
    EditConditionPattern,
    HIGH_LEVEL_SCHEDULE_VERSION,
    HML_INNER_SCHEDULE_VERSION,
    KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
    SAMPLE_RNG_VERSION,
    STAGE_C_EDIT20_SCHEDULE_VERSION,
    STAGE_C_SAFE_MIX_SCHEDULE_VERSION,
    STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION,
    TrainingPhase,
    UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION,
    UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
    UNIFIED_EDIT_SCHEDULE_VERSIONS,
    UNIFIED_EDIT_V2_SCHEDULE_VERSION,
    UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION,
    bernoulli_from_draw,
    optimizer_group_hparams,
    phase_for_step,
    probability_units_for_step,
    sample_key_u64,
)
from models.raw_motion.flow_schedule import (
    LEGACY_SPLIT_CONTACT_PROTOCOL,
    UNIFIED_273_CONTACT_PROTOCOL,
    build_flow_state,
    build_unified_273_flow_state,
    sample_timesteps,
    uses_unified_273_flow,
)
from models.raw_motion.hy273_constraints import (
    KimodoControlCurriculum,
    build_kimodo_control_curriculum_batch,
)
from models.raw_motion.hy273_ease import EASE_DIM, EASE_STATS_FORMAT
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    ConditionBatch,
    TaskId,
    TrainStream,
)
from models.raw_motion.hy273_multitask_losses import (
    HY273MultitaskLossBundle,
    HY273MultitaskLossWeights,
    R13_UNIFIED_CONTACT_WEIGHT,
    R13_UNIFIED_CONTROL_CONTACT_WEIGHT,
    REPRESENTATION_LOSS_SPACES,
    SEMANTIC_SLICES,
    compute_hy273_multitask_loss,
    compute_hy273_unified_flow_loss,
)
from models.raw_motion.hy273_unified_edit_losses import (
    PhysicalTemporalEditLossBundle,
    SourceAnchorLossBundle,
    SourceTargetDiscrepancyLossBundle,
    SourceTargetDiscrepancyMask,
    UnifiedEditLossBundle,
    UnifiedEditLossWeights,
    build_source_target_discrepancy_mask,
    compute_physical_temporal_edit_loss,
    compute_source_anchor_loss,
    compute_source_target_discrepancy_x0_loss,
    compute_unified_edit_loss,
)
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    CONT_DIM,
    DIM_HY273,
    HEADING_SLICE,
)
from models.raw_motion.hytext_cache import (
    LLM2VEC_CACHE_FORMAT,
    PROFILE_CACHE_FORMAT,
)
from models.raw_motion.kimodo_context_flow_dit import (
    HY273KimodoContextFlow,
    KimodoContextFlowOutput,
)
from models.raw_motion.kimodo_like_flow_dit import (
    TEXT_GLOBAL_CONDITIONING_MODES,
)


R11_TRAIN_CONTRACT = "hy273_multitask_train_contract_v2"
R12_TRAIN_CONTRACT = "hy273_multitask_train_contract_v3_rootmask"
R13_TRAIN_CONTRACT = "hy273_multitask_train_contract_v4_unified273"
TRAIN_CONTRACT = R11_TRAIN_CONTRACT
SUPPORTED_TRAIN_CONTRACTS = (
    R11_TRAIN_CONTRACT,
    R12_TRAIN_CONTRACT,
    R13_TRAIN_CONTRACT,
)
CHECKPOINT_FORMAT = "hy273_multitask_checkpoint_v2"
EMA_SCHEDULE = "post_optimizer_preincrement_mod10_v1"
R12_ORIGIN_PARENT_SHA256 = (
    "e06b397df60e9b68e628fa68bede687c97ecb9bb25e556f3d96a311423e1744e"
)
R12_ORIGIN_PARENT_RUN_NAME = (
    "hy273_multitask_r11_stage_a_t2m_ddp8_20260715_1510"
)
R12_ORIGIN_PARENT_RUN_UUID = "8805e8ff-6c53-4d0e-9d68-562e471babe8"
R12_ORIGIN_PARENT_BASE_CONTRACT_SHA256 = (
    "7aa80f026c32f3c6eba3175b15228c6d1b53064c5f46841ff56fbaa5fae00485"
)
R12_ORIGIN_PARENT_CONFIG_SHA256 = (
    "cc3a3af87c6ba7fb43012465ad20adc3d9991e95bb49fac85e2727941fe0d2d1"
)
R12_ORIGIN_PARENT_CODE_IDENTITY_SHA256 = (
    "093b3a36ab4cdd6735eb91175a5e6d4522058356eda61f61bdf3b04054fc5ede"
)
R12_B1_250K_CODE_IDENTITY_SHA256 = (
    "cd3a9edfb33b04c0c218d88b5ec341e2f2c1e10b6d1d0cc5e8ef0990b1b4b6c1"
)
R12_B1_250K_SAMPLER_SHA256 = (
    "fe5cc57911409fcdc9190221189be2d7a82d45e653618aea754f91e5aa5827a2"
)
R13_EDIT_RESEARCH_PARENT_RUN_NAME = (
    "hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507"
)
R13_EDIT_RESEARCH_PARENT_RUN_UUID = "c36725f7-a468-4c89-9ac2-ddbbb1502733"
DEFAULT_SAME_SOURCE_EDIT_GROUPS = Path(
    "outputs/hy273_multitask/diagnostics/"
    "r13_edit_objective_pilot_405k_20260722/"
    "tiny_overfit_candidate_groups.json"
)
OPTIMIZER_GROUP_ORDER = (
    "G0_existing",
    "G1_context_weight",
    "G2_context_bias",
)
EASE_OPTIMIZER_GROUP_ORDER = (
    "G3_ease_weight",
    "G4_ease_bias",
)
CONDITIONING_ARCHITECTURES = (
    "hytext_flux",
    "llm2vec_flux",
    "llm2vec_kimodo_prefix",
)

# Named research treatments keep every Edit pilot attributable to one declared
# objective. The formal R13 path uses no treatment and remains unchanged.
EDIT_RESEARCH_TREATMENTS: dict[str, dict[str, float | str]] = {
    "baseline": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "shuffled_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_mode": "hinge",
        "instruction_rank_temperature": 0.01,
        "instruction_rank_multiplier": 1.0,
        "instruction_negative_scope": "all",
    },
    "same_source_contrast": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "same_source_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_mode": "hinge",
        "instruction_rank_temperature": 0.01,
        "instruction_rank_multiplier": 1.0,
        "instruction_negative_scope": "all",
    },
    "same_source_hinge_only": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "same_source_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_mode": "hinge",
        "instruction_rank_temperature": 0.01,
        "instruction_rank_multiplier": 1.0,
        "instruction_negative_scope": "same_source_only",
    },
    "same_source_softplus_only": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "same_source_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_mode": "softplus",
        "instruction_rank_temperature": 0.01,
        "instruction_rank_multiplier": 0.575,
        "instruction_negative_scope": "same_source_only",
    },
    "no_rank_positive_only": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "none",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_multiplier": 0.0,
        "discrepancy_sample_scope": "all",
    },
    "physical_temporal_positive_only": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "none",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_multiplier": 0.0,
        "discrepancy_sample_scope": "all",
        # Parent-400K no-update calibration: 0.00053 gives about 15% of the
        # pre-temporal Edit objective's output-gradient RMS.
        "temporal_scale": 0.00053,
        "temporal_vector_weight": 0.50,
        "temporal_speed_weight": 0.50,
        "temporal_background_weight": 0.10,
        "temporal_change_scale_mps": 0.25,
        "temporal_smooth_l1_beta_mps": 0.10,
    },
    "same_source_changed_positive_only": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "none",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_multiplier": 0.0,
        "discrepancy_x0_scale": 0.05,
        "discrepancy_fraction": 0.20,
        "discrepancy_sample_scope": "same_source_only",
    },
    "source_token_block_positive_only": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "none",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "instruction_rank_multiplier": 0.0,
        "source_fusion_mode": "token_block",
    },
    "source_target_discrepancy_x0": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "shuffled_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
        "discrepancy_x0_scale": 0.01,
        "discrepancy_fraction": 0.20,
    },
    "anchored_identity": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "source_identity",
        "identity_base_scale": 0.10,
        "source_anchor_scale": 0.05,
        "source_anchor_relative_margin": 0.10,
    },
    "anchored_identity_low_t": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.5,
        "low_t_max": 0.2,
        "secondary_branch": "source_identity",
        "identity_base_scale": 0.10,
        "source_anchor_scale": 0.05,
        "source_anchor_relative_margin": 0.10,
    },
    "clean_x0_mse": {
        "representation_loss_space": "clean_x0_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "shuffled_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
    },
    "clean_x0_smooth_l1": {
        "representation_loss_space": "clean_x0_smooth_l1",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.0,
        "low_t_max": 0.2,
        "secondary_branch": "shuffled_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
    },
    "low_t_only": {
        "representation_loss_space": "velocity_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.5,
        "low_t_max": 0.2,
        "secondary_branch": "shuffled_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
    },
    "clean_x0_mse_low_t": {
        "representation_loss_space": "clean_x0_mse",
        "contact_loss_space": "velocity_mse",
        "representation_multiplier": 1.0,
        "low_t_mix_prob": 0.5,
        "low_t_max": 0.2,
        "secondary_branch": "shuffled_instruction",
        "identity_base_scale": 0.0,
        "source_anchor_scale": 0.0,
        "source_anchor_relative_margin": 0.10,
    },
}


def resolve_edit_research_treatment(name: str) -> dict[str, float | str]:
    """Return one immutable-by-convention treatment specification."""

    normalized = str(name).strip()
    if not normalized:
        treatment: dict[str, float | str] = {
            "name": "formal_default",
            "representation_loss_space": "velocity_mse",
            "contact_loss_space": "velocity_mse",
            "representation_multiplier": 1.0,
            "low_t_mix_prob": 0.0,
            "low_t_max": 0.2,
            "secondary_branch": "shuffled_instruction",
            "identity_base_scale": 0.0,
            "source_anchor_scale": 0.0,
            "source_anchor_relative_margin": 0.10,
            "instruction_rank_mode": "hinge",
            "instruction_rank_temperature": 0.01,
            "instruction_rank_multiplier": 1.0,
            "instruction_negative_scope": "all",
        }
    else:
        if normalized not in EDIT_RESEARCH_TREATMENTS:
            raise ValueError(
                f"Unknown Edit research treatment {normalized!r}; expected one of "
                f"{tuple(EDIT_RESEARCH_TREATMENTS)}"
            )
        treatment = {"name": normalized, **EDIT_RESEARCH_TREATMENTS[normalized]}
    treatment.setdefault("discrepancy_x0_scale", 0.0)
    treatment.setdefault("discrepancy_fraction", 0.20)
    treatment.setdefault("instruction_rank_mode", "hinge")
    treatment.setdefault("instruction_rank_temperature", 0.01)
    treatment.setdefault("instruction_rank_multiplier", 1.0)
    treatment.setdefault("instruction_negative_scope", "all")
    treatment.setdefault("discrepancy_sample_scope", "all")
    treatment.setdefault("temporal_scale", 0.0)
    treatment.setdefault("temporal_vector_weight", 0.5)
    treatment.setdefault("temporal_speed_weight", 0.5)
    treatment.setdefault("temporal_background_weight", 0.10)
    treatment.setdefault("temporal_change_scale_mps", 0.25)
    treatment.setdefault("temporal_smooth_l1_beta_mps", 0.10)
    treatment.setdefault("source_fusion_mode", "additive")
    if treatment["secondary_branch"] not in {
        "none",
        "shuffled_instruction",
        "same_source_instruction",
        "source_identity",
    }:
        raise ValueError("Unknown Edit research secondary branch")
    if treatment["source_fusion_mode"] not in {"additive", "token_block"}:
        raise ValueError("Unknown source fusion mode")
    if treatment["instruction_rank_mode"] not in {"hinge", "softplus"}:
        raise ValueError("Unknown Edit instruction rank mode")
    if treatment["instruction_negative_scope"] not in {"all", "same_source_only"}:
        raise ValueError("Unknown Edit instruction negative scope")
    if treatment["discrepancy_sample_scope"] not in {"all", "same_source_only"}:
        raise ValueError("Unknown Edit discrepancy sample scope")
    for field in (
        "representation_multiplier",
        "identity_base_scale",
        "source_anchor_scale",
        "source_anchor_relative_margin",
        "discrepancy_x0_scale",
        "instruction_rank_multiplier",
        "temporal_scale",
        "temporal_vector_weight",
        "temporal_speed_weight",
    ):
        value = float(treatment[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Edit research treatment {field} must be finite and non-negative")
    rank_temperature = float(treatment["instruction_rank_temperature"])
    if not math.isfinite(rank_temperature) or rank_temperature <= 0.0:
        raise ValueError("Edit research treatment rank temperature must be positive")
    discrepancy_fraction = float(treatment["discrepancy_fraction"])
    if not 0.0 < discrepancy_fraction <= 1.0:
        raise ValueError("Edit research treatment discrepancy_fraction must be in (0,1]")
    temporal_component_sum = float(treatment["temporal_vector_weight"]) + float(
        treatment["temporal_speed_weight"]
    )
    if temporal_component_sum <= 0.0:
        raise ValueError("Edit temporal component weights must have positive sum")
    temporal_background = float(treatment["temporal_background_weight"])
    if not math.isfinite(temporal_background) or not 0.0 <= temporal_background <= 1.0:
        raise ValueError("Edit temporal background weight must be in [0,1]")
    for field in ("temporal_change_scale_mps", "temporal_smooth_l1_beta_mps"):
        value = float(treatment[field])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Edit research treatment {field} must be finite and positive")
    return treatment

# Every value below changes the R11 experiment identity. Asset paths are bound
# separately by validate_assets(); output paths are operational rather than
# mathematical. Rejecting unknown section fields prevents a new knob from being
# silently added without deciding whether it belongs in this contract.
FROZEN_CONFIG_SECTION_FIELDS = {
    "contract": {"name", "version"},
    "data": {
        "manifest_dir",
        "train_manifest",
        "stats_root",
        "verify_payload_hash",
        "max_target_frames",
        "materialize_workers",
        "sort_window_batches",
    },
    "text": {
        "encoder",
        "cache_dir",
        "max_tokens",
        "ctxt_dim",
        "vtxt_dim",
        "max_open_shards",
        "strict_cache",
    },
    "model": {
        "hidden_dim",
        "num_heads",
        "root_depth_double",
        "root_depth_single",
        "body_depth_double",
        "body_depth_single",
        "mlp_ratio",
        "dropout",
        "detach_root_bridge",
        "self_conditioning",
        "stats_variance_eps",
    },
    "flow": {
        "prediction_type",
        "representation_space",
        "timestep_schedule",
        "timestep_mean",
        "timestep_std",
    },
    "loss": {
        "representation_scale",
        "semantic_weights",
        "contact",
        "clean_root_velocity",
        "clean_joint_velocity",
        "foot_lock",
        "fk_consistency",
        "fk_warmup_steps",
        "fk_scale_m",
        "control_continuous",
        "control_contact",
        "velocity_t_eps",
        "fps",
        "contact_threshold",
    },
    "control": {
        "mixed_prob",
        "max_sparse_keyframes",
        "curriculum_start_step",
        "curriculum_end_step",
        "dense_min_fraction",
        "endpoint_preset",
        "endpoint_subset_mode",
        "include_root_ref_for_endpoints",
        "include_endpoint_rotations",
        "include_contact_pattern",
    },
    "training": {
        "seed",
        "batch_size_per_rank",
        "precision",
        "gradient_clip",
        "weight_decay",
        "ema_decay",
        "ema_every",
        "log_every",
        "latest_every",
        "archive_every",
        "gradient_sync_mode",
        "gradient_sync_bucket_cap_mb",
        "output_dir",
        "max_global_step",
    },
    "production": {
        "nonregression_artifact",
        "nonregression_artifact_sha256",
        "asset_preflight_artifact",
        "asset_preflight_artifact_sha256",
    },
    "stage": {
        "name",
        "expected_start_step",
        "stop_step",
        "phase_id",
        "schedule_version",
    },
}

R12_FROZEN_CONFIG_SECTION_FIELDS = {
    section: set(fields) for section, fields in FROZEN_CONFIG_SECTION_FIELDS.items()
}
R12_FROZEN_CONFIG_SECTION_FIELDS["control"].add("root_heading_probability")
R13_FROZEN_CONFIG_SECTION_FIELDS = {
    section: set(fields) for section, fields in R12_FROZEN_CONFIG_SECTION_FIELDS.items()
}
R13_FROZEN_CONFIG_SECTION_FIELDS["flow"].add("contact_protocol")

UNIFIED_EDIT_OBJECTIVE_FIELDS = {
    "target_x0_scale",
    "hard_x0_scale",
    "hard_fraction",
    "instruction_rank_scale",
    "instruction_relative_margin",
}

FROZEN_CONFIG_VALUES = {
    "contract.name": "hy273_multitask_r11_v2",
    "contract.version": TRAIN_CONTRACT,
    "data.verify_payload_hash": False,
    "data.max_target_frames": 300,
    "data.materialize_workers": 4,
    "data.sort_window_batches": 8,
    "text.encoder": "hy_cache",
    "text.max_tokens": 128,
    "text.ctxt_dim": 4096,
    "text.vtxt_dim": 768,
    "text.max_open_shards": 64,
    "text.strict_cache": True,
    "model.hidden_dim": 1024,
    "model.num_heads": 8,
    "model.root_depth_double": 3,
    "model.root_depth_single": 6,
    "model.body_depth_double": 3,
    "model.body_depth_single": 6,
    "model.mlp_ratio": 2.0,
    "model.dropout": 0.0,
    "model.detach_root_bridge": True,
    "model.self_conditioning": False,
    "model.stats_variance_eps": 1.0e-5,
    "flow.prediction_type": "x0",
    "flow.representation_space": "velocity",
    "flow.timestep_schedule": "logit_normal",
    "flow.timestep_mean": -0.8,
    "flow.timestep_std": 0.8,
    "control.mixed_prob": 0.25,
    "control.max_sparse_keyframes": 20,
    "control.curriculum_start_step": 200_000,
    "control.curriculum_end_step": 400_000,
    "control.dense_min_fraction": 1.0,
    "control.endpoint_preset": "kimodo_ee",
    "control.endpoint_subset_mode": "random_nonempty",
    "control.include_root_ref_for_endpoints": True,
    "control.include_endpoint_rotations": True,
    "control.include_contact_pattern": True,
    "training.seed": 20_260_715,
    "training.batch_size_per_rank": 16,
    "training.precision": "bf16",
    "training.gradient_clip": 1.0,
    "training.weight_decay": 0.01,
    "training.ema_decay": 0.995,
    "training.ema_every": 10,
    "training.log_every": 20,
    "training.latest_every": 10_000,
    "training.archive_every": 50_000,
    "training.gradient_sync_mode": "fixed_bucket",
    "training.gradient_sync_bucket_cap_mb": 100.0,
    "training.max_global_step": 500_000,
    "production.nonregression_artifact": (
        "/mnt/afs/mogeflow-control/outside_doc/"
        "HY273_multitask_nonregression_baseline_v1.json"
    ),
    "production.nonregression_artifact_sha256": (
        "2332badc52da7c603c8550e43e432be43dd0f858e7bc88e62df6160c59db47c0"
    ),
    "production.asset_preflight_artifact": (
        "/mnt/afs/mogeflow-control/outputs/hy273_multitask/gates/"
        "full_asset_preflight_r11_v2.json"
    ),
    "production.asset_preflight_artifact_sha256": (
        "010bb74124219be811a1ae6caf8e81668e77165e39f3e9dc5150d32005b621cd"
    ),
}

R12_FROZEN_CONFIG_VALUES = dict(FROZEN_CONFIG_VALUES)
R12_FROZEN_CONFIG_VALUES.update(
    {
        "contract.name": "hy273_multitask_r12_rootmask_v1",
        "contract.version": R12_TRAIN_CONTRACT,
        "control.root_heading_probability": 0.5,
    }
)
R13_FROZEN_CONFIG_VALUES = dict(R12_FROZEN_CONFIG_VALUES)
R13_FROZEN_CONFIG_VALUES.update(
    {
        "contract.name": "hy273_multitask_r13_unified273_v1",
        "contract.version": R13_TRAIN_CONTRACT,
        "flow.contact_protocol": UNIFIED_273_CONTACT_PROTOCOL,
        "loss.contact": R13_UNIFIED_CONTACT_WEIGHT,
        "loss.control_contact": R13_UNIFIED_CONTROL_CONTACT_WEIGHT,
    }
)

FROZEN_STAGE_CONTRACTS = {
    "stage_a_t2m": (
        0,
        200_000,
        int(TrainingPhase.STAGE_A),
        HIGH_LEVEL_SCHEDULE_VERSION,
    ),
    "stage_b1_control_bootstrap": (
        200_000,
        250_000,
        int(TrainingPhase.STAGE_B1),
        HIGH_LEVEL_SCHEDULE_VERSION,
    ),
    "stage_b2_joint_adapt": (
        250_000,
        400_000,
        int(TrainingPhase.STAGE_B2),
        HIGH_LEVEL_SCHEDULE_VERSION,
    ),
    "stage_b_r16_fixed_control": (
        200_000,
        400_000,
        int(TrainingPhase.STAGE_B1),
        R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
    ),
    "stage_be_kencoder_edit": (
        200_000,
        250_000,
        int(TrainingPhase.STAGE_B1),
        KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    ),
    "stage_bc_kencoder_ease_control": (
        250_000,
        400_000,
        int(TrainingPhase.STAGE_B2),
        KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    ),
    "stage_c_consolidate": (
        400_000,
        500_000,
        int(TrainingPhase.STAGE_C),
        HIGH_LEVEL_SCHEDULE_VERSION,
    ),
    "stage_c_safe_mix_probe": (
        400_000,
        405_000,
        int(TrainingPhase.STAGE_C),
        STAGE_C_SAFE_MIX_SCHEDULE_VERSION,
    ),
    "stage_c_edit20_research": (
        400_000,
        500_000,
        int(TrainingPhase.STAGE_C),
        STAGE_C_EDIT20_SCHEDULE_VERSION,
    ),
    "stage_d_edit_condition_calibration_pilot": (
        500_000,
        510_000,
        int(TrainingPhase.STAGE_C),
        STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION,
    ),
    "stage_d_edit_condition_calibration_extend": (
        510_000,
        550_000,
        int(TrainingPhase.STAGE_C),
        STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION,
    ),
    "stage_c_unified_edit_v2": (
        400_000,
        500_000,
        int(TrainingPhase.STAGE_C),
        UNIFIED_EDIT_V2_SCHEDULE_VERSION,
    ),
    "stage_c_unified_edit_v2_edit40": (
        450_000,
        500_000,
        int(TrainingPhase.STAGE_C),
        UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION,
    ),
    "stage_c1_r13_unified_edit_v2": (
        400_000,
        450_000,
        int(TrainingPhase.STAGE_C),
        UNIFIED_EDIT_V2_SCHEDULE_VERSION,
    ),
    "stage_c1_r13_decomposed_cfg_edit": (
        400_000,
        450_000,
        int(TrainingPhase.STAGE_C),
        UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
    ),
    "stage_c2_r15_edit80_from_positive450k": (
        450_000,
        500_000,
        int(TrainingPhase.STAGE_C),
        UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION,
    ),
    "stage_c2_r13_unified_edit40": (
        450_000,
        500_000,
        int(TrainingPhase.STAGE_C),
        UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION,
    ),
}

CODE_IDENTITY_FILES = (
    "train_hy273_multitask.py",
    "common/fixed_bucket_ddp.py",
    "data/hy273_multitask_batcher.py",
    "data/hy273_multitask_manifest_dataset.py",
    "data/hy273_multitask_scheduler.py",
    "models/codeflow/dit_blocks.py",
    "models/raw_motion/flow_schedule.py",
    "models/raw_motion/hy273_constraints.py",
    "models/raw_motion/hy273_unified_edit_losses.py",
    "models/raw_motion/hy273_multitask_condition.py",
    "models/raw_motion/hy273_multitask_losses.py",
    "models/raw_motion/hy273_normalizer.py",
    "models/raw_motion/hy273_slices.py",
    "models/raw_motion/hytext_cache.py",
    "models/raw_motion/llm2vec_cache.py",
    "models/raw_motion/kimodo_prefix_transformer.py",
    "models/raw_motion/kimodo_context_flow_dit.py",
    "models/raw_motion/kimodo_like_flow_dit.py",
    "models/raw_motion/raw_flow_dit.py",
    "sample_hy273_multitask.py",
    "tools/preflight_hy273_multitask_assets.py",
)

TEXT_FUSION_IMPLEMENTATION_FILES = (
    "train_hy273_multitask.py",
    "models/codeflow/dit_blocks.py",
    "models/raw_motion/kimodo_context_flow_dit.py",
    "models/raw_motion/kimodo_like_flow_dit.py",
    "models/raw_motion/raw_flow_dit.py",
)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = value
    return output


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    import yaml

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Training config is missing: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    base_name = payload.get("base_config")
    if base_name:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = resolved.parent / base_path
        base, _ = load_config(base_path)
        payload = deep_merge(base, payload)
    return payload, resolved


def cfg_get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Required config field is missing: {dotted}")
        value = value[key]
    return value


def contact_protocol_for_config(config: dict[str, Any]) -> str:
    protocol = config.get("flow", {}).get(
        "contact_protocol", LEGACY_SPLIT_CONTACT_PROTOCOL
    )
    uses_unified_273_flow(str(protocol))
    return str(protocol)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def base_contract_sha(config: dict[str, Any]) -> str:
    """Hash model/data invariants while excluding stage-local objectives."""

    payload = {
        key: value
        for key, value in config.items()
        if key not in {"stage", "edit_objective", "ease"}
    }
    payload = json.loads(json.dumps(payload))
    payload["training"].pop("output_dir", None)
    return canonical_sha(payload)


def current_code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = {name: sha256_file(root / name) for name in CODE_IDENTITY_FILES}
    return {"files": files, "identity_sha256": canonical_sha(files)}


def validate_resume_code_identity(
    checkpoint: dict[str, Any],
    current: dict[str, Any],
    *,
    allow_r12_b1_sampler_cfg_migration: bool,
) -> str:
    """Allow the known post-B1 sampler CFG-default change without rewriting state."""

    previous = checkpoint.get("code_identity")
    if previous == current:
        return "exact"
    if not allow_r12_b1_sampler_cfg_migration:
        raise RuntimeError("Checkpoint/current code identity mismatch")
    if not (
        checkpoint.get("train_contract") == R12_TRAIN_CONTRACT
        and checkpoint.get("next_global_step") == 250_000
        and isinstance(previous, dict)
        and previous.get("identity_sha256") == R12_B1_250K_CODE_IDENTITY_SHA256
    ):
        raise RuntimeError(
            "Sampler-CFG resume migration is restricted to the known R12 B1 250K checkpoint"
        )
    previous_files = previous.get("files")
    current_files = current.get("files")
    if not isinstance(previous_files, dict) or not isinstance(current_files, dict):
        raise RuntimeError("Malformed code identity during sampler-CFG resume migration")
    if previous_files.get("sample_hy273_multitask.py") != R12_B1_250K_SAMPLER_SHA256:
        raise RuntimeError("R12 B1 checkpoint has an unexpected sampler identity")

    # This file necessarily changes to add the compatibility path. Everything
    # defining data, model, loss, scheduling, and DDP behavior must still match.
    allowed = {"train_hy273_multitask.py", "sample_hy273_multitask.py"}
    mismatches = {
        name
        for name in set(previous_files) | set(current_files)
        if previous_files.get(name) != current_files.get(name)
    }
    if not mismatches or not mismatches.issubset(allowed):
        raise RuntimeError(
            "Sampler-CFG resume migration found training-code changes: "
            f"{sorted(mismatches - allowed)}"
        )
    return "r12_b1_250k_sampler_cfg_defaults_2p0"


def describe_research_resume_code_identity(
    checkpoint: dict[str, Any], current: dict[str, Any]
) -> str:
    """Report code drift without making a research resume depend on file hashes."""

    previous = checkpoint.get("code_identity")
    if previous == current:
        return "exact"
    previous_files = previous.get("files") if isinstance(previous, dict) else None
    current_files = current.get("files")
    if not isinstance(previous_files, dict) or not isinstance(current_files, dict):
        return "research_code_identity_unavailable"
    mismatches = sorted(
        name
        for name in set(previous_files) | set(current_files)
        if previous_files.get(name) != current_files.get(name)
    )
    return "research_code_drift:" + ",".join(mismatches)


def validate_research_resume_objective(
    checkpoint_runtime: dict[str, Any] | None,
    current_overrides: dict[str, Any],
) -> None:
    checkpoint_overrides = (
        checkpoint_runtime.get("research_overrides")
        if isinstance(checkpoint_runtime, dict)
        else None
    )
    objective_keys = (
        "research_treatment",
        "edit_objective",
        "same_source_instruction_donors",
        "base_representation_loss_space",
        "base_contact_loss_space",
    )

    def normalized_objective(overrides: Any) -> dict[str, Any] | None:
        if not isinstance(overrides, dict):
            return None
        objective = {
            key: overrides[key] for key in objective_keys if key in overrides
        }
        objective.setdefault("base_representation_loss_space", "velocity_mse")
        objective.setdefault("base_contact_loss_space", "velocity_mse")
        return objective

    checkpoint_objective = normalized_objective(checkpoint_overrides)
    current_objective = normalized_objective(current_overrides)
    if checkpoint_objective != current_objective:
        raise RuntimeError(
            "Research objective changed across resume: "
            f"checkpoint={checkpoint_objective!r}, requested={current_objective!r}"
        )


def validate_stage_c_schedule_fork_code_identity(
    checkpoint: dict[str, Any], current: dict[str, Any]
) -> str:
    """Allow only the code files required by the registered Stage-C mix fork."""

    if (
        checkpoint.get("train_contract") != R12_TRAIN_CONTRACT
        or int(checkpoint.get("next_global_step", -1)) != 400_000
    ):
        raise RuntimeError("Stage-C schedule fork requires the R12 400K checkpoint")
    previous = checkpoint.get("code_identity")
    if not isinstance(previous, dict):
        raise RuntimeError("Stage-C schedule fork parent has no code identity")
    previous_files = previous.get("files")
    current_files = current.get("files")
    if not isinstance(previous_files, dict) or not isinstance(current_files, dict):
        raise RuntimeError("Malformed code identity for Stage-C schedule fork")
    allowed = {
        "train_hy273_multitask.py",
        "data/hy273_multitask_batcher.py",
        "data/hy273_multitask_manifest_dataset.py",
        "data/hy273_multitask_scheduler.py",
    }
    mismatches = {
        name
        for name in set(previous_files) | set(current_files)
        if previous_files.get(name) != current_files.get(name)
    }
    if not mismatches or not mismatches.issubset(allowed):
        raise RuntimeError(
            "Stage-C schedule fork has unrelated code changes: "
            f"{sorted(mismatches - allowed)}"
        )
    if "data/hy273_multitask_scheduler.py" not in mismatches:
        raise RuntimeError("Stage-C schedule fork did not change the scheduler")
    return "r12_stage_c_400k_compatible_code_v1"


def validate_unified_edit40_fork_code_identity(
    checkpoint: dict[str, Any], current: dict[str, Any]
) -> str:
    """Restrict the 450K Edit40 fork to its scheduler-only training change."""

    if (
        checkpoint.get("train_contract") != R12_TRAIN_CONTRACT
        or int(checkpoint.get("next_global_step", -1)) != 450_000
        or checkpoint.get("high_level_schedule_version")
        != UNIFIED_EDIT_V2_SCHEDULE_VERSION
    ):
        raise RuntimeError(
            "Unified Edit40 fork requires the R12 Unified Edit V2 450K checkpoint"
        )
    previous = checkpoint.get("code_identity")
    if not isinstance(previous, dict):
        raise RuntimeError("Unified Edit40 fork parent has no code identity")
    previous_files = previous.get("files")
    current_files = current.get("files")
    if not isinstance(previous_files, dict) or not isinstance(current_files, dict):
        raise RuntimeError("Malformed code identity for Unified Edit40 fork")
    allowed = {
        "train_hy273_multitask.py",
        "data/hy273_multitask_scheduler.py",
    }
    mismatches = {
        name
        for name in set(previous_files) | set(current_files)
        if previous_files.get(name) != current_files.get(name)
    }
    if not mismatches or not mismatches.issubset(allowed):
        raise RuntimeError(
            "Unified Edit40 fork has unrelated training-code changes: "
            f"{sorted(mismatches - allowed)}"
        )
    if "data/hy273_multitask_scheduler.py" not in mismatches:
        raise RuntimeError("Unified Edit40 fork did not change the scheduler")
    return "r12_unified_edit40_450k_compatible_code_v1"


def validate_unified_edit40_objective(
    parent_config: dict[str, Any], current_config: dict[str, Any]
) -> None:
    """Keep the Edit objective fixed when only the task mix is being changed."""

    if parent_config.get("edit_objective") != current_config.get("edit_objective"):
        raise RuntimeError("Unified Edit40 fork changed the Edit objective")


def _verified_file(path: Any, expected_sha: Any, label: str) -> Path:
    if not path or not expected_sha:
        raise RuntimeError(f"Production gate {label} is missing path/SHA")
    resolved = Path(str(path)).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Production gate {label} is missing: {resolved}")
    actual = sha256_file(resolved)
    if actual != str(expected_sha).lower():
        raise RuntimeError(
            f"Production gate {label} SHA mismatch: expected={expected_sha}, actual={actual}"
        )
    return resolved


def validate_production_gate(
    config: dict[str, Any], cli_artifact: str = ""
) -> dict[str, Any]:
    configured = Path(cfg_get(config, "production.nonregression_artifact")).resolve()
    if cli_artifact and Path(cli_artifact).expanduser().resolve() != configured:
        raise RuntimeError("Production non-regression artifact path is frozen by config")
    artifact_path = _verified_file(
        configured,
        cfg_get(config, "production.nonregression_artifact_sha256"),
        "bundle",
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("format") != "hy273_multitask_nonregression_baseline_v1":
        raise RuntimeError("Production non-regression bundle format mismatch")
    if artifact.get("status") != "ready":
        raise RuntimeError(
            f"Production gate is not ready: {artifact_path} status={artifact.get('status')!r}"
        )

    for checkpoint_name in ("stage1", "stage2_complete"):
        item = artifact.get(checkpoint_name, {})
        _verified_file(item.get("checkpoint_path"), item.get("checkpoint_sha256"), checkpoint_name)
        _verified_file(item.get("config_path"), item.get("config_sha256"), f"{checkpoint_name}.config")

    required = artifact.get("required_gate_artifacts", {})
    if required.get("status") != "validated":
        raise RuntimeError("Production required_gate_artifacts are not validated")
    for name in (
        "gate_matrix",
        "paired_bootstrap",
        "t2m_artifact",
        "comparator_self_test",
    ):
        item = required.get(name, {})
        if item.get("status") != "validated":
            raise RuntimeError(f"Production gate artifact {name} is not validated")
        if item.get("row_count") is None or item.get("case_count") is None:
            raise RuntimeError(f"Production gate artifact {name} lacks counts")
        expected_count = item.get("expected_case_count")
        if expected_count is not None and int(item["case_count"]) != int(expected_count):
            raise RuntimeError(f"Production gate artifact {name} case count mismatch")
        _verified_file(item.get("path"), item.get("sha256"), name)
        if name == "comparator_self_test":
            _verified_file(
                item.get("comparator_code_path"),
                item.get("comparator_code_sha256"),
                "comparator code",
            )
        identity = item.get("protocol_identity")
        if not isinstance(identity, dict) or any(value is None for value in identity.values()):
            raise RuntimeError(f"Production gate artifact {name} has incomplete protocol identity")
    return {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "format": artifact["format"],
    }


def validate_run_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(name)):
        raise ValueError("--name must be a safe basename")
    return str(name)


def setup_distributed() -> tuple[torch.device, int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl", device_id=device)
        return device, dist.get_rank(), dist.get_world_size(), local_rank
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, -1


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def globally_normalized_eligible_denominator(
    sample_mask: torch.Tensor,
    *,
    world_size: int,
) -> torch.Tensor:
    """Return the per-rank divisor whose DDP average is a global sample mean."""

    if sample_mask.ndim != 1:
        raise ValueError("sample_mask must have shape [B]")
    count = sample_mask.to(dtype=torch.float32).sum()
    if is_distributed():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    per_rank = count / float(world_size)
    return torch.where(count > 0, per_rank, torch.ones_like(per_rank))


def contiguous_ratio_rank_groups(
    world_size: int, group_size: int
) -> tuple[tuple[int, ...], ...]:
    """Return fixed physical-rank groups for one preserved ratio partition."""

    world = int(world_size)
    size = int(group_size)
    if world <= 0 or size <= 0 or world % size != 0:
        raise ValueError("ratio group size must divide the distributed world size")
    return tuple(
        tuple(range(start, start + size)) for start in range(0, world, size)
    )


def create_local_ratio_process_group(
    *, world_size: int, rank: int, group_size: int
) -> Any | None:
    """Create all fixed ratio groups and return the group containing this rank."""

    if int(group_size) == 1:
        return None
    if not is_distributed():
        raise RuntimeError("Multi-rank ratio partitions require distributed training")
    local_group = None
    for ranks in contiguous_ratio_rank_groups(world_size, group_size):
        process_group = dist.new_group(ranks=list(ranks))
        if int(rank) in ranks:
            local_group = process_group
    if local_group is None:
        raise RuntimeError("Current rank was not assigned to a ratio process group")
    return local_group


def apply_preserved_ratio_partition(
    bundle: HY273MultitaskLossBundle,
    *,
    process_group: Any | None,
    group_size: int,
) -> HY273MultitaskLossBundle:
    """Make several physical ranks reproduce one parent rank-local ratio.

    Each local numerator is scaled by the physical group size and divided by
    the partition's summed denominator. The later world-size gradient average
    then equals the parent's average over its larger rank-local partitions.
    """

    size = int(group_size)
    if size == 1:
        return bundle
    if process_group is None or not is_distributed():
        raise RuntimeError("Preserved ratio loss requires its distributed group")
    terms = tuple(bundle.terms.values())
    denominators = torch.stack(
        [term.denominator.detach().to(dtype=torch.float32) for term in terms]
    )
    dist.all_reduce(denominators, op=dist.ReduceOp.SUM, group=process_group)
    for term, denominator in zip(terms, denominators):
        term.backward_denominator = denominator
        term.backward_numerator_scale = float(size)
    bundle.total = sum(
        (term.weighted for term in terms),
        bundle.total * 0.0,
    )
    return bundle


def validate_exact_kencoder_stage_b_reshard(
    checkpoint: dict[str, Any],
    *,
    current_world_size: int,
    current_batch_size: int,
) -> None:
    """Close the formal K-Encoder 4x32 -> 8x16 boundary contract."""

    if int(checkpoint.get("next_global_step", -1)) != 200_000:
        raise ValueError("K-Encoder Stage-B reshard requires the exact 200K checkpoint")
    if checkpoint.get("high_level_schedule_version") != HIGH_LEVEL_SCHEDULE_VERSION:
        raise ValueError("K-Encoder Stage-B reshard requires a Stage-A scheduler")
    batcher_state = checkpoint.get("batcher")
    if not isinstance(batcher_state, dict):
        raise ValueError("K-Encoder Stage-B checkpoint has no batcher state")
    source_world_size = int(batcher_state.get("world_size", 0))
    source_batch_size = int(batcher_state.get("batch_size_per_rank", 0))
    if (source_world_size, source_batch_size) != (4, 32):
        raise ValueError(
            "K-Encoder Stage-B source topology must be exactly 4x32"
        )
    if (int(current_world_size), int(current_batch_size)) != (8, 16):
        raise ValueError(
            "K-Encoder Stage-B destination topology must be exactly 8x16"
        )
    ratio_partition = batcher_state.get("ratio_partition")
    if ratio_partition is not None and (
        not isinstance(ratio_partition, dict)
        or int(ratio_partition.get("world_size", 0)) != 4
        or int(ratio_partition.get("batch_size", 0)) != 32
    ):
        raise ValueError(
            "K-Encoder Stage-A checkpoint has an unexpected ratio partition"
        )


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def seed_model_initialization(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def cuda_rng_states_for_resume(
    saved_states: Sequence[torch.Tensor],
    *,
    device_count: int,
    allow_same_global_batch_reshard: bool,
) -> list[torch.Tensor]:
    """Resize legacy CUDA RNG snapshots only for an explicit topology reshard."""

    count = int(device_count)
    if count < 0:
        raise ValueError("device_count cannot be negative")
    states = list(saved_states)
    if len(states) == count:
        return states
    if not allow_same_global_batch_reshard:
        raise RuntimeError(
            "CUDA RNG topology changed across resume: "
            f"checkpoint={len(states)}, runtime={count}"
        )
    if count == 0:
        return []
    if not states:
        raise RuntimeError("Cannot reshard an empty CUDA RNG snapshot")
    if len(states) > count:
        return states[:count]
    return states + [states[-1].clone() for _ in range(count - len(states))]


def validate_frozen_contract(config: dict[str, Any]) -> HY273MultitaskLossWeights:
    edit_objective = config.get("edit_objective")
    base_config = {
        key: value for key, value in config.items() if key != "edit_objective"
    }
    if edit_objective is not None:
        if not isinstance(edit_objective, dict) or set(edit_objective) != UNIFIED_EDIT_OBJECTIVE_FIELDS:
            raise ValueError("Invalid unified Edit objective config schema")
        UnifiedEditLossWeights(
            **{key: float(value) for key, value in edit_objective.items()}
        ).validate()
    contract_name = str(cfg_get(config, "contract.name"))
    stage_name = str(cfg_get(config, "stage.name"))
    if contract_name == "hy273_multitask_r11_v2":
        expected_sections = FROZEN_CONFIG_SECTION_FIELDS
        expected_values = FROZEN_CONFIG_VALUES
        contract_label = "R11"
    elif contract_name == "hy273_multitask_r12_rootmask_v1":
        expected_sections = R12_FROZEN_CONFIG_SECTION_FIELDS
        expected_values = dict(R12_FROZEN_CONFIG_VALUES)
        if stage_name.startswith("stage_d_edit_condition_calibration_"):
            expected_values["training.max_global_step"] = 550_000
        contract_label = "R12 root-mask"
    elif contract_name == "hy273_multitask_r13_unified273_v1":
        expected_sections = {
            section: set(fields)
            for section, fields in R13_FROZEN_CONFIG_SECTION_FIELDS.items()
        }
        expected_values = dict(R13_FROZEN_CONFIG_VALUES)
        if stage_name == "stage_bc_kencoder_ease_control":
            expected_sections["ease"] = {"enabled", "stats_dir"}
            expected_values["ease.enabled"] = True
        contract_label = "R13 unified-273"
    else:
        raise ValueError(f"Unknown HY273 multitask contract: {contract_name!r}")

    if set(base_config) != set(expected_sections):
        raise ValueError(
            f"Resolved {contract_label} config sections changed: "
            f"actual={sorted(base_config)} expected={sorted(expected_sections)}"
        )
    for section, expected_fields in expected_sections.items():
        value = base_config.get(section)
        if not isinstance(value, dict) or set(value) != expected_fields:
            actual = sorted(value) if isinstance(value, dict) else type(value).__name__
            raise ValueError(
                f"Resolved {contract_label} config fields changed for {section}: "
                f"actual={actual} expected={sorted(expected_fields)}"
            )
    for key, expected in expected_values.items():
        actual = cfg_get(config, key)
        if actual != expected:
            raise ValueError(
                f"Frozen {contract_label} field changed: {key}={actual!r}, expected={expected!r}"
            )
    if stage_name not in FROZEN_STAGE_CONTRACTS:
        raise ValueError(f"Unknown frozen {contract_label} stage: {stage_name!r}")
    actual_stage = (
        int(cfg_get(config, "stage.expected_start_step")),
        int(cfg_get(config, "stage.stop_step")),
        int(cfg_get(config, "stage.phase_id")),
        str(cfg_get(config, "stage.schedule_version")),
    )
    if actual_stage != FROZEN_STAGE_CONTRACTS[stage_name]:
        raise ValueError(
            f"Frozen {contract_label} stage changed: {stage_name}={actual_stage}, "
            f"expected={FROZEN_STAGE_CONTRACTS[stage_name]}"
        )
    if [float(x) for x in cfg_get(config, "loss.semantic_weights")] != [
        10.0,
        2.0,
        10.0,
        10.0,
        3.0,
    ]:
        raise ValueError("Frozen semantic block weights changed")
    weights = HY273MultitaskLossWeights(
        representation_scale=float(cfg_get(config, "loss.representation_scale")),
        contact=float(cfg_get(config, "loss.contact")),
        clean_root_velocity=float(cfg_get(config, "loss.clean_root_velocity")),
        clean_joint_velocity=float(cfg_get(config, "loss.clean_joint_velocity")),
        foot_lock=float(cfg_get(config, "loss.foot_lock")),
        fk_consistency=float(cfg_get(config, "loss.fk_consistency")),
        control_continuous=float(cfg_get(config, "loss.control_continuous")),
        control_contact=float(cfg_get(config, "loss.control_contact")),
        velocity_t_eps=float(cfg_get(config, "loss.velocity_t_eps")),
        fk_warmup_steps=int(cfg_get(config, "loss.fk_warmup_steps")),
        fk_scale_m=float(cfg_get(config, "loss.fk_scale_m")),
        fps=float(cfg_get(config, "loss.fps")),
        contact_threshold=float(cfg_get(config, "loss.contact_threshold")),
    )
    frozen = HY273MultitaskLossWeights()
    if contract_name == "hy273_multitask_r13_unified273_v1":
        # Kimodo uses gamma_contact=4 against continuous semantic weights
        # 10/2/10/10/3. Preserve the calibrated continuous aggregate scale.
        frozen = replace(
            frozen,
            contact=R13_UNIFIED_CONTACT_WEIGHT,
            control_contact=R13_UNIFIED_CONTROL_CONTACT_WEIGHT,
        )
    if weights != frozen:
        raise ValueError(
            f"Resolved loss contract differs from the shared baseline: {weights} != {frozen}"
        )
    return weights


def validate_assets(
    config: dict[str, Any],
    *,
    include_full_preflight: bool = False,
    text_cache_dir: str | Path | None = None,
    expected_text_cache_format: str = PROFILE_CACHE_FORMAT,
) -> dict[str, Any]:
    manifest_dir = Path(cfg_get(config, "data.manifest_dir")).expanduser().resolve()
    train_manifest = Path(cfg_get(config, "data.train_manifest")).expanduser().resolve()
    stats_root = Path(cfg_get(config, "data.stats_root")).expanduser().resolve()
    text_cache = Path(
        text_cache_dir or cfg_get(config, "text.cache_dir")
    ).expanduser().resolve()
    required = [
        manifest_dir / "manifest.sha256",
        manifest_dir / "summary.json",
        train_manifest,
        stats_root / "manifest.json",
        stats_root / "full" / "Mean.npy",
        stats_root / "full" / "Std.npy",
        stats_root / "local_root" / "Mean.npy",
        stats_root / "local_root" / "Std.npy",
        text_cache / "manifest.json",
        text_cache / "index.json",
        text_cache / "coverage_report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required training assets are missing: {missing}")
    summary = json.loads((manifest_dir / "summary.json").read_text(encoding="utf-8"))
    bundle_files = {
        name: sha256_file(manifest_dir / name)
        for name in sorted((*summary["files"].keys(), "schema.json", "summary.json"))
    }
    bundle_sha = canonical_sha(bundle_files)
    recorded_bundle_sha = (manifest_dir / "manifest.sha256").read_text().strip()
    if bundle_sha != recorded_bundle_sha:
        raise RuntimeError(
            f"Unified manifest bundle SHA mismatch: {bundle_sha}/{recorded_bundle_sha}"
        )
    for name, record in summary["files"].items():
        if bundle_files[name] != record.get("sha256"):
            raise RuntimeError(f"Unified manifest member SHA mismatch: {name}")
    train_sha = sha256_file(train_manifest)
    if summary["files"]["train.jsonl"]["sha256"] != train_sha:
        raise RuntimeError("Unified train manifest SHA does not match summary")
    stats_manifest = json.loads((stats_root / "manifest.json").read_text(encoding="utf-8"))
    unified_273 = uses_unified_273_flow(contact_protocol_for_config(config))
    expected_stats_format = (
        "hy273_multitask_target_stats_v2_unified273"
        if unified_273
        else "hy273_multitask_target_stats_v1"
    )
    if stats_manifest.get("format") != expected_stats_format:
        raise ValueError("Joint stats format mismatch")
    if stats_manifest.get("train_manifest_sha256") != train_sha:
        raise RuntimeError("Joint stats were not built from this train manifest")
    if not bool(stats_manifest.get("target_only")) or bool(stats_manifest.get("source_in_stats")):
        raise RuntimeError("Joint stats violate target-only contract")
    stats_array_sha = {}
    for relative, expected in stats_manifest.get("array_sha256", {}).items():
        actual = sha256_file(stats_root / relative)
        if actual != expected:
            raise RuntimeError(f"Joint stats array SHA mismatch: {relative}")
        stats_array_sha[relative] = actual
    if len(stats_array_sha) != 8:
        raise RuntimeError("Joint stats manifest must bind all eight canonical arrays")
    cache_manifest = json.loads((text_cache / "manifest.json").read_text(encoding="utf-8"))
    coverage = json.loads((text_cache / "coverage_report.json").read_text(encoding="utf-8"))
    if cache_manifest.get("format") != expected_text_cache_format:
        raise ValueError(
            "Text cache format mismatch: "
            f"expected={expected_text_cache_format!r}, "
            f"actual={cache_manifest.get('format')!r}"
        )
    if not bool(coverage.get("passed")):
        raise RuntimeError("HYText profile coverage report did not pass")
    if coverage.get("required_manifest_sha256") != cache_manifest.get(
        "required_manifest_sha256"
    ):
        raise RuntimeError("HYText coverage/manifest source hashes differ")
    cache_index_sha = sha256_file(text_cache / "index.json")
    if cache_manifest.get("index_sha256") != cache_index_sha:
        raise RuntimeError("HYText index SHA does not match cache manifest")
    if coverage.get("cache_index_sha256") != cache_index_sha:
        raise RuntimeError("HYText index SHA does not match coverage report")
    profile_rows_sha = sha256_file(text_cache / "profile_rows.jsonl")
    if cache_manifest.get("profile_rows_sha256") != profile_rows_sha:
        raise RuntimeError("HYText profile-row SHA mismatch")
    mean = np.load(stats_root / "full" / "Mean.npy")
    std = np.load(stats_root / "full" / "Std.npy")
    if unified_273:
        if not bool(stats_manifest.get("contacts_normalized")):
            raise RuntimeError("R13 stats must normalize contact channels")
        if not (
            np.isfinite(mean[CONTACT_SLICE]).all()
            and np.isfinite(std[CONTACT_SLICE]).all()
            and (std[CONTACT_SLICE] > 1e-3).all()
            and (std[CONTACT_SLICE] < 1.0).all()
        ):
            raise RuntimeError("R13 contact statistics are invalid")
    elif not (
        not bool(stats_manifest.get("contacts_normalized"))
        and np.array_equal(mean[CONTACT_SLICE], np.zeros(4, dtype=mean.dtype))
        and np.array_equal(std[CONTACT_SLICE], np.ones(4, dtype=std.dtype))
    ):
        raise RuntimeError("Legacy contact statistics must remain raw 0/1")
    identity = {
        "manifest_bundle_sha256": bundle_sha,
        "manifest_bundle_members_sha256": canonical_sha(bundle_files),
        "train_manifest_sha256": train_sha,
        "stats_manifest_sha256": sha256_file(stats_root / "manifest.json"),
        "hytext_manifest_sha256": sha256_file(text_cache / "manifest.json"),
        "stats_arrays_sha256": canonical_sha(stats_array_sha),
        "hytext_index_sha256": cache_index_sha,
        "hytext_profile_rows_sha256": profile_rows_sha,
        "hytext_coverage_sha256": sha256_file(text_cache / "coverage_report.json"),
    }
    identity["identity_sha256"] = canonical_sha(identity)
    if not include_full_preflight:
        return identity
    preflight_path = _verified_file(
        cfg_get(config, "production.asset_preflight_artifact"),
        cfg_get(config, "production.asset_preflight_artifact_sha256"),
        "full asset preflight",
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("format") != "hy273_multitask_full_asset_preflight_v1":
        raise RuntimeError("Full asset preflight format mismatch")
    if preflight.get("status") != "passed":
        raise RuntimeError("Full asset preflight did not pass")
    if preflight.get("asset_identity") != identity:
        raise RuntimeError("Full asset preflight/data identity mismatch")
    report_content = {
        key: value
        for key, value in preflight.items()
        if key != "report_content_sha256"
    }
    if preflight.get("report_content_sha256") != canonical_sha(report_content):
        raise RuntimeError("Full asset preflight embedded content SHA mismatch")
    scanner_path = Path(__file__).resolve().parent / "tools/preflight_hy273_multitask_assets.py"
    if preflight.get("scanner_code_sha256") != sha256_file(scanner_path):
        raise RuntimeError("Full asset preflight scanner code has changed")
    if int(preflight.get("unique_k273_payloads", 0)) <= 0:
        raise RuntimeError("Full asset preflight has no verified K273 payloads")
    payload_sha = str(preflight.get("payload_records_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", payload_sha):
        raise RuntimeError("Full asset preflight payload digest is invalid")
    base_identity_sha = identity["identity_sha256"]
    identity.update(
        {
            "base_identity_sha256": base_identity_sha,
            "full_asset_preflight_sha256": sha256_file(preflight_path),
            "full_asset_payload_records_sha256": payload_sha,
            "full_asset_unique_k273_payloads": int(preflight["unique_k273_payloads"]),
        }
    )
    identity["identity_sha256"] = canonical_sha(
        {key: value for key, value in identity.items() if key != "identity_sha256"}
    )
    return identity


def validate_ease_stats(config: dict[str, Any]) -> dict[str, Any] | None:
    """Bind Ease normalization to its scientific data semantics."""

    ease = config.get("ease")
    if not isinstance(ease, dict) or not bool(ease.get("enabled", False)):
        return None
    root = Path(str(ease.get("stats_dir", ""))).expanduser().resolve()
    mean_path = root / "Mean.npy"
    std_path = root / "Std.npy"
    metadata_path = root / "metadata.json"
    missing = [
        str(path)
        for path in (mean_path, std_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Required Ease stats assets are missing: {missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != EASE_STATS_FORMAT:
        raise ValueError("Ease stats format mismatch")
    if int(metadata.get("feature_dim", -1)) != EASE_DIM:
        raise ValueError("Ease stats feature_dim mismatch")
    source_manifest = Path(
        str(metadata.get("source_manifest", ""))
    ).expanduser().resolve()
    train_manifest = Path(cfg_get(config, "data.train_manifest")).expanduser().resolve()
    if source_manifest != train_manifest:
        raise RuntimeError(
            "Ease stats were not built from the configured train manifest"
        )
    if metadata.get("split") != "train":
        raise RuntimeError("Ease stats must be computed from the train split")
    if metadata.get("dataset") != "humanml3d_k273":
        raise RuntimeError("Ease stats must use the HumanML3D K273 target distribution")
    for field in ("row_count", "caption_occurrences", "unique_target_assets"):
        if int(metadata.get(field, 0)) <= 0:
            raise RuntimeError(f"Ease stats metadata has invalid {field}")

    mean = np.asarray(np.load(mean_path), dtype=np.float32)
    std = np.asarray(np.load(std_path), dtype=np.float32)
    if mean.shape != (EASE_DIM,) or std.shape != (EASE_DIM,):
        raise ValueError("Ease Mean/Std must each have shape [6]")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Ease Mean/Std must be finite with positive Std")

    return {
        "format": str(metadata["format"]),
        "feature_dim": EASE_DIM,
        "dataset": str(metadata["dataset"]),
        "split": str(metadata["split"]),
        "source_manifest": str(source_manifest),
        "row_count": int(metadata["row_count"]),
        "caption_occurrences": int(metadata["caption_occurrences"]),
        "unique_target_assets": int(metadata["unique_target_assets"]),
        "physical_label": str(metadata.get("physical_label", "")),
        "weighting": str(metadata.get("weighting", "")),
        "yaw_statistics": str(metadata.get("yaw_statistics", "")),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def assert_model_ease_stats(
    model: HY273KimodoContextFlow,
    identity: dict[str, Any] | None,
) -> None:
    conditioner = model.ease_conditioner
    if identity is None:
        if conditioner is not None:
            raise RuntimeError("Ease-disabled run unexpectedly constructed an Ease conditioner")
        return
    if conditioner is None:
        raise RuntimeError("Ease-enabled run has no Ease conditioner")
    expected_mean = torch.tensor(
        identity["mean"],
        device=conditioner.normalizer.mean.device,
        dtype=conditioner.normalizer.mean.dtype,
    )
    expected_std = torch.tensor(
        identity["std"],
        device=conditioner.normalizer.std.device,
        dtype=conditioner.normalizer.std.dtype,
    )
    if not torch.equal(conditioner.normalizer.mean, expected_mean) or not torch.equal(
        conditioner.normalizer.std, expected_std
    ):
        raise RuntimeError(
            "Model Ease normalization buffers differ from the configured Ease stats"
        )


def create_model(
    config: dict[str, Any],
    *,
    source_fusion_mode: str = "additive",
    text_global_conditioning: str = "pooled_adaln",
    text_fusion_mode: str = "f00",
    conditioning_architecture: str = "hytext_flux",
    llm2vec_cache_dir: str = "",
) -> HY273KimodoContextFlow:
    conditioning_architecture = str(conditioning_architecture)
    if conditioning_architecture not in CONDITIONING_ARCHITECTURES:
        raise ValueError(
            "Unknown conditioning architecture "
            f"{conditioning_architecture!r}"
        )
    if conditioning_architecture == "hytext_flux":
        text_encoder = str(cfg_get(config, "text.encoder"))
        text_cache_dir = str(cfg_get(config, "text.cache_dir"))
        text_max_tokens = int(cfg_get(config, "text.max_tokens"))
        text_ctxt_dim = int(cfg_get(config, "text.ctxt_dim"))
        text_vtxt_dim = int(cfg_get(config, "text.vtxt_dim"))
        backbone_type = "flux"
    else:
        if not llm2vec_cache_dir:
            raise ValueError(
                f"{conditioning_architecture} requires --llm2vec_cache_dir"
            )
        text_encoder = "llm2vec_cache"
        text_cache_dir = str(Path(llm2vec_cache_dir).expanduser().resolve())
        text_max_tokens = 1
        text_ctxt_dim = 4096
        text_vtxt_dim = 1
        backbone_type = (
            "flux"
            if conditioning_architecture == "llm2vec_flux"
            else "kimodo_prefix"
        )
    stats_root = Path(cfg_get(config, "data.stats_root"))
    normalize_contacts = uses_unified_273_flow(contact_protocol_for_config(config))
    ease_config = config.get("ease", {})
    use_ease = bool(ease_config.get("enabled", False))
    ease_stats_dir = str(ease_config.get("stats_dir", ""))
    if use_ease and not ease_stats_dir:
        raise ValueError("Ease-enabled model requires ease.stats_dir")
    return HY273KimodoContextFlow(
        hidden_dim=int(cfg_get(config, "model.hidden_dim")),
        num_heads=int(cfg_get(config, "model.num_heads")),
        root_depth_double=int(cfg_get(config, "model.root_depth_double")),
        root_depth_single=int(cfg_get(config, "model.root_depth_single")),
        body_depth_double=int(cfg_get(config, "model.body_depth_double")),
        body_depth_single=int(cfg_get(config, "model.body_depth_single")),
        mlp_ratio=float(cfg_get(config, "model.mlp_ratio")),
        dropout=float(cfg_get(config, "model.dropout")),
        max_text_tokens=text_max_tokens,
        text_encoder=text_encoder,
        hytext_cache_dir=text_cache_dir,
        hytext_ctxt_dim=text_ctxt_dim,
        hytext_vtxt_dim=text_vtxt_dim,
        hytext_max_open_shards=int(cfg_get(config, "text.max_open_shards")),
        hytext_strict_cache=bool(cfg_get(config, "text.strict_cache")),
        motion_stats_dir=str(stats_root / "full"),
        local_root_stats_dir=str(stats_root / "local_root"),
        fps=float(cfg_get(config, "loss.fps")),
        stats_variance_eps=float(cfg_get(config, "model.stats_variance_eps")),
        detach_root_bridge=bool(cfg_get(config, "model.detach_root_bridge")),
        self_conditioning=bool(cfg_get(config, "model.self_conditioning")),
        max_frames=int(cfg_get(config, "data.max_target_frames")),
        normalize_contacts=normalize_contacts,
        source_fusion_mode=source_fusion_mode,
        text_global_conditioning=text_global_conditioning,
        text_fusion_mode=text_fusion_mode,
        backbone_type=backbone_type,
        use_ease=use_ease,
        ease_stats_dir=ease_stats_dir,
    )


def source_fusion_mode_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Recover the parameter-free source fusion treatment saved with a checkpoint."""

    mode = "additive"
    runtime = checkpoint.get("runtime_identity")
    if isinstance(runtime, dict):
        overrides = runtime.get("research_overrides")
        if isinstance(overrides, dict):
            treatment = overrides.get("research_treatment")
            if isinstance(treatment, dict):
                mode = str(treatment.get("source_fusion_mode", mode))
    if mode not in {"additive", "token_block"}:
        raise ValueError(f"Checkpoint contains unknown source fusion mode {mode!r}")
    return mode


def text_global_conditioning_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Recover the text-conditioning architecture saved with a checkpoint."""

    mode = "pooled_adaln"
    runtime = checkpoint.get("runtime_identity")
    if isinstance(runtime, dict):
        overrides = runtime.get("research_overrides")
        if isinstance(overrides, dict):
            mode = str(overrides.get("text_global_conditioning", mode))
    if mode not in TEXT_GLOBAL_CONDITIONING_MODES:
        raise ValueError(
            f"Checkpoint contains unknown text global conditioning mode {mode!r}"
        )
    return mode


def text_fusion_mode_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    """Recover the text/motion attention topology saved with a checkpoint."""

    mode = "f00"
    runtime = checkpoint.get("runtime_identity")
    if isinstance(runtime, dict):
        overrides = runtime.get("research_overrides")
        if isinstance(overrides, dict):
            mode = str(overrides.get("text_fusion_mode", mode)).lower()
    if mode not in TEXT_FUSION_MODES:
        raise ValueError(f"Checkpoint contains unknown text fusion mode {mode!r}")
    return mode


def conditioning_architecture_from_checkpoint(
    checkpoint: dict[str, Any],
) -> str:
    mode = "hytext_flux"
    runtime = checkpoint.get("runtime_identity")
    if isinstance(runtime, dict):
        overrides = runtime.get("research_overrides")
        if isinstance(overrides, dict):
            mode = str(overrides.get("conditioning_architecture", mode))
    if mode not in CONDITIONING_ARCHITECTURES:
        raise ValueError(
            f"Checkpoint contains unknown conditioning architecture {mode!r}"
        )
    return mode


def llm2vec_cache_dir_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    runtime = checkpoint.get("runtime_identity")
    if not isinstance(runtime, dict):
        return ""
    overrides = runtime.get("research_overrides")
    if not isinstance(overrides, dict):
        return ""
    return str(overrides.get("llm2vec_cache_dir", ""))


def base_loss_spaces_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[str, str]:
    representation = "velocity_mse"
    contact = "velocity_mse"
    runtime = checkpoint.get("runtime_identity")
    if isinstance(runtime, dict):
        overrides = runtime.get("research_overrides")
        if isinstance(overrides, dict):
            representation = str(
                overrides.get(
                    "base_representation_loss_space",
                    representation,
                )
            )
            contact = str(
                overrides.get("base_contact_loss_space", contact)
            )
    for name, value in (
        ("base_representation_loss_space", representation),
        ("base_contact_loss_space", contact),
    ):
        if value not in REPRESENTATION_LOSS_SPACES:
            raise ValueError(
                f"Checkpoint contains unknown {name}={value!r}"
            )
    return representation, contact


def validate_checkpoint_text_fusion_mode(
    checkpoint: dict[str, Any],
    requested_mode: str,
) -> str:
    checkpoint_mode = text_fusion_mode_from_checkpoint(checkpoint)
    requested_mode = str(requested_mode).lower()
    if requested_mode not in TEXT_FUSION_MODES:
        raise ValueError(f"Unknown requested text fusion mode {requested_mode!r}")
    if checkpoint_mode != requested_mode:
        raise RuntimeError(
            "Text fusion mode changed across checkpoint load: "
            f"checkpoint={checkpoint_mode!r}, requested={requested_mode!r}"
        )
    return checkpoint_mode


def validate_checkpoint_text_fusion_implementation(
    checkpoint: dict[str, Any],
    current_identity: dict[str, Any],
    *,
    enforce: bool = True,
) -> str:
    """Reject implementation drift for checkpoints from explicit fusion runs."""

    if not enforce:
        return "research_mode_structural_load"

    runtime = checkpoint.get("runtime_identity")
    overrides = (
        runtime.get("research_overrides") if isinstance(runtime, dict) else None
    )
    if not isinstance(overrides, dict) or "text_fusion_mode" not in overrides:
        return "legacy_f00_unpinned"

    previous_identity = checkpoint.get("code_identity")
    previous_files = (
        previous_identity.get("files")
        if isinstance(previous_identity, dict)
        else None
    )
    current_files = current_identity.get("files")
    if not isinstance(previous_files, dict) or not isinstance(current_files, dict):
        raise RuntimeError(
            "Explicit text-fusion checkpoint has no usable code identity"
        )
    mismatches = [
        name
        for name in TEXT_FUSION_IMPLEMENTATION_FILES
        if previous_files.get(name) != current_files.get(name)
    ]
    if mismatches:
        raise RuntimeError(
            "Text fusion implementation changed across checkpoint load: "
            + ", ".join(mismatches)
        )
    return "exact"


def create_model_from_checkpoint(
    checkpoint: dict[str, Any],
) -> HY273KimodoContextFlow:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing its resolved config")
    validate_checkpoint_text_fusion_implementation(
        checkpoint,
        current_code_identity(),
        enforce=False,
    )
    return create_model(
        config,
        source_fusion_mode=source_fusion_mode_from_checkpoint(checkpoint),
        text_global_conditioning=text_global_conditioning_from_checkpoint(checkpoint),
        text_fusion_mode=text_fusion_mode_from_checkpoint(checkpoint),
        conditioning_architecture=conditioning_architecture_from_checkpoint(
            checkpoint
        ),
        llm2vec_cache_dir=llm2vec_cache_dir_from_checkpoint(checkpoint),
    )


def parameter_name_sha(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def optimizer_groups(
    model: HY273KimodoContextFlow,
    step: int,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
    g0_lr_override: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    named = dict(model.named_parameters())
    by_id = {id(param): name for name, param in named.items()}
    group_parameters = {
        "G0_existing": model.base_parameters(),
        "G1_context_weight": model.context_weight_parameters(),
        "G2_context_bias": model.context_bias_parameters(),
    }
    group_order = OPTIMIZER_GROUP_ORDER
    if model.use_ease:
        group_parameters.update(
            {
                "G3_ease_weight": model.ease_weight_parameters(),
                "G4_ease_bias": model.ease_bias_parameters(),
            }
        )
        group_order = (*OPTIMIZER_GROUP_ORDER, *EASE_OPTIMIZER_GROUP_ORDER)
    seen: set[int] = set()
    groups: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"order": list(group_order), "groups": {}}
    hparams = optimizer_group_hparams(step, schedule_version)
    if set(hparams) != set(group_order):
        raise RuntimeError(
            f"Optimizer hparameter groups differ from model groups: "
            f"hparams={sorted(hparams)} model={list(group_order)}"
        )
    for group_name in group_order:
        params = tuple(group_parameters[group_name])
        names = [by_id[id(param)] for param in params]
        if not params or len(set(map(id, params))) != len(params):
            raise RuntimeError(f"Invalid or empty optimizer group {group_name}")
        if seen.intersection(map(id, params)):
            raise RuntimeError(f"Optimizer parameter groups overlap at {group_name}")
        seen.update(map(id, params))
        expected = dict(hparams[group_name])
        if group_name == "G0_existing" and g0_lr_override is not None:
            expected["lr"] = float(g0_lr_override)
        groups.append(
            {
                "params": list(params),
                "group_name": group_name,
                "lr": float(expected["lr"]),
                "weight_decay": float(expected["weight_decay"]),
            }
        )
        manifest["groups"][group_name] = {
            "parameter_count": sum(param.numel() for param in params),
            "tensor_count": len(params),
            "ordered_parameter_name_sha256": hashlib.sha256(
                "\n".join(names).encode("utf-8")
            ).hexdigest(),
            "parameter_names": names,
        }
    trainable = {id(param) for param in model.parameters() if param.requires_grad}
    if seen != trainable:
        raise RuntimeError(
            f"Optimizer groups do not cover trainable parameters: missing={len(trainable-seen)} extra={len(seen-trainable)}"
        )
    manifest["manifest_sha256"] = canonical_sha(manifest)
    return groups, manifest


def apply_optimizer_phase(
    optimizer: torch.optim.Optimizer,
    step: int,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
    g0_lr_override: float | None = None,
) -> None:
    expected = optimizer_group_hparams(step, schedule_version)
    actual_names = tuple(str(group.get("group_name")) for group in optimizer.param_groups)
    if actual_names != tuple(expected):
        raise RuntimeError(f"Optimizer group order changed: {actual_names}")
    for group in optimizer.param_groups:
        group_name = str(group["group_name"])
        values = dict(expected[group_name])
        if group_name == "G0_existing" and g0_lr_override is not None:
            values["lr"] = float(g0_lr_override)
        group["lr"] = float(values["lr"])
        group["weight_decay"] = float(values["weight_decay"])
    for group in optimizer.param_groups:
        group_name = str(group["group_name"])
        values = dict(expected[group_name])
        if group_name == "G0_existing" and g0_lr_override is not None:
            values["lr"] = float(g0_lr_override)
        if group["lr"] != values["lr"] or group["weight_decay"] != values["weight_decay"]:
            raise RuntimeError("Optimizer phase hparameters failed to resolve exactly")


def _optimizer_parameter_ids_by_name(
    optimizer_state: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, int]:
    order = tuple(manifest.get("order", ()))
    saved_groups = optimizer_state.get("param_groups")
    manifest_groups = manifest.get("groups")
    if not isinstance(saved_groups, list) or not isinstance(manifest_groups, dict):
        raise ValueError("Optimizer checkpoint/manifest is malformed")
    if len(saved_groups) != len(order):
        raise ValueError("Optimizer group count differs from its manifest")
    output: dict[str, int] = {}
    for group_name, saved_group in zip(order, saved_groups):
        if str(saved_group.get("group_name")) != str(group_name):
            raise ValueError("Optimizer group order/name differs from its manifest")
        record = manifest_groups.get(group_name)
        names = record.get("parameter_names") if isinstance(record, dict) else None
        parameter_ids = saved_group.get("params")
        if not isinstance(names, list) or not isinstance(parameter_ids, list):
            raise ValueError(f"Optimizer manifest group {group_name} is malformed")
        if len(names) != len(parameter_ids):
            raise ValueError(f"Optimizer group {group_name} tensor count mismatch")
        for name, parameter_id in zip(names, parameter_ids):
            if name in output:
                raise ValueError(f"Duplicate optimizer parameter name {name}")
            output[str(name)] = int(parameter_id)
    return output


def migrate_optimizer_state_with_new_ease(
    optimizer: torch.optim.Optimizer,
    *,
    parent_state: dict[str, Any],
    parent_manifest: dict[str, Any],
    current_manifest: dict[str, Any],
) -> None:
    """Retain every parent Adam moment while leaving new Ease state empty."""

    current_state = optimizer.state_dict()
    parent_ids = _optimizer_parameter_ids_by_name(parent_state, parent_manifest)
    current_ids = _optimizer_parameter_ids_by_name(current_state, current_manifest)
    ease_names = {
        name for name in current_ids if name.startswith("ease_conditioner.")
    }
    if not ease_names:
        raise RuntimeError("Ease optimizer migration found no Ease parameters")
    expected_parent_names = set(current_ids) - ease_names
    if set(parent_ids) != expected_parent_names:
        missing = sorted(expected_parent_names - set(parent_ids))
        unexpected = sorted(set(parent_ids) - expected_parent_names)
        raise RuntimeError(
            "Parent/current non-Ease optimizer parameters differ: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    migrated_state: dict[int, Any] = {}
    parent_slots = parent_state.get("state")
    if not isinstance(parent_slots, dict):
        raise ValueError("Parent optimizer state is malformed")
    for name in expected_parent_names:
        parent_id = parent_ids[name]
        if parent_id in parent_slots:
            migrated_state[current_ids[name]] = parent_slots[parent_id]
    current_state["state"] = migrated_state
    optimizer.load_state_dict(current_state)
    if any(
        parameter in optimizer.state
        for group in optimizer.param_groups[-len(EASE_OPTIMIZER_GROUP_ORDER) :]
        for parameter in group["params"]
    ):
        raise RuntimeError("New Ease parameters unexpectedly acquired Adam state")


def load_parent_model_with_new_ease(
    model: HY273KimodoContextFlow,
    parent_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    """Strictly load all parent tensors, allowing only new Ease tensors."""

    incompatible = model.load_state_dict(parent_state, strict=False)
    missing = tuple(sorted(incompatible.missing_keys))
    unexpected = tuple(sorted(incompatible.unexpected_keys))
    expected_missing = tuple(
        sorted(
            name
            for name in model.state_dict()
            if name.startswith("ease_conditioner.")
        )
    )
    if unexpected or missing != expected_missing:
        raise RuntimeError(
            "Ease model migration mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )
    return missing


def initialize_ema_with_new_ease(
    model: HY273KimodoContextFlow,
    parent_ema: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    ema = initialize_ema(model)
    expected_parent = {
        name for name in ema if not name.startswith("ease_conditioner.")
    }
    if set(parent_ema) != expected_parent:
        raise RuntimeError("Parent EMA tensors differ outside the new Ease module")
    for name, value in parent_ema.items():
        if ema[name].shape != value.shape or ema[name].dtype != value.dtype:
            raise RuntimeError(f"Parent EMA tensor schema changed at {name}")
        ema[name] = value.to(device=ema[name].device)
    return ema


def _plan_draw(plan: Any, manifest_sha256: str, run_seed: int, stream_id: str) -> int:
    task = TaskId.GENERATE if plan.train_stream_id == TrainStream.HML_MIXED else TaskId.EDIT
    return sample_key_u64(
        manifest_sha256=manifest_sha256,
        run_seed=run_seed,
        global_sample_ordinal=plan.global_sample_ordinal,
        train_stream_id=plan.train_stream_id,
        task_id=int(task),
        uid=plan.uid,
        random_stream_id=stream_id,
    )


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def build_stateless_flow_inputs(
    *,
    plans: list[Any],
    x0_norm: torch.Tensor,
    manifest_sha256: str,
    run_seed: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    timesteps, noises, contacts = [], [], []
    for plan in plans:
        t_generator = _generator(
            x0_norm.device, _plan_draw(plan, manifest_sha256, run_seed, "flow_t")
        )
        noise_generator = _generator(
            x0_norm.device,
            _plan_draw(plan, manifest_sha256, run_seed, "continuous_noise"),
        )
        contact_generator = _generator(
            x0_norm.device,
            _plan_draw(plan, manifest_sha256, run_seed, "contact_aux"),
        )
        timesteps.append(
            sample_timesteps(
                1,
                device=x0_norm.device,
                schedule=str(cfg_get(config, "flow.timestep_schedule")),
                p_mean=float(cfg_get(config, "flow.timestep_mean")),
                p_std=float(cfg_get(config, "flow.timestep_std")),
                generator=t_generator,
            )
        )
        noises.append(
            torch.randn(
                x0_norm.shape[1],
                CONT_DIM,
                device=x0_norm.device,
                dtype=x0_norm.dtype,
                generator=noise_generator,
            )
        )
        contacts.append(
            torch.rand(
                x0_norm.shape[1],
                4,
                device=x0_norm.device,
                dtype=x0_norm.dtype,
                generator=contact_generator,
            )
        )
    return torch.cat(timesteps), torch.stack(noises), torch.stack(contacts)


def build_stateless_unified_273_flow_inputs(
    *,
    plans: list[Any],
    x0_norm: torch.Tensor,
    manifest_sha256: str,
    run_seed: int,
    config: dict[str, Any],
    edit_low_t_mix_prob: float = 0.0,
    edit_low_t_max: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample replayable t and one Gaussian source tensor over all 273 channels."""

    if not 0.0 <= float(edit_low_t_mix_prob) <= 1.0:
        raise ValueError("edit_low_t_mix_prob must be in [0,1]")
    if not 0.0 < float(edit_low_t_max) < 1.0:
        raise ValueError("edit_low_t_max must be in (0,1)")

    timesteps, noises, low_t_selected = [], [], []
    for plan in plans:
        t_generator = _generator(
            x0_norm.device, _plan_draw(plan, manifest_sha256, run_seed, "flow_t")
        )
        noise_generator = _generator(
            x0_norm.device,
            _plan_draw(plan, manifest_sha256, run_seed, "unified_273_noise"),
        )
        use_low_t = (
            plan.train_stream_id == TrainStream.MOTION_EDIT
            and float(edit_low_t_mix_prob) > 0.0
            and bernoulli_from_draw(
                _plan_draw(
                    plan,
                    manifest_sha256,
                    run_seed,
                    "edit_low_t_selector",
                ),
                float(edit_low_t_mix_prob),
            )
        )
        if use_low_t:
            low_t_generator = _generator(
                x0_norm.device,
                _plan_draw(
                    plan,
                    manifest_sha256,
                    run_seed,
                    "edit_low_t_value",
                ),
            )
            timestep = (
                torch.rand(
                    1,
                    device=x0_norm.device,
                    generator=low_t_generator,
                )
                * float(edit_low_t_max)
            ).clamp(1e-4, 1.0 - 1e-4)
        else:
            timestep = sample_timesteps(
                1,
                device=x0_norm.device,
                schedule=str(cfg_get(config, "flow.timestep_schedule")),
                p_mean=float(cfg_get(config, "flow.timestep_mean")),
                p_std=float(cfg_get(config, "flow.timestep_std")),
                generator=t_generator,
            )
        timesteps.append(timestep)
        low_t_selected.append(bool(use_low_t))
        noises.append(
            torch.randn(
                x0_norm.shape[1],
                DIM_HY273,
                device=x0_norm.device,
                dtype=x0_norm.dtype,
                generator=noise_generator,
            )
        )
    return (
        torch.cat(timesteps),
        torch.stack(noises),
        torch.tensor(low_t_selected, device=x0_norm.device, dtype=torch.bool),
    )


def build_equal_length_source_identity_flow(
    *,
    condition: ConditionBatch,
    normalizer: HY273Normalizer,
    timesteps: torch.Tensor,
    unified_noise: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the paired no-edit branch without interpolating K273 features."""

    bsz, target_frames = condition.target_valid.shape
    if unified_noise.shape != (bsz, target_frames, DIM_HY273):
        raise ValueError("unified_noise must match the padded target [B,T,273]")
    if condition.source_slots != 1:
        raise ValueError("Source identity flow requires exactly one source slot")
    requested = condition.requested_target_len.long()
    source_lengths = condition.source_native_lengths[:, 0].long()
    source_present = condition.source_present[:, 0]
    exact_pair = source_present & (source_lengths == requested)
    exact_pair = exact_pair & (
        condition.target_valid.sum(dim=-1).long() == requested
    )

    source_physical = unified_noise.new_zeros(
        (bsz, target_frames, DIM_HY273), dtype=torch.float32
    )
    copy_frames = min(target_frames, condition.source_frames)
    source_physical[:, :copy_frames] = condition.source_motion[
        :, 0, :copy_frames
    ].to(device=unified_noise.device, dtype=torch.float32)
    source_norm = normalizer.normalize(source_physical)
    identity_valid = condition.target_valid & exact_pair[:, None]
    zero_observed = torch.zeros_like(source_norm)
    zero_mask = torch.zeros_like(source_norm, dtype=torch.bool)
    flow_state = build_unified_273_flow_state(
        source_norm,
        zero_observed,
        zero_mask,
        timesteps,
        noise=unified_noise,
    )
    return {
        "exact_pair": exact_pair,
        "target_valid": identity_valid,
        "x0_physical": source_physical,
        "x0_norm": source_norm,
        "hard_mask": zero_mask,
        "observed_norm": zero_observed,
        "model_in": flow_state["model_in"],
        "z_imp": flow_state["z_imp"],
    }


def build_hard_controls(
    *,
    target_physical: torch.Tensor,
    condition: ConditionBatch,
    plans: list[Any],
    global_step: int,
    config: dict[str, Any],
    manifest_sha256: str,
    run_seed: int,
    forced_mode_schedule: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    observed = torch.zeros_like(target_physical)
    hard_mask = torch.zeros_like(target_physical, dtype=torch.bool)
    modes: list[str] = []
    start = int(cfg_get(config, "control.curriculum_start_step"))
    end = int(cfg_get(config, "control.curriculum_end_step"))
    progress = min(max((int(global_step) - start + 1) / float(end - start), 0.0), 1.0)
    if forced_mode_schedule is not None and len(forced_mode_schedule) != len(plans):
        raise ValueError("forced_mode_schedule must match the number of SamplePlans")
    for index, plan in enumerate(plans):
        capability = CapabilityId(int(condition.capability_id[index].item()))
        intended = capability in {
            CapabilityId.KIMODO_CONTROL,
            CapabilityId.MOTION_EDIT_CONTROL,
        }
        if not intended:
            modes.append("none")
            continue
        generator = _generator(target_physical.device, plan.control_u64)
        root_heading_generator = _generator(
            target_physical.device,
            _plan_draw(
                plan,
                manifest_sha256,
                run_seed,
                "root_heading_presence",
            ),
        )
        result = build_kimodo_control_curriculum_batch(
            target_physical[index : index + 1],
            lengths=condition.requested_target_len[index : index + 1],
            progress=progress,
            config=KimodoControlCurriculum(
                none_prob=0.0,
                mixed_prob=float(cfg_get(config, "control.mixed_prob")),
                max_sparse_keyframes=int(
                    cfg_get(config, "control.max_sparse_keyframes")
                ),
                dense_min_fraction=float(
                    cfg_get(config, "control.dense_min_fraction")
                ),
                endpoint_preset=str(cfg_get(config, "control.endpoint_preset")),
                endpoint_subset_mode=str(
                    cfg_get(config, "control.endpoint_subset_mode")
                ),
                include_root_ref_for_endpoints=bool(
                    cfg_get(config, "control.include_root_ref_for_endpoints")
                ),
                include_endpoint_rotations=bool(
                    cfg_get(config, "control.include_endpoint_rotations")
                ),
                include_contact_pattern=bool(
                    cfg_get(config, "control.include_contact_pattern")
                ),
                root_heading_probability=float(
                    config["control"].get("root_heading_probability", 1.0)
                ),
            ),
            generator=generator,
            root_heading_generator=root_heading_generator,
            mode_schedule=(
                None
                if forced_mode_schedule is None
                else [forced_mode_schedule[index]]
            ),
        )
        observed[index] = result.observed_motion[0]
        hard_mask[index] = result.motion_mask[0]
        modes.append(result.mode_ids[0])
        valid_mask = condition.target_valid[index, :, None]
        if not bool((hard_mask[index] & valid_mask).any()):
            raise RuntimeError(f"Intended control plan emitted an empty mask: {plan.uid}")
    if bool((hard_mask & ~condition.target_valid[..., None]).any()):
        raise RuntimeError("Control compiler wrote into target padding")
    return observed, hard_mask, modes


class LossMetricWindow:
    """Accumulate every update, then all-reduce interval statistics at log time."""

    def __init__(self, term_names: tuple[str, ...], device: torch.device) -> None:
        self.term_names = term_names
        self.device = device
        self.scope_names = (
            "overall",
            "stream/HML_MIXED",
            "stream/MOTION_EDIT",
            *(f"capability/{capability.name}" for capability in CapabilityId),
        )
        self._scope_index = {name: index for index, name in enumerate(self.scope_names)}
        shape = (len(self.scope_names), len(term_names))
        # numerator, denominator, local raw sum, local weighted sum
        self.values = torch.zeros(*shape, 4, device=device, dtype=torch.float64)
        self.scope_steps = torch.zeros(len(self.scope_names), device=device, dtype=torch.float64)
        self.scope_samples = torch.zeros(len(self.scope_names), device=device, dtype=torch.float64)
        self.total_sum = torch.zeros(len(self.scope_names), device=device, dtype=torch.float64)
        self.fk_distance_num = torch.zeros(len(self.scope_names), device=device, dtype=torch.float64)
        self.fk_distance_den = torch.zeros(len(self.scope_names), device=device, dtype=torch.float64)

    def _add_scope(
        self,
        scope: str,
        terms: dict[str, Any],
        total: torch.Tensor,
        sample_count: int,
        fk_distance: Any | None = None,
    ) -> None:
        index = self._scope_index[scope]
        self.scope_steps[index] += 1.0
        self.scope_samples[index] += float(sample_count)
        self.total_sum[index] += total.detach().double()
        for term_index, name in enumerate(self.term_names):
            term = terms[name]
            self.values[index, term_index, 0] += term.numerator.detach().double()
            self.values[index, term_index, 1] += term.denominator.detach().double()
            self.values[index, term_index, 2] += term.raw.detach().double()
            self.values[index, term_index, 3] += term.weighted.detach().double()
        if fk_distance is not None:
            self.fk_distance_num[index] += fk_distance.numerator.detach().double()
            self.fk_distance_den[index] += fk_distance.denominator.detach().double()

    def add(
        self,
        bundle: HY273MultitaskLossBundle,
        condition: ConditionBatch,
        stream: TrainStream,
    ) -> None:
        batch_size = condition.batch_size
        with torch.no_grad():
            self._add_scope(
                "overall",
                bundle.terms,
                bundle.total,
                batch_size,
                bundle.fk_distance_cm,
            )
            stream_name = f"stream/{stream.name}"
            self._add_scope(
                stream_name,
                bundle.terms,
                bundle.total,
                batch_size,
                bundle.fk_distance_cm,
            )
            for capability in CapabilityId:
                selector = condition.capability_id == int(capability)
                count = int(selector.sum().item())
                if count == 0:
                    continue
                terms = bundle.terms_for_samples(selector)
                capability_total = sum(
                    (term.weighted for term in terms.values()),
                    bundle.total.detach() * 0.0,
                )
                self._add_scope(
                    f"capability/{capability.name}",
                    terms,
                    capability_total,
                    count,
                    bundle.fk_distance_for_samples(selector),
                )

    def reduce_and_reset(self) -> dict[str, float]:
        tensors = (
            self.values,
            self.scope_steps,
            self.scope_samples,
            self.total_sum,
            self.fk_distance_num,
            self.fk_distance_den,
        )
        if is_distributed():
            for tensor in tensors:
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metrics: dict[str, float] = {}
        for scope_index, scope in enumerate(self.scope_names):
            steps = float(self.scope_steps[scope_index].item())
            if steps <= 0:
                continue
            total = float(self.total_sum[scope_index].item() / steps)
            prefix = f"loss/{scope}"
            metrics[f"{prefix}/backward_total"] = total
            metrics[f"{prefix}/rank_steps"] = steps
            metrics[f"{prefix}/samples"] = float(self.scope_samples[scope_index].item())
            group_weighted: dict[str, float] = {}
            for term_index, name in enumerate(self.term_names):
                numerator, denominator, raw_sum, weighted_sum = (
                    self.values[scope_index, term_index].tolist()
                )
                term_prefix = f"{prefix}/{name}"
                global_ratio = numerator / max(denominator, 1.0)
                local_ratio_mean = raw_sum / steps
                weighted_mean = weighted_sum / steps
                metrics[f"{term_prefix}/numerator"] = numerator
                metrics[f"{term_prefix}/denominator"] = denominator
                metrics[f"{term_prefix}/global_ratio"] = global_ratio
                metrics[f"{term_prefix}/backward_local_ratio_mean"] = local_ratio_mean
                metrics[f"{term_prefix}/weighted_contribution"] = weighted_mean
                metrics[f"{term_prefix}/percent_total"] = (
                    100.0 * weighted_mean / max(total, 1e-30)
                )
                if name.startswith("repr_"):
                    group = "representation"
                else:
                    group = name
                group_weighted[group] = group_weighted.get(group, 0.0) + weighted_mean
            for group, contribution in group_weighted.items():
                metrics[f"{prefix}/group_{group}/weighted_contribution"] = contribution
                metrics[f"{prefix}/group_{group}/percent_total"] = (
                    100.0 * contribution / max(total, 1e-30)
                )
            fk_den = float(self.fk_distance_den[scope_index].item())
            metrics[f"{prefix}/fk_distance_cm"] = float(
                self.fk_distance_num[scope_index].item() / max(fk_den, 1.0)
            )
        for tensor in tensors:
            tensor.zero_()
        return metrics


def tensor_group_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return 0.0
    norms = torch._foreach_norm(grads, 2.0)
    return float(torch.linalg.vector_norm(torch.stack(norms), 2.0).item())


def tensor_masked_rms(values: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.to(device=values.device, dtype=torch.bool)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    count = int(expanded.sum().item())
    if count == 0:
        return 0.0
    selected = values.float().masked_select(expanded)
    return float(torch.sqrt(selected.square().mean()).item())


def feature_mask_fraction(
    feature_mask: torch.Tensor,
    valid: torch.Tensor,
    feature_slice: slice,
) -> float:
    selected = feature_mask[..., feature_slice]
    support = valid.to(device=selected.device, dtype=torch.bool).unsqueeze(-1).expand_as(
        selected
    )
    denominator = int(support.sum().item())
    if denominator == 0:
        return 0.0
    return float((selected & support).sum().item() / denominator)


class UpdateSampler:
    """Measure exact context updates and a deterministic 1M-element base sketch."""

    def __init__(self, model: HY273KimodoContextFlow, base_budget: int = 1_000_000) -> None:
        self.groups = {
            "base": self._select(model.base_parameters(), int(base_budget)),
            "context": self._select(
                (*model.context_weight_parameters(), *model.context_bias_parameters()),
                sum(
                    parameter.numel()
                    for parameter in (
                        *model.context_weight_parameters(),
                        *model.context_bias_parameters(),
                    )
                ),
            ),
        }

    @staticmethod
    def _select(
        parameters: Iterable[torch.nn.Parameter], budget: int
    ) -> list[tuple[torch.nn.Parameter, int]]:
        selected = []
        remaining = int(budget)
        for parameter in parameters:
            if remaining <= 0:
                break
            count = min(remaining, parameter.numel())
            selected.append((parameter, count))
            remaining -= count
        return selected

    def snapshot(self) -> dict[str, list[torch.Tensor]]:
        return {
            name: [parameter.detach().view(-1)[:count].clone() for parameter, count in rows]
            for name, rows in self.groups.items()
        }

    def differences(self, before: dict[str, list[torch.Tensor]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, rows in self.groups.items():
            square_sum = 0.0
            elements = 0
            for (parameter, count), previous in zip(rows, before[name]):
                delta = parameter.detach().view(-1)[:count] - previous
                square_sum += float(delta.float().square().sum().item())
                elements += count
            metrics[f"update/{name}/sampled_norm"] = math.sqrt(square_sum)
            metrics[f"update/{name}/sampled_rms"] = math.sqrt(
                square_sum / max(elements, 1)
            )
            metrics[f"update/{name}/sampled_elements"] = float(elements)
        return metrics


def assert_and_mask_context_gradients(
    model: HY273KimodoContextFlow,
    *,
    context_active: bool,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    schedule_version: str = HIGH_LEVEL_SCHEDULE_VERSION,
) -> None:
    context = (*model.context_weight_parameters(), *model.context_bias_parameters())
    phase = phase_for_step(global_step)
    context_phase_active = phase in {
        TrainingPhase.STAGE_B2,
        TrainingPhase.STAGE_C,
    }
    if (
        schedule_version == KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION
        and phase == TrainingPhase.STAGE_B1
    ):
        context_phase_active = True
    if (
        schedule_version == R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION
        and phase in {TrainingPhase.STAGE_B1, TrainingPhase.STAGE_B2}
    ):
        context_phase_active = False
    should_update = context_active and context_phase_active
    if context_active and not should_update:
        raise RuntimeError("Source/task context appeared in a context-frozen stage")
    for parameter in context:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError("Context parameter unexpectedly has grad=None after backward")
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Context gradient is non-finite")
        if should_update:
            continue
        if bool(torch.count_nonzero(gradient)):
            raise RuntimeError(
                f"Context-inactive/frozen gradient is not exact zero at step={global_step}"
            )
        parameter.grad = None
        if not context_phase_active and parameter in optimizer.state:
            raise RuntimeError("Frozen context parameter acquired optimizer state")


def assert_and_mask_ease_gradients(
    model: HY273KimodoContextFlow,
    *,
    ease_active: bool,
    optimizer: torch.optim.Optimizer,
) -> None:
    parameters = (*model.ease_weight_parameters(), *model.ease_bias_parameters())
    if not parameters:
        if ease_active:
            raise RuntimeError("Ease-present batch reached an Ease-disabled model")
        return
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError("Ease parameter unexpectedly has grad=None after backward")
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Ease gradient is non-finite")
        if ease_active:
            continue
        if bool(torch.count_nonzero(gradient)):
            raise RuntimeError("Ease-absent global batch produced a nonzero Ease gradient")
        parameter.grad = None


def _ease_optimizer_steps(
    optimizer: torch.optim.Optimizer,
    model: HY273KimodoContextFlow,
) -> dict[int, int]:
    output = {}
    for parameter in (
        *model.ease_weight_parameters(),
        *model.ease_bias_parameters(),
    ):
        state = optimizer.state.get(parameter)
        if not state or "step" not in state:
            continue
        value = state["step"]
        output[id(parameter)] = (
            int(value.item()) if torch.is_tensor(value) else int(value)
        )
    return output


def _assert_ease_optimizer_count(
    optimizer: torch.optim.Optimizer,
    model: HY273KimodoContextFlow,
    ease_update_count: int,
) -> None:
    parameters = (*model.ease_weight_parameters(), *model.ease_bias_parameters())
    if not parameters:
        if ease_update_count != 0:
            raise RuntimeError("Ease-disabled checkpoint has a nonzero update count")
        return
    steps = _ease_optimizer_steps(optimizer, model)
    if ease_update_count == 0:
        if steps:
            raise RuntimeError("Untrained Ease module has unexpected Adam state")
        return
    if len(steps) != len(parameters) or set(steps.values()) != {
        int(ease_update_count)
    }:
        raise RuntimeError(
            "Ease Adam step/count mismatch: "
            f"states={sorted(steps.values())}, updates={ease_update_count}"
        )


def initialize_ema(model: HY273KimodoContextFlow) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


@torch.no_grad()
def update_ema(
    ema: dict[str, torch.Tensor], model: HY273KimodoContextFlow, decay: float
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
    model: HY273KimodoContextFlow,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    config: dict[str, Any],
    config_path: Path,
    next_global_step: int,
    batcher: HY273MultitaskStepBatcher,
    context_update_count: int,
    ease_update_count: int,
    optimizer_manifest: dict[str, Any],
    asset_identity: dict[str, Any],
    ease_stats_identity: dict[str, Any] | None,
    normalizer: HY273Normalizer,
    run_name: str,
    run_uuid: str,
    ema_update_count: int,
    code_identity: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> None:
    payload = {
        "format": CHECKPOINT_FORMAT,
        "train_contract": str(cfg_get(config, "contract.version")),
        "ema_schedule_version": EMA_SCHEDULE,
        "high_level_schedule_version": str(
            cfg_get(config, "stage.schedule_version")
        ),
        "hml_inner_schedule_version": HML_INNER_SCHEDULE_VERSION,
        "sample_rng_version": SAMPLE_RNG_VERSION,
        "bucket_plan_version": BUCKET_PLAN_VERSION,
        "run_name": run_name,
        "run_uuid": run_uuid,
        "config": config,
        "config_path": str(config_path),
        "config_sha256": canonical_sha(config),
        "base_contract_sha256": base_contract_sha(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema,
        "next_global_step": int(next_global_step),
        "phase_id": int(phase_for_step(next_global_step)),
        "batcher": batcher.state_dict(),
        "context_update_count": int(context_update_count),
        "ease_update_count": int(ease_update_count),
        "ema_update_count": int(ema_update_count),
        "code_identity": code_identity,
        "runtime_identity": runtime_identity,
        "optimizer_group_manifest": optimizer_manifest,
        "asset_identity": asset_identity,
        "ease_stats_identity": ease_stats_identity,
        "normalizer": normalizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _context_optimizer_steps(
    optimizer: torch.optim.Optimizer, model: HY273KimodoContextFlow
) -> dict[int, int]:
    output = {}
    for parameter in (
        *model.context_weight_parameters(),
        *model.context_bias_parameters(),
    ):
        state = optimizer.state.get(parameter)
        if not state or "step" not in state:
            continue
        value = state["step"]
        output[id(parameter)] = int(value.item()) if torch.is_tensor(value) else int(value)
    return output


def _assert_context_optimizer_count(
    optimizer: torch.optim.Optimizer,
    model: HY273KimodoContextFlow,
    context_update_count: int,
) -> None:
    parameters = (*model.context_weight_parameters(), *model.context_bias_parameters())
    steps = _context_optimizer_steps(optimizer, model)
    if context_update_count == 0:
        if steps:
            raise RuntimeError("Frozen context has unexpected Adam state")
        return
    if len(steps) != len(parameters) or set(steps.values()) != {int(context_update_count)}:
        raise RuntimeError(
            "Context Adam step/count mismatch: "
            f"states={sorted(steps.values())}, updates={context_update_count}"
        )


def expected_ema_updates(next_global_step: int, every: int) -> int:
    if next_global_step <= 0:
        return 0
    return (int(next_global_step) - 1) // int(every) + 1


def _normalizer_matches(
    normalizer: HY273Normalizer, checkpoint_state: dict[str, torch.Tensor]
) -> bool:
    current = normalizer.state_dict()
    return set(current) == set(checkpoint_state) and all(
        torch.equal(current[key].cpu(), checkpoint_state[key].cpu()) for key in current
    )


def _copy_checkpoint_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    os.replace(temporary, target)


def r12_origin_parent_identity(checkpoint_path: str | Path) -> dict[str, Any]:
    return {
        "kind": "r11_stage_a_200k",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "checkpoint_sha256": R12_ORIGIN_PARENT_SHA256,
        "run_name": R12_ORIGIN_PARENT_RUN_NAME,
        "run_uuid": R12_ORIGIN_PARENT_RUN_UUID,
        "train_contract": R11_TRAIN_CONTRACT,
        "next_global_step": 200_000,
        "base_contract_sha256": R12_ORIGIN_PARENT_BASE_CONTRACT_SHA256,
        "config_sha256": R12_ORIGIN_PARENT_CONFIG_SHA256,
        "code_identity_sha256": R12_ORIGIN_PARENT_CODE_IDENTITY_SHA256,
    }


def validate_r12_origin_parent_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("R12 run is missing its immutable origin parent")
    expected = r12_origin_parent_identity(value.get("checkpoint", ""))
    if value != expected:
        raise RuntimeError("R12 immutable origin parent identity mismatch")
    return value


def validate_r12_origin_checkpoint(
    checkpoint: dict[str, Any], *, checkpoint_sha256: str
) -> None:
    expected = {
        "train_contract": R11_TRAIN_CONTRACT,
        "run_name": R12_ORIGIN_PARENT_RUN_NAME,
        "run_uuid": R12_ORIGIN_PARENT_RUN_UUID,
        "next_global_step": 200_000,
        "context_update_count": 0,
        "ema_update_count": 20_000,
        "base_contract_sha256": R12_ORIGIN_PARENT_BASE_CONTRACT_SHA256,
        "config_sha256": R12_ORIGIN_PARENT_CONFIG_SHA256,
    }
    for key, required in expected.items():
        if checkpoint.get(key) != required:
            raise RuntimeError(
                f"R12 origin checkpoint {key} mismatch: "
                f"{checkpoint.get(key)!r}/{required!r}"
            )
    code_identity = checkpoint.get("code_identity")
    if not isinstance(code_identity, dict) or code_identity.get(
        "identity_sha256"
    ) != R12_ORIGIN_PARENT_CODE_IDENTITY_SHA256:
        raise RuntimeError("R12 origin checkpoint code identity mismatch")
    if checkpoint_sha256.lower() != R12_ORIGIN_PARENT_SHA256:
        raise RuntimeError("R12 origin checkpoint content SHA mismatch")


def unified_edit_loss_weights(
    config: dict[str, Any],
) -> UnifiedEditLossWeights | None:
    payload = config.get("edit_objective")
    if payload is None:
        return None
    weights = UnifiedEditLossWeights(
        **{key: float(value) for key, value in payload.items()}
    )
    weights.validate()
    return weights


def mismatched_instruction_texts(texts: Sequence[str]) -> list[str]:
    """Choose a deterministic non-identical in-batch instruction per sample."""

    rows = [str(value) for value in texts]
    normalized = [" ".join(value.split()).casefold() for value in rows]
    if len(rows) < 2:
        raise ValueError("Instruction ranking requires at least two Edit samples")
    mismatched: list[str] = []
    for index, own_text in enumerate(normalized):
        donor = None
        for offset in range(1, len(rows)):
            candidate = (index + offset) % len(rows)
            if normalized[candidate] != own_text:
                donor = rows[candidate]
                break
        if donor is None:
            raise RuntimeError("Edit batch has no non-identical instruction donor")
        mismatched.append(donor)
    return mismatched


def load_same_source_instruction_donors(
    path: str | Path,
    *,
    minimum_target_pair_mse: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Load semantically separable instruction siblings for an identical source.

    The diagnostic group file contains two MotionFix rows with byte-identical
    source K273 and equal-length targets. Filtering on target-pair distance avoids
    treating two effectively equivalent edits as negatives.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Same-source Edit group file is missing: {resolved}")
    minimum = float(minimum_target_pair_mse)
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_target_pair_mse must be finite and non-negative")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Same-source Edit group file must contain a JSON list")

    donors: dict[str, str] = {}
    selected_groups = 0
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Same-source Edit group rows must be JSON objects")
        pair_ids = row.get("pair_ids")
        texts = row.get("texts")
        pair_mse = float(row.get("target_pair_mse", float("nan")))
        source_sha = str(row.get("source_sha256", ""))
        if (
            not isinstance(pair_ids, list)
            or not isinstance(texts, list)
            or len(pair_ids) != 2
            or len(texts) != 2
            or len({str(value) for value in pair_ids}) != 2
            or not source_sha
            or not math.isfinite(pair_mse)
        ):
            raise ValueError("Malformed same-source Edit group row")
        if pair_mse < minimum:
            continue
        normalized_texts = [" ".join(str(value).split()).casefold() for value in texts]
        if not normalized_texts[0] or not normalized_texts[1]:
            raise ValueError("Same-source Edit donor instructions must be non-empty")
        if normalized_texts[0] == normalized_texts[1]:
            continue
        uids = [f"motionfix:{str(value)}" for value in pair_ids]
        for uid, donor_text in zip(uids, reversed([str(value) for value in texts])):
            previous = donors.get(uid)
            if previous is not None and previous != donor_text:
                raise ValueError(f"Conflicting same-source instruction donor for {uid}")
            donors[uid] = donor_text
        selected_groups += 1
    if not donors:
        raise RuntimeError(
            "Same-source Edit group filter selected no instruction donors"
        )
    return donors, {
        "path": str(resolved),
        "minimum_target_pair_mse": minimum,
        "selected_groups": selected_groups,
        "eligible_rows": len(donors),
    }


def same_source_instruction_texts(
    texts: Sequence[str],
    uids: Sequence[str],
    donors: dict[str, str],
    *,
    fallback_mode: str = "mismatched",
) -> tuple[list[str], list[bool]]:
    """Prefer a separable sibling instruction with an explicit fallback policy."""

    if len(texts) != len(uids):
        raise ValueError("texts and uids must have the same length")
    if fallback_mode == "mismatched":
        fallback = mismatched_instruction_texts(texts)
    elif fallback_mode == "self":
        fallback = [str(value) for value in texts]
    else:
        raise ValueError("fallback_mode must be 'mismatched' or 'self'")
    output: list[str] = []
    used_same_source: list[bool] = []
    for text, uid, fallback_text in zip(texts, uids, fallback):
        donor = donors.get(str(uid))
        own = " ".join(str(text).split()).casefold()
        donor_normalized = (
            "" if donor is None else " ".join(str(donor).split()).casefold()
        )
        use_donor = bool(donor_normalized and donor_normalized != own)
        output.append(str(donor) if use_donor else fallback_text)
        used_same_source.append(use_donor)
    return output, used_same_source


def repeat_condition_batch(
    condition: ConditionBatch, count: int, *, v1_strict: bool = True
) -> ConditionBatch:
    values: dict[str, Any] = {}
    for field_name in condition.__dataclass_fields__:
        value = getattr(condition, field_name)
        if torch.is_tensor(value):
            values[field_name] = torch.cat([value] * int(count), dim=0)
        elif value is None:
            values[field_name] = None
        elif field_name == "text_encoding_profile":
            values[field_name] = tuple(value) * int(count)
        else:
            raise TypeError(
                f"Unsupported ConditionBatch field {field_name}: {type(value)}"
            )
    repeated = replace(condition, **values)
    repeated.validate(v1_strict=v1_strict)
    return repeated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/hy273_multitask_stage_a_t2m.yaml"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume_sha256", default="")
    parser.add_argument(
        "--fork_from_r11_stage_a",
        action="store_true",
        help="Start the R12 root-mask protocol from an immutable R11 200K Stage-A checkpoint.",
    )
    parser.add_argument(
        "--allow_r12_b1_sampler_cfg_migration",
        action="store_true",
        help=(
            "Resume the known R12 B1 250K state after the sampler-only default CFG "
            "change to text/control 2.0."
        ),
    )
    parser.add_argument(
        "--fork_stage_c_schedule",
        action="store_true",
        help=(
            "Fork the exact R12 400K state into the registered fixed-18/80/2 "
            "Stage-C safety probe."
        ),
    )
    parser.add_argument(
        "--fork_stage_c_research",
        action="store_true",
        help=(
            "Fork the exact R12 400K state into a new research run while "
            "retaining the standard Stage-C schedule."
        ),
    )
    parser.add_argument(
        "--fork_stage_d_edit_calibration",
        action="store_true",
        help=(
            "Fork the completed R12 500K edit20 checkpoint into the Stage-D "
            "research calibration schedule. Model/optimizer state is retained; "
            "the first 10K changes only edit condition dropout, then a separate "
            "continuation may increase the edit replay ratio."
        ),
    )
    parser.add_argument(
        "--fork_stage_c_unified_edit",
        action="store_true",
        help=(
            "Fork the 400K shared model into the no-new-parameters unified "
            "T2M/control/Edit training protocol."
        ),
    )
    parser.add_argument(
        "--fork_stage_c_unified_edit40",
        action="store_true",
        help=(
            "Fork the Unified Edit V2 450K checkpoint into a fixed "
            "30/30/40 T2M/control/Edit continuation."
        ),
    )
    parser.add_argument(
        "--fork_kencoder_stage_be_edit",
        action="store_true",
        help=(
            "Start the registered 200K K-Encoder Edit bootstrap as a new "
            "research run while preserving the 4x32 global batch under 8x16."
        ),
    )
    parser.add_argument(
        "--fork_kencoder_stage_bc_ease_control",
        action="store_true",
        help=(
            "Fork the completed 250K K-Encoder Edit model into the registered "
            "10/70/20 T2M/control/Edit bootstrap with Ease conditioning."
        ),
    )
    parser.add_argument(
        "--research_fork",
        action="store_true",
        help=(
            "Start a new non-production R13 run from an existing checkpoint. "
            "This is intended for controlled scientific pilots."
        ),
    )
    parser.add_argument(
        "--research_treatment",
        choices=tuple(EDIT_RESEARCH_TREATMENTS),
        default="",
        help=(
            "Named R13 Edit pilot objective. Use with --research_fork for a new "
            "400K fork, or repeat the same name when resuming that pilot."
        ),
    )
    parser.add_argument(
        "--text_global_conditioning",
        choices=TEXT_GLOBAL_CONDITIONING_MODES,
        default="pooled_adaln",
        help=(
            "Text path used by all stages. qwen_tokens_only keeps Qwen token "
            "attention but removes CLIP pooled conditioning from AdaLN."
        ),
    )
    parser.add_argument(
        "--text_fusion_mode",
        choices=TEXT_FUSION_MODES,
        default="f00",
        help=(
            "Text/motion attention topology: f00 shared+bidirectional, "
            "f10 shared+asymmetric, f01 separate+bidirectional, "
            "f11 separate+asymmetric."
        ),
    )
    parser.add_argument(
        "--conditioning_architecture",
        choices=CONDITIONING_ARCHITECTURES,
        default="hytext_flux",
        help=(
            "hytext_flux is the existing Qwen-token/CLIP-pooled model; "
            "llm2vec_flux changes only the sentence encoder; "
            "llm2vec_kimodo_prefix also uses Kimodo prefix Transformers."
        ),
    )
    parser.add_argument(
        "--llm2vec_cache_dir",
        default="",
        help="Profile-aware offline LLM2Vec cache used by llm2vec treatments.",
    )
    parser.add_argument(
        "--ease_stats_dir",
        default="",
        help="Optional research override for the configured Ease stats directory.",
    )
    parser.add_argument(
        "--base_representation_loss_space",
        choices=REPRESENTATION_LOSS_SPACES,
        default="velocity_mse",
        help=(
            "Primary continuous HY273 objective for T2M/control samples. "
            "The default preserves the existing flow-velocity-equivalent loss."
        ),
    )
    parser.add_argument(
        "--base_contact_loss_space",
        choices=REPRESENTATION_LOSS_SPACES,
        default="velocity_mse",
        help=(
            "Primary unified-contact objective for T2M/control samples. "
            "Set with the representation objective for Kimodo-style x0 pilots."
        ),
    )
    parser.add_argument(
        "--g0_lr_override",
        type=float,
        default=None,
        help=(
            "Research override for the existing G0 backbone learning rate. "
            "Context-group learning rates and all other optimizer settings stay unchanged."
        ),
    )
    parser.add_argument(
        "--edit_same_source_groups",
        default=str(DEFAULT_SAME_SOURCE_EDIT_GROUPS),
        help="Research-only identical-source, two-target instruction group JSON.",
    )
    parser.add_argument(
        "--edit_same_source_min_target_mse",
        type=float,
        default=0.10,
        help="Minimum normalized continuous target-pair MSE for a hard donor.",
    )
    parser.add_argument("--smoke_steps", type=int, default=0)
    parser.add_argument("--batch_size_per_rank", type=int, default=0)
    parser.add_argument("--materialize_workers", type=int, default=-1)
    parser.add_argument(
        "--research_reshard_same_global_batch",
        action="store_true",
        help=(
            "Allow a research fork to change world-size/per-rank batch while "
            "preserving the checkpoint's effective global batch."
        ),
    )
    parser.add_argument(
        "--research_no_update",
        action="store_true",
        help="Run a finite research calibration trace without optimizer or EMA updates.",
    )
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--production", action="store_true")
    parser.add_argument(
        "--nonregression_artifact",
        default="",
    )
    parser.add_argument("--save_smoke", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.name = validate_run_name(args.name)
    if args.g0_lr_override is not None and (
        not math.isfinite(args.g0_lr_override) or args.g0_lr_override <= 0.0
    ):
        raise ValueError("--g0_lr_override must be a finite positive value")
    config, config_path = load_config(args.config)
    if args.ease_stats_dir:
        if "ease" not in config:
            raise ValueError("--ease_stats_dir requires an Ease-enabled stage config")
        config["ease"]["stats_dir"] = str(
            Path(args.ease_stats_dir).expanduser().resolve()
        )
    loss_weights = validate_frozen_contract(config)
    current_train_contract = str(cfg_get(config, "contract.version"))
    uses_llm2vec = args.conditioning_architecture != "hytext_flux"
    if uses_llm2vec:
        if not args.llm2vec_cache_dir:
            raise ValueError(
                f"{args.conditioning_architecture} requires --llm2vec_cache_dir"
            )
        if args.text_global_conditioning != "llm2vec_tokens_only":
            raise ValueError(
                "LLM2Vec treatments require "
                "--text_global_conditioning llm2vec_tokens_only"
            )
        if args.text_fusion_mode != "f00":
            raise ValueError(
                "The first controlled LLM2Vec treatments freeze "
                "--text_fusion_mode to f00"
            )
    else:
        if args.llm2vec_cache_dir:
            raise ValueError(
                "--llm2vec_cache_dir is only valid for an LLM2Vec treatment"
            )
        if args.text_global_conditioning == "llm2vec_tokens_only":
            raise ValueError(
                "llm2vec_tokens_only requires an LLM2Vec conditioning architecture"
            )
    contact_protocol = contact_protocol_for_config(config)
    unified_273_flow = uses_unified_273_flow(contact_protocol)
    expected_start = int(cfg_get(config, "stage.expected_start_step"))
    stop_step = int(cfg_get(config, "stage.stop_step"))
    expected_phase = TrainingPhase(int(cfg_get(config, "stage.phase_id")))
    schedule_version = str(cfg_get(config, "stage.schedule_version"))
    edit80_replay_fork = (
        expected_start == 450_000
        and stop_step == 500_000
        and expected_phase == TrainingPhase.STAGE_C
        and schedule_version
        == UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION
    )
    edit_loss_weights = unified_edit_loss_weights(config)
    effective_treatment_name = args.research_treatment
    if schedule_version in {
        KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
        KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    }:
        if (
            effective_treatment_name
            and effective_treatment_name != "no_rank_positive_only"
        ):
            raise ValueError(
                "K-Encoder Stage-BE freezes the Edit treatment to "
                "no_rank_positive_only"
            )
        effective_treatment_name = "no_rank_positive_only"
    treatment = resolve_edit_research_treatment(effective_treatment_name)
    edit_representation_loss_space = str(treatment["representation_loss_space"])
    edit_contact_loss_space = str(treatment["contact_loss_space"])
    edit_representation_multiplier = float(treatment["representation_multiplier"])
    edit_low_t_mix_prob = float(treatment["low_t_mix_prob"])
    edit_low_t_max = float(treatment["low_t_max"])
    edit_secondary_branch = str(treatment["secondary_branch"])
    edit_identity_base_scale = float(treatment["identity_base_scale"])
    edit_source_anchor_scale = float(treatment["source_anchor_scale"])
    edit_source_anchor_relative_margin = float(
        treatment["source_anchor_relative_margin"]
    )
    edit_instruction_rank_mode = str(treatment["instruction_rank_mode"])
    edit_instruction_rank_temperature = float(
        treatment["instruction_rank_temperature"]
    )
    edit_instruction_rank_multiplier = float(
        treatment["instruction_rank_multiplier"]
    )
    edit_instruction_negative_scope = str(
        treatment["instruction_negative_scope"]
    )
    edit_discrepancy_sample_scope = str(
        treatment["discrepancy_sample_scope"]
    )
    edit_discrepancy_x0_scale = float(treatment["discrepancy_x0_scale"])
    edit_discrepancy_fraction = float(treatment["discrepancy_fraction"])
    edit_temporal_scale = float(treatment["temporal_scale"])
    edit_temporal_vector_weight = float(treatment["temporal_vector_weight"])
    edit_temporal_speed_weight = float(treatment["temporal_speed_weight"])
    edit_temporal_background_weight = float(
        treatment["temporal_background_weight"]
    )
    edit_temporal_change_scale_mps = float(
        treatment["temporal_change_scale_mps"]
    )
    edit_temporal_smooth_l1_beta_mps = float(
        treatment["temporal_smooth_l1_beta_mps"]
    )
    same_source_instruction_donors: dict[str, str] = {}
    same_source_donor_spec: dict[str, Any] | None = None
    if (
        edit_secondary_branch == "same_source_instruction"
        or edit_discrepancy_sample_scope == "same_source_only"
    ):
        same_source_instruction_donors, same_source_donor_spec = (
            load_same_source_instruction_donors(
                args.edit_same_source_groups,
                minimum_target_pair_mse=args.edit_same_source_min_target_mse,
            )
        )
    research_overrides = {
        "research_treatment": dict(treatment),
        "edit_objective": json.loads(json.dumps(config.get("edit_objective"))),
        "base_representation_loss_space": str(
            args.base_representation_loss_space
        ),
        "base_contact_loss_space": str(args.base_contact_loss_space),
        "g0_lr_override": (
            None
            if args.g0_lr_override is None
            else float(args.g0_lr_override)
        ),
        "text_global_conditioning": str(args.text_global_conditioning),
        "text_fusion_mode": str(args.text_fusion_mode),
        "conditioning_architecture": str(args.conditioning_architecture),
        "llm2vec_cache_dir": str(
            Path(args.llm2vec_cache_dir).expanduser().resolve()
            if args.llm2vec_cache_dir
            else ""
        ),
    }
    if same_source_donor_spec is not None:
        research_overrides["same_source_instruction_donors"] = same_source_donor_spec
    fork_flags = (
        args.fork_from_r11_stage_a,
        args.fork_stage_c_schedule,
        args.fork_stage_c_research,
        args.fork_stage_d_edit_calibration,
        args.fork_stage_c_unified_edit,
        args.fork_stage_c_unified_edit40,
        args.fork_kencoder_stage_be_edit,
        args.fork_kencoder_stage_bc_ease_control,
        args.research_fork,
    )
    if sum(bool(value) for value in fork_flags) > 1:
        raise ValueError("Stage/research fork modes are mutually exclusive")
    has_explicit_research_treatment = bool(args.research_treatment)
    has_effective_research_treatment = bool(effective_treatment_name)
    collect_research_diagnostics = has_explicit_research_treatment
    if args.research_fork and not has_explicit_research_treatment:
        raise ValueError("--research_fork requires one --research_treatment")
    if has_explicit_research_treatment and not args.resume:
        raise ValueError("Edit research treatments require a parent or pilot checkpoint")
    if (schedule_version in UNIFIED_EDIT_SCHEDULE_VERSIONS) != (
        edit_loss_weights is not None
    ):
        raise ValueError(
            "The unified Edit schedule and edit_objective config must be enabled together"
        )
    if phase_for_step(expected_start) != expected_phase:
        raise ValueError("Stage expected_start_step and phase_id disagree")
    if not expected_start < stop_step <= int(cfg_get(config, "training.max_global_step")):
        raise ValueError("Invalid stage step interval")
    if expected_start > 0 and not args.resume:
        raise ValueError("Stages B1/B2/C require an exact previous-stage checkpoint")
    if args.fork_kencoder_stage_be_edit:
        if (
            current_train_contract != R13_TRAIN_CONTRACT
            or expected_start != 200_000
            or stop_step != 250_000
            or expected_phase != TrainingPhase.STAGE_B1
            or schedule_version != KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION
        ):
            raise ValueError(
                "K-Encoder Stage-BE fork requires its registered "
                "200K->250K R13 config"
            )
        if not args.resume:
            raise ValueError("K-Encoder Stage-BE fork requires the 200K checkpoint")
        if not args.research_reshard_same_global_batch:
            raise ValueError(
                "The 4x32 Stage-A to 8x16 Stage-BE fork requires the "
                "same-global-batch reshard"
            )
    if args.fork_kencoder_stage_bc_ease_control:
        if (
            current_train_contract != R13_TRAIN_CONTRACT
            or expected_start != 250_000
            or stop_step != 400_000
            or expected_phase != TrainingPhase.STAGE_B2
            or schedule_version
            != KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION
        ):
            raise ValueError(
                "K-Encoder Ease-Control fork requires its registered "
                "250K->400K R13 config"
            )
        if not args.resume:
            raise ValueError(
                "K-Encoder Ease-Control fork requires the completed 250K checkpoint"
            )
        if not bool(config.get("ease", {}).get("enabled", False)):
            raise ValueError("K-Encoder Ease-Control fork requires ease.enabled=true")
    if args.fork_from_r11_stage_a:
        if current_train_contract != R12_TRAIN_CONTRACT:
            raise ValueError("--fork_from_r11_stage_a requires the R12 root-mask contract")
        if expected_start != 200_000 or expected_phase != TrainingPhase.STAGE_B1:
            raise ValueError("R11 Stage-A forking is only valid for the R12 B1 stage")
        if not args.resume_sha256:
            raise ValueError("R11 Stage-A forking requires a pinned --resume_sha256")
        if args.resume_sha256.lower() != R12_ORIGIN_PARENT_SHA256:
            raise ValueError(
                "R12 must fork from the frozen R11 200K checkpoint content SHA"
            )
    if args.fork_stage_c_schedule:
        if (
            args.fork_from_r11_stage_a
            or args.fork_stage_c_research
            or args.fork_stage_d_edit_calibration
            or args.fork_stage_c_unified_edit
            or args.fork_stage_c_unified_edit40
        ):
            raise ValueError("Stage fork modes are mutually exclusive")
        if current_train_contract != R12_TRAIN_CONTRACT:
            raise ValueError("Stage-C schedule fork requires the R12 contract")
        if not args.resume or not args.resume_sha256:
            raise ValueError("Stage-C schedule fork requires a pinned resume checkpoint")
        if (
            expected_start != 400_000
            or stop_step != 405_000
            or expected_phase != TrainingPhase.STAGE_C
            or schedule_version != STAGE_C_SAFE_MIX_SCHEDULE_VERSION
        ):
            raise ValueError("Stage-C schedule fork requires the registered 400K->405K config")
    if args.fork_stage_c_research:
        if (
            args.fork_from_r11_stage_a
            or args.fork_stage_c_schedule
            or args.fork_stage_d_edit_calibration
            or args.fork_stage_c_unified_edit
            or args.fork_stage_c_unified_edit40
        ):
            raise ValueError("Stage fork modes are mutually exclusive")
        if current_train_contract != R12_TRAIN_CONTRACT:
            raise ValueError("Stage-C research fork requires the R12 contract")
        if not args.resume or not args.resume_sha256:
            raise ValueError("Stage-C research fork requires a pinned resume checkpoint")
        if (
            expected_start != 400_000
            or stop_step != 500_000
            or expected_phase != TrainingPhase.STAGE_C
            or schedule_version != STAGE_C_EDIT20_SCHEDULE_VERSION
        ):
            raise ValueError(
                "Stage-C research fork requires the registered fixed-edit20 config"
            )
    if args.fork_stage_d_edit_calibration:
        if (
            args.fork_from_r11_stage_a
            or args.fork_stage_c_schedule
            or args.fork_stage_c_research
            or args.fork_stage_c_unified_edit
            or args.fork_stage_c_unified_edit40
        ):
            raise ValueError("Stage fork modes are mutually exclusive")
        if current_train_contract != R12_TRAIN_CONTRACT:
            raise ValueError("Stage-D calibration fork requires the R12 contract")
        if not args.resume:
            raise ValueError("Stage-D calibration fork requires the completed 500K checkpoint")
        if (
            expected_start != 500_000
            or stop_step != 510_000
            or expected_phase != TrainingPhase.STAGE_C
            or schedule_version != STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
        ):
            raise ValueError(
                "Stage-D calibration fork requires the registered 500K->510K pilot config"
            )
    if args.fork_stage_c_unified_edit:
        if (
            args.fork_from_r11_stage_a
            or args.fork_stage_c_schedule
            or args.fork_stage_c_research
            or args.fork_stage_d_edit_calibration
            or args.fork_stage_c_unified_edit40
        ):
            raise ValueError("Stage fork modes are mutually exclusive")
        if current_train_contract != R12_TRAIN_CONTRACT:
            raise ValueError("Unified Edit fork requires the R12 shared model")
        if not args.resume:
            raise ValueError("Unified Edit fork requires the completed 400K checkpoint")
        if (
            expected_start != 400_000
            or stop_step != 500_000
            or expected_phase != TrainingPhase.STAGE_C
            or schedule_version
            not in {
                UNIFIED_EDIT_V2_SCHEDULE_VERSION,
                UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
            }
        ):
            raise ValueError("Unified Edit fork requires its registered 400K->500K config")
    if args.fork_stage_c_unified_edit40:
        if (
            args.fork_from_r11_stage_a
            or args.fork_stage_c_schedule
            or args.fork_stage_c_research
            or args.fork_stage_d_edit_calibration
            or args.fork_stage_c_unified_edit
        ):
            raise ValueError("Stage fork modes are mutually exclusive")
        if current_train_contract != R12_TRAIN_CONTRACT:
            raise ValueError("Unified Edit40 fork requires the R12 shared model")
        if not args.resume or not args.resume_sha256:
            raise ValueError(
                "Unified Edit40 fork requires the pinned Unified Edit V2 450K checkpoint"
            )
        if (
            expected_start != 450_000
            or stop_step != 500_000
            or expected_phase != TrainingPhase.STAGE_C
            or schedule_version != UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
        ):
            raise ValueError(
                "Unified Edit40 fork requires its registered 450K->500K config"
            )
    if args.research_fork:
        if args.production:
            raise ValueError("--research_fork cannot be used with --production")
        if current_train_contract != R13_TRAIN_CONTRACT:
            raise ValueError("--research_fork currently supports only R13")
        if not args.resume:
            raise ValueError("--research_fork requires a parent checkpoint")
        standard_400k_fork = (
            expected_start == 400_000
            and expected_phase == TrainingPhase.STAGE_C
            and schedule_version
            in {
                UNIFIED_EDIT_V2_SCHEDULE_VERSION,
                UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
            }
        )
        if not (standard_400k_fork or edit80_replay_fork):
            raise ValueError(
                "R13 Edit research pilots must use either the registered 400K "
                "Stage-C config or the 450K Positive-only Edit80 continuation"
            )
        if edit80_replay_fork and args.research_treatment != "no_rank_positive_only":
            raise ValueError(
                "The Edit80 continuation must preserve the Positive-only objective"
            )
    if args.smoke_steps < 0:
        raise ValueError("--smoke_steps cannot be negative")
    stage_b_boundary_reshard = (
        bool(args.resume)
        and expected_start == 200_000
        and expected_phase == TrainingPhase.STAGE_B1
        and schedule_version
        in {
            R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
            KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
        }
    )
    if (
        args.research_reshard_same_global_batch
        and not args.research_fork
        and not stage_b_boundary_reshard
    ):
        raise ValueError(
            "Research topology reshard requires either --research_fork or the "
            "exact R16 Stage-A-to-Stage-B boundary"
        )
    if args.research_no_update:
        if not args.research_fork or args.smoke_steps <= 0:
            raise ValueError(
                "--research_no_update requires a finite --research_fork smoke trace"
            )
        if args.save_smoke:
            raise ValueError("A no-update calibration trace cannot save a checkpoint")
    if (
        current_train_contract == R12_TRAIN_CONTRACT
        and args.resume
        and not args.resume_sha256
        and not args.fork_stage_d_edit_calibration
        and not args.fork_stage_c_unified_edit
        and not args.fork_stage_c_unified_edit40
    ):
        raise ValueError("Every R12 resume requires a pinned --resume_sha256")
    if args.production and args.smoke_steps:
        raise ValueError("Production and smoke modes are mutually exclusive")
    production_gate_identity = None
    if args.production:
        production_gate_identity = validate_production_gate(
            config, args.nonregression_artifact
        )
        configured_batch = int(cfg_get(config, "training.batch_size_per_rank"))
        if args.batch_size_per_rank not in {0, configured_batch}:
            raise RuntimeError("Production batch size is frozen by config")
        if args.resume and not args.resume_sha256:
            raise RuntimeError("Production resume requires --resume_sha256")

    device, rank, world_size, local_rank = setup_distributed()
    try:
        if args.production and world_size != 8:
            raise RuntimeError(f"R11 production requires 8 ranks, got {world_size}")
        if device.type != "cuda" and args.production:
            raise RuntimeError("Production training requires CUDA")
        seed = int(cfg_get(config, "training.seed"))
        seed_model_initialization(seed)
        asset_identity = validate_assets(
            config,
            include_full_preflight=bool(args.production),
            text_cache_dir=(
                args.llm2vec_cache_dir if uses_llm2vec else None
            ),
            expected_text_cache_format=(
                LLM2VEC_CACHE_FORMAT
                if uses_llm2vec
                else PROFILE_CACHE_FORMAT
            ),
        )
        ease_stats_identity = validate_ease_stats(config)
        code_identity = current_code_identity()
        batch_size = (
            int(args.batch_size_per_rank)
            if args.batch_size_per_rank > 0
            else int(cfg_get(config, "training.batch_size_per_rank"))
        )
        workers = (
            int(args.materialize_workers)
            if args.materialize_workers >= 0
            else int(cfg_get(config, "data.materialize_workers"))
        )
        output_root = Path(
            args.output_dir or str(cfg_get(config, "training.output_dir"))
        ).expanduser().resolve()
        run_dir = output_root / args.name
        origin_parent: dict[str, Any] | None = None
        is_fresh_fork = bool(
            args.fork_from_r11_stage_a
            or args.fork_stage_c_schedule
            or args.fork_stage_c_research
            or args.fork_stage_d_edit_calibration
            or args.fork_stage_c_unified_edit
            or args.fork_stage_c_unified_edit40
            or args.fork_kencoder_stage_be_edit
            or args.fork_kencoder_stage_bc_ease_control
            or args.research_fork
        )
        if args.resume and not is_fresh_fork:
            if not run_dir.is_dir() or not (run_dir / "run_identity.json").is_file():
                raise RuntimeError("Resume requires the existing run directory and identity")
            run_identity = json.loads(
                (run_dir / "run_identity.json").read_text(encoding="utf-8")
            )
            if run_identity.get("run_name") != args.name:
                raise RuntimeError("Resume run identity/name mismatch")
            run_uuid = str(run_identity.get("run_uuid", ""))
            if not run_uuid:
                raise RuntimeError("Resume run identity has no UUID")
            if current_train_contract == R12_TRAIN_CONTRACT:
                origin_parent = validate_r12_origin_parent_identity(
                    run_identity.get("origin_parent")
                )
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"Fresh run refuses non-empty directory: {run_dir}")
            run_uuid = str(uuid.uuid4()) if rank == 0 else ""
            if args.fork_from_r11_stage_a:
                origin_parent = r12_origin_parent_identity(args.resume)
            elif args.fork_kencoder_stage_be_edit:
                source_identity_path = (
                    Path(args.resume).expanduser().resolve().parent.parent
                    / "run_identity.json"
                )
                if not source_identity_path.is_file():
                    raise RuntimeError(
                        "K-Encoder Stage-BE source run identity is missing"
                    )
                source_identity = json.loads(
                    source_identity_path.read_text(encoding="utf-8")
                )
                origin_parent = {
                    "kind": "kencoder_stage_be_edit_fork",
                    "source_run_identity": source_identity,
                    "checkpoint": str(Path(args.resume).expanduser().resolve()),
                }
            elif args.fork_kencoder_stage_bc_ease_control:
                source_identity_path = (
                    Path(args.resume).expanduser().resolve().parent.parent
                    / "run_identity.json"
                )
                if not source_identity_path.is_file():
                    raise RuntimeError(
                        "K-Encoder Ease-Control source run identity is missing"
                    )
                source_identity = json.loads(
                    source_identity_path.read_text(encoding="utf-8")
                )
                origin_parent = {
                    "kind": "kencoder_stage_bc_ease_control_fork",
                    "source_run_identity": source_identity,
                    "checkpoint": str(Path(args.resume).expanduser().resolve()),
                }
            elif (
                args.fork_stage_c_schedule
                or args.fork_stage_c_research
                or args.fork_stage_d_edit_calibration
                or args.fork_stage_c_unified_edit
                or args.fork_stage_c_unified_edit40
            ):
                source_identity_path = (
                    Path(args.resume).expanduser().resolve().parent.parent
                    / "run_identity.json"
                )
                if not source_identity_path.is_file():
                    raise RuntimeError("Stage-C fork source run identity is missing")
                source_identity = json.loads(
                    source_identity_path.read_text(encoding="utf-8")
                )
                origin_parent = validate_r12_origin_parent_identity(
                    source_identity.get("origin_parent")
                )
            elif args.research_fork:
                source_identity_path = (
                    Path(args.resume).expanduser().resolve().parent.parent
                    / "run_identity.json"
                )
                if not source_identity_path.is_file():
                    raise RuntimeError("Research-fork source run identity is missing")
                source_identity = json.loads(
                    source_identity_path.read_text(encoding="utf-8")
                )
                registered_source = source_identity
                if edit80_replay_fork:
                    registered_source = (
                        source_identity.get("origin_parent", {})
                        .get("source_run_identity", {})
                    )
                if (
                    registered_source.get("run_name")
                    != R13_EDIT_RESEARCH_PARENT_RUN_NAME
                    or registered_source.get("run_uuid")
                    != R13_EDIT_RESEARCH_PARENT_RUN_UUID
                ):
                    raise RuntimeError(
                        "R13 Edit research fork must descend from the registered "
                        "shared 400K run"
                    )
                origin_parent = {
                    "kind": "r13_research_fork",
                    "source_run_identity": source_identity,
                    "checkpoint": str(Path(args.resume).expanduser().resolve()),
                }
        if is_distributed():
            objects = [run_uuid]
            dist.broadcast_object_list(objects, src=0)
            run_uuid = str(objects[0])
        runtime_identity = {
            "format": "hy273_multitask_runtime_identity_v1",
            "run_name": args.name,
            "run_uuid": run_uuid,
            "world_size": world_size,
            "batch_size_per_rank": batch_size,
            "effective_global_batch": world_size * batch_size,
            "materialize_workers": workers,
            "gradient_sync": {
                "mode": str(cfg_get(config, "training.gradient_sync_mode")),
                "bucket_cap_mb": float(
                    cfg_get(config, "training.gradient_sync_bucket_cap_mb")
                ),
            },
            "config_path": str(config_path),
            "config_sha256": canonical_sha(config),
            "base_contract_sha256": base_contract_sha(config),
            "asset_identity": asset_identity,
            "ease_stats_identity": ease_stats_identity,
            "code_identity": code_identity,
            "production": bool(args.production),
            "research_reshard_same_global_batch": bool(
                args.research_reshard_same_global_batch
            ),
            "research_no_update": bool(args.research_no_update),
            "production_gate_identity": production_gate_identity,
            "research_overrides": research_overrides,
            "origin_parent": origin_parent,
            "immediate_resume_parent": (
                {
                    "checkpoint": str(Path(args.resume).expanduser().resolve()),
                    "checkpoint_sha256": str(args.resume_sha256).lower(),
                }
                if args.resume
                else None
            ),
        }
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config_resolved.json").write_text(
                json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            (run_dir / "asset_identity.json").write_text(
                json.dumps(asset_identity, indent=2, sort_keys=True), encoding="utf-8"
            )
            (run_dir / f"config_resolved_{cfg_get(config, 'stage.name')}.json").write_text(
                json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            if not args.resume or is_fresh_fork:
                (run_dir / "run_identity.json").write_text(
                    json.dumps(
                        {
                            "format": "hy273_multitask_run_identity_v1",
                            "run_name": args.name,
                            "run_uuid": run_uuid,
                            "origin_parent": origin_parent,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
        if is_distributed():
            dist.barrier()

        stats_root = Path(cfg_get(config, "data.stats_root"))
        normalizer = HY273Normalizer.from_data_root(
            stats_root / "full",
            stats_dir=stats_root / "full",
            variance_eps=float(cfg_get(config, "model.stats_variance_eps")),
            normalize_contacts=unified_273_flow,
        ).to(device)
        model = create_model(
            config,
            source_fusion_mode=str(treatment["source_fusion_mode"]),
            text_global_conditioning=str(args.text_global_conditioning),
            text_fusion_mode=str(args.text_fusion_mode),
            conditioning_architecture=str(args.conditioning_architecture),
            llm2vec_cache_dir=str(args.llm2vec_cache_dir),
        ).to(device)
        optimizer_group_defs, optimizer_manifest = optimizer_groups(
            model,
            expected_start,
            schedule_version,
            g0_lr_override=args.g0_lr_override,
        )
        gradient_sync_mode = str(cfg_get(config, "training.gradient_sync_mode"))
        if gradient_sync_mode not in {"fixed_bucket", "native_ddp"}:
            raise ValueError(f"Unknown gradient sync mode: {gradient_sync_mode!r}")
        gradient_synchronizer = (
            FixedBucketGradientSynchronizer(
                model,
                bucket_cap_mb=float(
                    cfg_get(config, "training.gradient_sync_bucket_cap_mb")
                ),
            )
            if gradient_sync_mode == "fixed_bucket"
            else None
        )
        optimizer_manifest.pop("manifest_sha256")
        optimizer_manifest["gradient_sync"] = {
            "mode": gradient_sync_mode,
            "plan": (
                gradient_synchronizer.manifest
                if gradient_synchronizer is not None
                else None
            ),
        }
        optimizer_manifest["manifest_sha256"] = canonical_sha(optimizer_manifest)
        optimizer = torch.optim.AdamW(optimizer_group_defs, betas=(0.9, 0.999), eps=1e-8)
        schedule_fork_step = None
        if (
            current_train_contract == R13_TRAIN_CONTRACT
            and expected_start == 200_000
            and schedule_version
            in {
                R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
                KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
            }
        ):
            schedule_fork_step = expected_start
        elif (
            args.fork_kencoder_stage_bc_ease_control
            and expected_start == 250_000
            and schedule_version
            == KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION
        ):
            schedule_fork_step = 250_000
        elif (
            current_train_contract == R13_TRAIN_CONTRACT
            and expected_start in {400_000, 450_000}
            and schedule_version in UNIFIED_EDIT_SCHEDULE_VERSIONS
        ):
            schedule_fork_step = expected_start
        elif args.fork_stage_c_unified_edit40:
            schedule_fork_step = 450_000
        elif args.fork_stage_d_edit_calibration:
            schedule_fork_step = 500_000
        elif (
            args.fork_stage_c_schedule
            or args.fork_stage_c_research
            or args.fork_stage_c_unified_edit
        ):
            schedule_fork_step = 400_000
        batcher = HY273MultitaskStepBatcher(
            train_manifest=cfg_get(config, "data.train_manifest"),
            run_seed=seed,
            world_size=world_size,
            rank=rank,
            batch_size_per_rank=batch_size,
            materialize_workers=workers,
            sort_window_batches=int(cfg_get(config, "data.sort_window_batches")),
            verify_payload_hash=bool(cfg_get(config, "data.verify_payload_hash")),
            schedule_version=schedule_version,
            allow_schedule_fork_at_step=schedule_fork_step,
        )
        ema = initialize_ema(model)
        context_update_count = 0
        ease_update_count = 0
        ema_update_count = 0
        global_step = expected_start
        preserve_parent_rank_ratio_objective = False

        if args.resume:
            resume_path = Path(args.resume).expanduser().resolve()
            if args.resume_sha256:
                actual = sha256_file(resume_path)
                if actual != args.resume_sha256.lower():
                    raise RuntimeError(
                        f"Resume SHA mismatch: expected={args.resume_sha256}, actual={actual}"
                    )
            checkpoint = torch.load(
                resume_path, map_location="cpu", mmap=True, weights_only=False
            )
            if checkpoint.get("format") != CHECKPOINT_FORMAT:
                raise ValueError("Checkpoint format mismatch")
            if stage_b_boundary_reshard and args.research_reshard_same_global_batch:
                validate_exact_kencoder_stage_b_reshard(
                    checkpoint,
                    current_world_size=world_size,
                    current_batch_size=batch_size,
                )
                preserve_parent_rank_ratio_objective = True
            if args.fork_kencoder_stage_be_edit:
                if (
                    conditioning_architecture_from_checkpoint(checkpoint)
                    != "llm2vec_flux"
                    or text_global_conditioning_from_checkpoint(checkpoint)
                    != "llm2vec_tokens_only"
                    or text_fusion_mode_from_checkpoint(checkpoint) != "f00"
                ):
                    raise RuntimeError(
                        "Stage-BE parent is not the selected K-Encoder architecture"
                    )
                parent_representation, parent_contact = (
                    base_loss_spaces_from_checkpoint(checkpoint)
                )
                if (
                    parent_representation != "velocity_mse"
                    or parent_contact != "velocity_mse"
                ):
                    raise RuntimeError(
                        "Stage-BE parent does not use the K-Encoder velocity loss"
                    )
                parent_cache = Path(
                    llm2vec_cache_dir_from_checkpoint(checkpoint)
                ).expanduser().resolve()
                requested_cache = Path(args.llm2vec_cache_dir).expanduser().resolve()
                if parent_cache != requested_cache:
                    raise RuntimeError(
                        "Stage-BE parent and runtime use different LLM2Vec caches"
                    )
            if args.fork_kencoder_stage_bc_ease_control:
                if int(checkpoint.get("next_global_step", -1)) != 250_000:
                    raise RuntimeError(
                        "Ease-Control must fork from the completed 250K parent"
                    )
                if (
                    checkpoint.get("high_level_schedule_version")
                    != KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION
                ):
                    raise RuntimeError(
                        "Ease-Control parent is not the K-Encoder Stage-BE run"
                    )
                if (
                    conditioning_architecture_from_checkpoint(checkpoint)
                    != "llm2vec_flux"
                    or text_global_conditioning_from_checkpoint(checkpoint)
                    != "llm2vec_tokens_only"
                    or text_fusion_mode_from_checkpoint(checkpoint) != "f00"
                ):
                    raise RuntimeError(
                        "Ease-Control parent is not the selected K-Encoder architecture"
                    )
                parent_representation, parent_contact = (
                    base_loss_spaces_from_checkpoint(checkpoint)
                )
                if (
                    parent_representation != "velocity_mse"
                    or parent_contact != "velocity_mse"
                ):
                    raise RuntimeError(
                        "Ease-Control parent does not use the selected velocity loss"
                    )
                parent_cache = Path(
                    llm2vec_cache_dir_from_checkpoint(checkpoint)
                ).expanduser().resolve()
                requested_cache = Path(
                    args.llm2vec_cache_dir
                ).expanduser().resolve()
                if parent_cache != requested_cache:
                    raise RuntimeError(
                        "Ease-Control parent and runtime use different LLM2Vec caches"
                    )
            if (
                args.research_fork
                and int(checkpoint.get("next_global_step", -1)) != expected_start
            ):
                raise ValueError(
                    "A fresh R13 Edit research fork must start at its configured boundary"
                )
            if args.research_fork and not edit80_replay_fork and (
                checkpoint.get("run_name") != R13_EDIT_RESEARCH_PARENT_RUN_NAME
                or checkpoint.get("run_uuid") != R13_EDIT_RESEARCH_PARENT_RUN_UUID
            ):
                raise RuntimeError(
                    "R13 Edit research checkpoint is not from the registered shared run"
                )
            if args.research_fork and edit80_replay_fork:
                if (
                    checkpoint.get("high_level_schedule_version")
                    != UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION
                ):
                    raise RuntimeError(
                        "The Edit80 continuation requires a decomposed-CFG 450K parent"
                    )
                parent_runtime = checkpoint.get("runtime_identity")
                parent_treatment = (
                    parent_runtime.get("research_overrides", {})
                    .get("research_treatment", {})
                    if isinstance(parent_runtime, dict)
                    else {}
                )
                if parent_treatment.get("name") != "no_rank_positive_only":
                    raise RuntimeError(
                        "The Edit80 continuation requires the Positive-only 450K parent"
                    )
            if args.fork_from_r11_stage_a:
                validate_r12_origin_checkpoint(
                    checkpoint, checkpoint_sha256=args.resume_sha256
                )
            elif checkpoint.get("train_contract") != current_train_contract:
                raise ValueError("Checkpoint training contract mismatch")
            if checkpoint.get("ema_schedule_version") != EMA_SCHEDULE:
                raise ValueError("Checkpoint EMA schedule mismatch")
            embedded_config = checkpoint.get("config")
            if not isinstance(embedded_config, dict):
                raise ValueError("Checkpoint is missing its resolved config")
            if checkpoint.get("config_sha256") != canonical_sha(embedded_config):
                raise RuntimeError("Checkpoint embedded config SHA is invalid")
            if args.fork_kencoder_stage_be_edit and (
                cfg_get(embedded_config, "stage.name") != "stage_a_t2m"
                or embedded_config.get("edit_objective") is not None
            ):
                raise RuntimeError(
                    "Stage-BE must fork from the T2M-only Stage-A config"
                )
            if args.fork_kencoder_stage_bc_ease_control:
                if (
                    cfg_get(embedded_config, "stage.name")
                    != "stage_be_kencoder_edit"
                    or embedded_config.get("edit_objective")
                    != config.get("edit_objective")
                    or embedded_config.get("ease") is not None
                ):
                    raise RuntimeError(
                        "Ease-Control must preserve the Stage-BE Edit objective "
                        "and add Ease only at the 250K fork"
                    )
            if args.fork_stage_c_unified_edit40:
                validate_unified_edit40_objective(embedded_config, config)
            expected_base_contract_sha = base_contract_sha(config)
            if args.fork_from_r11_stage_a:
                validate_frozen_contract(embedded_config)
                if cfg_get(embedded_config, "contract.version") != R11_TRAIN_CONTRACT:
                    raise RuntimeError("R12 fork source embedded config is not R11")
                if checkpoint.get("base_contract_sha256") != base_contract_sha(
                    embedded_config
                ):
                    raise RuntimeError("R12 fork source embedded base contract is invalid")
            elif args.fork_stage_d_edit_calibration:
                validate_frozen_contract(embedded_config)
                if checkpoint.get("base_contract_sha256") != base_contract_sha(
                    embedded_config
                ):
                    raise RuntimeError(
                        "Stage-D parent embedded base contract is inconsistent"
                    )
            else:
                if checkpoint.get("base_contract_sha256") != expected_base_contract_sha:
                    raise RuntimeError("Checkpoint/current cross-stage base contract mismatch")
                if base_contract_sha(embedded_config) != expected_base_contract_sha:
                    raise RuntimeError("Checkpoint embedded base contract is inconsistent")
            checkpoint_schedule_version = checkpoint.get(
                "high_level_schedule_version"
            )
            if checkpoint_schedule_version != schedule_version:
                valid_schedule_fork = (
                    (
                        args.fork_stage_c_schedule
                        or args.fork_stage_c_research
                        or args.fork_stage_c_unified_edit
                    )
                    and checkpoint_schedule_version == HIGH_LEVEL_SCHEDULE_VERSION
                    and schedule_version
                    in {
                        STAGE_C_SAFE_MIX_SCHEDULE_VERSION,
                        STAGE_C_EDIT20_SCHEDULE_VERSION,
                        UNIFIED_EDIT_V2_SCHEDULE_VERSION,
                        UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
                    }
                    and int(checkpoint.get("next_global_step", -1)) == 400_000
                )
                valid_schedule_fork = valid_schedule_fork or (
                    args.fork_stage_d_edit_calibration
                    and checkpoint_schedule_version == STAGE_C_EDIT20_SCHEDULE_VERSION
                    and schedule_version == STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION
                    and int(checkpoint.get("next_global_step", -1)) == 500_000
                )
                valid_schedule_fork = valid_schedule_fork or (
                    args.fork_stage_c_unified_edit40
                    and checkpoint_schedule_version == UNIFIED_EDIT_V2_SCHEDULE_VERSION
                    and schedule_version == UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
                    and int(checkpoint.get("next_global_step", -1)) == 450_000
                )
                valid_schedule_fork = valid_schedule_fork or (
                    args.fork_kencoder_stage_bc_ease_control
                    and checkpoint_schedule_version
                    == KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION
                    and schedule_version
                    == KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION
                    and int(checkpoint.get("next_global_step", -1)) == 250_000
                )
                valid_schedule_fork = valid_schedule_fork or (
                    current_train_contract == R13_TRAIN_CONTRACT
                    and (
                        (
                            checkpoint_schedule_version
                            in {
                                HIGH_LEVEL_SCHEDULE_VERSION,
                                R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
                            }
                            and schedule_version
                            in {
                                UNIFIED_EDIT_V2_SCHEDULE_VERSION,
                                UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION,
                            }
                            and int(checkpoint.get("next_global_step", -1)) == 400_000
                        )
                        or (
                            checkpoint_schedule_version
                            == UNIFIED_EDIT_V2_SCHEDULE_VERSION
                            and schedule_version
                            == UNIFIED_EDIT_V2_EDIT40_SCHEDULE_VERSION
                            and int(checkpoint.get("next_global_step", -1)) == 450_000
                        )
                        or (
                            checkpoint_schedule_version
                            == UNIFIED_EDIT_DECOMPOSED_CFG_SCHEDULE_VERSION
                            and schedule_version
                            == UNIFIED_EDIT_DECOMPOSED_CFG_EDIT80_SCHEDULE_VERSION
                            and int(checkpoint.get("next_global_step", -1)) == 450_000
                        )
                        or (
                            checkpoint_schedule_version
                            == HIGH_LEVEL_SCHEDULE_VERSION
                            and schedule_version
                            in {
                                R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
                                KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
                            }
                            and int(checkpoint.get("next_global_step", -1)) == 200_000
                        )
                    )
                )
                if not valid_schedule_fork:
                    raise RuntimeError("Checkpoint high_level_schedule_version mismatch")
            for key, expected in {
                "hml_inner_schedule_version": HML_INNER_SCHEDULE_VERSION,
                "sample_rng_version": SAMPLE_RNG_VERSION,
                "bucket_plan_version": BUCKET_PLAN_VERSION,
            }.items():
                if checkpoint.get(key) != expected:
                    raise RuntimeError(f"Checkpoint {key} mismatch")
            if not args.fork_from_r11_stage_a:
                if args.fork_stage_d_edit_calibration:
                    resume_code_identity_mode = "stage_d_500k_research_fork"
                elif args.fork_stage_c_unified_edit:
                    resume_code_identity_mode = "stage_c_unified_edit_research_fork"
                elif args.fork_stage_c_unified_edit40:
                    resume_code_identity_mode = (
                        validate_unified_edit40_fork_code_identity(
                            checkpoint, code_identity
                        )
                    )
                elif args.research_fork:
                    resume_code_identity_mode = describe_research_resume_code_identity(
                        checkpoint, code_identity
                    )
                elif args.fork_stage_c_schedule or args.fork_stage_c_research:
                    resume_code_identity_mode = (
                        validate_stage_c_schedule_fork_code_identity(
                            checkpoint, code_identity
                        )
                    )
                elif not args.production:
                    resume_code_identity_mode = (
                        describe_research_resume_code_identity(
                            checkpoint, code_identity
                        )
                    )
                else:
                    resume_code_identity_mode = validate_resume_code_identity(
                        checkpoint,
                        code_identity,
                        allow_r12_b1_sampler_cfg_migration=(
                            args.allow_r12_b1_sampler_cfg_migration
                        ),
                    )
                if rank == 0 and resume_code_identity_mode != "exact":
                    print(
                        "resume_code_identity_mode=" + resume_code_identity_mode,
                        flush=True,
                    )
            if not is_fresh_fork and (
                checkpoint.get("run_name") != args.name
                or checkpoint.get("run_uuid") != run_uuid
            ):
                raise RuntimeError("Checkpoint run lineage mismatch")
            if not is_fresh_fork:
                checkpoint_runtime = checkpoint.get("runtime_identity")
                checkpoint_text_mode = text_global_conditioning_from_checkpoint(
                    checkpoint
                )
                if checkpoint_text_mode != args.text_global_conditioning:
                    raise RuntimeError(
                        "Text global conditioning changed across resume: "
                        f"checkpoint={checkpoint_text_mode!r}, "
                        f"requested={args.text_global_conditioning!r}"
                    )
                checkpoint_architecture = (
                    conditioning_architecture_from_checkpoint(checkpoint)
                )
                if checkpoint_architecture != args.conditioning_architecture:
                    raise RuntimeError(
                        "Conditioning architecture changed across resume: "
                        f"checkpoint={checkpoint_architecture!r}, "
                        f"requested={args.conditioning_architecture!r}"
                    )
                checkpoint_base_representation, checkpoint_base_contact = (
                    base_loss_spaces_from_checkpoint(checkpoint)
                )
                if (
                    checkpoint_base_representation
                    != args.base_representation_loss_space
                    or checkpoint_base_contact
                    != args.base_contact_loss_space
                ):
                    raise RuntimeError(
                        "Base loss space changed across resume: "
                        f"checkpoint=({checkpoint_base_representation!r}, "
                        f"{checkpoint_base_contact!r}), "
                        f"requested=({args.base_representation_loss_space!r}, "
                        f"{args.base_contact_loss_space!r})"
                    )
                checkpoint_overrides = (
                    checkpoint_runtime.get("research_overrides", {})
                    if isinstance(checkpoint_runtime, dict)
                    else {}
                )
                checkpoint_g0_lr_override = checkpoint_overrides.get(
                    "g0_lr_override"
                )
                if checkpoint_g0_lr_override != args.g0_lr_override:
                    raise RuntimeError(
                        "G0 learning-rate override changed across resume: "
                        f"checkpoint={checkpoint_g0_lr_override!r}, "
                        f"requested={args.g0_lr_override!r}"
                    )
                if uses_llm2vec:
                    checkpoint_cache = str(
                        Path(
                            llm2vec_cache_dir_from_checkpoint(checkpoint)
                        ).expanduser().resolve()
                    )
                    requested_cache = str(
                        Path(args.llm2vec_cache_dir).expanduser().resolve()
                    )
                    if checkpoint_cache != requested_cache:
                        raise RuntimeError(
                            "LLM2Vec cache changed across resume: "
                            f"checkpoint={checkpoint_cache!r}, "
                            f"requested={requested_cache!r}"
                        )
                if has_effective_research_treatment:
                    validate_research_resume_objective(
                        checkpoint_runtime, research_overrides
                    )
            validate_checkpoint_text_fusion_mode(
                checkpoint, args.text_fusion_mode
            )
            validate_checkpoint_text_fusion_implementation(
                checkpoint,
                code_identity,
                enforce=bool(args.production),
            )
            if current_train_contract == R12_TRAIN_CONTRACT and not args.fork_from_r11_stage_a:
                checkpoint_runtime = checkpoint.get("runtime_identity")
                if not isinstance(checkpoint_runtime, dict):
                    raise RuntimeError("R12 checkpoint has no runtime identity")
                checkpoint_origin = validate_r12_origin_parent_identity(
                    checkpoint_runtime.get("origin_parent")
                )
                if checkpoint_origin != origin_parent:
                    raise RuntimeError("R12 checkpoint/run origin lineage mismatch")
            if checkpoint.get("asset_identity") != asset_identity:
                raise RuntimeError("Checkpoint/data asset identity mismatch")
            if not args.fork_kencoder_stage_bc_ease_control and (
                checkpoint.get("ease_stats_identity") != ease_stats_identity
            ):
                raise RuntimeError("Checkpoint/configured Ease stats semantics mismatch")
            if args.fork_kencoder_stage_bc_ease_control:
                load_parent_model_with_new_ease(model, checkpoint["model"])
                migrate_optimizer_state_with_new_ease(
                    optimizer,
                    parent_state=checkpoint["optimizer"],
                    parent_manifest=checkpoint["optimizer_group_manifest"],
                    current_manifest=optimizer_manifest,
                )
            else:
                if checkpoint.get("optimizer_group_manifest") != optimizer_manifest:
                    raise RuntimeError(
                        "Checkpoint optimizer parameter-name manifest mismatch"
                    )
                model.load_state_dict(checkpoint["model"], strict=True)
                optimizer.load_state_dict(checkpoint["optimizer"])
            assert_model_ease_stats(model, ease_stats_identity)
            global_step = int(checkpoint["next_global_step"])
            if int(checkpoint.get("phase_id", -1)) != int(phase_for_step(global_step)):
                raise RuntimeError("Checkpoint saved phase_id is inconsistent")
            if not expected_start <= global_step < stop_step:
                raise RuntimeError(
                    f"Checkpoint step {global_step} is outside stage interval "
                    f"[{expected_start},{stop_step})"
                )
            phase_matches_stage = phase_for_step(global_step) == expected_phase
            if (
                schedule_version == R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION
                and expected_phase == TrainingPhase.STAGE_B1
                and 200_000 <= global_step < 400_000
            ):
                phase_matches_stage = True
            if not phase_matches_stage:
                raise RuntimeError("Checkpoint next step resolves to a different phase")
            batcher.load_state_dict(
                checkpoint["batcher"],
                allow_same_global_batch_reshard=bool(
                    args.research_reshard_same_global_batch
                ),
                preserve_source_rank_ratio_objective=(
                    preserve_parent_rank_ratio_objective
                ),
            )
            if batcher.scheduler.state.next_step != global_step:
                raise RuntimeError("Batcher/checkpoint next step mismatch")
            context_update_count = int(checkpoint["context_update_count"])
            ease_update_count = int(checkpoint.get("ease_update_count", 0))
            ema_update_count = int(checkpoint.get("ema_update_count", -1))
            expected_ema_count = expected_ema_updates(
                global_step, int(cfg_get(config, "training.ema_every"))
            )
            if ema_update_count != expected_ema_count:
                raise RuntimeError(
                    f"Checkpoint EMA update count mismatch: {ema_update_count}/{expected_ema_count}"
                )
            _assert_context_optimizer_count(
                optimizer, model, context_update_count
            )
            _assert_ease_optimizer_count(
                optimizer, model, ease_update_count
            )
            if args.fork_kencoder_stage_bc_ease_control:
                ema = initialize_ema_with_new_ease(model, checkpoint["ema"])
            else:
                ema = {
                    key: value.to(device=device)
                    for key, value in checkpoint["ema"].items()
                }
            if not _normalizer_matches(normalizer, checkpoint["normalizer"]):
                raise RuntimeError("Checkpoint normalizer differs from frozen stats")
            random.setstate(checkpoint["rng"]["python"])
            np.random.set_state(checkpoint["rng"]["numpy"])
            torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
            if torch.cuda.is_available():
                cuda_rng = cuda_rng_states_for_resume(
                    checkpoint["rng"]["torch_cuda"],
                    device_count=torch.cuda.device_count(),
                    allow_same_global_batch_reshard=bool(
                        args.research_reshard_same_global_batch
                    ),
                )
                torch.cuda.set_rng_state_all(cuda_rng)
            del checkpoint
        ratio_process_group = create_local_ratio_process_group(
            world_size=world_size,
            rank=rank,
            group_size=batcher.ratio_group_size,
        )
        runtime_identity["ratio_partition"] = {
            "world_size": batcher.ratio_partition_world_size,
            "batch_size": batcher.ratio_partition_batch_size,
            "physical_group_size": batcher.ratio_group_size,
            "preserved_from_parent": batcher.ratio_group_size > 1,
            "introduced_at_this_resume": preserve_parent_rank_ratio_objective,
        }
        if rank == 0:
            (run_dir / "runtime_identity.json").write_text(
                json.dumps(
                    runtime_identity,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if is_distributed():
            dist.barrier()
        apply_optimizer_phase(
            optimizer,
            global_step,
            schedule_version,
            g0_lr_override=args.g0_lr_override,
        )

        raw_model = model
        if world_size > 1:
            training_model: torch.nn.Module = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
                static_graph=gradient_sync_mode == "native_ddp",
            )
        else:
            training_model = model
        training_model.train()
        precision = str(cfg_get(config, "training.precision"))
        if precision not in {"bf16", "fp32"}:
            raise ValueError("R11 trainer supports only bf16 or fp32")
        autocast_enabled = precision == "bf16" and device.type == "cuda"
        autocast_dtype = torch.bfloat16
        term_names = (
            "repr_root_xyz",
            "repr_heading",
            "repr_joint_position",
            "repr_global_rot6d",
            "repr_velocity",
            "contact_all",
            "clean_root_velocity",
            "clean_joint_velocity",
            "foot_lock",
            "fk_consistency",
            "control_continuous",
            "control_contact",
        )
        loss_window = LossMetricWindow(term_names, device)
        update_sampler = UpdateSampler(raw_model)
        aux_sums: dict[str, float] = {}
        aux_steps = 0
        log_every = int(cfg_get(config, "training.log_every"))
        latest_every = int(cfg_get(config, "training.latest_every"))
        archive_every = int(cfg_get(config, "training.archive_every"))
        ema_every = int(cfg_get(config, "training.ema_every"))
        ema_decay = float(cfg_get(config, "training.ema_decay"))
        grad_clip = float(cfg_get(config, "training.gradient_clip"))
        actual_stop = (
            min(stop_step, global_step + int(args.smoke_steps))
            if args.smoke_steps
            else stop_step
        )
        if rank == 0:
            total_parameters = sum(parameter.numel() for parameter in raw_model.parameters())
            trainable_parameters = sum(
                parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad
            )
            print(
                json.dumps(
                    {
                        "event": "training_start",
                        "run": args.name,
                        "stage": cfg_get(config, "stage.name"),
                        "schedule_version": schedule_version,
                        "contact_protocol": contact_protocol,
                        "start_step": global_step,
                        "stop_step": actual_stop,
                        "world_size": world_size,
                        "batch_size_per_rank": batch_size,
                        "effective_global_batch": world_size * batch_size,
                        "ratio_partition_world_size": (
                            batcher.ratio_partition_world_size
                        ),
                        "ratio_partition_batch_size": (
                            batcher.ratio_partition_batch_size
                        ),
                        "ratio_physical_group_size": batcher.ratio_group_size,
                        "total_parameters": total_parameters,
                        "trainable_parameters": trainable_parameters,
                        "gradient_sync": optimizer_manifest["gradient_sync"],
                        "loss_weights": loss_weights.__dict__,
                        "edit_objective": (
                            None
                            if edit_loss_weights is None
                            else edit_loss_weights.__dict__
                        ),
                        "research_overrides": research_overrides,
                        "asset_identity": asset_identity,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

        while global_step < actual_stop:
            step_started = time.perf_counter()
            apply_optimizer_phase(
                optimizer,
                global_step,
                schedule_version,
                g0_lr_override=args.g0_lr_override,
            )
            batch_started = time.perf_counter()
            batch, stream, plan_trace = batcher.next_batch(global_step)
            data_seconds = time.perf_counter() - batch_started
            if is_distributed():
                stream_tensor = torch.tensor(
                    [int(stream), int(stream)], device=device, dtype=torch.long
                )
                dist.all_reduce(stream_tensor[:1], op=dist.ReduceOp.MIN)
                dist.all_reduce(stream_tensor[1:], op=dist.ReduceOp.MAX)
                if stream_tensor[0] != stream_tensor[1]:
                    raise RuntimeError("DDP ranks selected different high-level streams")

            target_physical = batch["target_motion"].to(device=device, dtype=torch.float32)
            condition = batch["condition"].to(device)
            unified_edit_step = (
                edit_loss_weights is not None and stream == TrainStream.MOTION_EDIT
            )
            if unified_edit_step:
                decomposed_cfg_edit = (
                    schedule_version in DECOMPOSED_CFG_EDIT_SCHEDULE_VERSIONS
                )
                allowed_patterns = (
                    {
                        EditConditionPattern.SOURCE_TEXT,
                        EditConditionPattern.SOURCE_IDENTITY,
                        EditConditionPattern.TEXT_ONLY,
                        EditConditionPattern.UNCONDITIONAL,
                        EditConditionPattern.SOURCE_TEXT_CONTROL,
                    }
                    if decomposed_cfg_edit
                    else {
                        EditConditionPattern.SOURCE_TEXT,
                        EditConditionPattern.SOURCE_TEXT_CONTROL,
                    }
                )
                if any(
                    plan.edit_pattern not in allowed_patterns
                    for plan in batch["plans"]
                ):
                    raise RuntimeError(
                        "Unified Edit batch contains a forbidden condition pattern"
                    )
                if not bool((condition.task_id == int(TaskId.EDIT)).all()):
                    raise RuntimeError("Unified Edit batch contains a non-Edit task")
                expected_source_present = torch.tensor(
                    [bool(plan.edit_pattern.uses_source) for plan in batch["plans"]],
                    device=device,
                    dtype=torch.bool,
                )
                actual_source_present = condition.source_present.any(dim=-1)
                if not torch.equal(actual_source_present, expected_source_present):
                    raise RuntimeError(
                        "Unified Edit source presence disagrees with its condition pattern"
                    )
                expected_text_present = [
                    bool(plan.edit_pattern.uses_text) for plan in batch["plans"]
                ]
                actual_text_present = [bool(str(value).strip()) for value in batch["texts"]]
                if actual_text_present != expected_text_present:
                    raise RuntimeError(
                        "Unified Edit text presence disagrees with its condition pattern"
                    )
                allowed_capabilities = torch.tensor(
                    [
                        int(CapabilityId.MOTION_EDIT),
                        int(CapabilityId.MOTION_EDIT_CONTROL),
                    ],
                    device=device,
                    dtype=condition.capability_id.dtype,
                )
                if not bool(
                    torch.isin(
                        condition.capability_id, allowed_capabilities
                    ).all()
                ):
                    raise RuntimeError(
                        "Unified Edit batch contains an invalid capability"
                    )
            has_source_free_edit = bool(
                (
                    (condition.task_id == int(TaskId.EDIT))
                    & ~condition.source_present.any(dim=1)
                ).any()
            )
            condition.validate(
                max_target_frames=int(cfg_get(config, "data.max_target_frames")),
                v1_strict=not has_source_free_edit,
            )
            observed_physical, hard_mask, control_modes = build_hard_controls(
                target_physical=target_physical,
                condition=condition,
                plans=batch["plans"],
                global_step=global_step,
                config=config,
                manifest_sha256=batcher.manifest_sha256,
                run_seed=seed,
            )
            x0_norm = normalizer.normalize(target_physical)
            observed_norm = normalizer.normalize(observed_physical)
            if unified_273_flow:
                timesteps, unified_noise, edit_low_t_selected = (
                    build_stateless_unified_273_flow_inputs(
                        plans=batch["plans"],
                        x0_norm=x0_norm,
                        manifest_sha256=batcher.manifest_sha256,
                        run_seed=seed,
                        config=config,
                        edit_low_t_mix_prob=edit_low_t_mix_prob,
                        edit_low_t_max=edit_low_t_max,
                    )
                )
                flow_state = build_unified_273_flow_state(
                    x0_norm,
                    observed_norm,
                    hard_mask,
                    timesteps,
                    noise=unified_noise,
                )
                identity_state = (
                    build_equal_length_source_identity_flow(
                        condition=condition,
                        normalizer=normalizer,
                        timesteps=timesteps,
                        unified_noise=unified_noise,
                    )
                    if unified_edit_step
                    and edit_secondary_branch == "source_identity"
                    else None
                )
            else:
                identity_state = None
                edit_low_t_selected = torch.zeros(
                    target_physical.shape[0], device=device, dtype=torch.bool
                )
                timesteps, continuous_noise, contact_aux = build_stateless_flow_inputs(
                    plans=batch["plans"],
                    x0_norm=x0_norm,
                    manifest_sha256=batcher.manifest_sha256,
                    run_seed=seed,
                    config=config,
                )
                flow_state = build_flow_state(
                    x0_norm,
                    observed_norm,
                    hard_mask,
                    timesteps,
                    noise_cont=continuous_noise,
                    contact_aux=contact_aux,
                )
            joint_edit_condition_mask = torch.tensor(
                [
                    plan.edit_pattern
                    in {
                        EditConditionPattern.SOURCE_TEXT,
                        EditConditionPattern.SOURCE_TEXT_CONTROL,
                    }
                    for plan in batch["plans"]
                ],
                device=device,
                dtype=torch.bool,
            )
            temporal_sample_mask = joint_edit_condition_mask
            temporal_active_mask = torch.zeros_like(temporal_sample_mask)
            if unified_edit_step and edit_temporal_scale > 0.0:
                if condition.source_native_lengths.shape[1] != 1:
                    raise RuntimeError(
                        "Physical temporal Edit treatment requires one source slot"
                    )
                target_lengths = condition.target_valid.sum(dim=-1).long()
                source_lengths = condition.source_native_lengths[:, 0].long()
                temporal_active_mask = (
                    temporal_sample_mask
                    & condition.source_present[:, 0]
                    & (source_lengths == target_lengths)
                    & (target_lengths >= 2)
                )
            same_source_eligible_mask = torch.zeros(
                target_physical.shape[0], device=device, dtype=torch.bool
            )
            if unified_edit_step and same_source_instruction_donors:
                same_source_eligible_mask = torch.tensor(
                    [str(uid) in same_source_instruction_donors for uid in batch["uids"]],
                    device=device,
                    dtype=torch.bool,
                ) & joint_edit_condition_mask
            discrepancy_sample_mask = (
                same_source_eligible_mask
                if edit_discrepancy_sample_scope == "same_source_only"
                else torch.ones_like(same_source_eligible_mask)
            )
            discrepancy_mask_bundle: SourceTargetDiscrepancyMask | None = None
            if (
                unified_edit_step
                and collect_research_diagnostics
                and edit_discrepancy_x0_scale > 0.0
            ):
                discrepancy_mask_bundle = build_source_target_discrepancy_mask(
                    source_physical=condition.source_motion,
                    source_lengths=condition.source_native_lengths,
                    target_physical=target_physical,
                    target_valid=condition.target_valid,
                    hard_mask=hard_mask,
                    fraction=edit_discrepancy_fraction,
                )
                discrepancy_mask_bundle.mask = (
                    discrepancy_mask_bundle.mask
                    & discrepancy_sample_mask[:, None, None]
                )
            optimizer.zero_grad(set_to_none=True)
            manual_sync = (
                gradient_sync_mode == "fixed_bucket"
                and world_size > 1
                and isinstance(training_model, DDP)
            )
            sync_context = training_model.no_sync() if manual_sync else nullcontext()
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_enabled,
                ):
                    secondary_prediction = None
                    paired_prediction_for_grad = None
                    paired_details_for_grad: KimodoContextFlowOutput | None = None
                    same_source_negative_fraction = 0.0
                    instruction_rank_sample_mask = None
                    if unified_edit_step and edit_secondary_branch != "none":
                        if edit_secondary_branch in {
                            "shuffled_instruction",
                            "same_source_instruction",
                        }:
                            secondary_model_in = flow_state["model_in"]
                            if edit_secondary_branch == "same_source_instruction":
                                secondary_texts, used_same_source = (
                                    same_source_instruction_texts(
                                        batch["texts"],
                                        batch["uids"],
                                        same_source_instruction_donors,
                                        fallback_mode=(
                                            "self"
                                            if edit_instruction_negative_scope
                                            == "same_source_only"
                                            else "mismatched"
                                        ),
                                    )
                                )
                                same_source_negative_fraction = sum(
                                    used_same_source
                                ) / float(len(used_same_source))
                                if (
                                    edit_instruction_negative_scope
                                    == "same_source_only"
                                ):
                                    instruction_rank_sample_mask = torch.tensor(
                                        used_same_source,
                                        device=device,
                                        dtype=torch.bool,
                                    )
                            else:
                                secondary_texts = mismatched_instruction_texts(
                                    batch["texts"]
                                )
                        elif edit_secondary_branch == "source_identity":
                            assert identity_state is not None
                            secondary_model_in = identity_state["model_in"]
                            secondary_texts = [""] * len(batch["texts"])
                        else:
                            raise RuntimeError(
                                f"Unknown Edit secondary branch {edit_secondary_branch!r}"
                            )
                        repeated_condition = repeat_condition_batch(condition, 2)
                        paired_output = training_model(
                            torch.cat(
                                [flow_state["model_in"], secondary_model_in],
                                dim=0,
                            ),
                            t=torch.cat([timesteps, timesteps], dim=0),
                            c_dir=torch.cat(
                                [condition.frame_gauge_dir] * 2, dim=0
                            ),
                            text=[*batch["texts"], *secondary_texts],
                            length_mask=torch.cat(
                                [condition.target_valid] * 2, dim=0
                            ),
                            x_self_cond=None,
                            text_drop_prob=0.0,
                            condition=repeated_condition,
                            return_details=collect_research_diagnostics,
                        )
                        if collect_research_diagnostics:
                            if not isinstance(paired_output, KimodoContextFlowOutput):
                                raise TypeError("Research Edit forward must return details")
                            paired_details_for_grad = paired_output
                            paired_prediction = paired_output.prediction
                            paired_prediction.retain_grad()
                            for diagnostic_tensor in (
                                paired_output.text_tokens,
                                paired_output.text_pooled,
                                paired_output.context_root,
                                paired_output.context_body,
                            ):
                                diagnostic_tensor.retain_grad()
                            paired_prediction_for_grad = paired_prediction
                        else:
                            if not torch.is_tensor(paired_output):
                                raise TypeError("Standard Edit forward must return a tensor")
                            paired_prediction = paired_output
                        prediction, secondary_prediction = paired_prediction.chunk(
                            2, dim=0
                        )
                    else:
                        single_output = training_model(
                            flow_state["model_in"],
                            t=timesteps,
                            c_dir=condition.frame_gauge_dir,
                            text=batch["texts"],
                            length_mask=condition.target_valid,
                            x_self_cond=None,
                            text_drop_prob=0.0,
                            condition=condition,
                            return_details=bool(
                                unified_edit_step and collect_research_diagnostics
                            ),
                        )
                        if unified_edit_step and collect_research_diagnostics:
                            if not isinstance(single_output, KimodoContextFlowOutput):
                                raise TypeError(
                                    "Research Edit forward must return details"
                                )
                            paired_details_for_grad = single_output
                            prediction = single_output.prediction
                            prediction.retain_grad()
                            for diagnostic_tensor in (
                                single_output.text_tokens,
                                single_output.text_pooled,
                                single_output.context_root,
                                single_output.context_body,
                            ):
                                diagnostic_tensor.retain_grad()
                            paired_prediction_for_grad = prediction
                        else:
                            if not torch.is_tensor(single_output):
                                raise TypeError("Model forward must return a tensor")
                            prediction = single_output
                    common_loss_kwargs = dict(
                        x0_target_norm=x0_norm,
                        x0_target_physical=target_physical,
                        hard_observed_norm=observed_norm,
                        hard_mask=hard_mask,
                        target_valid=condition.target_valid,
                        timesteps=timesteps,
                        normalizer=normalizer,
                        global_step=global_step,
                        weights=loss_weights,
                    )
                    if unified_273_flow:
                        loss_bundle = compute_hy273_unified_flow_loss(
                            x0_hat_norm=prediction,
                            z_imputed=flow_state["z_imp"],
                            representation_loss_space=(
                                edit_representation_loss_space
                                if unified_edit_step
                                else args.base_representation_loss_space
                            ),
                            contact_loss_space=(
                                edit_contact_loss_space
                                if unified_edit_step
                                else args.base_contact_loss_space
                            ),
                            representation_multiplier=(
                                edit_representation_multiplier
                                if unified_edit_step
                                else 1.0
                            ),
                            **common_loss_kwargs,
                        )
                    else:
                        loss_bundle = compute_hy273_multitask_loss(
                            x0_hat_cont=prediction[..., :CONT_DIM],
                            contact_logits=prediction[..., CONTACT_SLICE],
                            z_cont_imputed=flow_state["z_cont_imp"],
                            **common_loss_kwargs,
                        )
                    loss_bundle = apply_preserved_ratio_partition(
                        loss_bundle,
                        process_group=ratio_process_group,
                        group_size=batcher.ratio_group_size,
                    )
                    edit_loss_bundle: UnifiedEditLossBundle | None = None
                    identity_loss_bundle: HY273MultitaskLossBundle | None = None
                    source_anchor_bundle: SourceAnchorLossBundle | None = None
                    discrepancy_loss_bundle: (
                        SourceTargetDiscrepancyLossBundle | None
                    ) = None
                    temporal_loss_bundle: PhysicalTemporalEditLossBundle | None = None
                    total_loss = loss_bundle.total
                    if unified_edit_step:
                        assert edit_loss_weights is not None
                        edit_loss_bundle = compute_unified_edit_loss(
                            correct_x0_hat_cont=prediction[..., :CONT_DIM],
                            shuffled_x0_hat_cont=(
                                secondary_prediction[..., :CONT_DIM]
                                if edit_secondary_branch
                                in {
                                    "shuffled_instruction",
                                    "same_source_instruction",
                                }
                                else None
                            ),
                            x0_target_norm=x0_norm,
                            target_valid=condition.target_valid,
                            hard_mask=hard_mask,
                            weights=replace(
                                edit_loss_weights,
                                instruction_rank_scale=(
                                    edit_loss_weights.instruction_rank_scale
                                    * edit_instruction_rank_multiplier
                                ),
                            ),
                            instruction_rank_mode=edit_instruction_rank_mode,
                            instruction_rank_temperature=(
                                edit_instruction_rank_temperature
                            ),
                            instruction_rank_sample_mask=(
                                instruction_rank_sample_mask
                            ),
                            instruction_rank_denominator=(
                                globally_normalized_eligible_denominator(
                                    instruction_rank_sample_mask,
                                    world_size=world_size,
                                )
                                if instruction_rank_sample_mask is not None
                                else None
                            ),
                        )
                        total_loss = total_loss + edit_loss_bundle.total
                        if edit_secondary_branch == "source_identity":
                            assert identity_state is not None
                            assert secondary_prediction is not None
                            identity_loss_bundle = compute_hy273_unified_flow_loss(
                                x0_hat_norm=secondary_prediction,
                                z_imputed=identity_state["z_imp"],
                                x0_target_norm=identity_state["x0_norm"],
                                x0_target_physical=identity_state["x0_physical"],
                                hard_observed_norm=identity_state["observed_norm"],
                                hard_mask=identity_state["hard_mask"],
                                target_valid=identity_state["target_valid"],
                                timesteps=timesteps,
                                normalizer=normalizer,
                                global_step=global_step,
                                weights=loss_weights,
                                representation_loss_space="velocity_mse",
                                contact_loss_space="velocity_mse",
                            )
                            identity_loss_bundle = apply_preserved_ratio_partition(
                                identity_loss_bundle,
                                process_group=ratio_process_group,
                                group_size=batcher.ratio_group_size,
                            )
                            source_anchor_bundle = compute_source_anchor_loss(
                                correct_x0_hat_cont=prediction[..., :CONT_DIM],
                                source_x0_norm=identity_state["x0_norm"],
                                x0_target_norm=x0_norm,
                                target_valid=condition.target_valid,
                                hard_mask=hard_mask,
                                sample_mask=identity_state["exact_pair"],
                                scale=edit_source_anchor_scale,
                                relative_margin=edit_source_anchor_relative_margin,
                            )
                            total_loss = (
                                total_loss
                                + edit_identity_base_scale * identity_loss_bundle.total
                                + source_anchor_bundle.total
                            )
                        objective_without_discrepancy = total_loss
                        if discrepancy_mask_bundle is not None:
                            discrepancy_loss_bundle = (
                                compute_source_target_discrepancy_x0_loss(
                                    correct_x0_hat_cont=prediction[..., :CONT_DIM],
                                    x0_target_norm=x0_norm,
                                    discrepancy_mask=discrepancy_mask_bundle.mask,
                                    scale=edit_discrepancy_x0_scale,
                                    active_denominator=(
                                        globally_normalized_eligible_denominator(
                                            discrepancy_mask_bundle.mask.reshape(
                                                discrepancy_mask_bundle.mask.shape[0], -1
                                            ).any(dim=-1),
                                            world_size=world_size,
                                        )
                                    ),
                                )
                            )
                            total_loss = total_loss + discrepancy_loss_bundle.total
                        objective_without_temporal = total_loss
                        if edit_temporal_scale > 0.0:
                            temporal_loss_bundle = compute_physical_temporal_edit_loss(
                                x0_hat_norm=prediction,
                                x0_target_physical=target_physical,
                                source_physical=condition.source_motion,
                                source_lengths=condition.source_native_lengths,
                                target_valid=condition.target_valid,
                                sample_mask=temporal_sample_mask,
                                normalizer=normalizer,
                                scale=edit_temporal_scale,
                                vector_weight=edit_temporal_vector_weight,
                                speed_weight=edit_temporal_speed_weight,
                                background_weight=edit_temporal_background_weight,
                                change_scale_mps=edit_temporal_change_scale_mps,
                                smooth_l1_beta_mps=(
                                    edit_temporal_smooth_l1_beta_mps
                                ),
                                fps=float(loss_weights.fps),
                                active_denominator=(
                                    globally_normalized_eligible_denominator(
                                        temporal_active_mask,
                                        world_size=world_size,
                                    )
                                ),
                            )
                            total_loss = total_loss + temporal_loss_bundle.total
                    else:
                        objective_without_discrepancy = total_loss
                        objective_without_temporal = total_loss
                if not bool(torch.isfinite(total_loss.detach())):
                    raise RuntimeError(f"Non-finite loss at step={global_step}")
                loss_window.add(loss_bundle, condition, stream)
                discrepancy_unit_output_grad_rms = 0.0
                discrepancy_weighted_output_grad_rms = 0.0
                temporal_unit_output_grad_rms = 0.0
                temporal_weighted_output_grad_rms = 0.0
                objective_output_grad_rms = 0.0
                if discrepancy_loss_bundle is not None:
                    objective_output_grad = torch.autograd.grad(
                        objective_without_discrepancy,
                        prediction,
                        retain_graph=True,
                    )[0]
                    discrepancy_unit_output_grad = torch.autograd.grad(
                        discrepancy_loss_bundle.raw,
                        prediction,
                        retain_graph=True,
                    )[0]
                    objective_output_grad_rms = tensor_masked_rms(
                        objective_output_grad, condition.target_valid
                    )
                    discrepancy_unit_output_grad_rms = tensor_masked_rms(
                        discrepancy_unit_output_grad, condition.target_valid
                    )
                    discrepancy_weighted_output_grad_rms = (
                        discrepancy_unit_output_grad_rms * edit_discrepancy_x0_scale
                    )
                if temporal_loss_bundle is not None:
                    objective_output_grad = torch.autograd.grad(
                        objective_without_temporal,
                        prediction,
                        retain_graph=True,
                    )[0]
                    temporal_unit_output_grad = torch.autograd.grad(
                        temporal_loss_bundle.raw,
                        prediction,
                        retain_graph=True,
                    )[0]
                    objective_output_grad_rms = tensor_masked_rms(
                        objective_output_grad, condition.target_valid
                    )
                    temporal_unit_output_grad_rms = tensor_masked_rms(
                        temporal_unit_output_grad, condition.target_valid
                    )
                    temporal_weighted_output_grad_rms = (
                        temporal_unit_output_grad_rms * edit_temporal_scale
                    )
                total_loss.backward()
                correct_output_grad_rms = 0.0
                secondary_output_grad_rms = 0.0
                text_token_grad_rms = 0.0
                text_pooled_grad_rms = 0.0
                source_root_token_grad_rms = 0.0
                source_body_token_grad_rms = 0.0
                if (
                    paired_prediction_for_grad is not None
                    and paired_prediction_for_grad.grad is not None
                ):
                    if paired_prediction_for_grad.shape[0] == condition.batch_size:
                        correct_grad = paired_prediction_for_grad.grad
                        secondary_grad = None
                    else:
                        correct_grad, secondary_grad = paired_prediction_for_grad.grad.chunk(
                            2, dim=0
                        )
                    correct_output_grad_rms = tensor_masked_rms(
                        correct_grad, condition.target_valid
                    )
                    if secondary_grad is not None:
                        secondary_valid = (
                            identity_state["target_valid"]
                            if identity_state is not None
                            else condition.target_valid
                        )
                        secondary_output_grad_rms = tensor_masked_rms(
                            secondary_grad, secondary_valid
                        )
                if paired_details_for_grad is not None:
                    details = paired_details_for_grad
                    batch_count = condition.target_valid.shape[0]
                    text_tokens_grad = details.text_tokens.grad
                    text_pooled_grad = details.text_pooled.grad
                    context_root_grad = details.context_root.grad
                    context_body_grad = details.context_body.grad
                    if text_tokens_grad is not None:
                        text_token_grad_rms = tensor_masked_rms(
                            text_tokens_grad[:batch_count],
                            ~details.text_padding_mask[:batch_count],
                        )
                    if text_pooled_grad is not None:
                        text_pooled_grad_rms = tensor_masked_rms(
                            text_pooled_grad[:batch_count],
                            torch.ones(
                                batch_count,
                                device=text_pooled_grad.device,
                                dtype=torch.bool,
                            ),
                        )
                    if context_root_grad is not None:
                        source_root_token_grad_rms = tensor_masked_rms(
                            context_root_grad[:batch_count],
                            condition.target_valid,
                        )
                    if context_body_grad is not None:
                        source_body_token_grad_rms = tensor_masked_rms(
                            context_body_grad[:batch_count],
                            condition.target_valid,
                        )
            gradient_sync_started = time.perf_counter()
            if manual_sync:
                assert gradient_synchronizer is not None
                gradient_synchronizer.synchronize()
            gradient_sync_seconds = time.perf_counter() - gradient_sync_started
            source_present = bool(condition.source_present.any().item())
            edit_task_present = bool((condition.task_id == int(TaskId.EDIT)).any().item())
            context_active = source_present or edit_task_present
            ease_active_tensor = torch.tensor(
                [int(bool(condition.ease_present.any().item()))],
                device=device,
                dtype=torch.int32,
            )
            if is_distributed():
                dist.all_reduce(ease_active_tensor, op=dist.ReduceOp.MAX)
            ease_active = bool(ease_active_tensor.item())
            assert_and_mask_context_gradients(
                raw_model,
                context_active=context_active,
                global_step=global_step,
                optimizer=optimizer,
                schedule_version=schedule_version,
            )
            assert_and_mask_ease_gradients(
                raw_model,
                ease_active=ease_active,
                optimizer=optimizer,
            )
            base_grad_norm = tensor_group_norm(raw_model.base_parameters())
            context_grad_norm = tensor_group_norm(
                (*raw_model.context_weight_parameters(), *raw_model.context_bias_parameters())
            )
            ease_grad_norm = tensor_group_norm(
                (*raw_model.ease_weight_parameters(), *raw_model.ease_bias_parameters())
            )
            context_steps_before = _context_optimizer_steps(optimizer, raw_model)
            ease_steps_before = _ease_optimizer_steps(optimizer, raw_model)
            before_update = update_sampler.snapshot()
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip).item()
            )
            if not math.isfinite(total_grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at step={global_step}")
            if not args.research_no_update:
                optimizer.step()
                if not context_active:
                    context_steps_after = _context_optimizer_steps(optimizer, raw_model)
                    if context_steps_before != context_steps_after:
                        raise RuntimeError("Context-inactive step advanced context Adam state")
                else:
                    context_update_count += 1
                if not ease_active:
                    ease_steps_after = _ease_optimizer_steps(optimizer, raw_model)
                    if ease_steps_before != ease_steps_after:
                        raise RuntimeError(
                            "Ease-inactive step advanced Ease Adam state"
                        )
                else:
                    ease_update_count += 1
            update_metrics = update_sampler.differences(before_update)
            if not args.research_no_update and global_step % ema_every == 0:
                update_ema(ema, raw_model, ema_decay)
                ema_update_count += 1
            next_global_step = global_step + 1
            step_seconds = time.perf_counter() - step_started
            units = probability_units_for_step(global_step, schedule_version)
            has_root_pattern = torch.tensor(
                [
                    any(
                        token.startswith(("root_sparse", "root_dense"))
                        for token in mode.removeprefix("mixed:").split("+")
                    )
                    for mode in control_modes
                ],
                device=hard_mask.device,
                dtype=torch.bool,
            )
            has_effective_heading = hard_mask[..., HEADING_SLICE].any(dim=(1, 2))
            root_position_only = int(
                (has_root_pattern & ~has_effective_heading).sum().item()
            )
            root_position_rotation = int(
                (has_root_pattern & has_effective_heading).sum().item()
            )
            identity_base_raw = (
                0.0
                if identity_loss_bundle is None
                else float(identity_loss_bundle.total.detach().item())
            )
            identity_base_weighted = edit_identity_base_scale * identity_base_raw
            source_anchor_raw = (
                0.0
                if source_anchor_bundle is None
                else float(source_anchor_bundle.raw.detach().item())
            )
            source_anchor_weighted = (
                0.0
                if source_anchor_bundle is None
                else float(source_anchor_bundle.weighted.detach().item())
            )
            standard_edit_aux = (
                0.0
                if edit_loss_bundle is None
                else float(edit_loss_bundle.total.detach().item())
            )
            discrepancy_raw = (
                0.0
                if discrepancy_loss_bundle is None
                else float(discrepancy_loss_bundle.raw.detach().item())
            )
            discrepancy_weighted = (
                0.0
                if discrepancy_loss_bundle is None
                else float(discrepancy_loss_bundle.weighted.detach().item())
            )
            temporal_raw = (
                0.0
                if temporal_loss_bundle is None
                else float(temporal_loss_bundle.raw.detach().item())
            )
            temporal_weighted = (
                0.0
                if temporal_loss_bundle is None
                else float(temporal_loss_bundle.weighted.detach().item())
            )
            complete_edit_aux = (
                standard_edit_aux
                + identity_base_weighted
                + source_anchor_weighted
                + discrepancy_weighted
                + temporal_weighted
            )
            discrepancy_metrics: dict[str, float] = {}
            if discrepancy_mask_bundle is not None:
                valid_count = max(int(condition.target_valid.sum().item()), 1)
                joint_support = valid_count * int(
                    discrepancy_mask_bundle.body_joint_time_mask.shape[-1]
                )
                discrepancy_metrics.update(
                    {
                        "batch/edit_discrepancy_root_time_fraction_sum": float(
                            discrepancy_mask_bundle.root_time_mask.sum().item()
                            / valid_count
                        ),
                        "batch/edit_discrepancy_body_joint_time_fraction_sum": float(
                            discrepancy_mask_bundle.body_joint_time_mask.sum().item()
                            / max(joint_support, 1)
                        ),
                        "batch/edit_discrepancy_velocity_joint_time_fraction_sum": float(
                            discrepancy_mask_bundle.velocity_joint_time_mask.sum().item()
                            / max(joint_support, 1)
                        ),
                        "batch/edit_discrepancy_root_active_fraction_sum": float(
                            discrepancy_mask_bundle.root_time_mask.any(dim=1)
                            .float()
                            .mean()
                            .item()
                        ),
                        "batch/edit_discrepancy_empty_pre_fraction_sum": float(
                            (~discrepancy_mask_bundle.pre_intersection_mask.reshape(
                                batch_size, -1
                            ).any(dim=-1))
                            .float()
                            .mean()
                            .item()
                        ),
                        "batch/edit_discrepancy_empty_post_fraction_sum": float(
                            (~discrepancy_mask_bundle.mask.reshape(batch_size, -1).any(dim=-1))
                            .float()
                            .mean()
                            .item()
                        ),
                        "batch/edit_discrepancy_exact_identity_fraction_sum": float(
                            discrepancy_mask_bundle.exact_identity.float().mean().item()
                        ),
                        "batch/edit_discrepancy_equal_length_fraction_sum": float(
                            discrepancy_mask_bundle.equal_length.float().mean().item()
                        ),
                    }
                )
                for block_name, block_slice in SEMANTIC_SLICES.items():
                    discrepancy_metrics[
                        f"batch/edit_discrepancy_pre_{block_name}_fraction_sum"
                    ] = feature_mask_fraction(
                        discrepancy_mask_bundle.pre_intersection_mask,
                        condition.target_valid,
                        block_slice,
                    )
                    discrepancy_metrics[
                        f"batch/edit_discrepancy_post_{block_name}_fraction_sum"
                    ] = feature_mask_fraction(
                        discrepancy_mask_bundle.mask,
                        condition.target_valid,
                        block_slice,
                    )
            aux = {
                "time/data_seconds": data_seconds,
                "time/gradient_sync_seconds": gradient_sync_seconds,
                "time/step_seconds": step_seconds,
                "throughput/samples_per_second": (world_size * batch_size) / max(step_seconds, 1e-9),
                "grad/total_preclip": total_grad_norm,
                "grad/base_preclip": base_grad_norm,
                "grad/context_preclip": context_grad_norm,
                "grad/ease_preclip": ease_grad_norm,
                "grad/clip_active": float(total_grad_norm > grad_clip),
                "batch/mask_fraction": float(hard_mask.float().mean().item()),
                "batch/source_present": float(source_present),
                "batch/edit_task_present": float(edit_task_present),
                "batch/context_active": float(context_active),
                "batch/ease_active": float(ease_active),
                "batch/ease_present_fraction": float(
                    condition.ease_present.float().mean().item()
                ),
                "batch/source_present_fraction": float(
                    condition.source_present.any(dim=1).float().mean().item()
                ),
                "batch/text_present_fraction": sum(bool(text) for text in batch["texts"])
                / float(batch_size),
                "batch/stream_hml": float(stream == TrainStream.HML_MIXED),
                "batch/stream_edit": float(stream == TrainStream.MOTION_EDIT),
                "batch/timestep_mean": float(timesteps.float().mean().item()),
                "batch/timestep_below_0p1_fraction": float(
                    (timesteps < 0.1).float().mean().item()
                ),
                "batch/timestep_below_edit_low_max_fraction": float(
                    (timesteps < edit_low_t_max).float().mean().item()
                ),
                "batch/edit_low_t_selected_fraction": float(
                    edit_low_t_selected.float().mean().item()
                ),
                "batch/edit_step_count": float(unified_edit_step),
                "batch/edit_timestep_mean_sum": (
                    float(timesteps.float().mean().item())
                    if unified_edit_step
                    else 0.0
                ),
                "batch/edit_timestep_below_0p1_fraction_sum": (
                    float((timesteps < 0.1).float().mean().item())
                    if unified_edit_step
                    else 0.0
                ),
                "batch/edit_low_t_selected_fraction_sum": (
                    float(edit_low_t_selected.float().mean().item())
                    if unified_edit_step
                    else 0.0
                ),
                "batch/edit_equal_length_identity_fraction_sum": (
                    0.0
                    if identity_state is None
                    else float(identity_state["exact_pair"].float().mean().item())
                ),
                "batch/edit_same_source_negative_fraction_sum": (
                    same_source_negative_fraction if unified_edit_step else 0.0
                ),
                "batch/edit_same_source_eligible_fraction_sum": (
                    float(same_source_eligible_mask.float().mean().item())
                    if unified_edit_step
                    else 0.0
                ),
                "batch/edit_discrepancy_sample_fraction_sum": (
                    float(discrepancy_sample_mask.float().mean().item())
                    if unified_edit_step and edit_discrepancy_x0_scale > 0.0
                    else 0.0
                ),
                "batch/edit_temporal_sample_fraction_sum": (
                    float(temporal_sample_mask.float().mean().item())
                    if unified_edit_step and edit_temporal_scale > 0.0
                    else 0.0
                ),
                "batch/edit_temporal_active_fraction_sum": (
                    float(temporal_active_mask.float().mean().item())
                    if unified_edit_step and edit_temporal_scale > 0.0
                    else 0.0
                ),
                "batch/root_position_only_per_sample": root_position_only
                / float(batch_size),
                "batch/root_position_rotation_per_sample": root_position_rotation
                / float(batch_size),
                "schedule/p_t2m": units.t2m / 5_000_000.0,
                "schedule/p_control": units.control / 5_000_000.0,
                "schedule/p_edit": units.edit / 5_000_000.0,
                "schedule/fk_warmup_factor": loss_bundle.fk_warmup_factor,
                "loss/total_with_edit_aux_sum": float(total_loss.detach().item()),
                "loss/edit_aux_count": float(edit_loss_bundle is not None),
                "loss/edit_aux_total_sum": float(
                    complete_edit_aux
                ),
                "loss/edit_identity_base_raw_sum": identity_base_raw,
                "loss/edit_identity_base_weighted_sum": identity_base_weighted,
                "loss/edit_source_anchor_raw_sum": source_anchor_raw,
                "loss/edit_source_anchor_weighted_sum": source_anchor_weighted,
                "edit/source_anchor_correct_distance_sum": (
                    0.0
                    if source_anchor_bundle is None
                    else float(source_anchor_bundle.correct_distance.detach().item())
                ),
                "edit/source_anchor_baseline_distance_sum": (
                    0.0
                    if source_anchor_bundle is None
                    else float(
                        source_anchor_bundle.source_baseline_distance.detach().item()
                    )
                ),
                "edit/source_anchor_active_fraction_sum": (
                    0.0
                    if source_anchor_bundle is None
                    else float(source_anchor_bundle.active_fraction.detach().item())
                ),
                "grad/edit_correct_output_rms_sum": (
                    correct_output_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_secondary_output_rms_sum": (
                    secondary_output_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_text_token_rms_sum": (
                    text_token_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_text_pooled_rms_sum": (
                    text_pooled_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_source_root_token_rms_sum": (
                    source_root_token_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_source_body_token_rms_sum": (
                    source_body_token_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_objective_output_rms_sum": (
                    objective_output_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_discrepancy_unit_output_rms_sum": (
                    discrepancy_unit_output_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_discrepancy_weighted_output_rms_sum": (
                    discrepancy_weighted_output_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_discrepancy_weighted_to_objective_ratio_sum": (
                    discrepancy_weighted_output_grad_rms
                    / max(objective_output_grad_rms, 1e-30)
                    if unified_edit_step
                    else 0.0
                ),
                "grad/edit_temporal_unit_output_rms_sum": (
                    temporal_unit_output_grad_rms if unified_edit_step else 0.0
                ),
                "grad/edit_temporal_weighted_output_rms_sum": (
                    temporal_weighted_output_grad_rms
                    if unified_edit_step
                    else 0.0
                ),
                "grad/edit_temporal_weighted_to_objective_ratio_sum": (
                    temporal_weighted_output_grad_rms
                    / max(objective_output_grad_rms, 1e-30)
                    if unified_edit_step
                    else 0.0
                ),
                "loss/edit_discrepancy_x0_raw_sum": discrepancy_raw,
                "loss/edit_discrepancy_x0_weighted_sum": discrepancy_weighted,
                "edit/discrepancy_active_fraction_sum": (
                    0.0
                    if discrepancy_loss_bundle is None
                    else float(discrepancy_loss_bundle.active_fraction.detach().item())
                ),
                "loss/edit_temporal_raw_sum": temporal_raw,
                "loss/edit_temporal_vector_raw_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(temporal_loss_bundle.vector_raw.detach().item())
                ),
                "loss/edit_temporal_speed_raw_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(temporal_loss_bundle.speed_raw.detach().item())
                ),
                "loss/edit_temporal_weighted_sum": temporal_weighted,
                "edit/temporal_active_fraction_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(temporal_loss_bundle.active_fraction.detach().item())
                ),
                "edit/temporal_equal_length_fraction_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(
                        temporal_loss_bundle.equal_length_fraction.detach().item()
                    )
                ),
                "edit/temporal_importance_mean_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(temporal_loss_bundle.importance_mean.detach().item())
                ),
                "edit/temporal_high_importance_fraction_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(
                        temporal_loss_bundle.high_importance_fraction.detach().item()
                    )
                ),
                "edit/temporal_source_target_velocity_delta_mps_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(
                        temporal_loss_bundle.source_target_velocity_delta_mps.detach().item()
                    )
                ),
                "edit/temporal_source_target_speed_delta_mps_sum": (
                    0.0
                    if temporal_loss_bundle is None
                    else float(
                        temporal_loss_bundle.source_target_speed_delta_mps.detach().item()
                    )
                ),
                "loss/edit_target_x0_raw_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.target_x0_raw.detach().item()
                ),
                "loss/edit_hard_x0_raw_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.hard_x0_raw.detach().item()
                ),
                "loss/edit_instruction_rank_raw_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_raw.detach().item()
                ),
                "loss/edit_instruction_rank_ce_raw_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_ce_raw.detach().item()
                ),
                "loss/edit_target_x0_weighted_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.target_x0_weighted.detach().item()
                ),
                "loss/edit_hard_x0_weighted_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.hard_x0_weighted.detach().item()
                ),
                "loss/edit_instruction_rank_weighted_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_weighted.detach().item()
                ),
                "edit/correct_distance_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.correct_distance.detach().item()
                ),
                "edit/shuffled_distance_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.shuffled_distance.detach().item()
                ),
                "edit/instruction_gap_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_gap.detach().item()
                ),
                "edit/instruction_rank_active_fraction_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_active_fraction.detach().item()
                ),
                "edit/instruction_rank_active_all_fraction_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_active_all_fraction.detach().item()
                ),
                "edit/instruction_rank_eligible_fraction_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_eligible_fraction.detach().item()
                ),
                "edit/instruction_rank_slope_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_slope.detach().item()
                ),
                "edit/instruction_rank_margin_sum": float(
                    0.0
                    if edit_loss_bundle is None
                    else edit_loss_bundle.instruction_rank_margin.detach().item()
                ),
                **discrepancy_metrics,
                **update_metrics,
            }
            for pattern in EditConditionPattern:
                aux[f"batch/edit_pattern_{pattern.name.lower()}_fraction"] = sum(
                    plan.edit_pattern == pattern for plan in batch["plans"]
                ) / float(batch_size)
            for root_kind in ("root_sparse", "root_dense"):
                has_kind = torch.tensor(
                    [
                        any(
                            token.startswith(root_kind)
                            for token in mode.removeprefix("mixed:").split("+")
                        )
                        for mode in control_modes
                    ],
                    device=hard_mask.device,
                    dtype=torch.bool,
                )
                is_mixed = torch.tensor(
                    [mode.startswith("mixed:") for mode in control_modes],
                    device=hard_mask.device,
                    dtype=torch.bool,
                )
                for mixture_name, mixture_selector in (
                    ("pure", ~is_mixed),
                    ("mixed", is_mixed),
                ):
                    selected = has_kind & mixture_selector
                    aux[
                        f"batch/{root_kind}_{mixture_name}_position_only_per_sample"
                    ] = float((selected & ~has_effective_heading).sum().item()) / float(
                        batch_size
                    )
                    aux[
                        f"batch/{root_kind}_{mixture_name}_position_rotation_per_sample"
                    ] = float((selected & has_effective_heading).sum().item()) / float(
                        batch_size
                    )
            for key, value in aux.items():
                aux_sums[key] = aux_sums.get(key, 0.0) + float(value)
            aux_steps += 1
            global_step = next_global_step

            should_log = global_step == 1 or global_step % log_every == 0 or global_step == actual_stop
            if should_log:
                metrics = loss_window.reduce_and_reset()
                keys = sorted(aux_sums)
                aux_tensor = torch.tensor(
                    [aux_sums[key] for key in keys], device=device, dtype=torch.float64
                )
                count_tensor = torch.tensor([aux_steps], device=device, dtype=torch.float64)
                if is_distributed():
                    dist.all_reduce(aux_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                denominator = max(float(count_tensor.item()), 1.0)
                metrics.update(
                    {key: float(value / denominator) for key, value in zip(keys, aux_tensor.tolist())}
                )
                edit_count = metrics.get("loss/edit_aux_count", 0.0)
                if edit_count > 0.0:
                    for source_key, target_key in (
                        ("loss/edit_aux_total_sum", "loss/edit_aux_total_per_edit"),
                        ("loss/edit_target_x0_raw_sum", "loss/edit_target_x0_raw_per_edit"),
                        ("loss/edit_hard_x0_raw_sum", "loss/edit_hard_x0_raw_per_edit"),
                        (
                            "loss/edit_instruction_rank_raw_sum",
                            "loss/edit_instruction_rank_raw_per_edit",
                        ),
                        (
                            "loss/edit_instruction_rank_ce_raw_sum",
                            "loss/edit_instruction_rank_ce_raw_per_edit",
                        ),
                        (
                            "loss/edit_target_x0_weighted_sum",
                            "loss/edit_target_x0_weighted_per_edit",
                        ),
                        (
                            "loss/edit_hard_x0_weighted_sum",
                            "loss/edit_hard_x0_weighted_per_edit",
                        ),
                        (
                            "loss/edit_instruction_rank_weighted_sum",
                            "loss/edit_instruction_rank_weighted_per_edit",
                        ),
                        (
                            "loss/edit_identity_base_raw_sum",
                            "loss/edit_identity_base_raw_per_edit",
                        ),
                        (
                            "loss/edit_identity_base_weighted_sum",
                            "loss/edit_identity_base_weighted_per_edit",
                        ),
                        (
                            "loss/edit_source_anchor_raw_sum",
                            "loss/edit_source_anchor_raw_per_edit",
                        ),
                        (
                            "loss/edit_source_anchor_weighted_sum",
                            "loss/edit_source_anchor_weighted_per_edit",
                        ),
                        ("edit/correct_distance_sum", "edit/correct_distance"),
                        ("edit/shuffled_distance_sum", "edit/shuffled_distance"),
                        ("edit/instruction_gap_sum", "edit/instruction_gap"),
                        (
                            "edit/instruction_rank_active_fraction_sum",
                            "edit/instruction_rank_active_given_eligible_fraction",
                        ),
                        (
                            "edit/instruction_rank_active_all_fraction_sum",
                            "edit/instruction_rank_active_all_fraction",
                        ),
                        (
                            "edit/instruction_rank_eligible_fraction_sum",
                            "edit/instruction_rank_eligible_fraction",
                        ),
                        (
                            "edit/instruction_rank_slope_sum",
                            "edit/instruction_rank_slope",
                        ),
                        (
                            "edit/instruction_rank_margin_sum",
                            "edit/instruction_rank_margin",
                        ),
                        (
                            "edit/source_anchor_correct_distance_sum",
                            "edit/source_anchor_correct_distance",
                        ),
                        (
                            "edit/source_anchor_baseline_distance_sum",
                            "edit/source_anchor_baseline_distance",
                        ),
                        (
                            "edit/source_anchor_active_fraction_sum",
                            "edit/source_anchor_active_fraction",
                        ),
                        (
                            "grad/edit_correct_output_rms_sum",
                            "grad/edit_correct_output_rms",
                        ),
                        (
                            "grad/edit_secondary_output_rms_sum",
                            "grad/edit_secondary_output_rms",
                        ),
                        (
                            "grad/edit_text_token_rms_sum",
                            "grad/edit_text_token_rms",
                        ),
                        (
                            "grad/edit_text_pooled_rms_sum",
                            "grad/edit_text_pooled_rms",
                        ),
                        (
                            "grad/edit_source_root_token_rms_sum",
                            "grad/edit_source_root_token_rms",
                        ),
                        (
                            "grad/edit_source_body_token_rms_sum",
                            "grad/edit_source_body_token_rms",
                        ),
                        (
                            "grad/edit_objective_output_rms_sum",
                            "grad/edit_objective_output_rms",
                        ),
                        (
                            "grad/edit_discrepancy_unit_output_rms_sum",
                            "grad/edit_discrepancy_unit_output_rms",
                        ),
                        (
                            "grad/edit_discrepancy_weighted_output_rms_sum",
                            "grad/edit_discrepancy_weighted_output_rms",
                        ),
                        (
                            "grad/edit_discrepancy_weighted_to_objective_ratio_sum",
                            "grad/edit_discrepancy_weighted_to_objective_ratio",
                        ),
                        (
                            "loss/edit_discrepancy_x0_raw_sum",
                            "loss/edit_discrepancy_x0_raw_per_edit",
                        ),
                        (
                            "loss/edit_discrepancy_x0_weighted_sum",
                            "loss/edit_discrepancy_x0_weighted_per_edit",
                        ),
                        (
                            "edit/discrepancy_active_fraction_sum",
                            "edit/discrepancy_active_fraction",
                        ),
                        (
                            "grad/edit_temporal_unit_output_rms_sum",
                            "grad/edit_temporal_unit_output_rms",
                        ),
                        (
                            "grad/edit_temporal_weighted_output_rms_sum",
                            "grad/edit_temporal_weighted_output_rms",
                        ),
                        (
                            "grad/edit_temporal_weighted_to_objective_ratio_sum",
                            "grad/edit_temporal_weighted_to_objective_ratio",
                        ),
                        (
                            "loss/edit_temporal_raw_sum",
                            "loss/edit_temporal_raw_per_edit",
                        ),
                        (
                            "loss/edit_temporal_vector_raw_sum",
                            "loss/edit_temporal_vector_raw_per_edit",
                        ),
                        (
                            "loss/edit_temporal_speed_raw_sum",
                            "loss/edit_temporal_speed_raw_per_edit",
                        ),
                        (
                            "loss/edit_temporal_weighted_sum",
                            "loss/edit_temporal_weighted_per_edit",
                        ),
                        (
                            "edit/temporal_active_fraction_sum",
                            "edit/temporal_active_fraction",
                        ),
                        (
                            "edit/temporal_equal_length_fraction_sum",
                            "edit/temporal_equal_length_fraction",
                        ),
                        (
                            "edit/temporal_importance_mean_sum",
                            "edit/temporal_importance_mean",
                        ),
                        (
                            "edit/temporal_high_importance_fraction_sum",
                            "edit/temporal_high_importance_fraction",
                        ),
                        (
                            "edit/temporal_source_target_velocity_delta_mps_sum",
                            "edit/temporal_source_target_velocity_delta_mps",
                        ),
                        (
                            "edit/temporal_source_target_speed_delta_mps_sum",
                            "edit/temporal_source_target_speed_delta_mps",
                        ),
                        (
                            "batch/edit_timestep_mean_sum",
                            "batch/edit_timestep_mean",
                        ),
                        (
                            "batch/edit_timestep_below_0p1_fraction_sum",
                            "batch/edit_timestep_below_0p1_fraction",
                        ),
                        (
                            "batch/edit_low_t_selected_fraction_sum",
                            "batch/edit_low_t_selected_fraction_conditional",
                        ),
                        (
                            "batch/edit_equal_length_identity_fraction_sum",
                            "batch/edit_equal_length_identity_fraction",
                        ),
                        (
                            "batch/edit_same_source_negative_fraction_sum",
                            "batch/edit_same_source_negative_fraction",
                        ),
                        (
                            "batch/edit_same_source_eligible_fraction_sum",
                            "batch/edit_same_source_eligible_fraction",
                        ),
                        (
                            "batch/edit_discrepancy_sample_fraction_sum",
                            "batch/edit_discrepancy_sample_fraction",
                        ),
                        (
                            "batch/edit_temporal_sample_fraction_sum",
                            "batch/edit_temporal_sample_fraction",
                        ),
                        (
                            "batch/edit_temporal_active_fraction_sum",
                            "batch/edit_temporal_active_fraction",
                        ),
                    ):
                        metrics[target_key] = metrics[source_key] / edit_count
                    for source_key in tuple(metrics):
                        if (
                            source_key.startswith("batch/edit_discrepancy_")
                            and source_key.endswith("_sum")
                        ):
                            metrics[source_key.removesuffix("_sum")] = (
                                metrics[source_key] / edit_count
                            )
                    edit_base = metrics.get(
                        "loss/stream/MOTION_EDIT/backward_total", 0.0
                    )
                    edit_aux = metrics["loss/edit_aux_total_per_edit"]
                    edit_total = edit_base + edit_aux
                    if edit_total > 0.0:
                        metrics["loss/edit_base_percent"] = (
                            100.0 * edit_base / edit_total
                        )
                        metrics["loss/edit_target_x0_percent"] = (
                            100.0
                            * metrics["loss/edit_target_x0_weighted_per_edit"]
                            / edit_total
                        )
                        metrics["loss/edit_hard_x0_percent"] = (
                            100.0
                            * metrics["loss/edit_hard_x0_weighted_per_edit"]
                            / edit_total
                        )
                        metrics["loss/edit_instruction_rank_percent"] = (
                            100.0
                            * metrics[
                                "loss/edit_instruction_rank_weighted_per_edit"
                            ]
                            / edit_total
                        )
                        metrics["loss/edit_identity_base_percent"] = (
                            100.0
                            * metrics["loss/edit_identity_base_weighted_per_edit"]
                            / edit_total
                        )
                        metrics["loss/edit_source_anchor_percent"] = (
                            100.0
                            * metrics["loss/edit_source_anchor_weighted_per_edit"]
                            / edit_total
                        )
                        metrics["loss/edit_discrepancy_x0_percent"] = (
                            100.0
                            * metrics["loss/edit_discrepancy_x0_weighted_per_edit"]
                            / edit_total
                        )
                        metrics["loss/edit_temporal_percent"] = (
                            100.0
                            * metrics["loss/edit_temporal_weighted_per_edit"]
                            / edit_total
                        )
                metrics["train/next_global_step"] = float(global_step)
                metrics["train/context_update_count"] = float(context_update_count)
                metrics["train/ease_update_count"] = float(ease_update_count)
                metrics["train/ema_update_count"] = float(ema_update_count)
                metrics["train/hml_t2m_expected"] = batcher.hml_t2m_integrity.expected
                metrics["train/hml_t2m_realized"] = float(
                    batcher.hml_t2m_integrity.realized
                )
                metrics["train/hml_t2m_integrity_bound"] = batcher.hml_t2m_integrity.bound
                metrics["train/hml_t2m_integrity_passed"] = float(
                    batcher.hml_t2m_integrity.passed
                )
                max_memory = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(device) / (1024**3)
                        if device.type == "cuda"
                        else 0.0
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if is_distributed():
                    dist.all_reduce(max_memory, op=dist.ReduceOp.MAX)
                metrics["memory/max_allocated_gib"] = float(max_memory.item())
                if rank == 0:
                    record = {
                        "step": global_step,
                        "stage": str(cfg_get(config, "stage.name")),
                        "plan_trace": plan_trace,
                        "control_modes_last_batch": control_modes,
                        "metrics": metrics,
                    }
                    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    groups = {
                        key: value
                        for key, value in metrics.items()
                        if key.startswith("loss/overall/group_") and key.endswith("percent_total")
                    }
                    print(
                        f"[train] step={global_step} "
                        f"loss={metrics.get('loss/overall/backward_total', float('nan')):.6f} "
                        f"step_s={metrics.get('time/step_seconds', 0.0):.3f} "
                        f"samples_s={metrics.get('throughput/samples_per_second', 0.0):.2f} "
                        f"grad={metrics.get('grad/total_preclip', 0.0):.3f} "
                        f"loss_pct={json.dumps(groups, sort_keys=True)}",
                        flush=True,
                    )
                aux_sums.clear()
                aux_steps = 0

            save_latest = latest_every > 0 and global_step % latest_every == 0
            save_archive = archive_every > 0 and global_step % archive_every == 0
            save_stage_end = global_step == actual_stop and (
                not args.smoke_steps or bool(args.save_smoke)
            )
            if args.research_no_update:
                save_latest = save_archive = save_stage_end = False
            if save_latest or save_archive or save_stage_end:
                if is_distributed():
                    dist.barrier()
                if rank == 0:
                    model_dir = run_dir / "model"
                    if save_archive or (
                        save_stage_end
                        and schedule_version
                        in {
                            STAGE_C_SAFE_MIX_SCHEDULE_VERSION,
                            STAGE_D_EDIT_CALIBRATION_SCHEDULE_VERSION,
                        }
                    ):
                        destination = model_dir / f"step_{global_step:08d}.pt"
                    elif args.smoke_steps and save_stage_end:
                        destination = model_dir / f"smoke_step_{global_step:08d}.pt"
                    else:
                        destination = model_dir / "latest.pt"
                    save_checkpoint(
                        destination,
                        model=raw_model,
                        optimizer=optimizer,
                        ema=ema,
                        config=config,
                        config_path=config_path,
                        next_global_step=global_step,
                        batcher=batcher,
                        context_update_count=context_update_count,
                        ease_update_count=ease_update_count,
                        optimizer_manifest=optimizer_manifest,
                        asset_identity=asset_identity,
                        ease_stats_identity=ease_stats_identity,
                        normalizer=normalizer,
                        run_name=args.name,
                        run_uuid=run_uuid,
                        ema_update_count=ema_update_count,
                        code_identity=code_identity,
                        runtime_identity=runtime_identity,
                    )
                    if save_archive or (save_stage_end and not args.smoke_steps):
                        _copy_checkpoint_link(destination, model_dir / "latest.pt")
                    print(f"[checkpoint] {destination}", flush=True)
                if is_distributed():
                    dist.barrier()

        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "stage_complete",
                        "run": args.name,
                        "next_global_step": global_step,
                        "context_update_count": context_update_count,
                        "ease_update_count": ease_update_count,
                        "hml_t2m_integrity": batcher.state_dict()["hml_t2m_integrity"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
