from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from train_hy273_multitask import (
    R11_TRAIN_CONTRACT,
    R12_B1_250K_CODE_IDENTITY_SHA256,
    R12_B1_250K_SAMPLER_SHA256,
    R12_ORIGIN_PARENT_BASE_CONTRACT_SHA256,
    R12_ORIGIN_PARENT_CODE_IDENTITY_SHA256,
    R12_ORIGIN_PARENT_CONFIG_SHA256,
    R12_ORIGIN_PARENT_RUN_NAME,
    R12_ORIGIN_PARENT_RUN_UUID,
    R12_ORIGIN_PARENT_SHA256,
    R12_TRAIN_CONTRACT,
    load_config,
    r12_origin_parent_identity,
    sha256_file,
    validate_frozen_contract,
    validate_production_gate,
    validate_r12_origin_checkpoint,
    validate_r12_origin_parent_identity,
    validate_resume_code_identity,
    validate_unified_edit40_fork_code_identity,
    validate_unified_edit40_objective,
)


def _config():
    return load_config("configs/hy273_multitask_stage_a_t2m.yaml")[0]


def _sampler_cfg_migration_identities():
    previous_files = {
        "train_hy273_multitask.py": "old-trainer",
        "sample_hy273_multitask.py": R12_B1_250K_SAMPLER_SHA256,
        "models/raw_motion/hy273_multitask_losses.py": "same-loss",
    }
    current_files = {
        **previous_files,
        "train_hy273_multitask.py": "new-trainer",
        "sample_hy273_multitask.py": "new-sampler",
    }
    previous = {
        "files": previous_files,
        "identity_sha256": R12_B1_250K_CODE_IDENTITY_SHA256,
    }
    current = {"files": current_files, "identity_sha256": "new-identity"}
    checkpoint = {
        "train_contract": R12_TRAIN_CONTRACT,
        "next_global_step": 250_000,
        "code_identity": previous,
    }
    return checkpoint, current


def test_known_r12_sampler_cfg_resume_migration_is_narrowly_allowed():
    checkpoint, current = _sampler_cfg_migration_identities()
    assert validate_resume_code_identity(
        checkpoint,
        current,
        allow_r12_b1_sampler_cfg_migration=True,
    ) == "r12_b1_250k_sampler_cfg_defaults_2p0"


def test_r12_sampler_cfg_resume_migration_requires_explicit_opt_in():
    checkpoint, current = _sampler_cfg_migration_identities()
    with pytest.raises(RuntimeError, match="code identity mismatch"):
        validate_resume_code_identity(
            checkpoint,
            current,
            allow_r12_b1_sampler_cfg_migration=False,
        )


def test_r12_sampler_cfg_resume_migration_rejects_training_changes():
    checkpoint, current = _sampler_cfg_migration_identities()
    current["files"]["models/raw_motion/hy273_multitask_losses.py"] = "changed-loss"
    with pytest.raises(RuntimeError, match="training-code changes"):
        validate_resume_code_identity(
            checkpoint,
            current,
            allow_r12_b1_sampler_cfg_migration=True,
        )


def _unified_edit40_fork_identities():
    previous_files = {
        "train_hy273_multitask.py": "old-trainer",
        "data/hy273_multitask_scheduler.py": "old-scheduler",
        "models/raw_motion/hy273_multitask_losses.py": "same-loss",
    }
    current_files = {
        **previous_files,
        "train_hy273_multitask.py": "new-trainer",
        "data/hy273_multitask_scheduler.py": "new-scheduler",
    }
    checkpoint = {
        "train_contract": R12_TRAIN_CONTRACT,
        "next_global_step": 450_000,
        "high_level_schedule_version": (
            "hy273_multitask_unified_fixed_30_40_30_joint_edit_v1"
        ),
        "code_identity": {"files": previous_files},
    }
    current = {"files": current_files, "identity_sha256": "new-identity"}
    return checkpoint, current


def test_unified_edit40_fork_allows_only_trainer_and_scheduler_changes():
    checkpoint, current = _unified_edit40_fork_identities()
    assert validate_unified_edit40_fork_code_identity(checkpoint, current) == (
        "r12_unified_edit40_450k_compatible_code_v1"
    )


def test_unified_edit40_fork_rejects_wrong_step_or_unrelated_change():
    checkpoint, current = _unified_edit40_fork_identities()
    checkpoint["next_global_step"] = 449_999
    with pytest.raises(RuntimeError, match="450K checkpoint"):
        validate_unified_edit40_fork_code_identity(checkpoint, current)

    checkpoint, current = _unified_edit40_fork_identities()
    current["files"]["models/raw_motion/hy273_multitask_losses.py"] = "changed-loss"
    with pytest.raises(RuntimeError, match="unrelated training-code changes"):
        validate_unified_edit40_fork_code_identity(checkpoint, current)


def test_unified_edit40_fork_keeps_edit_objective_exactly_fixed():
    parent = load_config(
        "configs/hy273_multitask_r12_stage_c_unified_edit_v2.yaml"
    )[0]
    current = load_config(
        "configs/hy273_multitask_r12_stage_c_unified_edit_v2_edit40.yaml"
    )[0]
    validate_unified_edit40_objective(parent, current)

    changed = deepcopy(current)
    changed["edit_objective"]["target_x0_scale"] *= 2.0
    with pytest.raises(RuntimeError, match="changed the Edit objective"):
        validate_unified_edit40_objective(parent, changed)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("flow", "timestep_mean", -0.7),
        ("flow", "timestep_std", 1.0),
        ("model", "dropout", 0.1),
        ("control", "mixed_prob", 0.5),
        ("control", "max_sparse_keyframes", 8),
        ("training", "gradient_clip", 0.5),
        ("training", "ema_decay", 0.999),
        ("training", "ema_every", 1),
        ("training", "archive_every", 10_000),
    ],
)
def test_every_training_identity_knob_is_frozen(section, key, value):
    config = deepcopy(_config())
    config[section][key] = value
    with pytest.raises(ValueError, match="Frozen R11 field changed"):
        validate_frozen_contract(config)


def test_unknown_field_fails_closed():
    config = deepcopy(_config())
    config["training"]["new_silent_knob"] = True
    with pytest.raises(ValueError, match="config fields changed"):
        validate_frozen_contract(config)


def test_all_stage_configs_match_the_frozen_contract():
    for path in (
        "configs/hy273_multitask_stage_a_t2m.yaml",
        "configs/hy273_multitask_stage_b1_control_bootstrap.yaml",
        "configs/hy273_multitask_stage_b2_joint_adapt.yaml",
        "configs/hy273_multitask_stage_c_consolidate.yaml",
    ):
        validate_frozen_contract(load_config(path)[0])


def test_all_r12_rootmask_stage_configs_match_the_frozen_contract():
    for path in (
        "configs/hy273_multitask_r12_stage_b1_control_bootstrap.yaml",
        "configs/hy273_multitask_r12_stage_b2_joint_adapt.yaml",
        "configs/hy273_multitask_r12_stage_c_consolidate.yaml",
        "configs/hy273_multitask_r12_stage_c_safe_mix_probe.yaml",
        "configs/hy273_multitask_r12_stage_c_edit20_research.yaml",
    ):
        config = load_config(path)[0]
        validate_frozen_contract(config)
        assert config["control"]["root_heading_probability"] == 0.5


def test_r12_root_heading_probability_is_frozen():
    config = load_config(
        "configs/hy273_multitask_r12_stage_b1_control_bootstrap.yaml"
    )[0]
    config["control"]["root_heading_probability"] = 0.75
    with pytest.raises(ValueError, match="Frozen R12 root-mask field changed"):
        validate_frozen_contract(config)


def test_r12_origin_parent_identity_is_content_frozen(tmp_path: Path):
    identity = r12_origin_parent_identity(tmp_path / "parent.pt")
    assert validate_r12_origin_parent_identity(identity) == identity
    changed = deepcopy(identity)
    changed["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="origin parent identity mismatch"):
        validate_r12_origin_parent_identity(changed)


def test_r12_origin_checkpoint_requires_the_exact_r11_identity():
    checkpoint = {
        "train_contract": R11_TRAIN_CONTRACT,
        "run_name": R12_ORIGIN_PARENT_RUN_NAME,
        "run_uuid": R12_ORIGIN_PARENT_RUN_UUID,
        "next_global_step": 200_000,
        "context_update_count": 0,
        "ema_update_count": 20_000,
        "base_contract_sha256": R12_ORIGIN_PARENT_BASE_CONTRACT_SHA256,
        "config_sha256": R12_ORIGIN_PARENT_CONFIG_SHA256,
        "code_identity": {
            "identity_sha256": R12_ORIGIN_PARENT_CODE_IDENTITY_SHA256
        },
    }
    validate_r12_origin_checkpoint(
        checkpoint, checkpoint_sha256=R12_ORIGIN_PARENT_SHA256
    )
    checkpoint["run_uuid"] = "different"
    with pytest.raises(RuntimeError, match="run_uuid mismatch"):
        validate_r12_origin_checkpoint(
            checkpoint, checkpoint_sha256=R12_ORIGIN_PARENT_SHA256
        )


def test_fake_ready_production_bundle_cannot_bypass_typed_payload_gate(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "fake_ready.json"
    artifact.write_text(
        json.dumps(
            {
                "format": "hy273_multitask_nonregression_baseline_v1",
                "status": "ready",
            }
        ),
        encoding="utf-8",
    )
    config = deepcopy(_config())
    config["production"]["nonregression_artifact"] = str(artifact)
    config["production"]["nonregression_artifact_sha256"] = sha256_file(artifact)
    with pytest.raises(RuntimeError, match="stage1 is missing path/SHA"):
        validate_production_gate(config)
