from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.raw_motion.evidence_hash import canonical_sha256, state_dict_sha256
from models.raw_motion.kimodo_context_flow_dit import HY273KimodoContextFlow
from train_hy273_context_only import (
    _filtered_state,
    _no_source_probe_sha,
    assert_context_optimizer_step,
    context_optimizer_groups,
    freeze_base_for_context_only,
    resolve_parent_model_config,
    update_context_ema,
)
from train_hy273_multitask import load_config


def _model(tmp_path: Path) -> HY273KimodoContextFlow:
    full = tmp_path / "full"
    local = tmp_path / "local"
    full.mkdir()
    local.mkdir()
    np.save(full / "Mean.npy", np.zeros(273, dtype=np.float32))
    np.save(full / "Std.npy", np.ones(273, dtype=np.float32))
    np.save(local / "Mean.npy", np.zeros(4, dtype=np.float32))
    np.save(local / "Std.npy", np.ones(4, dtype=np.float32))
    return HY273KimodoContextFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        mlp_ratio=1.0,
        dropout=0.0,
        max_text_tokens=4,
        text_encoder="null",
        motion_stats_dir=full,
        local_root_stats_dir=local,
        max_frames=8,
    )


def _optimizer_config() -> dict:
    return {
        "optimizer": {
            "context_weight_lr": 5e-5,
            "context_bias_lr": 5e-5,
            "context_weight_decay": 0.01,
            "context_bias_decay": 0.0,
        }
    }


def test_context_only_partition_and_optimizer_exclude_base(tmp_path: Path) -> None:
    model = _model(tmp_path)
    manifest = freeze_base_for_context_only(model)
    groups, optimizer_manifest = context_optimizer_groups(model, _optimizer_config())

    assert manifest["context_tensor_count"] == 9
    assert manifest["context_parameter_count"] == sum(
        parameter.numel()
        for parameter in (
            *model.context_weight_parameters(),
            *model.context_bias_parameters(),
        )
    )
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if not name.startswith("source_context."))
    optimizer_ids = {
        id(parameter) for group in groups for parameter in group["params"]
    }
    assert optimizer_ids == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert set(optimizer_manifest["groups"]) == {
        "G1_context_weight",
        "G2_context_bias",
    }


def test_context_updates_preserve_base_ema_and_no_source_output(tmp_path: Path) -> None:
    model = _model(tmp_path).eval()
    freeze_base_for_context_only(model)
    ema = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    base_sha = state_dict_sha256(
        _filtered_state(model.state_dict(), context=False)
    )
    probe_sha = _no_source_probe_sha(model, torch.device("cpu"))

    with torch.no_grad():
        for parameter in (
            *model.context_weight_parameters(),
            *model.context_bias_parameters(),
        ):
            parameter.add_(0.125)
    update_context_ema(ema, model, decay=0.5)

    assert state_dict_sha256(
        _filtered_state(model.state_dict(), context=False)
    ) == base_sha
    assert state_dict_sha256(_filtered_state(ema, context=False)) == base_sha
    assert _no_source_probe_sha(model, torch.device("cpu")) == probe_sha


def test_fresh_context_optimizer_step_count(tmp_path: Path) -> None:
    model = _model(tmp_path)
    freeze_base_for_context_only(model)
    groups, _ = context_optimizer_groups(model, _optimizer_config())
    optimizer = torch.optim.AdamW(groups)
    assert_context_optimizer_step(optimizer, model, 0)

    loss = sum(
        parameter.square().sum()
        for parameter in (
            *model.context_weight_parameters(),
            *model.context_bias_parameters(),
        )
    )
    loss.backward()
    optimizer.step()
    assert_context_optimizer_step(optimizer, model, 1)


def test_archived_parent_config_migration_only_adds_schedule_metadata() -> None:
    config, _ = load_config("configs/hy273_multitask_r12_stage_b2_joint_adapt.yaml")
    archived = {key: value for key, value in config.items()}
    archived["stage"] = dict(config["stage"])
    archived["stage"].pop("schedule_version")
    checkpoint = {
        "config": archived,
        "config_sha256": canonical_sha256(archived),
    }

    resolved, migrations = resolve_parent_model_config(checkpoint)

    assert resolved["stage"]["schedule_version"] == (
        "hy273_multitask_weighted_deficit_v1"
    )
    assert migrations == [
        {
            "field": "stage.schedule_version",
            "from": "absent_in_archived_checkpoint",
            "to": "hy273_multitask_weighted_deficit_v1",
            "scope": "metadata_only",
        }
    ]
