from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from data.hy273_multitask_scheduler import (
    KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
)
from models.raw_motion.hy273_multitask_condition import (
    ABSOLUTE_TEXT_PROFILE,
    RELATIVE_EDIT_TEXT_PROFILE,
    CapabilityId,
    ConditionBatch,
    FramePolicy,
    SourceRole,
    TargetOp,
    TaskId,
    TrainStream,
    make_absent_condition,
)
from models.raw_motion.kimodo_context_flow_dit import HY273KimodoContextFlow
from train_hy273_multitask import (
    _context_optimizer_steps,
    apply_optimizer_phase,
    assert_and_mask_context_gradients,
    initialize_ema,
    initialize_ema_with_new_ease,
    load_parent_model_with_new_ease,
    migrate_optimizer_state_with_new_ease,
    optimizer_groups,
)
from models.raw_motion.hy273_ease import EASE_STATS_FORMAT


def _stats(root: Path) -> tuple[Path, Path]:
    full = root / "full"
    local = root / "local"
    full.mkdir(parents=True)
    local.mkdir(parents=True)
    np.save(full / "Mean.npy", np.zeros(273, dtype=np.float32))
    np.save(full / "Std.npy", np.ones(273, dtype=np.float32))
    np.save(local / "Mean.npy", np.zeros(4, dtype=np.float32))
    np.save(local / "Std.npy", np.ones(4, dtype=np.float32))
    return full, local


def _model(tmp_path: Path) -> HY273KimodoContextFlow:
    full, local = _stats(tmp_path)
    return HY273KimodoContextFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        text_encoder="none",
        max_text_tokens=4,
        motion_stats_dir=full,
        local_root_stats_dir=local,
        max_frames=8,
    )


def _ease_model(tmp_path: Path) -> HY273KimodoContextFlow:
    full, local = _stats(tmp_path)
    ease = tmp_path / "ease"
    ease.mkdir()
    np.save(ease / "Mean.npy", np.zeros(6, dtype=np.float32))
    np.save(ease / "Std.npy", np.ones(6, dtype=np.float32))
    (ease / "metadata.json").write_text(
        '{"feature_dim": 6, "format": "' + EASE_STATS_FORMAT + '"}'
    )
    return HY273KimodoContextFlow(
        hidden_dim=16,
        num_heads=4,
        root_depth_double=1,
        root_depth_single=1,
        body_depth_double=1,
        body_depth_single=1,
        text_encoder="none",
        max_text_tokens=4,
        motion_stats_dir=full,
        local_root_stats_dir=local,
        max_frames=8,
        use_ease=True,
        ease_stats_dir=ease,
    )


def _edit_condition(frames: int = 4) -> ConditionBatch:
    valid = torch.ones(1, frames, dtype=torch.bool)
    source = torch.zeros(1, 1, frames, 273)
    source[..., 3] = 1.0
    condition = ConditionBatch(
        train_stream_id=torch.tensor([int(TrainStream.MOTION_EDIT)]),
        task_id=torch.tensor([int(TaskId.EDIT)]),
        capability_id=torch.tensor([int(CapabilityId.MOTION_EDIT)]),
        text_encoding_profile=(RELATIVE_EDIT_TEXT_PROFILE,),
        target_valid=valid,
        target_op_id=torch.full((1, frames), int(TargetOp.EDIT)),
        source_motion=source,
        source_present=torch.ones(1, 1, dtype=torch.bool),
        source_time_valid=torch.ones(1, 1, frames, dtype=torch.bool),
        source_value_mask=torch.ones(1, 1, frames, 273, dtype=torch.bool),
        source_role_id=torch.tensor([[int(SourceRole.SELF)]]),
        source_native_lengths=torch.tensor([[frames]]),
        requested_target_len=torch.tensor([frames]),
        frame_gauge_dir=torch.tensor([[1.0, 0.0]]),
        frame_policy_id=torch.tensor([int(FramePolicy.INDEPENDENT_SEQUENCE)]),
        ease_physical=torch.zeros(1, 6),
        ease_present=torch.zeros(1, dtype=torch.bool),
    )
    condition.validate(max_target_frames=8)
    return condition


def _source_free_edit_condition(frames: int = 4) -> ConditionBatch:
    condition = make_absent_condition(
        batch_size=1,
        target_frames=frames,
        capability=CapabilityId.T2M,
    )
    condition = replace(
        condition,
        train_stream_id=torch.tensor([int(TrainStream.MOTION_EDIT)]),
        task_id=torch.tensor([int(TaskId.EDIT)]),
        capability_id=torch.tensor([int(CapabilityId.MOTION_EDIT)]),
        text_encoding_profile=(RELATIVE_EDIT_TEXT_PROFILE,),
        target_op_id=torch.full((1, frames), int(TargetOp.EDIT)),
    )
    condition.validate(max_target_frames=8, v1_strict=False)
    return condition


def _backward(
    model: HY273KimodoContextFlow, condition: ConditionBatch
) -> None:
    frames = condition.target_frames
    model_in = torch.randn(1, frames, 546)
    prediction = model(
        model_in,
        t=torch.tensor([0.5]),
        text=[""],
        length_mask=condition.target_valid,
        condition=condition,
    )
    prediction.square().mean().backward()


def test_context_adam_state_only_advances_on_edit(tmp_path: Path) -> None:
    model = _model(tmp_path)
    groups, _ = optimizer_groups(model, 250_000)
    optimizer = torch.optim.AdamW(groups)
    absent = make_absent_condition(
        batch_size=1,
        target_frames=4,
        capability=CapabilityId.T2M,
    )
    assert absent.text_encoding_profile == (ABSOLUTE_TEXT_PROFILE,)

    apply_optimizer_phase(optimizer, 250_000)
    _backward(model, absent)
    assert_and_mask_context_gradients(
        model,
        context_active=False,
        global_step=250_000,
        optimizer=optimizer,
    )
    optimizer.step()
    assert _context_optimizer_steps(optimizer, model) == {}

    optimizer.zero_grad(set_to_none=True)
    _backward(model, _edit_condition())
    assert_and_mask_context_gradients(
        model,
        context_active=True,
        global_step=250_001,
        optimizer=optimizer,
    )
    optimizer.step()
    steps_after_edit = _context_optimizer_steps(optimizer, model)
    assert steps_after_edit
    assert set(steps_after_edit.values()) == {1}

    optimizer.zero_grad(set_to_none=True)
    _backward(model, absent)
    assert_and_mask_context_gradients(
        model,
        context_active=False,
        global_step=250_002,
        optimizer=optimizer,
    )
    optimizer.step()
    assert _context_optimizer_steps(optimizer, model) == steps_after_edit


def test_r16_stage_b_keeps_edit_context_frozen(tmp_path: Path) -> None:
    model = _model(tmp_path)
    groups, _ = optimizer_groups(
        model,
        300_000,
        R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
    )
    optimizer = torch.optim.AdamW(groups)
    assert {
        group["group_name"]: group["lr"] for group in optimizer.param_groups
    }["G1_context_weight"] == 0.0

    _backward(model, _edit_condition())
    with pytest.raises(
        RuntimeError,
        match="Source/task context appeared in a context-frozen stage",
    ):
        assert_and_mask_context_gradients(
            model,
            context_active=True,
            global_step=300_000,
            optimizer=optimizer,
            schedule_version=R16_STAGE_B_FIXED_CONTROL_SCHEDULE_VERSION,
        )


def test_kencoder_stage_be_updates_edit_context_in_stage_b1(tmp_path: Path) -> None:
    model = _model(tmp_path)
    groups, _ = optimizer_groups(
        model,
        200_000,
        KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    )
    optimizer = torch.optim.AdamW(groups)
    resolved = {
        group["group_name"]: group["lr"] for group in optimizer.param_groups
    }
    assert resolved == {
        "G0_existing": 5e-5,
        "G1_context_weight": 1e-4,
        "G2_context_bias": 1e-4,
    }

    _backward(model, _edit_condition())
    assert_and_mask_context_gradients(
        model,
        context_active=True,
        global_step=200_000,
        optimizer=optimizer,
        schedule_version=KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    )
    optimizer.step()
    assert set(_context_optimizer_steps(optimizer, model).values()) == {1}


def test_ease_fork_preserves_model_ema_and_adam_by_parameter_name(
    tmp_path: Path,
) -> None:
    parent_root = tmp_path / "parent"
    child_root = tmp_path / "child"
    parent = _model(parent_root)
    parent_groups, parent_manifest = optimizer_groups(
        parent,
        200_000,
        KENCODER_STAGE_BE_EDIT_SCHEDULE_VERSION,
    )
    parent_optimizer = torch.optim.AdamW(parent_groups)
    for parameter in parent.parameters():
        parameter.grad = torch.ones_like(parameter)
    parent_optimizer.step()
    parent_ema = initialize_ema(parent)

    child = _ease_model(child_root)
    missing = load_parent_model_with_new_ease(child, parent.state_dict())
    assert missing
    assert all(name.startswith("ease_conditioner.") for name in missing)
    child_groups, child_manifest = optimizer_groups(
        child,
        250_000,
        KENCODER_STAGE_BC_EASE_CONTROL_SCHEDULE_VERSION,
    )
    child_optimizer = torch.optim.AdamW(child_groups)
    migrate_optimizer_state_with_new_ease(
        child_optimizer,
        parent_state=parent_optimizer.state_dict(),
        parent_manifest=parent_manifest,
        current_manifest=child_manifest,
    )
    parent_names = dict(parent.named_parameters())
    child_names = dict(child.named_parameters())
    for name, parent_parameter in parent_names.items():
        assert torch.equal(child_names[name], parent_parameter)
        if parent_parameter.requires_grad:
            assert child_names[name] in child_optimizer.state
    for name, parameter in child_names.items():
        if name.startswith("ease_conditioner."):
            assert parameter not in child_optimizer.state

    child_ema = initialize_ema_with_new_ease(child, parent_ema)
    for name in parent_ema:
        assert torch.equal(child_ema[name], parent_ema[name])
    assert all(
        name in child_ema
        for name in child.state_dict()
        if name.startswith("ease_conditioner.")
    )

    condition = make_absent_condition(batch_size=1, target_frames=4)
    model_in = torch.randn(1, 4, 546)
    parent.eval()
    child.eval()
    with torch.no_grad():
        parent_output = parent(
            model_in,
            t=torch.tensor([0.5]),
            text=[""],
            length_mask=condition.target_valid,
            condition=condition,
        )
        child_output = child(
            model_in,
            t=torch.tensor([0.5]),
            text=[""],
            length_mask=condition.target_valid,
            condition=condition,
        )
    assert torch.equal(parent_output, child_output)


def test_phase_learning_rates_are_overwritten_after_load(tmp_path: Path) -> None:
    model = _model(tmp_path)
    groups, _ = optimizer_groups(model, 399_999)
    source_optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    tracked_parameter = source_optimizer.param_groups[0]["params"][0]
    tracked_parameter.grad = torch.ones_like(tracked_parameter)
    source_optimizer.step()
    saved_step = source_optimizer.state[tracked_parameter]["step"].clone()

    loaded_optimizer = torch.optim.AdamW(groups, betas=(0.8, 0.88), eps=1e-6)
    loaded_optimizer.load_state_dict(source_optimizer.state_dict())
    for group in loaded_optimizer.param_groups:
        group["lr"] = 9.0
        group["weight_decay"] = 9.0
    apply_optimizer_phase(loaded_optimizer, 400_000)
    resolved = {
        group["group_name"]: (group["lr"], group["weight_decay"])
        for group in loaded_optimizer.param_groups
    }
    assert resolved == {
        "G0_existing": (2e-5, 0.01),
        "G1_context_weight": (5e-5, 0.01),
        "G2_context_bias": (5e-5, 0.0),
    }
    assert torch.equal(
        loaded_optimizer.state[tracked_parameter]["step"],
        saved_step,
    )
    assert loaded_optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    assert loaded_optimizer.param_groups[0]["eps"] == 1e-8


def test_g0_lr_override_changes_only_base_group(tmp_path: Path) -> None:
    model = _model(tmp_path)
    groups, _ = optimizer_groups(model, 0, g0_lr_override=2e-5)
    optimizer = torch.optim.AdamW(groups)
    initial = {
        group["group_name"]: (group["lr"], group["weight_decay"])
        for group in optimizer.param_groups
    }
    assert initial == {
        "G0_existing": (2e-5, 0.01),
        "G1_context_weight": (0.0, 0.01),
        "G2_context_bias": (0.0, 0.0),
    }

    for step, expected_context_lr in ((0, 0.0), (250_000, 1e-4), (400_000, 5e-5)):
        for group in optimizer.param_groups:
            group["lr"] = 9.0
            group["weight_decay"] = 9.0
        apply_optimizer_phase(optimizer, step, g0_lr_override=2e-5)
        restored = {
            group["group_name"]: (group["lr"], group["weight_decay"])
            for group in optimizer.param_groups
        }
        assert restored == {
            "G0_existing": (2e-5, 0.01),
            "G1_context_weight": (expected_context_lr, 0.01),
            "G2_context_bias": (expected_context_lr, 0.0),
        }


def test_source_free_edit_advances_task_context_optimizer(tmp_path: Path) -> None:
    model = _model(tmp_path)
    groups, _ = optimizer_groups(model, 500_000)
    optimizer = torch.optim.AdamW(groups)
    condition = _source_free_edit_condition()

    output = model.source_context(condition, target_frames=4, dtype=torch.float32)
    assert output.context_present.tolist() == [True]
    output.root.sum().backward()
    assert torch.count_nonzero(
        model.source_context.task_embed.weight.grad[int(TaskId.EDIT)]
    )
    assert torch.count_nonzero(
        model.source_context.op_embed.weight.grad[int(TargetOp.EDIT)]
    )
    assert_and_mask_context_gradients(
        model,
        context_active=True,
        global_step=500_000,
        optimizer=optimizer,
    )
    optimizer.step()
    steps = _context_optimizer_steps(optimizer, model)
    assert len(steps) == len(
        (*model.context_weight_parameters(), *model.context_bias_parameters())
    )
    assert set(steps.values()) == {1}
