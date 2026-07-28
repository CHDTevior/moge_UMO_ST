from __future__ import annotations

from types import SimpleNamespace

import torch

import train_hy273_multitask as trainer
from data.hy273_multitask_scheduler import SamplePlan
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    TrainStream,
    make_absent_condition,
)


def test_hard_control_compiler_consumes_sample_plan_seed(monkeypatch) -> None:
    expected_seed = 123_456_789
    seen_heading_seeds = []

    def fake_compiler(motion, *, generator, root_heading_generator, config, **kwargs):
        assert generator.initial_seed() == expected_seed
        seen_heading_seeds.append(root_heading_generator.initial_seed())
        assert config.root_heading_probability == 1.0
        mask = torch.zeros_like(motion, dtype=torch.bool)
        mask[:, 0, 0] = True
        return SimpleNamespace(
            observed_motion=motion * mask,
            motion_mask=mask,
            mode_ids=["root_sparse"],
        )

    monkeypatch.setattr(
        trainer, "build_kimodo_control_curriculum_batch", fake_compiler
    )
    condition = make_absent_condition(
        batch_size=1,
        target_frames=8,
        capability=CapabilityId.KIMODO_CONTROL,
    )
    plan = SamplePlan(
        global_step=200_000,
        global_sample_ordinal=17,
        train_stream_id=TrainStream.HML_MIXED,
        capability_id=CapabilityId.KIMODO_CONTROL,
        row_index=3,
        uid="humanml3d:test",
        caption_index=0,
        yaw_u64=11,
        control_u64=expected_seed,
        text_drop=False,
        edit_pattern=None,
    )
    config = {
        "control": {
            "curriculum_start_step": 200_000,
            "curriculum_end_step": 400_000,
            "mixed_prob": 0.25,
            "max_sparse_keyframes": 20,
            "dense_min_fraction": 1.0,
            "endpoint_preset": "kimodo_ee",
            "endpoint_subset_mode": "random_nonempty",
            "include_root_ref_for_endpoints": True,
            "include_endpoint_rotations": True,
            "include_contact_pattern": True,
        }
    }

    observed, mask, modes = trainer.build_hard_controls(
        target_physical=torch.ones(1, 8, 273),
        condition=condition,
        plans=[plan],
        global_step=200_000,
        config=config,
        manifest_sha256="manifest-sha-must-not-redraw-control",
        run_seed=999,
    )

    assert modes == ["root_sparse"]
    assert torch.equal(observed, mask.float())
    assert mask.sum().item() == 1
    assert seen_heading_seeds == [
        trainer._plan_draw(
            plan,
            "manifest-sha-must-not-redraw-control",
            999,
            "root_heading_presence",
        )
    ]


def test_hard_control_gate_can_force_every_kimodo_pattern(monkeypatch) -> None:
    seen = []

    def fake_compiler(motion, *, mode_schedule, **kwargs):
        seen.extend(mode_schedule)
        mask = torch.zeros_like(motion, dtype=torch.bool)
        mask[:, 0, 0] = True
        return SimpleNamespace(
            observed_motion=motion * mask,
            motion_mask=mask,
            mode_ids=list(mode_schedule),
        )

    monkeypatch.setattr(trainer, "build_kimodo_control_curriculum_batch", fake_compiler)
    modes = ["root_sparse", "root_dense", "endpoints", "fullpose", "contact"]
    condition = make_absent_condition(
        batch_size=len(modes),
        target_frames=8,
        capability=CapabilityId.KIMODO_CONTROL,
    )
    plans = [
        SamplePlan(
            global_step=400_000,
            global_sample_ordinal=index,
            train_stream_id=TrainStream.HML_MIXED,
            capability_id=CapabilityId.KIMODO_CONTROL,
            row_index=index,
            uid=f"humanml3d:{index}",
            caption_index=0,
            yaw_u64=1,
            control_u64=index + 1,
            text_drop=False,
            edit_pattern=None,
        )
        for index in range(len(modes))
    ]
    config = {
        "control": {
            "curriculum_start_step": 200_000,
            "curriculum_end_step": 400_000,
            "mixed_prob": 0.25,
            "max_sparse_keyframes": 20,
            "dense_min_fraction": 1.0,
            "endpoint_preset": "kimodo_ee",
            "endpoint_subset_mode": "random_nonempty",
            "include_root_ref_for_endpoints": True,
            "include_endpoint_rotations": True,
            "include_contact_pattern": True,
        }
    }
    _, mask, actual = trainer.build_hard_controls(
        target_physical=torch.ones(len(modes), 8, 273),
        condition=condition,
        plans=plans,
        global_step=400_000,
        config=config,
        manifest_sha256="unused",
        run_seed=1,
        forced_mode_schedule=modes,
    )
    assert seen == modes
    assert actual == modes
    assert mask[:, 0, 0].all()
