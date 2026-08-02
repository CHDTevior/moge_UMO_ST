from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import torch

from data.hy273_reaction_dataset import (
    HY273ReactionDataset,
    ReactionConditionPattern,
    ReactionSamplePlan,
    apply_shared_reaction_gauge,
    reaction_pattern_from_draw,
)
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    SourceRole,
    TaskId,
)
from models.raw_motion.hy273_normalizer import apply_yaw_rotation
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_SLICE,
    reconstruct_global_joints_from_features,
)
from models.raw_motion.kimodo_context_flow_dit import HY273SourceContext
from sample_hy273_multitask import (
    make_instruction_only_edit_diagnostic_condition,
    make_reaction_condition,
    prepare_reaction_source,
    restore_reaction_world,
    validate_sampling_mode_checkpoint,
)


def _motion(frames: int, *, root_y: float, root_x: float, root_z: float) -> torch.Tensor:
    motion = torch.zeros(frames, DIM_HY273)
    motion[:, ROOT_SLICE] = torch.tensor([root_x, root_y, root_z])
    motion[:, HEADING_SLICE] = torch.tensor([1.0, 0.0])
    joints = torch.linspace(-0.4, 0.4, 22 * 3).reshape(22, 3)
    motion[:, JOINT_POS_SLICE] = joints.reshape(-1)
    identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    motion[:, GLOBAL_ROT_SLICE] = identity_6d.repeat(22)
    motion[:, CONTACT_SLICE] = 0.0
    return motion


def test_reaction_dropout_pattern_is_exact_90_5_5() -> None:
    counts = {pattern: 0 for pattern in ReactionConditionPattern}
    for draw in range(100):
        counts[reaction_pattern_from_draw(draw)] += 1
    assert counts == {
        ReactionConditionPattern.SOURCE_AND_TEXT: 90,
        ReactionConditionPattern.SOURCE_ONLY: 5,
        ReactionConditionPattern.UNCONDITIONAL: 5,
    }


def test_reaction_dataset_uses_official_actor_reactor_order(tmp_path: Path) -> None:
    frames = 16
    p1 = _motion(frames, root_y=1.0, root_x=2.0, root_z=-1.0)
    p2 = _motion(frames, root_y=2.0, root_x=-3.0, root_z=4.0)
    np.save(tmp_path / "p1.npy", p1.numpy().astype(np.float32))
    np.save(tmp_path / "p2.npy", p2.numpy().astype(np.float32))
    row = {
        "dataset": "interx",
        "split": "train",
        "clip_id": "sample",
        "frames": frames,
        "fps": 30,
        "dim": DIM_HY273,
        "person1": "p1.npy",
        "person2": "p2.npy",
        "has_text": True,
        "texts": ["one person reacts to the other"],
    }
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    with (tmp_path / "interaction_order.pkl").open("wb") as handle:
        pickle.dump({"sample": 1}, handle)

    dataset = HY273ReactionDataset(tmp_path, max_frames=32)
    plan = ReactionSamplePlan(
        global_step=100_000,
        global_sample_ordinal=0,
        row_index=0,
        uid="sample",
        caption_index=0,
        crop_start=0,
        yaw_u64=1 << 63,
        condition_pattern=ReactionConditionPattern.SOURCE_AND_TEXT,
    )
    sample = dataset.materialize(plan)
    assert sample["actor_person_index"] == 1
    assert float(sample["source_motion"][0, 0, ROOT_SLICE.start + 1]) == 2.0
    assert float(sample["target_motion"][0, ROOT_SLICE.start + 1]) == 1.0
    assert int(sample["task_id"]) == int(TaskId.REACTION)
    assert int(sample["capability_id"]) == int(CapabilityId.TEXT_REACTION)
    assert int(sample["source_role_id"][0]) == int(
        SourceRole.OTHER_ACTOR_SECOND_PERSON
    )
    torch.testing.assert_close(
        sample["source_motion"][0, 0, [ROOT_SLICE.start, ROOT_SLICE.start + 2]],
        torch.zeros(2),
        atol=1e-6,
        rtol=0.0,
    )


def test_reaction_condition_encodes_observed_caption_person() -> None:
    source = _motion(8, root_y=1.0, root_x=0.0, root_z=0.0).repeat(2, 1, 1)
    condition = make_reaction_condition(
        source,
        target_lengths=torch.tensor([8, 8]),
        source_person_index=torch.tensor([0, 1]),
    )
    assert condition.source_role_id[:, 0].tolist() == [
        int(SourceRole.OTHER_ACTOR_FIRST_PERSON),
        int(SourceRole.OTHER_ACTOR_SECOND_PERSON),
    ]


def test_reaction_sampler_rejects_wrong_or_untrained_checkpoint() -> None:
    config = {
        "data": {"paired_task": "reaction"},
        "model": {
            "source_fusion_mode": "token_block",
            "text_token_sequence": "sentence_plus_context",
        },
    }
    checkpoint = {
        "format": "hy273_unified_actor_checkpoint_v1",
        "next_global_step": 100_000,
    }
    try:
        validate_sampling_mode_checkpoint("reaction", checkpoint, config)
    except RuntimeError as error:
        assert "Stage-B" in str(error)
    else:
        raise AssertionError("Stage-A Reaction checkpoint was accepted")
    validate_sampling_mode_checkpoint(
        "reaction",
        checkpoint,
        config,
        allow_stage_a_reaction_diagnostic=True,
    )
    wrong = {**config, "data": {"paired_task": "interaction"}}
    try:
        validate_sampling_mode_checkpoint(
            "reaction", {**checkpoint, "next_global_step": 200_000}, wrong
        )
    except RuntimeError as error:
        assert "paired_task=reaction" in str(error)
    else:
        raise AssertionError("Archived Interaction checkpoint was accepted as Reaction")


def test_shared_reaction_gauge_preserves_inter_actor_geometry() -> None:
    source = _motion(8, root_y=1.0, root_x=2.0, root_z=-1.0)
    target = _motion(8, root_y=1.1, root_x=-0.5, root_z=3.0)
    target[:, JOINT_POS_SLICE] += 0.2
    before = torch.stack(
        [
            reconstruct_global_joints_from_features(source),
            reconstruct_global_joints_from_features(target),
        ]
    )
    source_g, target_g, _ = apply_shared_reaction_gauge(source, target, phi=1.2)
    after = torch.stack(
        [
            reconstruct_global_joints_from_features(source_g),
            reconstruct_global_joints_from_features(target_g),
        ]
    )
    torch.testing.assert_close(
        torch.cdist(after[0], after[1]),
        torch.cdist(before[0], before[1]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_reaction_sampling_gauge_round_trip() -> None:
    source = _motion(8, root_y=1.0, root_x=2.0, root_z=-1.0).unsqueeze(0)
    target = _motion(8, root_y=1.1, root_x=-0.5, root_z=3.0).unsqueeze(0)
    gauge = prepare_reaction_source(source, heading_rad=1.1)
    target_g = target.clone()
    target_g[..., ROOT_SLICE.start] -= gauge.root_anchor_xz[:, 0, None]
    target_g[..., ROOT_SLICE.start + 2] -= gauge.root_anchor_xz[:, 1, None]
    target_g = apply_yaw_rotation(target_g, gauge.yaw_delta)
    restored = restore_reaction_world(target_g, gauge)
    torch.testing.assert_close(restored, target, rtol=1e-5, atol=1e-5)


def test_global_task_conditioning_keeps_source_free_branch_without_source_tokens(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "Mean.npy", np.zeros(DIM_HY273, dtype=np.float32))
    np.save(tmp_path / "Std.npy", np.ones(DIM_HY273, dtype=np.float32))
    condition = make_instruction_only_edit_diagnostic_condition(
        target_lengths=torch.tensor([4]),
    )
    context = HY273SourceContext(
        hidden_dim=8,
        motion_stats_dir=tmp_path,
        max_frames=8,
        normalize_contacts=True,
        num_tasks=4,
        global_task_conditioning=True,
    )
    output = context(condition, target_frames=4, dtype=torch.float32)
    assert not bool(output.context_present.any())
    assert torch.count_nonzero(output.root) == 0
    assert torch.count_nonzero(output.body) == 0
