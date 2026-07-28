from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.hy273_multitask_batcher import HY273MultitaskStepBatcher
from data.hy273_multitask_manifest_dataset import (
    HY273MultitaskManifestDataset,
    collate_hy273_multitask,
)
from data.hy273_multitask_scheduler import EditConditionPattern, SamplePlan
from models.raw_motion.hy273_multitask_condition import CapabilityId, TrainStream


def _asset(path: Path, frames: int) -> dict:
    motion = np.zeros((frames, 273), dtype=np.float32)
    motion[:, 3] = 1.0
    np.save(path, motion)
    return {
        "path": str(path),
        "sha256": "not-computed",
        "frames": frames,
        "fps": 30.0,
        "feature_dim": 273,
        "representation_version": "test",
    }


def _manifest(path: Path) -> None:
    rows = []
    hml_lengths = (8, 16, 64, 72)
    for index in range(4):
        target = _asset(path.parent / f"hml_{index}.npy", hml_lengths[index])
        texts = [
            {
                "value": f"motion {index}",
                "encoding_profile": "hytext_absolute_motion_v1",
                "target_k273_asset": target,
            }
        ]
        if index == 0:
            segment = _asset(path.parent / "hml_0_segment.npy", 5)
            texts.append(
                {
                    "value": "short segment zero",
                    "encoding_profile": "hytext_absolute_motion_v1",
                    "target_k273_asset": segment,
                }
            )
        rows.append(
            {
                "schema_version": "hy273_multitask_manifest_v1",
                "uid": f"humanml3d:{index}",
                "dataset": "humanml3d_k273",
                "split": "train",
                "source_motion": None,
                "target_motion": {"k273_asset": target},
                "texts": texts,
            }
        )
    for index in range(4):
        source = _asset(path.parent / f"source_{index}.npy", 15 + index)
        target = _asset(path.parent / f"target_{index}.npy", 16 + index)
        rows.append(
            {
                "schema_version": "hy273_multitask_manifest_v1",
                "uid": f"motionfix:{index}",
                "dataset": "motionfix_k273",
                "split": "train",
                "source_motion": {"k273_asset": source},
                "target_motion": {"k273_asset": target},
                "texts": [
                    {
                        "value": f"edit {index}",
                        "encoding_profile": "hytext_relative_edit_v1",
                    }
                ],
                "pair": {"frame_policy_id": "independent_sequence_frame_v1"},
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_batcher_resume_replays_next_batch(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    first = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=7,
        world_size=1,
        rank=0,
        batch_size_per_rank=2,
        materialize_workers=1,
        sort_window_batches=1,
    )
    batch0, stream0, trace0 = first.next_batch(0)
    state = first.state_dict()
    batch1, stream1, trace1 = first.next_batch(1)

    resumed = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=7,
        world_size=1,
        rank=0,
        batch_size_per_rank=2,
        materialize_workers=1,
        sort_window_batches=1,
    )
    resumed.load_state_dict(state)
    resumed_batch, resumed_stream, resumed_trace = resumed.next_batch(1)
    assert stream0 == TrainStream.HML_MIXED
    assert stream1 == resumed_stream
    assert trace1 == resumed_trace
    assert batch1["uids"] == resumed_batch["uids"]
    assert trace0 != trace1
    assert batch0["condition"].source_present.any().item() is False


def test_research_reshard_preserves_next_global_plan(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    source = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=23,
        world_size=2,
        rank=0,
        batch_size_per_rank=2,
        materialize_workers=1,
        sort_window_batches=1,
    )
    state = source.state_dict()
    _, source_stream, source_trace = source.next_batch(0)

    resharded = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=23,
        world_size=1,
        rank=0,
        batch_size_per_rank=4,
        materialize_workers=1,
        sort_window_batches=1,
    )
    with pytest.raises(ValueError, match="world_size"):
        resharded.load_state_dict(state)
    resharded.load_state_dict(state, allow_same_global_batch_reshard=True)
    batch, stream, trace = resharded.next_batch(0)
    assert stream == source_stream
    assert trace == source_trace
    assert len(batch["uids"]) == 4


def test_expansion_reshard_preserves_parent_ratio_partitions(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    template = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=29,
        world_size=4,
        rank=0,
        batch_size_per_rank=32,
        materialize_workers=1,
        sort_window_batches=1,
    )
    state = template.state_dict()
    source_ranks = []
    for rank in range(4):
        batcher = HY273MultitaskStepBatcher(
            train_manifest=manifest,
            run_seed=29,
            world_size=4,
            rank=rank,
            batch_size_per_rank=32,
            materialize_workers=1,
            sort_window_batches=1,
        )
        batcher.load_state_dict(state)
        source_ranks.append(batcher)
    expanded_ranks = []
    for rank in range(8):
        batcher = HY273MultitaskStepBatcher(
            train_manifest=manifest,
            run_seed=29,
            world_size=8,
            rank=rank,
            batch_size_per_rank=16,
            materialize_workers=1,
            sort_window_batches=1,
        )
        batcher.load_state_dict(
            state,
            allow_same_global_batch_reshard=True,
            preserve_source_rank_ratio_objective=True,
        )
        expanded_ranks.append(batcher)

    for step in range(4):
        source_batches = [batcher.next_batch(step)[0] for batcher in source_ranks]
        expanded_batches = [
            batcher.next_batch(step)[0] for batcher in expanded_ranks
        ]
        for virtual_rank in range(4):
            combined_plans = (
                expanded_batches[2 * virtual_rank]["plans"]
                + expanded_batches[2 * virtual_rank + 1]["plans"]
            )
            assert combined_plans == source_batches[virtual_rank]["plans"]

    saved = expanded_ranks[0].state_dict()
    assert saved["ratio_partition"] == {
        "format": "hy273_rank_ratio_partition_v1",
        "world_size": 4,
        "batch_size": 32,
    }
    resumed = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=29,
        world_size=8,
        rank=0,
        batch_size_per_rank=16,
        materialize_workers=1,
        sort_window_batches=1,
    )
    resumed.load_state_dict(saved)
    assert resumed.ratio_group_size == 2
    assert resumed.ratio_partition_world_size == 4
    assert resumed.ratio_partition_batch_size == 32
    expected_batch = expanded_ranks[0].next_batch(4)[0]
    resumed_batch = resumed.next_batch(4)[0]
    assert resumed_batch["plans"] == expected_batch["plans"]


def test_hml_plan_bucket_uses_selected_caption_segment_length(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    dataset = HY273MultitaskManifestDataset(manifest, TrainStream.HML_MIXED)
    plan = SamplePlan(
        global_step=0,
        global_sample_ordinal=0,
        train_stream_id=TrainStream.HML_MIXED,
        capability_id=CapabilityId.T2M,
        row_index=0,
        uid="humanml3d:0",
        caption_index=1,
        yaw_u64=0,
        control_u64=0,
        text_drop=False,
        edit_pattern=None,
    )
    assert dataset.plan_lengths(plan) == (5, 0)
    assert dataset.plan_bucket_key(plan) == (0, 0)
    sample = dataset.materialize(plan)
    assert sample["target_motion"].shape == (5, 273)


@pytest.mark.parametrize(
    ("pattern", "expected_text"),
    [
        (EditConditionPattern.TEXT_ONLY, "edit 0"),
        (EditConditionPattern.UNCONDITIONAL, ""),
    ],
)
def test_edit_cfg_source_free_patterns_use_valid_absent_sentinel(
    tmp_path: Path,
    pattern: EditConditionPattern,
    expected_text: str,
) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    dataset = HY273MultitaskManifestDataset(manifest, TrainStream.MOTION_EDIT)
    plan = SamplePlan(
        global_step=500_000,
        global_sample_ordinal=0,
        train_stream_id=TrainStream.MOTION_EDIT,
        capability_id=CapabilityId.MOTION_EDIT,
        row_index=0,
        uid="motionfix:0",
        caption_index=None,
        yaw_u64=0,
        control_u64=0,
        text_drop=not pattern.uses_text,
        edit_pattern=pattern,
    )
    sample = dataset.materialize(plan)
    assert sample["text"] == expected_text
    assert sample["source_motion"].shape == (1, 1, 273)
    assert not sample["source_present"].any()
    assert not sample["source_time_valid"].any()
    assert not sample["source_value_mask"].any()
    batch = collate_hy273_multitask([sample])
    batch["condition"].validate(v1_strict=False)


def test_source_identity_pattern_uses_transformed_source_as_no_edit_target(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    dataset = HY273MultitaskManifestDataset(manifest, TrainStream.MOTION_EDIT)
    source_path = Path(
        dataset.rows[0]["source_motion"]["k273_asset"]["path"]
    )
    source = np.load(source_path)
    source[:, 5] = np.linspace(0.0, 0.25, source.shape[0], dtype=np.float32)
    np.save(source_path, source)
    plan = SamplePlan(
        global_step=0,
        global_sample_ordinal=0,
        train_stream_id=TrainStream.MOTION_EDIT,
        capability_id=CapabilityId.MOTION_EDIT,
        row_index=0,
        uid="motionfix:0",
        caption_index=None,
        yaw_u64=0,
        control_u64=0,
        text_drop=True,
        edit_pattern=EditConditionPattern.SOURCE_IDENTITY,
    )

    assert dataset.plan_lengths(plan) == (15, 15)
    sample = dataset.materialize(plan)
    assert sample["text"] == ""
    assert sample["target_motion"].shape == (15, 273)
    assert torch.equal(sample["target_motion"], sample["source_motion"][0])
    assert sample["source_present"].tolist() == [True]


def test_plan_first_bucketing_covers_rows_and_rotates_length_chunks(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    rank0 = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=17,
        world_size=2,
        rank=0,
        batch_size_per_rank=2,
        materialize_workers=1,
        sort_window_batches=1,
    )
    rank1 = HY273MultitaskStepBatcher(
        train_manifest=manifest,
        run_seed=17,
        world_size=2,
        rank=1,
        batch_size_per_rank=2,
        materialize_workers=1,
        sort_window_batches=1,
    )

    batch00, _, trace00 = rank0.next_batch(0)
    batch10, _, trace10 = rank1.next_batch(0)
    assert trace00 == trace10
    assert len(set(batch00["uids"] + batch10["uids"])) == 4
    assert set(batch00["uids"]).isdisjoint(batch10["uids"])
    assert batch00["target_motion"].shape[1] <= batch10["target_motion"].shape[1]

    batch01, _, trace01 = rank0.next_batch(1)
    batch11, _, trace11 = rank1.next_batch(1)
    assert trace01 == trace11
    assert len(set(batch01["uids"] + batch11["uids"])) == 4
    assert set(batch01["uids"]).isdisjoint(batch11["uids"])
    assert batch01["target_motion"].shape[1] >= batch11["target_motion"].shape[1]


def test_dataset_rejects_nonbinary_contact_payload(tmp_path: Path) -> None:
    manifest = tmp_path / "train.jsonl"
    _manifest(manifest)
    dataset = HY273MultitaskManifestDataset(manifest, TrainStream.HML_MIXED)
    row = dataset.rows[0]
    asset_path = Path(row["texts"][0]["target_k273_asset"]["path"])
    motion = np.load(asset_path)
    motion[0, 269] = 0.5
    np.save(asset_path, motion)
    plan = SamplePlan(
        global_step=0,
        global_sample_ordinal=0,
        train_stream_id=TrainStream.HML_MIXED,
        capability_id=CapabilityId.T2M,
        row_index=0,
        uid="humanml3d:0",
        caption_index=0,
        yaw_u64=0,
        control_u64=0,
        text_drop=False,
        edit_pattern=None,
    )
    with pytest.raises(ValueError, match="exact binary 0/1"):
        dataset.materialize(plan)
