from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import torch

from sample_hy273_multitask import make_reaction_condition
from tools.eval_hy273_reaction import (
    _action_category,
    _bootstrap_mean,
    _build_split_donor_map,
    _case_noise_seed,
    _clear_source,
    _load_final_protocol_lock,
    _matched_advantage,
    _unrelated_source,
    _validate_final_protocol_runtime,
)


def _condition():
    source = torch.zeros(2, 8, 273)
    source[0, :, 0] = 1.0
    source[1, :, 0] = 2.0
    return make_reaction_condition(
        source,
        target_lengths=torch.tensor([8, 8]),
        source_person_index=torch.tensor([0, 1]),
    )


def test_reaction_ablation_conditions_change_only_intended_fields() -> None:
    condition = _condition()
    empty = _clear_source(condition)
    assert not bool(empty.source_present.any())
    assert torch.count_nonzero(empty.source_motion) == 0
    assert torch.equal(empty.task_id, condition.task_id)

    donor_source = torch.zeros(2, 6, 273)
    donor_source[0, :, 0] = 3.0
    donor_source[1, :, 0] = 4.0
    donor = make_reaction_condition(
        donor_source,
        target_lengths=torch.tensor([8, 8]),
        source_lengths=torch.tensor([6, 6]),
        target_frames=8,
        source_person_index=torch.tensor([0, 1]),
    )
    unrelated = _unrelated_source(condition, donor)
    torch.testing.assert_close(unrelated.source_motion[0], donor.source_motion[0])
    torch.testing.assert_close(unrelated.source_motion[1], donor.source_motion[1])
    assert torch.equal(unrelated.source_role_id, condition.source_role_id)
    assert unrelated.source_frames == 6
    assert unrelated.target_frames == 8
    assert unrelated.source_native_lengths.tolist() == [[6], [6]]


def test_bootstrap_and_matched_advantage_direction() -> None:
    summary = _bootstrap_mean(
        np.asarray([1.0, 2.0, 3.0]), seed=1, resamples=100, confidence=0.95
    )
    assert summary["mean"] == 2.0
    correct = [
        {
            "uid": "a",
            "reactor_fk_mpjpe_cm": 1.0,
            "reactor_contact_f1": 0.9,
            "reactor_prediction_fk_jerk_mps3": 10.0,
        },
        {
            "uid": "b",
            "reactor_fk_mpjpe_cm": 2.0,
            "reactor_contact_f1": 0.8,
            "reactor_prediction_fk_jerk_mps3": 11.0,
        },
    ]
    ablated = [
        {
            "uid": "a",
            "reactor_fk_mpjpe_cm": 3.0,
            "reactor_contact_f1": 0.5,
            "reactor_prediction_fk_jerk_mps3": 5.0,
        },
        {
            "uid": "b",
            "reactor_fk_mpjpe_cm": 4.0,
            "reactor_contact_f1": 0.4,
            "reactor_prediction_fk_jerk_mps3": 6.0,
        },
    ]
    result = _matched_advantage(
        correct, ablated, seed=1, resamples=20, confidence=0.95
    )
    assert result["reactor_fk_mpjpe_cm"]["mean"] == 2.0
    assert result["reactor_contact_f1"]["mean"] == 0.4
    assert "reactor_prediction_fk_jerk_mps3" not in result


class _TinyReactionDataset:
    def __init__(self) -> None:
        self.max_frames = 300
        self.rows = [
            {"clip_id": "G001T001A000R000", "frames": 100, "texts": ["a"]},
            {"clip_id": "G002T001A000R001", "frames": 110, "texts": ["b"]},
            {"clip_id": "G003T001A001R000", "frames": 120, "texts": ["c"]},
            {"clip_id": "G004T001A001R001", "frames": 130, "texts": ["d"]},
            {"clip_id": "G005T001A002R000", "frames": 140, "texts": ["e"]},
            {"clip_id": "G006T001A002R001", "frames": 150, "texts": ["f"]},
        ]
        self.actor_order = {str(row["clip_id"]): 0 for row in self.rows}

    def __len__(self) -> int:
        return len(self.rows)

    def uid(self, index: int) -> str:
        return str(self.rows[int(index)]["clip_id"])


def test_reaction_donor_map_is_split_wide_deterministic_and_action_disjoint() -> None:
    dataset = _TinyReactionDataset()
    first, protocol = _build_split_donor_map(dataset)  # type: ignore[arg-type]
    second, _ = _build_split_donor_map(dataset)  # type: ignore[arg-type]
    assert first == second
    assert set(first) == set(range(len(dataset)))
    assert len(set(first.values())) == len(dataset)
    for index, donor_index in first.items():
        assert index != donor_index
        assert _action_category(dataset.uid(index)) != _action_category(
            dataset.uid(donor_index)
        )
    assert protocol["one_to_one"] is True


def test_reaction_case_noise_seed_is_uid_stable() -> None:
    first = _case_noise_seed(7, "G001T001A000R000", 1)
    assert first == _case_noise_seed(7, "G001T001A000R000", 1)
    assert first != _case_noise_seed(7, "G001T001A000R000", 2)
    assert first != _case_noise_seed(7, "G001T001A001R000", 1)


def test_reaction_final_protocol_lock_requires_exact_runtime_contract(tmp_path) -> None:
    checkpoint = (tmp_path / "step_00200000.pt").resolve()
    checkpoint.touch()
    lock_path = tmp_path / "protocol_lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "format": "hy273_reaction_eval_cfg_lock_v1",
                "checkpoint": str(checkpoint),
                "checkpoint_next_global_step": 200000,
                "weight_source": "ema",
                "num_steps": 32,
                "source_cfg_scale": 2.0,
                "text_cfg_scale": 2.0,
                "caption_policy": "uid_balanced",
                "seed": 20260801,
                "selection_policy": "preregistered_fixed_before_val_and_test",
                "splits": ["val", "test"],
            }
        )
    )
    lock = _load_final_protocol_lock(
        lock_path,
        checkpoint_path=checkpoint,
        checkpoint_step=200000,
        weight_source="ema",
        num_steps=32,
        source_cfg_scale=2.0,
        text_cfg_scale=2.0,
        caption_policy="uid_balanced",
        seed=20260801,
    )
    assert lock["path"] == str(lock_path.resolve())

    try:
        _load_final_protocol_lock(
            lock_path,
            checkpoint_path=checkpoint,
            checkpoint_step=200000,
            weight_source="ema",
            num_steps=32,
            source_cfg_scale=2.0,
            text_cfg_scale=3.0,
            caption_policy="uid_balanced",
            seed=20260801,
        )
    except RuntimeError as error:
        assert "text_cfg_scale" in str(error)
    else:
        raise AssertionError("Mismatched final Reaction CFG was accepted")

    noncanonical_path = tmp_path / "self_consistent_but_noncanonical_lock.json"
    payload = json.loads(lock_path.read_text())
    payload["num_steps"] = 16
    noncanonical_path.write_text(json.dumps(payload))
    try:
        _load_final_protocol_lock(
            noncanonical_path,
            checkpoint_path=checkpoint,
            checkpoint_step=200000,
            weight_source="ema",
            num_steps=16,
            source_cfg_scale=2.0,
            text_cfg_scale=2.0,
            caption_policy="uid_balanced",
            seed=20260801,
        )
    except RuntimeError as error:
        assert "canonical protocol" in str(error)
        assert "num_steps" in str(error)
    else:
        raise AssertionError("Self-consistent noncanonical final protocol was accepted")


def test_reaction_final_protocol_lock_accepts_explicit_250k_checkpoint(tmp_path) -> None:
    checkpoint = (tmp_path / "step_00250000.pt").resolve()
    checkpoint.touch()
    lock_path = tmp_path / "protocol_lock_250k.json"
    lock_path.write_text(
        json.dumps(
            {
                "format": "hy273_reaction_eval_cfg_lock_v1",
                "checkpoint": str(checkpoint),
                "checkpoint_next_global_step": 250000,
                "weight_source": "ema",
                "num_steps": 32,
                "source_cfg_scale": 2.0,
                "text_cfg_scale": 2.0,
                "caption_policy": "uid_balanced",
                "seed": 20260801,
                "selection_policy": "preregistered_fixed_before_val_and_test",
                "splits": ["val", "test"],
            }
        )
    )
    lock = _load_final_protocol_lock(
        lock_path,
        checkpoint_path=checkpoint,
        checkpoint_step=250000,
        weight_source="ema",
        num_steps=32,
        source_cfg_scale=2.0,
        text_cfg_scale=2.0,
        caption_policy="uid_balanced",
        seed=20260801,
    )
    assert lock["checkpoint_next_global_step"] == 250000


def test_reaction_final_protocol_rejects_partial_split() -> None:
    for start_index, num_samples in ((1, None), (0, 1)):
        try:
            _validate_final_protocol_runtime(
                checkpoint_step=200000,
                weight_source="ema",
                num_steps=32,
                source_cfg_scale=2.0,
                text_cfg_scale=2.0,
                caption_policy="uid_balanced",
                seed=20260801,
                start_index=start_index,
                num_samples=num_samples,
            )
        except RuntimeError as error:
            assert "complete requested split" in str(error)
        else:
            raise AssertionError("Partial split was accepted as final Reaction evaluation")
