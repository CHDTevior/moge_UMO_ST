from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import train_hy273_multitask as trainer
from data.hy273_multitask_scheduler import HIGH_LEVEL_SCHEDULE_VERSION
from models.raw_motion.hy273_multitask_losses import RatioLossTerm
from train_hy273_multitask import (
    apply_preserved_ratio_partition,
    contiguous_ratio_rank_groups,
    cuda_rng_states_for_resume,
    validate_exact_kencoder_stage_b_reshard,
)
from tools.gate_hy273_multitask_phase_resume import _assert_exact, _state_sha


def test_phase_resume_state_sha_is_order_stable_and_tensor_sensitive() -> None:
    left = {"b": [torch.tensor([1.0, 2.0])], "a": {"step": 3}}
    reordered = {"a": {"step": 3}, "b": [torch.tensor([1.0, 2.0])]}
    changed = {"a": {"step": 3}, "b": [torch.tensor([1.0, 2.001])]}
    assert _state_sha(left) == _state_sha(reordered)
    assert _state_sha(left) != _state_sha(changed)


def test_phase_resume_state_sha_supports_scalar_optimizer_steps() -> None:
    state = {"step": torch.tensor(1.0), "exp_avg": torch.tensor([0.25])}
    assert _state_sha(state) == _state_sha(state)


def test_phase_resume_exact_compare_checks_nested_tensors() -> None:
    expected = {
        "model": {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        "optimizer": {"state": [torch.tensor(2, dtype=torch.int64)]},
    }
    _assert_exact(expected, expected)
    changed = {
        "model": {"weight": expected["model"]["weight"].clone()},
        "optimizer": {"state": [torch.tensor(3, dtype=torch.int64)]},
    }
    with pytest.raises(RuntimeError, match="tensor mismatch"):
        _assert_exact(changed, expected)


def test_phase_resume_exact_compare_rejects_tensor_metadata_drift() -> None:
    with pytest.raises(RuntimeError, match="tensor metadata mismatch"):
        _assert_exact(torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float32))


def test_cuda_rng_states_expand_only_for_explicit_same_batch_reshard() -> None:
    states = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]
    with pytest.raises(RuntimeError, match="CUDA RNG topology changed"):
        cuda_rng_states_for_resume(
            states,
            device_count=4,
            allow_same_global_batch_reshard=False,
        )
    expanded = cuda_rng_states_for_resume(
        states,
        device_count=4,
        allow_same_global_batch_reshard=True,
    )
    assert [state.item() for state in expanded] == [1, 2, 2, 2]
    assert expanded[2] is not states[-1]


def test_preserved_ratio_partition_reconstructs_parent_rank_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.tensor(3.0, requires_grad=True)
    term = RatioLossTerm(
        name="term",
        group="representation",
        numerator=2.0 * parameter,
        denominator=torch.tensor(3.0),
        weight=0.5,
    )
    bundle = SimpleNamespace(terms={"term": term}, total=term.weighted)

    monkeypatch.setattr(trainer, "is_distributed", lambda: True)

    def fake_all_reduce(
        value: torch.Tensor, *, op: object, group: object
    ) -> None:
        del op, group
        value.fill_(10.0)

    monkeypatch.setattr(trainer.dist, "all_reduce", fake_all_reduce)
    apply_preserved_ratio_partition(
        bundle,
        process_group=object(),
        group_size=2,
    )
    assert term.denominator.item() == pytest.approx(3.0)
    assert term.backward_denominator is not None
    assert term.backward_denominator.item() == pytest.approx(10.0)
    assert term.backward_numerator_scale == pytest.approx(2.0)
    assert bundle.total.item() == pytest.approx(0.6)
    bundle.total.backward()
    assert parameter.grad.item() == pytest.approx(0.2)


def test_preserved_ratio_partition_handles_single_valid_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    term = RatioLossTerm(
        name="term",
        group="control_contact",
        numerator=parameter,
        denominator=torch.tensor(1.0),
        weight=1.0,
    )
    bundle = SimpleNamespace(terms={"term": term}, total=term.weighted)
    monkeypatch.setattr(trainer, "is_distributed", lambda: True)

    def fake_all_reduce(
        value: torch.Tensor, *, op: object, group: object
    ) -> None:
        del op, group
        value.fill_(1.0)

    monkeypatch.setattr(trainer.dist, "all_reduce", fake_all_reduce)
    apply_preserved_ratio_partition(
        bundle,
        process_group=object(),
        group_size=2,
    )
    bundle.total.backward()
    assert bundle.total.item() == pytest.approx(4.0)
    assert parameter.grad.item() == pytest.approx(2.0)


def test_kencoder_stage_b_reshard_gate_is_exact() -> None:
    checkpoint = {
        "next_global_step": 200_000,
        "high_level_schedule_version": HIGH_LEVEL_SCHEDULE_VERSION,
        "batcher": {
            "world_size": 4,
            "batch_size_per_rank": 32,
        },
    }
    validate_exact_kencoder_stage_b_reshard(
        checkpoint,
        current_world_size=8,
        current_batch_size=16,
    )
    assert contiguous_ratio_rank_groups(8, 2) == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    )

    checkpoint["next_global_step"] = 250_000
    with pytest.raises(ValueError, match="exact 200K"):
        validate_exact_kencoder_stage_b_reshard(
            checkpoint,
            current_world_size=8,
            current_batch_size=16,
        )

    checkpoint["next_global_step"] = 200_000
    checkpoint["high_level_schedule_version"] = "wrong_schedule"
    with pytest.raises(ValueError, match="Stage-A scheduler"):
        validate_exact_kencoder_stage_b_reshard(
            checkpoint,
            current_world_size=8,
            current_batch_size=16,
        )

    checkpoint["high_level_schedule_version"] = HIGH_LEVEL_SCHEDULE_VERSION
    checkpoint["batcher"]["world_size"] = 8
    checkpoint["batcher"]["batch_size_per_rank"] = 16
    with pytest.raises(ValueError, match="source topology"):
        validate_exact_kencoder_stage_b_reshard(
            checkpoint,
            current_world_size=8,
            current_batch_size=16,
        )

    checkpoint["batcher"]["world_size"] = 4
    checkpoint["batcher"]["batch_size_per_rank"] = 32
    with pytest.raises(ValueError, match="destination topology"):
        validate_exact_kencoder_stage_b_reshard(
            checkpoint,
            current_world_size=4,
            current_batch_size=32,
        )
