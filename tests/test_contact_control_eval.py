from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.raw_motion.hy273_kimodo_contact_benchmark import (
    V5_CONTACT_SUBTYPES,
    compile_kimodo_contact_constraint,
    controlled_contact_metrics,
)
from models.raw_motion.hy273_kimodo_benchmark import KIMODO_CONTROL_SUBTYPES
from models.raw_motion.hy273_normalizer import HY273Normalizer
from tools.eval_hy273_kimodo_v5_contact import (
    ContactCase,
    MULTITASK_CHECKPOINT_FORMAT,
    MULTITASK_CHECKPOINT_KIND,
    SCIENTIFIC_BENCHMARK_PROFILE,
    TEXT_REGIMES,
    V5_ALL_SUBTYPES,
    _checkpoint_kind,
    _constraint_payload_sha256,
    _initial_control_noise,
    _physical_exact_clamp,
    _sample_control_case,
    _sampling_identity,
    _validate_production_profile,
    build_plan,
)
from train_hy273_multitask import R11_TRAIN_CONTRACT, R12_TRAIN_CONTRACT


def _motion(frames: int = 40) -> torch.Tensor:
    motion = torch.zeros(frames, 273)
    motion[:, 3] = 1.0
    pattern = torch.arange(frames)[:, None] % torch.tensor([2, 3, 4, 5])[None]
    motion[:, 269:273] = (pattern == 0).float()
    return motion


def test_v5_contact_compiler_covers_contact_only_and_compositions() -> None:
    motion = _motion()
    for subtype in V5_CONTACT_SUBTYPES:
        constraint = compile_kimodo_contact_constraint(
            motion, subtype, seed=17, max_sparse_keyframes=8
        )
        assert constraint.contact_metric_mask.any()
        assert constraint.motion_mask[:, 269:273].any()
        assert torch.equal(
            constraint.observed_motion[constraint.motion_mask],
            motion[constraint.motion_mask],
        )
        if subtype == "contact_only_sparse":
            assert not constraint.motion_mask[:, :269].any()
        else:
            assert constraint.motion_mask[:, :269].any()


def test_controlled_contact_metrics_separate_raw_adherence_from_exact_clamp() -> None:
    target = _motion(8)[:, 269:273]
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[[0, 2, 5, 7]] = True
    raw = torch.full_like(target, 0.6)
    exact = raw.clone()
    exact[mask] = target[mask]
    raw_metrics = controlled_contact_metrics(raw, target, mask)
    exact_metrics = controlled_contact_metrics(exact, target, mask)
    assert raw_metrics["controlled_contact_entries"] == int(mask.sum())
    assert raw_metrics["controlled_contact_exact_equality"] < 1.0
    assert raw_metrics["controlled_contact_brier"] > 0.0
    assert exact_metrics["controlled_contact_exact_equality"] == 1.0
    assert exact_metrics["controlled_contact_bce"] < 1e-5
    assert exact_metrics["controlled_contact_brier"] == 0.0


def test_control_evaluator_clamps_bit_exactly_in_physical_space() -> None:
    normalizer = HY273Normalizer(
        torch.linspace(-0.2, 0.3, 273),
        torch.linspace(0.4, 1.7, 273),
        variance_eps=1.0e-5,
    )
    observed = torch.linspace(-1.3, 2.1, 4 * 273).reshape(4, 273)
    observed[:, 269:273] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    prediction = normalizer.denormalize(
        normalizer.normalize(observed.unsqueeze(0))
    )[0]
    mask = torch.zeros_like(observed, dtype=torch.bool)
    mask[1, 0:273:7] = True
    mask[2, 269:273] = True
    assert not torch.equal(prediction[mask], observed[mask])

    exact = _physical_exact_clamp(prediction, observed, mask)
    assert torch.equal(exact[mask], observed[mask])
    assert torch.equal(exact[~mask], prediction[~mask])


def test_v5_plan_retains_legacy_and_adds_contact_subtypes() -> None:
    assert V5_ALL_SUBTYPES == (*KIMODO_CONTROL_SUBTYPES, *V5_CONTACT_SUBTYPES)
    plan = build_plan(len(V5_ALL_SUBTYPES), seed=3407)
    assert len(plan) == len(V5_ALL_SUBTYPES) * len(TEXT_REGIMES)
    assert {case.subtype for case in plan} == set(V5_ALL_SUBTYPES)


def test_research_control_profile_allows_registered_pilot_subset() -> None:
    values = dict(SCIENTIFIC_BENCHMARK_PROFILE)
    values.update(weight_source="model", cases_per_subtype=32)
    _validate_production_profile(SimpleNamespace(profile="research", **values))
    with pytest.raises(RuntimeError, match="production benchmark profile mismatch"):
        _validate_production_profile(SimpleNamespace(profile="production", **values))


def test_control_noise_uses_deterministic_independent_cpu_streams() -> None:
    case = ContactCase(3, "contact_only_sparse", "withtext", 1234)
    first_cont, first_contact, first_evidence = _initial_control_noise(case, 40)
    second_cont, second_contact, second_evidence = _initial_control_noise(case, 40)
    assert first_cont.dtype == torch.float32 and first_cont.device.type == "cpu"
    assert first_contact.dtype == torch.float32 and first_contact.device.type == "cpu"
    assert torch.equal(first_cont, second_cont)
    assert torch.equal(first_contact, second_contact)
    assert first_evidence == second_evidence
    assert first_evidence["initial_continuous_noise_sha256"] != first_evidence[
        "initial_contact_noise_sha256"
    ]


class _ZeroControlModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, model_in: torch.Tensor, **_kwargs) -> torch.Tensor:
        return torch.zeros_like(model_in[..., :273]) + self.anchor * 0.0


@pytest.mark.parametrize("unified", [False, True])
def test_multitask_control_routes_noise_from_contact_protocol(unified: bool) -> None:
    case = ContactCase(0, "contact_only_sparse", "notext", 17)
    target = _motion(8)
    constraint = compile_kimodo_contact_constraint(
        target, case.subtype, seed=case.sample_seed, max_sparse_keyframes=4
    )
    continuous, contacts, evidence = _initial_control_noise(
        case, len(target), unified=unified
    )
    evidence = {**evidence, "text": ""}
    prepared = {
        "target_cpu": target,
        "constraint_cpu": constraint,
        "is_contact_case": True,
        "c_dir_cpu": torch.tensor([1.0, 0.0]),
        "initial_continuous_noise": continuous,
        "initial_contact_noise": contacts,
        "evidence": evidence,
    }
    args = SimpleNamespace(
        num_steps=1,
        cfg_scale=2.0,
        control_cfg_scale=2.0,
        max_sparse_keyframes=4,
    )
    normalizer = HY273Normalizer(
        torch.zeros(273), torch.ones(273), normalize_contacts=unified
    )
    output = _sample_control_case(
        kind=MULTITASK_CHECKPOINT_KIND,
        model=_ZeroControlModel(),
        normalizer=normalizer,
        self_conditioning=False,
        prepared=prepared,
        args=args,
        device=torch.device("cpu"),
    )
    identity = _sampling_identity(args, unified_273_flow=unified)
    assert evidence["initial_noise_protocol"] == identity["initial_noise"]
    assert output.protocol["initial_noise_source"] == (
        "provided_unified_273d" if unified else "legacy_split_state"
    )


def test_constraint_payload_hash_binds_values_masks_metrics_and_direction() -> None:
    constraint = compile_kimodo_contact_constraint(
        _motion(), "root_sparse_contact", seed=17, max_sparse_keyframes=8
    )
    baseline = _constraint_payload_sha256(constraint, torch.tensor([1.0, 0.0]))
    repeated = _constraint_payload_sha256(constraint, torch.tensor([1.0, 0.0]))
    rotated = _constraint_payload_sha256(constraint, torch.tensor([0.0, 1.0]))
    assert baseline == repeated
    assert baseline != rotated


def test_control_evaluator_accepts_only_registered_multitask_contracts() -> None:
    for contract in (R11_TRAIN_CONTRACT, R12_TRAIN_CONTRACT):
        assert (
            _checkpoint_kind(
                {"format": MULTITASK_CHECKPOINT_FORMAT, "train_contract": contract}
            )
            == MULTITASK_CHECKPOINT_KIND
        )
    try:
        _checkpoint_kind(
            {
                "format": MULTITASK_CHECKPOINT_FORMAT,
                "train_contract": "unknown-contract",
            }
        )
    except RuntimeError as error:
        assert "train contract" in str(error)
    else:
        raise AssertionError("Unknown multitask contract was accepted")
