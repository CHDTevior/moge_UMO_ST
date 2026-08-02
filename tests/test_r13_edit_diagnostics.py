from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from models.raw_motion.hy273_normalizer import HY273Normalizer, apply_yaw_rotation
from tools.eval_hy273_edit_same_source_fixed_t import (
    aggregate_physical_records,
    aggregate_group_records,
    paired_bootstrap_direct_comparison,
    parse_system_expectation,
    paired_bootstrap_comparisons,
    validate_checkpoint_systems,
)
from tools.eval_hy273_r13_same_source_ab_decision import (
    PRIMARY_EDIT_SUBSET,
    edit_guardrail,
    evaluate_directional_metrics,
    relative_degradation,
)
from tools.diagnose_hy273_r13_edit_fixed_t import (
    aggregate,
    build_directional_donor_map,
    target_errors,
    yaw_pair_normalized_noise,
)
from tools.overfit_hy273_edit_instruction_pairs import (
    CandidateGroup,
    MaterializedGroup,
    collate_groups,
    paired_assignment_metrics,
    select_candidate_groups,
)


def test_directional_donor_map_is_explicit_and_opposite_is_counterfactual() -> None:
    pair_ids = ("up_a", "up_b", "down_a", "down_b")
    directions = {
        "up_a": "increase",
        "up_b": "increase",
        "down_a": "decrease",
        "down_b": "decrease",
    }
    donors = build_directional_donor_map(pair_ids, directions)
    for pair_id in pair_ids:
        row = donors[pair_id]
        assert row["source_donor_id"] != pair_id
        assert directions[row["same_direction_text_donor_id"]] == directions[pair_id]
        assert directions[row["opposite_direction_text_donor_id"]] != directions[pair_id]


def test_yaw_paired_noise_rotates_through_physical_hy273_space() -> None:
    generator = torch.Generator().manual_seed(7)
    mean = torch.randn(273, generator=generator)
    std = torch.rand(273, generator=generator) + 0.2
    normalizer = HY273Normalizer(mean, std, normalize_contacts=True)
    base = torch.randn(2, 5, 273, generator=generator)
    phi = math.pi / 2
    paired = yaw_pair_normalized_noise(base, normalizer, phi)
    expected = normalizer.normalize(
        apply_yaw_rotation(normalizer.denormalize(base), torch.tensor(phi))
    )
    torch.testing.assert_close(paired, expected)
    torch.testing.assert_close(paired[..., 269:273], base[..., 269:273])


def test_fixed_t_aggregate_keeps_yaw_as_an_independent_variable() -> None:
    records = [
        {
            "checkpoint": "model_a.pt",
            "checkpoint_step": 400000,
            "yaw_degrees": yaw,
            "t": 0.1,
            "branch": "source_correct/text_correct",
            "metrics": {"error": yaw + 1.0},
        }
        for yaw in (0.0, 90.0)
    ]
    rows = aggregate(records)
    assert len(rows) == 2
    assert {row["yaw_degrees"] for row in rows} == {0.0, 90.0}


def test_fixed_t_aggregate_keeps_same_step_checkpoints_separate() -> None:
    records = [
        {
            "checkpoint": checkpoint,
            "checkpoint_step": 405000,
            "yaw_degrees": 0.0,
            "t": 0.0,
            "branch": "source_correct/text_correct",
            "metrics": {"error": error},
        }
        for checkpoint, error in (("baseline.pt", 1.0), ("treatment.pt", 2.0))
    ]
    rows = aggregate(records)
    assert len(rows) == 2
    assert {row["checkpoint"] for row in rows} == {"baseline.pt", "treatment.pt"}
    assert {row["metrics_mean"]["error"] for row in rows} == {1.0, 2.0}


def test_fixed_t_selected_region_error_separates_mask_and_complement() -> None:
    target_norm = torch.zeros(2, 273)
    prediction_norm = torch.zeros_like(target_norm)
    prediction_norm[0, 0] = 2.0
    prediction_norm[1, 1] = 3.0
    selected = torch.zeros(2, 269, dtype=torch.bool)
    selected[0, 0] = True
    metrics = target_errors(
        prediction_norm,
        target_norm,
        prediction_norm,
        target_norm,
        length=2,
        discrepancy_mask=selected,
    )
    assert metrics["selected_continuous_normalized_mse"] == 4.0
    assert metrics["unselected_continuous_normalized_mse"] == pytest.approx(9.0 / 537.0)
    assert metrics["selected_continuous_fraction"] == pytest.approx(1.0 / 538.0)


def test_tiny_overfit_candidate_selection_keeps_unique_source_pairs() -> None:
    payload = [
        {
            "target_pair_mse": 9.0,
            "frames": 60,
            "source_sha256": "bad-same-text",
            "source_base_motion_id": "bad",
            "pair_ids": ["bad0", "bad1"],
            "texts": ["same edit", "  SAME   EDIT "],
        },
        {
            "target_pair_mse": 4.0,
            "frames": 60,
            "source_sha256": "source-a",
            "source_base_motion_id": "a",
            "pair_ids": ["a0", "a1"],
            "texts": ["edit a", "edit b"],
        },
        {
            "target_pair_mse": 3.5,
            "frames": 60,
            "source_sha256": "source-a",
            "source_base_motion_id": "a-duplicate",
            "pair_ids": ["a2", "a3"],
            "texts": ["edit e", "edit f"],
        },
        {
            "target_pair_mse": 3.0,
            "frames": 90,
            "source_sha256": "source-b",
            "source_base_motion_id": "b",
            "pair_ids": ["b0", "b1"],
            "texts": ["edit c", "edit d"],
        },
    ]
    selected = select_candidate_groups(
        payload, count=2, max_frames=90, minimum_target_pair_mse=3.0
    )
    assert [group.pair_ids for group in selected] == [("a0", "a1"), ("b0", "b1")]
    assert [group.group_index for group in selected] == [0, 1]


def test_tiny_overfit_batch_has_text_only_target_selecting_input() -> None:
    candidate = CandidateGroup(
        group_index=0,
        source_sha256="source",
        source_base_motion_id="base",
        pair_ids=("a", "b"),
        texts=("first edit", "second edit"),
        frames=4,
        target_pair_mse=1.0,
    )
    source = torch.randn(4, 273)
    source[:, 269:273] = torch.randint(0, 2, (4, 4)).float()
    targets = (torch.randn(4, 273), torch.randn(4, 273))
    group = MaterializedGroup(
        candidate=candidate,
        source=source,
        targets=targets,
        texts=candidate.texts,
        noise=torch.randn(4, 273),
    )
    batch = collate_groups([group], [0], device=torch.device("cpu"))
    torch.testing.assert_close(batch["source"][0], batch["source"][1])
    torch.testing.assert_close(batch["noise"][0], batch["noise"][1])
    assert batch["texts"] == ["first edit", "second edit"]
    assert batch["swapped_texts"] == ["second edit", "first edit"]
    assert batch["condition"].task_id.tolist() == [1, 1]

    candidate_two = CandidateGroup(
        group_index=1,
        source_sha256="source-two",
        source_base_motion_id="base-two",
        pair_ids=("c", "d"),
        texts=("third edit", "fourth edit"),
        frames=4,
        target_pair_mse=1.0,
    )
    source_two = torch.randn(4, 273)
    source_two[:, 269:273] = torch.randint(0, 2, (4, 4)).float()
    group_two = MaterializedGroup(
        candidate=candidate_two,
        source=source_two,
        targets=(torch.randn(4, 273), torch.randn(4, 273)),
        texts=candidate_two.texts,
        noise=torch.randn(4, 273),
    )
    batch_two = collate_groups(
        [group, group_two], [0, 1], device=torch.device("cpu")
    )
    assert batch_two["swapped_texts"] == [
        "second edit",
        "first edit",
        "fourth edit",
        "third edit",
    ]


def test_tiny_overfit_assignment_metric_detects_instruction_selection() -> None:
    target = torch.zeros(2, 3, 273)
    target[1, :, :269] = 2.0
    records, aggregate = paired_assignment_metrics(
        target.clone(),
        target,
        torch.tensor([3, 3]),
        [0, 0],
        ["a", "b"],
    )
    assert len(records) == 1
    continuous = aggregate["spaces"]["continuous_269"]
    assert continuous["correct_mse"] == 0.0
    assert continuous["swapped_mse"] == 4.0
    assert continuous["assignment_correct"] == 1.0
    assert continuous["row_assignment_accuracy"] == 1.0
    assert aggregate["memorized"] is True


def test_tiny_overfit_gate_rejects_assignment_with_large_common_error() -> None:
    target = torch.zeros(2, 3, 273)
    target[1, :, :269] = 2.0
    prediction = torch.full_like(target, 100.0)
    prediction[1, :, :269] = 102.0
    records, aggregate = paired_assignment_metrics(
        prediction,
        target,
        torch.tensor([3, 3]),
        [0, 0],
        ["a", "b"],
    )
    assert records[0]["spaces"]["continuous_269"]["assignment_correct"] == 1.0
    assert records[0]["memorized"] is False
    assert aggregate["memorized"] is False


def test_tiny_overfit_gate_includes_contact_channels() -> None:
    target = torch.zeros(2, 3, 273)
    target[1, :, :269] = 2.0
    target[1, :, 269:273] = 1.0
    prediction = target.clone()
    prediction[0, :, 269:273] = 1.0
    prediction[1, :, 269:273] = 0.0
    records, aggregate = paired_assignment_metrics(
        prediction,
        target,
        torch.tensor([3, 3]),
        [0, 0],
        ["a", "b"],
    )
    assert records[0]["spaces"]["continuous_269"]["correct_mse"] == 0.0
    assert records[0]["spaces"]["contact_4"]["assignment_correct"] == 0.0
    assert aggregate["memorized"] is False


def test_same_source_eval_bootstraps_reconstruction_and_instruction_metrics() -> None:
    spaces = ("full_273", "continuous_269", "contact_4")

    def system(correct: float, margin: float, effect: float, advantage: float) -> dict:
        row = {
            "pair_ids": ["a", "b"],
            "spaces": {
                space: {
                    "correct_instruction_mse": correct,
                    "swapped_instruction_mse": correct + margin,
                    "instruction_margin": margin,
                    "text_effect_rms": effect,
                    "correct_assignment": 1.0,
                    "swapped_assignment": 1.0 - advantage,
                    "assignment_advantage": advantage,
                    "empty_instruction_mse": correct + 0.25,
                    "correct_vs_empty_mse_gap": 0.25,
                    "correct_vs_empty_effect_rms": effect / 2,
                    "source_copy_mse": 3.0,
                    "correct_vs_source_copy_mse_gain": 3.0 - correct,
                }
                for space in spaces
            },
        }
        return {
            "timesteps": {
                "0.0": {
                    "groups": [row],
                    "aggregate": aggregate_group_records([row]),
                }
            }
        }

    comparisons = paired_bootstrap_comparisons(
        {
            "baseline": system(2.0, 0.1, 0.2, 0.0),
            "treatment": system(1.5, 0.4, 0.5, 1.0),
        },
        baseline_label="baseline",
        subsets={"target_disjoint": {("a", "b")}},
        bootstrap_samples=100,
        seed=7,
    )
    metrics = comparisons["treatment"]["target_disjoint"]["0.0"]["full_273"]
    assert metrics["correct_instruction_mse"][
        "mean_delta_candidate_minus_baseline"
    ] == -0.5
    assert metrics["instruction_margin"][
        "mean_delta_candidate_minus_baseline"
    ] == pytest.approx(0.3)
    assert metrics["assignment_advantage"][
        "mean_delta_candidate_minus_baseline"
    ] == 1.0
    assert metrics["text_effect_rms"]["mean_delta_candidate_minus_baseline"] == 0.3
    assert metrics["text_effect_rms"]["better_direction"] == "diagnostic_two_sided"
    assert metrics["correct_vs_source_copy_mse_gain"][
        "mean_delta_candidate_minus_baseline"
    ] == 0.5
    assert metrics["correct_vs_source_copy_mse_gain"]["better_direction"] == "positive"


def test_same_source_eval_aggregates_optional_physical_metrics_by_branch() -> None:
    records = [
        {
            "branch": "correct",
            "metrics": {
                "changed_region_target_error_m": 0.1,
                "unchanged_region_source_error_m": None,
            },
        },
        {
            "branch": "correct",
            "metrics": {
                "changed_region_target_error_m": 0.3,
                "unchanged_region_source_error_m": 0.02,
            },
        },
        {
            "branch": "empty",
            "metrics": {
                "changed_region_target_error_m": 0.4,
                "unchanged_region_source_error_m": 0.01,
            },
        },
    ]

    aggregate = aggregate_physical_records(records)

    assert aggregate["correct"]["cases"] == 2
    assert aggregate["correct"]["changed_region_target_error_m"] == pytest.approx(0.2)
    assert aggregate["correct"]["unchanged_region_source_error_m"] == pytest.approx(
        0.02
    )
    assert aggregate["empty"]["changed_region_target_error_m"] == pytest.approx(0.4)


def test_same_source_eval_directly_bootstraps_hinge_vs_softplus() -> None:
    def system(advantage: float) -> dict:
        row = {
            "pair_ids": ["a", "b"],
            "spaces": {
                space: {
                    "correct_instruction_mse": 1.0,
                    "swapped_instruction_mse": 1.0 + advantage,
                    "instruction_margin": advantage,
                    "text_effect_rms": advantage,
                    "correct_assignment": 1.0,
                    "swapped_assignment": 1.0 - advantage,
                    "assignment_advantage": advantage,
                    "empty_instruction_mse": 1.25,
                    "correct_vs_empty_mse_gap": 0.25,
                    "correct_vs_empty_effect_rms": 0.1,
                    "source_copy_mse": 1.5,
                    "correct_vs_source_copy_mse_gain": 0.5,
                }
                for space in ("full_273", "continuous_269", "contact_4")
            },
        }
        step = {"groups": [row], "aggregate": aggregate_group_records([row])}
        return {"timesteps": {"0.0": step}, "ode": step}

    result = paired_bootstrap_direct_comparison(
        {"hinge": system(0.2), "softplus": system(0.5)},
        left_label="hinge",
        right_label="softplus",
        subsets={"target_disjoint": {("a", "b")}},
        bootstrap_samples=100,
        seed=9,
    )
    subset = result["subsets"]["target_disjoint"]
    fixed = subset["timesteps"]["0.0"]["full_273"]["assignment_advantage"]
    ode = subset["ode"]["full_273"]["assignment_advantage"]
    assert fixed["mean_delta_softplus_minus_hinge"] == pytest.approx(0.3)
    assert fixed["paired_bootstrap_ci95"] == pytest.approx([0.3, 0.3])
    assert ode["mean_delta_softplus_minus_hinge"] == pytest.approx(0.3)


def test_ab_guardrail_uses_directional_relative_degradation() -> None:
    assert relative_degradation(1.04, 1.0, "lower") == pytest.approx(0.04)
    assert relative_degradation(0.96, 1.0, "higher") == pytest.approx(0.04)
    result = evaluate_directional_metrics(
        {"error": 1.0, "accuracy": 0.8},
        {"error": 1.04, "accuracy": 0.77},
        {"error": "lower", "accuracy": "higher"},
        maximum_degradation=0.05,
    )
    assert result["passed"] is True
    assert result["metrics"]["error"]["passed"] is True
    assert result["metrics"]["accuracy"]["passed"] is True


def test_ab_guardrail_requires_parent_relative_ode_fidelity() -> None:
    def endpoint(correct: float, empty_gap: float) -> dict:
        return {
            "correct_instruction_mse": correct,
            "correct_vs_empty_mse_gap": empty_gap,
        }

    fixed = {
        "systems": {
            "parent": {
                "timesteps": {
                    "0.0": {
                        "subset_aggregates": {
                            PRIMARY_EDIT_SUBSET: {"full_273": endpoint(1.0, -0.1)}
                        }
                    }
                },
                "ode": {
                    "subset_aggregates": {
                        PRIMARY_EDIT_SUBSET: {"full_273": endpoint(1.0, -0.1)}
                    }
                },
            },
            "candidate": {
                "timesteps": {
                    "0.0": {
                        "subset_aggregates": {
                            PRIMARY_EDIT_SUBSET: {"full_273": endpoint(1.0, 0.1)}
                        }
                    }
                },
                "ode": {
                    "subset_aggregates": {
                        PRIMARY_EDIT_SUBSET: {"full_273": endpoint(1.06, 0.1)}
                    }
                },
            },
        },
        "comparisons": {
            "candidate": {
                PRIMARY_EDIT_SUBSET: {
                    "0.0": {
                        "full_273": {
                            "assignment_advantage": {
                                "mean_delta_candidate_minus_baseline": 0.1
                            }
                        }
                    }
                }
            }
        },
    }
    failed = edit_guardrail(
        fixed,
        parent_label="parent",
        candidate_label="candidate",
        maximum_degradation=0.05,
    )
    assert failed["checks"]["ode32_correct_not_worse_than_empty"] is True
    assert failed["checks"]["ode32_correct_mse_nonregression"] is False
    assert failed["passed"] is False

    fixed["systems"]["candidate"]["ode"]["subset_aggregates"][
        PRIMARY_EDIT_SUBSET
    ]["full_273"]["correct_instruction_mse"] = 1.04
    passed = edit_guardrail(
        fixed,
        parent_label="parent",
        candidate_label="candidate",
        maximum_degradation=0.05,
    )
    assert passed["checks"]["ode32_correct_mse_nonregression"] is True
    assert passed["passed"] is True


def test_same_source_eval_pairs_groups_by_key_before_bootstrap() -> None:
    spaces = ("full_273", "continuous_269", "contact_4")

    def row(pair_ids: tuple[str, str], correct: float) -> dict:
        return {
            "pair_ids": list(pair_ids),
            "spaces": {
                space: {
                    "correct_instruction_mse": correct,
                    "swapped_instruction_mse": correct + 1.0,
                    "instruction_margin": 1.0,
                    "text_effect_rms": 0.5,
                    "correct_assignment": 1.0,
                    "swapped_assignment": 0.0,
                    "assignment_advantage": 1.0,
                    "empty_instruction_mse": correct + 0.5,
                    "correct_vs_empty_mse_gap": 0.5,
                    "correct_vs_empty_effect_rms": 0.25,
                    "source_copy_mse": 20.0,
                    "correct_vs_source_copy_mse_gain": 20.0 - correct,
                }
                for space in spaces
            },
        }

    baseline_rows = [row(("a", "b"), 1.0), row(("c", "d"), 10.0)]
    treatment_rows = [row(("d", "c"), 8.0), row(("b", "a"), 0.5)]
    wrap = lambda rows: {
        "timesteps": {
            "0.0": {
                "groups": rows,
                "aggregate": aggregate_group_records(rows),
            }
        }
    }
    comparisons = paired_bootstrap_comparisons(
        {"baseline": wrap(baseline_rows), "treatment": wrap(treatment_rows)},
        baseline_label="baseline",
        subsets={"pair_level_all": {("a", "b"), ("c", "d")}},
        bootstrap_samples=100,
        seed=11,
    )
    metric = comparisons["treatment"]["pair_level_all"]["0.0"]["full_273"][
        "correct_instruction_mse"
    ]
    assert metric["groups"] == 2
    assert metric["mean_delta_candidate_minus_baseline"] == pytest.approx(-1.25)


def test_same_source_eval_parses_parent_and_treatment_expectations() -> None:
    assert parse_system_expectation("parent=400000,none") == (
        "parent",
        400000,
        "",
    )
    assert parse_system_expectation("hinge=405000,same_source_hinge_only") == (
        "hinge",
        405000,
        "same_source_hinge_only",
    )


def _write_continuation_checkpoint(
    path: Path,
    *,
    step: int,
    treatment: str,
    parent: Path | None,
    null_parent: bool = False,
) -> None:
    torch.save(
        {
            "next_global_step": step,
            "model": {},
            "config": {
                "contract": {"name": "hy273_multitask_r13_unified273_v1"},
                "flow": {"contact_protocol": "unified_273_clean_flow_v1"},
            },
            "runtime_identity": {
                "research_overrides": {
                    "research_treatment": {"name": treatment},
                },
                "immediate_resume_parent": (
                    None
                    if parent is None and null_parent
                    else (
                        {}
                        if parent is None
                        else {"checkpoint": str(parent.resolve())}
                    )
                ),
            },
        },
        path,
    )


def _write_unified_actor_checkpoint(
    path: Path,
    *,
    step: int,
    run_name: str = "unified_actor_test_run",
    variance_eps: float = 1e-5,
) -> None:
    torch.save(
        {
            "format": "hy273_unified_actor_checkpoint_v1",
            "run_name": run_name,
            "next_global_step": step,
            "model": {},
            "ema": {},
            "config": {
                "schedule": {
                    "segments": [
                        {
                            "start": 0,
                            "end": 100000,
                            "t2m": 100,
                            "edit": 0,
                            "interaction": 0,
                        },
                        {
                            "start": 100000,
                            "end": 200000,
                            "t2m": 30,
                            "edit": 35,
                            "interaction": 35,
                        },
                    ]
                },
                "flow": {"prediction_type": "x0", "loss_space": "velocity_mse"},
                "text": {"encoder": "llm2vec_cache"},
                "model": {
                    "text_global_conditioning": "llm2vec_tokens_only",
                    "source_fusion_mode": "additive",
                    "dropout": 0.0,
                },
                "training": {
                    "max_global_step": 200000,
                    "stage_a_adaptation_lr": 0.0,
                    "stage_b_adaptation_lr": 1e-4,
                    "batch_size_t2m_edit_per_rank": 16,
                    "batch_size_interaction_per_rank": 8,
                },
                "normalization": {
                    "normalize_contacts": True,
                    "variance_eps": variance_eps,
                },
            },
            "normalization": {
                "normalize_contacts": True,
                "variance_eps": variance_eps,
            },
            "normalizer": {
                "mean": torch.zeros(273),
                "std": torch.ones(273),
                "variance_eps": torch.tensor(variance_eps),
            },
        },
        path,
    )


def test_same_source_eval_accepts_unified_actor_checkpoints(
    tmp_path: Path,
) -> None:
    stage_a = tmp_path / "stage_a_100k.pt"
    stage_b = tmp_path / "stage_b_200k.pt"
    _write_unified_actor_checkpoint(stage_a, step=100000)
    _write_unified_actor_checkpoint(stage_b, step=200000)

    metadata = validate_checkpoint_systems(
        [("stage_a", stage_a), ("stage_b", stage_b)],
        expectations={
            "stage_a": (100000, "t2m"),
            "stage_b": (200000, "t2m_edit_interaction"),
        },
        weight_source="ema",
    )

    assert [row["step"] for row in metadata] == [100000, 200000]
    assert {row["format"] for row in metadata} == {
        "hy273_unified_actor_checkpoint_v1"
    }
    assert [row["expected_treatment_label"] for row in metadata] == [
        "t2m",
        "t2m_edit_interaction",
    ]


def test_same_source_eval_accepts_same_run_continuation_schedule_extension(
    tmp_path: Path,
) -> None:
    stage_a = tmp_path / "stage_a_100k.pt"
    stage_b = tmp_path / "stage_b_250k.pt"
    _write_unified_actor_checkpoint(stage_a, step=100000)
    _write_unified_actor_checkpoint(stage_b, step=250000)
    checkpoint = torch.load(stage_b, map_location="cpu", weights_only=False)
    checkpoint["config"]["schedule"]["segments"][-1]["end"] = 250000
    checkpoint["config"]["training"]["max_global_step"] = 250000
    torch.save(checkpoint, stage_b)

    metadata = validate_checkpoint_systems(
        [("stage_a", stage_a), ("stage_b", stage_b)],
        expectations={
            "stage_a": (100000, "t2m"),
            "stage_b": (250000, "t2m_edit_interaction"),
        },
        weight_source="ema",
    )
    assert [row["step"] for row in metadata] == [100000, 250000]


def test_same_source_eval_rejects_unified_normalizer_scale_mismatch(
    tmp_path: Path,
) -> None:
    first = tmp_path / "eps_1e5.pt"
    second = tmp_path / "eps_1e4.pt"
    _write_unified_actor_checkpoint(first, step=100000, variance_eps=1e-5)
    _write_unified_actor_checkpoint(second, step=200000, variance_eps=1e-4)

    with pytest.raises(
        RuntimeError, match="different resolved scientific configs"
    ):
        validate_checkpoint_systems(
            [("first", first), ("second", second)],
            expectations={
                "first": (100000, "t2m"),
                "second": (200000, "multitask"),
            },
            weight_source="ema",
        )


def test_same_source_eval_rejects_mixed_legacy_unified_comparison(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.pt"
    unified = tmp_path / "unified.pt"
    _write_continuation_checkpoint(
        legacy,
        step=450000,
        treatment="legacy",
        parent=None,
    )
    _write_unified_actor_checkpoint(unified, step=100000)

    with pytest.raises(RuntimeError, match="Mixed legacy/unified"):
        validate_checkpoint_systems(
            [("legacy", legacy), ("unified", unified)],
            expectations={
                "legacy": (450000, "legacy"),
                "unified": (100000, "t2m"),
            },
            weight_source="model",
        )


def test_same_source_eval_rejects_unrelated_unified_runs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "run_a.pt"
    second = tmp_path / "run_b.pt"
    _write_unified_actor_checkpoint(first, step=100000, run_name="run_a")
    _write_unified_actor_checkpoint(second, step=200000, run_name="run_b")

    with pytest.raises(RuntimeError, match="same training run"):
        validate_checkpoint_systems(
            [("first", first), ("second", second)],
            expectations={
                "first": (100000, "t2m"),
                "second": (200000, "multitask"),
            },
            weight_source="ema",
        )


def test_same_source_eval_accepts_explicit_matched_continuations(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent400.pt"
    additive_parent = tmp_path / "additive405.pt"
    token_parent = tmp_path / "token405.pt"
    additive = tmp_path / "additive425.pt"
    token = tmp_path / "token425.pt"
    _write_continuation_checkpoint(
        root, step=400000, treatment="formal_default", parent=None
    )
    _write_continuation_checkpoint(
        additive_parent,
        step=405000,
        treatment="no_rank_positive_only",
        parent=root,
    )
    _write_continuation_checkpoint(
        token_parent,
        step=405000,
        treatment="source_token_block_positive_only",
        parent=root,
    )
    _write_continuation_checkpoint(
        additive,
        step=425000,
        treatment="no_rank_positive_only",
        parent=additive_parent,
    )
    _write_continuation_checkpoint(
        token,
        step=425000,
        treatment="source_token_block_positive_only",
        parent=token_parent,
    )
    checkpoints = [("additive", additive), ("token", token)]
    expectations = {
        "additive": (425000, "no_rank_positive_only"),
        "token": (425000, "source_token_block_positive_only"),
    }

    with pytest.raises(RuntimeError, match="do not share one resume parent"):
        validate_checkpoint_systems(
            checkpoints,
            expectations=expectations,
            weight_source="model",
        )

    metadata = validate_checkpoint_systems(
        checkpoints,
        expectations=expectations,
        weight_source="model",
        allow_matched_continuations=True,
    )
    assert {row["continuation_parent_step"] for row in metadata} == {405000}
    assert {row["common_fork_parent"] for row in metadata} == {
        str(root.resolve())
    }


def test_same_source_eval_accepts_named_default_as_compared_parent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent200.pt"
    candidate = tmp_path / "edit210.pt"
    _write_continuation_checkpoint(
        root,
        step=200000,
        treatment="formal_default",
        parent=None,
        null_parent=True,
    )
    _write_continuation_checkpoint(
        candidate,
        step=210000,
        treatment="no_rank_positive_only",
        parent=root,
    )

    metadata = validate_checkpoint_systems(
        [("parent", root), ("candidate", candidate)],
        expectations={
            "parent": (200000, "formal_default"),
            "candidate": (210000, "no_rank_positive_only"),
        },
        weight_source="model",
    )

    rows = {row["label"]: row for row in metadata}
    assert rows["parent"]["parent"] is None
    assert rows["candidate"]["parent"] == str(root.resolve())


def test_same_source_eval_accepts_direct_and_resumed_common_fork_branches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent400.pt"
    additive = tmp_path / "additive450.pt"
    token_405 = tmp_path / "token405.pt"
    token_425 = tmp_path / "token425.pt"
    token_450 = tmp_path / "token450.pt"
    _write_continuation_checkpoint(
        root, step=400000, treatment="", parent=None
    )
    _write_continuation_checkpoint(
        additive,
        step=450000,
        treatment="no_rank_positive_only",
        parent=root,
    )
    _write_continuation_checkpoint(
        token_405,
        step=405000,
        treatment="source_token_block_positive_only",
        parent=root,
    )
    _write_continuation_checkpoint(
        token_425,
        step=425000,
        treatment="source_token_block_positive_only",
        parent=token_405,
    )
    _write_continuation_checkpoint(
        token_450,
        step=450000,
        treatment="source_token_block_positive_only",
        parent=token_425,
    )

    metadata = validate_checkpoint_systems(
        [("additive", additive), ("token", token_450)],
        expectations={
            "additive": (450000, "no_rank_positive_only"),
            "token": (450000, "source_token_block_positive_only"),
        },
        weight_source="model",
        allow_matched_continuations=True,
    )

    rows = {row["label"]: row for row in metadata}
    assert rows["additive"]["lineage_steps"] == [400000]
    assert rows["token"]["lineage_steps"] == [425000, 405000, 400000]
    assert {row["common_fork_parent"] for row in metadata} == {
        str(root.resolve())
    }
