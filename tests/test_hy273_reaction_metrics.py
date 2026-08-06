from __future__ import annotations

import json

import numpy as np
import torch

from tools.compare_hy273_reaction_matched import (
    REQUIRED_VARIANTS,
    _paired_ratio_comparison,
    _recompute_metrics_from_predictions,
    _validate_matched_rows,
)
from models.raw_motion.hy273_reaction_metrics import reaction_fixed_role_metrics
from models.raw_motion.hy273_slices import (
    DIM_HY273,
    GLOBAL_ROT_SLICE,
    HEADING_SLICE,
    JOINT_POS_SLICE,
    ROOT_SLICE,
)
from tools.eval_hy273_reaction import _aggregate_rows, _matched_advantage


def _motion(batch: int = 1, frames: int = 6) -> torch.Tensor:
    value = torch.zeros(batch, frames, DIM_HY273)
    value[..., HEADING_SLICE] = torch.tensor([1.0, 0.0])
    value[..., GLOBAL_ROT_SLICE] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ).repeat(22)
    return value


def test_fixed_role_reaction_metrics_are_zero_for_exact_reactor() -> None:
    source = _motion()
    target = _motion()
    target[..., ROOT_SLICE.start] = 1.0
    target[..., JOINT_POS_SLICE] = 0.1
    result = reaction_fixed_role_metrics(source, target, target)
    assert result["assignment_rule"] == "fixed_source_actor_to_target_reactor_no_swap"
    assert result["aggregate"]["reactor_position_mpjpe_cm"] == 0.0
    assert result["aggregate"]["reactor_fk_mpjpe_cm"] == 0.0
    assert result["per_sample"][0]["assignment"] == "fixed_actor_to_reactor"


def test_copying_source_is_not_hidden_by_actor_swap_or_pair_average() -> None:
    source = _motion()
    target = _motion()
    target[..., ROOT_SLICE.start] = 2.0
    copied_source = source.clone()
    result = reaction_fixed_role_metrics(source, copied_source, target)
    assert result["aggregate"]["reactor_position_mpjpe_cm"] > 190.0
    assert result["aggregate"]["reactor_root_error_cm"] > 190.0
    assert result["aggregate"]["prediction_to_source_position_mpjpe_cm"] == 0.0


def test_reaction_metrics_respect_lengths() -> None:
    source = _motion(batch=2)
    target = source.clone()
    prediction = target.clone()
    prediction[0, 3:, ROOT_SLICE.start] = 100.0
    result = reaction_fixed_role_metrics(
        source,
        prediction,
        target,
        lengths=torch.tensor([3, 6]),
    )
    assert result["per_sample"][0]["reactor_root_error_cm"] == 0.0


def test_reaction_metrics_report_root_radius_bearing_and_close_confusion() -> None:
    source = _motion(frames=6)
    target = _motion(frames=6)
    prediction = _motion(frames=6)
    target[:, :3, ROOT_SLICE.start] = 0.0
    target[:, 3:, ROOT_SLICE.start] = 3.0
    prediction[:, :3, ROOT_SLICE.start] = 3.0
    prediction[:, 3:, ROOT_SLICE.start] = 0.0

    result = reaction_fixed_role_metrics(source, prediction, target)
    aggregate = result["aggregate"]
    assert aggregate["close_20cm_precision"] == 0.0
    assert aggregate["close_20cm_recall"] == 0.0
    assert aggregate["false_close_rate_20cm"] == 1.0
    assert aggregate["missed_close_rate_20cm"] == 1.0
    assert aggregate["fk_close_20cm_precision"] == 0.0
    assert aggregate["fk_close_20cm_recall"] == 0.0
    assert aggregate["fk_false_close_rate_20cm"] == 1.0
    assert aggregate["fk_missed_close_rate_20cm"] == 1.0
    assert aggregate["relative_root_radius_error_cm"] == 300.0
    assert aggregate["relative_root_bearing_valid_frame_fraction"] == 0.5
    assert result["per_sample"][0]["relative_root_bearing_valid_frames"] == 3

    target[..., ROOT_SLICE.start] = 1.0
    prediction[..., ROOT_SLICE.start] = 0.0
    prediction[..., ROOT_SLICE.start + 2] = 1.0
    result = reaction_fixed_role_metrics(source, prediction, target)
    assert abs(result["aggregate"]["relative_root_bearing_error_deg"] - 90.0) < 1e-4
    assert abs(result["aggregate"]["partner_facing_error_deg"] - 90.0) < 1e-4


def test_reaction_metrics_keep_position_and_fk_close_paths_separate() -> None:
    source = _motion(frames=2)
    target = _motion(frames=2)
    prediction = _motion(frames=2)
    target[..., ROOT_SLICE.start] = 3.0
    prediction[..., ROOT_SLICE.start] = 3.0
    target[..., JOINT_POS_SLICE.start + 3] = -3.0
    prediction[..., JOINT_POS_SLICE.start + 3] = -3.0

    result = reaction_fixed_role_metrics(source, prediction, target)
    assert result["aggregate"]["close_20cm_f1"] == 1.0
    assert result["aggregate"]["fk_close_20cm_f1"] is None
    assert result["aggregate"]["target_min_inter_actor_joint_cm"] == 0.0
    assert result["aggregate"]["target_fk_min_inter_actor_joint_cm"] > 90.0


def test_reaction_metrics_report_fk_pair_contact_and_lifecycle() -> None:
    source = _motion(frames=4)
    target = _motion(frames=4)
    exact = target.clone()
    target[..., ROOT_SLICE.start] = torch.tensor([3.0, 0.0, 0.0, 3.0])
    exact.copy_(target)

    exact_result = reaction_fixed_role_metrics(source, exact, target)
    exact_aggregate = exact_result["aggregate"]
    assert exact_aggregate["fk_pair_close_15cm_f1"] == 1.0
    assert exact_aggregate["fk_pair_transition_15cm_f1"] == 1.0
    assert exact_aggregate["fk_contact_vector_error_cm_15cm"] == 0.0

    shifted = _motion(frames=4)
    shifted[..., ROOT_SLICE.start] = torch.tensor([3.0, 3.0, 0.0, 0.0])
    shifted_result = reaction_fixed_role_metrics(source, shifted, target)
    shifted_aggregate = shifted_result["aggregate"]
    assert shifted_aggregate["fk_pair_close_15cm_f1"] < 1.0
    assert shifted_aggregate["fk_pair_transition_15cm_f1"] < 1.0
    assert shifted_aggregate["fk_contact_vector_error_cm_15cm"] > 0.0
    row = shifted_result["per_sample"][0]
    assert row["fk_pair_close_15cm_target_positive"] > 0
    assert row["fk_pair_transition_15cm_target_positive"] > 0
    assert row["fk_contact_vector_target_pairs_15cm"] > 0


def test_reaction_metrics_do_not_match_release_to_onset() -> None:
    source = _motion(frames=2)
    target = _motion(frames=2)
    prediction = _motion(frames=2)
    target[..., ROOT_SLICE.start] = torch.tensor([3.0, 0.0])
    prediction[..., ROOT_SLICE.start] = torch.tensor([0.0, 3.0])

    result = reaction_fixed_role_metrics(source, prediction, target)
    aggregate = result["aggregate"]
    assert aggregate["fk_pair_transition_15cm_precision"] == 0.0
    assert aggregate["fk_pair_transition_15cm_recall"] == 0.0
    assert aggregate["fk_pair_transition_15cm_f1"] == 0.0


def test_reaction_metrics_ignore_padded_fk_pair_contact_errors() -> None:
    source = _motion(frames=3)
    target = _motion(frames=3)
    prediction = target.clone()
    target[..., ROOT_SLICE.start] = 3.0
    prediction.copy_(target)
    prediction[:, 2, ROOT_SLICE.start] = 0.0
    result = reaction_fixed_role_metrics(
        source,
        prediction,
        target,
        lengths=torch.tensor([2]),
    )
    assert result["aggregate"]["fk_pair_close_15cm_f1"] is None
    assert result["aggregate"]["fk_pair_transition_15cm_f1"] is None
    assert result["aggregate"]["fk_contact_vector_error_cm_15cm"] is None


def test_reaction_metrics_report_initial_precontact_and_close_timing() -> None:
    source = _motion(frames=20)
    target = _motion(frames=20)
    prediction = _motion(frames=20)

    # GT first reaches the source at frame 10; prediction reaches it at frame 5.
    target[:, :10, ROOT_SLICE.start] = 1.0
    prediction[:, :5, ROOT_SLICE.start] = 1.0
    prediction[:, :5, HEADING_SLICE] = torch.tensor([0.0, 1.0])

    aggregate = reaction_fixed_role_metrics(source, prediction, target)["aggregate"]
    assert aggregate["frame0_relative_root_error_cm"] == 0.0
    assert abs(aggregate["initial_15f_relative_root_error_cm"] - 100.0 / 3.0) < 1e-4
    assert abs(aggregate["precontact_relative_root_error_cm"] - 50.0) < 1e-4
    assert abs(aggregate["frame0_relative_heading_error_deg"] - 90.0) < 1e-4
    assert abs(aggregate["initial_15f_relative_heading_error_deg"] - 30.0) < 1e-4
    assert abs(aggregate["precontact_relative_heading_error_deg"] - 45.0) < 1e-4
    assert abs(aggregate["precontact_frame_fraction"] - 0.5) < 1e-6
    assert abs(aggregate["first_close_timing_error_s_20cm"] - 5.0 / 30.0) < 1e-6
    assert abs(aggregate["first_close_too_early_s_20cm"] - 5.0 / 30.0) < 1e-6
    assert aggregate["first_close_too_late_s_20cm"] == 0.0
    assert abs(aggregate["precontact_false_close_rate_20cm"] - 0.5) < 1e-6


def test_reaction_metrics_mark_precontact_errors_unavailable_at_frame0_contact() -> None:
    source = _motion(frames=4)
    target = _motion(frames=4)
    prediction = _motion(frames=4)
    prediction[..., ROOT_SLICE.start] = 1.0
    prediction[..., HEADING_SLICE] = torch.tensor([0.0, 1.0])

    result = reaction_fixed_role_metrics(source, prediction, target)
    row = result["per_sample"][0]
    assert row["precontact_valid_frames_20cm"] == 0
    assert row["precontact_relative_root_error_cm"] is None
    assert row["precontact_relative_heading_error_deg"] is None
    assert row["precontact_false_close_rate_20cm"] is None
    assert result["aggregate"]["precontact_relative_root_error_cm"] is None
    assert result["aggregate"]["precontact_relative_heading_error_deg"] is None
    aggregate = _aggregate_rows(
        result["per_sample"],
        seed=8,
        resamples=100,
        confidence=0.95,
    )
    assert aggregate["precontact_relative_root_error_cm"]["mean"] is None
    assert aggregate["precontact_relative_heading_error_deg"]["mean"] is None
    assert aggregate["precontact_false_close_rate_20cm"]["mean"] is None


def test_reaction_metrics_mark_zero_denominator_close_metrics_unavailable() -> None:
    source = _motion(frames=4)
    target = _motion(frames=4)
    prediction = _motion(frames=4)
    target[..., ROOT_SLICE.start] = 3.0
    prediction[..., ROOT_SLICE.start] = 3.0

    result = reaction_fixed_role_metrics(source, prediction, target)
    row = result["per_sample"][0]
    assert row["close_20cm_precision"] is None
    assert row["close_20cm_recall"] is None
    assert row["close_20cm_f1"] is None
    assert row["missed_close_rate_20cm"] is None
    assert result["aggregate"]["close_20cm_precision"] is None
    aggregate = _aggregate_rows(
        result["per_sample"],
        seed=12,
        resamples=100,
        confidence=0.95,
    )
    assert aggregate["close_20cm_precision"]["mean"] is None
    assert aggregate["close_20cm_recall"]["mean"] is None
    assert aggregate["missed_close_rate_20cm"]["mean"] is None


def test_reaction_eval_uses_micro_not_macro_close_precision() -> None:
    rows = []
    for tp, fp in ((1, 0), (1, 8)):
        row: dict[str, float | int] = {"dummy_metric": 1.0}
        for threshold in (10, 20, 30):
            prefix = f"close_{threshold}cm"
            row.update(
                {
                    f"{prefix}_precision": float(tp / (tp + fp)),
                    f"{prefix}_recall": 1.0,
                    f"{prefix}_f1": float(2 * tp / (2 * tp + fp)),
                    f"false_close_rate_{threshold}cm": float(fp / 10),
                    f"missed_close_rate_{threshold}cm": 0.0,
                    f"{prefix}_tp": tp,
                    f"{prefix}_fp": fp,
                    f"{prefix}_fn": 0,
                    f"{prefix}_target_positive": tp,
                    f"{prefix}_target_negative": 10,
                }
            )
        rows.append(row)
    aggregate = _aggregate_rows(
        rows,
        seed=7,
        resamples=100,
        confidence=0.95,
    )
    assert abs(float(aggregate["close_20cm_precision"]["mean"]) - 0.2) < 1e-8


def test_reaction_eval_micro_pools_fk_pair_events_and_contact_vectors() -> None:
    rows = []
    for tp, fp, vector_sum, vector_count in (
        (1, 0, 10.0, 1),
        (1, 8, 90.0, 9),
    ):
        row: dict[str, float | int] = {
            "fk_pair_close_15cm_precision": float(tp / (tp + fp)),
            "fk_pair_close_15cm_recall": 1.0,
            "fk_pair_close_15cm_f1": float(2 * tp / (2 * tp + fp)),
            "fk_pair_false_close_rate_15cm": float(fp / 10),
            "fk_pair_missed_close_rate_15cm": 0.0,
            "fk_pair_close_15cm_tp": tp,
            "fk_pair_close_15cm_fp": fp,
            "fk_pair_close_15cm_fn": 0,
            "fk_pair_close_15cm_target_positive": tp,
            "fk_pair_close_15cm_target_negative": 10,
            "fk_contact_vector_error_cm_15cm": vector_sum / vector_count,
            "fk_contact_vector_error_sum_cm_15cm": vector_sum,
            "fk_contact_vector_target_pairs_15cm": vector_count,
        }
        rows.append(row)
    aggregate = _aggregate_rows(
        rows,
        seed=19,
        resamples=100,
        confidence=0.95,
    )
    assert abs(
        float(aggregate["fk_pair_close_15cm_precision"]["mean"]) - 0.2
    ) < 1e-8
    assert abs(
        float(aggregate["fk_contact_vector_error_cm_15cm"]["mean"]) - 10.0
    ) < 1e-8


def test_reaction_eval_uses_micro_precontact_false_close_rate() -> None:
    rows = [
        {
            "precontact_false_close_rate_20cm": 1.0,
            "precontact_false_close_frames_20cm": 1,
            "precontact_valid_frames_20cm": 1,
        },
        {
            "precontact_false_close_rate_20cm": 0.1,
            "precontact_false_close_frames_20cm": 1,
            "precontact_valid_frames_20cm": 10,
        },
    ]
    aggregate = _aggregate_rows(
        rows,
        seed=9,
        resamples=100,
        confidence=0.95,
    )
    assert abs(
        float(aggregate["precontact_false_close_rate_20cm"]["mean"]) - 2.0 / 11.0
    ) < 1e-8


def test_reaction_eval_pools_precontact_errors_over_valid_frames() -> None:
    rows = [
        {
            "precontact_relative_root_error_cm": None,
            "precontact_relative_root_error_sum_cm": 0.0,
            "precontact_relative_heading_error_deg": None,
            "precontact_relative_heading_error_sum_deg": 0.0,
            "precontact_valid_frames_20cm": 0,
        },
        {
            "precontact_relative_root_error_cm": 10.0,
            "precontact_relative_root_error_sum_cm": 20.0,
            "precontact_relative_heading_error_deg": 30.0,
            "precontact_relative_heading_error_sum_deg": 60.0,
            "precontact_valid_frames_20cm": 2,
        },
        {
            "precontact_relative_root_error_cm": 20.0,
            "precontact_relative_root_error_sum_cm": 20.0,
            "precontact_relative_heading_error_deg": 60.0,
            "precontact_relative_heading_error_sum_deg": 60.0,
            "precontact_valid_frames_20cm": 1,
        },
    ]
    aggregate = _aggregate_rows(
        rows,
        seed=10,
        resamples=100,
        confidence=0.95,
    )
    assert abs(
        float(aggregate["precontact_relative_root_error_cm"]["mean"])
        - 40.0 / 3.0
    ) < 1e-8
    assert abs(
        float(aggregate["precontact_relative_heading_error_deg"]["mean"])
        - 40.0
    ) < 1e-8


def test_reaction_causal_advantage_uses_matched_micro_close_counts() -> None:
    correct = [
        {
            "uid": "a",
            "close_20cm_precision": 1.0,
            "close_20cm_tp": 1,
            "close_20cm_fp": 0,
        },
        {
            "uid": "b",
            "close_20cm_precision": 1.0 / 9.0,
            "close_20cm_tp": 1,
            "close_20cm_fp": 8,
        },
    ]
    ablated = [
        {
            "uid": "a",
            "close_20cm_precision": 0.0,
            "close_20cm_tp": 0,
            "close_20cm_fp": 0,
        },
        {
            "uid": "b",
            "close_20cm_precision": 1.0,
            "close_20cm_tp": 1,
            "close_20cm_fp": 0,
        },
    ]
    result = _matched_advantage(
        correct,
        ablated,
        seed=11,
        resamples=100,
        confidence=0.95,
    )
    assert abs(float(result["close_20cm_precision"]["mean"]) + 0.8) < 1e-8


def test_reaction_matched_comparison_ignores_batch_local_index() -> None:
    baseline = {"case": [{"uid": "case", "index": 0}]}
    candidate = {"case": [{"uid": "case", "index": 7}]}
    assert _validate_matched_rows(baseline, candidate) == ["case"]


def test_reaction_matched_pooled_ratio_uses_shared_valid_frames() -> None:
    baseline = np.asarray(((10.0, 2.0), (0.0, 0.0), (20.0, 1.0)))
    candidate = np.asarray(((8.0, 2.0), (0.0, 0.0), (18.0, 1.0)))
    result = _paired_ratio_comparison(
        baseline,
        candidate,
        rng=np.random.default_rng(17),
        resamples=1000,
    )
    assert abs(float(result["baseline"]) - 10.0) < 1e-8
    assert abs(float(result["candidate"]) - 26.0 / 3.0) < 1e-8
    assert abs(float(result["candidate_minus_baseline"]) + 4.0 / 3.0) < 1e-8


def test_reaction_matched_metrics_can_be_recomputed_from_saved_predictions(
    tmp_path,
) -> None:
    uid = "case"
    length = 4
    case_dir = tmp_path / uid
    case_dir.mkdir()
    source = _motion(frames=length)[0]
    target = source.clone()
    target[..., ROOT_SLICE.start] = 1.0
    np.save(case_dir / "source.npy", source.numpy())
    np.save(case_dir / "target.npy", target.numpy())
    for variant in REQUIRED_VARIANTS:
        np.save(case_dir / f"{variant}.npy", target.numpy())
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "uid": uid,
                "length": length,
                "caption_index": 0,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "per_sample": {
            variant: [
                {
                    "uid": uid,
                    "variant": variant,
                    "index": 99,
                    "dataset_index": 0,
                    "caption_index": 0,
                    "length": length,
                    "reactor_fk_mpjpe_cm": 0.0,
                    "reactor_root_error_cm": 0.0,
                    "fk_relation_distance_mae_cm": 0.0,
                }
            ]
            for variant in REQUIRED_VARIANTS
        }
    }
    result = _recompute_metrics_from_predictions(payload, tmp_path)
    row = result["per_sample"]["source_text"][0]
    assert row["index"] == 99
    assert row["frame0_relative_root_error_cm"] == 0.0
    assert "precontact_relative_root_error_sum_cm" in row
