from __future__ import annotations

import numpy as np
import pytest

from tools.compare_hy273_reaction_matched import (
    EXPECTED_SAMPLING,
    REQUIRED_VARIANTS,
    _cluster_counts,
    _recompute_metrics_from_predictions,
    _validate_protocol,
)
from models.raw_motion.hy273_slices import DIM_HY273


def _event_row(length: int) -> dict[str, int]:
    pair_total = length * 22 * 22
    transition_total = (length - 1) * 22 * 22 * 2
    return {
        "length": length,
        "fk_pair_close_15cm_tp": 4,
        "fk_pair_close_15cm_fp": 3,
        "fk_pair_close_15cm_fn": 2,
        "fk_pair_close_15cm_target_positive": 6,
        "fk_pair_close_15cm_target_negative": pair_total - 6,
        "fk_pair_transition_15cm_tp": 2,
        "fk_pair_transition_15cm_fp": 5,
        "fk_pair_transition_15cm_fn": 1,
        "fk_pair_transition_15cm_target_positive": 3,
        "fk_pair_transition_15cm_target_negative": transition_total - 3,
    }


def test_cluster_counts_supports_pair_contact_and_directional_transition() -> None:
    grouped = {"uid": [_event_row(length=4)]}

    pair = _cluster_counts(
        grouped,
        ["uid"],
        "fk_pair_",
        15,
        pairs_per_frame=22 * 22,
    )
    transition = _cluster_counts(
        grouped,
        ["uid"],
        "fk_pair_",
        15,
        event_name="transition",
        pairs_per_frame=22 * 22,
        frame_offset=1,
        event_channels=2,
    )

    np.testing.assert_array_equal(pair[0, :3], [4, 3, 2])
    np.testing.assert_array_equal(transition[0, :3], [2, 5, 1])


def test_cluster_counts_rejects_wrong_pair_denominator() -> None:
    row = _event_row(length=4)
    row["fk_pair_close_15cm_target_negative"] -= 1

    with pytest.raises(ValueError, match="event map"):
        _cluster_counts(
            {"uid": [row]},
            ["uid"],
            "fk_pair_",
            15,
            pairs_per_frame=22 * 22,
        )


def test_protocol_validation_accepts_fixed_200k_test_selection() -> None:
    rows = [{}] * 1_579
    payload = {
        "format": "hy273_fixed_role_reaction_eval_v2",
        "dataset": "Inter-X K273",
        "split": "test",
        "caption_policy": "uid_balanced",
        "weight_source": "ema",
        "assignment_rule": "fixed_source_actor_to_target_reactor_no_swap",
        "checkpoint_next_global_step": 200_000,
        "sampling": dict(EXPECTED_SAMPLING),
        "selection": {
            "count": 1_579,
            "dataset_count_after_filters": 1_579,
            "start_index": 0,
        },
        "per_sample": {variant: rows for variant in REQUIRED_VARIANTS},
        "protocols": {
            "causal_ablations": ["empty", "shuffled_text", "unrelated_source"]
        },
        "negative_donor_protocol": {"scope": "complete_filtered_split"},
    }

    _validate_protocol(
        payload,
        payload,
        expected_checkpoint_step=200_000,
        expected_split="test",
    )


def test_prediction_recomputation_rejects_mismatched_report(tmp_path) -> None:
    uid = "case"
    length = 4
    case_dir = tmp_path / uid
    case_dir.mkdir()
    motion = np.zeros((length, DIM_HY273), dtype=np.float32)
    np.save(case_dir / "source.npy", motion)
    np.save(case_dir / "target.npy", motion)
    for variant in REQUIRED_VARIANTS:
        np.save(case_dir / f"{variant}.npy", motion)
    (case_dir / "metadata.json").write_text(
        '{"uid":"case","length":4,"caption_index":0}\n',
        encoding="utf-8",
    )
    payload = {
        "per_sample": {
            variant: [
                {
                    "uid": uid,
                    "variant": variant,
                    "caption_index": 0,
                    "dataset_index": 0,
                    "length": length,
                    "reactor_fk_mpjpe_cm": 123.0,
                    "reactor_root_error_cm": 0.0,
                    "fk_relation_distance_mae_cm": 0.0,
                }
            ]
            for variant in REQUIRED_VARIANTS
        }
    }

    with pytest.raises(ValueError, match="do not match report row"):
        _recompute_metrics_from_predictions(payload, tmp_path)


def test_prediction_recomputation_requires_all_consistency_metrics(tmp_path) -> None:
    uid = "case"
    length = 4
    case_dir = tmp_path / uid
    case_dir.mkdir()
    motion = np.zeros((length, DIM_HY273), dtype=np.float32)
    np.save(case_dir / "source.npy", motion)
    np.save(case_dir / "target.npy", motion)
    for variant in REQUIRED_VARIANTS:
        np.save(case_dir / f"{variant}.npy", motion)
    (case_dir / "metadata.json").write_text(
        '{"uid":"case","length":4,"caption_index":0}\n',
        encoding="utf-8",
    )
    payload = {
        "per_sample": {
            variant: [
                {
                    "uid": uid,
                    "variant": variant,
                    "caption_index": 0,
                    "dataset_index": 0,
                    "length": length,
                    "reactor_fk_mpjpe_cm": 0.0,
                    "reactor_root_error_cm": 0.0,
                }
            ]
            for variant in REQUIRED_VARIANTS
        }
    }

    with pytest.raises(ValueError, match="missing required consistency metric"):
        _recompute_metrics_from_predictions(payload, tmp_path)
