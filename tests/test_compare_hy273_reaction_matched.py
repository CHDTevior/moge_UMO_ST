from __future__ import annotations

import json
import numpy as np
import pytest

from tools.compare_hy273_reaction_matched import (
    EXPECTED_SAMPLING,
    REQUIRED_VARIANTS,
    _cluster_counts,
    _replay_cursor_state,
    _recompute_metrics_from_predictions,
    _validate_prediction_input_identity,
    _validate_training_contract,
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
        baseline_checkpoint_step=200_000,
        candidate_checkpoint_step=200_000,
        expected_split="test",
    )


def _dose_configs() -> tuple[dict, dict]:
    base_config = {
        "schedule": {
            "segments": [
                {"start": 0, "end": 100_000, "t2m": 100, "edit": 0, "reaction": 0},
                {"start": 100_000, "end": 200_000, "t2m": 30, "edit": 35, "reaction": 35},
            ]
        },
        "training": {"max_global_step": 200_000},
    }
    candidate_config = {
        **base_config,
        "schedule": {
            "segments": [
                base_config["schedule"]["segments"][0],
                {**base_config["schedule"]["segments"][1], "end": 300_000},
            ]
        },
        "training": {"max_global_step": 300_000},
    }
    return base_config, candidate_config


def _dose_contract(
    step: int,
    config: dict,
    counts: tuple[int, int, int],
) -> dict:
    static = {
        "format": "batcher",
        "multitask_manifest": "manifest",
        "interaction_root": "interaction",
        "run_seed": 7,
        "world_size": 8,
        "interaction_exclude_overlength": True,
        "paired_task": "reaction",
        "local_batch_sizes": {"0": 16, "1": 8, "2": 8},
        "manifest_hashes": {"train": "hash"},
    }
    hml, edit, reaction = counts
    global_batches = (128, 64, 64)
    return {
        "path": f"step_{step:08d}.pt",
        "run_name": "same_run",
        "config_path": "config.yaml",
        "config": config,
        "rng_contract": "stateless",
        "ema_update_count": step // 10,
        "batcher": {
            **static,
            "scheduler": {
                "segments": config["schedule"]["segments"],
                "state": {
                    "next_step": step,
                    "debt_hml": 0,
                    "debt_edit": 0,
                    "debt_interaction": 0,
                    "realized_hml": hml,
                    "realized_edit": edit,
                    "realized_interaction": reaction,
                },
            },
            "cursors": {
                str(stream): {"global_batch_size": global_batch}
                for stream, global_batch in enumerate(global_batches)
            },
            "next_global_sample_ordinal": {
                "0": hml * global_batches[0],
                "1": edit * global_batches[1],
                "2": reaction * global_batches[2],
            },
        },
    }


def test_dose_contract_accepts_exact_same_mix_extension(monkeypatch) -> None:
    base_config, candidate_config = _dose_configs()

    contracts = iter(
        (
            _dose_contract(200_000, base_config, (130_000, 35_000, 35_000)),
            _dose_contract(300_000, candidate_config, (160_000, 70_000, 70_000)),
        )
    )
    monkeypatch.setattr(
        "tools.compare_hy273_reaction_matched._load_checkpoint_contract",
        lambda *args, **kwargs: next(contracts),
    )
    monkeypatch.setattr(
        "tools.compare_hy273_reaction_matched._validate_stream_continuity",
        lambda *args, **kwargs: {"exact_cursor_state_match": True},
    )
    result = _validate_training_contract(
        {},
        {},
        mode="same_run_dose_extension",
        baseline_checkpoint_step=200_000,
        candidate_checkpoint_step=300_000,
    )
    assert result["additional_task_updates"] == {
        "realized_hml": 30_000,
        "realized_edit": 35_000,
        "realized_interaction": 35_000,
    }
    assert result["same_run_metadata_and_stream_continuity_verified"] is True
    assert result["checkpoint_resume_lineage_verified_by_comparator"] is False
    assert result["parent_lineage_verified_by_comparator"] is False
    assert (
        result["parent_lineage_status"]
        == "not_revalidated_for_same_run_dose_comparison"
    )


def test_independent_run_contract_requires_external_parent_lineage(
    monkeypatch,
) -> None:
    baseline_config, _ = _dose_configs()
    baseline_config = {**baseline_config, "reaction_loss": {}}
    candidate_config = {
        **baseline_config,
        "reaction_loss": {
            "fk_contact_map_negative": 0.005,
            "fk_contact_map_positive": 0.001,
            "fk_contact_temperature_m": 0.02,
            "fk_contact_threshold_m": 0.15,
            "fk_contact_transition": 0.003,
            "fk_contact_transition_beta": 0.1,
            "fk_contact_vector": 0.002,
            "fk_contact_vector_scale_m": 0.05,
        },
    }
    baseline = _dose_contract(
        200_000,
        baseline_config,
        (130_000, 35_000, 35_000),
    )
    candidate = _dose_contract(
        200_000,
        candidate_config,
        (130_000, 35_000, 35_000),
    )
    baseline["run_name"] = "reaction_v5"
    candidate["run_name"] = "reaction_v5_1"
    contracts = iter((baseline, candidate))
    monkeypatch.setattr(
        "tools.compare_hy273_reaction_matched._load_checkpoint_contract",
        lambda *args, **kwargs: next(contracts),
    )

    result = _validate_training_contract(
        {},
        {},
        mode="reaction_v5_1_full_contact",
        baseline_checkpoint_step=200_000,
        candidate_checkpoint_step=200_000,
    )

    assert result["parent_lineage_status"] == "external_launch_record_required"
    assert result["parent_lineage_verified_by_comparator"] is False
    assert "external launch records" in result["parent_lineage_note"]


def _v5_2_gate_contracts(*, extra_candidate_change: bool = False) -> tuple[dict, dict]:
    baseline_config, _ = _dose_configs()
    baseline_config = {
        **baseline_config,
        "reaction_loss": {
            "fine_min_flow_t": 0.2,
            "joint_distance": 0.0273,
            "min_flow_t": 0.2,
        },
    }
    candidate_reaction_loss = {
        **baseline_config["reaction_loss"],
        "fine_min_flow_t": 0.0,
        "min_flow_t": 0.0,
    }
    if extra_candidate_change:
        candidate_reaction_loss["joint_distance"] = 0.03
    candidate_config = {
        **baseline_config,
        "reaction_loss": candidate_reaction_loss,
    }
    baseline = _dose_contract(
        200_000,
        baseline_config,
        (130_000, 35_000, 35_000),
    )
    candidate = _dose_contract(
        200_000,
        candidate_config,
        (130_000, 35_000, 35_000),
    )
    baseline["run_name"] = "reaction_v5_1"
    candidate["run_name"] = "reaction_v5_2"
    return baseline, candidate


def test_v5_2_contract_accepts_only_all_timestep_gate_change(monkeypatch) -> None:
    contracts = iter(_v5_2_gate_contracts())
    monkeypatch.setattr(
        "tools.compare_hy273_reaction_matched._load_checkpoint_contract",
        lambda *args, **kwargs: next(contracts),
    )

    result = _validate_training_contract(
        {},
        {},
        mode="reaction_v5_2_all_t_fine",
        baseline_checkpoint_step=200_000,
        candidate_checkpoint_step=200_000,
    )

    assert result["config_differences"] == [
        {
            "path": "reaction_loss.fine_min_flow_t",
            "baseline": 0.2,
            "candidate": 0.0,
        },
        {
            "path": "reaction_loss.min_flow_t",
            "baseline": 0.2,
            "candidate": 0.0,
        },
    ]


def test_v5_2_contract_rejects_an_extra_loss_change(monkeypatch) -> None:
    contracts = iter(_v5_2_gate_contracts(extra_candidate_change=True))
    monkeypatch.setattr(
        "tools.compare_hy273_reaction_matched._load_checkpoint_contract",
        lambda *args, **kwargs: next(contracts),
    )

    with pytest.raises(ValueError, match="do not implement"):
        _validate_training_contract(
            {},
            {},
            mode="reaction_v5_2_all_t_fine",
            baseline_checkpoint_step=200_000,
            candidate_checkpoint_step=200_000,
        )


def test_dose_contract_rejects_shifted_absolute_exposures(monkeypatch) -> None:
    base_config, candidate_config = _dose_configs()
    contracts = iter(
        (
            _dose_contract(200_000, base_config, (129_000, 36_000, 35_000)),
            _dose_contract(300_000, candidate_config, (159_000, 71_000, 70_000)),
        )
    )
    monkeypatch.setattr(
        "tools.compare_hy273_reaction_matched._load_checkpoint_contract",
        lambda *args, **kwargs: next(contracts),
    )
    monkeypatch.setattr(
        "tools.compare_hy273_reaction_matched._validate_stream_continuity",
        lambda *args, **kwargs: {"exact_cursor_state_match": True},
    )
    with pytest.raises(ValueError, match="wrong absolute task exposures"):
        _validate_training_contract(
            {},
            {},
            mode="same_run_dose_extension",
            baseline_checkpoint_step=200_000,
            candidate_checkpoint_step=300_000,
        )


def test_cursor_replay_composes_across_resume_boundary() -> None:
    saved = {
        "format": "hy273_sortish_stream_cursor_v1",
        "manifest_sha256": "manifest",
        "run_seed": 7,
        "stream": 0,
        "global_batch_size": 2,
        "sort_window_batches": 3,
        "row_count": 11,
        "cycle": 0,
        "offset": 0,
        "pending_batches": [],
    }
    kwargs = {
        "row_bucket_keys": tuple((index % 3, 0) for index in range(11)),
        "manifest_sha256": "manifest",
        "run_seed": 7,
        "stream": 0,
    }
    at_three = _replay_cursor_state(saved, updates=3, **kwargs)
    resumed = _replay_cursor_state(at_three, updates=4, **kwargs)
    uninterrupted = _replay_cursor_state(saved, updates=7, **kwargs)
    assert resumed == uninterrupted
    assert uninterrupted != saved


def _write_prediction_input_case(root, *, text: str, target_value: float) -> None:
    case = root / "case"
    case.mkdir(parents=True)
    source = np.zeros((4, DIM_HY273), dtype=np.float32)
    target = np.full((4, DIM_HY273), target_value, dtype=np.float32)
    np.save(case / "source.npy", source)
    np.save(case / "target.npy", target)
    (case / "metadata.json").write_text(
        json.dumps(
            {
                "uid": "case",
                "text": text,
                "caption_index": 0,
                "negative_donor_uid": "donor",
                "negative_donor_text": "other",
                "actor_person_index": 0,
                "length": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _prediction_identity_payload() -> dict:
    return {
        "per_sample": {
            "source_text": [
                {
                    "uid": "case",
                    "variant": "source_text",
                    "caption_index": 0,
                    "dataset_index": 0,
                }
            ]
        }
    }


def test_prediction_input_identity_checks_physical_data_and_text(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_prediction_input_case(baseline, text="react", target_value=1.0)
    _write_prediction_input_case(candidate, text="react", target_value=1.0)
    result = _validate_prediction_input_identity(
        _prediction_identity_payload(),
        _prediction_identity_payload(),
        baseline,
        candidate,
    )
    assert result["uids"] == 1

    np.save(
        candidate / "case" / "target.npy",
        np.full((4, DIM_HY273), 2.0, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="target.npy content differs"):
        _validate_prediction_input_identity(
            _prediction_identity_payload(),
            _prediction_identity_payload(),
            baseline,
            candidate,
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
