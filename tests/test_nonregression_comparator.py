from __future__ import annotations

import json

import numpy as np
import pytest

from tools.compare_hy273_nonregression import (
    PRODUCTION_METRICS_BY_SUBTYPE,
    PRODUCTION_PROTOCOL_VERSION,
    UNIT_INTERVAL_METRICS,
    _metric_names,
    _validate_metrics,
    align_records,
    allowed_degradation,
    build_artifacts,
    paired_bootstrap_mean_ci,
    run_self_test,
    signed_degradation,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _metric_payload(subtype: str, value: float) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for name in PRODUCTION_METRICS_BY_SUBTYPE[subtype]:
        if name == "controlled_contact_entries":
            payload[name] = 4
        elif name == "controlled_contact_positive_entries":
            payload[name] = 2
        elif name == "controlled_contact_exact_equality":
            payload[name] = 1.0
        elif name in UNIT_INTERVAL_METRICS:
            payload[name] = 0.9
        elif name == "constraint_end_effector_rotation_deg":
            payload[name] = 5.0
        else:
            payload[name] = value
    return payload


def _record(
    key: str,
    *,
    seed: int = 7,
    value: float = 0.1,
    subtype: str = "path_2dpos",
) -> dict:
    contact_entries = 4 if "controlled_contact_entries" in PRODUCTION_METRICS_BY_SUBTYPE[subtype] else 0
    metrics = _metric_payload(subtype, value)
    return {
        "status": "ok",
        "case_key": key,
        "dataset_index": 1,
        "motion_id": "000001",
        "subtype": subtype,
        "text_regime": "withtext",
        "sample_seed": seed,
        "length": 30,
        "components": {"root_path_2dpos": [0, 1]},
        "model_mask_fraction": 0.1,
        "contact_control_entries": contact_entries,
        "text": "a person walks",
        "target_asset_path": "/tmp/frozen_target.npy",
        "target_asset_sha256": SHA_A,
        "target_tensor_sha256": SHA_A,
        "observed_motion_sha256": SHA_A,
        "motion_mask_sha256": SHA_A,
        "c_dir_sha256": SHA_A,
        "constraint_payload_sha256": SHA_A,
        "initial_continuous_noise_sha256": SHA_A,
        "initial_contact_noise_sha256": SHA_A,
        "initial_noise_sha256": SHA_A,
        "metrics": {
            "generated_raw": dict(metrics),
            "ground_truth": dict(metrics),
            "diagnostic_exact_clamp": dict(metrics),
        },
    }


def _protocol(checkpoint: str = "baseline") -> dict:
    return {
        "protocol_version": PRODUCTION_PROTOCOL_VERSION,
        "checkpoint_sha256": checkpoint,
        "case_plan_sha256": "plan",
        "weight_source": "ema",
    }


def _manifest(
    *,
    path: str,
    result: str = "result",
    contract: str = "contract",
    selected_weight: str = SHA_A,
    inference_state: str = "",
) -> dict:
    return {
        "protocol_manifest_path": path,
        "protocol_contract_sha256": contract,
        "ordered_case_identity_sha256": "cases",
        "ordered_result_content_sha256": result,
        "selected_weight_state_sha256": selected_weight,
        "inference_state_sha256": inference_state or selected_weight,
    }


def test_signed_degradation_respects_metric_direction() -> None:
    baseline = np.asarray([1.0, 2.0])
    candidate = np.asarray([1.1, 1.8])
    np.testing.assert_allclose(
        signed_degradation(baseline, candidate, "lower_is_better"), [0.1, -0.2]
    )
    np.testing.assert_allclose(
        signed_degradation(baseline, candidate, "higher_is_better"), [-0.1, 0.2]
    )


def test_allowed_degradation_is_continuous_at_zero_with_scale_floor() -> None:
    assert allowed_degradation(0.1, 0.01, 0.05, 0.02) == pytest.approx(0.005)
    assert allowed_degradation(10.0, 0.01, 0.05, 0.02) == pytest.approx(0.01)
    assert allowed_degradation(0.0, 0.01, 0.05, 0.02) == pytest.approx(0.001)
    assert allowed_degradation(1e-13, 0.01, 0.05, 0.02) == pytest.approx(0.001)


def test_bootstrap_is_deterministic_paired_and_nondegenerate() -> None:
    values = np.asarray([-0.1, 0.0, 0.1, 0.2])
    first = paired_bootstrap_mean_ci(values, resamples=1000, confidence=0.95, seed=123)
    second = paired_bootstrap_mean_ci(values, resamples=1000, confidence=0.95, seed=123)
    assert first == second
    assert first[0] == pytest.approx(0.05)
    assert first[1] < first[2]
    assert first[1] <= first[0] <= first[2]


def test_alignment_fails_closed_on_seed_or_case_changes() -> None:
    baseline = [_record("case_a")]
    assert len(align_records(baseline, [_record("case_a")])) == 1
    with pytest.raises(RuntimeError, match="identity changed"):
        align_records(baseline, [_record("case_a", seed=8)])
    with pytest.raises(RuntimeError, match="case-key mismatch"):
        align_records(baseline, [_record("case_b")])


def test_alignment_fails_closed_on_constraint_or_noise_changes() -> None:
    baseline = _record("case_a")
    for field in ("constraint_payload_sha256", "initial_noise_sha256"):
        candidate = json.loads(json.dumps(baseline))
        candidate[field] = SHA_B
        with pytest.raises(RuntimeError, match="identity changed"):
            align_records([baseline], [candidate])


def test_alignment_fails_closed_on_protocol_count_changes() -> None:
    baseline = _record("case_a", subtype="contact_only_sparse")
    candidate = json.loads(json.dumps(baseline))
    candidate["metrics"]["generated_raw"]["controlled_contact_entries"] = 3
    with pytest.raises(RuntimeError, match="count metric"):
        align_records([baseline], [candidate])


def test_comparator_synthetic_self_test_covers_both_directions() -> None:
    payload = run_self_test(resamples=1000, confidence=0.95, seed=3407)
    assert payload["status"] == "validated"
    assert payload["case_count"] == 4
    assert all(row["passed"] and row["nondegenerate_ci"] for row in payload["cases"])


def test_self_calibration_cannot_claim_candidate_evidence() -> None:
    record = _record("case_a")
    protocol = _protocol()
    manifest = _manifest(path="/same")
    gate, bootstrap = build_artifacts(
        baseline_protocol=protocol,
        baseline_records=[record],
        baseline_case_manifest=manifest,
        candidate_protocol=protocol,
        candidate_records=[json.loads(json.dumps(record))],
        candidate_case_manifest=dict(manifest),
        resamples=100,
        confidence=0.95,
        relative_tolerance=0.05,
        seed=3407,
        profile="smoke",
    )
    assert gate["status"] == "smoke_validated"
    assert bootstrap["comparison_kind"] == "baseline_self_calibration"
    assert bootstrap["candidate_evidence"] is False
    assert bootstrap["nonregression_decision"] == "not_applicable"
    assert "all_passed" not in bootstrap
    assert "passed" not in bootstrap["rows"][0]
    assert bootstrap["calibration_all_zero_delta"] is True


def test_candidate_protocol_drift_fails_closed() -> None:
    record = _record("case_a")
    with pytest.raises(RuntimeError, match="protocol contracts differ"):
        build_artifacts(
            baseline_protocol=_protocol("baseline"),
            baseline_records=[record],
            baseline_case_manifest=_manifest(path="/base", contract="contract-a"),
            candidate_protocol=_protocol("candidate"),
            candidate_records=[json.loads(json.dumps(record))],
            candidate_case_manifest=_manifest(
                path="/candidate", contract="contract-b", selected_weight=SHA_B
            ),
            resamples=100,
            confidence=0.95,
            relative_tolerance=0.05,
            seed=3407,
            profile="smoke",
        )


def test_candidate_failure_has_explicit_nonregression_decision() -> None:
    baseline = _record("case_a", value=0.1)
    candidate = _record("case_a", value=0.2)
    _, bootstrap = build_artifacts(
        baseline_protocol=_protocol("baseline"),
        baseline_records=[baseline],
        baseline_case_manifest=_manifest(path="/base"),
        candidate_protocol=_protocol("candidate"),
        candidate_records=[candidate],
        candidate_case_manifest=_manifest(
            path="/candidate", result="candidate-result", selected_weight=SHA_B
        ),
        resamples=100,
        confidence=0.95,
        relative_tolerance=0.05,
        seed=3407,
        profile="smoke",
    )
    assert bootstrap["candidate_evidence"] is True
    assert bootstrap["nonregression_decision"] == "fail"
    assert bootstrap["all_passed"] is False
    assert any(row["passed"] is False for row in bootstrap["rows"])


def test_same_weights_with_changed_inference_state_is_a_candidate() -> None:
    baseline = _record("case_a", value=0.1)
    candidate = _record("case_a", value=0.1)
    _, bootstrap = build_artifacts(
        baseline_protocol=_protocol("same-container"),
        baseline_records=[baseline],
        baseline_case_manifest=_manifest(path="/base", inference_state=SHA_A),
        candidate_protocol=_protocol("same-container"),
        candidate_records=[candidate],
        candidate_case_manifest=_manifest(
            path="/candidate",
            result="candidate-result",
            selected_weight=SHA_A,
            inference_state=SHA_B,
        ),
        resamples=100,
        confidence=0.95,
        relative_tolerance=0.05,
        seed=3407,
        profile="smoke",
    )
    assert bootstrap["comparison_kind"] == "candidate_vs_baseline"
    assert bootstrap["candidate_evidence"] is True


def test_production_profile_rejects_partial_coverage() -> None:
    record = _record("case_a")
    with pytest.raises(RuntimeError, match="Production comparator profile mismatch"):
        build_artifacts(
            baseline_protocol=_protocol(),
            baseline_records=[record],
            baseline_case_manifest=_manifest(path="/same"),
            candidate_protocol=_protocol(),
            candidate_records=[record],
            candidate_case_manifest=_manifest(path="/same"),
            resamples=10_000,
            confidence=0.95,
            relative_tolerance=0.05,
            seed=3407,
            profile="production",
        )


def test_raw_bit_exact_contact_metric_is_diagnostic_only() -> None:
    record = _record("case_a")
    record["metrics"]["generated_raw"]["controlled_contact_exact_equality"] = 0.1
    assert "controlled_contact_exact_equality" not in _metric_names([record])


@pytest.mark.parametrize(
    ("subtype", "metric", "bad_value", "message"),
    [
        ("path_2dpos", "constraint_root2d_acc", 1.1, "outside"),
        ("contact_only_sparse", "controlled_contact_brier", 1.1, "exceeds one"),
        (
            "feet_posrot",
            "constraint_end_effector_rotation_deg",
            181.0,
            "exceeds 180",
        ),
    ],
)
def test_metric_domain_validation_rejects_invalid_scores(
    subtype: str, metric: str, bad_value: float, message: str
) -> None:
    record = _record("case_a", subtype=subtype)
    record["metrics"]["generated_raw"][metric] = bad_value
    with pytest.raises(RuntimeError, match=message):
        _validate_metrics(record)


def test_metric_schema_and_contact_counts_fail_closed() -> None:
    record = _record("case_a")
    record["metrics"]["generated_raw"].clear()
    with pytest.raises(RuntimeError, match="Metric schema mismatch"):
        _validate_metrics(record)

    contact = _record("case_b", subtype="contact_only_sparse")
    contact["metrics"]["generated_raw"]["controlled_contact_positive_entries"] = 5
    with pytest.raises(RuntimeError, match="count mismatch"):
        _validate_metrics(contact)
