#!/usr/bin/env python
"""Build and apply the frozen HY273 control non-regression comparator.

The comparator uses only ``generated_raw`` metrics for learned-control gates.
Exact-clamped metrics are retained as diagnostics and can never substitute for
raw model adherence. Candidate and baseline records must have identical case
identity, seed, requested length, and constraint payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.evidence_io import atomic_write_json


FORMAT_GATE_MATRIX = "hy273_nonregression_gate_matrix_v1"
FORMAT_BOOTSTRAP = "hy273_control_paired_bootstrap_v1"
FORMAT_GATE_MATRIX_SMOKE = "hy273_nonregression_gate_matrix_smoke_v1"
FORMAT_BOOTSTRAP_SMOKE = "hy273_control_paired_bootstrap_smoke_v1"
FORMAT_SELF_TEST = "hy273_nonregression_comparator_self_test_v1"
FORMAT_CASE_MANIFEST = "hy273_control_case_metrics_manifest_v4"
FORMAT_ARTIFACT_INDEX = "hy273_nonregression_artifact_index_v1"
FORMAT_THRESHOLD_AUDIT = "hy273_nonregression_threshold_audit_v1"
COMPARATOR_CONTRACT = "direction_aware_paired_ci_v3"
PRODUCTION_PROTOCOL_VERSION = "hy273_hml3d_kimodo_constraints_v5_contact_evidence_v2"
CONTROL_SUMMARY_FORMAT = "hy273_kimodo_v5_contact_summary_v3"
CONTROL_ARTIFACT_INDEX_FORMAT = "hy273_control_evidence_artifact_index_v1"
CONTROL_OUTPUT_FORMAT = "hy273_control_case_output_v1"

PRODUCTION_CASE_COUNT = 8_084
PRODUCTION_DATASET_SIZE = 4_042
PRODUCTION_RESAMPLES = 10_000
PRODUCTION_CONFIDENCE = 0.95
PRODUCTION_SEED = 3_407
PRODUCTION_RELATIVE_TOLERANCE = 0.05
PRODUCTION_CASE_PLAN_SHA256 = "e1bde6eca329e7d7212b12a6847b055a091352e8325956e30604e1b3feabb73b"
PRODUCTION_SUBTYPES = (
    "path_2dpos",
    "path_2dposrot",
    "waypoint_2dpos",
    "waypoint_2dposrot",
    "inbetweening",
    "random",
    "feet_posrot",
    "hands_posrot",
    "hands_feet_posrot",
    "root_ee_hands_feet_posrot_fullbody",
    "root_ee_hands_posrot",
    "root_ee_hands_posrot_fullbody",
    "root_path_fullbody",
    "contact_only_sparse",
    "root_sparse_contact",
    "root_dense_contact",
    "endpoints_contact",
    "fullpose_contact",
    "mixed_contact",
)
PRODUCTION_TEXT_REGIMES = ("withtext", "notext")
COUNT_METRICS = {
    "controlled_contact_entries",
    "controlled_contact_positive_entries",
}

# direction, threshold category, absolute tolerance, continuous relative scale floor
METRIC_SPECS: dict[str, tuple[str, str, float, float]] = {
    "constraint_root2d_err": (
        "lower_is_better", "position_error_m", 0.005, 0.01
    ),
    "constraint_root2d_acc": ("higher_is_better", "unit_score", 0.01, 0.05),
    "constraint_end_effector": (
        "lower_is_better", "position_error_m", 0.005, 0.01
    ),
    "constraint_end_effector_rotation_deg": (
        "lower_is_better",
        "rotation_error_deg",
        0.5,
        1.0,
    ),
    "constraint_fullbody_keyframe": (
        "lower_is_better",
        "position_error_m",
        0.005,
        0.01,
    ),
    "controlled_contact_bce": (
        "lower_is_better", "contact_bce", 0.02, 0.01
    ),
    "controlled_contact_brier": (
        "lower_is_better", "contact_brier", 0.01, 0.005
    ),
    "controlled_contact_accuracy": (
        "higher_is_better", "unit_score", 0.01, 0.05
    ),
    "controlled_contact_f1": (
        "higher_is_better", "unit_score", 0.01, 0.05
    ),
    "fk_position_rotation_consistency_cm": (
        "lower_is_better",
        "fk_consistency_cm",
        0.10,
        0.10,
    ),
    "foot_contact_consistency": (
        "higher_is_better", "unit_score", 0.01, 0.05
    ),
    "foot_skate_from_height": (
        "lower_is_better", "foot_skate_mps", 0.02, 0.05
    ),
    "foot_skate_from_pred_contacts": (
        "lower_is_better",
        "foot_skate_mps",
        0.02,
        0.05,
    ),
    "foot_skate_max_vel": (
        "lower_is_better", "foot_skate_mps", 0.02, 0.05
    ),
    "foot_skate_ratio": ("lower_is_better", "unit_score", 0.02, 0.05),
}

# This value measures whether a raw probability is bit-exactly 0/1. It is retained
# for diagnostics but is not a learned-control gate; BCE/Brier/F1 carry that signal.
RAW_DIAGNOSTIC_ONLY_METRICS = {"controlled_contact_exact_equality"}
UNIT_INTERVAL_METRICS = {
    "constraint_root2d_acc",
    "controlled_contact_accuracy",
    "controlled_contact_f1",
    "controlled_contact_exact_equality",
    "foot_contact_consistency",
    "foot_skate_ratio",
}

QUALITY_METRICS = frozenset(
    {
        "foot_contact_consistency",
        "foot_skate_from_height",
        "foot_skate_from_pred_contacts",
        "foot_skate_max_vel",
        "foot_skate_ratio",
    }
)
ROOT_METRICS = frozenset({"constraint_root2d_acc", "constraint_root2d_err"})
ENDPOINT_METRICS = frozenset(
    {"constraint_end_effector", "constraint_end_effector_rotation_deg"}
)
FULLBODY_METRICS = frozenset({"constraint_fullbody_keyframe"})
CONTACT_METRICS = frozenset(
    {
        "controlled_contact_accuracy",
        "controlled_contact_bce",
        "controlled_contact_brier",
        "controlled_contact_entries",
        "controlled_contact_exact_equality",
        "controlled_contact_f1",
        "controlled_contact_positive_entries",
        "fk_position_rotation_consistency_cm",
    }
)


def _metrics(*groups: frozenset[str]) -> frozenset[str]:
    output: frozenset[str] = frozenset()
    for group in groups:
        output = output | group
    return output


PRODUCTION_METRICS_BY_SUBTYPE: dict[str, frozenset[str]] = {
    "path_2dpos": _metrics(ROOT_METRICS, QUALITY_METRICS),
    "path_2dposrot": _metrics(ROOT_METRICS, QUALITY_METRICS),
    "waypoint_2dpos": _metrics(ROOT_METRICS, QUALITY_METRICS),
    "waypoint_2dposrot": _metrics(ROOT_METRICS, QUALITY_METRICS),
    "inbetweening": _metrics(FULLBODY_METRICS, QUALITY_METRICS),
    "random": _metrics(FULLBODY_METRICS, QUALITY_METRICS),
    "feet_posrot": _metrics(ENDPOINT_METRICS, QUALITY_METRICS),
    "hands_posrot": _metrics(ENDPOINT_METRICS, QUALITY_METRICS),
    "hands_feet_posrot": _metrics(ENDPOINT_METRICS, QUALITY_METRICS),
    "root_ee_hands_feet_posrot_fullbody": _metrics(
        ROOT_METRICS, ENDPOINT_METRICS, FULLBODY_METRICS, QUALITY_METRICS
    ),
    "root_ee_hands_posrot": _metrics(
        ROOT_METRICS, ENDPOINT_METRICS, QUALITY_METRICS
    ),
    "root_ee_hands_posrot_fullbody": _metrics(
        ROOT_METRICS, ENDPOINT_METRICS, FULLBODY_METRICS, QUALITY_METRICS
    ),
    "root_path_fullbody": _metrics(ROOT_METRICS, FULLBODY_METRICS, QUALITY_METRICS),
    "contact_only_sparse": _metrics(CONTACT_METRICS, QUALITY_METRICS),
    "root_sparse_contact": _metrics(
        ROOT_METRICS, CONTACT_METRICS, QUALITY_METRICS
    ),
    "root_dense_contact": _metrics(
        ROOT_METRICS, CONTACT_METRICS, QUALITY_METRICS
    ),
    "endpoints_contact": _metrics(
        ENDPOINT_METRICS, CONTACT_METRICS, QUALITY_METRICS
    ),
    "fullpose_contact": _metrics(
        FULLBODY_METRICS, CONTACT_METRICS, QUALITY_METRICS
    ),
    "mixed_contact": _metrics(
        ROOT_METRICS,
        ENDPOINT_METRICS,
        FULLBODY_METRICS,
        CONTACT_METRICS,
        QUALITY_METRICS,
    ),
}

CASE_IDENTITY_FIELDS = (
    "case_key",
    "dataset_index",
    "motion_id",
    "subtype",
    "text_regime",
    "sample_seed",
    "length",
    "components",
    "model_mask_fraction",
    "contact_control_entries",
    "target_asset_path",
    "target_asset_sha256",
    "target_tensor_sha256",
    "observed_motion_sha256",
    "motion_mask_sha256",
    "c_dir_sha256",
    "constraint_payload_sha256",
    "initial_continuous_noise_sha256",
    "initial_contact_noise_sha256",
    "initial_noise_sha256",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    value = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256_value(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _case_identity(record: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in CASE_IDENTITY_FIELDS if name not in record]
    if missing:
        raise RuntimeError(
            f"Record {record.get('case_key', '<unknown>')} lacks identity fields {missing}"
        )
    identity = {name: record[name] for name in CASE_IDENTITY_FIELDS}
    identity.update(
        {
            "text_sha256": hashlib.sha256(
                str(record.get("text", "")).encode("utf-8")
            ).hexdigest(),
            "target_identity_sha256": canonical_sha(
                {
                    "path": record["target_asset_path"],
                    "asset_sha256": record["target_asset_sha256"],
                    "transformed_tensor_sha256": record["target_tensor_sha256"],
                }
            ),
        }
    )
    return identity


def _record_content_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_identity": _case_identity(record),
        "metrics": record["metrics"],
        "output_path": record.get("output_path"),
        "output_sha256": record.get("output_sha256"),
        "protocol_version": record.get("protocol_version"),
        "weight_source": record.get("weight_source"),
        "selected_weight_state_sha256": record.get(
            "selected_weight_state_sha256"
        ),
        "inference_state_sha256": record.get("inference_state_sha256"),
        "preflight_manifest_sha256": record.get("preflight_manifest_sha256"),
        "protocol_manifest_sha256": record.get("protocol_manifest_sha256"),
    }


def _load_preflight_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    locator = protocol.get("preflight_manifest")
    if not isinstance(locator, dict) or not locator.get("path") or not locator.get("sha256"):
        raise RuntimeError("Control protocol lacks a pinned preflight manifest")
    path = Path(locator["path"]).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != locator["sha256"]:
        raise RuntimeError("Control preflight manifest is missing or changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError("Control preflight did not pass")
    if payload.get("checkpoint", {}).get("sha256") != protocol.get("checkpoint_sha256"):
        raise RuntimeError("Control preflight/checkpoint identity mismatch")
    checkpoint_path = Path(str(protocol.get("checkpoint", ""))).expanduser().resolve()
    if (
        not checkpoint_path.is_file()
        or sha256_file(checkpoint_path) != protocol.get("checkpoint_sha256")
    ):
        raise RuntimeError("Pinned control checkpoint is missing or changed")
    if payload.get("plan", {}).get("case_plan_sha256") != protocol.get(
        "case_plan_sha256"
    ):
        raise RuntimeError("Control preflight/case-plan identity mismatch")
    expected_inference_state = {
        "checkpoint_kind": protocol.get("checkpoint_kind"),
        "weight_source": protocol.get("weight_source"),
        "selected_weight_state_sha256": protocol.get(
            "selected_weight_state_sha256"
        ),
        "normalizer_state_sha256": protocol.get("normalizer_state_sha256"),
        "model_config_sha256": protocol.get("model_config_sha256"),
        "model_asset_identity_sha256": protocol.get(
            "model_asset_identity_sha256"
        ),
    }
    if (
        payload.get("inference_state") != expected_inference_state
        or canonical_sha(expected_inference_state)
        != protocol.get("inference_state_sha256")
        or payload.get("inference_state_sha256")
        != protocol.get("inference_state_sha256")
    ):
        raise RuntimeError("Control preflight/inference-state identity mismatch")
    expected_sampling = {
        "ode_steps": protocol.get("ode_steps"),
        "text_cfg_scale": protocol.get("text_cfg_scale"),
        "control_cfg_scale": protocol.get("control_cfg_scale"),
        "contact_init": protocol.get("contact_init"),
        "contact_feedback": protocol.get("contact_feedback"),
        "cfg_apply_contacts": protocol.get("cfg_apply_contacts"),
        "primary_output": protocol.get("primary_output"),
        "max_sparse_keyframes": protocol.get("max_sparse_keyframes"),
        "initial_noise": protocol.get("initial_noise"),
    }
    if payload.get("sampling") != expected_sampling:
        raise RuntimeError("Control preflight/protocol sampling contract mismatch")
    code = payload.get("code", {})
    code_rows = code.get("files", [])
    if not isinstance(code_rows, list) or not code_rows:
        raise RuntimeError("Control preflight lacks full execution code identity")
    if code.get("sha256") != canonical_sha(code_rows):
        raise RuntimeError("Control preflight code identity digest is invalid")
    root = Path(__file__).resolve().parents[1]
    for row in code_rows:
        path = Path(str(row.get("path", "")))
        resolved = path if path.is_absolute() else root / path
        if (
            not resolved.is_file()
            or int(resolved.stat().st_size) != int(row.get("size", -1))
            or sha256_file(resolved) != str(row.get("sha256", ""))
        ):
            raise RuntimeError(f"Pinned control execution code changed: {resolved}")
    selected_weight = payload.get("selected_weight")
    if not isinstance(selected_weight, dict):
        raise RuntimeError("Control preflight lacks selected tensor-state identity")
    if (
        selected_weight.get("source") != protocol.get("weight_source")
        or selected_weight.get("state_dict_sha256")
        != protocol.get("selected_weight_state_sha256")
    ):
        raise RuntimeError("Control preflight/protocol selected-weight identity mismatch")
    environment = payload.get("environment")
    if not isinstance(environment, dict) or canonical_sha(environment) != locator.get(
        "environment_sha256"
    ):
        raise RuntimeError("Control preflight environment identity is invalid")
    if canonical_sha(payload.get("model_assets")) != locator.get(
        "model_asset_identity_sha256"
    ):
        raise RuntimeError("Control preflight model-asset identity is invalid")
    return {
        "format": payload.get("format"),
        "execution_code": code,
        "execution_code_sha256": code["sha256"],
        "dataset": {
            "split": payload.get("dataset", {}).get("split"),
            "caption_policy": payload.get("dataset", {}).get("caption_policy"),
            "dataset_size": payload.get("dataset", {}).get("dataset_size"),
            "ordered_records_sha256": payload.get("dataset", {}).get(
                "ordered_records_sha256"
            ),
            "split_sha256": payload.get("dataset", {}).get("split_sha256"),
        },
        "sampling": expected_sampling,
        "environment_sha256": locator["environment_sha256"],
    }


def protocol_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "protocol_version",
        "dataset_size",
        "dataset_split",
        "case_count",
        "case_plan_sha256",
        "seed",
        "subtypes",
        "text_regimes",
        "legacy_subtypes",
        "contact_subtypes",
        "ode_steps",
        "text_cfg_scale",
        "control_cfg_scale",
        "contact_init",
        "contact_feedback",
        "cfg_apply_contacts",
        "max_sparse_keyframes",
        "initial_noise",
        "primary_output",
        "weight_source",
        "profile",
        "num_shards",
    )
    missing = [name for name in fields if name not in protocol]
    if missing:
        raise RuntimeError(f"Control protocol lacks comparison fields: {missing}")
    contract = {name: protocol[name] for name in fields}
    contract["preflight_contract"] = _load_preflight_contract(protocol)
    return contract


def _validate_metrics(record: dict[str, Any]) -> None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"Missing metrics payload: {record.get('case_key')}")
    subtype = str(record.get("subtype", ""))
    expected_names = PRODUCTION_METRICS_BY_SUBTYPE.get(subtype)
    if expected_names is None:
        raise RuntimeError(f"Unknown control subtype in metrics: {subtype!r}")
    for pass_name in ("generated_raw", "ground_truth", "diagnostic_exact_clamp"):
        if not isinstance(metrics.get(pass_name), dict):
            raise RuntimeError(
                f"Missing {pass_name} metrics: {record.get('case_key')}"
            )
        actual_names = frozenset(metrics[pass_name])
        if actual_names != expected_names:
            raise RuntimeError(
                f"Metric schema mismatch for {record.get('case_key')}/{pass_name}: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
        for name, value in metrics[pass_name].items():
            if name in COUNT_METRICS:
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise RuntimeError(
                        f"Invalid count metric {name} in {record.get('case_key')}"
                    )
                continue
            if not _numeric(value):
                raise RuntimeError(
                    f"Non-finite metric {pass_name}/{name} in {record.get('case_key')}"
                )
            numeric = float(value)
            if name in UNIT_INTERVAL_METRICS and not 0.0 <= numeric <= 1.0 + 1e-7:
                raise RuntimeError(
                    f"Metric {pass_name}/{name} is outside [0,1] in {record.get('case_key')}"
                )
            if name not in UNIT_INTERVAL_METRICS and numeric < -1e-10:
                raise RuntimeError(
                    f"Metric {pass_name}/{name} is negative in {record.get('case_key')}"
                )
            if name == "controlled_contact_brier" and numeric > 1.0 + 1e-7:
                raise RuntimeError(
                    f"Brier score exceeds one in {record.get('case_key')}/{pass_name}"
                )
            if name == "constraint_end_effector_rotation_deg" and numeric > 180.0 + 1e-5:
                raise RuntimeError(
                    f"Rotation error exceeds 180 degrees in {record.get('case_key')}/{pass_name}"
                )
    contact_entries = int(record.get("contact_control_entries", 0))
    raw = metrics["generated_raw"]
    exact = metrics["diagnostic_exact_clamp"]
    if contact_entries > 0:
        for name in COUNT_METRICS:
            if name not in raw or name not in exact:
                raise RuntimeError(
                    f"Controlled-contact record lacks {name}: {record.get('case_key')}"
                )
        expected_positive = None
        for pass_name in ("generated_raw", "ground_truth", "diagnostic_exact_clamp"):
            payload = metrics[pass_name]
            entries = int(payload["controlled_contact_entries"])
            positive = int(payload["controlled_contact_positive_entries"])
            if entries != contact_entries or not 0 <= positive <= entries:
                raise RuntimeError(
                    f"Controlled-contact count mismatch: {record.get('case_key')}/{pass_name}"
                )
            if expected_positive is None:
                expected_positive = positive
            elif positive != expected_positive:
                raise RuntimeError(
                    f"Controlled-contact positive count changed by output pass: {record.get('case_key')}"
                )
        if abs(float(exact.get("controlled_contact_exact_equality", -1.0)) - 1.0) > 1e-7:
            raise RuntimeError(
                f"Exact clamp failed controlled-contact equality: {record.get('case_key')}"
            )
    elif any(name in metrics["generated_raw"] for name in COUNT_METRICS):
        raise RuntimeError(
            f"Uncontrolled record unexpectedly reports contact counts: {record.get('case_key')}"
        )
    for name in CASE_IDENTITY_FIELDS:
        if name.endswith("sha256") and not _sha256_value(record.get(name)):
            raise RuntimeError(f"Invalid identity digest {name}: {record.get('case_key')}")
    if not Path(str(record["target_asset_path"])).is_absolute():
        raise RuntimeError(f"Target asset path is not absolute: {record.get('case_key')}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_control_output(
    root: Path, record: dict[str, Any]
) -> dict[str, str]:
    case_key = str(record.get("case_key", ""))
    expected_path = (root / "case_outputs" / f"{case_key}.npz").resolve()
    path = Path(str(record.get("output_path", ""))).expanduser().resolve()
    claimed_sha = str(record.get("output_sha256", ""))
    if path != expected_path or not path.is_file():
        raise RuntimeError(f"Control output path is missing or unexpected: {case_key}")
    if not _sha256_value(claimed_sha) or sha256_file(path) != claimed_sha:
        raise RuntimeError(f"Control output SHA mismatch: {case_key}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "format",
            "case_key",
            "length",
            "generated_raw",
            "diagnostic_exact_clamp",
        }:
            raise RuntimeError(f"Unexpected control output schema: {case_key}")
        length = int(archive["length"].item())
        raw = archive["generated_raw"]
        exact = archive["diagnostic_exact_clamp"]
        if (
            str(archive["format"].item()) != CONTROL_OUTPUT_FORMAT
            or str(archive["case_key"].item()) != case_key
            or length != int(record.get("length", -1))
            or raw.shape != (length, 273)
            or exact.shape != (length, 273)
            or raw.dtype != np.float32
            or exact.dtype != np.float32
            or not np.isfinite(raw).all()
            or not np.isfinite(exact).all()
        ):
            raise RuntimeError(f"Invalid control output payload: {case_key}")
    return {"case_key": case_key, "path": str(path), "sha256": claimed_sha}


def load_evaluation_dir(
    directory: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = Path(directory).expanduser().resolve()
    protocol_path = root / "protocol_manifest.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing protocol manifest: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PRODUCTION_PROTOCOL_VERSION:
        raise RuntimeError("Comparator requires the frozen v5-contact evidence-v2 protocol")
    for field in (
        "selected_weight_state_sha256",
        "normalizer_state_sha256",
        "model_config_sha256",
        "model_asset_identity_sha256",
        "inference_state_sha256",
    ):
        if not _sha256_value(protocol.get(field)):
            raise RuntimeError(f"Control protocol lacks valid {field}")
    num_shards = int(protocol.get("num_shards", 0))
    expected = int(protocol.get("case_count", 0))
    if num_shards <= 0 or expected <= 0:
        raise RuntimeError("Evaluation protocol has invalid shard/case counts")
    contract = protocol_contract(protocol)
    contract_sha = canonical_sha(contract)
    preflight_locator = protocol["preflight_manifest"]
    preflight_path = Path(preflight_locator["path"]).expanduser().resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    plan_rows = preflight.get("plan", {}).get("rows")
    if not isinstance(plan_rows, list) or len(plan_rows) != expected:
        raise RuntimeError("Control preflight lacks the complete frozen case plan")
    planned_by_key = {str(row.get("case_key", "")): row for row in plan_rows}
    if len(planned_by_key) != expected or not all(planned_by_key):
        raise RuntimeError("Control preflight case plan is incomplete or duplicated")
    expected_shard_by_key = {
        str(row["case_key"]): position % num_shards
        for position, row in enumerate(plan_rows)
    }

    records: list[dict[str, Any]] = []
    shard_rows = []
    for shard_id in range(num_shards):
        path = root / "shards" / f"shard_{shard_id:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation shard: {path}")
        rows = _read_jsonl(path)
        for row in rows:
            if int(row.get("shard_id", -1)) != shard_id:
                raise RuntimeError(f"Record is stored in the wrong shard: {row.get('case_key')}")
            planned = planned_by_key.get(str(row.get("case_key", "")))
            if planned is None or any(
                row.get(name) != planned.get(name)
                for name in (
                    "dataset_index",
                    "subtype",
                    "text_regime",
                    "sample_seed",
                )
            ):
                raise RuntimeError(
                    f"Record differs from frozen control plan: {row.get('case_key')}"
                )
            if expected_shard_by_key[str(row["case_key"])] != shard_id:
                raise RuntimeError(
                    f"Frozen control case is in the wrong shard: {row.get('case_key')}"
                )
        records.extend(rows)
        shard_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "content_sha256": canonical_sha(
                    [_record_content_identity(row) for row in rows]
                ),
            }
        )
    keys = [str(record.get("case_key", "")) for record in records]
    if len(records) != expected or len(set(keys)) != expected or not all(keys):
        raise RuntimeError(
            f"Evaluation records are incomplete or duplicated: rows={len(records)} "
            f"unique={len(set(keys))} expected={expected}"
        )
    protocol_sha = sha256_file(protocol_path)
    target_hash_cache: dict[str, str] = {}
    preflight_sha = protocol["preflight_manifest"]["sha256"]
    output_evidence = []
    for record in records:
        if record.get("status") != "ok":
            raise RuntimeError(f"Non-ok evaluation record: {record.get('case_key')}")
        if record.get("protocol_manifest_sha256") != protocol_sha:
            raise RuntimeError(
                f"Protocol SHA mismatch in record {record.get('case_key')}"
            )
        if record.get("preflight_manifest_sha256") != preflight_sha:
            raise RuntimeError(f"Preflight SHA mismatch in record {record.get('case_key')}")
        if record.get("weight_source") != protocol["weight_source"]:
            raise RuntimeError(f"Weight source mismatch in record {record.get('case_key')}")
        if (
            record.get("selected_weight_state_sha256")
            != protocol["selected_weight_state_sha256"]
        ):
            raise RuntimeError(
                f"Selected-weight SHA mismatch in record {record.get('case_key')}"
            )
        if record.get("inference_state_sha256") != protocol["inference_state_sha256"]:
            raise RuntimeError(
                f"Inference-state SHA mismatch in record {record.get('case_key')}"
            )
        metrics = record.get("metrics", {})
        if not isinstance(metrics.get("generated_raw"), dict):
            raise RuntimeError(f"Missing generated_raw metrics: {record.get('case_key')}")
        _case_identity(record)
        _validate_metrics(record)
        target_path = str(record["target_asset_path"])
        if target_path not in target_hash_cache:
            path = Path(target_path)
            if not path.is_file():
                raise FileNotFoundError(f"Pinned control target is missing: {path}")
            target_hash_cache[target_path] = sha256_file(path)
        if target_hash_cache[target_path] != record["target_asset_sha256"]:
            raise RuntimeError(f"Control target asset changed: {target_path}")
        output_evidence.append(_validate_control_output(root, record))

    ordered_case_identities = [_case_identity(record) for record in records]
    ordered_result_identities = [_record_content_identity(record) for record in records]
    output_evidence.sort(key=lambda row: row["path"])
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing validated control aggregate: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary_shards = [
        {"shard_id": index, "rows": row["rows"], "sha256": row["sha256"]}
        for index, row in enumerate(shard_rows)
    ]
    if (
        summary.get("format") != CONTROL_SUMMARY_FORMAT
        or summary.get("status") != "validated"
        or summary.get("protocol") != protocol
        or summary.get("protocol_manifest_sha256") != protocol_sha
        or int(summary.get("case_count", -1)) != len(records)
        or summary.get("case_rows_sha256") != canonical_sha(records)
        or summary.get("shards") != expected_summary_shards
        or summary.get("case_outputs") != output_evidence
    ):
        raise RuntimeError("Control aggregate summary does not bind current evidence")

    artifact_index_path = root / "artifact_index.json"
    if not artifact_index_path.is_file():
        raise FileNotFoundError(
            f"Missing validated control artifact index: {artifact_index_path}"
        )
    artifact_index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
    expected_artifacts = {
        "preflight_manifest": protocol["preflight_manifest"],
        "protocol_manifest": {
            "path": str(protocol_path),
            "sha256": protocol_sha,
        },
        "summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "shards": expected_summary_shards,
        "case_outputs": output_evidence,
    }
    if (
        artifact_index.get("format") != CONTROL_ARTIFACT_INDEX_FORMAT
        or artifact_index.get("status") != "validated"
        or artifact_index.get("profile") != protocol.get("profile")
        or int(artifact_index.get("case_count", -1)) != len(records)
        or artifact_index.get("protocol_version") != protocol.get("protocol_version")
        or artifact_index.get("checkpoint_sha256")
        != protocol.get("checkpoint_sha256")
        or artifact_index.get("inference_state_sha256")
        != protocol.get("inference_state_sha256")
        or artifact_index.get("artifacts") != expected_artifacts
    ):
        raise RuntimeError("Control artifact index does not bind current evidence")

    case_manifest = {
        "format": FORMAT_CASE_MANIFEST,
        "protocol_version": protocol["protocol_version"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "case_plan_sha256": protocol["case_plan_sha256"],
        "protocol_manifest_path": str(protocol_path),
        "protocol_manifest_sha256": protocol_sha,
        "protocol_contract": contract,
        "protocol_contract_sha256": contract_sha,
        "weight_source": protocol["weight_source"],
        "selected_weight_state_sha256": protocol["selected_weight_state_sha256"],
        "inference_state_sha256": protocol["inference_state_sha256"],
        "record_count": len(records),
        "unique_case_key_count": len(set(keys)),
        "case_identity_fields": list(CASE_IDENTITY_FIELDS),
        "ordered_case_identity_sha256": canonical_sha(ordered_case_identities),
        "ordered_result_content_sha256": canonical_sha(ordered_result_identities),
        "ordered_case_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in keys).encode("utf-8")
        ).hexdigest(),
        "sorted_case_keys_sha256": hashlib.sha256(
            "".join(f"{key}\n" for key in sorted(keys)).encode("utf-8")
        ).hexdigest(),
        "shards": shard_rows,
        "aggregate_summary": {
            "path": str(summary_path),
            "sha256": sha256_file(summary_path),
        },
        "control_artifact_index": {
            "path": str(artifact_index_path),
            "sha256": sha256_file(artifact_index_path),
        },
    }
    return protocol, records, case_manifest


def align_records(
    baseline: Iterable[dict[str, Any]], candidate: Iterable[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    baseline_rows = list(baseline)
    candidate_rows = list(candidate)
    baseline_by_key = {row["case_key"]: row for row in baseline_rows}
    candidate_by_key = {row["case_key"]: row for row in candidate_rows}
    if len(baseline_by_key) != len(baseline_rows) or len(candidate_by_key) != len(
        candidate_rows
    ):
        raise RuntimeError("Duplicate case keys are not valid paired evidence")
    if set(baseline_by_key) != set(candidate_by_key):
        missing = sorted(set(baseline_by_key) - set(candidate_by_key))[:5]
        extra = sorted(set(candidate_by_key) - set(baseline_by_key))[:5]
        raise RuntimeError(f"Candidate case-key mismatch: missing={missing}, extra={extra}")
    aligned = []
    for key in sorted(baseline_by_key):
        base = baseline_by_key[key]
        cand = candidate_by_key[key]
        if _case_identity(base) != _case_identity(cand):
            raise RuntimeError(f"Candidate case identity changed for {key}")
        base_raw = base["metrics"]["generated_raw"]
        cand_raw = cand["metrics"]["generated_raw"]
        if set(base_raw) != set(cand_raw):
            raise RuntimeError(f"Candidate metric schema changed for {key}")
        contact_controlled = int(base["contact_control_entries"]) > 0
        for count_metric in COUNT_METRICS:
            if contact_controlled and (
                count_metric not in base_raw or count_metric not in cand_raw
            ):
                raise RuntimeError(
                    f"Controlled-contact record lacks count metric {count_metric}: {key}"
                )
            base_value = base_raw.get(count_metric, 0)
            cand_value = cand_raw.get(count_metric, 0)
            if base_value != cand_value:
                raise RuntimeError(
                    f"Protocol count metric {count_metric} changed for {key}: "
                    f"{base_value} -> {cand_value}"
                )
        aligned.append((base, cand))
    return aligned


def signed_degradation(
    baseline: np.ndarray, candidate: np.ndarray, direction: str
) -> np.ndarray:
    if direction == "lower_is_better":
        return candidate - baseline
    if direction == "higher_is_better":
        return baseline - candidate
    raise ValueError(f"Unknown metric direction: {direction}")


def allowed_degradation(
    baseline_mean: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    scale_floor: float,
) -> float:
    """Use continuous absolute-and-relative non-inferiority tolerance."""

    if absolute_tolerance < 0 or relative_tolerance < 0 or scale_floor <= 0:
        raise ValueError("Comparator tolerances and scale floor must be valid")
    relative = max(abs(float(baseline_mean)), float(scale_floor)) * float(
        relative_tolerance
    )
    return min(float(absolute_tolerance), relative)


def paired_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    resamples: int,
    confidence: float,
    seed: int,
    chunk_size: int = 2048,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a non-empty finite vector")
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap resamples/confidence")
    rng = np.random.default_rng(int(seed))
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha], method="linear")
    return float(values.mean()), float(low), float(high)


def _row_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        (str(seed) + "\0" + "\0".join(parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _metric_names(records: list[dict[str, Any]]) -> list[str]:
    names = set()
    for record in records:
        names.update(record["metrics"]["generated_raw"])
    unknown = sorted(
        names - set(METRIC_SPECS) - COUNT_METRICS - RAW_DIAGNOSTIC_ONLY_METRICS
    )
    if unknown:
        raise RuntimeError(f"Unclassified v5 generated_raw metrics: {unknown}")
    return sorted(names & set(METRIC_SPECS))


def _validate_profile(
    *,
    profile: str,
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
    resamples: int,
    confidence: float,
    relative_tolerance: float,
    seed: int,
) -> None:
    if profile == "smoke":
        if not records:
            raise RuntimeError("Smoke comparator requires at least one case")
        return
    if profile != "production":
        raise ValueError(f"Unknown comparator profile: {profile}")
    expected_cells = {
        (regime, subtype)
        for regime in PRODUCTION_TEXT_REGIMES
        for subtype in PRODUCTION_SUBTYPES
    }
    actual_cells = {
        (str(record["text_regime"]), str(record["subtype"])) for record in records
    }
    counts = {
        cell: sum(
            1
            for record in records
            if (str(record["text_regime"]), str(record["subtype"])) == cell
        )
        for cell in actual_cells
    }
    expected_count_by_subtype = {
        subtype: (213 if index < PRODUCTION_DATASET_SIZE % len(PRODUCTION_SUBTYPES) else 212)
        for index, subtype in enumerate(PRODUCTION_SUBTYPES)
    }
    failures = []
    if len(records) != PRODUCTION_CASE_COUNT:
        failures.append(f"case_count={len(records)}")
    if int(protocol.get("dataset_size", -1)) != PRODUCTION_DATASET_SIZE:
        failures.append(f"dataset_size={protocol.get('dataset_size')}")
    if tuple(protocol.get("subtypes", ())) != PRODUCTION_SUBTYPES:
        failures.append("subtypes")
    if tuple(protocol.get("text_regimes", ())) != PRODUCTION_TEXT_REGIMES:
        failures.append("text_regimes")
    if protocol.get("protocol_version") != PRODUCTION_PROTOCOL_VERSION:
        failures.append(f"protocol_version={protocol.get('protocol_version')}")
    if protocol.get("case_plan_sha256") != PRODUCTION_CASE_PLAN_SHA256:
        failures.append(f"case_plan_sha256={protocol.get('case_plan_sha256')}")
    expected_protocol_fields = {
        "profile": "production",
        "dataset_split": "test",
        "num_shards": 8,
        "ode_steps": 32,
        "text_cfg_scale": 2.0,
        "control_cfg_scale": 2.0,
        "seed": PRODUCTION_SEED,
        "max_sparse_keyframes": 20,
        "weight_source": "ema",
        "primary_output": "raw_pre_exact_clamp",
    }
    for name, expected_value in expected_protocol_fields.items():
        if protocol.get(name) != expected_value:
            failures.append(f"{name}={protocol.get(name)!r}")
    if actual_cells != expected_cells:
        failures.append("cell_coverage")
    expected_counts = {
        (regime, subtype): expected_count_by_subtype[subtype]
        for regime in PRODUCTION_TEXT_REGIMES
        for subtype in PRODUCTION_SUBTYPES
    }
    if counts != expected_counts:
        failures.append("exact_cell_counts")
    for record in records:
        expected_metrics = PRODUCTION_METRICS_BY_SUBTYPE[str(record["subtype"])]
        for pass_name in ("generated_raw", "ground_truth", "diagnostic_exact_clamp"):
            if frozenset(record["metrics"][pass_name]) != expected_metrics:
                failures.append("metric_schema")
                break
        if failures and failures[-1] == "metric_schema":
            break
    if resamples != PRODUCTION_RESAMPLES:
        failures.append(f"resamples={resamples}")
    if not math.isclose(confidence, PRODUCTION_CONFIDENCE, abs_tol=1e-12):
        failures.append(f"confidence={confidence}")
    if not math.isclose(
        relative_tolerance, PRODUCTION_RELATIVE_TOLERANCE, abs_tol=1e-12
    ):
        failures.append(f"relative_tolerance={relative_tolerance}")
    if seed != PRODUCTION_SEED:
        failures.append(f"seed={seed}")
    if failures:
        raise RuntimeError(
            "Production comparator profile mismatch: " + ", ".join(failures)
        )


def _comparison_kind(
    baseline_protocol: dict[str, Any],
    baseline_case_manifest: dict[str, Any],
    candidate_protocol: dict[str, Any],
    candidate_case_manifest: dict[str, Any],
) -> str:
    same_inference_state = (
        baseline_case_manifest["inference_state_sha256"]
        == candidate_case_manifest["inference_state_sha256"]
    )
    same_locator = (
        baseline_case_manifest["protocol_manifest_path"]
        == candidate_case_manifest["protocol_manifest_path"]
    )
    if same_inference_state and same_locator:
        if (
            baseline_case_manifest["ordered_result_content_sha256"]
            != candidate_case_manifest["ordered_result_content_sha256"]
        ):
            raise RuntimeError("A self-calibration locator has inconsistent results")
        return "baseline_self_calibration"
    if same_inference_state:
        return "reproducibility_check"
    return "candidate_vs_baseline"


def build_artifacts(
    *,
    baseline_protocol: dict[str, Any],
    baseline_records: list[dict[str, Any]],
    baseline_case_manifest: dict[str, Any],
    candidate_protocol: dict[str, Any],
    candidate_records: list[dict[str, Any]],
    candidate_case_manifest: dict[str, Any],
    resamples: int,
    confidence: float,
    relative_tolerance: float,
    seed: int,
    profile: str = "smoke",
    frozen_gate_matrix: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_profile(
        profile=profile,
        protocol=baseline_protocol,
        records=baseline_records,
        resamples=resamples,
        confidence=confidence,
        relative_tolerance=relative_tolerance,
        seed=seed,
    )
    _validate_profile(
        profile=profile,
        protocol=candidate_protocol,
        records=candidate_records,
        resamples=resamples,
        confidence=confidence,
        relative_tolerance=relative_tolerance,
        seed=seed,
    )
    baseline_contract_sha = baseline_case_manifest["protocol_contract_sha256"]
    candidate_contract_sha = candidate_case_manifest["protocol_contract_sha256"]
    if baseline_contract_sha != candidate_contract_sha:
        raise RuntimeError("Candidate and baseline protocol contracts differ")
    if (
        baseline_case_manifest["ordered_case_identity_sha256"]
        != candidate_case_manifest["ordered_case_identity_sha256"]
    ):
        raise RuntimeError("Candidate and baseline ordered case identities differ")
    comparison_kind = _comparison_kind(
        baseline_protocol,
        baseline_case_manifest,
        candidate_protocol,
        candidate_case_manifest,
    )
    if (
        profile == "production"
        and comparison_kind != "baseline_self_calibration"
        and frozen_gate_matrix is None
    ):
        raise RuntimeError(
            "Production candidate/reproducibility comparisons require a frozen gate matrix"
        )
    aligned = align_records(baseline_records, candidate_records)
    cells: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for base, cand in aligned:
        key = (str(base["text_regime"]), str(base["subtype"]))
        cells.setdefault(key, []).append((base, cand))

    gate_rows = []
    bootstrap_rows = []
    for (regime, subtype), pairs in sorted(cells.items()):
        metric_names = _metric_names([base for base, _ in pairs])
        candidate_names = _metric_names([cand for _, cand in pairs])
        if metric_names != candidate_names:
            raise RuntimeError(f"Metric schema changed for {regime}/{subtype}")
        for metric in metric_names:
            baseline_values = np.asarray(
                [base["metrics"]["generated_raw"].get(metric) for base, _ in pairs],
                dtype=np.float64,
            )
            candidate_values = np.asarray(
                [cand["metrics"]["generated_raw"].get(metric) for _, cand in pairs],
                dtype=np.float64,
            )
            if not np.isfinite(baseline_values).all() or not np.isfinite(
                candidate_values
            ).all():
                raise RuntimeError(f"Missing/non-finite metric {regime}/{subtype}/{metric}")
            direction, category, absolute, scale_floor = METRIC_SPECS[metric]
            baseline_mean = float(baseline_values.mean())
            tolerance = allowed_degradation(
                baseline_mean, absolute, relative_tolerance, scale_floor
            )
            degradation = signed_degradation(
                baseline_values, candidate_values, direction
            )
            row_seed = _row_seed(seed, regime, subtype, metric)
            mean, ci_low, ci_high = paired_bootstrap_mean_ci(
                degradation,
                resamples=resamples,
                confidence=confidence,
                seed=row_seed,
            )
            common = {
                "text_regime": regime,
                "subtype": subtype,
                "metric": metric,
                "case_count": len(pairs),
                "direction": direction,
                "threshold_category": category,
                "baseline_mean": baseline_mean,
                "absolute_tolerance": float(absolute),
                "relative_tolerance": float(relative_tolerance),
                "relative_scale_floor": float(scale_floor),
                "allowed_degradation": tolerance,
                "tolerance_rule": "min_absolute_relative_with_scale_floor_v2",
                "gate_output": "generated_raw",
            }
            gate_rows.append(common)
            row = {
                **common,
                "candidate_mean": float(candidate_values.mean()),
                "signed_degradation_mean": mean,
                "signed_degradation_ci_low": ci_low,
                "signed_degradation_ci_high": ci_high,
                "bootstrap_seed": row_seed,
            }
            within_tolerance = bool(mean <= tolerance and ci_high <= tolerance)
            if comparison_kind == "candidate_vs_baseline":
                row["passed"] = within_tolerance
            elif comparison_kind == "reproducibility_check":
                row["reproducibility_within_tolerance"] = within_tolerance
            else:
                row["calibration_zero_delta"] = bool(
                    mean == 0.0 and ci_low == 0.0 and ci_high == 0.0
                )
            bootstrap_rows.append(row)

    expected_gate_rows = sum(
        len(metrics & set(METRIC_SPECS))
        for metrics in PRODUCTION_METRICS_BY_SUBTYPE.values()
    ) * len(PRODUCTION_TEXT_REGIMES)
    if profile == "production" and (
        len(gate_rows) != expected_gate_rows
        or len(bootstrap_rows) != expected_gate_rows
    ):
        raise RuntimeError(
            f"Production metric rows are incomplete: {len(gate_rows)}/{expected_gate_rows}"
        )

    protocol_identity = {
        "protocol_version": baseline_protocol["protocol_version"],
        "checkpoint_sha256": baseline_protocol["checkpoint_sha256"],
        "case_plan_sha256": baseline_protocol["case_plan_sha256"],
        "case_identity_sha256": baseline_case_manifest[
            "ordered_case_identity_sha256"
        ],
        "protocol_contract_sha256": baseline_contract_sha,
        "weight_source": baseline_protocol["weight_source"],
        "selected_weight_state_sha256": baseline_case_manifest[
            "selected_weight_state_sha256"
        ],
        "inference_state_sha256": baseline_case_manifest[
            "inference_state_sha256"
        ],
    }
    generated_gate = {
        "format": (
            FORMAT_GATE_MATRIX if profile == "production" else FORMAT_GATE_MATRIX_SMOKE
        ),
        "schema_version": 2,
        "status": "validated" if profile == "production" else "smoke_validated",
        "profile": profile,
        "comparator_contract": COMPARATOR_CONTRACT,
        "protocol_identity": protocol_identity,
        "case_count": len(baseline_records),
        "row_count": len(gate_rows),
        "relative_tolerance": float(relative_tolerance),
        "tolerance_policy_sha256": canonical_sha(METRIC_SPECS),
        "metric_schema_sha256": canonical_sha(
            [
                {
                    "text_regime": row["text_regime"],
                    "subtype": row["subtype"],
                    "metric": row["metric"],
                    "case_count": row["case_count"],
                }
                for row in gate_rows
            ]
        ),
        "primary_output": "generated_raw",
        "exact_clamp_policy": "diagnostic_only_never_substitutes_for_generated_raw",
        "raw_exact_equality_policy": "diagnostic_only_not_a_learned_control_gate",
        "rows": gate_rows,
    }
    if frozen_gate_matrix is not None:
        if frozen_gate_matrix != generated_gate:
            raise RuntimeError(
                "Frozen gate matrix differs from the current baseline metrics/policy"
            )
        gate = frozen_gate_matrix
    else:
        gate = generated_gate
    candidate_evidence = comparison_kind == "candidate_vs_baseline"
    if candidate_evidence:
        decision = (
            "pass" if all(bool(row["passed"]) for row in bootstrap_rows) else "fail"
        )
        evidence_scope = "candidate_nonregression"
    elif comparison_kind == "reproducibility_check":
        decision = "not_applicable"
        evidence_scope = "same_checkpoint_reproducibility_only"
    else:
        decision = "not_applicable"
        evidence_scope = "comparator_calibration_only"
    bootstrap = {
        "format": (
            FORMAT_BOOTSTRAP if profile == "production" else FORMAT_BOOTSTRAP_SMOKE
        ),
        "schema_version": 2,
        "status": "validated" if profile == "production" else "smoke_validated",
        "profile": profile,
        "comparator_contract": COMPARATOR_CONTRACT,
        "comparison_kind": comparison_kind,
        "evidence_scope": evidence_scope,
        "candidate_evidence": candidate_evidence,
        "nonregression_decision": decision,
        "protocol_identity": protocol_identity,
        "candidate_identity": {
            "checkpoint_sha256": candidate_protocol["checkpoint_sha256"],
            "case_plan_sha256": candidate_protocol["case_plan_sha256"],
            "case_identity_sha256": candidate_case_manifest[
                "ordered_case_identity_sha256"
            ],
            "protocol_contract_sha256": candidate_contract_sha,
            "weight_source": candidate_protocol["weight_source"],
            "selected_weight_state_sha256": candidate_case_manifest[
                "selected_weight_state_sha256"
            ],
            "inference_state_sha256": candidate_case_manifest[
                "inference_state_sha256"
            ],
        },
        "case_count": len(baseline_records),
        "row_count": len(bootstrap_rows),
        "resamples": int(resamples),
        "confidence": float(confidence),
        "ci_method": "paired_percentile_two_sided",
        "noninferiority_bound": "upper_97.5_percentile_for_confidence_0.95",
        "seed": int(seed),
        "rows": bootstrap_rows,
    }
    if candidate_evidence:
        bootstrap["all_passed"] = decision == "pass"
    elif comparison_kind == "reproducibility_check":
        bootstrap["reproducibility_all_within_tolerance"] = all(
            bool(row["reproducibility_within_tolerance"])
            for row in bootstrap_rows
        )
    else:
        bootstrap["calibration_all_zero_delta"] = all(
            bool(row["calibration_zero_delta"]) for row in bootstrap_rows
        )
    return gate, bootstrap


def run_self_test(*, resamples: int, confidence: float, seed: int) -> dict[str, Any]:
    cases = []
    specifications = (
        (
            "lower_small_pass",
            "lower_is_better",
            [1.0] * 8,
            [1.002, 1.003, 1.004, 1.005, 1.006, 1.004, 1.003, 1.005],
            0.01,
            True,
        ),
        (
            "lower_large_fail",
            "lower_is_better",
            [1.0] * 8,
            [1.015, 1.018, 1.020, 1.022, 1.025, 1.019, 1.021, 1.024],
            0.01,
            False,
        ),
        (
            "higher_small_pass",
            "higher_is_better",
            [0.8] * 8,
            [0.798, 0.797, 0.796, 0.795, 0.794, 0.796, 0.797, 0.795],
            0.01,
            True,
        ),
        (
            "higher_large_fail",
            "higher_is_better",
            [0.8] * 8,
            [0.785, 0.782, 0.780, 0.778, 0.775, 0.781, 0.779, 0.776],
            0.01,
            False,
        ),
    )
    for name, direction, base, cand, tolerance, expected in specifications:
        degradation = signed_degradation(
            np.asarray(base, dtype=np.float64),
            np.asarray(cand, dtype=np.float64),
            direction,
        )
        case_seed = _row_seed(seed, "self_test", name)
        mean, low, high = paired_bootstrap_mean_ci(
            degradation,
            resamples=resamples,
            confidence=confidence,
            seed=case_seed,
        )
        actual = bool(mean <= tolerance and high <= tolerance)
        cases.append(
            {
                "name": name,
                "direction": direction,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "tolerance": tolerance,
                "expected_pass": expected,
                "actual_pass": actual,
                "passed": actual == expected,
                "nondegenerate_ci": bool(high > low),
            }
        )
    return {
        "format": FORMAT_SELF_TEST,
        "schema_version": 1,
        "status": (
            "validated"
            if all(case["passed"] and case["nondegenerate_ci"] for case in cases)
            else "failed"
        ),
        "comparator_contract": COMPARATOR_CONTRACT,
        "case_count": len(cases),
        "row_count": len(cases),
        "resamples": int(resamples),
        "confidence": float(confidence),
        "seed": int(seed),
        "cases": cases,
    }


def threshold_audit(relative_tolerance: float) -> dict[str, Any]:
    rationale = {
        "position_error_m": "5 mm absolute cap; also no more than 5% of baseline with a 1 cm scale floor.",
        "rotation_error_deg": "0.5 degree absolute cap; also no more than 5% with a 1 degree floor.",
        "unit_score": "One percentage-point absolute cap; also no more than 5% with a 0.05 score floor.",
        "contact_bce": "0.02 BCE cap with a 0.01 numerical scale floor.",
        "contact_brier": "0.01 probability-MSE cap with a 0.005 numerical scale floor.",
        "fk_consistency_cm": "0.1 cm absolute FK consistency cap.",
        "foot_skate_mps": "0.02 m/s absolute foot-skate cap with a 0.05 m/s scale floor.",
    }
    rows = []
    for metric, (direction, category, absolute, floor) in sorted(METRIC_SPECS.items()):
        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "threshold_category": category,
                "absolute_tolerance": absolute,
                "relative_tolerance": float(relative_tolerance),
                "relative_scale_floor": floor,
                "combination_rule": "min(absolute, relative*max(abs(baseline_mean), scale_floor))",
                "rationale": rationale[category],
            }
        )
    return {
        "format": FORMAT_THRESHOLD_AUDIT,
        "schema_version": 1,
        "status": "validated",
        "scope": "engineering_noninferiority_gate_not_publication_significance_claim",
        "policy_sha256": canonical_sha(METRIC_SPECS),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_eval_dir", required=True)
    parser.add_argument(
        "--candidate_eval_dir",
        default="",
        help="Defaults to baseline for comparator calibration before training.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile", choices=["production", "smoke"], default="smoke")
    parser.add_argument("--frozen_gate_matrix", default="")
    parser.add_argument("--frozen_gate_matrix_sha256", default="")
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--relative_tolerance", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("confidence must be in (0,1)")
    if not 0.0 <= args.relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be in [0,1)")
    baseline_protocol, baseline_records, baseline_manifest = load_evaluation_dir(
        args.baseline_eval_dir
    )
    candidate_dir = args.candidate_eval_dir or args.baseline_eval_dir
    candidate_protocol, candidate_records, candidate_manifest = load_evaluation_dir(
        candidate_dir
    )
    frozen_gate = None
    if args.frozen_gate_matrix:
        frozen_path = Path(args.frozen_gate_matrix).expanduser().resolve()
        frozen_sha = sha256_file(frozen_path)
        if (
            not args.frozen_gate_matrix_sha256
            or frozen_sha != args.frozen_gate_matrix_sha256.lower()
        ):
            raise RuntimeError("Frozen gate matrix SHA256 is missing or mismatched")
        frozen_gate = json.loads(frozen_path.read_text(encoding="utf-8"))
    gate, bootstrap = build_artifacts(
        baseline_protocol=baseline_protocol,
        baseline_records=baseline_records,
        baseline_case_manifest=baseline_manifest,
        candidate_protocol=candidate_protocol,
        candidate_records=candidate_records,
        candidate_case_manifest=candidate_manifest,
        resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        relative_tolerance=args.relative_tolerance,
        seed=args.seed,
        profile=args.profile,
        frozen_gate_matrix=frozen_gate,
    )
    self_test = run_self_test(
        resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    if self_test["status"] != "validated":
        raise RuntimeError("Comparator self-test failed")
    threshold_report = threshold_audit(args.relative_tolerance)

    output = Path(args.output_dir).expanduser().resolve()
    paths = {
        "baseline_case_manifest": output / "baseline_case_manifest.json",
        "candidate_case_manifest": output / "candidate_case_manifest.json",
        "gate_matrix": output / "gate_matrix.json",
        "paired_bootstrap": output / "paired_bootstrap.json",
        "comparator_self_test": output / "comparator_self_test.json",
        "threshold_audit": output / "threshold_audit.json",
    }
    _atomic_json(paths["baseline_case_manifest"], baseline_manifest)
    _atomic_json(paths["candidate_case_manifest"], candidate_manifest)
    _atomic_json(paths["gate_matrix"], gate)
    _atomic_json(paths["paired_bootstrap"], bootstrap)
    _atomic_json(paths["comparator_self_test"], self_test)
    _atomic_json(paths["threshold_audit"], threshold_report)
    comparator_code = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    artifact_index = {
        "format": FORMAT_ARTIFACT_INDEX,
        "schema_version": 1,
        "status": "validated",
        "profile": args.profile,
        "comparison_kind": bootstrap["comparison_kind"],
        "candidate_evidence": bootstrap["candidate_evidence"],
        "nonregression_decision": bootstrap["nonregression_decision"],
        "comparator_contract": COMPARATOR_CONTRACT,
        "comparator_code": comparator_code,
        "baseline": {
            "eval_dir": str(Path(args.baseline_eval_dir).expanduser().resolve()),
            "checkpoint_sha256": baseline_protocol["checkpoint_sha256"],
            "selected_weight_state_sha256": baseline_manifest[
                "selected_weight_state_sha256"
            ],
            "inference_state_sha256": baseline_manifest[
                "inference_state_sha256"
            ],
            "ordered_case_identity_sha256": baseline_manifest[
                "ordered_case_identity_sha256"
            ],
            "ordered_result_content_sha256": baseline_manifest[
                "ordered_result_content_sha256"
            ],
        },
        "candidate": {
            "eval_dir": str(Path(candidate_dir).expanduser().resolve()),
            "checkpoint_sha256": candidate_protocol["checkpoint_sha256"],
            "selected_weight_state_sha256": candidate_manifest[
                "selected_weight_state_sha256"
            ],
            "inference_state_sha256": candidate_manifest[
                "inference_state_sha256"
            ],
            "ordered_case_identity_sha256": candidate_manifest[
                "ordered_case_identity_sha256"
            ],
            "ordered_result_content_sha256": candidate_manifest[
                "ordered_result_content_sha256"
            ],
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    artifact_index_path = output / "artifact_index.json"
    _atomic_json(artifact_index_path, artifact_index)
    result = {
        "artifact_validated": True,
        "profile": args.profile,
        "comparison_kind": bootstrap["comparison_kind"],
        "candidate_evidence": bootstrap["candidate_evidence"],
        "nonregression_decision": bootstrap["nonregression_decision"],
        "case_count": len(baseline_records),
        "gate_rows": gate["row_count"],
        "comparator_code": comparator_code,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "artifact_index": {
            "path": str(artifact_index_path),
            "sha256": sha256_file(artifact_index_path),
        },
    }
    print(json.dumps(result, sort_keys=True))
    if (
        bootstrap["candidate_evidence"]
        and bootstrap["nonregression_decision"] != "pass"
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
