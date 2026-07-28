#!/usr/bin/env python
"""Materialize matched control-evaluation cases for the Kimodo gallery renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_hy273_kimodo_v5_contact import (  # noqa: E402
    ContactCase,
    V5_CONTACT_SUBTYPES,
    _dataset,
    _load_case_output,
    _prepare_case,
    _validate_case_evidence,
)


IDENTITY_FIELDS = (
    "dataset_index",
    "subtype",
    "text_regime",
    "sample_seed",
    "length",
    "motion_id",
    "text",
    "constraint_payload_sha256",
    "motion_mask_sha256",
    "observed_motion_sha256",
    "c_dir_sha256",
    "initial_continuous_noise_sha256",
    "initial_contact_noise_sha256",
    "initial_noise_sha256",
    "target_tensor_sha256",
)


def load_records(eval_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((eval_dir / "shards").glob("shard_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = str(record["case_key"])
                if key in records:
                    raise RuntimeError(f"Duplicate case key: {key}")
                records[key] = record
    if not records:
        raise RuntimeError(f"No evaluation records found in {eval_dir}")
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def metric_masks(constraint: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = constraint.base if hasattr(constraint, "base") else constraint
    return (
        base.root_metric_frames.cpu().numpy(),
        base.fullbody_metric_frames.cpu().numpy(),
        base.endpoint_position_metric_mask.cpu().numpy(),
        base.endpoint_rotation_metric_mask.cpu().numpy(),
    )


def materialize_variant(
    *,
    eval_dir: Path,
    record: dict[str, Any],
    prepared: dict[str, Any],
    output_dir: Path,
    selection_index: int,
    variant: str,
    variant_label: str,
    checkpoint_label: str,
) -> dict[str, Any]:
    case = ContactCase(
        dataset_index=int(record["dataset_index"]),
        subtype=str(record["subtype"]),
        text_regime=str(record["text_regime"]),
        sample_seed=int(record["sample_seed"]),
    )
    raw, exact = _load_case_output(record, case, prepared, eval_dir)
    constraint = prepared["constraint_cpu"]
    root_frames, fullbody_frames, endpoint_pos, endpoint_rot = metric_masks(constraint)
    case_dir = output_dir / "cases" / f"{selection_index:02d}_{variant}_{case.key}"
    case_dir.mkdir(parents=True, exist_ok=False)
    arrays = {
        "generated_raw": raw.numpy(),
        "generated_exact_clamped": exact.numpy(),
        "target": prepared["target_cpu"].numpy(),
        "observed": constraint.observed_motion.cpu().numpy(),
        "mask": constraint.motion_mask.cpu().numpy(),
        "root_metric_frames": root_frames,
        "fullbody_metric_frames": fullbody_frames,
        "endpoint_position_metric_mask": endpoint_pos,
        "endpoint_rotation_metric_mask": endpoint_rot,
    }
    for name, value in arrays.items():
        np.save(case_dir / f"{name}.npy", value)

    text = str(record["text"]).strip() or "<no text>"
    protocol = json.loads((eval_dir / "protocol_manifest.json").read_text())
    metadata = {
        "format": "hy273_matched_control_pair_gallery_case_v1",
        "case_key": case.key,
        "selection_index": selection_index,
        "target_quantile": 0.11 if variant == "r11" else 0.12,
        "composite_percentile": 0.5,
        "metric_percentiles": {},
        "subtype": case.subtype,
        "family": variant_label,
        "variant": variant,
        "variant_label": variant_label,
        "checkpoint_label": checkpoint_label,
        "dataset_index": case.dataset_index,
        "motion_id": str(record["motion_id"]),
        "length": int(record["length"]),
        "caption": f"[{variant_label} / {case.text_regime}] {text}",
        "sample_seed": case.sample_seed,
        "constraint_components": constraint.components,
        "formal_metrics": record["metrics"]["generated_raw"],
        "regenerated_metrics": record["metrics"]["generated_raw"],
        "diagnostic_exact_metrics": record["metrics"]["diagnostic_exact_clamp"],
        "protocol": {
            "num_steps": int(protocol["ode_steps"]),
            "cfg_scale_text": float(protocol["text_cfg_scale"]),
            "cfg_scale_control": float(protocol["control_cfg_scale"]),
        },
    }
    write_json(case_dir / "metadata.json", metadata)
    return {
        "selection_index": selection_index,
        "case_key": case.key,
        "variant": variant,
        "case_dir": str(case_dir.relative_to(output_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_eval_dir", required=True)
    parser.add_argument("--candidate_eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--case_keys", required=True)
    parser.add_argument("--baseline_label", default="R11 B1")
    parser.add_argument("--candidate_label", default="R12 root-mask B1")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_eval_dir).expanduser().resolve()
    candidate_dir = Path(args.candidate_eval_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    case_keys = [item.strip() for item in args.case_keys.split(",") if item.strip()]
    if not case_keys or len(case_keys) != len(set(case_keys)):
        raise ValueError("case_keys must be a non-empty unique comma-separated list")

    baseline = load_records(baseline_dir)
    candidate = load_records(candidate_dir)
    missing = [key for key in case_keys if key not in baseline or key not in candidate]
    if missing:
        raise KeyError(f"Missing case keys: {missing}")

    baseline_protocol = json.loads((baseline_dir / "protocol_manifest.json").read_text())
    candidate_protocol = json.loads((candidate_dir / "protocol_manifest.json").read_text())
    sampling_fields = (
        "case_plan_sha256",
        "seed",
        "ode_steps",
        "text_cfg_scale",
        "control_cfg_scale",
        "contact_init",
        "contact_feedback",
        "cfg_apply_contacts",
        "max_sparse_keyframes",
        "initial_noise",
    )
    changed = [
        name
        for name in sampling_fields
        if baseline_protocol.get(name) != candidate_protocol.get(name)
    ]
    if changed:
        raise RuntimeError(f"Evaluation protocols are not matched: {changed}")

    candidate_preflight = json.loads(
        (candidate_dir / "preflight_manifest.json").read_text()
    )
    dataset_info = candidate_preflight["dataset"]
    dataset_args = argparse.Namespace(
        data_root=str(dataset_info["data_root"]),
        text_root=str(dataset_info["text_root"]),
        max_frames=300,
    )
    dataset = _dataset(dataset_args, str(dataset_info["split"]))
    output_dir.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    target_sha_cache: dict[str, str] = {}
    for pair_index, key in enumerate(case_keys):
        baseline_record = baseline[key]
        candidate_record = candidate[key]
        mismatched = [
            name
            for name in IDENTITY_FIELDS
            if baseline_record.get(name) != candidate_record.get(name)
        ]
        if mismatched:
            raise RuntimeError(f"Case {key} is not matched: {mismatched}")
        case = ContactCase(
            dataset_index=int(candidate_record["dataset_index"]),
            subtype=str(candidate_record["subtype"]),
            text_regime=str(candidate_record["text_regime"]),
            sample_seed=int(candidate_record["sample_seed"]),
        )
        prepared = _prepare_case(
            dataset,
            case,
            max_sparse_keyframes=int(candidate_protocol["max_sparse_keyframes"]),
            target_asset_sha_cache=target_sha_cache,
        )
        _validate_case_evidence(candidate_record, case, prepared)
        _validate_case_evidence(baseline_record, case, prepared)
        for offset, (variant, label, directory, record) in enumerate(
            (
                ("r11", args.baseline_label, baseline_dir, baseline_record),
                ("r12", args.candidate_label, candidate_dir, candidate_record),
            )
        ):
            manifest_rows.append(
                materialize_variant(
                    eval_dir=directory,
                    record=record,
                    prepared=prepared,
                    output_dir=output_dir,
                    selection_index=2 * pair_index + offset,
                    variant=variant,
                    variant_label=label,
                    checkpoint_label=label,
                )
            )

    write_json(
        output_dir / "gallery_manifest.json",
        {
            "format": "hy273_matched_control_pair_gallery_v1",
            "num_cases": len(manifest_rows),
            "num_pairs": len(case_keys),
            "case_keys": case_keys,
            "cases": manifest_rows,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "pairs": len(case_keys)}))


if __name__ == "__main__":
    main()
