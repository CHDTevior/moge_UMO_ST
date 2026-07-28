#!/usr/bin/env python
"""Build a renderable gallery directly from a validated v5 control evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_hy273_kimodo_v5_contact import (
    CONTROL_SUMMARY_FORMAT,
    PROTOCOL_VERSION,
    ContactCase,
    CompiledKimodoContactConstraint,
    V5_ALL_SUBTYPES,
    _dataset,
    _load_case_output,
    _prepare_case,
    _validate_case_evidence,
)


SELECTION_FORMAT = "hy273_kimodo_v5_gallery_selection_v1"
GALLERY_FORMAT = "hy273_kimodo_gallery_manifest_v1"
ERROR_METRICS = (
    "constraint_root2d_err",
    "constraint_fullbody_keyframe",
    "constraint_end_effector",
    "constraint_end_effector_rotation_deg",
    "controlled_contact_bce",
    "controlled_contact_brier",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def _parse_quantiles(value: str) -> tuple[float, ...]:
    quantiles = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not quantiles or any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError(f"Invalid quantiles: {value!r}")
    return quantiles


def _percentile_ranks(
    rows: list[dict[str, Any]], metric_name: str
) -> dict[str, float]:
    values = sorted(
        (
            float(row["metrics"]["generated_raw"][metric_name]),
            str(row["case_key"]),
        )
        for row in rows
        if metric_name in row["metrics"]["generated_raw"]
    )
    if not values:
        return {}
    denominator = max(len(values) - 1, 1)
    output: dict[str, float] = {}
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end][0] == values[start][0]:
            end += 1
        rank = 0.5 * (start + end - 1) / denominator
        for _, case_key in values[start:end]:
            output[case_key] = float(rank)
        start = end
    return output


def _select_cases(
    records: list[dict[str, Any]], quantiles: tuple[float, ...]
) -> list[dict[str, Any]]:
    with_text = [row for row in records if row.get("text_regime") == "withtext"]
    if len(with_text) != 4042:
        raise RuntimeError(f"Expected 4042 with-text cases, found {len(with_text)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in with_text:
        grouped[str(row["subtype"])].append(row)
    if set(grouped) != set(V5_ALL_SUBTYPES):
        raise RuntimeError("Control records do not cover all v5 subtypes")

    selected: list[dict[str, Any]] = []
    for subtype in V5_ALL_SUBTYPES:
        rows = grouped[subtype]
        rank_maps = {
            metric: ranks
            for metric in ERROR_METRICS
            if (ranks := _percentile_ranks(rows, metric))
        }
        if not rank_maps:
            raise RuntimeError(f"No control-error metric is available for {subtype}")
        scored: list[tuple[float, dict[str, Any], dict[str, float]]] = []
        for row in rows:
            case_key = str(row["case_key"])
            ranks = {
                metric: values[case_key]
                for metric, values in rank_maps.items()
                if case_key in values
            }
            score = float(sum(ranks.values()) / len(ranks))
            scored.append((score, row, ranks))
        scored.sort(key=lambda item: (item[0], str(item[1]["case_key"])))

        used: set[str] = set()
        for quantile in quantiles:
            candidates = [item for item in scored if str(item[1]["case_key"]) not in used]
            score, row, ranks = min(
                candidates,
                key=lambda item: (abs(item[0] - quantile), str(item[1]["case_key"])),
            )
            used.add(str(row["case_key"]))
            selected.append(
                {
                    "selection_index": len(selected),
                    "target_quantile": float(quantile),
                    "composite_percentile": float(score),
                    "metric_percentiles": ranks,
                    "formal_record": row,
                }
            )
    return selected


def _family(subtype: str) -> str:
    if "contact" in subtype:
        return "contact"
    if subtype.startswith(("path_", "waypoint_")):
        return "root_path"
    if subtype in {"inbetweening", "random"}:
        return "full_pose"
    if subtype.startswith(("hands", "feet")):
        return "end_effector"
    return "composite"


def _case_dir(output_dir: Path, selected: dict[str, Any]) -> Path:
    record = selected["formal_record"]
    quantile = int(round(100.0 * float(selected["target_quantile"])))
    return output_dir / "cases" / (
        f"{int(selected['selection_index']):02d}_{record['subtype']}_"
        f"q{quantile:02d}_{record['motion_id']}"
    )


def build(args: argparse.Namespace) -> None:
    eval_dir = Path(args.formal_eval_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "validated" or summary.get("format") != CONTROL_SUMMARY_FORMAT:
        raise RuntimeError("The input is not a validated v5 control summary")
    protocol = summary["protocol"]
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("The control protocol is not v5 contact evidence v2")
    if int(summary.get("case_count", -1)) != 8084:
        raise RuntimeError("The formal control evaluation is incomplete")

    shard_paths = sorted((eval_dir / "shards").glob("shard_*.jsonl"))
    if len(shard_paths) != 8:
        raise RuntimeError(f"Expected eight shard files, found {len(shard_paths)}")
    records = [row for path in shard_paths for row in _load_jsonl(path)]
    if len(records) != 8084 or any(row.get("status") != "ok" for row in records):
        raise RuntimeError("Formal control records are incomplete")
    quantiles = _parse_quantiles(args.quantiles)
    selected = _select_cases(records, quantiles)

    preflight = json.loads(
        (eval_dir / "preflight_manifest.json").read_text(encoding="utf-8")
    )
    dataset_info = preflight["dataset"]
    dataset_args = argparse.Namespace(
        data_root=dataset_info["data_root"],
        text_root=dataset_info["text_root"],
        max_frames=300,
    )
    dataset = _dataset(dataset_args, str(dataset_info["split"]))
    target_asset_sha_cache: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []
    render_protocol = {
        "num_steps": int(protocol["ode_steps"]),
        "cfg_scale_text": float(protocol["text_cfg_scale"]),
        "cfg_scale_control": float(protocol["control_cfg_scale"]),
        "cfg_apply_contacts": bool(protocol["cfg_apply_contacts"]),
        "contact_init": str(protocol["contact_init"]),
        "contact_feedback": str(protocol["contact_feedback"]),
        "max_sparse_keyframes": int(protocol["max_sparse_keyframes"]),
    }
    checkpoint_stem = Path(str(protocol["checkpoint"])).stem
    checkpoint_step = int(checkpoint_stem.removeprefix("step_"))
    weight_source = str(protocol["weight_source"])
    checkpoint_label = (
        f"{'EMA' if weight_source == 'ema' else 'Model'} "
        f"{checkpoint_step // 1000}K"
    )

    for selected_case in selected:
        record = selected_case["formal_record"]
        case = ContactCase(
            dataset_index=int(record["dataset_index"]),
            subtype=str(record["subtype"]),
            text_regime="withtext",
            sample_seed=int(record["sample_seed"]),
        )
        prepared = _prepare_case(
            dataset,
            case,
            max_sparse_keyframes=int(protocol["max_sparse_keyframes"]),
            target_asset_sha_cache=target_asset_sha_cache,
            unified_noise=bool(protocol.get("unified_273_flow", False)),
        )
        _validate_case_evidence(
            record,
            case,
            prepared,
            output_dir=eval_dir,
            verify_output=True,
        )
        generated_raw, generated_exact = _load_case_output(
            record, case, prepared, eval_dir
        )
        constraint = prepared["constraint_cpu"]
        base = constraint.base if isinstance(constraint, CompiledKimodoContactConstraint) else constraint
        contact_mask = (
            constraint.contact_metric_mask
            if isinstance(constraint, CompiledKimodoContactConstraint)
            else np.zeros((int(record["length"]), 4), dtype=bool)
        )
        if not isinstance(contact_mask, np.ndarray):
            contact_mask = contact_mask.cpu().numpy()

        case_dir = _case_dir(output_dir, selected_case)
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "generated_raw.npy", generated_raw.cpu().numpy())
        np.save(case_dir / "generated_exact_clamped.npy", generated_exact.cpu().numpy())
        np.save(case_dir / "target.npy", prepared["target_cpu"].cpu().numpy())
        np.save(case_dir / "observed.npy", constraint.observed_motion.cpu().numpy())
        np.save(case_dir / "mask.npy", constraint.motion_mask.cpu().numpy())
        np.save(case_dir / "root_metric_frames.npy", base.root_metric_frames.cpu().numpy())
        np.save(case_dir / "fullbody_metric_frames.npy", base.fullbody_metric_frames.cpu().numpy())
        np.save(
            case_dir / "endpoint_position_metric_mask.npy",
            base.endpoint_position_metric_mask.cpu().numpy(),
        )
        np.save(
            case_dir / "endpoint_rotation_metric_mask.npy",
            base.endpoint_rotation_metric_mask.cpu().numpy(),
        )
        np.save(case_dir / "contact_metric_mask.npy", contact_mask)

        metadata = {
            "format": "hy273_kimodo_v5_gallery_case_v1",
            "case_key": case.key,
            "selection_index": int(selected_case["selection_index"]),
            "target_quantile": float(selected_case["target_quantile"]),
            "composite_percentile": float(selected_case["composite_percentile"]),
            "metric_percentiles": selected_case["metric_percentiles"],
            "subtype": case.subtype,
            "family": _family(case.subtype),
            "dataset_index": case.dataset_index,
            "motion_id": str(record["motion_id"]),
            "length": int(record["length"]),
            "caption": str(record["text"]),
            "sample_seed": case.sample_seed,
            "constraint_components": constraint.components,
            "formal_metrics": record["metrics"]["generated_raw"],
            "regenerated_metrics": record["metrics"]["generated_raw"],
            "diagnostic_exact_metrics": record["metrics"]["diagnostic_exact_clamp"],
            "protocol": render_protocol,
            "checkpoint_label": checkpoint_label,
            "source_evaluation": str(eval_dir),
        }
        _atomic_json(case_dir / "metadata.json", metadata)
        manifest_rows.append(
            {
                "case_dir": str(case_dir.relative_to(output_dir)),
                "metadata": str((case_dir / "metadata.json").relative_to(output_dir)),
                **metadata,
            }
        )
        print(f"GALLERY_CASE {case.key}", flush=True)

    selection_payload = {
        "format": SELECTION_FORMAT,
        "quantiles": list(quantiles),
        "case_count": len(selected),
        "source_evaluation": str(eval_dir),
        "cases": selected,
    }
    _atomic_json(output_dir / "selection.json", selection_payload)
    _atomic_json(
        output_dir / "gallery_manifest.json",
        {
            "format": GALLERY_FORMAT,
            "num_cases": len(manifest_rows),
            "selection_manifest": "selection.json",
            "cases": manifest_rows,
        },
    )
    print(
        json.dumps(
            {
                "gallery_manifest": str(output_dir / "gallery_manifest.json"),
                "num_cases": len(manifest_rows),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_eval_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--quantiles", default="0.25,0.50,0.75")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
