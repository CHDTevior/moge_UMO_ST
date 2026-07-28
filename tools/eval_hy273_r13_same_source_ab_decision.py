#!/usr/bin/env python3
"""Apply the frozen scientific decision rules for the R13 same-source A/B pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


T2M_DIRECTIONS = {
    "foot_skate_from_height": "lower",
    "foot_skate_from_pred_contacts": "lower",
    "foot_skate_ratio": "lower",
    "foot_contact_consistency": "higher",
    "fk_jerk_mps3": "lower",
    "position_channel_jerk_mps3": "lower",
}

CONTROL_DIRECTIONS = {
    "constraint_end_effector": "lower",
    "constraint_end_effector_rotation_deg": "lower",
    "constraint_fullbody_keyframe": "lower",
    "constraint_root2d_err": "lower",
    "constraint_root2d_acc": "higher",
    "controlled_contact_brier": "lower",
    "controlled_contact_accuracy": "higher",
    "controlled_contact_f1": "higher",
}
PRIMARY_EDIT_SUBSET = "target_disjoint_asset_nonoverlap"


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label.strip() or not path.is_file():
        raise ValueError(f"Invalid labeled path: {value!r}")
    return label.strip(), path


def load_labeled(values: list[str]) -> dict[str, dict[str, Any]]:
    output = {}
    for value in values:
        label, path = parse_labeled_path(value)
        if label in output:
            raise ValueError(f"Duplicate label: {label}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["__input_path"] = str(path)
        output[label] = payload
    return output


def relative_degradation(candidate: float, parent: float, direction: str) -> float:
    denominator = max(abs(float(parent)), 1e-8)
    if direction == "lower":
        return (float(candidate) - float(parent)) / denominator
    if direction == "higher":
        return (float(parent) - float(candidate)) / denominator
    raise ValueError(f"Unknown metric direction: {direction}")


def evaluate_directional_metrics(
    parent: dict[str, float],
    candidate: dict[str, float],
    directions: dict[str, str],
    *,
    maximum_degradation: float,
) -> dict[str, Any]:
    rows = {}
    for metric, direction in directions.items():
        if metric not in parent or metric not in candidate:
            raise KeyError(f"Missing guardrail metric: {metric}")
        degradation = relative_degradation(candidate[metric], parent[metric], direction)
        rows[metric] = {
            "direction": direction,
            "parent": float(parent[metric]),
            "candidate": float(candidate[metric]),
            "relative_degradation": float(degradation),
            "passed": bool(degradation <= float(maximum_degradation)),
        }
    return {
        "passed": all(row["passed"] for row in rows.values()),
        "metrics": rows,
    }


def t2m_guardrail(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_degradation: float,
) -> dict[str, Any]:
    def metadata(payload: dict[str, Any]) -> dict[str, Any]:
        path = Path(payload["__input_path"]).parent / "metadata.json"
        if not path.is_file():
            raise RuntimeError(f"Missing T2M protocol metadata: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    parent_metadata = metadata(parent)
    candidate_metadata = metadata(candidate)
    protocol_fields = (
        "ode_steps",
        "text_cfg_scale",
        "source_cfg_scale",
        "edit_cfg_scale",
        "control_cfg_scale",
        "cfg_apply_contacts",
        "contact_protocol",
        "initial_noise_source",
        "seed",
        "texts",
        "lengths",
        "weight_source",
    )
    protocol = {name: parent_metadata.get(name) for name in protocol_fields}
    candidate_protocol = {
        name: candidate_metadata.get(name) for name in protocol_fields
    }
    if protocol != candidate_protocol:
        raise RuntimeError("T2M systems do not share the frozen sampling protocol")
    expected = {
        "ode_steps": 32,
        "text_cfg_scale": 2.0,
        "seed": 3407,
        "weight_source": "model",
    }
    if any(protocol[name] != value for name, value in expected.items()):
        raise RuntimeError(f"Unexpected T2M pilot protocol: {protocol}")
    parent_rows = parent.get("per_sample", [])
    candidate_rows = candidate.get("per_sample", [])
    parent_keys = [
        (row.get("index"), row.get("text"), row.get("length")) for row in parent_rows
    ]
    candidate_keys = [
        (row.get("index"), row.get("text"), row.get("length"))
        for row in candidate_rows
    ]
    if not parent_rows or parent_keys != candidate_keys:
        raise RuntimeError("T2M systems do not share the fixed prompt/length plan")
    result = evaluate_directional_metrics(
        parent["aggregate"],
        candidate["aggregate"],
        T2M_DIRECTIONS,
        maximum_degradation=maximum_degradation,
    )
    result["cases"] = len(parent_rows)
    result["matched_prompt_length_seed_protocol"] = True
    result["protocol"] = protocol
    return result


def control_rows(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    if summary.get("status") != "validated":
        raise RuntimeError("Control benchmark did not finish validated")
    output = {}
    for row in summary.get("rows", []):
        if row.get("subtype") == "all":
            output[str(row["text_regime"])] = row["generated_raw"]
    if set(output) != {"withtext", "notext"}:
        raise RuntimeError("Control summary lacks all-subtype text regimes")
    return output


def control_guardrail(
    parent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_degradation: float,
) -> dict[str, Any]:
    protocol_fields = (
        "case_count",
        "case_plan_sha256",
        "ode_steps",
        "text_cfg_scale",
        "control_cfg_scale",
        "seed",
        "max_sparse_keyframes",
        "subtypes",
        "text_regimes",
        "contact_feedback",
        "cfg_apply_contacts",
        "primary_output",
        "weight_source",
    )
    parent_protocol = {
        name: parent.get("protocol", {}).get(name) for name in protocol_fields
    }
    candidate_protocol = {
        name: candidate.get("protocol", {}).get(name) for name in protocol_fields
    }
    if parent_protocol != candidate_protocol:
        raise RuntimeError("Control systems do not share the frozen case/sampling protocol")
    expected = {
        "ode_steps": 32,
        "text_cfg_scale": 2.0,
        "control_cfg_scale": 2.0,
        "seed": 3407,
        "weight_source": "model",
    }
    if any(parent_protocol[name] != value for name, value in expected.items()):
        raise RuntimeError(f"Unexpected control pilot protocol: {parent_protocol}")
    parent_rows = control_rows(parent)
    candidate_rows = control_rows(candidate)
    regimes = {
        regime: evaluate_directional_metrics(
            parent_rows[regime],
            candidate_rows[regime],
            CONTROL_DIRECTIONS,
            maximum_degradation=maximum_degradation,
        )
        for regime in ("withtext", "notext")
    }
    return {
        "passed": all(row["passed"] for row in regimes.values()),
        "regimes": regimes,
        "protocol": parent_protocol,
        "exact_observed_channel_overwrite": (
            "passed implicitly: validated control evaluator checks torch.equal on every case"
        ),
    }


def edit_guardrail(
    fixed: dict[str, Any],
    *,
    parent_label: str,
    candidate_label: str,
    maximum_degradation: float,
) -> dict[str, Any]:
    parent = fixed["systems"][parent_label]
    candidate = fixed["systems"][candidate_label]
    parent_fixed = parent["timesteps"]["0.0"]["subset_aggregates"][
        PRIMARY_EDIT_SUBSET
    ]["full_273"]
    candidate_fixed = candidate["timesteps"]["0.0"]["subset_aggregates"][
        PRIMARY_EDIT_SUBSET
    ]["full_273"]
    comparison = fixed["comparisons"][candidate_label][PRIMARY_EDIT_SUBSET]["0.0"][
        "full_273"
    ]
    if "ode" not in candidate:
        raise RuntimeError("Raw-primary fixed evaluation lacks matched-noise ODE")
    parent_ode = parent["ode"]["subset_aggregates"][PRIMARY_EDIT_SUBSET][
        "full_273"
    ]
    candidate_ode = candidate["ode"]["subset_aggregates"][PRIMARY_EDIT_SUBSET][
        "full_273"
    ]
    reconstruction_degradation = relative_degradation(
        candidate_fixed["correct_instruction_mse"],
        parent_fixed["correct_instruction_mse"],
        "lower",
    )
    ode_reconstruction_degradation = relative_degradation(
        candidate_ode["correct_instruction_mse"],
        parent_ode["correct_instruction_mse"],
        "lower",
    )
    assignment_delta = comparison["assignment_advantage"][
        "mean_delta_candidate_minus_baseline"
    ]
    checks = {
        "parent_relative_assignment_advantage_positive": bool(assignment_delta > 0.0),
        "fixed_t_correct_better_than_empty": bool(
            candidate_fixed["correct_vs_empty_mse_gap"] > 0.0
        ),
        "fixed_t_correct_mse_nonregression": bool(
            reconstruction_degradation <= float(maximum_degradation)
        ),
        "ode32_correct_not_worse_than_empty": bool(
            candidate_ode["correct_vs_empty_mse_gap"] >= 0.0
        ),
        "ode32_correct_mse_nonregression": bool(
            ode_reconstruction_degradation <= float(maximum_degradation)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "parent_relative_assignment_advantage": float(assignment_delta),
        "fixed_t_correct_mse_relative_degradation": float(
            reconstruction_degradation
        ),
        "ode32_correct_mse_relative_degradation": float(
            ode_reconstruction_degradation
        ),
        "fixed_t": candidate_fixed,
        "parent_ode32": parent_ode,
        "ode32": candidate_ode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed_t_raw", type=Path, required=True)
    parser.add_argument("--t2m", action="append", required=True)
    parser.add_argument("--control", action="append", required=True)
    parser.add_argument("--parent_label", default="parent400k")
    parser.add_argument(
        "--candidate_label",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--direct_comparison",
        default="",
        help="LEFT,RIGHT candidate labels; defaults to candidate order.",
    )
    parser.add_argument("--maximum_relative_degradation", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed = json.loads(args.fixed_t_raw.expanduser().resolve().read_text(encoding="utf-8"))
    if fixed.get("weight_source") != "model":
        raise RuntimeError("The pilot decision must use raw model weights")
    t2m = load_labeled(args.t2m)
    control = load_labeled(args.control)
    candidate_labels = args.candidate_label or ["hinge405k", "softplus405k"]
    if len(candidate_labels) != 2 or len(set(candidate_labels)) != 2:
        raise ValueError("The controlled A/B decision requires two candidate labels")
    required = {args.parent_label, *candidate_labels}
    if required - set(fixed["systems"]) or required - set(t2m) or required - set(control):
        raise RuntimeError("Fixed-t, T2M, and control inputs must share every label")

    candidates = {}
    for label in candidate_labels:
        edit = edit_guardrail(
            fixed,
            parent_label=args.parent_label,
            candidate_label=label,
            maximum_degradation=float(args.maximum_relative_degradation),
        )
        t2m_row = t2m_guardrail(
            t2m[args.parent_label],
            t2m[label],
            maximum_degradation=float(args.maximum_relative_degradation),
        )
        control_row = control_guardrail(
            control[args.parent_label],
            control[label],
            maximum_degradation=float(args.maximum_relative_degradation),
        )
        candidates[label] = {
            "passed": bool(edit["passed"] and t2m_row["passed"] and control_row["passed"]),
            "edit": edit,
            "t2m_physical": t2m_row,
            "kimodo_like_control": control_row,
        }

    if args.direct_comparison:
        parts = [part.strip() for part in args.direct_comparison.split(",")]
        if len(parts) != 2 or set(parts) != set(candidate_labels):
            raise ValueError(
                "--direct_comparison must contain the two candidate labels"
            )
        left_label, right_label = parts
    else:
        left_label, right_label = candidate_labels
    direct_key = f"{right_label}_minus_{left_label}"
    direct = fixed.get("direct_comparisons", {}).get(direct_key)
    if direct is None:
        raise RuntimeError(f"Missing direct paired bootstrap: {direct_key}")
    direct_metric = direct["subsets"][PRIMARY_EDIT_SUBSET]["timesteps"]["0.0"][
        "full_273"
    ]["assignment_advantage"]
    ci_low, ci_high = direct_metric["paired_bootstrap_ci95"]
    eligible = [label for label, row in candidates.items() if row["passed"]]
    if left_label in eligible and right_label in eligible:
        if ci_low > 0.0:
            selection = [right_label]
        elif ci_high < 0.0:
            selection = [left_label]
        else:
            selection = [left_label, right_label]
    else:
        selection = eligible

    result = {
        "format": "hy273_r13_same_source_ab_decision_v1",
        "status": "passed" if selection else "no_candidate_passed",
        "maximum_relative_degradation": float(args.maximum_relative_degradation),
        "candidates": candidates,
        "direct_candidate_comparison": {
            "left_label": left_label,
            "right_label": right_label,
            "metric": direct_metric,
        },
        "selected_for_next_experiment": selection,
        "claim_limit": (
            "This selects between two time-matched continuations from the same parent; "
            "the conclusion is limited to the frozen held-out Edit, T2M, and raw-control "
            "protocols reported here."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selection": selection}, sort_keys=True))


if __name__ == "__main__":
    main()
