"""Shardable internal K273 evaluation for MotionFix editing capabilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_multitask_manifest_dataset import sha256_file as dataset_sha256_file
from models.raw_motion.hy273_motionfix_metrics import (
    INTERNAL_METRIC_PROTOCOL,
    PHYSICAL_TIMEWARP_PROTOCOL,
    evaluate_motionfix_internal_case,
    physical_timewarp_hy201_to_k273,
)
from models.raw_motion.hy273_kimodo_benchmark import (
    KIMODO_CONTROL_SUBTYPES,
    compile_kimodo_constraint,
    evaluate_kimodo_constraint_case,
)
from models.raw_motion.hy273_kimodo_contact_benchmark import (
    V5_CONTACT_SUBTYPES,
    compile_kimodo_contact_constraint,
    evaluate_kimodo_contact_case,
)
from models.raw_motion.evidence_hash import tensor_sha256
from models.raw_motion.flow_schedule import (
    LEGACY_SPLIT_CONTACT_PROTOCOL,
    UNIFIED_273_CONTACT_PROTOCOL,
    uses_unified_273_flow,
)
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    make_absent_condition,
)
from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    CONT_DIM,
    DIM_HY273,
    HEADING_SLICE,
)
from sample_hy273_multitask import (
    CONTACT_PROTOCOL_VERSION,
    EDIT_SOURCE_BASELINE_MODES,
    SAMPLING_PROTOCOL_VERSION,
    UNIFIED_CONTACT_PROTOCOL_VERSION,
    make_edit_condition,
    make_instruction_only_edit_diagnostic_condition,
    normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from train_hy273_multitask import (
    create_model_from_checkpoint,
    contact_protocol_for_config,
    current_code_identity,
    source_fusion_mode_from_checkpoint,
    text_fusion_mode_from_checkpoint,
    text_global_conditioning_from_checkpoint,
    validate_assets,
    validate_frozen_contract,
)


EQUAL_PROTOCOL = "motionfix_equal_length_952_internal_k273_v1"
FULL_PROTOCOL = "motionfix_full_requested_length_1013_internal_k273_v1"
VAL_PROTOCOL = "motionfix_val_selected_internal_k273_v1"
PREFLIGHT_FORMAT = "hy273_motionfix_edit_preflight_v8_full_closure_sealed"
PROTOCOL_FORMAT = "hy273_motionfix_edit_protocol_v3_full_closure_sealed"
SUMMARY_FORMAT = "hy273_motionfix_edit_summary_v8_control_safety_integrated"
ARTIFACT_INDEX_FORMAT = "hy273_motionfix_artifact_index_v2_canonical_reverse_sealed"
ARTIFACT_INDEX_CONTRACT_FORMAT = (
    "hy273_motionfix_artifact_index_immutable_contract_v1"
)
DEFAULT_MANIFEST = (
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/test.jsonl"
)
DEFAULT_TRAIN_MANIFEST = (
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/train.jsonl"
)
DEFAULT_COUNTERFACTUAL_MANIFEST = str(
    ROOT
    / "outputs"
    / "hy273_multitask"
    / "gates"
    / "motionfix_edit_counterfactual_manifest_v1.jsonl"
)
DEFAULT_COUNTERFACTUAL_SUMMARY = str(
    ROOT
    / "outputs"
    / "hy273_multitask"
    / "gates"
    / "motionfix_edit_counterfactual_manifest_v1_summary.json"
)
COUNTERFACTUAL_FORMAT = "motionfix_edit_counterfactual_manifest_v1"
DEFAULT_COUNTERFACTUAL_MANIFEST_SHA256 = (
    "8bc5f61a44906312c581540b65792cee0bdd03c244a24389330c8755b317e6c3"
)
DEFAULT_COUNTERFACTUAL_SUMMARY_SHA256 = (
    "fb8a2f673b1c506cfbf7818ef5d0c944c24e3bcb6be4f226c47b5523eff84013"
)
SYSTEM_SOURCE_COPY = "source_copy"
SYSTEM_MODEL = "source_instruction_model"
SYSTEM_SOURCE_SHUFFLE = "shuffled_source_instruction_model"
SYSTEM_INSTRUCTION_SHUFFLE = "source_shuffled_instruction_model"
SYSTEM_INSTRUCTION_DROP = "source_only_model"
SYSTEM_INSTRUCTION_ONLY = "relative_instruction_only_ood"
SYSTEM_EDIT_CONTROL = "source_instruction_control_model"
SYSTEM_STANDALONE_CONTROL = "standalone_control_model"
ALL_SYSTEMS = (
    SYSTEM_SOURCE_COPY,
    SYSTEM_MODEL,
    SYSTEM_SOURCE_SHUFFLE,
    SYSTEM_INSTRUCTION_SHUFFLE,
    SYSTEM_INSTRUCTION_DROP,
    SYSTEM_INSTRUCTION_ONLY,
    SYSTEM_EDIT_CONTROL,
    SYSTEM_STANDALONE_CONTROL,
)
MODEL_SYSTEMS = frozenset(ALL_SYSTEMS) - {SYSTEM_SOURCE_COPY}
COUNTERFACTUAL_SYSTEMS = frozenset(
    {SYSTEM_SOURCE_SHUFFLE, SYSTEM_INSTRUCTION_SHUFFLE}
)
EDIT_CONTROL_SUBTYPES = (*KIMODO_CONTROL_SUBTYPES, *V5_CONTACT_SUBTYPES)
CONTROL_SYSTEMS = frozenset(
    {SYSTEM_EDIT_CONTROL, SYSTEM_STANDALONE_CONTROL}
)
EDITING_VISUAL_SYSTEMS = tuple(
    system for system in ALL_SYSTEMS if system not in CONTROL_SYSTEMS
)
EDITING_CONTACT_PROTOCOL = "editing_selected_contact_v1"
EDITING_CONTACT_CFG_PROTOCOL = "editing_hierarchical_contact_cfg_v1"
GENERATE_CONTROL_CONTACT_PROTOCOL = "legacy_generate_control_contact_cfg_v1"
EXACT_CLAMP_PROTOCOL = "post_denormalization_physical_overwrite_v1"
EXPECTED_CASE_FORMAT = "hy273_motionfix_expected_cases_v3_control_safety_schema"
MOTION_OUTPUT_FORMAT = "hy273_raw_exact_npz_v4_full_overwrite_contract"
ALIGNED_SOURCE_FORMAT = "hy273_evaluator_aligned_source_npy_v1"
VISUAL_SELECTION_FORMAT = (
    "hy273_motionfix_visual_review_selection_v7_canonical_reverse_sealed"
)
VISUAL_SELECTION_POLICY = (
    "first_4_equal_first_2_unequal_per_system_plus_first_per_control_subtype_v1"
)
EQUAL_ONLY_REGIONAL_METRICS = frozenset(
    {
        "changed_joint_entries",
        "unchanged_joint_entries",
        "ambiguous_joint_entries",
        "changed_region_target_error_m",
        "unchanged_region_source_error_m",
        "changed_region_target_rotation_error_deg",
        "unchanged_region_source_rotation_error_deg",
        "changed_position_threshold_m",
        "changed_rotation_threshold_deg",
        "unchanged_position_threshold_m",
        "unchanged_rotation_threshold_deg",
        "changed_temporal_dilation_frames",
    }
)
CONTROL_SAFETY_METRICS = (
    "prediction_jerk_mps3",
    "foot_skate_from_height",
    "foot_skate_from_pred_contacts",
    "foot_skate_max_vel",
    "foot_skate_ratio",
    "foot_contact_consistency",
    "fk_position_rotation_consistency_cm",
)


def _checkpoint_step(checkpoint: dict[str, Any]) -> int:
    if "next_global_step" in checkpoint:
        return int(checkpoint["next_global_step"])
    return int(checkpoint.get("step", -1))


@dataclass(frozen=True)
class EditCase:
    row_index: int
    pair_id: str
    system: str
    sample_seed: int
    control_subtype: str | None = None

    @property
    def key(self) -> str:
        suffix = f"__{self.control_subtype}" if self.control_subtype else ""
        return f"pair_{self.pair_id}__{self.system}{suffix}"


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha(payload: Any) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_equal_strict(first: Any, second: Any) -> bool:
    """Compare JSON values without Python's bool/int/float aliasing."""

    return _canonical_json(first) == _canonical_json(second)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"Duplicate JSON object key: {key}")
        value[key] = item
    return value


def _load_json_object_strict(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(
        resolved.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {resolved}")
    return payload


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _normalize_counterfactual_text(value: str) -> str:
    return " ".join(str(value).split())


def stable_seed(seed: int, pair_id: str) -> int:
    digest = hashlib.sha256(f"motionfix-edit:{seed}:{pair_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as writer:
        for row in rows:
            writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _artifact_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": dataset_sha256_file(resolved),
        "size": int(resolved.stat().st_size),
    }


def _required_render_views() -> list[dict[str, Any]]:
    common = {
        "fps": 30,
        "stride": 4,
        "trail_frames": 12,
        "output_format": "gif",
    }
    return [
        {
            "view_id": f"{prediction_key}_{family}",
            "prediction_key": prediction_key,
            "systems": list(systems),
            "max_cases": max_cases,
            **common,
        }
        for prediction_key in ("raw", "exact")
        for family, systems, max_cases in (
            ("editing6", EDITING_VISUAL_SYSTEMS, 6),
            (
                "contact12",
                (SYSTEM_EDIT_CONTROL, SYSTEM_STANDALONE_CONTROL),
                12,
            ),
        )
    ]


def _file_stat(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "inode": int(stat.st_ino),
        "device": int(stat.st_dev),
    }


def _checkpoint_preflight_identity(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    *,
    checkpoint_sha256: str,
    file_stat: dict[str, int | str] | None = None,
) -> dict[str, Any]:
    """Build the checkpoint identity shared by preflight creation and loading."""

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint has no resolved multitask config")
    unified_273_flow = uses_unified_273_flow(contact_protocol_for_config(config))
    return {
        **(file_stat if file_stat is not None else _file_stat(checkpoint_path)),
        "sha256": str(checkpoint_sha256),
        "format": checkpoint.get("format"),
        "step": _checkpoint_step(checkpoint),
        "unified_273_flow": unified_273_flow,
    }


def _git_dependency_identity(
    repo: Path, relative_paths: Iterable[Path]
) -> dict[str, Any]:
    repo = repo.expanduser().resolve()
    if not (repo / ".git").is_dir():
        raise RuntimeError(f"Evaluation dependency is not a Git repository: {repo}")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    files = []
    for relative in sorted({Path(value) for value in relative_paths}, key=str):
        path = (repo / relative).resolve()
        if not path.is_file() or not path.is_relative_to(repo):
            raise RuntimeError(f"Missing evaluation dependency file: {path}")
        files.append(
            {
                "path": str(relative),
                "size": int(path.stat().st_size),
                "sha256": dataset_sha256_file(path),
            }
        )
    payload = {
        "repo": str(repo),
        "commit": commit,
        "dirty": bool(status_lines),
        "status_lines": status_lines,
        "status_sha256": canonical_sha(status_lines),
        "files": files,
    }
    payload["identity_sha256"] = canonical_sha(payload)
    return payload


def external_dependency_identity() -> dict[str, Any]:
    hy201_repo = Path("/mnt/afs/UMO_debug/hy201_to_kimodo273")
    kimodo_repo = Path("/mnt/afs/UMO_debug/outside_material/kimodo")
    hy201_files = sorted(
        path.relative_to(hy201_repo)
        for path in (hy201_repo / "hy201_to_kimodo273").rglob("*.py")
    )
    # Importing kimodo.motion_rep executes kimodo/__init__.py and its model
    # imports. Seal the complete Python package rather than a hand-picked subset.
    kimodo_files = sorted(
        path.relative_to(kimodo_repo)
        for path in (kimodo_repo / "kimodo").rglob("*.py")
    )
    kimodo_files.append(Path("kimodo/assets/skeletons/smplx22/joints.p"))
    payload = {
        "hy201_to_kimodo273": _git_dependency_identity(hy201_repo, hy201_files),
        "kimodo_runtime": _git_dependency_identity(kimodo_repo, kimodo_files),
    }
    payload["identity_sha256"] = canonical_sha(payload)
    return payload


def evaluation_code_identity() -> dict[str, Any]:
    paths = {
        Path(__file__).resolve(),
        ROOT / "sample_hy273_multitask.py",
        ROOT / "train_hy273_multitask.py",
        ROOT / "tools" / "build_motionfix_edit_counterfactual_manifest.py",
        ROOT / "tools" / "render_hy273_motionfix_review.py",
        ROOT / "models" / "__init__.py",
        ROOT
        / "external_repos"
        / "kimodo"
        / "kimodo"
        / "assets"
        / "skeletons"
        / "smplx22"
        / "joints.p",
        Path(
            "/mnt/afs/UMO_debug/hy201_to_kimodo273/"
            "hy201_to_kimodo273/__init__.py"
        ),
        Path(
            "/mnt/afs/UMO_debug/hy201_to_kimodo273/"
            "hy201_to_kimodo273/convert.py"
        ),
        Path(
            "/mnt/afs/UMO_debug/hy201_to_kimodo273/"
            "hy201_to_kimodo273/geometry.py"
        ),
        Path(
            "/mnt/afs/UMO_debug/hy201_to_kimodo273/"
            "hy201_to_kimodo273/kimodo_bridge.py"
        ),
        *sorted((ROOT / "common").glob("*.py")),
        *sorted((ROOT / "data").glob("*.py")),
        *sorted((ROOT / "models" / "codeflow").glob("*.py")),
        *sorted((ROOT / "models" / "raw_motion").glob("*.py")),
    }
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing local inference dependencies: {missing}")
    rows = [
        {
            "path": (
                str(path.relative_to(ROOT))
                if path.is_relative_to(ROOT)
                else str(path)
            ),
            "size": int(path.stat().st_size),
            "sha256": dataset_sha256_file(path),
        }
        for path in sorted(paths, key=str)
    ]
    payload = {
        "files": rows,
        "external_dependencies": external_dependency_identity(),
    }
    payload["sha256"] = canonical_sha(payload)
    return payload


def _validate_checkpoint_code_identity(
    checkpoint: dict[str, Any], *, allow_code_drift: bool = False
) -> dict[str, Any]:
    current = current_code_identity()
    if checkpoint.get("code_identity") != current:
        if not allow_code_drift:
            raise RuntimeError(
                "Checkpoint training-code identity differs from the current inference runtime"
            )
        checkpoint_identity = checkpoint.get("code_identity")
        if not isinstance(checkpoint_identity, dict):
            raise RuntimeError("Checkpoint has no valid training-code identity")
        return checkpoint_identity
    return current


def _validate_aggregate_training_code_identity(
    preflight: dict[str, Any],
    *,
    allow_code_drift: bool = False,
) -> dict[str, Any]:
    current = current_code_identity()
    if preflight.get("checkpoint_training_code_identity") != current:
        if not allow_code_drift:
            raise RuntimeError(
                "MotionFix checkpoint training-code identity differs before aggregation"
            )
        identity = preflight.get("checkpoint_training_code_identity")
        if not isinstance(identity, dict):
            raise RuntimeError("Preflight has no valid checkpoint training-code identity")
        return identity
    return current


def _hytext_profile_identity(config: dict[str, Any]) -> dict[str, Any]:
    cache_dir = Path(config["text"]["cache_dir"]).expanduser().resolve()
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_profile = "hytext_relative_edit_v1"
    if relative_profile not in manifest.get("profiles", {}):
        raise RuntimeError("HYText cache has no relative-edit profile")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": dataset_sha256_file(manifest_path),
        "cache_format": str(manifest.get("format")),
        "encoder_identity": str(manifest.get("encoder_identity")),
        "prompt_template_version": str(manifest.get("prompt_template_version")),
        "profile_contract_sha256": str(manifest.get("profile_contract_sha256")),
        "relative_profile_prompt_sha256": str(
            manifest["profile_prompt_sha256"][relative_profile]
        ),
        "relative_profile_crop_start": int(
            manifest["profile_crop_starts"][relative_profile]
        ),
    }


def load_motionfix_rows(
    manifest_path: str | Path,
    protocol: str = FULL_PROTOCOL,
) -> list[dict[str, Any]]:
    path = Path(manifest_path).expanduser().resolve()
    expected_split = "val" if protocol == VAL_PROTOCOL else "test"
    expected_count = 330 if protocol == VAL_PROTOCOL else 1013
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != "motionfix_k273":
                continue
            if row.get("split") != expected_split:
                raise RuntimeError(
                    f"MotionFix evaluator received non-{expected_split} row "
                    f"at line {line_number}"
                )
            if row.get("schema_version") != "hy273_multitask_manifest_v1":
                raise RuntimeError("MotionFix evaluator manifest schema mismatch")
            rows.append(row)
    pair_ids = [str(row["pair"]["official_pair_id"]) for row in rows]
    if len(rows) != expected_count or len(set(pair_ids)) != expected_count:
        raise RuntimeError(
            f"MotionFix {expected_split} protocol requires {expected_count} "
            f"unique pairs, got {len(rows)}"
        )
    if protocol == VAL_PROTOCOL:
        return rows
    equal = sum(row["pair"]["length_relation"] == "equal" for row in rows)
    if equal != 952:
        raise RuntimeError(f"Equal-length protocol requires 952 pairs, got {equal}")
    return rows


def protocol_rows(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    if protocol == EQUAL_PROTOCOL:
        selected = [row for row in rows if row["pair"]["length_relation"] == "equal"]
        if len(selected) != 952:
            raise RuntimeError("equal-length-952 selection changed")
        return selected
    if protocol == FULL_PROTOCOL:
        if len(rows) != 1013:
            raise RuntimeError("full-requested-length-1013 selection changed")
        return list(rows)
    if protocol == VAL_PROTOCOL:
        if len(rows) != 330:
            raise RuntimeError("validation-330 selection changed")
        return list(rows)
    raise ValueError(f"Unknown MotionFix protocol: {protocol}")


def selected_protocol_rows(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    selected = protocol_rows(rows, args.protocol)
    requested_ids = parse_csv(args.pair_ids)
    if requested_ids:
        if args.max_pairs > 0:
            raise ValueError("--pair_ids and --max_pairs are mutually exclusive")
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("--pair_ids contains duplicates")
        by_id = {
            str(row["pair"]["official_pair_id"]): row for row in selected
        }
        missing = [pair_id for pair_id in requested_ids if pair_id not in by_id]
        if missing:
            raise ValueError(f"Requested MotionFix pair ids are unavailable: {missing}")
        return [by_id[pair_id] for pair_id in requested_ids]
    if args.max_pairs > 0:
        selected = selected[: args.max_pairs]
    return selected


def _load_train_seen_index(path: str | Path) -> dict[str, set[Any]]:
    train_path = Path(path).expanduser().resolve()
    index: dict[str, set[Any]] = {
        "base_motion_ids": set(),
        "motion_payload_sha256s": set(),
        "source_target_base_pairs": set(),
        "source_target_payload_pairs": set(),
        "instruction_target_payload_pairs": set(),
        "instruction_target_base_pairs": set(),
        "official_pair_ids": set(),
    }
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "hy273_multitask_manifest_v1":
                raise RuntimeError(
                    f"Train manifest schema mismatch at line {line_number}"
                )
            if row.get("split") != "train":
                raise RuntimeError(
                    f"Seen-index manifest contains non-train row at line {line_number}"
                )
            source = row.get("source_motion")
            target = row.get("target_motion")
            if isinstance(source, dict):
                index["base_motion_ids"].add(str(source["base_motion_id"]))
                index["motion_payload_sha256s"].add(
                    str(source["k273_asset"]["sha256"])
                )
            if isinstance(target, dict):
                index["base_motion_ids"].add(str(target["base_motion_id"]))
                index["motion_payload_sha256s"].add(
                    str(target["k273_asset"]["sha256"])
                )
            if row.get("dataset") != "motionfix_k273":
                continue
            if not isinstance(source, dict) or not isinstance(target, dict):
                raise RuntimeError("MotionFix train row is missing source/target")
            source_base = str(source["base_motion_id"])
            target_base = str(target["base_motion_id"])
            source_sha = str(source["k273_asset"]["sha256"])
            target_sha = str(target["k273_asset"]["sha256"])
            index["source_target_base_pairs"].add((source_base, target_base))
            index["source_target_payload_pairs"].add((source_sha, target_sha))
            index["official_pair_ids"].add(str(row["pair"]["official_pair_id"]))
            for text_row in row.get("texts", []):
                instruction = _normalize_counterfactual_text(text_row["value"])
                index["instruction_target_payload_pairs"].add(
                    (instruction, target_sha)
                )
                index["instruction_target_base_pairs"].add(
                    (instruction, target_base)
                )
    return index


def _seen_strata(
    row: dict[str, Any], train_index: dict[str, set[Any]]
) -> dict[str, Any]:
    source = row["source_motion"]
    target = row["target_motion"]
    source_base = str(source["base_motion_id"])
    target_base = str(target["base_motion_id"])
    source_sha = str(source["k273_asset"]["sha256"])
    target_sha = str(target["k273_asset"]["sha256"])
    instruction = _normalize_counterfactual_text(row["texts"][0]["value"])
    source_seen = source_base in train_index["base_motion_ids"]
    target_seen = target_base in train_index["base_motion_ids"]
    source_payload_seen = source_sha in train_index["motion_payload_sha256s"]
    target_payload_seen = target_sha in train_index["motion_payload_sha256s"]
    return {
        "source_base_seen": source_seen,
        "target_base_seen": target_seen,
        "source_payload_seen": source_payload_seen,
        "target_payload_seen": target_payload_seen,
        "source_target_base_pair_seen": (
            source_base,
            target_base,
        )
        in train_index["source_target_base_pairs"],
        "exact_source_target_payload_pair_seen": (
            source_sha,
            target_sha,
        )
        in train_index["source_target_payload_pairs"],
        "instruction_target_payload_seen": (
            instruction,
            target_sha,
        )
        in train_index["instruction_target_payload_pairs"],
        "instruction_target_base_seen": (
            instruction,
            target_base,
        )
        in train_index["instruction_target_base_pairs"],
        "official_pair_id_seen": str(row["pair"]["official_pair_id"])
        in train_index["official_pair_ids"],
        "source_target_base_seen_category": (
            "both"
            if source_seen and target_seen
            else "source_only"
            if source_seen
            else "target_only"
            if target_seen
            else "neither"
        ),
        "source_target_payload_seen_category": (
            "both"
            if source_payload_seen and target_payload_seen
            else "source_only"
            if source_payload_seen
            else "target_only"
            if target_payload_seen
            else "neither"
        ),
    }


def load_counterfactual_rows(
    path: str | Path,
    motionfix_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    counterfactual_path = Path(path).expanduser().resolve()
    records: dict[str, dict[str, Any]] = {}
    with counterfactual_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("format") != COUNTERFACTUAL_FORMAT:
                raise RuntimeError(
                    f"Counterfactual schema mismatch at line {line_number}"
                )
            pair_id = str(row.get("official_pair_id", ""))
            if not pair_id or pair_id in records:
                raise RuntimeError(
                    f"Invalid/duplicate counterfactual pair at line {line_number}"
                )
            instruction = row.get("instruction_shuffle")
            source = row.get("source_shuffle")
            if not isinstance(instruction, dict) or not isinstance(source, dict):
                raise RuntimeError("Counterfactual row is missing source/instruction data")
            if instruction.get("encoding_profile") != "hytext_relative_edit_v1":
                raise RuntimeError("Instruction shuffle changed the HYText profile")
            if source.get("status") not in {
                "matched",
                "failed_no_legal_same_bucket_donor",
            }:
                raise RuntimeError("Unknown source-shuffle status")
            records[pair_id] = row

    expected = {
        str(row["pair"]["official_pair_id"]): row for row in motionfix_rows
    }
    if set(records) != set(expected):
        missing = sorted(set(expected) - set(records))[:5]
        extra = sorted(set(records) - set(expected))[:5]
        raise RuntimeError(
            f"Counterfactual/test pair mismatch: missing={missing}, extra={extra}"
        )
    rows_by_uid = {str(row["uid"]): row for row in motionfix_rows}
    instruction_donor_uids: set[str] = set()
    source_donor_uids: set[str] = set()
    for pair_id, row in expected.items():
        cf = records[pair_id]
        if str(cf.get("case_uid")) != str(row["uid"]):
            raise RuntimeError(f"Counterfactual uid mismatch for pair {pair_id}")
        if int(cf.get("requested_target_len", -1)) != int(
            row["pair"]["target_frames"]
        ):
            raise RuntimeError(
                f"Counterfactual target length mismatch for pair {pair_id}"
            )
        if int(cf.get("source_len", -1)) != int(row["pair"]["source_frames"]):
            raise RuntimeError(
                f"Counterfactual source length mismatch for pair {pair_id}"
            )

        source_len = int(row["source_motion"]["k273_asset"]["frames"])
        source_bucket = (source_len - 1) // 8
        if int(cf.get("source_length_bucket_8", -1)) != source_bucket:
            raise RuntimeError(
                f"Counterfactual source bucket mismatch for pair {pair_id}"
            )

        instruction = cf["instruction_shuffle"]
        instruction_uid = str(instruction.get("donor_uid", ""))
        if instruction_uid == str(row["uid"]) or instruction_uid not in rows_by_uid:
            raise RuntimeError(
                f"Instruction shuffle is not a valid derangement for pair {pair_id}"
            )
        instruction_donor = rows_by_uid[instruction_uid]
        donor_text_row = instruction_donor["texts"][0]
        donor_text = _normalize_counterfactual_text(donor_text_row["value"])
        original_text = _normalize_counterfactual_text(row["texts"][0]["value"])
        if donor_text == original_text:
            raise RuntimeError(
                f"Instruction shuffle preserved normalized text for pair {pair_id}"
            )
        if str(instruction.get("donor_text_id")) != str(donor_text_row["text_id"]):
            raise RuntimeError(
                f"Instruction donor text id mismatch for pair {pair_id}"
            )
        if str(instruction.get("donor_instruction")) != donor_text:
            raise RuntimeError(
                f"Instruction donor text mismatch for pair {pair_id}"
            )
        if str(instruction.get("donor_instruction_sha256")) != hashlib.sha256(
            donor_text.encode("utf-8")
        ).hexdigest():
            raise RuntimeError(
                f"Instruction donor SHA256 mismatch for pair {pair_id}"
            )
        instruction_donor_uids.add(instruction_uid)

        shuffled = cf["source_shuffle"]
        if shuffled.get("target_to_source_time_map_policy") != "normalized_progress_v1":
            raise RuntimeError(
                f"Source shuffle time-map policy mismatch for pair {pair_id}"
            )
        donor_fields = (
            "donor_uid",
            "donor_source_motion_uid",
            "donor_source_base_motion_id",
            "donor_source_path",
            "donor_source_sha256",
            "donor_source_len",
            "donor_length_bucket_8",
        )
        if shuffled["status"] == "failed_no_legal_same_bucket_donor":
            if any(shuffled.get(name) is not None for name in donor_fields):
                raise RuntimeError(
                    f"Fail-closed source shuffle carries a donor for pair {pair_id}"
                )
            original_source = row["source_motion"]
            original_target = row["target_motion"]
            for candidate in motionfix_rows:
                candidate_source = candidate["source_motion"]
                candidate_asset = candidate_source["k273_asset"]
                candidate_len = int(candidate_asset["frames"])
                if (candidate_len - 1) // 8 != source_bucket:
                    continue
                if str(candidate["uid"]) == str(row["uid"]):
                    continue
                candidate_identity = (
                    str(candidate_source["motion_uid"]),
                    str(candidate_source["base_motion_id"]),
                    str(candidate_asset["path"]),
                    str(candidate_asset["sha256"]),
                )
                source_identity = (
                    str(original_source["motion_uid"]),
                    str(original_source["base_motion_id"]),
                    str(original_source["k273_asset"]["path"]),
                    str(original_source["k273_asset"]["sha256"]),
                )
                target_identity = (
                    str(original_target["motion_uid"]),
                    str(original_target["base_motion_id"]),
                    str(original_target["k273_asset"]["path"]),
                    str(original_target["k273_asset"]["sha256"]),
                )
                if all(a != b for a, b in zip(candidate_identity, source_identity)) and all(
                    a != b for a, b in zip(candidate_identity, target_identity)
                ):
                    raise RuntimeError(
                        f"Source shuffle failed closed despite a legal donor for pair {pair_id}"
                    )
            continue

        donor_uid = str(shuffled.get("donor_uid", ""))
        if donor_uid == str(row["uid"]) or donor_uid not in rows_by_uid:
            raise RuntimeError(
                f"Source shuffle is not a valid derangement for pair {pair_id}"
            )
        donor = rows_by_uid[donor_uid]["source_motion"]
        donor_asset = donor["k273_asset"]
        expected_source_fields = {
            "donor_source_motion_uid": str(donor["motion_uid"]),
            "donor_source_base_motion_id": str(donor["base_motion_id"]),
            "donor_source_path": str(donor_asset["path"]),
            "donor_source_sha256": str(donor_asset["sha256"]),
            "donor_source_len": int(donor_asset["frames"]),
            "donor_length_bucket_8": (int(donor_asset["frames"]) - 1) // 8,
        }
        if any(shuffled.get(name) != value for name, value in expected_source_fields.items()):
            raise RuntimeError(f"Source donor metadata mismatch for pair {pair_id}")
        if int(shuffled["donor_length_bucket_8"]) != source_bucket:
            raise RuntimeError(f"Source donor left the frozen bucket for pair {pair_id}")
        original_source = row["source_motion"]
        original_target = row["target_motion"]
        donor_identity = (
            str(donor["motion_uid"]),
            str(donor["base_motion_id"]),
            str(donor_asset["path"]),
            str(donor_asset["sha256"]),
        )
        source_identity = (
            str(original_source["motion_uid"]),
            str(original_source["base_motion_id"]),
            str(original_source["k273_asset"]["path"]),
            str(original_source["k273_asset"]["sha256"]),
        )
        target_identity = (
            str(original_target["motion_uid"]),
            str(original_target["base_motion_id"]),
            str(original_target["k273_asset"]["path"]),
            str(original_target["k273_asset"]["sha256"]),
        )
        if not all(a != b for a, b in zip(donor_identity, source_identity)):
            raise RuntimeError(f"Source donor aliases the original source for pair {pair_id}")
        if not all(a != b for a, b in zip(donor_identity, target_identity)):
            raise RuntimeError(f"Source donor aliases the GT target for pair {pair_id}")
        source_donor_uids.add(donor_uid)

    if len(instruction_donor_uids) != len(motionfix_rows):
        raise RuntimeError("Instruction shuffle is not a one-to-one permutation")
    matched_source_count = sum(
        row["source_shuffle"]["status"] == "matched" for row in records.values()
    )
    if len(source_donor_uids) != matched_source_count:
        raise RuntimeError("Matched source shuffle is not a one-to-one permutation")
    return records


def build_plan(
    rows: list[dict[str, Any]],
    *,
    systems: Iterable[str],
    seed: int,
    counterfactual_rows: dict[str, dict[str, Any]] | None = None,
) -> list[EditCase]:
    systems = tuple(systems)
    unknown = sorted(set(systems) - set(ALL_SYSTEMS))
    if not systems or unknown:
        raise ValueError(f"Invalid evaluation systems: {unknown or systems}")
    plan = []
    for row_index, row in enumerate(rows):
        pair_id = str(row["pair"]["official_pair_id"])
        case_seed = stable_seed(seed, pair_id)
        for system in systems:
            if system in COUNTERFACTUAL_SYSTEMS:
                if counterfactual_rows is None:
                    raise ValueError(
                        f"System {system} requires a counterfactual manifest"
                    )
                if (
                    system == SYSTEM_SOURCE_SHUFFLE
                    and counterfactual_rows[pair_id]["source_shuffle"]["status"]
                    != "matched"
                ):
                    continue
            control_subtype = (
                EDIT_CONTROL_SUBTYPES[row_index % len(EDIT_CONTROL_SUBTYPES)]
                if system in CONTROL_SYSTEMS
                else None
            )
            plan.append(
                EditCase(
                    row_index,
                    pair_id,
                    system,
                    case_seed,
                    control_subtype,
                )
            )
    return plan


def plan_identity(
    rows: list[dict[str, Any]], plan: list[EditCase], args: argparse.Namespace
) -> dict[str, Any]:
    pair_rows = [
        {
            "pair_id": str(row["pair"]["official_pair_id"]),
            "source_frames": int(row["pair"]["source_frames"]),
            "target_frames": int(row["pair"]["target_frames"]),
            "length_relation": str(row["pair"]["length_relation"]),
        }
        for row in rows
    ]
    return {
        "protocol": args.protocol,
        "pair_count": len(rows),
        "case_count": len(plan),
        "smoke_subset": bool(args.max_pairs > 0 or parse_csv(args.pair_ids)),
        "max_pairs": int(args.max_pairs),
        "pair_ids": list(parse_csv(args.pair_ids)),
        "systems": list(parse_csv(args.systems)),
        "seed": int(args.seed),
        "pair_rows_sha256": canonical_sha(pair_rows),
        "case_keys_sha256": canonical_sha([case.key for case in plan]),
        "num_shards": int(args.num_shards),
        "control_subtypes": list(EDIT_CONTROL_SUBTYPES),
        "case_system_counts": {
            system: sum(case.system == system for case in plan)
            for system in parse_csv(args.systems)
        },
    }


def _evaluation_argument_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Arguments that can change generated values or aggregate statistics."""

    return {
        "protocol": str(args.protocol),
        "systems": list(parse_csv(args.systems)),
        "pair_ids": list(parse_csv(args.pair_ids)),
        "max_pairs": int(args.max_pairs),
        "num_shards": int(args.num_shards),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "source_cfg_scale": float(args.source_cfg_scale),
        "edit_cfg_scale": float(args.edit_cfg_scale),
        "edit_source_baseline": str(args.edit_source_baseline),
        "generate_text_cfg_scale": float(args.generate_text_cfg_scale),
        "control_cfg_scale": float(args.control_cfg_scale),
        "cfg_apply_contacts": bool(args.cfg_apply_contacts),
        "generate_cfg_apply_contacts": bool(args.generate_cfg_apply_contacts),
        "contact_init": str(args.contact_init),
        "contact_feedback": str(args.contact_feedback),
        "max_sparse_keyframes": int(args.max_sparse_keyframes),
        "weight_source": str(args.weight_source),
        "save_motion_outputs": bool(args.save_motion_outputs),
    }


def _run_contract(
    args: argparse.Namespace,
    *,
    checkpoint_sha256: str,
    plan_sha256: str,
    counterfactual_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    arguments = _evaluation_argument_identity(args)
    return {
        **arguments,
        "metric_protocol": INTERNAL_METRIC_PROTOCOL,
        "official_motionfix_claim": False,
        "target_length_protocol": (
            "equal_length_only"
            if args.protocol == EQUAL_PROTOCOL
            else "gt_requested_target_length"
        ),
        "editing_contact_protocol_id": (
            EDITING_CONTACT_CFG_PROTOCOL
            if args.cfg_apply_contacts
            else EDITING_CONTACT_PROTOCOL
        ),
        "generate_control_contact_protocol_id": GENERATE_CONTROL_CONTACT_PROTOCOL,
        "diagnostic_exact_clamp_protocol": EXACT_CLAMP_PROTOCOL,
        "checkpoint_sha256": str(checkpoint_sha256),
        "plan_sha256": str(plan_sha256),
        "source_copy_unequal_protocol": PHYSICAL_TIMEWARP_PROTOCOL,
        "direct_k273_interpolation": False,
        "counterfactual_manifest": counterfactual_identity,
        "edit_control_subtypes": list(EDIT_CONTROL_SUBTYPES),
        "same_noise_across_systems": True,
    }


def _protocol_manifest_from_preflight(
    preflight: dict[str, Any],
    *,
    preflight_path: str | Path,
    preflight_sha256: str,
) -> dict[str, Any]:
    """Build the only protocol envelope permitted for a frozen preflight."""

    run_contract = preflight.get("run_contract")
    plan = preflight.get("plan")
    checkpoint = preflight.get("checkpoint")
    expected_case_manifest = preflight.get("expected_case_manifest")
    train_manifest = preflight.get("train_manifest")
    fail_closed_pairs = preflight.get("counterfactual_source_fail_closed_pairs")
    if (
        not isinstance(run_contract, dict)
        or preflight.get("run_contract_sha256") != canonical_sha(run_contract)
        or not isinstance(plan, dict)
        or not isinstance(checkpoint, dict)
        or not isinstance(expected_case_manifest, dict)
        or not isinstance(train_manifest, dict)
        or not isinstance(fail_closed_pairs, list)
        or any(not isinstance(value, str) for value in fail_closed_pairs)
        or fail_closed_pairs != sorted(set(fail_closed_pairs))
        or run_contract.get("checkpoint_sha256") != checkpoint.get("sha256")
        or run_contract.get("plan_sha256") != plan.get("case_keys_sha256")
    ):
        raise RuntimeError("MotionFix preflight cannot define a protocol envelope")
    return {
        "format": PROTOCOL_FORMAT,
        **run_contract,
        "run_contract_sha256": canonical_sha(run_contract),
        "pair_count": int(plan["pair_count"]),
        "case_count": int(plan["case_count"]),
        "preflight_manifest": str(Path(preflight_path).expanduser().resolve()),
        "preflight_sha256": str(preflight_sha256).lower(),
        "train_manifest": train_manifest,
        "counterfactual_source_fail_closed_pairs": fail_closed_pairs,
        "expected_case_manifest": expected_case_manifest,
    }


def _validate_protocol_run_contract(
    preflight: dict[str, Any],
    protocol: dict[str, Any],
    args: argparse.Namespace,
    *,
    preflight_path: str | Path,
    preflight_sha256: str,
) -> None:
    run_contract = preflight.get("run_contract")
    if (
        not isinstance(run_contract, dict)
        or preflight.get("run_contract_sha256") != canonical_sha(run_contract)
        or protocol.get("run_contract_sha256") != canonical_sha(run_contract)
        or not _json_equal_strict(
            {key: protocol.get(key) for key in run_contract}, run_contract
        )
    ):
        raise RuntimeError("MotionFix protocol differs from the frozen run contract")
    argument_identity = _evaluation_argument_identity(args)
    if not _json_equal_strict(
        {key: run_contract.get(key) for key in argument_identity},
        argument_identity,
    ):
        raise RuntimeError("MotionFix aggregate arguments differ from the frozen run contract")
    expected = _protocol_manifest_from_preflight(
        preflight,
        preflight_path=preflight_path,
        preflight_sha256=preflight_sha256,
    )
    if not _json_equal_strict(protocol, expected):
        differing = sorted(
            key
            for key in set(protocol) | set(expected)
            if key not in protocol
            or key not in expected
            or not _json_equal_strict(protocol[key], expected[key])
        )
        raise RuntimeError(
            f"MotionFix protocol envelope differs from preflight: {differing[:8]}"
        )


def _asset_contract(ref: dict[str, Any]) -> dict[str, Any]:
    required = {
        "path",
        "sha256",
        "frames",
        "feature_dim",
        "fps",
        "representation_version",
    }
    missing = sorted(required - set(ref))
    if missing:
        raise RuntimeError(f"Evaluation asset contract is incomplete: {missing}")
    return {
        "path": str(Path(ref["path"]).expanduser().resolve()),
        "sha256": str(ref["sha256"]),
        "frames": int(ref["frames"]),
        "feature_dim": int(ref["feature_dim"]),
        "fps": float(ref["fps"]),
        "representation_version": str(ref["representation_version"]),
    }


def _metric_schema(values: dict[str, Any]) -> dict[str, str]:
    schema: dict[str, str] = {}
    for name, value in values.items():
        if value is None:
            schema[name] = "none"
        elif isinstance(value, bool):
            raise RuntimeError(f"Boolean metric is not allowed: {name}")
        elif isinstance(value, (int, np.integer)):
            schema[name] = "integer"
        elif isinstance(value, (float, np.floating)):
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite metric in expected schema: {name}")
            schema[name] = "float"
        elif isinstance(value, str):
            schema[name] = f"literal:{value}"
        else:
            raise RuntimeError(f"Unsupported metric value for {name}: {type(value)}")
    return dict(sorted(schema.items()))


def _control_metrics_with_safety(
    control_metrics: dict[str, float | int],
    internal_metrics: dict[str, float | int | None],
) -> dict[str, float | int]:
    output = dict(control_metrics)
    for name in CONTROL_SAFETY_METRICS:
        value = internal_metrics.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"Missing numeric controlled safety metric: {name}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError(f"Non-finite controlled safety metric: {name}")
        if name in output and float(output[name]) != numeric:
            raise RuntimeError(f"Control/internal safety metric mismatch: {name}")
        output[name] = numeric
    return output


def _validate_metric_schema(
    values: Any, expected_schema: dict[str, str], *, label: str
) -> None:
    if not isinstance(values, dict) or set(values) != set(expected_schema):
        actual = sorted(values) if isinstance(values, dict) else type(values).__name__
        raise RuntimeError(
            f"{label} schema mismatch: expected={sorted(expected_schema)}, actual={actual}"
        )
    actual_schema = _metric_schema(values)
    if actual_schema != expected_schema:
        raise RuntimeError(f"{label} value-type/nullability schema changed")


def _control_identity(
    subtype: str,
    constraint: Any,
    mask: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, Any]:
    return {
        "subtype": str(subtype),
        "components": constraint.components,
        "motion_mask_fraction": float(mask.float().mean().item()),
        "motion_mask_sha256": tensor_sha256(mask.contiguous()),
        "observed_motion_sha256": tensor_sha256(observed.contiguous()),
    }


def _expected_sampling_protocol(
    case: EditCase,
    args: argparse.Namespace,
    source_copy_info: dict[str, Any],
    *,
    unified_273_flow: bool = False,
    source_fusion_mode: str = "additive",
    text_global_conditioning: str = "pooled_adaln",
    text_fusion_mode: str = "f00",
) -> dict[str, Any]:
    if case.system == SYSTEM_SOURCE_COPY:
        return {"system": SYSTEM_SOURCE_COPY, "source_copy": source_copy_info}
    standalone = case.system == SYSTEM_STANDALONE_CONTROL
    diagnostic = case.system == SYSTEM_INSTRUCTION_ONLY
    has_control = case.system in CONTROL_SYSTEMS
    route = "generate" if standalone else "edit"
    if standalone:
        branch_names = ["joint", "text", "control", "empty"]
    elif has_control:
        branch_names = ["empty", "source", "edit", "all"]
    else:
        branch_names = ["empty", "source", "joint"]
    unified = bool(unified_273_flow)
    protocol = {
        "sampling_protocol_version": SAMPLING_PROTOCOL_VERSION,
        "contact_protocol_version": (
            UNIFIED_CONTACT_PROTOCOL_VERSION
            if unified
            else CONTACT_PROTOCOL_VERSION
        ),
        "contact_protocol": (
            UNIFIED_273_CONTACT_PROTOCOL
            if unified
            else LEGACY_SPLIT_CONTACT_PROTOCOL
        ),
        "route": route,
        "has_control": has_control,
        "source_fusion_mode": str(source_fusion_mode),
        "text_global_conditioning": str(text_global_conditioning),
        "text_fusion_mode": str(text_fusion_mode),
        "branch_names": branch_names,
        "ode_steps": int(args.num_steps),
        "ode_state_persistent_clamp": False,
        "text_cfg_scale": float(args.generate_text_cfg_scale if standalone else 2.0),
        "source_cfg_scale": float(args.source_cfg_scale),
        "edit_cfg_scale": float(args.edit_cfg_scale),
        "edit_source_baseline": (
            "learned" if standalone or diagnostic else str(args.edit_source_baseline)
        ),
        "control_cfg_scale": float(args.control_cfg_scale),
        "cfg_apply_contacts": (
            True
            if unified
            else bool(
                args.generate_cfg_apply_contacts
                if standalone
                else args.cfg_apply_contacts
            )
        ),
        "contact_init": "unified_273d_state" if unified else str(args.contact_init),
        "contact_feedback": "ode_273d" if unified else str(args.contact_feedback),
        "initial_noise_source": (
            "provided_unified_273d" if unified else "legacy_split_state"
        ),
        "initial_noise_protocol": (
            "unified_gaussian_273d_v1"
            if unified
            else "legacy_split_contact_aux_v1"
        ),
        "primary_output": "raw_pre_exact_clamp",
        "diagnostic_allow_source_absent_edit": diagnostic,
        "evaluator_contact_protocol_id": (
            UNIFIED_CONTACT_PROTOCOL_VERSION
            if unified
            else (
                GENERATE_CONTROL_CONTACT_PROTOCOL
                if standalone
                else (
                    EDITING_CONTACT_CFG_PROTOCOL
                    if args.cfg_apply_contacts
                    else EDITING_CONTACT_PROTOCOL
                )
            )
        ),
    }
    if has_control:
        protocol["diagnostic_exact_clamp_protocol"] = EXACT_CLAMP_PROTOCOL
    return protocol


def _sampling_protocol_identity(protocol: Any) -> dict[str, Any]:
    if not isinstance(protocol, dict):
        raise RuntimeError("Sampling protocol is not a mapping")
    return {
        key: value
        for key, value in protocol.items()
        if key
        not in {
            "initial_continuous_noise_sha256",
            "initial_contact_noise_sha256",
            "initial_unified_noise_sha256",
        }
    }


def _expected_case_rows(
    rows: list[dict[str, Any]],
    plan: list[EditCase],
    *,
    args: argparse.Namespace,
    counterfactual_rows: dict[str, dict[str, Any]] | None,
    train_seen_index: dict[str, set[Any]],
    output_dir: Path,
    unified_273_flow: bool = False,
    source_fusion_mode: str = "additive",
    text_global_conditioning: str = "pooled_adaln",
    text_fusion_mode: str = "f00",
) -> list[dict[str, Any]]:
    pair_cache: dict[int, dict[str, Any]] = {}
    control_cache: dict[tuple[int, str], tuple[dict[str, Any], dict[str, str]]] = {}
    output: list[dict[str, Any]] = []
    for case_index, case in enumerate(plan):
        row = rows[case.row_index]
        pair = row["pair"]
        shard_id = case.row_index % int(args.num_shards)
        cached = pair_cache.get(case.row_index)
        if cached is None:
            target_frames = int(pair["target_frames"])
            source_aligned_native, source_copy_info = _aligned_source(row, target_frames)
            target_native = _load_k273(
                row["target_motion"]["k273_asset"],
                expected_sha256=str(row["target_motion"]["k273_asset"]["sha256"]),
            )
            source_native = _load_k273(
                row["source_motion"]["k273_asset"],
                expected_sha256=str(row["source_motion"]["k273_asset"]["sha256"]),
            )
            phi = _phi(case.sample_seed)
            _, source_delta = _to_gauge(source_native, phi)
            source_aligned, aligned_delta = _to_gauge(source_aligned_native, phi)
            target, target_delta = _to_gauge(target_native, phi)
            identity_metrics = evaluate_motionfix_internal_case(
                target, target, source_aligned
            )
            if str(pair["length_relation"]) != "equal":
                for metric_name in EQUAL_ONLY_REGIONAL_METRICS:
                    identity_metrics[metric_name] = None
            cached = {
                "target": target,
                "source_aligned": source_aligned,
                "source_copy_info": source_copy_info,
                "metric_schema": _metric_schema(identity_metrics),
                "aligned_source_tensor_sha256": tensor_sha256(
                    source_aligned.contiguous()
                ),
                "phi": phi,
                "source_delta": source_delta,
                "target_delta": target_delta,
                "aligned_delta": aligned_delta,
            }
            pair_cache[case.row_index] = cached

        case_counterfactual = (
            None
            if counterfactual_rows is None
            else counterfactual_rows[case.pair_id]
        )
        source_ref, model_text, condition_provenance = _resolve_model_inputs(
            row, case, case_counterfactual
        )
        conditioning_source = (
            _asset_contract(source_ref)
            if condition_provenance["source_condition_present"]
            else None
        )
        if bool(condition_provenance["source_condition_present"]) != (
            conditioning_source is not None
        ):
            raise RuntimeError("Conditioning-source provenance is internally inconsistent")
        if source_ref is None:
            model_source_delta = None
        else:
            source_for_model_native = _load_k273(
                source_ref,
                expected_sha256=str(source_ref["sha256"]),
            )
            _, model_source_delta = _to_gauge(
                source_for_model_native, cached["phi"]
            )

        control_identity = None
        control_metric_schema = None
        if case.system in CONTROL_SYSTEMS:
            cache_key = (case.row_index, str(case.control_subtype))
            cached_control = control_cache.get(cache_key)
            if cached_control is None:
                constraint, evaluator = _compile_edit_control(
                    cached["target"],
                    case,
                    max_sparse_keyframes=args.max_sparse_keyframes,
                )
                control_identity = _control_identity(
                    str(case.control_subtype),
                    constraint,
                    constraint.motion_mask,
                    constraint.observed_motion,
                )
                control_metric_schema = _metric_schema(
                    _control_metrics_with_safety(
                        evaluator(cached["target"], cached["target"], constraint),
                        identity_metrics,
                    )
                )
                cached_control = (control_identity, control_metric_schema)
                control_cache[cache_key] = cached_control
            control_identity, control_metric_schema = cached_control

        reference_hy201 = (
            _asset_contract(row["source_motion"]["hy201_asset"])
            if str(pair["length_relation"]) != "equal"
            else None
        )
        regional_protocol = (
            "equal_length_frozen_regions_v1"
            if str(pair["length_relation"]) == "equal"
            else "not_applicable_unequal_length"
        )
        aligned_source_path = (
            output_dir
            / "reference_sources"
            / f"shard_{shard_id:02d}"
            / f"pair_{case.pair_id}.npy"
        ).resolve()
        motion_output_path = (
            output_dir
            / "motion_outputs"
            / f"shard_{shard_id:02d}"
            / f"{case.key}.npz"
        ).resolve()
        scientific_identity = {
            "case_uid": str(row["uid"]),
            "pair": {
                "source_frames": int(pair["source_frames"]),
                "target_frames": int(pair["target_frames"]),
                "length_relation": str(pair["length_relation"]),
                "target_length_protocol": (
                    "equal_length_only"
                    if args.protocol == EQUAL_PROTOCOL
                    else "gt_requested_target_length"
                ),
                "frame_policy_id": str(pair["frame_policy_id"]),
                "shared_world_frame": bool(pair["shared_world_frame"]),
            },
            "assets": {
                "reference_source_k273": _asset_contract(
                    row["source_motion"]["k273_asset"]
                ),
                "reference_source_hy201": reference_hy201,
                "target_k273": _asset_contract(row["target_motion"]["k273_asset"]),
                "conditioning_source_k273": conditioning_source,
            },
            "condition_provenance": condition_provenance,
            "gauge": {
                "output_gauge_phi": cached["phi"],
                "source_applied_yaw_delta": cached["source_delta"],
                "model_source_applied_yaw_delta": model_source_delta,
                "target_applied_yaw_delta": cached["target_delta"],
                "aligned_source_applied_yaw_delta": cached["aligned_delta"],
            },
            "instruction": {
                "original_sha256": hashlib.sha256(
                    str(row["texts"][0]["value"]).encode("utf-8")
                ).hexdigest(),
                "model_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
                "model_text": model_text,
            },
            "seen_strata": _seen_strata(row, train_seen_index),
            "source_copy_protocol": cached["source_copy_info"],
            "regional_metric_protocol": regional_protocol,
            "sampling_protocol": _expected_sampling_protocol(
                case,
                args,
                cached["source_copy_info"],
                unified_273_flow=unified_273_flow,
                source_fusion_mode=source_fusion_mode,
                text_global_conditioning=text_global_conditioning,
                text_fusion_mode=text_fusion_mode,
            ),
            "metric_schema": cached["metric_schema"],
            "control": control_identity,
            "control_metric_schema": (
                {
                    pass_name: control_metric_schema
                    for pass_name in (
                        "generated_raw",
                        "diagnostic_exact_clamp",
                        "ground_truth",
                    )
                }
                if control_metric_schema is not None
                else None
            ),
            "aligned_reference_source": {
                "format": ALIGNED_SOURCE_FORMAT,
                "path": str(aligned_source_path),
                "shape": [int(pair["target_frames"]), DIM_HY273],
                "tensor_sha256": cached["aligned_source_tensor_sha256"],
            },
        }
        output.append(
            {
                "case_index": case_index,
                "case_key": case.key,
                "row_index": case.row_index,
                "pair_id": case.pair_id,
                "system": case.system,
                "sample_seed": case.sample_seed,
                "control_subtype": case.control_subtype,
                "expected_shard_id": shard_id,
                "expected_motion_output_path": str(motion_output_path),
                "scientific_identity": scientific_identity,
            }
        )
    return output


def _ensure_expected_case_manifest(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = output_dir / "expected_cases.jsonl"
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("Existing MotionFix expected-case manifest changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("Concurrent expected-case manifest creation differed")
    return {
        "format": EXPECTED_CASE_FORMAT,
        "path": str(path),
        "sha256": dataset_sha256_file(path),
        "count": len(rows),
    }


def _load_expected_case_manifest(identity: dict[str, Any]) -> list[dict[str, Any]]:
    if identity.get("format") != EXPECTED_CASE_FORMAT:
        raise RuntimeError("Unknown MotionFix expected-case manifest format")
    path = Path(identity["path"]).expanduser().resolve()
    if dataset_sha256_file(path) != identity.get("sha256"):
        raise RuntimeError("MotionFix expected-case manifest SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != int(identity.get("count", -1)):
        raise RuntimeError("MotionFix expected-case manifest count mismatch")
    keys = [row["case_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("MotionFix expected-case manifest has duplicate keys")
    return rows


def _counterfactual_manifest_identity(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    systems = set(parse_csv(args.systems))
    if not systems & COUNTERFACTUAL_SYSTEMS:
        return None
    path = Path(args.counterfactual_manifest).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Counterfactual manifest is missing: {path}")
    sha = dataset_sha256_file(path)
    expected = str(args.counterfactual_manifest_sha256).lower()
    if expected and sha != expected:
        raise RuntimeError("Counterfactual manifest SHA256 mismatch")
    summary_path = Path(args.counterfactual_summary).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Counterfactual summary is missing: {summary_path}")
    summary_sha = dataset_sha256_file(summary_path)
    expected_summary_sha = str(args.counterfactual_summary_sha256).lower()
    if expected_summary_sha and summary_sha != expected_summary_sha:
        raise RuntimeError("Counterfactual summary SHA256 mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("format") != COUNTERFACTUAL_FORMAT
        or summary.get("status") != "passed"
        or summary.get("jsonl_sha256") != sha
        or int(summary.get("seed", -1)) != int(args.seed)
        or summary.get("text_normalizer")
        != "unicode_preserving_whitespace_collapse_v1"
    ):
        raise RuntimeError("Counterfactual summary contract mismatch")
    builder_path = ROOT / "tools" / "build_motionfix_edit_counterfactual_manifest.py"
    if summary.get("builder_code_sha256") != dataset_sha256_file(builder_path):
        raise RuntimeError("Counterfactual builder code SHA256 mismatch")
    return {
        **_file_stat(path),
        "sha256": sha,
        "format": COUNTERFACTUAL_FORMAT,
        "summary": {
            **_file_stat(summary_path),
            "sha256": summary_sha,
            "builder_code_sha256": str(summary["builder_code_sha256"]),
            "text_normalizer": str(summary["text_normalizer"]),
            "rows_sha256": str(summary["rows_sha256"]),
        },
    }


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    stat_before = _file_stat(checkpoint_path)
    checkpoint_sha = dataset_sha256_file(checkpoint_path)
    stat_after = _file_stat(checkpoint_path)
    if stat_before != stat_after:
        raise RuntimeError("Checkpoint changed while hashing")
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256.lower():
        raise RuntimeError("Checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Checkpoint has no resolved multitask config")
    validate_frozen_contract(config)
    unified_273_flow = uses_unified_273_flow(contact_protocol_for_config(config))
    source_fusion_mode = source_fusion_mode_from_checkpoint(checkpoint)
    text_global_conditioning = text_global_conditioning_from_checkpoint(
        checkpoint
    )
    text_fusion_mode = text_fusion_mode_from_checkpoint(checkpoint)
    training_code_identity = _validate_checkpoint_code_identity(
        checkpoint, allow_code_drift=bool(args.research_allow_code_drift)
    )
    asset_identity = validate_assets(config)
    if checkpoint.get("asset_identity") != asset_identity:
        raise RuntimeError("Checkpoint/data asset identity mismatch")
    all_rows = load_motionfix_rows(args.manifest, args.protocol)
    counterfactual_identity = _counterfactual_manifest_identity(args)
    counterfactual_rows = (
        load_counterfactual_rows(args.counterfactual_manifest, all_rows)
        if counterfactual_identity is not None
        else None
    )
    selected = selected_protocol_rows(all_rows, args)
    plan = build_plan(
        selected,
        systems=parse_csv(args.systems),
        seed=args.seed,
        counterfactual_rows=counterfactual_rows,
    )
    train_seen_index = _load_train_seen_index(args.train_manifest)
    output_dir = Path(args.output_dir).expanduser().resolve()
    expected_rows = _expected_case_rows(
        selected,
        plan,
        args=args,
        counterfactual_rows=counterfactual_rows,
        train_seen_index=train_seen_index,
        output_dir=output_dir,
        unified_273_flow=unified_273_flow,
        source_fusion_mode=source_fusion_mode,
        text_global_conditioning=text_global_conditioning,
        text_fusion_mode=text_fusion_mode,
    )
    expected_case_manifest = _ensure_expected_case_manifest(
        output_dir, expected_rows
    )
    frozen_plan = plan_identity(selected, plan, args)
    run_contract = _run_contract(
        args,
        checkpoint_sha256=checkpoint_sha,
        plan_sha256=str(frozen_plan["case_keys_sha256"]),
        counterfactual_identity=counterfactual_identity,
    )
    payload = {
        "format": PREFLIGHT_FORMAT,
        "status": "passed",
        "host": socket.gethostname(),
        "checkpoint": _checkpoint_preflight_identity(
            checkpoint_path,
            checkpoint,
            checkpoint_sha256=checkpoint_sha,
            file_stat=stat_after,
        ),
        "manifest": {
            **_file_stat(Path(args.manifest).expanduser().resolve()),
            "sha256": dataset_sha256_file(args.manifest),
        },
        "train_manifest": {
            **_file_stat(Path(args.train_manifest).expanduser().resolve()),
            "sha256": dataset_sha256_file(args.train_manifest),
        },
        "counterfactual_manifest": counterfactual_identity,
        "counterfactual_source_fail_closed_pairs": sorted(
            pair_id
            for pair_id, row in (counterfactual_rows or {}).items()
            if row["source_shuffle"]["status"] != "matched"
        ),
        "asset_identity_sha256": canonical_sha(asset_identity),
        "checkpoint_training_code_identity": training_code_identity,
        "hytext_profile_identity": _hytext_profile_identity(config),
        "code": evaluation_code_identity(),
        "plan": frozen_plan,
        "run_contract": run_contract,
        "run_contract_sha256": canonical_sha(run_contract),
        "expected_case_manifest": expected_case_manifest,
        "metric_protocol": INTERNAL_METRIC_PROTOCOL,
        "source_copy_unequal_protocol": PHYSICAL_TIMEWARP_PROTOCOL,
        "official_motionfix_claim": False,
    }
    del checkpoint
    return payload


def load_preflight(
    args: argparse.Namespace,
    *,
    checkpoint: dict[str, Any],
    rows: list[dict[str, Any]],
    plan: list[EditCase],
) -> tuple[Path, dict[str, Any], str]:
    path = (
        Path(args.preflight_manifest).expanduser().resolve()
        if args.preflight_manifest
        else Path(args.output_dir).expanduser().resolve() / "preflight_manifest.json"
    )
    if not path.is_file() or not args.preflight_sha256:
        raise RuntimeError("Shard launch requires a pinned preflight manifest and SHA256")
    sha = dataset_sha256_file(path)
    if sha != args.preflight_sha256.lower():
        raise RuntimeError("MotionFix preflight SHA256 mismatch")
    payload = _load_json_object_strict(path)
    if payload.get("format") != PREFLIGHT_FORMAT or payload.get("status") != "passed":
        raise RuntimeError("Invalid MotionFix preflight")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    current_checkpoint = _checkpoint_preflight_identity(
        checkpoint_path,
        checkpoint,
        checkpoint_sha256=payload["checkpoint"]["sha256"],
    )
    if current_checkpoint != payload["checkpoint"]:
        raise RuntimeError("Checkpoint changed after MotionFix preflight")
    manifest_path = Path(args.manifest).expanduser().resolve()
    current_manifest = {
        **_file_stat(manifest_path),
        "sha256": dataset_sha256_file(manifest_path),
    }
    if current_manifest != payload["manifest"]:
        raise RuntimeError("MotionFix manifest changed after preflight")
    train_manifest_path = Path(args.train_manifest).expanduser().resolve()
    current_train_manifest = {
        **_file_stat(train_manifest_path),
        "sha256": dataset_sha256_file(train_manifest_path),
    }
    if current_train_manifest != payload["train_manifest"]:
        raise RuntimeError("MotionFix train manifest changed after preflight")
    counterfactual_identity = _counterfactual_manifest_identity(args)
    if payload.get("counterfactual_manifest") != counterfactual_identity:
        raise RuntimeError("MotionFix counterfactual manifest changed after preflight")
    if payload.get("code") != evaluation_code_identity():
        raise RuntimeError("MotionFix evaluator code changed after preflight")
    if payload.get("checkpoint_training_code_identity") != (
        _validate_checkpoint_code_identity(
            checkpoint, allow_code_drift=bool(args.research_allow_code_drift)
        )
    ):
        raise RuntimeError("MotionFix checkpoint training-code identity changed")
    if payload.get("hytext_profile_identity") != _hytext_profile_identity(
        checkpoint["config"]
    ):
        raise RuntimeError("MotionFix HYText profile changed after preflight")
    runtime_plan = plan_identity(rows, plan, args)
    if payload.get("plan") != runtime_plan:
        raise RuntimeError("MotionFix evaluation plan changed after preflight")
    runtime_contract = _run_contract(
        args,
        checkpoint_sha256=str(payload["checkpoint"]["sha256"]),
        plan_sha256=str(runtime_plan["case_keys_sha256"]),
        counterfactual_identity=counterfactual_identity,
    )
    if (
        payload.get("run_contract") != runtime_contract
        or payload.get("run_contract_sha256") != canonical_sha(runtime_contract)
    ):
        raise RuntimeError("MotionFix run contract changed after preflight")
    expected_rows = _load_expected_case_manifest(payload["expected_case_manifest"])
    expected_core = [
        (
            row["case_key"],
            int(row["row_index"]),
            row["pair_id"],
            row["system"],
            int(row["sample_seed"]),
            row.get("control_subtype"),
        )
        for row in expected_rows
    ]
    runtime_core = [
        (
            case.key,
            case.row_index,
            case.pair_id,
            case.system,
            case.sample_seed,
            case.control_subtype,
        )
        for case in plan
    ]
    if expected_core != runtime_core:
        raise RuntimeError("Expected-case manifest differs from the runtime plan")
    verification_dir = Path(args.output_dir).expanduser().resolve()
    verification_dir.mkdir(parents=True, exist_ok=True)
    lock_path = verification_dir / ".checkpoint_content_verification.lock"
    stamp_path = verification_dir / "checkpoint_content_verification.json"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        expected_stamp = {
            "format": "hy273_checkpoint_content_verification_v1",
            "preflight_sha256": sha,
            "checkpoint": payload["checkpoint"],
        }
        if stamp_path.is_file():
            current_stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            if current_stamp != expected_stamp:
                raise RuntimeError("Checkpoint verification stamp mismatch")
        else:
            stat_before = _file_stat(checkpoint_path)
            checkpoint_sha = dataset_sha256_file(checkpoint_path)
            stat_after = _file_stat(checkpoint_path)
            if stat_before != stat_after or checkpoint_sha != payload["checkpoint"]["sha256"]:
                raise RuntimeError("Checkpoint changed during shard-launch verification")
            _atomic_json(stamp_path, expected_stamp)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return path, payload, sha


def _load_k273(
    ref: dict[str, Any], *, expected_sha256: str = ""
) -> torch.Tensor:
    if expected_sha256:
        _verify_asset_payload(
            str(Path(ref["path"]).expanduser().resolve()),
            str(expected_sha256),
            int(ref["frames"]),
            DIM_HY273,
        )
    value = np.load(ref["path"])
    expected = (int(ref["frames"]), DIM_HY273)
    if value.shape != expected or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid K273 evaluation asset {ref['path']}: {value.shape}")
    tensor = torch.from_numpy(value.astype(np.float32, copy=False)).clone()
    contacts = tensor[:, CONTACT_SLICE]
    if not bool(((contacts == 0.0) | (contacts == 1.0)).all()):
        raise RuntimeError(f"K273 evaluation contacts are not binary: {ref['path']}")
    return tensor


def _to_gauge(motion: torch.Tensor, phi: float) -> tuple[torch.Tensor, float]:
    shifted = root_origin_shift(motion)
    heading = shifted[0, HEADING_SLICE]
    current = torch.atan2(heading[1], heading[0])
    delta = torch.as_tensor(phi, dtype=shifted.dtype) - current
    return apply_yaw_rotation(shifted, delta), float(delta.item())


def _phi(seed: int) -> float:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return float((torch.rand((), generator=generator) * 2.0 * math.pi - math.pi).item())


def _aligned_source(row: dict[str, Any], target_frames: int) -> tuple[torch.Tensor, dict[str, Any]]:
    source_frames = int(row["pair"]["source_frames"])
    if source_frames == target_frames:
        asset = row["source_motion"]["k273_asset"]
        return _load_k273(asset, expected_sha256=str(asset["sha256"])), {
            "protocol": "native_equal_length_k273_identity_v1",
            "source_frames": source_frames,
            "target_frames": target_frames,
        }
    hy201_asset = row["source_motion"]["hy201_asset"]
    _verify_asset_payload(
        str(Path(hy201_asset["path"]).expanduser().resolve()),
        str(hy201_asset["sha256"]),
        int(hy201_asset["frames"]),
        int(hy201_asset["feature_dim"]),
    )
    hy201 = np.load(hy201_asset["path"])
    return physical_timewarp_hy201_to_k273(hy201, target_frames)


def _model_and_normalizer(
    checkpoint: dict[str, Any], weight_source: str, device: torch.device
) -> tuple[torch.nn.Module, Any]:
    model = create_model_from_checkpoint(checkpoint).to(device)
    weights = checkpoint["ema"] if weight_source == "ema" else checkpoint["model"]
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model, normalizer_from_checkpoint(checkpoint, device)


def _resolve_model_inputs(
    row: dict[str, Any],
    case: EditCase,
    counterfactual: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    original_source = row["source_motion"]["k273_asset"]
    original_text = str(row["texts"][0]["value"])
    source_ref = original_source
    text = original_text
    provenance: dict[str, Any] = {
        "source_case_uid": str(row["uid"]),
        "source_motion_uid": str(row["source_motion"]["motion_uid"]),
        "source_base_motion_id": str(row["source_motion"]["base_motion_id"]),
        "source_sha256": str(original_source["sha256"]),
        "source_role": "original_pair_source",
        "instruction_text_id": str(row["texts"][0]["text_id"]),
        "instruction_sha256": hashlib.sha256(original_text.encode("utf-8")).hexdigest(),
        "source_condition_present": True,
        "instruction_condition_present": True,
        "source_shuffled": False,
        "instruction_shuffled": False,
        "instruction_dropped": False,
    }
    if case.system == SYSTEM_SOURCE_COPY:
        source_ref = None
        text = ""
        provenance.update(
            {
                "source_case_uid": None,
                "source_motion_uid": None,
                "source_base_motion_id": None,
                "source_sha256": None,
                "source_role": "absent_non_model_baseline",
                "instruction_text_id": None,
                "instruction_sha256": None,
                "source_condition_present": False,
                "instruction_condition_present": False,
                "instruction_dropped": True,
                "comparator_role": "source_copy_baseline",
            }
        )
    elif case.system == SYSTEM_STANDALONE_CONTROL:
        source_ref = None
        text = ""
        provenance.update(
            {
                "source_case_uid": None,
                "source_motion_uid": None,
                "source_base_motion_id": None,
                "source_sha256": None,
                "source_role": "absent_generate_control",
                "instruction_text_id": None,
                "instruction_sha256": None,
                "source_condition_present": False,
                "instruction_condition_present": False,
                "instruction_dropped": True,
                "comparator_role": "matched_generate_control",
            }
        )
    elif case.system == SYSTEM_INSTRUCTION_ONLY:
        source_ref = None
        provenance.update(
            {
                "source_case_uid": None,
                "source_motion_uid": None,
                "source_base_motion_id": None,
                "source_sha256": None,
                "source_role": "absent_instruction_only_diagnostic",
                "source_condition_present": False,
                "comparator_role": "source_absent_edit_diagnostic",
            }
        )
    elif case.system == SYSTEM_SOURCE_SHUFFLE:
        if counterfactual is None:
            raise RuntimeError("Source shuffle requested without counterfactual data")
        shuffled = counterfactual["source_shuffle"]
        if shuffled["status"] != "matched":
            raise RuntimeError("A fail-closed source-shuffle case entered the plan")
        source_ref = {
            "path": shuffled["donor_source_path"],
            "frames": int(shuffled["donor_source_len"]),
            "sha256": str(shuffled["donor_source_sha256"]),
            "feature_dim": DIM_HY273,
            "fps": 30.0,
            "representation_version": "kimodo273_smplx22_v1",
        }
        provenance.update(
            {
                "source_case_uid": str(shuffled["donor_uid"]),
                "source_motion_uid": str(shuffled["donor_source_motion_uid"]),
                "source_base_motion_id": str(
                    shuffled["donor_source_base_motion_id"]
                ),
                "source_sha256": str(shuffled["donor_source_sha256"]),
                "source_role": "shuffled_donor_source",
                "source_shuffled": True,
                "target_to_source_time_map_policy": str(
                    shuffled["target_to_source_time_map_policy"]
                ),
            }
        )
    elif case.system == SYSTEM_INSTRUCTION_SHUFFLE:
        if counterfactual is None:
            raise RuntimeError("Instruction shuffle requested without counterfactual data")
        shuffled = counterfactual["instruction_shuffle"]
        text = str(shuffled["donor_instruction"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest != shuffled["donor_instruction_sha256"]:
            raise RuntimeError("Shuffled instruction SHA256 mismatch")
        provenance.update(
            {
                "instruction_case_uid": str(shuffled["donor_uid"]),
                "instruction_text_id": str(shuffled["donor_text_id"]),
                "instruction_sha256": digest,
                "instruction_shuffled": True,
                "encoding_profile": str(shuffled["encoding_profile"]),
            }
        )
    elif case.system == SYSTEM_INSTRUCTION_DROP:
        text = ""
        provenance.update(
            {
                "instruction_text_id": None,
                "instruction_sha256": None,
                "instruction_condition_present": False,
                "instruction_dropped": True,
            }
        )
    return source_ref, text, provenance


def _compile_edit_control(
    target: torch.Tensor,
    case: EditCase,
    *,
    max_sparse_keyframes: int,
) -> tuple[Any, Any]:
    subtype = case.control_subtype
    if not subtype:
        raise ValueError("Edit-control case has no control subtype")
    if subtype in V5_CONTACT_SUBTYPES:
        return (
            compile_kimodo_contact_constraint(
                target,
                subtype,
                seed=case.sample_seed,
                max_sparse_keyframes=max_sparse_keyframes,
            ),
            evaluate_kimodo_contact_case,
        )
    return (
        compile_kimodo_constraint(
            target,
            subtype,
            seed=case.sample_seed,
            max_sparse_keyframes=max_sparse_keyframes,
        ),
        evaluate_kimodo_constraint_case,
    )


def _generate_model_case(
    *,
    model: torch.nn.Module,
    normalizer: Any,
    source: torch.Tensor,
    source_anchor: torch.Tensor | None,
    text: str,
    target_frames: int,
    phi: float,
    case: EditCase,
    observed: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    lengths = torch.tensor([target_frames], dtype=torch.long)
    gauge = torch.tensor([[math.cos(phi), math.sin(phi)]], dtype=torch.float32)
    diagnostic = case.system == SYSTEM_INSTRUCTION_ONLY
    standalone_control = case.system == SYSTEM_STANDALONE_CONTROL
    if standalone_control:
        condition = make_absent_condition(
            batch_size=1,
            target_frames=target_frames,
            target_lengths=lengths,
            capability=CapabilityId.KIMODO_CONTROL,
        )
        condition = replace(condition, frame_gauge_dir=gauge)
        condition.validate()
    elif diagnostic:
        condition = make_instruction_only_edit_diagnostic_condition(
            target_lengths=lengths,
            target_frames=target_frames,
            frame_gauge_dir=gauge,
        )
    else:
        condition = make_edit_condition(
            source,
            target_lengths=lengths,
            target_frames=target_frames,
            frame_gauge_dir=gauge,
            capability=(
                CapabilityId.MOTION_EDIT_CONTROL
                if case.system == SYSTEM_EDIT_CONTROL
                else CapabilityId.MOTION_EDIT
            ),
        )
    if observed.shape != (target_frames, DIM_HY273) or mask.shape != (
        target_frames,
        DIM_HY273,
    ):
        raise ValueError("Edit evaluation control tensors have invalid shapes")
    model_parameter = next(model.parameters())
    sample_device = model_parameter.device
    sample_dtype = model_parameter.dtype
    generator = torch.Generator(device=sample_device).manual_seed(case.sample_seed)
    unified_273_flow = bool(normalizer.normalize_contacts)
    initial_unified_noise = None
    if unified_273_flow:
        initial_unified_noise = torch.randn(
            1,
            target_frames,
            DIM_HY273,
            device=sample_device,
            dtype=sample_dtype,
            generator=generator,
        )
        initial_continuous_noise = initial_unified_noise[..., :CONT_DIM]
        initial_contact_noise = initial_unified_noise[..., CONTACT_SLICE]
    else:
        initial_continuous_noise = torch.randn(
            1,
            target_frames,
            CONT_DIM,
            device=sample_device,
            dtype=sample_dtype,
            generator=generator,
        )
        if args.contact_init == "random":
            initial_contact_noise = torch.rand(
                1,
                target_frames,
                4,
                device=sample_device,
                dtype=sample_dtype,
                generator=generator,
            )
        elif args.contact_init == "half":
            initial_contact_noise = torch.full(
                (1, target_frames, 4),
                0.5,
                device=sample_device,
                dtype=sample_dtype,
            )
        else:
            initial_contact_noise = torch.zeros(
                1,
                target_frames,
                4,
                device=sample_device,
                dtype=sample_dtype,
            )
    output = sample_hy273_multitask_ode(
        model,
        normalizer,
        condition,
        [text],
        observed.unsqueeze(0),
        mask.unsqueeze(0),
        num_steps=args.num_steps,
        text_cfg_scale=(
            args.generate_text_cfg_scale if standalone_control else None
        ),
        source_cfg_scale=args.source_cfg_scale,
        edit_cfg_scale=args.edit_cfg_scale,
        control_cfg_scale=args.control_cfg_scale,
        contact_init=args.contact_init,
        contact_feedback=args.contact_feedback,
        cfg_apply_contacts=(
            args.generate_cfg_apply_contacts
            if standalone_control
            else args.cfg_apply_contacts
        ),
        initial_unified_noise=initial_unified_noise,
        initial_continuous_noise=(
            None if unified_273_flow else initial_continuous_noise
        ),
        initial_contact_noise=None if unified_273_flow else initial_contact_noise,
        diagnostic_allow_source_absent_edit=diagnostic,
        edit_source_baseline=(
            "learned"
            if standalone_control or diagnostic
            else args.edit_source_baseline
        ),
        edit_source_anchor_physical=(
            None if source_anchor is None else source_anchor.unsqueeze(0)
        ),
    )
    protocol = dict(output.protocol)
    protocol["evaluator_contact_protocol_id"] = (
        UNIFIED_CONTACT_PROTOCOL_VERSION
        if unified_273_flow
        else (
            GENERATE_CONTROL_CONTACT_PROTOCOL
            if standalone_control
            else (
                EDITING_CONTACT_CFG_PROTOCOL
                if args.cfg_apply_contacts
                else EDITING_CONTACT_PROTOCOL
            )
        )
    )
    protocol["initial_continuous_noise_sha256"] = tensor_sha256(
        initial_continuous_noise
    )
    protocol["initial_contact_noise_sha256"] = tensor_sha256(
        initial_contact_noise
    )
    protocol["initial_noise_protocol"] = (
        "unified_gaussian_273d_v1"
        if unified_273_flow
        else "legacy_split_contact_aux_v1"
    )
    if initial_unified_noise is not None:
        protocol["initial_unified_noise_sha256"] = tensor_sha256(
            initial_unified_noise
        )
    return (
        output.raw_motion[0, :target_frames].cpu(),
        output.exact_clamped_motion[0, :target_frames].cpu(),
        protocol,
    )


def _validate_exact_overwrite_contract(
    raw: torch.Tensor,
    exact: torch.Tensor,
    *,
    observed: torch.Tensor | None,
    mask: torch.Tensor | None,
    label: str,
) -> None:
    if raw.shape != exact.shape or raw.dtype != exact.dtype or raw.device != exact.device:
        raise RuntimeError(f"{label}: raw/exact tensor contract mismatch")
    if observed is None or mask is None:
        if observed is not None or mask is not None:
            raise RuntimeError(f"{label}: incomplete exact-overwrite control tensors")
        if not torch.equal(exact, raw):
            raise RuntimeError(f"{label}: uncontrolled exact output differs from raw")
        return
    if (
        observed.shape != raw.shape
        or observed.dtype != raw.dtype
        or observed.device != raw.device
        or mask.shape != raw.shape
        or mask.dtype != torch.bool
        or mask.device != raw.device
    ):
        raise RuntimeError(f"{label}: exact-overwrite control tensor contract mismatch")
    expected = torch.where(mask, observed, raw)
    if not torch.equal(exact, expected):
        controlled_mismatches = int(torch.count_nonzero(exact[mask] != observed[mask]))
        uncontrolled_mismatches = int(torch.count_nonzero(exact[~mask] != raw[~mask]))
        raise RuntimeError(
            f"{label}: exact output is not where(mask, observed, raw): "
            f"controlled_mismatches={controlled_mismatches}, "
            f"uncontrolled_mismatches={uncontrolled_mismatches}"
        )


def _physical_exact_clamp(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the diagnostic clamp after returning to physical K273 space."""

    if prediction.shape != observed.shape or mask.shape != prediction.shape:
        raise ValueError("Physical exact-clamp tensors must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("Physical exact-clamp mask must be bool")
    result = torch.where(mask, observed, prediction)
    _validate_exact_overwrite_contract(
        prediction,
        result,
        observed=observed,
        mask=mask,
        label="post-denormalization physical clamp",
    )
    return result


def _write_aligned_reference_source(
    output_dir: Path,
    case: EditCase,
    *,
    shard_id: int,
    source_aligned: torch.Tensor,
    overwrite: bool,
) -> dict[str, Any]:
    path = (
        output_dir
        / "reference_sources"
        / f"shard_{int(shard_id):02d}"
        / f"pair_{case.pair_id}.npy"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_tensor_sha = tensor_sha256(source_aligned.contiguous())
    if overwrite or not path.is_file():
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.npy")
        np.save(
            temporary,
            source_aligned.detach().cpu().float().contiguous().numpy(),
        )
        os.replace(temporary, path)
    loaded = np.load(path, allow_pickle=False)
    expected_shape = tuple(source_aligned.shape)
    if (
        loaded.shape != expected_shape
        or loaded.dtype != np.float32
        or not np.isfinite(loaded).all()
        or tensor_sha256(torch.from_numpy(loaded).contiguous()) != expected_tensor_sha
    ):
        raise RuntimeError(f"Evaluator-aligned source artifact is invalid: {path}")
    return {
        "format": ALIGNED_SOURCE_FORMAT,
        "path": str(path.resolve()),
        "sha256": dataset_sha256_file(path),
        "shape": list(expected_shape),
        "tensor_sha256": expected_tensor_sha,
    }


def _write_motion_output(
    output_dir: Path,
    case: EditCase,
    *,
    shard_id: int,
    prediction: torch.Tensor,
    exact_prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    overwrite: bool,
) -> dict[str, Any]:
    path = (
        output_dir
        / "motion_outputs"
        / f"shard_{int(shard_id):02d}"
        / f"{case.key}.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_exact_overwrite_contract(
        prediction,
        exact_prediction,
        observed=observed,
        mask=mask,
        label=f"MotionFix output {case.key}",
    )
    arrays = {
        "raw": prediction.detach().cpu().float().contiguous().numpy(),
        "exact": exact_prediction.detach().cpu().float().contiguous().numpy(),
    }
    if bool(mask.any()):
        arrays["observed"] = observed.detach().cpu().float().contiguous().numpy()
        arrays["mask"] = mask.detach().cpu().bool().contiguous().numpy()
    if overwrite or not path.is_file():
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(arrays):
            raise RuntimeError(f"Stale MotionFix output schema at {path}")
        loaded = {name: payload[name].copy() for name in payload.files}
    for name, expected_value in arrays.items():
        actual_value = loaded[name]
        if (
            actual_value.shape != expected_value.shape
            or actual_value.dtype != expected_value.dtype
            or not np.array_equal(actual_value, expected_value)
        ):
            raise RuntimeError(f"Stale MotionFix output tensor {name} at {path}")
    return {
        "path": str(path.resolve()),
        "sha256": dataset_sha256_file(path),
        "format": MOTION_OUTPUT_FORMAT,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": tensor_sha256(
                    torch.from_numpy(value.copy()).contiguous()
                ),
            }
            for name, value in sorted(loaded.items())
        },
    }


def _record_scientific_identity(record: dict[str, Any]) -> dict[str, Any]:
    control_metrics = record.get("control_metrics")
    control_metric_schema = (
        {
            pass_name: _metric_schema(control_metrics[pass_name])
            for pass_name in (
                "generated_raw",
                "diagnostic_exact_clamp",
                "ground_truth",
            )
        }
        if isinstance(control_metrics, dict)
        else None
    )
    aligned = record.get("aligned_reference_source")
    aligned_identity = None
    if isinstance(aligned, dict):
        aligned_identity = {
            key: aligned[key]
            for key in ("format", "path", "shape", "tensor_sha256")
        }
    return {
        "case_uid": record.get("case_uid"),
        "pair": {
            "source_frames": record.get("source_frames"),
            "target_frames": record.get("target_frames"),
            "length_relation": record.get("length_relation"),
            "target_length_protocol": record.get("target_length_protocol"),
            "frame_policy_id": record.get("frame_policy_id"),
            "shared_world_frame": record.get("shared_world_frame"),
        },
        "assets": record.get("assets"),
        "condition_provenance": record.get("condition_provenance"),
        "gauge": {
            "output_gauge_phi": record.get("output_gauge_phi"),
            "source_applied_yaw_delta": record.get("source_applied_yaw_delta"),
            "model_source_applied_yaw_delta": record.get(
                "model_source_applied_yaw_delta"
            ),
            "target_applied_yaw_delta": record.get("target_applied_yaw_delta"),
            "aligned_source_applied_yaw_delta": record.get(
                "aligned_source_applied_yaw_delta"
            ),
        },
        "instruction": {
            "original_sha256": hashlib.sha256(
                str(record.get("instruction", "")).encode("utf-8")
            ).hexdigest(),
            "model_sha256": hashlib.sha256(
                str(record.get("model_instruction", "")).encode("utf-8")
            ).hexdigest(),
            "model_text": record.get("model_instruction"),
        },
        "seen_strata": record.get("seen_strata"),
        "source_copy_protocol": record.get("source_copy_protocol"),
        "regional_metric_protocol": record.get("regional_metric_protocol"),
        "sampling_protocol": _sampling_protocol_identity(
            record.get("sampling_protocol")
        ),
        "metric_schema": _metric_schema(record.get("metrics", {})),
        "control": record.get("control"),
        "control_metric_schema": control_metric_schema,
        "aligned_reference_source": aligned_identity,
    }


def _validate_scientific_identity(
    record: dict[str, Any], expected: dict[str, Any]
) -> None:
    actual = _record_scientific_identity(record)
    if actual != expected:
        raise RuntimeError(
            f"MotionFix scientific identity mismatch: {record.get('case_key', '<unknown>')}"
        )


def _validate_shard_ownership(
    record: dict[str, Any], expected: dict[str, Any], shard_id: int
) -> None:
    if (
        int(record.get("shard_id", -1)) != int(shard_id)
        or int(expected.get("expected_shard_id", -1)) != int(shard_id)
    ):
        raise RuntimeError(
            f"MotionFix shard {shard_id} contains a foreign case: "
            f"{record.get('case_key', '<unknown>')}"
        )


@lru_cache(maxsize=None)
def _verify_asset_payload(
    path_string: str, sha256: str, frames: int, feature_dim: int
) -> None:
    path = Path(path_string).expanduser().resolve()
    if not path.is_file() or dataset_sha256_file(path) != sha256:
        raise RuntimeError(f"Evaluation asset payload changed: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != (int(frames), int(feature_dim)):
        raise RuntimeError(f"Evaluation asset shape changed: {path} {value.shape}")


def _validate_asset_identity_tree(assets: Any, *, case_key: str) -> None:
    if not isinstance(assets, dict):
        raise RuntimeError(f"MotionFix case has no asset identity: {case_key}")
    for name, asset in assets.items():
        if asset is None:
            continue
        if not isinstance(asset, dict):
            raise RuntimeError(f"Invalid asset identity {name}: {case_key}")
        _verify_asset_payload(
            str(asset["path"]),
            str(asset["sha256"]),
            int(asset["frames"]),
            int(asset["feature_dim"]),
        )


def _validate_metric_values(
    recorded: Any, recomputed: dict[str, Any], *, label: str
) -> None:
    if recorded != recomputed:
        differing = []
        if isinstance(recorded, dict):
            differing = sorted(
                key
                for key in set(recorded) | set(recomputed)
                if recorded.get(key) != recomputed.get(key)
            )[:8]
        raise RuntimeError(f"{label} values differ from saved motion: {differing}")


def _replay_record_metrics(
    record: dict[str, Any],
    *,
    raw: torch.Tensor,
    exact: torch.Tensor,
    aligned_source: torch.Tensor,
    observed: torch.Tensor | None,
    mask: torch.Tensor | None,
    max_sparse_keyframes: int,
) -> None:
    controlled = record.get("control") is not None
    if controlled:
        if observed is None or mask is None:
            raise RuntimeError(
                f"Controlled metric replay lacks tensors: {record['case_key']}"
            )
        _validate_exact_overwrite_contract(
            raw,
            exact,
            observed=observed,
            mask=mask,
            label=f"Metric replay {record['case_key']}",
        )
    else:
        if observed is not None or mask is not None:
            raise RuntimeError(
                f"Uncontrolled metric replay has control tensors: {record['case_key']}"
            )
        _validate_exact_overwrite_contract(
            raw,
            exact,
            observed=None,
            mask=None,
            label=f"Metric replay {record['case_key']}",
        )
    target_asset = record["assets"]["target_k273"]
    target_native = _load_k273(
        target_asset, expected_sha256=str(target_asset["sha256"])
    )
    target, _ = _to_gauge(target_native, float(record["output_gauge_phi"]))
    if target.shape != raw.shape or aligned_source.shape != raw.shape:
        raise RuntimeError(f"Metric replay shape mismatch: {record['case_key']}")
    main_metrics = evaluate_motionfix_internal_case(raw, target, aligned_source)
    if str(record["length_relation"]) != "equal":
        for metric_name in EQUAL_ONLY_REGIONAL_METRICS:
            main_metrics[metric_name] = None
    _validate_metric_values(
        record.get("metrics"), main_metrics, label=f"{record['case_key']}.metrics"
    )

    if not controlled:
        if record.get("control_metrics") is not None:
            raise RuntimeError(f"Uncontrolled metric replay has control tensors: {record['case_key']}")
        return
    assert observed is not None and mask is not None
    if (
        tensor_sha256(mask.contiguous())
        != record["control"]["motion_mask_sha256"]
        or tensor_sha256(observed.contiguous())
        != record["control"]["observed_motion_sha256"]
    ):
        raise RuntimeError(f"Control payload hash mismatch: {record['case_key']}")
    case = EditCase(
        row_index=int(record["row_index"]),
        pair_id=str(record["pair_id"]),
        system=str(record["system"]),
        sample_seed=int(record["sample_seed"]),
        control_subtype=str(record["control_subtype"]),
    )
    constraint, evaluator = _compile_edit_control(
        target, case, max_sparse_keyframes=int(max_sparse_keyframes)
    )
    if not torch.equal(mask, constraint.motion_mask) or not torch.equal(
        observed, constraint.observed_motion
    ):
        raise RuntimeError(f"Saved control tensors differ from compiler: {record['case_key']}")
    if _control_identity(
        str(case.control_subtype), constraint, mask, observed
    ) != record.get("control"):
        raise RuntimeError(f"Recompiled control identity differs: {record['case_key']}")
    exact_internal_metrics = evaluate_motionfix_internal_case(
        exact, target, aligned_source
    )
    ground_truth_internal_metrics = evaluate_motionfix_internal_case(
        target, target, aligned_source
    )
    replayed = {
        "generated_raw": _control_metrics_with_safety(
            evaluator(raw, target, constraint), main_metrics
        ),
        "diagnostic_exact_clamp": _control_metrics_with_safety(
            evaluator(exact, target, constraint), exact_internal_metrics
        ),
        "ground_truth": _control_metrics_with_safety(
            evaluator(target, target, constraint), ground_truth_internal_metrics
        ),
    }
    for pass_name, values in replayed.items():
        _validate_metric_values(
            record.get("control_metrics", {}).get(pass_name),
            values,
            label=f"{record['case_key']}.control_metrics.{pass_name}",
        )


def _validate_case_record_identity(
    record: dict[str, Any],
    expected: dict[str, Any],
    *,
    protocol_sha256: str,
    preflight_sha256: str,
    verify_motion_output: bool,
    max_sparse_keyframes: int,
) -> None:
    for field in (
        "case_key",
        "row_index",
        "pair_id",
        "system",
        "sample_seed",
        "control_subtype",
    ):
        if record.get(field) != expected.get(field):
            raise RuntimeError(
                f"MotionFix case identity mismatch for {expected['case_key']}: {field}"
            )
    if record.get("status") != "ok":
        raise RuntimeError(f"MotionFix case is not successful: {expected['case_key']}")
    if int(record.get("shard_id", -1)) != int(expected["expected_shard_id"]):
        raise RuntimeError(f"MotionFix case entered the wrong shard: {expected['case_key']}")
    if record.get("protocol_manifest_sha256") != protocol_sha256:
        raise RuntimeError(f"MotionFix case protocol SHA mismatch: {expected['case_key']}")
    if record.get("preflight_sha256") != preflight_sha256:
        raise RuntimeError(f"MotionFix case preflight SHA mismatch: {expected['case_key']}")
    _validate_scientific_identity(record, expected["scientific_identity"])
    actual_scientific_identity = _record_scientific_identity(record)
    _validate_asset_identity_tree(
        actual_scientific_identity["assets"], case_key=expected["case_key"]
    )
    _validate_metric_schema(
        record.get("metrics"),
        expected["scientific_identity"]["metric_schema"],
        label=f"{expected['case_key']}.metrics",
    )
    expected_control_schema = expected["scientific_identity"][
        "control_metric_schema"
    ]
    if expected_control_schema is None:
        if record.get("control") is not None or record.get("control_metrics") is not None:
            raise RuntimeError(
                f"Uncontrolled case carries control data: {expected['case_key']}"
            )
    else:
        if not isinstance(record.get("control_metrics"), dict):
            raise RuntimeError(f"Controlled case lacks metrics: {expected['case_key']}")
        for pass_name, schema in expected_control_schema.items():
            _validate_metric_schema(
                record["control_metrics"].get(pass_name),
                schema,
                label=f"{expected['case_key']}.control_metrics.{pass_name}",
            )
    aligned = record.get("aligned_reference_source")
    if not isinstance(aligned, dict) or aligned.get("format") != ALIGNED_SOURCE_FORMAT:
        raise RuntimeError(
            f"MotionFix case has no evaluator-aligned source: {expected['case_key']}"
        )
    aligned_path = Path(aligned["path"]).expanduser().resolve()
    if not aligned_path.is_file() or dataset_sha256_file(aligned_path) != aligned.get(
        "sha256"
    ):
        raise RuntimeError(f"Aligned source payload changed: {aligned_path}")
    aligned_value = np.load(aligned_path, mmap_mode="r", allow_pickle=False)
    if (
        list(aligned_value.shape) != list(aligned["shape"])
        or aligned_value.dtype != np.float32
        or tensor_sha256(torch.from_numpy(np.asarray(aligned_value).copy()).contiguous())
        != aligned["tensor_sha256"]
    ):
        raise RuntimeError(f"Aligned source payload is invalid: {aligned_path}")
    output = record.get("motion_output")
    if not verify_motion_output:
        if output is not None:
            raise RuntimeError("Motion outputs exist despite a disabled output protocol")
        return
    if not isinstance(output, dict) or output.get("format") != MOTION_OUTPUT_FORMAT:
        raise RuntimeError(f"MotionFix case has no valid motion output: {expected['case_key']}")
    output_path = Path(output["path"]).expanduser().resolve()
    if output_path != Path(expected["expected_motion_output_path"]).resolve():
        raise RuntimeError(
            f"MotionFix motion output entered the wrong shard path: {expected['case_key']}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"MotionFix motion output is missing: {output_path}")
    if dataset_sha256_file(output_path) != output.get("sha256"):
        raise RuntimeError(f"MotionFix motion output SHA mismatch: {output_path}")
    with np.load(output_path, allow_pickle=False) as payload:
        expected_shape = (int(record["target_frames"]), DIM_HY273)
        if set(payload.files) not in ({"raw", "exact"}, {"raw", "exact", "observed", "mask"}):
            raise RuntimeError(f"MotionFix motion output schema mismatch: {output_path}")
        loaded = {name: payload[name].copy() for name in payload.files}
        raw = loaded["raw"]
        exact = loaded["exact"]
        if (
            raw.shape != expected_shape
            or exact.shape != expected_shape
            or raw.dtype != np.float32
            or exact.dtype != np.float32
            or not np.isfinite(raw).all()
            or not np.isfinite(exact).all()
        ):
            raise RuntimeError(f"MotionFix motion output payload is invalid: {output_path}")
        array_identity = {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": tensor_sha256(
                    torch.from_numpy(value.copy()).contiguous()
                ),
            }
            for name, value in sorted(loaded.items())
        }
        if output.get("arrays") != array_identity:
            raise RuntimeError(f"MotionFix output tensor identity mismatch: {output_path}")
        if record.get("control") is not None:
            if "observed" not in loaded or "mask" not in loaded:
                raise RuntimeError(f"Controlled output lacks observations: {output_path}")
            observed = loaded["observed"]
            mask = loaded["mask"]
            if (
                observed.shape != expected_shape
                or mask.shape != expected_shape
                or observed.dtype != np.float32
                or mask.dtype != np.bool_
                or not np.isfinite(observed).all()
            ):
                raise RuntimeError(f"Controlled output payload is invalid: {output_path}")
            _validate_exact_overwrite_contract(
                torch.from_numpy(raw).contiguous(),
                torch.from_numpy(exact).contiguous(),
                observed=torch.from_numpy(observed).contiguous(),
                mask=torch.from_numpy(mask).contiguous(),
                label=f"Controlled output {output_path}",
            )
        elif set(payload.files) != {"raw", "exact"}:
            raise RuntimeError(f"Uncontrolled output carries a control payload: {output_path}")
        else:
            _validate_exact_overwrite_contract(
                torch.from_numpy(raw).contiguous(),
                torch.from_numpy(exact).contiguous(),
                observed=None,
                mask=None,
                label=f"Uncontrolled output {output_path}",
            )
    _replay_record_metrics(
        record,
        raw=torch.from_numpy(raw).contiguous(),
        exact=torch.from_numpy(exact).contiguous(),
        aligned_source=torch.from_numpy(np.asarray(aligned_value).copy()).contiguous(),
        observed=(
            torch.from_numpy(loaded["observed"]).contiguous()
            if "observed" in loaded
            else None
        ),
        mask=(
            torch.from_numpy(loaded["mask"]).contiguous()
            if "mask" in loaded
            else None
        ),
        max_sparse_keyframes=int(max_sparse_keyframes),
    )


def run_shard(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0,num_shards)")
    checkpoint = torch.load(
        Path(args.checkpoint).expanduser().resolve(),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    all_rows = load_motionfix_rows(args.manifest, args.protocol)
    counterfactual_identity = _counterfactual_manifest_identity(args)
    counterfactual_rows = (
        load_counterfactual_rows(args.counterfactual_manifest, all_rows)
        if counterfactual_identity is not None
        else None
    )
    rows = selected_protocol_rows(all_rows, args)
    plan = build_plan(
        rows,
        systems=parse_csv(args.systems),
        seed=args.seed,
        counterfactual_rows=counterfactual_rows,
    )
    train_seen_index = _load_train_seen_index(args.train_manifest)
    preflight_path, preflight, preflight_sha = load_preflight(
        args, checkpoint=checkpoint, rows=rows, plan=plan
    )
    systems = set(parse_csv(args.systems))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = normalizer = None
    if systems & MODEL_SYSTEMS:
        model, normalizer = _model_and_normalizer(checkpoint, args.weight_source, device)
        if bool(normalizer.normalize_contacts) != bool(
            preflight["checkpoint"]["unified_273_flow"]
        ):
            raise RuntimeError(
                "MotionFix runtime contact protocol differs from preflight"
            )
    del checkpoint

    output_dir = Path(args.output_dir).expanduser().resolve()
    expected_case_manifest = preflight.get("expected_case_manifest")
    if not isinstance(expected_case_manifest, dict):
        raise RuntimeError("Preflight has no expected-case manifest")
    expected_rows = _load_expected_case_manifest(expected_case_manifest)
    expected_by_key = {row["case_key"]: row for row in expected_rows}
    if len(expected_rows) != len(plan):
        raise RuntimeError("Preflight expected-case count differs from runtime plan")
    run_contract = preflight.get("run_contract")
    if not isinstance(run_contract, dict):
        raise RuntimeError("Preflight has no frozen run contract")
    protocol = _protocol_manifest_from_preflight(
        preflight,
        preflight_path=preflight_path,
        preflight_sha256=preflight_sha,
    )
    _validate_protocol_run_contract(
        preflight,
        protocol,
        args,
        preflight_path=preflight_path,
        preflight_sha256=preflight_sha,
    )
    protocol_path = output_dir / "protocol_manifest.json"
    if protocol_path.is_file():
        if not _json_equal_strict(_load_json_object_strict(protocol_path), protocol):
            raise RuntimeError("Existing MotionFix protocol differs from requested run")
    else:
        _atomic_json(protocol_path, protocol)
    protocol_sha = dataset_sha256_file(protocol_path)
    shard_path = output_dir / "shards" / f"shard_{args.shard_id:02d}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        shard_path.unlink(missing_ok=True)
    existing = []
    if shard_path.is_file():
        existing = [json.loads(line) for line in shard_path.read_text().splitlines() if line]
    for record in existing:
        expected = expected_by_key.get(record.get("case_key"))
        if expected is None:
            raise RuntimeError("Existing shard contains an unplanned MotionFix case")
        _validate_shard_ownership(record, expected, args.shard_id)
        _validate_case_record_identity(
            record,
            expected,
            protocol_sha256=protocol_sha,
            preflight_sha256=preflight_sha,
            verify_motion_output=bool(args.save_motion_outputs),
            max_sparse_keyframes=int(args.max_sparse_keyframes),
        )
    by_key = {row["case_key"]: row for row in existing}
    if len(by_key) != len(existing):
        raise RuntimeError("Existing MotionFix shard contains duplicate case keys")
    shard_pair_indices = {
        index for index in range(len(rows)) if index % args.num_shards == args.shard_id
    }
    shard_cases = [case for case in plan if case.row_index in shard_pair_indices]
    started = time.perf_counter()
    new_count = 0
    with shard_path.open("a", encoding="utf-8") as writer:
        for case in shard_cases:
            if case.key in by_key:
                continue
            row = rows[case.row_index]
            pair = row["pair"]
            target_frames = int(pair["target_frames"])
            source_asset = row["source_motion"]["k273_asset"]
            target_asset = row["target_motion"]["k273_asset"]
            source_native = _load_k273(
                source_asset, expected_sha256=str(source_asset["sha256"])
            )
            target_native = _load_k273(
                target_asset, expected_sha256=str(target_asset["sha256"])
            )
            source_aligned_native, source_copy_info = _aligned_source(row, target_frames)
            phi = _phi(case.sample_seed)
            source, source_delta = _to_gauge(source_native, phi)
            target, target_delta = _to_gauge(target_native, phi)
            source_aligned, aligned_delta = _to_gauge(source_aligned_native, phi)
            original_text = str(row["texts"][0]["value"])
            case_counterfactual = (
                None
                if counterfactual_rows is None
                else counterfactual_rows[case.pair_id]
            )
            source_ref, model_text, condition_provenance = _resolve_model_inputs(
                row, case, case_counterfactual
            )
            if source_ref is None:
                if condition_provenance.get("source_condition_present") is not False:
                    raise RuntimeError("Absent source reference has present provenance")
                source_for_model = torch.empty(0, DIM_HY273)
                model_source_delta = None
                conditioning_source_asset = None
            else:
                if condition_provenance.get("source_condition_present") is not True:
                    raise RuntimeError("Present source reference has absent provenance")
                source_for_model_native = _load_k273(
                    source_ref,
                    expected_sha256=str(source_ref.get("sha256", ""))
                    or str(condition_provenance.get("source_sha256", "")),
                )
                source_for_model, model_source_delta = _to_gauge(
                    source_for_model_native, phi
                )
                conditioning_source_asset = _asset_contract(source_ref)
            observed = torch.zeros(target_frames, DIM_HY273)
            mask = torch.zeros_like(observed, dtype=torch.bool)
            control_constraint = control_evaluator = None
            if case.system in CONTROL_SYSTEMS:
                control_constraint, control_evaluator = _compile_edit_control(
                    target,
                    case,
                    max_sparse_keyframes=args.max_sparse_keyframes,
                )
                observed = control_constraint.observed_motion
                mask = control_constraint.motion_mask
            if case.system == SYSTEM_SOURCE_COPY:
                prediction = source_aligned
                exact_prediction = prediction
                sampling_protocol = {
                    "system": SYSTEM_SOURCE_COPY,
                    "source_copy": source_copy_info,
                }
            else:
                if model is None or normalizer is None:
                    raise AssertionError("Model system requested without a loaded model")
                prediction, exact_prediction, sampling_protocol = _generate_model_case(
                    model=model,
                    normalizer=normalizer,
                    source=source_for_model,
                    source_anchor=(
                        None
                        if args.edit_source_baseline != "exact"
                        or case.system
                        in {SYSTEM_STANDALONE_CONTROL, SYSTEM_INSTRUCTION_ONLY}
                        else (
                            source_aligned
                            if case.system != SYSTEM_SOURCE_SHUFFLE
                            else (
                                source_for_model
                                if source_for_model.shape[0] == target_frames
                                else None
                            )
                        )
                    ),
                    text=model_text,
                    target_frames=target_frames,
                    phi=phi,
                    case=case,
                    observed=observed,
                    mask=mask,
                    args=args,
                )
            if control_constraint is not None:
                exact_prediction = _physical_exact_clamp(
                    prediction,
                    control_constraint.observed_motion,
                    control_constraint.motion_mask,
                )
                sampling_protocol = dict(sampling_protocol)
                sampling_protocol[
                    "diagnostic_exact_clamp_protocol"
                ] = EXACT_CLAMP_PROTOCOL
            metrics = evaluate_motionfix_internal_case(
                prediction, target, source_aligned
            )
            regional_metric_protocol = "equal_length_frozen_regions_v1"
            if str(pair["length_relation"]) != "equal":
                for metric_name in EQUAL_ONLY_REGIONAL_METRICS:
                    metrics[metric_name] = None
                regional_metric_protocol = "not_applicable_unequal_length"
            control_metrics = None
            control_identity = None
            if control_constraint is not None:
                if control_evaluator is None:
                    raise AssertionError("Control evaluator was not resolved")
                if not torch.equal(
                    exact_prediction[mask], control_constraint.observed_motion[mask]
                ):
                    raise RuntimeError("Exact edit-control output violates observations")
                exact_internal_metrics = evaluate_motionfix_internal_case(
                    exact_prediction, target, source_aligned
                )
                ground_truth_internal_metrics = evaluate_motionfix_internal_case(
                    target, target, source_aligned
                )
                control_metrics = {
                    "generated_raw": _control_metrics_with_safety(
                        control_evaluator(prediction, target, control_constraint),
                        metrics,
                    ),
                    "diagnostic_exact_clamp": _control_metrics_with_safety(
                        control_evaluator(
                            exact_prediction, target, control_constraint
                        ),
                        exact_internal_metrics,
                    ),
                    "ground_truth": _control_metrics_with_safety(
                        control_evaluator(target, target, control_constraint),
                        ground_truth_internal_metrics,
                    ),
                }
                control_identity = _control_identity(
                    str(case.control_subtype),
                    control_constraint,
                    mask,
                    observed,
                )
            aligned_reference_source = _write_aligned_reference_source(
                output_dir,
                case,
                shard_id=args.shard_id,
                source_aligned=source_aligned,
                overwrite=args.overwrite,
            )
            motion_output = (
                _write_motion_output(
                    output_dir,
                    case,
                    shard_id=args.shard_id,
                    prediction=prediction,
                    exact_prediction=exact_prediction,
                    observed=observed,
                    mask=mask,
                    overwrite=args.overwrite,
                )
                if args.save_motion_outputs
                else None
            )
            record = {
                "status": "ok",
                "case_uid": str(row["uid"]),
                "case_key": case.key,
                "row_index": case.row_index,
                "pair_id": case.pair_id,
                "system": case.system,
                "sample_seed": case.sample_seed,
                "control_subtype": case.control_subtype,
                "instruction": original_text,
                "model_instruction": model_text,
                "condition_provenance": condition_provenance,
                "seen_strata": _seen_strata(row, train_seen_index),
                "source_frames": int(pair["source_frames"]),
                "target_frames": target_frames,
                "length_relation": str(pair["length_relation"]),
                "target_length_protocol": protocol["target_length_protocol"],
                "frame_policy_id": str(pair["frame_policy_id"]),
                "shared_world_frame": bool(pair["shared_world_frame"]),
                "output_gauge_phi": phi,
                "source_applied_yaw_delta": source_delta,
                "model_source_applied_yaw_delta": model_source_delta,
                "target_applied_yaw_delta": target_delta,
                "aligned_source_applied_yaw_delta": aligned_delta,
                "source_copy_protocol": source_copy_info,
                "sampling_protocol": sampling_protocol,
                "assets": {
                    "reference_source_k273": _asset_contract(
                        row["source_motion"]["k273_asset"]
                    ),
                    "reference_source_hy201": (
                        _asset_contract(row["source_motion"]["hy201_asset"])
                        if str(pair["length_relation"]) != "equal"
                        else None
                    ),
                    "target_k273": _asset_contract(
                        row["target_motion"]["k273_asset"]
                    ),
                    "conditioning_source_k273": conditioning_source_asset,
                },
                "aligned_reference_source": aligned_reference_source,
                "source_k273_path": str(row["source_motion"]["k273_asset"]["path"]),
                "target_k273_path": str(row["target_motion"]["k273_asset"]["path"]),
                "motion_output": motion_output,
                "metrics": metrics,
                "regional_metric_protocol": regional_metric_protocol,
                "control": control_identity,
                "control_metrics": control_metrics,
                "protocol_manifest_sha256": protocol_sha,
                "preflight_sha256": preflight_sha,
                "shard_id": args.shard_id,
            }
            _validate_case_record_identity(
                record,
                expected_by_key[case.key],
                protocol_sha256=protocol_sha,
                preflight_sha256=preflight_sha,
                verify_motion_output=bool(args.save_motion_outputs),
                max_sparse_keyframes=int(args.max_sparse_keyframes),
            )
            writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
            by_key[case.key] = record
            new_count += 1
            print(
                json.dumps(
                    {
                        "shard_id": args.shard_id,
                        "completed": len(by_key),
                        "total": len(shard_cases),
                        "cases_per_second": new_count
                        / max(time.perf_counter() - started, 1e-9),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _atomic_json(
        output_dir / "shards" / f"shard_{args.shard_id:02d}_summary.json",
        {
            "shard_id": args.shard_id,
            "expected": len(shard_cases),
            "records": len(by_key),
            "complete": len(by_key) == len(shard_cases),
            "protocol_manifest_sha256": protocol_sha,
        },
    )


def _bootstrap_mean_ci(
    values: list[float], *, seed: int, samples: int
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty paired statistic")
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    chunk = 256
    for start in range(0, int(samples), chunk):
        count = min(chunk, int(samples) - start)
        indices = generator.integers(0, array.size, size=(count, array.size))
        means[start : start + count] = array[indices].mean(axis=1)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }


def _stat_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"motionfix-stat:{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _paired_counterfactual_summary(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    all_comparison_systems = (
        SYSTEM_SOURCE_SHUFFLE,
        SYSTEM_INSTRUCTION_SHUFFLE,
        SYSTEM_INSTRUCTION_DROP,
    )
    protocol_systems = protocol.get("systems")
    requested_comparison_systems = (
        all_comparison_systems
        if protocol_systems is None
        else tuple(
            system
            for system in all_comparison_systems
            if system in set(protocol_systems)
        )
    )
    by_pair_system = {
        (record["pair_id"], record["system"]): record for record in records
    }
    correct_records = [
        record for record in records if record["system"] == SYSTEM_MODEL
    ]
    output: dict[str, Any] = {
        "primary_system": SYSTEM_MODEL,
        "same_noise_required": True,
        "source_shuffle_fail_closed_pairs": list(
            protocol.get("counterfactual_source_fail_closed_pairs", [])
        ),
        "requested_comparison_systems": list(requested_comparison_systems),
        "subsets": {},
    }
    subsets: list[tuple[str, Any]] = [
        ("all", lambda record: True),
        ("equal_length", lambda record: record["length_relation"] == "equal"),
        ("unequal_length", lambda record: record["length_relation"] != "equal"),
    ]
    if correct_records:
        for field_name in sorted(correct_records[0]["seen_strata"]):
            values = sorted(
                {record["seen_strata"][field_name] for record in correct_records},
                key=str,
            )
            for value in values:
                label = str(value).lower() if isinstance(value, bool) else str(value)
                subsets.append(
                    (
                        f"seen_strata.{field_name}={label}",
                        lambda record, field_name=field_name, value=value: record[
                            "seen_strata"
                        ][field_name]
                        == value,
                    )
                )
    for subset_name, predicate in subsets:
        primary = [record for record in correct_records if predicate(record)]
        if not primary:
            continue
        edit_gain = []
        copy_ratio = []
        for record in primary:
            metrics = record["metrics"]
            source_delta = float(metrics["source_target_position_delta_m"])
            target_error = float(metrics["global_joint_target_error_m"])
            source_error = float(metrics["global_joint_source_error_m"])
            edit_gain.append(
                (source_delta - target_error) / max(source_delta, 1.0e-4)
            )
            copy_ratio.append(source_error / max(source_delta, 1.0e-4))
        subset: dict[str, Any] = {
            "correct_model_case_count": len(primary),
            "edit_gain": _bootstrap_mean_ci(
                edit_gain,
                seed=_stat_seed(args.seed, f"{subset_name}:edit_gain"),
                samples=args.bootstrap_samples,
            ),
            "copy_ratio_median": float(np.median(copy_ratio)),
        }
        comparisons = {}
        for counterfactual_system in requested_comparison_systems:
            position_degradation = []
            rotation_degradation = []
            missing_pairs = []
            for correct in primary:
                other = by_pair_system.get(
                    (correct["pair_id"], counterfactual_system)
                )
                if other is None:
                    missing_pairs.append(correct["pair_id"])
                    continue
                if int(other["sample_seed"]) != int(correct["sample_seed"]):
                    raise RuntimeError("Counterfactual systems did not share noise seed")
                for noise_name in (
                    "initial_continuous_noise_sha256",
                    "initial_contact_noise_sha256",
                ):
                    if correct["sampling_protocol"].get(noise_name) != other[
                        "sampling_protocol"
                    ].get(noise_name):
                        raise RuntimeError(
                            "Counterfactual systems did not share actual initial noise: "
                            f"pair={correct['pair_id']} system={counterfactual_system} "
                            f"field={noise_name}"
                        )
                position_degradation.append(
                    float(other["metrics"]["global_joint_target_error_m"])
                    - float(correct["metrics"]["global_joint_target_error_m"])
                )
                rotation_degradation.append(
                    float(other["metrics"]["global_rotation_target_error_deg"])
                    - float(correct["metrics"]["global_rotation_target_error_deg"])
                )
            if position_degradation:
                comparisons[counterfactual_system] = {
                    "target_position_error_degradation_m": _bootstrap_mean_ci(
                        position_degradation,
                        seed=_stat_seed(
                            args.seed,
                            f"{subset_name}:{counterfactual_system}:position",
                        ),
                        samples=args.bootstrap_samples,
                    ),
                    "target_rotation_error_degradation_deg": _bootstrap_mean_ci(
                        rotation_degradation,
                        seed=_stat_seed(
                            args.seed,
                            f"{subset_name}:{counterfactual_system}:rotation",
                        ),
                        samples=args.bootstrap_samples,
                    ),
                }
            allowed_missing = (
                set(protocol.get("counterfactual_source_fail_closed_pairs", []))
                if counterfactual_system == SYSTEM_SOURCE_SHUFFLE
                else set()
            )
            unexpected_missing = sorted(set(missing_pairs) - allowed_missing)
            unexpected_present_exclusion = sorted(
                allowed_missing.intersection(
                    record["pair_id"] for record in primary
                )
                - set(missing_pairs)
            )
            if unexpected_missing or unexpected_present_exclusion:
                raise RuntimeError(
                    "Counterfactual pairing is incomplete or violated its fail-closed "
                    f"manifest for {counterfactual_system}: "
                    f"missing={unexpected_missing[:5]}, "
                    f"unexpected_present={unexpected_present_exclusion[:5]}"
                )
            expected_pair_count = len(primary) - len(set(missing_pairs))
            if expected_pair_count == 0:
                comparisons[counterfactual_system] = {
                    "status": "all_subset_pairs_declared_fail_closed",
                    "paired_count": 0,
                    "excluded_pair_ids": sorted(set(missing_pairs)),
                }
                continue
            if counterfactual_system not in comparisons:
                raise RuntimeError(
                    f"No paired counterfactual values for {counterfactual_system}"
                )
            comparisons[counterfactual_system]["excluded_pair_ids"] = sorted(
                set(missing_pairs)
            )
            for statistic in (
                "target_position_error_degradation_m",
                "target_rotation_error_degradation_deg",
            ):
                if int(comparisons[counterfactual_system][statistic]["count"]) != int(
                    expected_pair_count
                ):
                    raise RuntimeError(
                        f"Counterfactual CI count mismatch for {counterfactual_system}.{statistic}"
                    )
            comparisons[counterfactual_system]["paired_count"] = expected_pair_count
        subset["comparisons"] = comparisons
        output["subsets"][subset_name] = subset
    return output


def _aggregate_control_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    controlled = [record for record in records if record.get("control_metrics")]
    rows = []
    for system in CONTROL_SYSTEMS:
        system_records = [
            record for record in controlled if record["system"] == system
        ]
        groups = [
            *[
                (
                    "subtype",
                    subtype,
                    [
                        record
                        for record in system_records
                        if record["control"]["subtype"] == subtype
                    ],
                )
                for subtype in EDIT_CONTROL_SUBTYPES
            ],
            ("all", "all", system_records),
        ]
        for level, name, selected in groups:
            if not selected:
                continue
            passes = {}
            for pass_name in (
                "generated_raw",
                "diagnostic_exact_clamp",
                "ground_truth",
            ):
                names = sorted(
                    {
                        metric
                        for record in selected
                        for metric, value in record["control_metrics"][
                            pass_name
                        ].items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                )
                means = {}
                valid_counts = {}
                missing_counts = {}
                for metric in names:
                    values = []
                    for record in selected:
                        value = record["control_metrics"][pass_name].get(metric)
                        if value is None:
                            continue
                        numeric = float(value)
                        if not math.isfinite(numeric):
                            raise RuntimeError(
                                f"Non-finite control metric {pass_name}.{metric}"
                            )
                        values.append(numeric)
                    if not values:
                        raise RuntimeError(
                            f"Control metric {pass_name}.{metric} has no values"
                        )
                    means[metric] = float(np.mean(values))
                    valid_counts[metric] = len(values)
                    missing_counts[metric] = len(selected) - len(values)
                passes[pass_name] = {
                    "metrics": means,
                    "valid_counts": valid_counts,
                    "missing_counts": missing_counts,
                }
            rows.append(
                {
                    "system": system,
                    "level": level,
                    "name": name,
                    "case_count": len(selected),
                    "passes": passes,
                }
            )
    return rows


_CONTROL_HIGHER_IS_BETTER = frozenset(
    {
        "constraint_root2d_acc",
        "foot_contact_consistency",
        "controlled_contact_accuracy",
        "controlled_contact_f1",
        "controlled_contact_exact_equality",
    }
)
_CONTROL_LOWER_IS_BETTER = frozenset(
    {
        "foot_skate_from_height",
        "foot_skate_from_pred_contacts",
        "foot_skate_max_vel",
        "foot_skate_ratio",
        "constraint_root2d_err",
        "constraint_fullbody_keyframe",
        "constraint_end_effector",
        "constraint_end_effector_rotation_deg",
        "controlled_contact_bce",
        "controlled_contact_brier",
        "fk_position_rotation_consistency_cm",
        "prediction_jerk_mps3",
    }
)
_CONTROL_COUNT_METRICS = frozenset(
    {"controlled_contact_entries", "controlled_contact_positive_entries"}
)


def _control_metric_direction(name: str) -> str:
    if name in _CONTROL_COUNT_METRICS:
        return "count"
    if name in _CONTROL_HIGHER_IS_BETTER:
        return "higher"
    if name in _CONTROL_LOWER_IS_BETTER:
        return "lower"
    raise RuntimeError(f"Unregistered control metric direction: {name}")


def _paired_edit_control_summary(
    records: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    edit_records = {
        record["pair_id"]: record
        for record in records
        if record["system"] == SYSTEM_EDIT_CONTROL
    }
    standalone_records = {
        record["pair_id"]: record
        for record in records
        if record["system"] == SYSTEM_STANDALONE_CONTROL
    }
    if set(edit_records) != set(standalone_records):
        raise RuntimeError("Edit+control and standalone-control case sets differ")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for pair_id in sorted(edit_records):
        edit = edit_records[pair_id]
        standalone = standalone_records[pair_id]
        if (
            edit["sample_seed"] != standalone["sample_seed"]
            or edit["control_subtype"] != standalone["control_subtype"]
            or edit["control"]["motion_mask_sha256"]
            != standalone["control"]["motion_mask_sha256"]
            or edit["control"]["observed_motion_sha256"]
            != standalone["control"]["observed_motion_sha256"]
            or edit["sampling_protocol"]["initial_continuous_noise_sha256"]
            != standalone["sampling_protocol"][
                "initial_continuous_noise_sha256"
            ]
            or edit["sampling_protocol"]["initial_contact_noise_sha256"]
            != standalone["sampling_protocol"]["initial_contact_noise_sha256"]
        ):
            raise RuntimeError(
                f"Matched control comparator differs for pair {pair_id}"
            )
        edit_schema = set(edit["control_metrics"]["generated_raw"])
        standalone_schema = set(standalone["control_metrics"]["generated_raw"])
        if edit_schema != standalone_schema:
            raise RuntimeError(
                f"Matched control metric schema differs for pair {pair_id}"
            )
        for metric_name in edit_schema:
            _control_metric_direction(metric_name)
        pairs.append((edit, standalone))

    groups: list[
        tuple[str, str, list[tuple[dict[str, Any], dict[str, Any]]]]
    ] = [
        ("all", "all", pairs),
        (
            "length_relation",
            "equal",
            [pair for pair in pairs if pair[0]["length_relation"] == "equal"],
        ),
        (
            "length_relation",
            "unequal",
            [pair for pair in pairs if pair[0]["length_relation"] != "equal"],
        ),
    ]
    groups.extend(
        (
            "subtype",
            subtype,
            [pair for pair in pairs if pair[0]["control_subtype"] == subtype],
        )
        for subtype in EDIT_CONTROL_SUBTYPES
    )
    if pairs:
        for field_name in sorted(pairs[0][0]["seen_strata"]):
            values = sorted(
                {pair[0]["seen_strata"][field_name] for pair in pairs},
                key=str,
            )
            groups.extend(
                (
                    f"seen_strata.{field_name}",
                    str(value).lower() if isinstance(value, bool) else str(value),
                    [
                        pair
                        for pair in pairs
                        if pair[0]["seen_strata"][field_name] == value
                    ],
                )
                for value in values
            )
    output_rows = []
    for level, name, selected in groups:
        if not selected:
            continue
        metric_names = sorted(
            set.intersection(
                *[
                    set(edit["control_metrics"]["generated_raw"])
                    & set(standalone["control_metrics"]["generated_raw"])
                    for edit, standalone in selected
                ]
            )
        )
        metrics = {}
        for metric_name in metric_names:
            direction = _control_metric_direction(metric_name)
            if direction == "count":
                continue
            directional_deltas = []
            for edit, standalone in selected:
                edit_value = edit["control_metrics"]["generated_raw"][metric_name]
                standalone_value = standalone["control_metrics"][
                    "generated_raw"
                ][metric_name]
                if edit_value is None or standalone_value is None:
                    raise RuntimeError(
                        f"Matched control metric is unexpectedly null: {metric_name}"
                    )
                edit_numeric = float(edit_value)
                standalone_numeric = float(standalone_value)
                if not math.isfinite(edit_numeric) or not math.isfinite(
                    standalone_numeric
                ):
                    raise RuntimeError(
                        f"Non-finite matched control metric {metric_name}"
                    )
                directional_deltas.append(
                    edit_numeric - standalone_numeric
                    if direction == "higher"
                    else standalone_numeric - edit_numeric
                )
            if len(directional_deltas) != len(selected):
                raise RuntimeError(
                    f"Matched control CI count mismatch for {level}.{name}.{metric_name}"
                )
            statistic = _bootstrap_mean_ci(
                directional_deltas,
                seed=_stat_seed(
                    args.seed,
                    f"edit-control:{level}:{name}:{metric_name}",
                ),
                samples=args.bootstrap_samples,
            )
            if int(statistic["count"]) != len(selected):
                raise RuntimeError(
                    f"Matched control bootstrap count mismatch for {metric_name}"
                )
            metrics[metric_name] = {
                "direction": direction,
                "positive_means_edit_control_better": True,
                "directional_edit_minus_standalone": statistic,
            }
        output_rows.append(
            {
                "level": level,
                "name": name,
                "pair_count": len(selected),
                "metrics": metrics,
            }
        )
    return {
        "edit_control_system": SYSTEM_EDIT_CONTROL,
        "standalone_control_system": SYSTEM_STANDALONE_CONTROL,
        "same_pair_seed_subtype_mask_observation_required": True,
        "rows": output_rows,
    }


def _aggregate_metric_dicts(
    records: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty MotionFix group")
    schema = set(records[0][field])
    for record in records:
        values = record.get(field)
        if not isinstance(values, dict) or set(values) != schema:
            raise RuntimeError(f"MotionFix {field} metric schema changed across cases")
    means: dict[str, float] = {}
    valid_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    for name in sorted(schema):
        values: list[float] = []
        saw_numeric = False
        for record in records:
            value = record[field][name]
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                saw_numeric = True
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(
                        f"Non-finite MotionFix metric {field}.{name} in {record['case_key']}"
                    )
                values.append(numeric)
        if not saw_numeric and not values:
            continue
        means[name] = float(np.mean(values)) if values else float("nan")
        valid_counts[name] = len(values)
        missing_counts[name] = len(records) - len(values)
        if not values or not math.isfinite(means[name]):
            raise RuntimeError(f"MotionFix metric {field}.{name} has no finite values")
    return {
        "metrics": means,
        "valid_counts": valid_counts,
        "missing_counts": missing_counts,
    }


def _aggregate_system_rows(
    records: list[dict[str, Any]], systems: Iterable[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system in systems:
        selected = [record for record in records if record["system"] == system]
        groups: list[tuple[str, str, list[dict[str, Any]]]] = [
            ("all", "all", selected),
            (
                "length_relation",
                "equal",
                [record for record in selected if record["length_relation"] == "equal"],
            ),
            (
                "length_relation",
                "unequal",
                [record for record in selected if record["length_relation"] != "equal"],
            ),
        ]
        if system == SYSTEM_SOURCE_COPY:
            protocols = sorted(
                {
                    str(record["source_copy_protocol"]["protocol"])
                    for record in selected
                }
            )
            groups.extend(
                (
                    "source_copy_protocol",
                    protocol_name,
                    [
                        record
                        for record in selected
                        if record["source_copy_protocol"]["protocol"]
                        == protocol_name
                    ],
                )
                for protocol_name in protocols
            )
        seen_fields = sorted(selected[0]["seen_strata"]) if selected else []
        for field_name in seen_fields:
            values = sorted(
                {record["seen_strata"][field_name] for record in selected},
                key=str,
            )
            groups.extend(
                (
                    f"seen_strata.{field_name}",
                    str(value).lower() if isinstance(value, bool) else str(value),
                    [
                        record
                        for record in selected
                        if record["seen_strata"][field_name] == value
                    ],
                )
                for value in values
            )
        for stratum_type, stratum, group in groups:
            if not group:
                continue
            rows.append(
                {
                    "system": system,
                    "stratum_type": stratum_type,
                    "stratum": stratum,
                    "case_count": len(group),
                    **_aggregate_metric_dicts(group, field="metrics"),
                }
            )
    return rows


def _select_visual_review_rows(
    ordered: list[dict[str, Any]], audit_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    selected_keys: set[str] = set()
    for system in ALL_SYSTEMS:
        for relation, limit in (("equal", 4), ("unequal", 2)):
            candidates = [
                record
                for record in ordered
                if record["system"] == system
                and (
                    record["length_relation"] == "equal"
                    if relation == "equal"
                    else record["length_relation"] != "equal"
                )
            ]
            selected_keys.update(record["case_key"] for record in candidates[:limit])
    for system in CONTROL_SYSTEMS:
        for subtype in EDIT_CONTROL_SUBTYPES:
            candidates = [
                record
                for record in ordered
                if record["system"] == system
                and record.get("control_subtype") == subtype
            ]
            if candidates:
                selected_keys.add(candidates[0]["case_key"])
    return [row for row in audit_rows if row["case_key"] in selected_keys]


def _write_visual_review_manifests(
    output_dir: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    records_with_output = [
        record for record in records if isinstance(record.get("motion_output"), dict)
    ]
    if not records_with_output:
        return {"enabled": False, "reason": "motion_outputs_disabled"}
    ordered = sorted(
        records_with_output,
        key=lambda record: (
            record["system"],
            record.get("control_subtype") or "",
            record["pair_id"],
        ),
    )
    audit_rows = [
        {
            "case_key": record["case_key"],
            "pair_id": record["pair_id"],
            "system": record["system"],
            "control_subtype": record.get("control_subtype"),
            "length_relation": record["length_relation"],
            "instruction": record["instruction"],
            "model_instruction": record["model_instruction"],
            "condition_provenance": record["condition_provenance"],
            "assets": record["assets"],
            "aligned_reference_source": record["aligned_reference_source"],
            "output_gauge_phi": record["output_gauge_phi"],
            "source_copy_protocol": record["source_copy_protocol"],
            "motion_output": record["motion_output"],
            "control": record["control"],
            "metrics": {
                name: record["metrics"].get(name)
                for name in (
                    "prediction_jerk_mps3",
                    "target_jerk_mps3",
                    "foot_skate_from_height",
                    "foot_skate_from_pred_contacts",
                    "foot_skate_max_vel",
                    "foot_skate_ratio",
                    "foot_contact_consistency",
                )
            },
            "delta_views": [
                "evaluator_source_target",
                "evaluator_source_prediction",
                "target_prediction",
            ],
        }
        for record in ordered
    ]
    audit_path = output_dir / "visual_review_manifest.jsonl"
    _atomic_jsonl(audit_path, audit_rows)

    selection = _select_visual_review_rows(ordered, audit_rows)
    selection_path = output_dir / "visual_review_selection.json"
    from tools.render_hy273_motionfix_review import renderer_dependency_identity

    _atomic_json(
        selection_path,
        {
            "format": VISUAL_SELECTION_FORMAT,
            "selection_policy": VISUAL_SELECTION_POLICY,
            "all_cases_manifest": {
                "path": str(audit_path),
                "sha256": dataset_sha256_file(audit_path),
                "count": len(audit_rows),
            },
            "renderer": renderer_dependency_identity(),
            "case_count": len(selection),
            "selected_case_keys_sha256": canonical_sha(
                [row["case_key"] for row in selection]
            ),
            "cases": selection,
        },
    )
    return {
        "enabled": True,
        "all_cases": {
            "path": str(audit_path),
            "sha256": dataset_sha256_file(audit_path),
            "count": len(audit_rows),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": dataset_sha256_file(selection_path),
            "count": len(selection),
        },
    }


def _write_core_artifact_index(
    output_dir: Path,
    *,
    protocol_path: Path,
    protocol: dict[str, Any],
    summary_path: Path,
    visual_review: dict[str, Any],
) -> Path:
    if visual_review.get("enabled") is not True:
        raise RuntimeError("MotionFix artifact index requires saved visual inputs")
    preflight_path = Path(protocol["preflight_manifest"]).expanduser().resolve()
    preflight = _load_json_object_strict(preflight_path)
    expected_case_path = Path(
        protocol["expected_case_manifest"]["path"]
    ).expanduser().resolve()
    shard_artifacts = []
    for shard_id in range(int(protocol["num_shards"])):
        shard_artifacts.append(
            {
                "shard_id": shard_id,
                "records": _artifact_file(
                    output_dir / "shards" / f"shard_{shard_id:02d}.jsonl"
                ),
                "summary": _artifact_file(
                    output_dir
                    / "shards"
                    / f"shard_{shard_id:02d}_summary.json"
                ),
            }
        )
    required_views = _required_render_views()
    artifacts = {
        "preflight_manifest": _artifact_file(preflight_path),
        "protocol_manifest": _artifact_file(protocol_path),
        "expected_case_manifest": _artifact_file(expected_case_path),
        "checkpoint_content_verification": _artifact_file(
            output_dir / "checkpoint_content_verification.json"
        ),
        "summary": _artifact_file(summary_path),
        "visual_review_manifest": _artifact_file(
            visual_review["all_cases"]["path"]
        ),
        "visual_review_selection": _artifact_file(
            visual_review["selection"]["path"]
        ),
        "shards": shard_artifacts,
    }
    immutable_contract = {
        "format": ARTIFACT_INDEX_CONTRACT_FORMAT,
        "evaluation_code_identity_sha256": preflight["code"]["sha256"],
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "protocol_manifest_sha256": artifacts["protocol_manifest"]["sha256"],
        "summary_sha256": artifacts["summary"]["sha256"],
        "core_artifacts_sha256": canonical_sha(artifacts),
        "required_render_views_sha256": canonical_sha(required_views),
        "required_render_view_ids_sha256": canonical_sha(
            [view["view_id"] for view in required_views]
        ),
    }
    payload = {
        "format": ARTIFACT_INDEX_FORMAT,
        "status": "awaiting_required_renders",
        **{
            key: value
            for key, value in immutable_contract.items()
            if key != "format"
        },
        "core_artifacts": artifacts,
        "required_render_views": required_views,
        "immutable_contract": immutable_contract,
        "immutable_contract_sha256": canonical_sha(immutable_contract),
        "render_manifests": {},
        "render_manifests_sha256": canonical_sha({}),
    }
    index_path = output_dir / "artifact_index.json"
    _atomic_json(index_path, payload)
    return index_path


def _validate_aggregate_preflight(
    output_dir: Path,
    protocol: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    preflight_path = Path(protocol["preflight_manifest"]).expanduser().resolve()
    if not preflight_path.is_file():
        raise RuntimeError("MotionFix preflight manifest disappeared before aggregation")
    preflight_sha = dataset_sha256_file(preflight_path)
    if preflight_sha != protocol.get("preflight_sha256"):
        raise RuntimeError("MotionFix preflight changed before aggregation")
    preflight = _load_json_object_strict(preflight_path)
    if preflight.get("format") != PREFLIGHT_FORMAT or preflight.get("status") != "passed":
        raise RuntimeError("MotionFix aggregate received an invalid preflight")
    if preflight.get("code") != evaluation_code_identity():
        raise RuntimeError("MotionFix evaluator/dependency code changed before aggregation")
    _validate_aggregate_training_code_identity(
        preflight, allow_code_drift=bool(args.research_allow_code_drift)
    )
    _validate_protocol_run_contract(
        preflight,
        protocol,
        args,
        preflight_path=preflight_path,
        preflight_sha256=preflight_sha,
    )
    if preflight.get("expected_case_manifest") != protocol.get(
        "expected_case_manifest"
    ):
        raise RuntimeError("MotionFix expected-case identity changed before aggregation")
    expected_path = Path(
        preflight["expected_case_manifest"]["path"]
    ).expanduser().resolve()
    if expected_path.parent != output_dir:
        raise RuntimeError("MotionFix expected-case manifest escaped the output directory")
    _load_expected_case_manifest(preflight["expected_case_manifest"])
    for field_name in ("manifest", "train_manifest"):
        frozen = preflight[field_name]
        path = Path(frozen["path"]).expanduser().resolve()
        current = {**_file_stat(path), "sha256": dataset_sha256_file(path)}
        if current != frozen:
            raise RuntimeError(f"MotionFix {field_name} changed before aggregation")
    if preflight.get("counterfactual_manifest") != _counterfactual_manifest_identity(
        args
    ):
        raise RuntimeError("MotionFix counterfactual data changed before aggregation")
    hytext = preflight["hytext_profile_identity"]
    hytext_manifest = Path(hytext["manifest_path"]).expanduser().resolve()
    if dataset_sha256_file(hytext_manifest) != hytext["manifest_sha256"]:
        raise RuntimeError("MotionFix HYText profile changed before aggregation")
    checkpoint = preflight["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).expanduser().resolve()
    current_checkpoint_stat = _file_stat(checkpoint_path)
    frozen_checkpoint_stat = {
        key: value
        for key, value in checkpoint.items()
        if key in current_checkpoint_stat
    }
    if current_checkpoint_stat != frozen_checkpoint_stat:
        raise RuntimeError("MotionFix checkpoint changed before aggregation")
    stamp_path = output_dir / "checkpoint_content_verification.json"
    expected_stamp = {
        "format": "hy273_checkpoint_content_verification_v1",
        "preflight_sha256": preflight_sha,
        "checkpoint": checkpoint,
    }
    if not stamp_path.is_file() or json.loads(
        stamp_path.read_text(encoding="utf-8")
    ) != expected_stamp:
        raise RuntimeError("MotionFix checkpoint content verification is missing")
    return preflight


def aggregate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    protocol_path = output_dir / "protocol_manifest.json"
    protocol = _load_json_object_strict(protocol_path)
    protocol_sha = dataset_sha256_file(protocol_path)
    aggregate_identity = {
        "protocol": args.protocol,
        "systems": list(parse_csv(args.systems)),
        "num_shards": int(args.num_shards),
        "seed": int(args.seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "pair_ids": list(parse_csv(args.pair_ids)),
        "max_pairs": int(args.max_pairs),
    }
    frozen_identity = {
        name: protocol[name] for name in aggregate_identity
    }
    if aggregate_identity != frozen_identity:
        raise RuntimeError(
            "Aggregate arguments differ from the frozen evaluation protocol"
        )
    _validate_aggregate_preflight(output_dir, protocol, args)
    expected_rows = _load_expected_case_manifest(protocol["expected_case_manifest"])
    expected_by_key = {row["case_key"]: row for row in expected_rows}
    if len(expected_rows) != int(protocol["case_count"]):
        raise RuntimeError("Protocol and expected-case manifest counts differ")
    records = []
    for shard_id in range(int(protocol["num_shards"])):
        path = output_dir / "shards" / f"shard_{shard_id:02d}.jsonl"
        summary_path = output_dir / "shards" / f"shard_{shard_id:02d}_summary.json"
        if not path.is_file() or not summary_path.is_file():
            raise RuntimeError(f"MotionFix shard {shard_id} is incomplete")
        shard_records = [
            json.loads(line) for line in path.read_text().splitlines() if line
        ]
        shard_summary = json.loads(summary_path.read_text())
        expected_shard_count = sum(
            int(row["expected_shard_id"]) == shard_id for row in expected_rows
        )
        if (
            shard_summary.get("complete") is not True
            or int(shard_summary.get("shard_id", -1)) != shard_id
            or int(shard_summary.get("expected", -1)) != expected_shard_count
            or int(shard_summary.get("records", -1)) != len(shard_records)
            or len(shard_records) != expected_shard_count
            or shard_summary.get("protocol_manifest_sha256") != protocol_sha
        ):
            raise RuntimeError(f"MotionFix shard {shard_id} summary is invalid")
        for record in shard_records:
            expected = expected_by_key.get(record.get("case_key"))
            if expected is None:
                raise RuntimeError(
                    f"MotionFix shard file {shard_id} contains a foreign case"
                )
            _validate_shard_ownership(record, expected, shard_id)
        records.extend(shard_records)
    by_key = {record["case_key"]: record for record in records}
    if len(by_key) != len(records):
        raise RuntimeError("Duplicate MotionFix evaluation case keys")
    if set(by_key) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(by_key))[:5]
        extra = sorted(set(by_key) - set(expected_by_key))[:5]
        raise RuntimeError(
            f"MotionFix evaluated the wrong case set: missing={missing}, extra={extra}"
        )
    for record in records:
        _validate_case_record_identity(
            record,
            expected_by_key[record["case_key"]],
            protocol_sha256=protocol_sha,
            preflight_sha256=str(protocol["preflight_sha256"]),
            verify_motion_output=bool(protocol["save_motion_outputs"]),
            max_sparse_keyframes=int(protocol["max_sparse_keyframes"]),
        )
    rows = _aggregate_system_rows(records, protocol["systems"])
    visual_review = _write_visual_review_manifests(output_dir, records)
    summary = {
        "format": SUMMARY_FORMAT,
        "status": "validated",
        "protocol": protocol,
        "case_count": len(records),
        "case_rows_sha256": canonical_sha(records),
        "rows": rows,
        "paired_counterfactual": _paired_counterfactual_summary(
            records, args, protocol
        ),
        "edit_control_rows": _aggregate_control_metrics(records),
        "paired_edit_control": _paired_edit_control_summary(records, args),
        "visual_review": visual_review,
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(summary_path, summary)
    artifact_index_path = _write_core_artifact_index(
        output_dir,
        protocol_path=protocol_path,
        protocol=protocol,
        summary_path=summary_path,
        visual_review=visual_review,
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "artifact_index": str(artifact_index_path),
                "cases": len(records),
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint_sha256", default="")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--train_manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument(
        "--counterfactual_manifest", default=DEFAULT_COUNTERFACTUAL_MANIFEST
    )
    parser.add_argument(
        "--counterfactual_manifest_sha256",
        default=DEFAULT_COUNTERFACTUAL_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--counterfactual_summary", default=DEFAULT_COUNTERFACTUAL_SUMMARY
    )
    parser.add_argument(
        "--counterfactual_summary_sha256",
        default=DEFAULT_COUNTERFACTUAL_SUMMARY_SHA256,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--protocol",
        choices=[EQUAL_PROTOCOL, FULL_PROTOCOL, VAL_PROTOCOL],
        required=True,
    )
    parser.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    parser.add_argument("--preflight_manifest", default="")
    parser.add_argument("--preflight_sha256", default="")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--weight_source", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument(
        "--pair_ids",
        default="",
        help="Comma-separated frozen pair ids for coverage-oriented smoke gates",
    )
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--source_cfg_scale", type=float, default=1.0)
    parser.add_argument("--edit_cfg_scale", type=float, default=1.0)
    parser.add_argument(
        "--edit_source_baseline",
        choices=EDIT_SOURCE_BASELINE_MODES,
        default="learned",
    )
    parser.add_argument("--generate_text_cfg_scale", type=float, default=2.0)
    parser.add_argument("--control_cfg_scale", type=float, default=2.0)
    parser.add_argument(
        "--cfg_apply_contacts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--generate_cfg_apply_contacts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max_sparse_keyframes", type=int, default=20)
    parser.add_argument("--contact_init", choices=["random", "half", "zeros"], default="random")
    parser.add_argument("--contact_feedback", choices=["blend", "prob", "fixed"], default="blend")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument(
        "--save_motion_outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--research_allow_code_drift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow inference-only research changes while retaining model state compatibility checks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    if args.max_sparse_keyframes < 1:
        raise ValueError("max_sparse_keyframes must be positive")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not args.save_motion_outputs:
        raise ValueError(
            "MotionFix evidence protocols require saved motion outputs for metric replay"
        )
    if args.preflight_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for preflight")
        payload = build_preflight(args)
        path = Path(args.output_dir).expanduser().resolve() / "preflight_manifest.json"
        _atomic_json(path, payload)
        print(
            json.dumps(
                {
                    "passed": True,
                    "preflight_manifest": str(path),
                    "preflight_sha256": dataset_sha256_file(path),
                    "pair_count": payload["plan"]["pair_count"],
                    "case_count": payload["plan"]["case_count"],
                },
                sort_keys=True,
            )
        )
    elif args.aggregate:
        aggregate(args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for model evaluation")
        run_shard(args)


if __name__ == "__main__":
    main()
