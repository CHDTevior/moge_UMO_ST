"""Render provenance-sealed MotionFix source/target/prediction review panels."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.raw_motion.evidence_hash import tensor_sha256
from models.raw_motion.hy273_motionfix_metrics import (
    INTERNAL_METRIC_PROTOCOL,
    evaluate_motionfix_internal_case,
)
from models.raw_motion.hy273_kimodo_benchmark import (
    SMPLX22_METRIC_JOINTS_PATH,
    load_smplx22_metric_joints,
)
from models.raw_motion.hy273_normalizer import apply_yaw_rotation, root_origin_shift
from models.raw_motion.hy273_slices import (
    CONTACT_JOINTS,
    CONTACT_SLICE,
    HEADING_SLICE,
    SMPLX22_PARENTS,
    fk_positions_from_global_rot6d,
)


SELECTION_FORMAT = (
    "hy273_motionfix_visual_review_selection_v7_canonical_reverse_sealed"
)
SELECTION_POLICY = (
    "first_4_equal_first_2_unequal_per_system_plus_first_per_control_subtype_v1"
)
RENDER_FORMAT = "hy273_motionfix_rendered_review_v8_canonical_reverse_sealed"
RENDERER_IDENTITY_FORMAT = (
    "hy273_motionfix_renderer_dependency_identity_v4_full_closure_sealed"
)
ARTIFACT_INDEX_FORMAT = "hy273_motionfix_artifact_index_v2_canonical_reverse_sealed"
ARTIFACT_INDEX_CONTRACT_FORMAT = (
    "hy273_motionfix_artifact_index_immutable_contract_v1"
)
TITLE_METRIC_PROTOCOL = "selected_prediction_internal_k273_v1"
DECODER_PROTOCOL = "global_rot6d_fk_smplx22_metric_exact_project_root_asset_v2"
MOTION_OUTPUT_FORMAT = "hy273_raw_exact_npz_v4_full_overwrite_contract"
ALIGNED_SOURCE_FORMAT = "hy273_evaluator_aligned_source_npy_v1"
ALL_SYSTEMS = (
    "source_copy",
    "source_instruction_model",
    "shuffled_source_instruction_model",
    "source_shuffled_instruction_model",
    "source_only_model",
    "relative_instruction_only_ood",
    "source_instruction_control_model",
    "standalone_control_model",
)
CONTROL_SYSTEMS = (
    "source_instruction_control_model",
    "standalone_control_model",
)
CONTACT_CONTROL_SUBTYPES = (
    "contact_only_sparse",
    "root_sparse_contact",
    "root_dense_contact",
    "endpoints_contact",
    "fullpose_contact",
    "mixed_contact",
)
EDITING_VISUAL_SYSTEMS = tuple(
    system for system in ALL_SYSTEMS if system not in CONTROL_SYSTEMS
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha(payload: Any) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_equal_strict(first: Any, second: Any) -> bool:
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _artifact_refs(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"path", "sha256", "size"} <= set(value):
            refs.append(value)
        else:
            for child in value.values():
                refs.extend(_artifact_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.extend(_artifact_refs(child))
    return refs


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
            ("contact12", CONTROL_SYSTEMS, 12),
        )
    ]


def _validate_artifact_ref(ref: dict[str, Any], *, label: str) -> Path:
    path = Path(str(ref["path"])).expanduser().resolve()
    if (
        not path.is_file()
        or int(path.stat().st_size) != int(ref["size"])
        or _sha256_file(path) != str(ref["sha256"])
    ):
        raise RuntimeError(f"{label} changed or disappeared: {path}")
    return path


def _render_view_signature(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_key": str(value["prediction_key"]),
        "systems": sorted(str(system) for system in value["systems"]),
        "max_cases": int(value["max_cases"]),
        "fps": int(value["fps"]),
        "stride": int(value["stride"]),
        "trail_frames": int(value["trail_frames"]),
        "output_format": str(value["output_format"]),
    }


def _resolve_required_render_view(
    artifact_index: dict[str, Any], parameters: dict[str, Any]
) -> str | None:
    signature = _render_view_signature(parameters)
    matches = [
        str(view["view_id"])
        for view in artifact_index.get("required_render_views", [])
        if _render_view_signature(view) == signature
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous required MotionFix render view: {matches}")
    return matches[0] if matches else None


def _view_cases(
    selection: dict[str, Any], expected_view: dict[str, Any]
) -> list[dict[str, Any]]:
    systems = set(expected_view["systems"])
    cases = [row for row in selection["cases"] if row["system"] in systems]
    cases = _stratified_render_cases(
        cases, max_cases=int(expected_view["max_cases"])
    )
    if len(cases) != int(expected_view["max_cases"]):
        raise RuntimeError(
            f"Required render view {expected_view['view_id']} has "
            f"{len(cases)} cases, expected {expected_view['max_cases']}"
        )
    return cases


def _expected_render_parameters(
    selection: dict[str, Any], expected_view: dict[str, Any]
) -> dict[str, Any]:
    cases = _view_cases(selection, expected_view)
    return {
        "decoder": DECODER_PROTOCOL,
        "prediction_key": expected_view["prediction_key"],
        "title_metric_protocol": TITLE_METRIC_PROTOCOL,
        "systems": list(expected_view["systems"]),
        "max_cases": int(expected_view["max_cases"]),
        "fps": int(expected_view["fps"]),
        "stride": int(expected_view["stride"]),
        "trail_frames": int(expected_view["trail_frames"]),
        "output_format": expected_view["output_format"],
        "selected_case_keys": [row["case_key"] for row in cases],
    }


def _validate_selection(
    artifact_index: dict[str, Any],
    *,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    core = artifact_index["core_artifacts"]
    selection_ref = core["visual_review_selection"]
    selection_path = _validate_artifact_ref(
        selection_ref, label="MotionFix visual selection"
    )
    if expected_path is not None and selection_path != expected_path.resolve():
        raise RuntimeError("Artifact index points to a different visual selection")
    if expected_sha256 is not None and selection_ref["sha256"] != expected_sha256:
        raise RuntimeError("Artifact index visual-selection SHA256 mismatch")
    selection = _load_json_object_strict(selection_path)
    if selection.get("format") != SELECTION_FORMAT:
        raise RuntimeError("MotionFix visual-selection format mismatch")
    if selection.get("selection_policy") != SELECTION_POLICY:
        raise RuntimeError("MotionFix visual-selection policy mismatch")
    all_cases_ref = selection.get("all_cases_manifest")
    core_all_cases = core["visual_review_manifest"]
    if not isinstance(all_cases_ref, dict) or any(
        all_cases_ref.get(key) != core_all_cases.get(key)
        for key in ("path", "sha256")
    ):
        raise RuntimeError("Visual selection is not bound to the core all-cases manifest")
    all_cases_path = _validate_artifact_ref(
        core_all_cases, label="MotionFix all-cases visual manifest"
    )
    all_rows = [
        json.loads(line)
        for line in all_cases_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(all_rows) != int(all_cases_ref.get("count", -1)):
        raise RuntimeError("MotionFix all-cases visual manifest count mismatch")
    replayed = _replay_frozen_selection(all_rows)
    if (
        int(selection.get("case_count", -1)) != len(selection.get("cases", []))
        or not _json_equal_strict(selection.get("cases"), replayed)
        or selection.get("selected_case_keys_sha256")
        != _canonical_sha([row["case_key"] for row in replayed])
    ):
        raise RuntimeError("MotionFix visual-selection replay mismatch")
    if not _json_equal_strict(
        selection.get("renderer"), renderer_dependency_identity()
    ):
        raise RuntimeError("MotionFix visual selection has stale renderer identity")
    return selection


def _immutable_artifact_contract(
    artifact_index: dict[str, Any], required_views: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "format": ARTIFACT_INDEX_CONTRACT_FORMAT,
        "evaluation_code_identity_sha256": artifact_index.get(
            "evaluation_code_identity_sha256"
        ),
        "checkpoint_sha256": artifact_index.get("checkpoint_sha256"),
        "protocol_manifest_sha256": artifact_index.get(
            "protocol_manifest_sha256"
        ),
        "summary_sha256": artifact_index.get("summary_sha256"),
        "core_artifacts_sha256": _canonical_sha(artifact_index.get("core_artifacts")),
        "required_render_views_sha256": _canonical_sha(required_views),
        "required_render_view_ids_sha256": _canonical_sha(
            [view["view_id"] for view in required_views]
        ),
    }


def _validate_registered_render_manifest(
    entry: dict[str, Any],
    *,
    artifact_root: Path,
    selection: dict[str, Any],
    selection_ref: dict[str, Any],
    expected_view: dict[str, Any],
) -> dict[str, Any]:
    expected_entry_keys = {
        "view_id",
        "path",
        "sha256",
        "size",
        "video_count",
        "selected_case_keys_sha256",
    }
    if set(entry) != expected_entry_keys:
        raise RuntimeError("Registered render-manifest entry schema changed")
    path = _validate_artifact_ref(entry, label="Registered render manifest")
    if not path.is_relative_to(artifact_root.resolve()):
        raise RuntimeError("Registered render manifest escaped the evaluation directory")
    manifest = _load_json_object_strict(path)
    expected_manifest_keys = {
        "format",
        "artifact_view_id",
        "selection",
        "renderer",
        "parameters",
        "videos",
    }
    expected_parameters = _expected_render_parameters(selection, expected_view)
    expected_selection = {
        "path": str(Path(selection_ref["path"]).resolve()),
        "sha256": selection_ref["sha256"],
        "format": SELECTION_FORMAT,
    }
    expected_cases = _view_cases(selection, expected_view)
    videos = manifest.get("videos")
    if not isinstance(videos, list):
        raise RuntimeError(f"Registered render manifest has no video list: {path}")
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("format") != RENDER_FORMAT
        or manifest.get("artifact_view_id") != expected_view["view_id"]
        or entry.get("view_id") != expected_view["view_id"]
        or not _json_equal_strict(manifest.get("selection"), expected_selection)
        or not _json_equal_strict(manifest.get("renderer"), selection["renderer"])
        or not _json_equal_strict(
            manifest.get("renderer"), renderer_dependency_identity()
        )
        or not _json_equal_strict(manifest.get("parameters"), expected_parameters)
        or int(entry.get("video_count", -1)) != len(videos)
        or int(entry.get("video_count", -1)) != len(expected_cases)
        or entry.get("selected_case_keys_sha256")
        != _canonical_sha(expected_parameters["selected_case_keys"])
    ):
        raise RuntimeError(f"Registered render manifest contract changed: {path}")
    seen_paths: set[Path] = set()
    expected_video_keys = {
        "case_key",
        "path",
        "sha256",
        "size",
        "title_metric_protocol",
        "title_metrics",
        "inputs",
    }
    expected_title_keys = {
        "prediction_jerk_mps3",
        "foot_skate_ratio",
        "foot_contact_consistency",
    }
    for video, case in zip(videos, expected_cases, strict=True):
        if not isinstance(video, dict):
            raise RuntimeError(f"Registered rendered video schema changed: {path}")
        expected_inputs = {
            "assets": case["assets"],
            "aligned_reference_source": case["aligned_reference_source"],
            "motion_output": case["motion_output"],
        }
        video_path = Path(str(video["path"])).expanduser().resolve()
        if (
            set(video) != expected_video_keys
            or video.get("case_key") != case["case_key"]
            or video_path.parent != path.parent
            or video_path.name
            != f"{case['case_key']}.{expected_view['output_format']}"
            or video_path in seen_paths
            or not video_path.is_file()
            or int(video_path.stat().st_size) != int(video["size"])
            or _sha256_file(video_path) != str(video["sha256"])
            or video.get("title_metric_protocol") != TITLE_METRIC_PROTOCOL
            or set(video.get("title_metrics", {})) != expected_title_keys
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
                for value in video.get("title_metrics", {}).values()
            )
            or not _json_equal_strict(video.get("inputs"), expected_inputs)
        ):
            raise RuntimeError(f"Registered rendered video changed: {video_path}")
        seen_paths.add(video_path)
    return manifest


def _validate_artifact_index(
    artifact_index_path: Path,
    artifact_index: dict[str, Any],
    *,
    expected_selection_path: Path | None = None,
    expected_selection_sha256: str | None = None,
) -> dict[str, Any]:
    if artifact_index.get("format") != ARTIFACT_INDEX_FORMAT:
        raise RuntimeError("MotionFix artifact-index format mismatch")
    required_views = _required_render_views()
    if not _json_equal_strict(
        artifact_index.get("required_render_views"), required_views
    ):
        raise RuntimeError("MotionFix required render views changed")
    core = artifact_index.get("core_artifacts")
    required_core = {
        "preflight_manifest",
        "protocol_manifest",
        "expected_case_manifest",
        "checkpoint_content_verification",
        "summary",
        "visual_review_manifest",
        "visual_review_selection",
        "shards",
    }
    if not isinstance(core, dict) or set(core) != required_core:
        raise RuntimeError("MotionFix core artifact schema changed")
    for ref in _artifact_refs(core):
        _validate_artifact_ref(ref, label="Core MotionFix artifact")
    immutable = _immutable_artifact_contract(artifact_index, required_views)
    if (
        not _json_equal_strict(artifact_index.get("immutable_contract"), immutable)
        or artifact_index.get("immutable_contract_sha256") != _canonical_sha(immutable)
        or any(
            artifact_index.get(key) != value
            for key, value in immutable.items()
            if key != "format"
        )
    ):
        raise RuntimeError("MotionFix artifact-index immutable contract changed")

    preflight = _load_json_object_strict(core["preflight_manifest"]["path"])
    protocol = _load_json_object_strict(core["protocol_manifest"]["path"])
    summary = _load_json_object_strict(core["summary"]["path"])
    if (
        preflight.get("code", {}).get("sha256")
        != artifact_index["evaluation_code_identity_sha256"]
        or preflight.get("checkpoint", {}).get("sha256")
        != artifact_index["checkpoint_sha256"]
        or core["protocol_manifest"]["sha256"]
        != artifact_index["protocol_manifest_sha256"]
        or core["summary"]["sha256"] != artifact_index["summary_sha256"]
        or protocol.get("preflight_manifest")
        != core["preflight_manifest"]["path"]
        or protocol.get("preflight_sha256")
        != core["preflight_manifest"]["sha256"]
        or not _json_equal_strict(summary.get("protocol"), protocol)
    ):
        raise RuntimeError("MotionFix artifact-index core provenance changed")
    selection = _validate_selection(
        artifact_index,
        expected_path=expected_selection_path,
        expected_sha256=expected_selection_sha256,
    )
    registered = artifact_index.get("render_manifests")
    if not isinstance(registered, dict) or artifact_index.get(
        "render_manifests_sha256"
    ) != _canonical_sha(registered):
        raise RuntimeError("MotionFix registered-render digest mismatch")
    by_id = {view["view_id"]: view for view in required_views}
    if not set(registered) <= set(by_id):
        raise RuntimeError("Artifact index contains a foreign render view")
    selection_ref = core["visual_review_selection"]
    for view_id, entry in registered.items():
        _validate_registered_render_manifest(
            entry,
            artifact_root=artifact_index_path.parent,
            selection=selection,
            selection_ref=selection_ref,
            expected_view=by_id[view_id],
        )
    expected_status = (
        "validated"
        if set(registered) == set(by_id)
        else "awaiting_required_renders"
    )
    if artifact_index.get("status") != expected_status:
        raise RuntimeError("MotionFix artifact-index status is inconsistent")
    return selection


def _register_render_manifest(
    *,
    artifact_index_path: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    lock_path = artifact_index_path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        artifact_index = _load_json_object_strict(artifact_index_path)
        selection = _validate_artifact_index(
            artifact_index_path, artifact_index
        )
        persisted_manifest = _load_json_object_strict(manifest_path)
        if not _json_equal_strict(persisted_manifest, manifest):
            raise RuntimeError("In-memory and persisted render manifests differ")
        if (
            set(manifest)
            != {
                "format",
                "artifact_view_id",
                "selection",
                "renderer",
                "parameters",
                "videos",
            }
            or not isinstance(manifest.get("parameters"), dict)
            or not isinstance(manifest.get("videos"), list)
        ):
            raise RuntimeError("Persisted render manifest schema changed")
        view_id = _resolve_required_render_view(
            artifact_index, manifest["parameters"]
        )
        if view_id is None or manifest.get("artifact_view_id") != view_id:
            raise RuntimeError("Render manifest is not a frozen required view")
        expected_view = {
            view["view_id"]: view for view in _required_render_views()
        }[view_id]
        entry = {
            "view_id": view_id,
            "path": str(manifest_path.resolve()),
            "sha256": _sha256_file(manifest_path),
            "size": int(manifest_path.stat().st_size),
            "video_count": len(manifest["videos"]),
            "selected_case_keys_sha256": _canonical_sha(
                manifest["parameters"]["selected_case_keys"]
            ),
        }
        _validate_registered_render_manifest(
            entry,
            artifact_root=artifact_index_path.parent,
            selection=selection,
            selection_ref=artifact_index["core_artifacts"][
                "visual_review_selection"
            ],
            expected_view=expected_view,
        )
        registered = dict(artifact_index.get("render_manifests", {}))
        previous = registered.get(view_id)
        if previous is not None and not _json_equal_strict(previous, entry):
            raise RuntimeError(f"Required render view was already sealed: {view_id}")
        registered[view_id] = entry
        required_ids = {view["view_id"] for view in _required_render_views()}
        artifact_index["render_manifests"] = dict(sorted(registered.items()))
        artifact_index["render_manifests_sha256"] = _canonical_sha(
            artifact_index["render_manifests"]
        )
        artifact_index["status"] = (
            "validated"
            if set(registered) == required_ids
            else "awaiting_required_renders"
        )
        _validate_artifact_index(artifact_index_path, artifact_index)
        _atomic_json(artifact_index_path, artifact_index)
        return artifact_index


def renderer_dependency_identity() -> dict[str, Any]:
    paths = {
        Path(__file__).resolve(),
        ROOT / "models" / "__init__.py",
        *sorted((ROOT / "models" / "raw_motion").glob("*.py")),
        SMPLX22_METRIC_JOINTS_PATH,
    }
    files = []
    for path in sorted(paths, key=str):
        resolved = path.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(ROOT):
            raise RuntimeError(f"Missing renderer dependency: {resolved}")
        files.append(
            {
                "path": str(resolved.relative_to(ROOT)),
                "size": int(resolved.stat().st_size),
                "sha256": _sha256_file(resolved),
            }
        )
    payload = {
        "format": RENDERER_IDENTITY_FORMAT,
        "files": files,
        "protocols": {
            "internal_metric": INTERNAL_METRIC_PROTOCOL,
            "title_metric": TITLE_METRIC_PROTOCOL,
            "decoder": DECODER_PROTOCOL,
            "skeleton_resolution": "project_root_explicit_path_v1",
            "skeleton_asset": str(SMPLX22_METRIC_JOINTS_PATH.relative_to(ROOT)),
        },
    }
    payload["identity_sha256"] = _canonical_sha(payload)
    return payload


def _validate_renderer_identity(frozen: Any) -> dict[str, Any]:
    current = renderer_dependency_identity()
    if not _json_equal_strict(frozen, current):
        raise RuntimeError("MotionFix renderer dependencies changed after selection was frozen")
    return current


def _to_gauge(features: np.ndarray, phi: float) -> np.ndarray:
    value = torch.from_numpy(np.asarray(features, dtype=np.float32)).clone()
    shifted = root_origin_shift(value)
    heading = shifted[0, HEADING_SLICE]
    delta = torch.as_tensor(phi, dtype=shifted.dtype) - torch.atan2(
        heading[1], heading[0]
    )
    return apply_yaw_rotation(shifted, delta).cpu().numpy()


def _joints(features: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    neutral_joints = load_smplx22_metric_joints(
        device=tensor.device,
        dtype=tensor.dtype,
    )
    with torch.no_grad():
        value = fk_positions_from_global_rot6d(
            tensor,
            neutral_joints=neutral_joints,
        )
    return value.cpu().numpy().astype(np.float32, copy=False)


def _selected_prediction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    reference_source: np.ndarray,
) -> dict[str, float]:
    values = evaluate_motionfix_internal_case(
        torch.from_numpy(np.asarray(prediction, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(target, dtype=np.float32).copy()),
        torch.from_numpy(np.asarray(reference_source, dtype=np.float32).copy()),
    )
    selected = {
        name: float(values[name])
        for name in (
            "prediction_jerk_mps3",
            "foot_skate_ratio",
            "foot_contact_consistency",
        )
    }
    if not all(np.isfinite(value) for value in selected.values()):
        raise RuntimeError("Selected prediction produced non-finite title metrics")
    return selected


def _normalized_time_joints(joints: np.ndarray, target_frames: int) -> np.ndarray:
    if joints.shape[0] == target_frames:
        return joints
    source_time = np.linspace(0.0, 1.0, joints.shape[0], dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
    flat = joints.reshape(joints.shape[0], -1)
    aligned = np.stack(
        [
            np.interp(target_time, source_time, flat[:, axis])
            for axis in range(flat.shape[1])
        ],
        axis=-1,
    )
    return aligned.reshape(target_frames, 22, 3).astype(np.float32)


def _load_asset(identity: dict[str, Any]) -> np.ndarray:
    path = Path(identity["path"]).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != identity["sha256"]:
        raise RuntimeError(f"Render input asset changed: {path}")
    value = np.load(path, allow_pickle=False)
    expected = (int(identity["frames"]), int(identity["feature_dim"]))
    if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
        raise RuntimeError(f"Render input asset is invalid: {path} {value.shape}")
    return value


def _load_aligned_source(identity: dict[str, Any]) -> np.ndarray:
    if identity.get("format") != ALIGNED_SOURCE_FORMAT:
        raise RuntimeError("Unknown evaluator-aligned source format")
    path = Path(identity["path"]).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != identity["sha256"]:
        raise RuntimeError(f"Evaluator-aligned source changed: {path}")
    value = np.load(path, allow_pickle=False)
    if (
        list(value.shape) != list(identity["shape"])
        or value.dtype != np.float32
        or not np.isfinite(value).all()
        or tensor_sha256(torch.from_numpy(value.copy()).contiguous())
        != identity["tensor_sha256"]
    ):
        raise RuntimeError(f"Evaluator-aligned source is invalid: {path}")
    return value


def _load_motion_output(
    identity: dict[str, Any], prediction_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if identity.get("format") != MOTION_OUTPUT_FORMAT:
        raise RuntimeError("Unknown MotionFix motion-output format")
    path = Path(identity["path"]).expanduser().resolve()
    if not path.is_file() or _sha256_file(path) != identity["sha256"]:
        raise RuntimeError(f"MotionFix render output changed: {path}")
    with np.load(path, allow_pickle=False) as output:
        loaded = {name: output[name].copy() for name in output.files}
        prediction = loaded[prediction_key].astype(np.float32, copy=False)
        mask = (
            loaded["mask"].astype(bool, copy=False)
            if "mask" in loaded
            else np.zeros_like(prediction, dtype=bool)
        )
        observed = (
            loaded["observed"].astype(np.float32, copy=False)
            if "observed" in loaded
            else np.zeros_like(prediction, dtype=np.float32)
        )
    array_identity = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": tensor_sha256(torch.from_numpy(value.copy()).contiguous()),
        }
        for name, value in sorted(loaded.items())
    }
    if identity.get("arrays") != array_identity:
        raise RuntimeError(f"MotionFix render tensor identity changed: {path}")
    if (
        prediction.ndim != 2
        or prediction.shape[1] != 273
        or mask.shape != prediction.shape
        or observed.shape != prediction.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(observed).all()
    ):
        raise RuntimeError(f"MotionFix render payload is invalid: {path}")
    return prediction, mask, observed


def _controlled_joint_mask(mask: np.ndarray) -> np.ndarray:
    frames = mask.shape[0]
    result = np.zeros((frames, 22), dtype=bool)
    result |= mask[:, 5:71].reshape(frames, 22, 3).any(axis=-1)
    result |= mask[:, 71:203].reshape(frames, 22, 6).any(axis=-1)
    result[:, 0] |= mask[:, 0:5].any(axis=-1)
    contact_mask = mask[:, 269:273]
    for contact_index, joint_index in enumerate(CONTACT_JOINTS):
        result[:, joint_index] |= contact_mask[:, contact_index]
    return result


def _edges() -> list[tuple[int, int]]:
    return [
        (int(parent), child)
        for child, parent in enumerate(SMPLX22_PARENTS.tolist())
        if parent >= 0
    ]


def _limits(*motions: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    present = [motion for motion in motions if motion is not None]
    points = np.concatenate([motion.reshape(-1, 3) for motion in present], axis=0)
    points = points[np.isfinite(points).all(axis=-1)]
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) * 0.5
    half = max(float((upper - lower).max()) * 0.55, 1.0)
    result_lower = center - half
    result_upper = center + half
    result_lower[1] = min(float(result_lower[1]), 0.0)
    return result_lower, result_upper


def _setup_axis(ax: Any, lower: np.ndarray, upper: np.ndarray, title: str) -> None:
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[2], upper[2])
    ax.set_zlim(lower[1], upper[1])
    ax.view_init(elev=18, azim=-68)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def _draw(ax: Any, joints: np.ndarray, color: str, alpha: float, width: float) -> None:
    for parent, child in _edges():
        segment = joints[[parent, child]]
        ax.plot(
            segment[:, 0],
            segment[:, 2],
            segment[:, 1],
            color=color,
            alpha=alpha,
            linewidth=width,
        )


def _load_case(row: dict[str, Any], prediction_key: str) -> dict[str, Any]:
    assets = row["assets"]
    provenance = row["condition_provenance"]
    condition_identity = assets.get("conditioning_source_k273")
    condition_present = bool(provenance["source_condition_present"])
    if condition_present != (condition_identity is not None):
        raise RuntimeError("Visual condition-source provenance is inconsistent")

    phi = float(row["output_gauge_phi"])
    target_features = _to_gauge(_load_asset(assets["target_k273"]), phi)
    reference_features = _load_aligned_source(row["aligned_reference_source"])
    prediction_features, mask, observed = _load_motion_output(
        row["motion_output"], prediction_key
    )
    target_frames = prediction_features.shape[0]
    if target_features.shape[0] != target_frames or reference_features.shape[0] != target_frames:
        raise RuntimeError("Visual target/reference length differs from evaluated prediction")
    condition_joints = None
    if condition_identity is not None:
        condition_features = _to_gauge(_load_asset(condition_identity), phi)
        condition_joints = _normalized_time_joints(
            _joints(condition_features), target_frames
        )
    title_metrics = _selected_prediction_metrics(
        prediction_features,
        target_features,
        reference_features,
    )
    return {
        "conditioning_source": condition_joints,
        "reference_source": _joints(reference_features),
        "target": _joints(target_features),
        "prediction": _joints(prediction_features),
        "prediction_contacts": prediction_features[:, CONTACT_SLICE] >= 0.5,
        "target_contacts": target_features[:, CONTACT_SLICE] >= 0.5,
        "controlled_contacts": mask[:, CONTACT_SLICE],
        "observed_contacts": observed[:, CONTACT_SLICE] >= 0.5,
        "control_mask": _controlled_joint_mask(mask),
        "title_metrics": title_metrics,
    }


def render_case(
    row: dict[str, Any],
    save_path: Path,
    *,
    prediction_key: str,
    fps: int,
    stride: int,
    trail_frames: int,
) -> dict[str, float]:
    data = _load_case(row, prediction_key)
    conditioning_source = (
        None
        if data["conditioning_source"] is None
        else data["conditioning_source"][::stride]
    )
    reference_source = data["reference_source"][::stride]
    target = data["target"][::stride]
    prediction = data["prediction"][::stride]
    control_mask = data["control_mask"][::stride]
    predicted_contacts = data["prediction_contacts"][::stride]
    target_contacts = data["target_contacts"][::stride]
    controlled_contacts = data["controlled_contacts"][::stride]
    observed_contacts = data["observed_contacts"][::stride]
    lower, upper = _limits(conditioning_source, reference_source, target, prediction)
    figure = plt.figure(figsize=(19, 4.9))
    axes = [figure.add_subplot(1, 5, index + 1, projection="3d") for index in range(5)]
    instruction = str(row["model_instruction"] or row["instruction"])
    metrics = data["title_metrics"]
    quality = (
        f"{prediction_key} FK jerk={metrics['prediction_jerk_mps3']:.1f} "
        f"FK skate={metrics['foot_skate_ratio']:.3f} "
        f"contact={metrics['foot_contact_consistency']:.3f}"
    )
    figure.suptitle(
        f"{row['pair_id']} | {row['system']} | {instruction[:96]} | {quality}",
        fontsize=9,
    )

    def update(frame: int) -> None:
        for axis in axes:
            axis.clear()
        _setup_axis(axes[0], lower, upper, "Condition source (FK)")
        _setup_axis(axes[1], lower, upper, "Reference source (metric FK)")
        _setup_axis(axes[2], lower, upper, "Target (FK)")
        _setup_axis(axes[3], lower, upper, f"Prediction ({prediction_key}, FK)")
        _setup_axis(axes[4], lower, upper, "Reference/target/pred FK + control")
        if conditioning_source is None:
            axes[0].text2D(
                0.5,
                0.5,
                "ABSENT",
                transform=axes[0].transAxes,
                ha="center",
                va="center",
                fontsize=14,
                color="#991b1b",
            )
        else:
            _draw(axes[0], conditioning_source[frame], "#7c3aed", 1.0, 2.0)
            axes[0].text2D(
                0.02,
                0.02,
                "normalized-time display",
                transform=axes[0].transAxes,
                fontsize=7,
            )
        _draw(axes[1], reference_source[frame], "#6b7280", 1.0, 2.0)
        _draw(axes[2], target[frame], "#059669", 1.0, 2.0)
        _draw(axes[3], prediction[frame], "#2563eb", 1.0, 2.0)
        _draw(axes[4], reference_source[frame], "#6b7280", 0.30, 1.2)
        _draw(axes[4], target[frame], "#059669", 0.80, 1.8)
        _draw(axes[4], prediction[frame], "#2563eb", 0.80, 1.8)

        changed_target = np.linalg.norm(
            target[frame] - reference_source[frame], axis=-1
        )
        changed_prediction = np.linalg.norm(
            prediction[frame] - reference_source[frame], axis=-1
        )
        joint_ids = np.argsort(np.maximum(changed_target, changed_prediction))[-5:]
        for joint_id in joint_ids:
            source_point = reference_source[frame, joint_id]
            for endpoint, color in (
                (target[frame, joint_id], "#059669"),
                (prediction[frame, joint_id], "#2563eb"),
            ):
                axes[4].plot(
                    [source_point[0], endpoint[0]],
                    [source_point[2], endpoint[2]],
                    [source_point[1], endpoint[1]],
                    color=color,
                    alpha=0.55,
                    linewidth=1.0,
                )

        controlled = control_mask[frame]
        if controlled.any():
            points = target[frame, controlled]
            for axis in (axes[2], axes[3], axes[4]):
                axis.scatter(
                    points[:, 0],
                    points[:, 2],
                    points[:, 1],
                    color="#dc2626",
                    s=48,
                    depthshade=False,
                )
        for joint_id in np.flatnonzero(control_mask.any(axis=0)):
            all_valid = control_mask[:, joint_id]
            full_trail = target[:, joint_id][all_valid]
            if full_trail.shape[0] > 1:
                for axis in (axes[2], axes[3], axes[4]):
                    axis.plot(
                        full_trail[:, 0],
                        full_trail[:, 2],
                        full_trail[:, 1],
                        color="#dc2626",
                        alpha=0.22,
                        linewidth=1.2,
                    )
        if controlled.any():
            trail_start = max(0, frame - trail_frames)
            for joint_id in np.flatnonzero(
                control_mask[trail_start : frame + 1].any(axis=0)
            ):
                valid = control_mask[trail_start : frame + 1, joint_id]
                trail = target[trail_start : frame + 1, joint_id][valid]
                if trail.shape[0] > 1:
                    axes[4].plot(
                        trail[:, 0],
                        trail[:, 2],
                        trail[:, 1],
                        color="#dc2626",
                        alpha=0.55,
                        linewidth=1.5,
                    )

        pred_contact = "".join("1" if value else "0" for value in predicted_contacts[frame])
        target_contact = "".join("1" if value else "0" for value in target_contacts[frame])
        controlled_values = [
            str(int(observed_contacts[frame, index]))
            if controlled_contacts[frame, index]
            else "-"
            for index in range(4)
        ]
        axes[3].text2D(
            0.02,
            0.02,
            f"contact pred/gt: {pred_contact}/{target_contact}\ncontrol: {''.join(controlled_values)}",
            transform=axes[3].transAxes,
            fontsize=7,
        )

    animation_object = FuncAnimation(
        figure,
        update,
        frames=prediction.shape[0],
        interval=1000.0 / max(fps // stride, 1),
        repeat=False,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output_fps = max(fps // stride, 1)
    if save_path.suffix.lower() == ".gif":
        animation_object.save(save_path, writer=animation.PillowWriter(fps=output_fps))
    else:
        animation_object.save(save_path, fps=output_fps)
    plt.close(figure)
    return metrics


def _replay_frozen_selection(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        all_rows,
        key=lambda row: (
            row["system"],
            row.get("control_subtype") or "",
            row["pair_id"],
        ),
    )
    selected_keys: set[str] = set()
    for system in ALL_SYSTEMS:
        for relation, limit in (("equal", 4), ("unequal", 2)):
            candidates = [
                row
                for row in ordered
                if row["system"] == system
                and (
                    row["length_relation"] == "equal"
                    if relation == "equal"
                    else row["length_relation"] != "equal"
                )
            ]
            selected_keys.update(row["case_key"] for row in candidates[:limit])
    subtypes = sorted(
        {
            str(row["control_subtype"])
            for row in ordered
            if row.get("control_subtype") is not None
        }
    )
    for system in CONTROL_SYSTEMS:
        for subtype in subtypes:
            candidates = [
                row
                for row in ordered
                if row["system"] == system and row.get("control_subtype") == subtype
            ]
            if candidates:
                selected_keys.add(candidates[0]["case_key"])
    return [row for row in ordered if row["case_key"] in selected_keys]


def _stratified_render_cases(
    cases: list[dict[str, Any]], *, max_cases: int
) -> list[dict[str, Any]]:
    if max_cases <= 0 or max_cases >= len(cases):
        return cases
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()

    def add_first(predicate: Any) -> None:
        for row in cases:
            if row["case_key"] not in selected_keys and predicate(row):
                selected.append(row)
                selected_keys.add(row["case_key"])
                return

    systems = [system for system in ALL_SYSTEMS if any(r["system"] == system for r in cases)]
    for system in systems:
        if system not in CONTROL_SYSTEMS:
            add_first(lambda row, system=system: row["system"] == system)
    for subtype in CONTACT_CONTROL_SUBTYPES:
        for system in CONTROL_SYSTEMS:
            add_first(
                lambda row, system=system, subtype=subtype: row["system"] == system
                and row.get("control_subtype") == subtype
            )
    for system in systems:
        add_first(lambda row, system=system: row["system"] == system)
    for row in cases:
        if row["case_key"] not in selected_keys:
            selected.append(row)
            selected_keys.add(row["case_key"])
    return selected[:max_cases]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection_sha256", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prediction_key", choices=["raw", "exact"], default="raw")
    parser.add_argument("--systems", default="")
    parser.add_argument("--max_cases", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--trail_frames", type=int, default=12)
    parser.add_argument("--format", choices=["gif", "mp4", "auto"], default="auto")
    args = parser.parse_args()

    selection_path = Path(args.selection).expanduser().resolve()
    selection_sha = _sha256_file(selection_path)
    if selection_sha != str(args.selection_sha256).lower():
        raise RuntimeError("MotionFix visual selection SHA256 mismatch")
    payload = _load_json_object_strict(selection_path)
    if payload.get("format") != SELECTION_FORMAT:
        raise RuntimeError("MotionFix visual selection format mismatch")
    if int(payload.get("case_count", -1)) != len(payload.get("cases", [])):
        raise RuntimeError("MotionFix visual selection count mismatch")
    renderer_identity = _validate_renderer_identity(payload.get("renderer"))
    all_cases = payload["all_cases_manifest"]
    if (
        not Path(all_cases["path"]).is_file()
        or _sha256_file(all_cases["path"]) != all_cases["sha256"]
    ):
        raise RuntimeError("MotionFix all-cases visual manifest changed")
    all_rows = [
        json.loads(line)
        for line in Path(all_cases["path"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(all_rows) != int(all_cases.get("count", -1)):
        raise RuntimeError("MotionFix all-cases visual manifest count mismatch")
    replayed = _replay_frozen_selection(all_rows)
    if (
        payload.get("selection_policy") != SELECTION_POLICY
        or payload.get("cases") != replayed
        or payload.get("selected_case_keys_sha256")
        != _canonical_sha([row["case_key"] for row in replayed])
    ):
        raise RuntimeError("MotionFix visual selection policy replay mismatch")

    cases = list(payload["cases"])
    requested_systems_ordered = tuple(
        value.strip() for value in args.systems.split(",") if value.strip()
    )
    requested_systems = set(requested_systems_ordered)
    if requested_systems:
        cases = [row for row in cases if row["system"] in requested_systems]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_format = args.format
    if output_format == "auto":
        output_format = "mp4" if shutil.which("ffmpeg") else "gif"
    parameters = {
        "decoder": DECODER_PROTOCOL,
        "prediction_key": args.prediction_key,
        "title_metric_protocol": TITLE_METRIC_PROTOCOL,
        "systems": list(requested_systems_ordered),
        "max_cases": max(int(args.max_cases), 0),
        "fps": max(int(args.fps), 1),
        "stride": max(int(args.stride), 1),
        "trail_frames": max(int(args.trail_frames), 0),
        "output_format": output_format,
    }
    cases = _stratified_render_cases(cases, max_cases=parameters["max_cases"])
    parameters["selected_case_keys"] = [row["case_key"] for row in cases]
    artifact_index_path = selection_path.parent / "artifact_index.json"
    artifact_index = _load_json_object_strict(artifact_index_path)
    indexed_selection = _validate_artifact_index(
        artifact_index_path,
        artifact_index,
        expected_selection_path=selection_path,
        expected_selection_sha256=selection_sha,
    )
    if not _json_equal_strict(indexed_selection, payload):
        raise RuntimeError("CLI and artifact-index visual selections differ")
    artifact_view_id = _resolve_required_render_view(artifact_index, parameters)
    if artifact_view_id is not None:
        expected_view = {
            view["view_id"]: view for view in _required_render_views()
        }[artifact_view_id]
        cases = _view_cases(payload, expected_view)
        parameters = _expected_render_parameters(payload, expected_view)
    written = []
    for row in cases:
        save_path = output_dir / f"{row['case_key']}.{output_format}"
        title_metrics = render_case(
            row,
            save_path,
            prediction_key=parameters["prediction_key"],
            fps=parameters["fps"],
            stride=parameters["stride"],
            trail_frames=parameters["trail_frames"],
        )
        written.append(
            {
                "case_key": row["case_key"],
                "path": str(save_path),
                "sha256": _sha256_file(save_path),
                "size": int(save_path.stat().st_size),
                "title_metric_protocol": TITLE_METRIC_PROTOCOL,
                "title_metrics": title_metrics,
                "inputs": {
                    "assets": row["assets"],
                    "aligned_reference_source": row["aligned_reference_source"],
                    "motion_output": row["motion_output"],
                },
            }
        )
    manifest = {
        "format": RENDER_FORMAT,
        "artifact_view_id": artifact_view_id,
        "selection": {
            "path": str(selection_path),
            "sha256": selection_sha,
            "format": SELECTION_FORMAT,
        },
        "renderer": renderer_identity,
        "parameters": parameters,
        "videos": written,
    }
    manifest_path = output_dir / "render_manifest.json"
    _atomic_json(manifest_path, manifest)
    registered_status = "ad_hoc_unregistered"
    if artifact_view_id is not None:
        updated_index = _register_render_manifest(
            artifact_index_path=artifact_index_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        registered_status = str(updated_index["status"])
    print(
        json.dumps(
            {
                "render_manifest": str(manifest_path),
                "artifact_view_id": artifact_view_id,
                "artifact_index": str(artifact_index_path),
                "artifact_index_status": registered_status,
                "videos": len(written),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
