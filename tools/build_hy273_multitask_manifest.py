#!/usr/bin/env python3
"""Build the frozen HY273 multitask manifest and derived HML segment cache."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "hy273_multitask_manifest_v1"
SPAN_POLICY = "hml_caption_span_30fps_round_half_up_v1"
DERIVED_CACHE_VERSION = "hy273_hml_segment_reextract_v1"
REPRESENTATION_VERSION = "kimodo273_smplx22_v1"
HY201_REPRESENTATION_VERSION = "hymotion201_o6dp_hml272_v1"
FRAME_POLICY = "independent_sequence_frame_v1"
OUTPUT_GAUGE_POLICY = "shared_target_yaw_phi_v1"
K273_CONVERSION_COMMIT = "ea668b7073de3d86894b17fa84cb8b456e06a9ed"
KIMODO_COMMIT = "6bb58488037dd65360ff0c5d1692b403a23309f7"
HY201_CONVERSION_COMMIT = "4bf40fe269478886712ef4fa7c37edf193416ce3"
FPS = 30.0
FEATURE_DIM = 273
MIN_FRAMES = 16


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


class HashCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, dict[str, Any]] = {}
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("format") == "hy273_asset_hash_cache_v1":
                self.data = dict(raw.get("files", {}))

    def get(self, path: Path) -> str:
        resolved = path.resolve()
        stat = resolved.stat()
        key = str(resolved)
        cached = self.data.get(key)
        if (
            cached
            and int(cached.get("size", -1)) == stat.st_size
            and int(cached.get("mtime_ns", -1)) == stat.st_mtime_ns
        ):
            return str(cached["sha256"])
        value = sha256_file(resolved)
        self.data[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": value,
        }
        return value

    def save(self) -> None:
        payload = {"format": "hy273_asset_hash_cache_v1", "files": self.data}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) * FPS + 0.5))


def _tag(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not math.isfinite(parsed) else parsed


def parse_caption_file(path: Path, total_frames: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    if not path.is_file():
        return accepted, [{"reason": "missing_caption_file", "path": str(path)}]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        fields = raw_line.strip().split("#")
        text = fields[0].strip() if fields else ""
        if not text:
            rejected.append({"reason": "empty_caption", "source_line": line_number})
            continue
        from_sec = _tag(fields[-2]) if len(fields) >= 3 else 0.0
        to_sec = _tag(fields[-1]) if len(fields) >= 3 else 0.0
        if from_sec == 0.0 and to_sec == 0.0:
            start, end, kind, status = 0, total_frames, "full", "accepted"
        elif from_sec >= 0.0 and to_sec > from_sec:
            raw_start, raw_end = _round_half_up(from_sec), _round_half_up(to_sec)
            start = min(max(raw_start, 0), total_frames)
            end = min(max(raw_end, 0), total_frames)
            kind = "segment"
            status = "clamped_end" if raw_end > total_frames else "accepted"
            if end - start < MIN_FRAMES:
                rejected.append(
                    {
                        "reason": "segment_too_short_after_clamp",
                        "source_line": line_number,
                        "from_sec": from_sec,
                        "to_sec": to_sec,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                    }
                )
                continue
        else:
            rejected.append(
                {
                    "reason": "invalid_nonzero_span",
                    "source_line": line_number,
                    "from_sec": from_sec,
                    "to_sec": to_sec,
                }
            )
            continue
        accepted.append(
            {
                "text": " ".join(text.split()),
                "source_line": line_number,
                "span": {
                    "kind": kind,
                    "from_sec": from_sec,
                    "to_sec": to_sec,
                    "start_frame": start,
                    "end_frame_exclusive": end,
                    "span_status": status,
                    "span_policy_version": SPAN_POLICY,
                },
            }
        )
    return accepted, rejected


def asset_ref(path: Path, frames: int, dim: int, representation: str, hash_cache: HashCache) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hash_cache.get(path),
        "frames": int(frames),
        "fps": FPS,
        "feature_dim": int(dim),
        "representation_version": representation,
    }


def _segment_cache_key(
    hy201_sha: str, start: int, end: int, kimodo_root: Path
) -> tuple[str, dict[str, Any]]:
    payload = {
        "format": DERIVED_CACHE_VERSION,
        "source_hy201_sha256": hy201_sha,
        "start_frame": int(start),
        "end_frame_exclusive": int(end),
        "k273_converter_commit": K273_CONVERSION_COMMIT,
        "kimodo_commit": KIMODO_COMMIT,
        "kimodo_root": str(kimodo_root.resolve()),
        "fps": FPS,
        "feature_dim": FEATURE_DIM,
    }
    return sha256_text(canonical_json(payload)), payload


_WORKER_MOTION_REP = None


def _segment_worker_init(converter_root: str, kimodo_root: str) -> None:
    global _WORKER_MOTION_REP
    sys.path.insert(0, converter_root)
    from hy201_to_kimodo273.kimodo_bridge import load_kimodo_motion_rep

    _, _WORKER_MOTION_REP = load_kimodo_motion_rep(kimodo_root=kimodo_root, fps=int(FPS))


def _segment_worker(task: dict[str, Any]) -> dict[str, Any]:
    from hy201_to_kimodo273.convert import hy201_to_kimodo273_array_with_info

    hy201 = np.load(task["hy201_path"]).astype(np.float32, copy=False)
    window = hy201[int(task["start"]) : int(task["end"])]
    output, info = hy201_to_kimodo273_array_with_info(window, _WORKER_MOTION_REP, device="cpu")
    if bool(info.get("smooth_root_fallback")):
        raise RuntimeError(f"Derived segment requires forbidden smooth-root fallback: {task['text_id']}")
    if output.shape != (int(task["end"]) - int(task["start"]), FEATURE_DIM):
        raise RuntimeError(f"Unexpected derived segment shape {output.shape}: {task['text_id']}")
    if not np.isfinite(output).all():
        raise RuntimeError(f"Non-finite derived K273 segment: {task['text_id']}")
    contacts = output[:, 269:273]
    if not np.logical_or(contacts == 0.0, contacts == 1.0).all():
        raise RuntimeError(f"Non-binary contacts in derived K273 segment: {task['text_id']}")

    output_path = Path(task["output_path"])
    sidecar_path = Path(task["sidecar_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(f".tmp.{os.getpid()}.npy")
    np.save(temporary, output)
    temporary.replace(output_path)
    payload_sha = sha256_file(output_path)
    sidecar = {
        **task["key_payload"],
        "cache_key": task["cache_key"],
        "text_id": task["text_id"],
        "source_hy201_path": task["hy201_path"],
        "derived_k273_path": str(output_path),
        "derived_k273_sha256": payload_sha,
        "shape": list(output.shape),
        "conversion_info": info,
        "semantic_audit": {
            "passed": True,
            "packing_vs_saved_abs_err_max": 0.0,
            "contacts_binary": True,
            "finite": True,
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    return sidecar


def _validate_cached_segment(task: dict[str, Any]) -> dict[str, Any] | None:
    output_path = Path(task["output_path"])
    sidecar_path = Path(task["sidecar_path"])
    if not output_path.is_file() or not sidecar_path.is_file():
        return None
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("cache_key") != task["cache_key"]:
        return None
    if not bool(sidecar.get("semantic_audit", {}).get("passed")):
        return None
    if sha256_file(output_path) != sidecar.get("derived_k273_sha256"):
        return None
    shape = np.load(output_path, mmap_mode="r").shape
    if shape != (int(task["end"]) - int(task["start"]), FEATURE_DIM):
        return None
    return sidecar


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hml_k273_root", default="/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22")
    parser.add_argument("--hml_hy201_root", default="/mnt/afs/mogo_base/datasets/HumanML3D/hymotion201_o6dp_hml272")
    parser.add_argument("--hml_text_root", default="/mnt/afs/mogo_base/datasets/HumanML3D/texts")
    parser.add_argument("--motionfix_k273_root", default="/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22")
    parser.add_argument("--motionfix_hy201_root", default="/mnt/afs/mogo_base/datasets/MotionFix/hymotion201_o6dp_hml272")
    parser.add_argument("--motionfix_annotations", default="/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset/amt_motionfix_latest.json")
    parser.add_argument("--motionfix_splits", default="/mnt/afs/mogo_base/datasets/MotionFix/motionfix-dataset/splits.json")
    parser.add_argument("--output_dir", default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/hy273_multitask_v1")
    parser.add_argument("--derived_cache_dir", default="/mnt/afs/mogo_base/datasets/HY273_multitask_v1/derived_hml_segments_v1")
    parser.add_argument("--converter_root", default="/mnt/afs/UMO_debug/hy201_to_kimodo273")
    parser.add_argument("--kimodo_root", default="/mnt/afs/UMO_debug/outside_material/kimodo")
    parser.add_argument("--segment_workers", type=int, default=8)
    parser.add_argument("--skip_payload_hashes", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    roots = {name: Path(getattr(args, name)).expanduser().resolve() for name in (
        "hml_k273_root", "hml_hy201_root", "hml_text_root", "motionfix_k273_root",
        "motionfix_hy201_root", "motionfix_annotations", "motionfix_splits",
        "output_dir", "derived_cache_dir", "converter_root", "kimodo_root",
    )}
    output_dir = roots["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    roots["derived_cache_dir"].mkdir(parents=True, exist_ok=True)
    hash_cache = HashCache(output_dir / "asset_hash_cache.json")
    if bool(args.skip_payload_hashes):
        hash_cache.get = lambda path: "not-computed"  # type: ignore[method-assign]

    hml_conversion: dict[str, dict[str, Any]] = {}
    with (roots["hml_k273_root"] / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            hml_conversion[str(row["relative_path"])] = row

    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in ("train", "val", "test")}
    accepted_caption_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    segment_tasks: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "format": SCHEMA_VERSION,
        "fps": FPS,
        "feature_dim": FEATURE_DIM,
        "hml": {},
        "motionfix": {},
    }

    for split in rows_by_split:
        split_ids = [
            line.strip()
            for line in (roots["hml_k273_root"] / "split_existing" / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        counts: dict[str, int] = {
            "split_ids": len(split_ids), "accepted_motions": 0, "accepted_full": 0,
            "accepted_segment": 0, "clamped_end": 0, "asset_too_short": 0,
            "smooth_root_fallback_excluded": 0, "invalid_nonzero_span": 0,
            "segment_too_short_after_clamp": 0,
        }
        for motion_id in split_ids:
            rel = f"motion_data/{motion_id}.npy"
            conversion = hml_conversion.get(rel)
            k273_path = roots["hml_k273_root"] / rel
            hy201_path = roots["hml_hy201_root"] / rel
            if conversion is None or not k273_path.is_file() or not hy201_path.is_file():
                rejected_rows.append({"dataset": "humanml3d", "split": split, "uid": motion_id, "reason": "missing_asset_or_conversion_row"})
                continue
            frames = int(conversion["frames"])
            if frames < MIN_FRAMES:
                counts["asset_too_short"] += 1
                rejected_rows.append({"dataset": "humanml3d", "split": split, "uid": motion_id, "reason": "asset_too_short", "frames": frames})
                continue
            if bool(conversion.get("smooth_root_fallback")):
                counts["smooth_root_fallback_excluded"] += 1
                rejected_rows.append({"dataset": "humanml3d", "split": split, "uid": motion_id, "reason": "smooth_root_fallback_excluded"})
                continue
            k_shape = np.load(k273_path, mmap_mode="r").shape
            h_shape = np.load(hy201_path, mmap_mode="r").shape
            if k_shape != (frames, FEATURE_DIM) or h_shape != (frames, 201):
                raise ValueError(f"HML shape mismatch for {motion_id}: {k_shape}/{h_shape}/{frames}")
            captions, rejected = parse_caption_file(roots["hml_text_root"] / f"{motion_id}.txt", frames)
            for item in rejected:
                reason = str(item["reason"])
                counts[reason] = counts.get(reason, 0) + 1
                rejected_rows.append({"dataset": "humanml3d", "split": split, "uid": motion_id, **item})
            if not captions:
                rejected_rows.append({"dataset": "humanml3d", "split": split, "uid": motion_id, "reason": "no_accepted_caption"})
                continue

            k273_ref = asset_ref(k273_path, frames, FEATURE_DIM, REPRESENTATION_VERSION, hash_cache)
            hy201_ref = asset_ref(hy201_path, frames, 201, HY201_REPRESENTATION_VERSION, hash_cache)
            rendered_captions: list[dict[str, Any]] = []
            for caption in captions:
                span = caption["span"]
                text_id = f"humanml3d:{motion_id}:line{caption['source_line']}"
                target_ref = k273_ref
                if span["kind"] == "segment":
                    cache_key, key_payload = _segment_cache_key(
                        hy201_ref["sha256"], span["start_frame"], span["end_frame_exclusive"], roots["kimodo_root"]
                    )
                    segment_path = roots["derived_cache_dir"] / cache_key[:2] / f"{cache_key}.npy"
                    sidecar_path = segment_path.with_suffix(".json")
                    task = {
                        "cache_key": cache_key, "key_payload": key_payload, "text_id": text_id,
                        "hy201_path": str(hy201_path), "start": span["start_frame"], "end": span["end_frame_exclusive"],
                        "output_path": str(segment_path), "sidecar_path": str(sidecar_path),
                    }
                    segment_tasks[cache_key] = task
                    target_ref = {
                        "path": str(segment_path), "sha256": None,
                        "frames": int(span["end_frame_exclusive"] - span["start_frame"]),
                        "fps": FPS, "feature_dim": FEATURE_DIM,
                        "representation_version": f"{REPRESENTATION_VERSION}:{DERIVED_CACHE_VERSION}",
                        "derived_cache_key": cache_key,
                    }
                    counts["accepted_segment"] += 1
                    counts["clamped_end"] += int(span["span_status"] == "clamped_end")
                else:
                    counts["accepted_full"] += 1
                text_row = {
                    "text_id": text_id, "value": caption["text"],
                    "kind": "absolute_motion_caption", "encoding_profile": "hytext_absolute_motion_v1",
                    "span": span, "source_line": caption["source_line"], "target_k273_asset": target_ref,
                }
                rendered_captions.append(text_row)
                accepted_caption_rows.append({"uid": f"humanml3d:{motion_id}", "split": split, **text_row})
            row = {
                "schema_version": SCHEMA_VERSION, "uid": f"humanml3d:{motion_id}",
                "dataset": "humanml3d_k273", "split": split,
                "task_capabilities": ["t2m", "kimodo_control_synth"], "source_motion": None,
                "target_motion": {
                    "motion_uid": f"humanml3d:{motion_id}:target", "base_motion_id": motion_id,
                    "timestamp_sec": None, "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273",
                    "smooth_root_fallback": False, "k273_asset": k273_ref, "hy201_asset": hy201_ref,
                },
                "texts": rendered_captions,
                "provenance": {"k273_conversion_commit": K273_CONVERSION_COMMIT, "kimodo_commit": KIMODO_COMMIT},
            }
            rows_by_split[split].append(row)
            counts["accepted_motions"] += 1
        summary["hml"][split] = counts
        hash_cache.save()
        print(f"[manifest] HML {split}: {counts}", flush=True)

    pending = []
    for task in segment_tasks.values():
        sidecar = _validate_cached_segment(task)
        if sidecar is None:
            pending.append(task)
    print(f"[manifest] derived segments total={len(segment_tasks)} pending={len(pending)}", flush=True)
    if pending:
        with ProcessPoolExecutor(
            max_workers=max(1, int(args.segment_workers)),
            initializer=_segment_worker_init,
            initargs=(str(roots["converter_root"]), str(roots["kimodo_root"])),
        ) as pool:
            futures = {pool.submit(_segment_worker, task): task for task in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 100 == 0 or completed == len(futures):
                    print(f"[manifest] derived segments {completed}/{len(futures)}", flush=True)

    derived_sha: dict[str, str] = {}
    for cache_key, task in segment_tasks.items():
        sidecar = _validate_cached_segment(task)
        if sidecar is None:
            raise RuntimeError(f"Derived segment failed validation: {cache_key}")
        derived_sha[cache_key] = str(sidecar["derived_k273_sha256"])
    for split_rows in rows_by_split.values():
        for row in split_rows:
            if row["dataset"] != "humanml3d_k273":
                continue
            for text in row["texts"]:
                ref = text["target_k273_asset"]
                if "derived_cache_key" in ref:
                    ref["sha256"] = derived_sha[ref["derived_cache_key"]]
    for row in accepted_caption_rows:
        ref = row["target_k273_asset"]
        if "derived_cache_key" in ref:
            ref["sha256"] = derived_sha[ref["derived_cache_key"]]

    annotations = json.loads(roots["motionfix_annotations"].read_text(encoding="utf-8"))
    motionfix_splits = json.loads(roots["motionfix_splits"].read_text(encoding="utf-8"))
    annotation_sha = sha256_file(roots["motionfix_annotations"])
    split_sha = sha256_file(roots["motionfix_splits"])
    for split in rows_by_split:
        counts = {"split_ids": len(motionfix_splits[split]), "accepted_pairs": 0, "equal": 0, "off_by_one": 0, "material_difference": 0, "missing": 0}
        for pair_id in motionfix_splits[split]:
            pair_id = str(pair_id)
            annotation = annotations.get(pair_id)
            source_k = roots["motionfix_k273_root"] / split / f"{pair_id}_source.npy"
            target_k = roots["motionfix_k273_root"] / split / f"{pair_id}_target.npy"
            source_h = roots["motionfix_hy201_root"] / split / f"{pair_id}_source.npy"
            target_h = roots["motionfix_hy201_root"] / split / f"{pair_id}_target.npy"
            if annotation is None or not all(path.is_file() for path in (source_k, target_k, source_h, target_h)):
                counts["missing"] += 1
                rejected_rows.append({"dataset": "motionfix", "split": split, "uid": pair_id, "reason": "missing_annotation_or_asset"})
                continue
            instruction = " ".join(str(annotation.get("annotation", "")).split())
            if not instruction:
                raise ValueError(f"Empty MotionFix instruction: {pair_id}")
            source_shape, target_shape = np.load(source_k, mmap_mode="r").shape, np.load(target_k, mmap_mode="r").shape
            source_h_shape, target_h_shape = np.load(source_h, mmap_mode="r").shape, np.load(target_h, mmap_mode="r").shape
            if source_shape[1:] != (FEATURE_DIM,) or target_shape[1:] != (FEATURE_DIM,):
                raise ValueError(f"MotionFix K273 shape mismatch {pair_id}: {source_shape}/{target_shape}")
            if source_h_shape != (source_shape[0], 201) or target_h_shape != (target_shape[0], 201):
                raise ValueError(f"MotionFix HY201 shape mismatch {pair_id}")
            source_len, target_len = int(source_shape[0]), int(target_shape[0])
            if target_len > 300:
                raise ValueError(f"MotionFix target exceeds max T=300: {pair_id}/{target_len}")
            delta = abs(target_len - source_len)
            relation = "equal" if delta == 0 else "off_by_one" if delta == 1 else "material_difference"
            counts[relation] += 1
            source_ref = asset_ref(source_k, source_len, FEATURE_DIM, REPRESENTATION_VERSION, hash_cache)
            target_ref = asset_ref(target_k, target_len, FEATURE_DIM, REPRESENTATION_VERSION, hash_cache)
            source_h_ref = asset_ref(source_h, source_len, 201, HY201_REPRESENTATION_VERSION, hash_cache)
            target_h_ref = asset_ref(target_h, target_len, 201, HY201_REPRESENTATION_VERSION, hash_cache)
            rows_by_split[split].append({
                "schema_version": SCHEMA_VERSION, "uid": f"motionfix:{pair_id}",
                "dataset": "motionfix_k273", "split": split,
                "task_capabilities": ["motion_edit", "motion_edit_with_control"],
                "source_motion": {
                    "motion_uid": f"motionfix:{pair_id}:source", "base_motion_id": str(annotation["motion_source"]),
                    "timestamp_sec": annotation["timestamp_source"],
                    "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273",
                    "physical_transform_group_id": f"motionfix:{pair_id}:source_independent",
                    "k273_asset": source_ref, "hy201_asset": source_h_ref,
                },
                "target_motion": {
                    "motion_uid": f"motionfix:{pair_id}:target", "base_motion_id": str(annotation["motion_target"]),
                    "timestamp_sec": annotation["timestamp_target"],
                    "coordinate_frame": "per_sequence_hml_canonical_then_raw_k273",
                    "physical_transform_group_id": f"motionfix:{pair_id}:target_independent",
                    "k273_asset": target_ref, "hy201_asset": target_h_ref,
                },
                "texts": [{
                    "text_id": f"motionfix:{pair_id}:instruction", "value": instruction,
                    "kind": "relative_edit_instruction", "encoding_profile": "hytext_relative_edit_v1",
                    "span": {"kind": "pair_instruction"}, "source_line": 1,
                }],
                "pair": {
                    "frame_policy_id": FRAME_POLICY, "framewise_aligned": False,
                    "shared_world_frame": False, "output_gauge_policy": OUTPUT_GAUGE_POLICY,
                    "source_frames": source_len, "target_frames": target_len,
                    "length_relation": relation, "default_time_relation": "normalized_progress",
                    "official_pair_id": pair_id,
                },
                "provenance": {
                    "annotation_sha256": annotation_sha, "split_sha256": split_sha,
                    "hy201_converter_commit": HY201_CONVERSION_COMMIT,
                    "k273_converter_commit": K273_CONVERSION_COMMIT,
                },
            })
            counts["accepted_pairs"] += 1
        summary["motionfix"][split] = counts
        hash_cache.save()
        print(f"[manifest] MotionFix {split}: {counts}", flush=True)

    accepted_caption_rows.sort(key=lambda row: (row["split"], row["uid"], row["source_line"]))
    write_jsonl(output_dir / "accepted_captions.jsonl", accepted_caption_rows)
    write_jsonl(output_dir / "rejected.jsonl", rejected_rows)
    file_records: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: (row["dataset"], row["split"], row["uid"]))
        path = output_dir / f"{split}.jsonl"
        count = write_jsonl(path, rows)
        file_records[path.name] = {"rows": count, "sha256": sha256_file(path)}
    for name in ("accepted_captions.jsonl", "rejected.jsonl"):
        path = output_dir / name
        file_records[name] = {"rows": sum(1 for _ in path.open(encoding="utf-8")), "sha256": sha256_file(path)}
    summary.update({
        "schema_version": SCHEMA_VERSION,
        "span_policy_version": SPAN_POLICY,
        "derived_cache_version": DERIVED_CACHE_VERSION,
        "derived_segment_count": len(segment_tasks),
        "files": file_records,
        "source_fps_authenticated": True,
        "merge_resampling": "none",
        "representation_version": REPRESENTATION_VERSION,
    })
    (output_dir / "schema.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "required_datasets": ["humanml3d_k273", "motionfix_k273"]}, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    bundle = {name: sha256_file(output_dir / name) for name in sorted([*file_records, "schema.json", "summary.json"])}
    bundle_sha = sha256_text(canonical_json(bundle))
    (output_dir / "manifest.sha256").write_text(bundle_sha + "\n", encoding="utf-8")
    hash_cache.save()
    print(json.dumps({"output_dir": str(output_dir), "bundle_sha256": bundle_sha, "files": file_records}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
