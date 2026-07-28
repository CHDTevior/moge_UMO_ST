"""Official MotionFix edit retrieval evaluation.

This module keeps MotionFix's evaluator stack external: it imports the
official repo, loads its TMR checkpoint from ``eval-deps``, and reuses the
official motion-to-motion retrieval metric implementation. CodeFlow motions
are decoded as 272D MotionStreamer features, then mapped back to the raw
MotionFix pose layout expected by the official evaluator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import torch

from .edit_masks import build_instruction_edit_preserve_mask
from .eval_t2m import eval_motion_to_vq_space


MOTIONFIX_TO_MOTIONSTREAMER = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
MOTIONSTREAMER_TO_MOTIONFIX = MOTIONFIX_TO_MOTIONSTREAMER.T


def _normalize_np(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(norm, eps)


def rotation_6d_to_matrix_np(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = _normalize_np(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize_np(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2).astype(np.float32)


def matrix_to_rotation_6d_np(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6).astype(np.float32)


def yaw_remove_matrix_np(root_rotation: np.ndarray) -> np.ndarray:
    forward = root_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    yaw = np.arctan2(forward[..., 0], forward[..., 2])
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    heading = np.zeros((*yaw.shape, 3, 3), dtype=np.float32)
    heading[..., 0, 0] = cos
    heading[..., 0, 2] = sin
    heading[..., 1, 1] = 1.0
    heading[..., 2, 0] = -sin
    heading[..., 2, 2] = cos
    return np.swapaxes(heading, -1, -2).astype(np.float32)


def _motionfix_split_file(data_root: Path, split: str) -> Path:
    if split == "train":
        return data_root / "motionfix.pth.tar"
    return data_root / f"motionfix_{split}.pth.tar"


def infer_motionfix_split_from_manifest(manifest_path: Path) -> str:
    stem = manifest_path.stem.lower()
    for split in ("train", "val", "test"):
        if stem.endswith(f"_{split}") or f"_{split}_" in stem:
            return split
    return "test"


def _manifest_data_root(manifest_path: Path) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    if manifest_path.parent.name == "manifests":
        return manifest_path.parent.parent
    return manifest_path.parent


def _resolve_manifest_relative_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (_manifest_data_root(manifest_path) / path).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_jsonable(value: object) -> str:
    import hashlib
    import json

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_motionfix_unified_v2_meta_index(manifest_path: str | Path) -> Dict[str, Dict[str, Any]]:
    import json

    manifest = Path(manifest_path).expanduser().resolve()
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    invalid: List[str] = []
    missing_meta: List[str] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", ""))
            if record.get("conversion_version") != "motionfix_motionstreamer272_unified_v2":
                invalid.append(sample_id or f"line:{line_no}")
                continue
            meta_value = str(record.get("meta", ""))
            if not meta_value:
                missing_meta.append(sample_id or f"line:{line_no}")
                continue
            meta_path = _resolve_manifest_relative_path(meta_value, manifest)
            if not meta_path.is_file():
                raise FileNotFoundError(f"MotionFix unified_v2 meta file not found for {sample_id}: {meta_path}")
            meta = _read_json(meta_path)
            if meta.get("conversion_version") != "motionfix_motionstreamer272_unified_v2":
                raise ValueError(f"Unexpected MotionFix meta conversion_version at {meta_path}: {meta.get('conversion_version')}")
            meta["_meta_path"] = str(meta_path)
            meta_by_id[str(sample_id)] = meta

    if invalid or missing_meta:
        examples = (invalid + missing_meta)[:5]
        raise ValueError(
            "MotionFix official eval now requires the unified_v2 manifest with per-sample meta. "
            f"Found {len(invalid)} non-v2 records and {len(missing_meta)} records without meta in {manifest}; "
            f"examples={examples}. Regenerate/use motionfix_motionstreamer272_unified_v2_*.jsonl."
        )
    if not meta_by_id:
        raise RuntimeError(f"MotionFix unified_v2 manifest is empty: {manifest}")
    return meta_by_id


def infer_motionfix_272_conversion_version(manifest_path: str | Path) -> str:
    import json

    manifest = Path(manifest_path).expanduser().resolve()
    versions: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            versions.add(str(record.get("conversion_version") or "motionfix_motionstreamer272"))
    if not versions:
        raise RuntimeError(f"MotionFix manifest is empty: {manifest}")
    if len(versions) != 1:
        raise ValueError(f"MotionFix manifest mixes conversion versions: {sorted(versions)} at {manifest}")
    return next(iter(versions))


def load_motionfix_unified_v2_rotation_corrections(meta_by_id: Dict[str, Dict[str, Any]]) -> np.ndarray:
    alignment_cache: Dict[Path, Dict[str, Any]] = {}
    corrections: np.ndarray | None = None
    for sample_id, meta in meta_by_id.items():
        alignment_info = meta.get("rotation_alignment")
        if not isinstance(alignment_info, dict):
            raise ValueError(f"MotionFix unified_v2 meta for {sample_id} has no rotation_alignment block")
        alignment_path = Path(str(alignment_info.get("path", ""))).expanduser()
        if not alignment_path.is_absolute():
            meta_path = Path(str(meta.get("_meta_path", ""))).expanduser()
            alignment_path = (meta_path.parent / alignment_path).resolve()
        if not alignment_path.is_file():
            raise FileNotFoundError(f"MotionFix unified_v2 rotation alignment not found for {sample_id}: {alignment_path}")
        if alignment_path not in alignment_cache:
            alignment = _read_json(alignment_path)
            expected_sha = str(alignment.get("sha256", ""))
            actual_sha = _sha256_jsonable({k: v for k, v in alignment.items() if k != "sha256"})
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    f"Rotation alignment sha mismatch at {alignment_path}: expected {expected_sha}, actual {actual_sha}"
                )
            alignment_cache[alignment_path] = alignment
        alignment = alignment_cache[alignment_path]
        meta_sha = str(alignment_info.get("sha256", ""))
        alignment_sha = str(alignment.get("sha256", ""))
        if meta_sha and alignment_sha and meta_sha != alignment_sha:
            raise ValueError(
                f"MotionFix unified_v2 meta/alignment sha mismatch for {sample_id}: meta={meta_sha}, file={alignment_sha}"
            )
        current = np.asarray(alignment["joint_corrections"], dtype=np.float32)
        if current.shape != (22, 3, 3):
            raise ValueError(f"Expected rotation corrections [22,3,3], got {current.shape} at {alignment_path}")
        if corrections is None:
            corrections = current
        elif not np.allclose(corrections, current, atol=1e-6):
            raise ValueError("MotionFix unified_v2 manifest mixes incompatible rotation alignment files")
    if corrections is None:
        raise RuntimeError("No MotionFix unified_v2 rotation corrections loaded")
    return corrections.astype(np.float32)


def ensure_motionfix_imports(motionfix_repo: Path) -> None:
    repo = str(motionfix_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)


class OfficialMotionFixDataset:
    def __init__(self, raw_records: Dict[str, object], keyids: Sequence[str], normalizer, motionfix_repo: Path):
        ensure_motionfix_imports(motionfix_repo)
        self.raw_records = raw_records
        self.keyids = [str(key) for key in keyids if str(key) in raw_records]
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.keyids)

    def load_keyid(self, keyid: str) -> Dict[str, object]:
        from src.data.features import _get_body_orient, _get_body_pose, _get_body_transl_delta_pelv_infer

        item = self.raw_records[str(keyid)]
        source_m = item["motion_source"]
        target_m = item["motion_target"]
        text = item.get("text", "")

        pose6d_src = _get_body_pose(source_m["rots"])
        orient6d_src = _get_body_orient(source_m["rots"][..., :3])
        trans_delta_src = _get_body_transl_delta_pelv_infer(orient6d_src, source_m["trans"])
        features_source = torch.cat([trans_delta_src, pose6d_src, orient6d_src], dim=-1)

        pose6d_tgt = _get_body_pose(target_m["rots"])
        orient6d_tgt = _get_body_orient(target_m["rots"][..., :3])
        trans_delta_tgt = _get_body_transl_delta_pelv_infer(orient6d_tgt, target_m["trans"])
        features_target = torch.cat([trans_delta_tgt, pose6d_tgt, orient6d_tgt], dim=-1)

        if self.normalizer is not None:
            features_source = self.normalizer(features_source)
            features_target = self.normalizer(features_target)

        return {
            "motion_source": features_source,
            "motion_target": features_target,
            "text": text,
            "keyid": str(keyid),
        }


def _raw_rot_mats_motionstreamer(raw_motion: Dict[str, object], transform_body_pose) -> np.ndarray:
    rots = np.asarray(raw_motion["rots"], dtype=np.float32)
    frames = int(rots.shape[0])
    rot_aa = rots.reshape(frames, 22, 3)
    rot_raw = (
        transform_body_pose(torch.from_numpy(rot_aa.reshape(-1, 3)), "aa->rot")
        .detach()
        .cpu()
        .numpy()
        .reshape(frames, 22, 3, 3)
        .astype(np.float32)
    )
    return np.einsum("ij,tkjl,ml->tkim", MOTIONFIX_TO_MOTIONSTREAMER, rot_raw, MOTIONFIX_TO_MOTIONSTREAMER).astype(
        np.float32
    )


def motionstreamer272_to_motionfix_pose(
    motion_272: np.ndarray,
    reference_raw_motion: Dict[str, object],
    transform_body_pose,
) -> np.ndarray:
    """Legacy inverse kept for old offline comparisons.

    The official MotionFix edit evaluator no longer uses this path because it
    reconstructs the gauge from a raw target reference. unified_v2 manifests
    carry the source-frame gauge explicitly in per-sample metadata.
    """
    motion = np.asarray(motion_272, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 272:
        raise ValueError(f"Expected MotionStreamer272 motion [T,272], got {motion.shape}")
    frames = int(motion.shape[0])
    if frames <= 0:
        raise ValueError("Cannot convert an empty motion")

    ref_rots_ms = _raw_rot_mats_motionstreamer(reference_raw_motion, transform_body_pose)
    ref_joints_raw = np.asarray(reference_raw_motion["joint_positions"], dtype=np.float32)
    ref_joints_ms = np.einsum("ij,tkj->tki", MOTIONFIX_TO_MOTIONSTREAMER, ref_joints_raw).astype(np.float32)
    ref_trans_raw = np.asarray(reference_raw_motion["trans"], dtype=np.float32)
    trans_joint_offset = (ref_trans_raw - ref_joints_raw[:, 0]).mean(axis=0)

    yaw0 = yaw_remove_matrix_np(ref_rots_ms[:1, 0])[0]
    origin = ref_joints_ms[0, 0].copy()
    origin[1] = float(ref_joints_ms[:, :, 1].min())

    heading_delta = rotation_6d_to_matrix_np(motion[:, 2:8])
    yaw_rel = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
    for frame in range(1, frames):
        yaw_rel[frame] = heading_delta[frame] @ yaw_rel[frame - 1]
    yaw_abs = np.einsum("tij,jk->tik", yaw_rel, yaw0).astype(np.float32)

    local_vel = np.zeros((frames, 3), dtype=np.float32)
    local_vel[:, 0] = motion[:, 0]
    local_vel[:, 2] = motion[:, 1]
    canonical_root = np.zeros((frames, 3), dtype=np.float32)
    canonical_root[:, 1] = motion[:, 9]
    for frame in range(1, frames):
        canon_vel = yaw_rel[frame - 1].T @ local_vel[frame]
        canonical_root[frame, 0] = canonical_root[frame - 1, 0] + canon_vel[0]
        canonical_root[frame, 2] = canonical_root[frame - 1, 2] + canon_vel[2]
    root_ms = np.einsum("ji,tj->ti", yaw0, canonical_root).astype(np.float32)
    root_joint_ms = root_ms + origin[None]
    root_joint_raw = np.einsum("ij,tj->ti", MOTIONSTREAMER_TO_MOTIONFIX, root_joint_ms).astype(np.float32)
    trans_raw = root_joint_raw + trans_joint_offset[None]

    local_rots = rotation_6d_to_matrix_np(motion[:, 140:272].reshape(frames, 22, 6))
    rot_mats_ms = local_rots.copy()
    rot_mats_ms[:, 0] = np.einsum("tji,tjk->tik", yaw_abs, local_rots[:, 0]).astype(np.float32)
    rot_mats_raw = np.einsum("ia,tkab,bm->tkim", MOTIONSTREAMER_TO_MOTIONFIX, rot_mats_ms, MOTIONFIX_TO_MOTIONSTREAMER)
    rot6d_raw = matrix_to_rotation_6d_np(rot_mats_raw)
    pose = np.concatenate([trans_raw, rot6d_raw[:, 0], rot6d_raw[:, 1:].reshape(frames, 21 * 6)], axis=-1)
    return pose.astype(np.float32)


def motionstreamer272_unified_v2_to_motionfix_pose(
    motion_272: np.ndarray,
    meta: Dict[str, Any],
    rotation_corrections: np.ndarray,
) -> np.ndarray:
    motion = np.asarray(motion_272, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 272:
        raise ValueError(f"Expected MotionFix unified_v2 motion [T,272], got {motion.shape}")
    frames = int(motion.shape[0])
    if frames <= 0:
        raise ValueError("Cannot convert an empty MotionFix unified_v2 motion")
    if meta.get("conversion_version") != "motionfix_motionstreamer272_unified_v2":
        raise ValueError(f"Expected MotionFix unified_v2 meta, got conversion_version={meta.get('conversion_version')}")

    corrections = np.asarray(rotation_corrections, dtype=np.float32)
    if corrections.shape != (22, 3, 3):
        raise ValueError(f"Expected rotation corrections [22,3,3], got {corrections.shape}")

    source_yaw0 = np.asarray(meta["source_initial_yaw_remove_matrix"], dtype=np.float32)
    origin = np.asarray(meta["source_origin_motionstreamer"], dtype=np.float32)
    trans_joint_offset_raw = np.asarray(meta["source_trans_joint_offset_raw"], dtype=np.float32)
    basis_meta = np.asarray(meta.get("basis_motionfix_to_motionstreamer", MOTIONFIX_TO_MOTIONSTREAMER), dtype=np.float32)
    if source_yaw0.shape != (3, 3) or origin.shape != (3,) or trans_joint_offset_raw.shape != (3,):
        raise ValueError(f"Malformed MotionFix unified_v2 meta for sample {meta.get('id', '<unknown>')}")
    if basis_meta.shape != (3, 3) or not np.allclose(basis_meta, MOTIONFIX_TO_MOTIONSTREAMER, atol=1e-6):
        raise ValueError(f"Unexpected MotionFix basis in unified_v2 meta for sample {meta.get('id', '<unknown>')}")

    heading_delta = rotation_6d_to_matrix_np(motion[:, 2:8])
    yaw_rel = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
    yaw_rel[0] = heading_delta[0]
    for frame in range(1, frames):
        yaw_rel[frame] = heading_delta[frame] @ yaw_rel[frame - 1]
    yaw_abs = np.einsum("tij,jk->tik", yaw_rel, source_yaw0).astype(np.float32)

    local_positions = motion[:, 8:74].reshape(frames, 22, 3)
    local_vel = np.zeros((frames, 3), dtype=np.float32)
    local_vel[:, 0] = motion[:, 0]
    local_vel[:, 2] = motion[:, 1]

    canonical_root = np.zeros((frames, 3), dtype=np.float32)
    canonical_root[:, 1] = local_positions[:, 0, 1]
    canonical_root[0, 0] = local_vel[0, 0]
    canonical_root[0, 2] = local_vel[0, 2]
    for frame in range(1, frames):
        canonical_delta = yaw_rel[frame - 1].T @ local_vel[frame]
        canonical_root[frame, 0] = canonical_root[frame - 1, 0] + canonical_delta[0]
        canonical_root[frame, 2] = canonical_root[frame - 1, 2] + canonical_delta[2]

    root_ms = np.einsum("ji,tj->ti", source_yaw0, canonical_root).astype(np.float32)
    root_joint_ms = root_ms + origin[None]
    root_joint_raw = np.einsum("ij,tj->ti", MOTIONSTREAMER_TO_MOTIONFIX, root_joint_ms).astype(np.float32)
    trans_raw = root_joint_raw + trans_joint_offset_raw[None]

    local_rots_unified = rotation_6d_to_matrix_np(motion[:, 140:272].reshape(frames, 22, 6))
    local_rots_source_frame = np.einsum("tkij,kjl->tkil", local_rots_unified, np.swapaxes(corrections, -1, -2)).astype(
        np.float32
    )
    rot_mats_ms = local_rots_source_frame.copy()
    rot_mats_ms[:, 0] = np.einsum("tji,tjk->tik", yaw_abs, local_rots_source_frame[:, 0]).astype(np.float32)
    rot_mats_raw = np.einsum("ia,tkab,bm->tkim", MOTIONSTREAMER_TO_MOTIONFIX, rot_mats_ms, MOTIONFIX_TO_MOTIONSTREAMER)
    rot6d_raw = matrix_to_rotation_6d_np(rot_mats_raw)
    pose = np.concatenate([trans_raw, rot6d_raw[:, 0], rot6d_raw[:, 1:].reshape(frames, 21 * 6)], axis=-1)
    if not np.isfinite(pose).all():
        raise RuntimeError(f"MotionFix unified_v2 inverse produced non-finite values for sample {meta.get('id', '<unknown>')}")
    return pose.astype(np.float32)


def build_motionfix_official_eval_wrapper(
    *,
    motionfix_repo: str,
    motionfix_data_root: str,
    manifest_path: str,
    device: torch.device,
) -> Dict[str, object]:
    motionfix_repo_path = Path(motionfix_repo).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    split = infer_motionfix_split_from_manifest(manifest)
    data_root = Path(motionfix_data_root).expanduser().resolve()
    raw_path = _motionfix_split_file(data_root, split)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Official MotionFix split file not found: {raw_path}")
    conversion_version = infer_motionfix_272_conversion_version(manifest)
    if conversion_version == "motionfix_motionstreamer272_unified_v2":
        meta_by_id = load_motionfix_unified_v2_meta_index(manifest)
        rotation_corrections = load_motionfix_unified_v2_rotation_corrections(meta_by_id)
        inverse_mode = "motionstreamer272_unified_v2"
    elif conversion_version == "motionfix_motionstreamer272":
        meta_by_id = {}
        rotation_corrections = None
        inverse_mode = "motionstreamer272_legacy"
    elif conversion_version == "motionfix_motionstreamer272_hml":
        meta_by_id = {}
        rotation_corrections = None
        inverse_mode = "motionstreamer272_hml"
    else:
        raise ValueError(f"Unsupported MotionFix 272 conversion_version={conversion_version} in {manifest}")
    raw_records = joblib.load(raw_path)

    ensure_motionfix_imports(motionfix_repo_path)

    from src.tmr.data.motionfix_loader import Normalizer
    from src.tmr.load_model import load_model_from_cfg
    from tmr_evaluator.motion2motion_retr import read_config

    repo_eval_deps = motionfix_repo_path / "eval-deps"
    cfg = read_config(str(repo_eval_deps))
    model = load_model_from_cfg(cfg, "last", eval_mode=True, device=str(device))
    normalizer = Normalizer(repo_eval_deps / "stats" / "humanml3d" / "amass_feats")

    return {
        "backend": "motionfix_official",
        "model": model,
        "normalizer": normalizer,
        "raw_records": raw_records,
        "motionfix_inverse_mode": inverse_mode,
        "motionfix_conversion_version": conversion_version,
        "meta_by_id": meta_by_id,
        "rotation_corrections": rotation_corrections,
        "motionfix_repo": motionfix_repo_path,
        "split": split,
        "raw_path": raw_path,
        "manifest_path": manifest,
    }


def _official_generated_features(
    poses: Dict[str, np.ndarray],
    normalizer,
    device: torch.device,
    motionfix_repo: Path,
) -> Dict[str, torch.Tensor]:
    ensure_motionfix_imports(motionfix_repo)
    from src.data.features import _get_body_transl_delta_pelv_infer

    out: Dict[str, torch.Tensor] = {}
    for keyid, pose_np in poses.items():
        pose = torch.from_numpy(np.asarray(pose_np, dtype=np.float32))
        trans = pose[..., :3]
        global_orient_6d = pose[..., 3:9]
        body_pose_6d = pose[..., 9:]
        trans_delta = _get_body_transl_delta_pelv_infer(global_orient_6d, trans)
        features = torch.cat([trans_delta, body_pose_6d, global_orient_6d], dim=-1)
        if normalizer is not None:
            features = normalizer(features)
        out[str(keyid)] = features.to(device)
    return out


def _metrics_from_sim_matrix(sim_matrix: np.ndarray) -> Dict[str, float]:
    from src.tmr.metrics import all_contrastive_metrics_mot2mot

    metrics = all_contrastive_metrics_mot2mot(sim_matrix, rounding=None)
    return {
        "R@1": float(metrics["m2m/R01"]),
        "R@2": float(metrics["m2m/R02"]),
        "R@3": float(metrics["m2m/R03"]),
        "R@5": float(metrics["m2m/R05"]),
        "R@10": float(metrics["m2m/R10"]),
        "MedR": float(metrics["m2m/MedR"]),
        "AvgR": float(metrics["m2m/AvgR"]),
        "len": float(metrics["m2m/len"]),
    }


def _mean_metric_dict(metrics: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = sorted(set().union(*(m.keys() for m in metrics)))
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(m[key]) for m in metrics if key in m]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _flatten_official_metrics(
    full: Dict[str, Dict[str, float]],
    batches: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "eval_backend": "motionfix_official",
    }
    for prefix, metrics in (
        ("source_target_full", full.get("source_target", {})),
        ("target_generated_full", full.get("target_generated", {})),
    ):
        for key, value in metrics.items():
            out[f"motionfix_official_{prefix}_{key}"] = float(value)
    for prefix, metrics in (
        ("source_target_batch", batches.get("source_target", {})),
        ("target_generated_batch", batches.get("target_generated", {})),
    ):
        for key, value in metrics.items():
            out[f"motionfix_official_{prefix}_{key}"] = float(value)

    tg_full = full.get("target_generated", {})
    tg_batch = batches.get("target_generated", {})
    if "R@1" in tg_full:
        out["g2t_full_r1"] = float(tg_full["R@1"]) / 100.0
    if "R@2" in tg_full:
        out["g2t_full_r2"] = float(tg_full["R@2"]) / 100.0
    if "R@3" in tg_full:
        out["g2t_full_r3"] = float(tg_full["R@3"]) / 100.0
        out["rtop3"] = float(tg_full["R@3"])
    if "AvgR" in tg_full:
        out["g2t_full_avgr"] = float(tg_full["AvgR"])
        out["avgr"] = float(tg_full["AvgR"])
    if "R@1" in tg_batch:
        out["g2t_batch_r1"] = float(tg_batch["R@1"]) / 100.0
    if "R@2" in tg_batch:
        out["g2t_batch_r2"] = float(tg_batch["R@2"]) / 100.0
    if "R@3" in tg_batch:
        out["g2t_batch_r3"] = float(tg_batch["R@3"]) / 100.0
    if "AvgR" in tg_batch:
        out["g2t_batch_avgr"] = float(tg_batch["AvgR"])
    return out


@torch.no_grad()
def evaluate_codeflow_global_edit_motionfix_official(
    *,
    loader,
    model,
    official_wrapper: Dict[str, object],
    vq_mean: torch.Tensor,
    vq_std: torch.Tensor,
    cfg,
    repeat_id: int = 0,
) -> Dict[str, object]:
    ensure_motionfix_imports(Path(official_wrapper["motionfix_repo"]))
    from tmr_evaluator.motion2motion_retr import compute_sim_matrix, mat2name

    model.eval()
    official_model = official_wrapper["model"]
    normalizer = official_wrapper["normalizer"]
    raw_records: Dict[str, object] = official_wrapper["raw_records"]  # type: ignore[assignment]
    meta_by_id: Dict[str, Dict[str, Any]] = official_wrapper["meta_by_id"]  # type: ignore[assignment]
    rotation_corrections_value = official_wrapper.get("rotation_corrections")
    rotation_corrections = (
        np.asarray(rotation_corrections_value, dtype=np.float32)
        if rotation_corrections_value is not None
        else None
    )
    motionfix_repo: Path = official_wrapper["motionfix_repo"]  # type: ignore[assignment]
    inverse_mode = str(official_wrapper.get("motionfix_inverse_mode", ""))
    from src.tools.transforms3d import transform_body_pose

    generated_poses: Dict[str, np.ndarray] = {}
    ordered_keyids: List[str] = []
    nb_sample = 0
    mask_generated_frac_sum = 0.0
    mask_preserved_frac_sum = 0.0
    mask_batches = 0

    vq_mean_t = torch.as_tensor(vq_mean, device=model.device, dtype=torch.float32).view(1, 1, -1)
    vq_std_t = torch.as_tensor(vq_std, device=model.device, dtype=torch.float32).view(1, 1, -1)

    for batch_id, batch in enumerate(loader):
        if int(getattr(cfg, "max_batches", 0)) > 0 and batch_id >= int(cfg.max_batches):
            break
        _word_embeddings, _pos_one_hots, instructions, _sent_len, source_pose, target_pose, m_length, _tokens, *extra = batch
        sample_ids = [str(item) for item in extra[0]] if extra else []
        if not sample_ids:
            raise RuntimeError("MotionFix official eval requires sample ids from MotionEditDatasetEval")

        source_pose = source_pose.to(model.device).float()
        target_pose = target_pose.to(model.device).float()
        m_length = m_length.to(model.device).long()

        source_vq_motion = eval_motion_to_vq_space(source_pose, m_length, vq_mean, vq_std, vq_mean, vq_std)
        target_vq_motion = eval_motion_to_vq_space(target_pose, m_length, vq_mean, vq_std, vq_mean, vq_std)
        target_ids, target_embeddings = model.tokenizer.encode(target_vq_motion)
        _source_ids, source_embeddings = model.tokenizer.encode(source_vq_motion)
        token_lengths = (m_length // int(cfg.unit_length)).clamp(min=1, max=target_embeddings.shape[1])

        preserve_mask = None
        if str(getattr(cfg, "mask_mode", "none")) == "instruction":
            preserve_mask, _op_ids, mask_stats = build_instruction_edit_preserve_mask(
                instructions,
                token_lengths=token_lengths,
                latent_len=source_embeddings.shape[1],
                num_parts=source_embeddings.shape[2],
                device=model.device,
                temporal_dilate=int(getattr(cfg, "mask_temporal_dilate", 0)),
            )
            mask_generated_frac_sum += float(mask_stats["global_edit_infer_generated_cell_frac"].detach().cpu())
            mask_preserved_frac_sum += float(mask_stats["global_edit_infer_preserved_cell_frac"].detach().cpu())
            mask_batches += 1
        elif str(getattr(cfg, "mask_mode", "none")) not in {"none", ""}:
            raise ValueError(f"Unsupported global edit eval mask mode: {cfg.mask_mode}")

        pred_motion_vq, _pred_ids = model.generate_global_edit_motion(
            instructions,
            source_embeddings=source_embeddings,
            token_lengths=token_lengths,
            steps=int(cfg.steps),
            cond_scale=float(cfg.cond_scale),
            terminal_mode=cfg.terminal_mode,
            decode_mode=cfg.decode_mode,
            preserve_mask=preserve_mask,
        )
        pred_raw = pred_motion_vq[:, : target_pose.shape[1]].float() * vq_std_t + vq_mean_t
        pred_raw_np = pred_raw.detach().cpu().numpy()
        lengths_np = m_length.detach().cpu().numpy().astype(np.int64)

        for local_idx, sample_id in enumerate(sample_ids):
            if sample_id not in raw_records:
                raise KeyError(f"MotionFix official raw split has no sample id {sample_id}")
            length = int(lengths_np[local_idx])
            pred_motion_i = pred_raw_np[local_idx, :length]
            if inverse_mode == "motionstreamer272_unified_v2":
                if sample_id not in meta_by_id:
                    raise KeyError(f"MotionFix unified_v2 manifest/meta has no sample id {sample_id}")
                if rotation_corrections is None:
                    raise RuntimeError("MotionFix unified_v2 inverse requires rotation corrections")
                generated_poses[sample_id] = motionstreamer272_unified_v2_to_motionfix_pose(
                    pred_motion_i,
                    meta_by_id[sample_id],
                    rotation_corrections,
                )
            elif inverse_mode in {"motionstreamer272_legacy", "motionstreamer272_hml"}:
                raw_item = raw_records[sample_id]
                generated_poses[sample_id] = motionstreamer272_to_motionfix_pose(
                    pred_motion_i,
                    raw_item["motion_target"],
                    transform_body_pose,
                )
            else:
                raise RuntimeError(f"Unsupported MotionFix official inverse mode: {inverse_mode}")
            ordered_keyids.append(sample_id)
            nb_sample += 1

    if nb_sample == 0:
        raise RuntimeError("MotionFix official edit eval loader produced zero samples")

    gen_samples = _official_generated_features(generated_poses, normalizer, official_model.device, motionfix_repo)
    dataset = OfficialMotionFixDataset(raw_records, ordered_keyids, normalizer, motionfix_repo)
    if len(dataset) != nb_sample:
        missing = sorted(set(ordered_keyids) - set(dataset.keyids))
        raise RuntimeError(f"MotionFix official dataset missing {len(missing)} generated keys, first={missing[:5]}")

    full_result, _full_keys = compute_sim_matrix(
        official_model,
        dataset,
        np.asarray(dataset.keyids),
        gen_samples=gen_samples,
        batch_size=256,
        progress=False,
    )
    full_metrics = {
        mat2name[var]: _metrics_from_sim_matrix(sim_matrix)
        for var, sim_matrix in full_result.items()
    }
    batch_metrics_lists: Dict[str, List[Dict[str, float]]] = {"source_target": [], "target_generated": []}
    keyids_sorted = np.asarray(sorted(dataset.keyids))
    if len(keyids_sorted) >= int(getattr(cfg, "retrieval_batch_size", 32)):
        rng = np.random.RandomState(0)
        idx = np.arange(len(keyids_sorted))
        rng.shuffle(idx)
        bs_m2m = int(getattr(cfg, "retrieval_batch_size", 32))
        for start in range(0, (len(keyids_sorted) // bs_m2m) * bs_m2m, bs_m2m):
            batch_keyids = keyids_sorted[idx[start : start + bs_m2m]]
            batch_result, _batch_keys = compute_sim_matrix(
                official_model,
                dataset,
                batch_keyids,
                gen_samples=gen_samples,
                batch_size=256,
                progress=False,
            )
            for var, sim_matrix in batch_result.items():
                batch_metrics_lists[mat2name[var]].append(_metrics_from_sim_matrix(sim_matrix))
    batch_metrics = {key: _mean_metric_dict(value) for key, value in batch_metrics_lists.items()}
    metrics = _flatten_official_metrics(
        full_metrics,
        batch_metrics,
    )
    metrics["nb_sample"] = int(nb_sample)
    metrics["repeat_id"] = int(repeat_id)
    metrics["motionfix_official_split"] = str(official_wrapper.get("split", ""))
    metrics["motionfix_official_raw_path"] = str(official_wrapper.get("raw_path", ""))
    metrics["motionfix_official_inverse_mode"] = inverse_mode
    metrics["motionfix_official_manifest_path"] = str(official_wrapper.get("manifest_path", ""))
    if mask_batches > 0:
        metrics["mask_generated_cell_frac"] = float(mask_generated_frac_sum / float(mask_batches))
        metrics["mask_preserved_cell_frac"] = float(mask_preserved_frac_sum / float(mask_batches))

    print(
        f"--> \t CodeFlow Global Edit MotionFix Official Eval Repeat {repeat_id}: "
        f"R@1/R@2/R@3 {metrics.get('motionfix_official_target_generated_full_R@1', float('nan')):.2f}/"
        f"{metrics.get('motionfix_official_target_generated_full_R@2', float('nan')):.2f}/"
        f"{metrics.get('motionfix_official_target_generated_full_R@3', float('nan')):.2f}, "
        f"AvgR {metrics.get('motionfix_official_target_generated_full_AvgR', float('nan')):.2f}"
    )
    return metrics
