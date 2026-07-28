#!/usr/bin/env python
"""Shardable HY273 T2M absolute-floor evaluation on HumanML3D test."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval_hy273_kimodo_full_test import verify_checkpoint_assets_coordinated
from models.raw_motion.hy273_kimodo_benchmark import kimodo_motion_quality_metrics
from models.raw_motion.flow_schedule import uses_unified_273_flow
from models.raw_motion.evidence_hash import state_dict_sha256, tensor_sha256
from models.raw_motion.evidence_io import (
    atomic_write_json,
    atomic_write_npz,
    durable_replace,
)
from models.raw_motion.hy273_multitask_condition import (
    CapabilityId,
    make_absent_condition,
)
from models.raw_motion.hy273_slices import (
    CONTACT_SLICE,
    fk_positions_from_global_rot6d,
    reconstruct_global_joints_from_features,
)
from models.raw_motion.hy273_t2m_eval import (
    HY273_TO_MS272_PROTOCOL,
    hy273_to_motionstreamer272,
)
from sample_hy273_multitask import (
    normalizer_from_checkpoint as multitask_normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from sample_hy273_raw import (
    checkpoint_normalizer,
    checkpoint_weight_state,
    sample_ode,
)
from train_hy273_multitask import (
    CHECKPOINT_FORMAT as MULTITASK_CHECKPOINT_FORMAT,
    create_model_from_checkpoint as create_multitask_model_from_checkpoint,
    contact_protocol_for_config,
    validate_assets as validate_multitask_assets,
    validate_frozen_contract,
)
from train_hy273_raw_flow import create_model as create_archived_model


PREFLIGHT_FORMAT = "hy273_t2m_nonregression_preflight_v2"
SUMMARY_FORMAT = "hy273_t2m_nonregression_summary_v2"
ARTIFACT_INDEX_FORMAT = "hy273_t2m_nonregression_artifact_index_v1"
PROTOCOL_VERSION = "hy273_t2m_k273_oracle_canonical_test4042_ode32_internal_v2"
MAX_EVAL_FRAMES = 300
LEGACY_INITIAL_NOISE_PROTOCOL = "per_case_two_stream_cpu_float32_fixed300_v2"
UNIFIED_INITIAL_NOISE_PROTOCOL = "per_case_unified_gaussian_cpu_float32_fixed300_v3"
DEFAULT_MANIFEST = (
    "/mnt/afs/mogo_base/datasets/HY273_multitask_v1/manifests/"
    "hy273_multitask_v1/test.jsonl"
)
DEFAULT_GT272_ROOT = "/mnt/afs/MotionMillion/272-dim-HumanML3D/motion_data"
DEFAULT_TEST_SPLIT = "/mnt/afs/MotionMillion/272-dim-HumanML3D/split/test.txt"
DEFAULT_K273_ROOT = (
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22/motion_data"
)
DEFAULT_TEXT_ROOT = "/mnt/afs/mogo_base/datasets/HumanML3D/texts"
DEFAULT_EVALUATOR = (
    "/mnt/afs/HY-Motion-1.0/ckpts/evaluators/motionstreamer/"
    "Evaluator_272/epoch=99_state_dict.pt"
)
DEFAULT_DISTILBERT = (
    "/mnt/afs/HY-Motion-1.0/ckpts/evaluators/distilbert-base-uncased"
)
DEFAULT_MEAN272 = "/mnt/afs/MotionMillion/272-dim-HumanML3D/mean_std/Mean.npy"
DEFAULT_STD272 = "/mnt/afs/MotionMillion/272-dim-HumanML3D/mean_std/Std.npy"
DEFAULT_HYMOTION_ROOT = "/mnt/afs/HY-Motion-1.0"


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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    atomic_write_npz(path, **arrays)


def _stat(path: str | Path) -> dict[str, int | str]:
    resolved = Path(path).expanduser().resolve()
    value = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
        "inode": int(value.st_ino),
        "device": int(value.st_dev),
    }


def _tree_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    rows = [
        {
            "path": str(item.relative_to(root)),
            "size": int(item.stat().st_size),
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    return {"path": str(root), "files": rows, "sha256": canonical_sha(rows)}


def _code_identity(hymotion_root: str | Path) -> dict[str, Any]:
    external_root = Path(hymotion_root).expanduser().resolve()
    paths = {
        Path(__file__).resolve(),
        ROOT / "models" / "raw_motion" / "hy273_t2m_eval.py",
        ROOT / "eval_hy273_kimodo_full_test.py",
        ROOT / "sample_hy273_raw.py",
        ROOT / "sample_hy273_multitask.py",
        ROOT / "train_hy273_raw_flow.py",
        ROOT / "train_hy273_multitask.py",
        ROOT / "data" / "kimodo273_datasets.py",
        ROOT / "data" / "hy273_multitask_manifest_dataset.py",
        ROOT / "data" / "hy273_multitask_batcher.py",
        ROOT / "data" / "hy273_multitask_scheduler.py",
        ROOT / "models" / "codeflow" / "dit_blocks.py",
        *sorted((ROOT / "models" / "raw_motion").glob("*.py")),
        *sorted((external_root / "hymotion" / "eval").glob("*.py")),
    }
    rows = [
        {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    return {"files": rows, "sha256": canonical_sha(rows)}


def _environment_identity() -> dict[str, Any]:
    import scipy
    import transformers

    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "scipy_version": scipy.__version__,
        "transformers_version": transformers.__version__,
    }


def stable_seed(seed: int, uid: str) -> int:
    digest = hashlib.sha256(f"hy273-t2m:{seed}:{uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def _first_full_caption(row: dict[str, Any]) -> dict[str, Any]:
    for text in row.get("texts", []):
        if text.get("span", {}).get("kind") == "full":
            return text
    raise RuntimeError(f"HumanML3D test row has no full caption: {row.get('uid')}")


def _first_full_caption_file(path: Path, motion_id: str) -> dict[str, Any]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            start, end = float(parts[2]), float(parts[3])
        except ValueError:
            continue
        if (math.isnan(start) or start == 0.0) and (math.isnan(end) or end == 0.0):
            return {
                "value": parts[0],
                "text_id": f"humanml3d:{motion_id}:line{line_number}",
            }
    raise RuntimeError(f"HumanML3D test motion has no full caption: {motion_id}")


def load_plan(
    manifest: str | Path,
    gt272_root: str | Path,
    test_split: str | Path,
    k273_root: str | Path,
    text_root: str | Path,
    *,
    seed: int,
    audit_gt: bool,
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest).expanduser().resolve()
    gt_root = Path(gt272_root).expanduser().resolve()
    split_path = Path(test_split).expanduser().resolve()
    k273_motion_root = Path(k273_root).expanduser().resolve()
    caption_root = Path(text_root).expanduser().resolve()
    manifest_rows: dict[str, dict[str, Any]] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") != "humanml3d_k273":
            continue
        if row.get("split") != "test":
            raise RuntimeError("T2M evaluator received a non-test HumanML3D row")
        motion_id = str(row["target_motion"]["base_motion_id"])
        if motion_id in manifest_rows:
            raise RuntimeError(f"Duplicate HumanML3D motion in unified manifest: {motion_id}")
        manifest_rows[motion_id] = row

    split_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(split_ids) != 4_042 or len(set(split_ids)) != 4_042:
        raise RuntimeError(
            f"Frozen canonical HumanML3D test split requires 4,042 unique IDs, got {len(split_ids)}"
        )
    unexpected = sorted(set(manifest_rows) - set(split_ids))
    if unexpected:
        raise RuntimeError(f"Unified manifest contains non-canonical T2M IDs: {unexpected[:5]}")

    plan = []
    for motion_id in split_ids:
        row = manifest_rows.get(motion_id)
        if row is not None:
            target = row["target_motion"]["k273_asset"]
            k273_path = Path(target["path"]).expanduser().resolve()
            length = int(target["frames"])
            expected_k273_sha = str(target["sha256"])
            text = _first_full_caption(row)
            case_key = str(row["uid"])
            supplemental = False
        else:
            k273_path = (k273_motion_root / f"{motion_id}.npy").resolve()
            k273_value = np.load(k273_path, mmap_mode="r")
            if k273_value.ndim != 2 or k273_value.shape[1] != 273:
                raise RuntimeError(f"Invalid supplemental K273 motion: {k273_path}")
            length = int(k273_value.shape[0])
            expected_k273_sha = sha256_file(k273_path)
            text_path = (caption_root / f"{motion_id}.txt").resolve()
            text = _first_full_caption_file(text_path, motion_id)
            case_key = f"humanml3d:{motion_id}"
            supplemental = True
        if not 4 <= length <= MAX_EVAL_FRAMES:
            raise RuntimeError(f"Invalid T2M requested length {length} for {motion_id}")
        gt_path = (gt_root / f"{motion_id}.npy").resolve()
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing MotionStreamer272 GT: {gt_path}")
        gt_sha = None
        k273_sha = expected_k273_sha
        if audit_gt:
            gt = np.load(gt_path, mmap_mode="r")
            if gt.shape != (length, 272) or not np.isfinite(gt).all():
                raise RuntimeError(
                    f"Invalid MotionStreamer272 GT {gt_path}: {gt.shape}, expected {(length, 272)}"
                )
            gt_sha = sha256_file(gt_path)
            k273 = np.load(k273_path, mmap_mode="r")
            if k273.shape != (length, 273) or not np.isfinite(k273).all():
                raise RuntimeError(
                    f"Invalid K273 GT {k273_path}: {k273.shape}, expected {(length, 273)}"
                )
            k273_sha = sha256_file(k273_path)
            if k273_sha != expected_k273_sha:
                raise RuntimeError(f"K273 manifest SHA mismatch: {k273_path}")
        plan.append(
            {
                "case_key": case_key,
                "motion_id": motion_id,
                "caption": str(text["value"]),
                "text_id": str(text["text_id"]),
                "length": length,
                "sample_seed": stable_seed(seed, case_key),
                "k273_path": str(k273_path),
                "k273_sha256": k273_sha,
                "gt272_path": str(gt_path),
                "gt272_sha256": gt_sha,
                "supplemental_short_case": supplemental,
            }
        )
    if len(plan) != 4_042 or len({row["case_key"] for row in plan}) != 4_042:
        raise RuntimeError(f"Frozen T2M protocol requires 4,042 unique cases, got {len(plan)}")
    return plan


def checkpoint_kind(checkpoint: dict[str, Any]) -> str:
    if checkpoint.get("format") == MULTITASK_CHECKPOINT_FORMAT:
        return "multitask"
    train_args = checkpoint.get("args")
    if (
        checkpoint.get("format") is None
        and isinstance(train_args, dict)
        and str(train_args.get("architecture", "")) == "redenoise_kimodo_like"
    ):
        return "archived_kimodo_like"
    raise RuntimeError("Unsupported HY273 checkpoint format")


def _sampling_identity(
    kind: str, args: argparse.Namespace, *, unified_273_flow: bool
) -> dict[str, Any]:
    unified = bool(unified_273_flow)
    return {
        "ode_steps": int(args.num_steps),
        "text_cfg_scale": float(args.cfg_scale),
        "contact_init": "unified_273d_state" if unified else "random",
        "contact_feedback": "ode_273d" if unified else "blend",
        "cfg_apply_contacts": bool(unified),
        "frame_gauge": [1.0, 0.0],
        "weight_source": args.weight_source,
        "generation_batch_size": int(args.batch_size),
        "fixed_frame_extent": MAX_EVAL_FRAMES,
        "initial_noise": (
            UNIFIED_INITIAL_NOISE_PROTOCOL
            if unified
            else LEGACY_INITIAL_NOISE_PROTOCOL
        ),
    }


def _checkpoint_step(checkpoint: dict[str, Any]) -> int:
    if "next_global_step" in checkpoint:
        return int(checkpoint["next_global_step"])
    return int(checkpoint.get("step", -1))


def _persistent_attestation_path(args: argparse.Namespace) -> Path:
    return (
        Path(args.output_dir).expanduser().resolve()
        / "archived_asset_verification_attestation.json"
    )


def _evaluator_forward_smoke(
    args: argparse.Namespace, plan: list[dict[str, Any]]
) -> dict[str, Any]:
    if args.hymotion_root not in sys.path:
        sys.path.insert(0, args.hymotion_root)
    from hymotion.eval.metrics import (
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
    )
    from hymotion.eval.motionstreamer272 import MotionStreamer272Evaluator

    evaluator = MotionStreamer272Evaluator.from_checkpoint(
        checkpoint=args.evaluator_checkpoint,
        distilbert_path=args.distilbert_path,
        mean_path=args.mean272,
        std_path=args.std272,
        device=args.device,
    )
    selected = plan[:2]
    motions = np.zeros((2, MAX_EVAL_FRAMES, 272), dtype=np.float32)
    lengths = np.asarray([int(row["length"]) for row in selected], dtype=np.int64)
    for index, row in enumerate(selected):
        k273 = np.load(row["k273_path"]).astype(np.float32)
        motions[index, : len(k273)] = hy273_to_motionstreamer272(k273)
    text_embeddings = evaluator.encode_text([row["caption"] for row in selected], batch_size=2)
    motion_embeddings = evaluator.encode_motion(motions, lengths, batch_size=2)
    mu, covariance = calculate_activation_statistics(motion_embeddings)
    identity_fid = calculate_frechet_distance(mu, covariance, mu, covariance)
    diversity = calculate_diversity(
        motion_embeddings, 1, rng=np.random.default_rng(int(args.seed))
    )
    top, matching = _retrieval_rows(text_embeddings, motion_embeddings, batch_size=2)
    values = np.asarray([identity_fid, diversity, matching.mean(), top.mean()])
    if not np.isfinite(values).all() or abs(identity_fid) > 1e-3:
        raise RuntimeError(f"Evaluator forward/FID smoke failed: {values.tolist()}")
    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "passed",
        "case_count": 2,
        "identity_fid": float(identity_fid),
        "diversity": float(diversity),
        "mean_matching_distance": float(matching.mean()),
        "r_precision_mean": float(top.mean()),
    }


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    stat_before = _stat(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    stat_after = _stat(checkpoint_path)
    if stat_before != stat_after:
        raise RuntimeError("Checkpoint changed while hashing")
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256.lower():
        raise RuntimeError("Checkpoint SHA256 mismatch")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    kind = checkpoint_kind(checkpoint)
    unified_273_flow = False
    if args.weight_source not in checkpoint:
        raise RuntimeError(f"Checkpoint has no requested weight source: {args.weight_source}")
    selected_weight_state_sha256 = state_dict_sha256(checkpoint[args.weight_source])
    if kind == "archived_kimodo_like":
        train_args = argparse.Namespace(**checkpoint["args"])
        if str(getattr(train_args, "architecture", "")) != "redenoise_kimodo_like":
            raise RuntimeError("Archived T2M baseline must use the Kimodo-like backbone")
        attestation_path, reused = verify_checkpoint_assets_coordinated(
            train_args,
            requested_cache_path=str(_persistent_attestation_path(args)),
        )
        asset_identity = {
            "kind": "archived_asset_attestation",
            "path": str(attestation_path),
            "sha256": sha256_file(attestation_path),
            "reused": bool(reused),
        }
    else:
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise RuntimeError("Multitask checkpoint has no resolved config")
        validate_frozen_contract(config)
        unified_273_flow = uses_unified_273_flow(
            contact_protocol_for_config(config)
        )
        asset_identity = validate_multitask_assets(config)
        if checkpoint.get("asset_identity") != asset_identity:
            raise RuntimeError("Multitask checkpoint asset identity mismatch")

    plan = load_plan(
        args.manifest,
        args.gt272_root,
        args.test_split,
        args.k273_root,
        args.text_root,
        seed=args.seed,
        audit_gt=True,
    )
    evaluator_assets = {
        "checkpoint": {**_stat(args.evaluator_checkpoint), "sha256": sha256_file(args.evaluator_checkpoint)},
        "distilbert": _tree_identity(args.distilbert_path),
        "mean272": {**_stat(args.mean272), "sha256": sha256_file(args.mean272)},
        "std272": {**_stat(args.std272), "sha256": sha256_file(args.std272)},
    }
    payload = {
        "format": PREFLIGHT_FORMAT,
        "status": "passed",
        "host": socket.gethostname(),
        "checkpoint": {
            **stat_after,
            "sha256": checkpoint_sha,
            "kind": kind,
            "unified_273_flow": unified_273_flow,
            "step": _checkpoint_step(checkpoint),
            "has_ema": "ema" in checkpoint,
            "selected_weight_source": args.weight_source,
            "selected_weight_state_sha256": selected_weight_state_sha256,
        },
        "manifest": {
            **_stat(args.manifest),
            "sha256": sha256_file(args.manifest),
        },
        "canonical_test_split": {
            **_stat(args.test_split),
            "sha256": sha256_file(args.test_split),
            "case_count": 4_042,
            "short_supplemental_ids": [
                row["motion_id"] for row in plan if row["supplemental_short_case"]
            ],
        },
        "asset_identity": asset_identity,
        "evaluator_assets": evaluator_assets,
        "code": _code_identity(args.hymotion_root),
        "environment": _environment_identity(),
        "plan": {
            "protocol_version": PROTOCOL_VERSION,
            "case_count": len(plan),
            "case_plan_sha256": canonical_sha(plan),
            "num_shards": int(args.num_shards),
            "seed": int(args.seed),
            "rows": plan,
        },
        "sampling": _sampling_identity(
            kind, args, unified_273_flow=unified_273_flow
        ),
        "aggregation": {
            "motion_embedding_batch_size": 32,
            "retrieval_batch_size": 32,
            "diversity_pairs": 300,
            "diversity_rng": "independent_sha256_derived_numpy_stream_v1",
        },
        "bridge_protocol": HY273_TO_MS272_PROTOCOL,
        "reference_domain": "k273_gt_oracle_through_same_bridge",
        "official_benchmark_claim": False,
    }
    payload["evaluator_forward_smoke"] = _evaluator_forward_smoke(args, plan)
    del checkpoint
    return payload


def _evaluator_assets(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint": {
            **_stat(args.evaluator_checkpoint),
            "sha256": sha256_file(args.evaluator_checkpoint),
        },
        "distilbert": _tree_identity(args.distilbert_path),
        "mean272": {**_stat(args.mean272), "sha256": sha256_file(args.mean272)},
        "std272": {**_stat(args.std272), "sha256": sha256_file(args.std272)},
    }


def load_preflight(
    args: argparse.Namespace, *, verify_evaluator_assets: bool = False
) -> tuple[Path, dict[str, Any], str]:
    if not args.checkpoint:
        raise RuntimeError("A pinned --checkpoint is required with a T2M preflight")
    path = Path(args.preflight_manifest).expanduser().resolve()
    if not path.is_file() or not args.preflight_sha256:
        raise RuntimeError("Shard/aggregate requires a pinned preflight manifest and SHA")
    sha = sha256_file(path)
    if sha != args.preflight_sha256.lower():
        raise RuntimeError("T2M preflight SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PREFLIGHT_FORMAT or payload.get("status") != "passed":
        raise RuntimeError("Invalid T2M preflight")
    if _stat(args.checkpoint) != {
        key: payload["checkpoint"][key]
        for key in ("path", "size", "mtime_ns", "ctime_ns", "inode", "device")
    }:
        raise RuntimeError("Checkpoint changed after T2M preflight")
    if sha256_file(args.checkpoint) != payload["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint content changed after T2M preflight")
    if payload["code"] != _code_identity(args.hymotion_root):
        raise RuntimeError("T2M evaluation code changed after preflight")
    if payload.get("environment") != _environment_identity():
        raise RuntimeError("T2M Python/package environment changed after preflight")
    if _stat(args.manifest) != {
        key: payload["manifest"][key]
        for key in ("path", "size", "mtime_ns", "ctime_ns", "inode", "device")
    } or sha256_file(args.manifest) != payload["manifest"]["sha256"]:
        raise RuntimeError("T2M manifest changed after preflight")
    split_identity = payload.get("canonical_test_split", {})
    if _stat(args.test_split) != {
        key: split_identity.get(key)
        for key in ("path", "size", "mtime_ns", "ctime_ns", "inode", "device")
    } or sha256_file(args.test_split) != split_identity.get("sha256"):
        raise RuntimeError("Canonical T2M test split changed after preflight")
    asset_identity = payload.get("asset_identity", {})
    if asset_identity.get("kind") == "archived_asset_attestation":
        attestation = Path(str(asset_identity.get("path", "")))
        if not attestation.is_file() or sha256_file(attestation) != asset_identity.get(
            "sha256"
        ):
            raise RuntimeError("Archived asset attestation is missing or changed")
    if verify_evaluator_assets and _evaluator_assets(args) != payload["evaluator_assets"]:
        raise RuntimeError("T2M evaluator assets changed after preflight")
    return path, payload, sha


def _load_runtime(
    args: argparse.Namespace, preflight: dict[str, Any], device: torch.device
) -> tuple[str, torch.nn.Module, Any, dict[str, Any]]:
    checkpoint = torch.load(
        Path(args.checkpoint).expanduser().resolve(),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    kind = checkpoint_kind(checkpoint)
    if kind != preflight["checkpoint"]["kind"]:
        raise RuntimeError("Checkpoint kind changed after preflight")
    if kind == "archived_kimodo_like":
        train_args = argparse.Namespace(**checkpoint["args"])
        attestation_path, _ = verify_checkpoint_assets_coordinated(
            train_args,
            requested_cache_path=str(preflight["asset_identity"]["path"]),
        )
        if sha256_file(attestation_path) != preflight["asset_identity"]["sha256"]:
            raise RuntimeError("Archived asset attestation changed after T2M preflight")
        state, weight_source = checkpoint_weight_state(
            checkpoint, args.weight_source, args.checkpoint
        )
        model = create_archived_model(train_args).to(device)
        model.load_state_dict(state, strict=True)
        normalizer = checkpoint_normalizer(
            checkpoint, train_args, device, args.checkpoint
        )
        runtime = {
            "weight_source": weight_source,
            "prediction_type": str(train_args.prediction_type),
            "self_conditioning": bool(train_args.self_conditioning),
            "unified_273_flow": False,
        }
    else:
        config = checkpoint["config"]
        current_asset_identity = validate_multitask_assets(config)
        if (
            current_asset_identity != checkpoint.get("asset_identity")
            or current_asset_identity != preflight.get("asset_identity")
        ):
            raise RuntimeError("Multitask asset identity changed after T2M preflight")
        model = create_multitask_model_from_checkpoint(checkpoint).to(device)
        state = checkpoint[args.weight_source]
        model.load_state_dict(state, strict=True)
        normalizer = multitask_normalizer_from_checkpoint(checkpoint, device)
        runtime = {
            "weight_source": args.weight_source,
            "prediction_type": "x0",
            "unified_273_flow": bool(normalizer.normalize_contacts),
        }
    if bool(runtime["unified_273_flow"]) != bool(
        preflight["checkpoint"]["unified_273_flow"]
    ):
        raise RuntimeError("Checkpoint contact protocol changed after T2M preflight")
    selected_weight_state_sha256 = state_dict_sha256(state)
    if (
        args.weight_source != preflight["checkpoint"]["selected_weight_source"]
        or selected_weight_state_sha256
        != preflight["checkpoint"]["selected_weight_state_sha256"]
    ):
        raise RuntimeError("Selected model tensor state changed after T2M preflight")
    runtime["selected_weight_state_sha256"] = selected_weight_state_sha256
    model.eval()
    del checkpoint, state
    return kind, model, normalizer, runtime


def _noise_stream_seed(sample_seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"hy273-t2m-noise-v2:{sample_seed}:{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _initial_noise(
    cases: list[dict[str, Any]],
    frames: int = MAX_EVAL_FRAMES,
    *,
    unified: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, str]]]:
    if frames != MAX_EVAL_FRAMES:
        raise ValueError(f"Frozen T2M noise extent must be {MAX_EVAL_FRAMES}")
    continuous, contacts, evidence = [], [], []
    for case in cases:
        sample_seed = int(case["sample_seed"])
        if unified:
            unified_generator = torch.Generator(device="cpu").manual_seed(
                _noise_stream_seed(sample_seed, "unified_273d")
            )
            unified_value = torch.randn(frames, 273, generator=unified_generator)
            continuous_value = unified_value[..., :269]
            contact_value = unified_value[..., 269:273]
            continuous.append(continuous_value)
            contacts.append(contact_value)
            continuous_sha = tensor_sha256(continuous_value)
            contact_sha = tensor_sha256(contact_value)
            unified_sha = tensor_sha256(unified_value)
            evidence.append(
                {
                    "initial_noise_protocol": UNIFIED_INITIAL_NOISE_PROTOCOL,
                    "initial_continuous_noise_sha256": continuous_sha,
                    "initial_contact_noise_sha256": contact_sha,
                    "initial_unified_noise_sha256": unified_sha,
                    "initial_noise_sha256": unified_sha,
                }
            )
            continue
        continuous_generator = torch.Generator(device="cpu").manual_seed(
            _noise_stream_seed(sample_seed, "continuous")
        )
        contact_generator = torch.Generator(device="cpu").manual_seed(
            _noise_stream_seed(sample_seed, "contacts")
        )
        continuous_value = torch.randn(frames, 269, generator=continuous_generator)
        contact_value = torch.rand(frames, 4, generator=contact_generator)
        continuous.append(continuous_value)
        contacts.append(contact_value)
        continuous_sha = tensor_sha256(continuous_value)
        contact_sha = tensor_sha256(contact_value)
        evidence.append(
            {
                "initial_noise_protocol": LEGACY_INITIAL_NOISE_PROTOCOL,
                "initial_continuous_noise_sha256": continuous_sha,
                "initial_contact_noise_sha256": contact_sha,
                "initial_noise_sha256": canonical_sha(
                    {"continuous": continuous_sha, "contacts": contact_sha}
                ),
            }
        )
    return torch.stack(continuous), torch.stack(contacts), evidence


def _jerk_mps3(joints: torch.Tensor, fps: float = 30.0) -> float:
    if len(joints) < 4:
        return 0.0
    jerk = torch.diff(joints, n=3, dim=0) * float(fps) ** 3
    return float(jerk.norm(dim=-1).mean().item())


def _quality(motion: torch.Tensor) -> dict[str, float]:
    motion = motion.float()
    fk_joints = fk_positions_from_global_rot6d(motion)
    represented_joints = reconstruct_global_joints_from_features(motion)
    output = kimodo_motion_quality_metrics(
        fk_joints,
        motion[..., CONTACT_SLICE] > 0.5,
    )
    output["fk_jerk_mps3"] = _jerk_mps3(fk_joints)
    output["position_channel_jerk_mps3"] = _jerk_mps3(represented_joints)
    return {name: float(value) for name, value in output.items()}


def _sample_batch(
    *,
    kind: str,
    model: torch.nn.Module,
    normalizer: Any,
    runtime: dict[str, Any],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[dict[str, str]]]:
    frames = MAX_EVAL_FRAMES
    lengths = torch.tensor([int(case["length"]) for case in cases], dtype=torch.long)
    texts = [str(case["caption"]) for case in cases]
    observed = torch.zeros(len(cases), frames, 273)
    hard_mask = torch.zeros_like(observed, dtype=torch.bool)
    c_dir = torch.zeros(len(cases), 2)
    c_dir[:, 0] = 1.0
    unified = bool(runtime["unified_273_flow"])
    continuous, contacts, noise_evidence = _initial_noise(
        cases, frames, unified=unified
    )
    if kind == "archived_kimodo_like":
        output = sample_ode(
            model,
            normalizer,
            lengths,
            texts,
            observed,
            hard_mask,
            c_dir,
            num_steps=args.num_steps,
            self_conditioning=bool(runtime["self_conditioning"]),
            cfg_scale=args.cfg_scale,
            control_cfg_scale=1.0,
            contact_init="random",
            contact_feedback="blend",
            cfg_apply_contacts=False,
            prediction_type=str(runtime["prediction_type"]),
            return_details=True,
            initial_continuous_noise=continuous,
            initial_contact_noise=contacts,
        )
        return output.raw_motion.cpu(), noise_evidence
    condition = make_absent_condition(
        batch_size=len(cases),
        target_frames=frames,
        target_lengths=lengths,
        device=torch.device("cpu"),
        capability=CapabilityId.T2M,
    )
    noise_kwargs = (
        {"initial_unified_noise": torch.cat([continuous, contacts], dim=-1)}
        if unified
        else {
            "initial_continuous_noise": continuous,
            "initial_contact_noise": contacts,
        }
    )
    output = sample_hy273_multitask_ode(
        model,
        normalizer,
        condition,
        texts,
        observed,
        hard_mask,
        num_steps=args.num_steps,
        text_cfg_scale=args.cfg_scale,
        **noise_kwargs,
    )
    return output.raw_motion.cpu(), noise_evidence


def _frozen_chunk_path(
    output: Path,
    *,
    shard_id: int,
    frozen_batch_index: int,
    frozen_batch: list[dict[str, Any]],
) -> Path:
    identity = canonical_sha([case["case_key"] for case in frozen_batch])[:16]
    return (
        output
        / "chunks"
        / f"shard_{shard_id:02d}_batch_{frozen_batch_index:05d}_{identity}.npz"
    )


def _expected_chunk_assignments(
    output: Path,
    *,
    shard_id: int,
    shard_cases: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, tuple[str, int]]:
    assignments: dict[str, tuple[str, int]] = {}
    for start in range(0, len(shard_cases), batch_size):
        frozen_batch = shard_cases[start : start + batch_size]
        path = _frozen_chunk_path(
            output,
            shard_id=shard_id,
            frozen_batch_index=start // batch_size,
            frozen_batch=frozen_batch,
        )
        for row_index, case in enumerate(frozen_batch):
            assignments[case["case_key"]] = (str(path), row_index)
    return assignments


def _recover_partial_batches(
    *,
    record_path: Path,
    shard_cases: list[dict[str, Any]],
    batch_size: int,
    existing: list[dict[str, Any]],
    by_key: dict[str, dict[str, Any]],
    chunk_cache: "ChunkCache",
    shard_id: int,
    protocol_sha: str,
    preflight_sha: str,
    selected_weight_sha: str,
    unified_noise: bool = False,
) -> None:
    with record_path.open("a", encoding="utf-8") as recovery_writer:
        for start in range(0, len(shard_cases), batch_size):
            frozen_batch = shard_cases[start : start + batch_size]
            present = [case["case_key"] in by_key for case in frozen_batch]
            if not any(present) or all(present):
                continue
            exemplars = [
                by_key[case["case_key"]]
                for case in frozen_batch
                if case["case_key"] in by_key
            ]
            chunk_paths = {str(row["chunk_path"]) for row in exemplars}
            chunk_shas = {str(row["chunk_sha256"]) for row in exemplars}
            if len(chunk_paths) != 1 or len(chunk_shas) != 1:
                raise RuntimeError("Partial T2M batch references inconsistent chunks")
            payload = chunk_cache.payload(exemplars[0])
            payload_keys = [str(value) for value in payload["case_keys"]]
            expected_keys = [case["case_key"] for case in frozen_batch]
            if payload_keys != expected_keys:
                raise RuntimeError(
                    "Partial T2M batch chunk does not contain its frozen cases"
                )
            chunk_path = next(iter(chunk_paths))
            chunk_sha = next(iter(chunk_shas))
            for row_index, case in enumerate(frozen_batch):
                if case["case_key"] in by_key:
                    continue
                length = int(case["length"])
                raw = payload["generated_k273"][row_index, :length]
                noise_row = _initial_noise(
                    [case], MAX_EVAL_FRAMES, unified=unified_noise
                )[2][0]
                recovered = {
                    "status": "ok",
                    **case,
                    "quality": _quality(torch.from_numpy(raw)),
                    **noise_row,
                    "chunk_path": chunk_path,
                    "chunk_sha256": chunk_sha,
                    "chunk_row": row_index,
                    "chunk_frames": MAX_EVAL_FRAMES,
                    "shard_id": shard_id,
                    "protocol_manifest_sha256": protocol_sha,
                    "preflight_sha256": preflight_sha,
                    "selected_weight_state_sha256": selected_weight_sha,
                    "recovered_from_atomic_chunk": True,
                }
                _validate_record_provenance(
                    recovered,
                    case,
                    shard_id=shard_id,
                    protocol_sha=protocol_sha,
                    preflight_sha=preflight_sha,
                    selected_weight_sha=selected_weight_sha,
                    chunk_cache=chunk_cache,
                    verify_quality=True,
                )
                recovery_writer.write(
                    json.dumps(recovered, ensure_ascii=False, sort_keys=True) + "\n"
                )
                recovery_writer.flush()
                os.fsync(recovery_writer.fileno())
                existing.append(recovered)
                by_key[case["case_key"]] = recovered


def run_shard(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0,num_shards)")
    preflight_path, preflight, preflight_sha = load_preflight(args)
    if int(preflight["plan"]["num_shards"]) != args.num_shards:
        raise RuntimeError("T2M shard count differs from preflight")
    if preflight["sampling"] != _sampling_identity(
        str(preflight["checkpoint"]["kind"]),
        args,
        unified_273_flow=bool(preflight["checkpoint"]["unified_273_flow"]),
    ):
        raise RuntimeError("T2M sampling protocol differs from preflight")
    plan = list(preflight["plan"]["rows"])
    shard_cases = [
        case for index, case in enumerate(plan) if index % args.num_shards == args.shard_id
    ]
    output = Path(args.output_dir).expanduser().resolve()
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "checkpoint_kind": preflight["checkpoint"]["kind"],
        "weight_source": args.weight_source,
        "case_count": len(plan),
        "case_plan_sha256": preflight["plan"]["case_plan_sha256"],
        "num_shards": args.num_shards,
        "sampling": preflight["sampling"],
        "bridge_protocol": HY273_TO_MS272_PROTOCOL,
        "reference_domain": preflight["reference_domain"],
        "official_benchmark_claim": False,
        "selected_weight_state_sha256": preflight["checkpoint"][
            "selected_weight_state_sha256"
        ],
        "aggregation": preflight["aggregation"],
        "preflight_path": str(preflight_path),
        "preflight_sha256": preflight_sha,
    }
    protocol_path = output / "protocol_manifest.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing T2M protocol differs from requested run")
    else:
        _atomic_json(protocol_path, protocol)
    protocol_sha = sha256_file(protocol_path)
    record_path = output / "shards" / f"shard_{args.shard_id:02d}.jsonl"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_records(record_path)
    by_key = {row["case_key"]: row for row in existing}
    if len(by_key) != len(existing):
        raise RuntimeError("Duplicate T2M shard records")
    expected_by_key = {case["case_key"]: case for case in shard_cases}
    expected_chunk_by_key = _expected_chunk_assignments(
        output,
        shard_id=args.shard_id,
        shard_cases=shard_cases,
        batch_size=args.batch_size,
    )
    existing_cache = ChunkCache()
    for row in existing:
        case = expected_by_key.get(row["case_key"])
        if case is None:
            raise RuntimeError(f"Existing T2M record is outside shard plan: {row['case_key']}")
        expected_chunk_path, expected_chunk_row = expected_chunk_by_key[case["case_key"]]
        if (
            row.get("chunk_path") != expected_chunk_path
            or int(row.get("chunk_row", -1)) != expected_chunk_row
        ):
            raise RuntimeError(f"Existing T2M record has wrong chunk assignment: {row['case_key']}")
        _validate_record_provenance(
            row,
            case,
            shard_id=args.shard_id,
            protocol_sha=protocol_sha,
            preflight_sha=preflight_sha,
            selected_weight_sha=protocol["selected_weight_state_sha256"],
            chunk_cache=existing_cache,
            verify_quality=True,
        )
    _recover_partial_batches(
        record_path=record_path,
        shard_cases=shard_cases,
        batch_size=args.batch_size,
        existing=existing,
        by_key=by_key,
        chunk_cache=existing_cache,
        shard_id=args.shard_id,
        protocol_sha=protocol_sha,
        preflight_sha=preflight_sha,
        selected_weight_sha=protocol["selected_weight_state_sha256"],
        unified_noise=bool(preflight["checkpoint"]["unified_273_flow"]),
    )
    for start in range(0, len(shard_cases), args.batch_size):
        frozen_batch = shard_cases[start : start + args.batch_size]
        present = [case["case_key"] in by_key for case in frozen_batch]
        if any(present) and not all(present):
            raise RuntimeError("Partial T2M frozen batch was not fully recovered")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    kind, model, normalizer, runtime = _load_runtime(args, preflight, device)
    started = time.perf_counter()
    generated_count = 0
    with record_path.open("a", encoding="utf-8") as writer:
        for start in range(0, len(shard_cases), args.batch_size):
            frozen_batch = shard_cases[start : start + args.batch_size]
            if all(case["case_key"] in by_key for case in frozen_batch):
                continue
            if any(case["case_key"] in by_key for case in frozen_batch):
                raise RuntimeError("Refusing to sample a partial frozen T2M batch")
            batch = frozen_batch
            raw, noise_evidence = _sample_batch(
                kind=kind,
                model=model,
                normalizer=normalizer,
                runtime=runtime,
                cases=batch,
                args=args,
            )
            frames = int(raw.shape[1])
            if raw.shape != (len(batch), MAX_EVAL_FRAMES, 273):
                raise RuntimeError(
                    f"Unexpected T2M sampler shape {tuple(raw.shape)}; expected "
                    f"{(len(batch), MAX_EVAL_FRAMES, 273)}"
                )
            generated272 = np.zeros((len(batch), frames, 272), dtype=np.float32)
            generated_k273 = raw.numpy().astype(np.float32, copy=False)
            qualities = []
            for index, case in enumerate(batch):
                length = int(case["length"])
                generated272[index, :length] = hy273_to_motionstreamer272(
                    raw[index, :length].numpy()
                )
                qualities.append(_quality(raw[index, :length]))
            chunk_path = _frozen_chunk_path(
                output,
                shard_id=args.shard_id,
                frozen_batch_index=start // args.batch_size,
                frozen_batch=frozen_batch,
            )
            _atomic_npz(
                chunk_path,
                case_keys=np.asarray([case["case_key"] for case in batch]),
                lengths=np.asarray([case["length"] for case in batch], dtype=np.int64),
                generated272=generated272,
                generated_k273=generated_k273,
            )
            chunk_sha = sha256_file(chunk_path)
            for index, (case, quality, noise_row) in enumerate(
                zip(batch, qualities, noise_evidence)
            ):
                record = {
                    "status": "ok",
                    **case,
                    "gt272_sha256": case["gt272_sha256"],
                    "quality": quality,
                    **noise_row,
                    "chunk_path": str(chunk_path),
                    "chunk_sha256": chunk_sha,
                    "chunk_row": index,
                    "chunk_frames": frames,
                    "shard_id": args.shard_id,
                    "protocol_manifest_sha256": protocol_sha,
                    "preflight_sha256": preflight_sha,
                    "selected_weight_state_sha256": protocol[
                        "selected_weight_state_sha256"
                    ],
                }
                writer.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                writer.flush()
                os.fsync(writer.fileno())
                by_key[case["case_key"]] = record
                generated_count += 1
            print(
                json.dumps(
                    {
                        "shard_id": args.shard_id,
                        "completed": len(by_key),
                        "total": len(shard_cases),
                        "samples_per_second": generated_count
                        / max(time.perf_counter() - started, 1e-9),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    _atomic_json(
        output / "shards" / f"shard_{args.shard_id:02d}_summary.json",
        {
            "shard_id": args.shard_id,
            "records": len(by_key),
            "expected": len(shard_cases),
            "complete": len(by_key) == len(shard_cases),
            "protocol_manifest_sha256": protocol_sha,
        },
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class ChunkCache:
    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity
        self.values: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.verified: dict[str, str] = {}

    def payload(self, record: dict[str, Any]) -> dict[str, np.ndarray]:
        path = str(record["chunk_path"])
        claimed_sha = str(record["chunk_sha256"])
        bound_sha = self.verified.get(path)
        if bound_sha is not None and bound_sha != claimed_sha:
            raise RuntimeError(f"Conflicting T2M generated chunk SHA claims: {path}")
        if path not in self.values:
            if sha256_file(path) != claimed_sha:
                raise RuntimeError(f"T2M generated chunk SHA mismatch: {path}")
            self.verified[path] = claimed_sha
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: archive[name].copy() for name in archive.files}
            if set(payload) != {
                "case_keys",
                "lengths",
                "generated272",
                "generated_k273",
            }:
                raise RuntimeError(f"Unexpected T2M chunk schema: {path}")
            count = len(payload["case_keys"])
            if (
                count == 0
                or len({str(value) for value in payload["case_keys"]}) != count
                or payload["lengths"].shape != (count,)
                or payload["generated272"].shape != (count, MAX_EVAL_FRAMES, 272)
                or payload["generated_k273"].shape != (count, MAX_EVAL_FRAMES, 273)
                or np.any(payload["lengths"] < 4)
                or np.any(payload["lengths"] > MAX_EVAL_FRAMES)
                or not np.isfinite(payload["generated272"]).all()
                or not np.isfinite(payload["generated_k273"]).all()
            ):
                raise RuntimeError(f"Invalid T2M chunk payload: {path}")
            for index, length in enumerate(payload["lengths"].tolist()):
                if np.any(payload["generated272"][index, int(length) :] != 0.0):
                    raise RuntimeError(f"Nonzero padded MotionStreamer272 values: {path}")
            self.values[path] = payload
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
        payload = self.values.pop(path)
        self.values[path] = payload
        return payload

    def row(self, record: dict[str, Any], field: str = "generated272") -> np.ndarray:
        payload = self.payload(record)
        row = int(record["chunk_row"])
        if not 0 <= row < len(payload["case_keys"]):
            raise RuntimeError("T2M chunk row is out of bounds")
        if str(payload["case_keys"][row]) != record["case_key"]:
            raise RuntimeError("T2M chunk row/case mismatch")
        if int(payload["lengths"][row]) != int(record["length"]):
            raise RuntimeError("T2M chunk row/length mismatch")
        return payload[field][row, : int(record["length"])]


def _validate_record_provenance(
    record: dict[str, Any],
    case: dict[str, Any],
    *,
    shard_id: int,
    protocol_sha: str,
    preflight_sha: str,
    selected_weight_sha: str,
    chunk_cache: ChunkCache,
    verify_quality: bool,
) -> None:
    plan_fields = (
        "case_key",
        "motion_id",
        "caption",
        "text_id",
        "length",
        "sample_seed",
        "k273_path",
        "k273_sha256",
        "gt272_path",
        "gt272_sha256",
        "supplemental_short_case",
    )
    if record.get("status") != "ok" or any(
        record.get(name) != case.get(name) for name in plan_fields
    ):
        raise RuntimeError(f"T2M record differs from frozen plan: {case['case_key']}")
    if (
        int(record.get("shard_id", -1)) != int(shard_id)
        or record.get("protocol_manifest_sha256") != protocol_sha
        or record.get("preflight_sha256") != preflight_sha
        or record.get("selected_weight_state_sha256") != selected_weight_sha
        or int(record.get("chunk_frames", -1)) != MAX_EVAL_FRAMES
    ):
        raise RuntimeError(f"T2M record provenance mismatch: {case['case_key']}")
    for name in (
        "initial_continuous_noise_sha256",
        "initial_contact_noise_sha256",
        "initial_noise_sha256",
        "chunk_sha256",
    ):
        value = str(record.get(name, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(f"Invalid {name}: {case['case_key']}")
    expected_noise = _initial_noise(
        [case],
        MAX_EVAL_FRAMES,
        unified=(
            record.get("initial_noise_protocol") == UNIFIED_INITIAL_NOISE_PROTOCOL
        ),
    )[2][0]
    if any(record.get(name) != value for name, value in expected_noise.items()):
        raise RuntimeError(f"T2M initial noise changed: {case['case_key']}")
    raw = chunk_cache.row(record, "generated_k273")
    generated272 = chunk_cache.row(record, "generated272")
    rebuilt272 = hy273_to_motionstreamer272(raw)
    if not np.array_equal(generated272, rebuilt272):
        raise RuntimeError(f"T2M K273/272 chunk bridge mismatch: {case['case_key']}")
    if verify_quality:
        recomputed = _quality(torch.from_numpy(raw))
        quality = record.get("quality")
        if not isinstance(quality, dict) or set(quality) != set(recomputed):
            raise RuntimeError(f"T2M quality schema mismatch: {case['case_key']}")
        for name, value in recomputed.items():
            if not math.isclose(float(quality[name]), value, rel_tol=0.0, abs_tol=1e-7):
                raise RuntimeError(f"T2M quality changed: {case['case_key']}/{name}")


def _retrieval_rows(
    text_embeddings: np.ndarray, motion_embeddings: np.ndarray, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    top = np.zeros((len(text_embeddings), 3), dtype=np.float64)
    matching = np.linalg.norm(text_embeddings - motion_embeddings, axis=1)
    for start in range(0, len(text_embeddings), batch_size):
        text = text_embeddings[start : start + batch_size]
        motion = motion_embeddings[start : start + batch_size]
        distance = np.linalg.norm(text[:, None] - motion[None], axis=-1)
        rank_matrix = np.argsort(np.argsort(distance, axis=1), axis=1)
        diagonal = np.arange(len(text))
        rank = rank_matrix[diagonal, diagonal]
        for k in range(3):
            top[start : start + len(text), k] = rank <= k
    return top, matching


def _metric_rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"hy273-t2m-metric-v1:{seed}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def aggregate(args: argparse.Namespace) -> None:
    _, preflight, preflight_sha = load_preflight(
        args, verify_evaluator_assets=True
    )
    output = Path(args.output_dir).expanduser().resolve()
    protocol_path = output / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("preflight_sha256") != preflight_sha
        or protocol.get("checkpoint_sha256") != preflight["checkpoint"]["sha256"]
        or protocol.get("case_plan_sha256") != preflight["plan"]["case_plan_sha256"]
        or protocol.get("sampling") != preflight["sampling"]
        or protocol.get("bridge_protocol") != preflight["bridge_protocol"]
        or protocol.get("reference_domain") != preflight["reference_domain"]
        or protocol.get("official_benchmark_claim") is not False
        or protocol.get("aggregation") != preflight["aggregation"]
        or protocol.get("selected_weight_state_sha256")
        != preflight["checkpoint"]["selected_weight_state_sha256"]
    ):
        raise RuntimeError("T2M protocol manifest differs from pinned preflight")
    protocol_sha = sha256_file(protocol_path)
    records = []
    shard_evidence = []
    chunk_cache = ChunkCache()
    plan_rows = preflight["plan"]["rows"]
    plan_by_key = {row["case_key"]: row for row in plan_rows}
    expected_shard_by_key = {
        row["case_key"]: position % int(protocol["num_shards"])
        for position, row in enumerate(plan_rows)
    }
    expected_chunk_by_key: dict[str, tuple[str, int]] = {}
    generation_batch_size = int(protocol["sampling"]["generation_batch_size"])
    for shard_id in range(int(protocol["num_shards"])):
        shard_cases = [
            row
            for position, row in enumerate(plan_rows)
            if position % int(protocol["num_shards"]) == shard_id
        ]
        expected_chunk_by_key.update(
            _expected_chunk_assignments(
                output,
                shard_id=shard_id,
                shard_cases=shard_cases,
                batch_size=generation_batch_size,
            )
        )
    for shard_id in range(int(protocol["num_shards"])):
        shard_path = output / "shards" / f"shard_{shard_id:02d}.jsonl"
        shard_records = _read_records(shard_path)
        shard_evidence.append(
            {
                "shard_id": shard_id,
                "rows": len(shard_records),
                "path": str(shard_path),
                "sha256": sha256_file(shard_path),
            }
        )
        for record in shard_records:
            case = plan_by_key.get(str(record.get("case_key", "")))
            if case is None:
                raise RuntimeError(f"T2M shard contains unplanned case: {record.get('case_key')}")
            expected_shard = expected_shard_by_key[case["case_key"]]
            if expected_shard != shard_id:
                raise RuntimeError(f"T2M case is in wrong shard: {record['case_key']}")
            expected_chunk_path, expected_chunk_row = expected_chunk_by_key[
                case["case_key"]
            ]
            if (
                record.get("chunk_path") != expected_chunk_path
                or int(record.get("chunk_row", -1)) != expected_chunk_row
            ):
                raise RuntimeError(f"T2M case has wrong chunk assignment: {record['case_key']}")
            _validate_record_provenance(
                record,
                case,
                shard_id=shard_id,
                protocol_sha=protocol_sha,
                preflight_sha=preflight_sha,
                selected_weight_sha=protocol["selected_weight_state_sha256"],
                chunk_cache=chunk_cache,
                verify_quality=True,
            )
        records.extend(shard_records)
    by_key = {record["case_key"]: record for record in records}
    if len(records) != len(by_key) or len(records) != int(protocol["case_count"]):
        raise RuntimeError(
            f"Incomplete T2M evaluation: rows={len(records)} unique={len(by_key)} "
            f"expected={protocol['case_count']}"
        )
    ordered = [by_key[row["case_key"]] for row in preflight["plan"]["rows"]]

    if args.hymotion_root not in sys.path:
        sys.path.insert(0, args.hymotion_root)
    from hymotion.eval.metrics import (
        calculate_activation_statistics,
        calculate_diversity,
        calculate_frechet_distance,
    )
    from hymotion.eval.motionstreamer272 import MotionStreamer272Evaluator

    device = torch.device(args.device)
    evaluator = MotionStreamer272Evaluator.from_checkpoint(
        checkpoint=args.evaluator_checkpoint,
        distilbert_path=args.distilbert_path,
        mean_path=args.mean272,
        std_path=args.std272,
        device=device,
    )
    texts = [record["caption"] for record in ordered]
    text_embeddings = evaluator.encode_text(texts, batch_size=32)
    generated_embeddings = []
    oracle_embeddings = []
    native_embeddings = []
    k273_sha_cache: dict[str, str] = {}
    native_sha_cache: dict[str, str] = {}
    bridge_channel_sums = {
        "root_velocity": 0.0,
        "heading_delta": 0.0,
        "local_positions": 0.0,
        "local_velocities": 0.0,
        "local_rotations": 0.0,
    }
    bridge_channel_counts = {name: 0 for name in bridge_channel_sums}
    bridge_channel_max = {name: 0.0 for name in bridge_channel_sums}
    channel_slices = {
        "root_velocity": slice(0, 2),
        "heading_delta": slice(2, 8),
        "local_positions": slice(8, 74),
        "local_velocities": slice(74, 140),
        "local_rotations": slice(140, 272),
    }
    for start in range(0, len(ordered), 32):
        batch = ordered[start : start + 32]
        lengths = np.asarray([row["length"] for row in batch], dtype=np.int64)
        generated = np.zeros((len(batch), MAX_EVAL_FRAMES, 272), dtype=np.float32)
        oracle = np.zeros_like(generated)
        native = np.zeros_like(generated)
        for index, row in enumerate(batch):
            length = int(row["length"])
            generated[index, :length] = chunk_cache.row(row, "generated272")
            k273_path = str(row["k273_path"])
            if k273_path not in k273_sha_cache:
                k273_sha_cache[k273_path] = sha256_file(k273_path)
            if k273_sha_cache[k273_path] != row["k273_sha256"]:
                raise RuntimeError(f"K273 GT changed after preflight: {k273_path}")
            k273_value = np.load(k273_path).astype(np.float32)
            oracle_value = hy273_to_motionstreamer272(k273_value)
            oracle[index, :length] = oracle_value
            native_path = str(row["gt272_path"])
            if native_path not in native_sha_cache:
                native_sha_cache[native_path] = sha256_file(native_path)
            if native_sha_cache[native_path] != row["gt272_sha256"]:
                raise RuntimeError(f"GT272 changed after preflight: {row['gt272_path']}")
            native_value = np.load(native_path).astype(np.float32)
            native[index, :length] = native_value[:length]
            difference = np.abs(oracle_value - native_value[:length])
            for name, selected_slice in channel_slices.items():
                selected = difference[:, selected_slice]
                bridge_channel_sums[name] += float(selected.sum(dtype=np.float64))
                bridge_channel_counts[name] += int(selected.size)
                bridge_channel_max[name] = max(
                    bridge_channel_max[name], float(selected.max(initial=0.0))
                )
        generated_embeddings.append(evaluator.encode_motion(generated, lengths, batch_size=32))
        oracle_embeddings.append(evaluator.encode_motion(oracle, lengths, batch_size=32))
        native_embeddings.append(evaluator.encode_motion(native, lengths, batch_size=32))
    generated_embeddings = np.concatenate(generated_embeddings)
    oracle_embeddings = np.concatenate(oracle_embeddings)
    native_embeddings = np.concatenate(native_embeddings)
    generated_top, generated_mm = _retrieval_rows(text_embeddings, generated_embeddings)
    oracle_top, oracle_mm = _retrieval_rows(text_embeddings, oracle_embeddings)
    native_top, native_mm = _retrieval_rows(text_embeddings, native_embeddings)
    oracle_mu, oracle_cov = calculate_activation_statistics(oracle_embeddings)
    native_mu, native_cov = calculate_activation_statistics(native_embeddings)
    pred_mu, pred_cov = calculate_activation_statistics(generated_embeddings)
    quality_names = sorted({name for row in ordered for name in row["quality"]})
    quality = {
        name: float(np.mean([row["quality"][name] for row in ordered]))
        for name in quality_names
    }
    case_metric_path = output / "case_metrics.jsonl"
    case_metric_tmp = case_metric_path.with_name(
        f".{case_metric_path.name}.{os.getpid()}.tmp"
    )
    with case_metric_tmp.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(ordered):
            handle.write(
                json.dumps(
                    {
                        "case_key": row["case_key"],
                        "sample_seed": row["sample_seed"],
                        "length": row["length"],
                        "r_top1": float(generated_top[index, 0]),
                        "r_top2": float(generated_top[index, 1]),
                        "r_top3": float(generated_top[index, 2]),
                        "mm_dist": float(generated_mm[index]),
                        "oracle_gt_r_top1": float(oracle_top[index, 0]),
                        "oracle_gt_r_top2": float(oracle_top[index, 1]),
                        "oracle_gt_r_top3": float(oracle_top[index, 2]),
                        "oracle_gt_mm_dist": float(oracle_mm[index]),
                        "native_gt_r_top1": float(native_top[index, 0]),
                        "native_gt_r_top2": float(native_top[index, 1]),
                        "native_gt_r_top3": float(native_top[index, 2]),
                        "native_gt_mm_dist": float(native_mm[index]),
                        "quality": row["quality"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    durable_replace(case_metric_tmp, case_metric_path)
    chunk_evidence = [
        {"path": path, "sha256": sha}
        for path, sha in sorted(
            {
                (str(row["chunk_path"]), str(row["chunk_sha256"]))
                for row in ordered
            }
        )
    ]
    summary = {
        "format": SUMMARY_FORMAT,
        "status": "validated",
        "protocol": protocol,
        "preflight_sha256": preflight_sha,
        "case_count": len(ordered),
        "case_metrics_path": str(case_metric_path),
        "case_metrics_sha256": sha256_file(case_metric_path),
        "evidence": {
            "protocol_manifest_sha256": protocol_sha,
            "ordered_record_content_sha256": canonical_sha(ordered),
            "shards": shard_evidence,
            "chunks": chunk_evidence,
            "selected_weight_state_sha256": protocol[
                "selected_weight_state_sha256"
            ],
        },
        "metrics": {
            "fid": float(
                calculate_frechet_distance(oracle_mu, oracle_cov, pred_mu, pred_cov)
            ),
            "r_precision": {
                "top1": float(generated_top[:, 0].mean()),
                "top2": float(generated_top[:, 1].mean()),
                "top3": float(generated_top[:, 2].mean()),
            },
            "mm_dist": float(generated_mm.mean()),
            "diversity": float(
                calculate_diversity(
                    generated_embeddings,
                    300,
                    rng=_metric_rng(args.seed, "generated"),
                )
            ),
            "oracle_gt_r_precision": {
                "top1": float(oracle_top[:, 0].mean()),
                "top2": float(oracle_top[:, 1].mean()),
                "top3": float(oracle_top[:, 2].mean()),
            },
            "oracle_gt_mm_dist": float(oracle_mm.mean()),
            "oracle_gt_diversity": float(
                calculate_diversity(
                    oracle_embeddings,
                    300,
                    rng=_metric_rng(args.seed, "oracle_gt"),
                )
            ),
            "native_gt_r_precision": {
                "top1": float(native_top[:, 0].mean()),
                "top2": float(native_top[:, 1].mean()),
                "top3": float(native_top[:, 2].mean()),
            },
            "native_gt_mm_dist": float(native_mm.mean()),
            "native_gt_diversity": float(
                calculate_diversity(
                    native_embeddings,
                    300,
                    rng=_metric_rng(args.seed, "native_gt"),
                )
            ),
            "diagnostic_fid_to_native_gt272": float(
                calculate_frechet_distance(native_mu, native_cov, pred_mu, pred_cov)
            ),
            "conversion_floor_oracle_to_native_fid": float(
                calculate_frechet_distance(
                    native_mu, native_cov, oracle_mu, oracle_cov
                )
            ),
            "conversion_floor_channel_abs_error": {
                name: {
                    "mae": bridge_channel_sums[name]
                    / max(bridge_channel_counts[name], 1),
                    "max": bridge_channel_max[name],
                }
                for name in bridge_channel_sums
            },
            "quality": quality,
        },
    }
    summary_path = output / "summary.json"
    _atomic_json(summary_path, summary)
    artifact_index = {
        "format": ARTIFACT_INDEX_FORMAT,
        "schema_version": 1,
        "status": "validated",
        "case_count": len(ordered),
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "selected_weight_state_sha256": protocol[
            "selected_weight_state_sha256"
        ],
        "artifacts": {
            "preflight_manifest": {
                "path": str(Path(args.preflight_manifest).expanduser().resolve()),
                "sha256": preflight_sha,
            },
            "protocol_manifest": {
                "path": str(protocol_path),
                "sha256": protocol_sha,
            },
            "case_metrics": {
                "path": str(case_metric_path),
                "sha256": sha256_file(case_metric_path),
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "shards": shard_evidence,
            "chunks": chunk_evidence,
        },
    }
    artifact_index_path = output / "artifact_index.json"
    _atomic_json(artifact_index_path, artifact_index)
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "artifact_index": str(artifact_index_path),
                "cases": len(ordered),
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint_sha256", default="")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--gt272_root", default=DEFAULT_GT272_ROOT)
    parser.add_argument("--test_split", default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--k273_root", default=DEFAULT_K273_ROOT)
    parser.add_argument("--text_root", default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preflight_manifest", default="")
    parser.add_argument("--preflight_sha256", default="")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--weight_source", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--hymotion_root", default=DEFAULT_HYMOTION_ROOT)
    parser.add_argument("--evaluator_checkpoint", default=DEFAULT_EVALUATOR)
    parser.add_argument("--distilbert_path", default=DEFAULT_DISTILBERT)
    parser.add_argument("--mean272", default=DEFAULT_MEAN272)
    parser.add_argument("--std272", default=DEFAULT_STD272)
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
                    "case_count": payload["plan"]["case_count"],
                    "preflight_manifest": str(path),
                    "preflight_sha256": sha256_file(path),
                },
                sort_keys=True,
            )
        )
    elif args.aggregate:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for aggregate")
        aggregate(args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for shard generation")
        run_shard(args)


if __name__ == "__main__":
    main()
