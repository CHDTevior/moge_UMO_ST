"""Shardable Kimodo-like control benchmark for archived and multitask models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
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

from data.kimodo273_datasets import Kimodo273TextDataset
from models.raw_motion.hy273_kimodo_benchmark import (
    KIMODO_CONTROL_SUBTYPES,
    CompiledKimodoConstraint,
    compile_kimodo_constraint,
    evaluate_kimodo_constraint_case,
)
from models.raw_motion.hy273_kimodo_contact_benchmark import (
    V5_CONTACT_PROTOCOL,
    V5_CONTACT_SUBTYPES,
    CompiledKimodoContactConstraint,
    compile_kimodo_contact_constraint,
    evaluate_kimodo_contact_case,
)
from models.raw_motion.evidence_hash import (
    combined_tensor_sha256,
    state_dict_sha256,
    tensor_sha256,
)
from models.raw_motion.evidence_io import atomic_write_json, atomic_write_npz
from models.raw_motion.hy273_multitask_condition import CapabilityId, make_absent_condition
from models.raw_motion.hy273_normalizer import apply_kimodo_training_transform
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM, DIM_HY273
from models.raw_motion.flow_schedule import uses_unified_273_flow
from sample_hy273_multitask import (
    normalizer_from_checkpoint as multitask_normalizer_from_checkpoint,
    sample_hy273_multitask_ode,
)
from sample_hy273_raw import (
    ODESampleOutput,
    checkpoint_normalizer,
    checkpoint_weight_state,
    sample_ode,
)
from train_hy273_raw_flow import create_model as create_archived_model
from train_hy273_multitask import (
    CHECKPOINT_FORMAT as MULTITASK_CHECKPOINT_FORMAT,
    SUPPORTED_TRAIN_CONTRACTS as MULTITASK_TRAIN_CONTRACTS,
    contact_protocol_for_config,
    create_model_from_checkpoint as create_multitask_model_from_checkpoint,
    validate_assets as validate_multitask_assets,
    validate_frozen_contract as validate_multitask_contract,
)
from eval_hy273_kimodo_full_test import (
    checkpoint_metadata,
    verify_checkpoint_assets_coordinated,
)


TEXT_REGIMES = ("withtext", "notext")
V5_ALL_SUBTYPES = (*KIMODO_CONTROL_SUBTYPES, *V5_CONTACT_SUBTYPES)
PROTOCOL_VERSION = "hy273_hml3d_kimodo_constraints_v5_contact_evidence_v2"
PREFLIGHT_FORMAT = "hy273_kimodo_v5_contact_preflight_v2"
ASSET_ATTESTATION_FORMAT = "hy273_asset_verification_attestation_v1"
LEGACY_INITIAL_NOISE_PROTOCOL = "per_case_two_stream_cpu_float32_native_length_v1"
UNIFIED_INITIAL_NOISE_PROTOCOL = "per_case_unified_gaussian_cpu_float32_native_length_v2"
CONTROL_OUTPUT_FORMAT = "hy273_control_case_output_v1"
CONTROL_ARTIFACT_INDEX_FORMAT = "hy273_control_evidence_artifact_index_v1"
CONTROL_SUMMARY_FORMAT = "hy273_kimodo_v5_contact_summary_v3"
ARCHIVED_CHECKPOINT_KIND = "archived_kimodo_like"
MULTITASK_CHECKPOINT_KIND = "hy273_multitask_v2"
DEFAULT_DATA_ROOT = (
    "/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22"
)
DEFAULT_TEXT_ROOT = "/mnt/afs/mogo_base/datasets/HumanML3D/texts"

SCIENTIFIC_BENCHMARK_PROFILE = {
    "split": "test",
    "weight_source": "ema",
    "num_shards": 8,
    "num_steps": 32,
    "cfg_scale": 2.0,
    "control_cfg_scale": 2.0,
    "seed": 3407,
    "max_sparse_keyframes": 20,
    "cases_per_subtype": 0,
    "data_root": DEFAULT_DATA_ROOT,
    "text_root": DEFAULT_TEXT_ROOT,
    "max_frames": 300,
}


@dataclass(frozen=True)
class ContactCase:
    dataset_index: int
    subtype: str
    text_regime: str
    sample_seed: int

    @property
    def key(self) -> str:
        return f"index_{self.dataset_index:05d}__{self.subtype}__{self.text_regime}"


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


def _checkpoint_kind(checkpoint: dict[str, Any]) -> str:
    if checkpoint.get("format") == MULTITASK_CHECKPOINT_FORMAT:
        if checkpoint.get("train_contract") not in MULTITASK_TRAIN_CONTRACTS:
            raise RuntimeError("Multitask checkpoint train contract is incompatible")
        return MULTITASK_CHECKPOINT_KIND
    train_args = checkpoint.get("args")
    if (
        isinstance(train_args, dict)
        and str(train_args.get("architecture", "")) == "redenoise_kimodo_like"
        and {"model", "normalizer"} <= set(checkpoint)
    ):
        return ARCHIVED_CHECKPOINT_KIND
    raise RuntimeError("Unsupported checkpoint format for HY273 control evaluation")


def _environment_identity() -> dict[str, Any]:
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "name": properties.name,
                "total_memory": int(properties.total_memory),
                "capability": [int(properties.major), int(properties.minor)],
            }
        )
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": len(devices),
        "cuda_devices": devices,
    }


def _normalizer_state_sha256(checkpoint: dict[str, Any]) -> str:
    state = checkpoint.get("normalizer")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Control checkpoint has no pinned normalizer state")
    return state_dict_sha256(state)


def _runtime_metadata(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
    *,
    verify_assets: bool,
) -> dict[str, Any]:
    kind = _checkpoint_kind(checkpoint)
    unified_273_flow = False
    selected_state, resolved_source = checkpoint_weight_state(
        checkpoint, args.weight_source, str(Path(args.checkpoint).expanduser().resolve())
    )
    selected_sha = state_dict_sha256(selected_state)
    normalizer_sha = _normalizer_state_sha256(checkpoint)
    if kind == ARCHIVED_CHECKPOINT_KIND:
        train_args = argparse.Namespace(**checkpoint["args"])
        model_config_sha = canonical_sha(vars(train_args))
        model_assets: dict[str, Any] = {
            "kind": "archived_training_asset_manifest",
            "manifest": _asset_manifest_identity(train_args),
        }
        asset_verification: dict[str, Any] = {}
        if verify_assets:
            attestation_path, _ = verify_checkpoint_assets_coordinated(
                train_args,
                requested_cache_path=_asset_verification_cache(args),
                max_age_seconds=args.asset_verification_max_age,
            )
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            if (
                attestation.get("format") != ASSET_ATTESTATION_FORMAT
                or attestation.get("status") != "ok"
            ):
                raise RuntimeError("Archived asset verification did not pass")
            asset_verification = {
                "kind": "archived_full_asset_attestation",
                "path": str(attestation_path),
                "sha256": sha256_file(attestation_path),
                "payload": attestation,
            }
    else:
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise RuntimeError("Multitask checkpoint has no resolved config")
        validate_multitask_contract(config)
        unified_273_flow = uses_unified_273_flow(
            contact_protocol_for_config(config)
        )
        current_assets = validate_multitask_assets(
            config, include_full_preflight=False
        )
        recorded_config_sha = str(checkpoint.get("config_sha256", ""))
        model_config_sha = canonical_sha(config)
        if recorded_config_sha != model_config_sha:
            raise RuntimeError("Multitask checkpoint config SHA is invalid")
        model_assets = {
            "kind": "multitask_scientific_assets",
            "identity": current_assets,
            "matches_checkpoint_record": checkpoint.get("asset_identity") == current_assets,
        }
        asset_verification = {}
    model_asset_sha = canonical_sha(model_assets)
    inference_state = {
        "checkpoint_kind": kind,
        "unified_273_flow": unified_273_flow,
        "weight_source": resolved_source,
        "selected_weight_state_sha256": selected_sha,
        "normalizer_state_sha256": normalizer_sha,
        "model_config_sha256": model_config_sha,
        "model_asset_identity_sha256": model_asset_sha,
    }
    inference_state_sha = canonical_sha(inference_state)
    return {
        "checkpoint_kind": kind,
        "unified_273_flow": unified_273_flow,
        "resolved_weight_source": resolved_source,
        "selected_weight_state_sha256": selected_sha,
        "normalizer_state_sha256": normalizer_sha,
        "model_config_sha256": model_config_sha,
        "model_assets": model_assets,
        "asset_verification": asset_verification,
        "inference_state": inference_state,
        "inference_state_sha256": inference_state_sha,
    }


def _validate_production_profile(args: argparse.Namespace) -> None:
    if args.profile != "production":
        return
    mismatches = {
        name: (getattr(args, name), expected)
        for name, expected in SCIENTIFIC_BENCHMARK_PROFILE.items()
        if getattr(args, name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Control production benchmark profile mismatch: {mismatches}"
        )


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


def evaluation_code_identity() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "eval_hy273_kimodo_full_test.py",
        ROOT / "sample_hy273_raw.py",
        ROOT / "sample_hy273_multitask.py",
        ROOT / "train_hy273_raw_flow.py",
        ROOT / "train_hy273_multitask.py",
        ROOT / "data" / "kimodo273_datasets.py",
        ROOT / "models" / "codeflow" / "dit_blocks.py",
        *sorted((ROOT / "models" / "raw_motion").glob("*.py")),
    ]
    rows = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Evaluation dependency is missing: {resolved}")
        rows.append(
            {
                "path": str(resolved.relative_to(ROOT)),
                "size": int(resolved.stat().st_size),
                "sha256": sha256_file(resolved),
            }
        )
    return {"files": rows, "sha256": canonical_sha(rows)}


def _ordered_dataset_identity(
    dataset: Kimodo273TextDataset, *, split: str, caption_policy: str
) -> dict[str, Any]:
    split_path = dataset.data_root / "split_existing" / f"{split}.txt"
    split_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [
        {
            "dataset_index": index,
            "motion_id": str(record["id"]),
            "rel_path": str(record["rel"]),
            "frames": int(record["frames"]),
        }
        for index, record in enumerate(dataset.records)
    ]
    if split_ids != [row["motion_id"] for row in rows]:
        raise RuntimeError("Ordered test split does not match evaluator dataset records")
    return {
        "data_root": str(dataset.data_root.resolve()),
        "text_root": str(dataset.text_root.resolve()),
        "split": split,
        "caption_policy": caption_policy,
        "dataset_size": len(dataset),
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path),
        "ordered_records_sha256": canonical_sha(rows),
    }


def _asset_manifest_identity(train_args: argparse.Namespace) -> dict[str, Any]:
    path = Path(str(getattr(train_args, "asset_manifest_path", ""))).expanduser().resolve()
    expected_sha = str(getattr(train_args, "asset_manifest_sha256", "")).lower()
    if not path.is_file() or not expected_sha:
        raise RuntimeError("Archived Kimodo checkpoint has no pinned training asset manifest")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Training asset manifest SHA mismatch: expected={expected_sha}, actual={observed_sha}"
        )
    return {**_file_stat(path), "sha256": observed_sha}


def _plan_identity(plan: list[ContactCase], args: argparse.Namespace) -> dict[str, Any]:
    keys = [case.key for case in plan]
    return {
        "case_count": len(plan),
        "case_plan_sha256": canonical_sha(keys),
        "cases_per_subtype": int(args.cases_per_subtype),
        "seed": int(args.seed),
        "subtypes": list(V5_ALL_SUBTYPES),
        "text_regimes": list(TEXT_REGIMES),
        "num_shards": int(args.num_shards),
        "rows": [
            {
                "case_key": case.key,
                "dataset_index": case.dataset_index,
                "subtype": case.subtype,
                "text_regime": case.text_regime,
                "sample_seed": case.sample_seed,
            }
            for case in plan
        ],
    }


def _sampling_identity(
    args: argparse.Namespace, *, unified_273_flow: bool
) -> dict[str, Any]:
    unified = bool(unified_273_flow)
    return {
        "ode_steps": int(args.num_steps),
        "text_cfg_scale": float(args.cfg_scale),
        "control_cfg_scale": float(args.control_cfg_scale),
        "contact_init": "unified_273d_state" if unified else "random",
        "contact_feedback": "ode_273d" if unified else "blend",
        "cfg_apply_contacts": True,
        "primary_output": "raw_pre_exact_clamp",
        "max_sparse_keyframes": int(args.max_sparse_keyframes),
        "initial_noise": (
            UNIFIED_INITIAL_NOISE_PROTOCOL
            if unified
            else LEGACY_INITIAL_NOISE_PROTOCOL
        ),
    }


def _asset_verification_cache(args: argparse.Namespace) -> str:
    if args.asset_verification_cache:
        return str(Path(args.asset_verification_cache).expanduser().resolve())
    return str(
        Path(args.output_dir).expanduser().resolve()
        / "archived_asset_verification_attestation.json"
    )


def build_preflight_manifest(args: argparse.Namespace) -> dict[str, Any]:
    _validate_production_profile(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    stat_before = _file_stat(checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    stat_after = _file_stat(checkpoint_path)
    if stat_before != stat_after:
        raise RuntimeError("Checkpoint changed while its full SHA256 was computed")
    if args.checkpoint_sha256 and checkpoint_sha != args.checkpoint_sha256.lower():
        raise RuntimeError("Checkpoint SHA256 mismatch")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    runtime = _runtime_metadata(
        checkpoint, args, verify_assets=args.profile == "production"
    )
    dataset = _dataset(args, args.split)
    plan = build_plan(
        len(dataset), seed=args.seed, cases_per_subtype=args.cases_per_subtype
    )
    payload = {
        "format": PREFLIGHT_FORMAT,
        "status": "passed",
        "host": socket.gethostname(),
        "checkpoint": {
            **stat_after,
            "sha256": checkpoint_sha,
            "metadata": {
                **checkpoint_metadata(checkpoint),
                "next_global_step": int(checkpoint.get("next_global_step", -1)),
                "kind": runtime["checkpoint_kind"],
            },
        },
        "selected_weight": {
            "source": runtime["resolved_weight_source"],
            "state_dict_sha256": runtime["selected_weight_state_sha256"],
        },
        "inference_state": runtime["inference_state"],
        "inference_state_sha256": runtime["inference_state_sha256"],
        "code": evaluation_code_identity(),
        "environment": _environment_identity(),
        "dataset": _ordered_dataset_identity(
            dataset, split=args.split, caption_policy="first_full_motion"
        ),
        "plan": _plan_identity(plan, args),
        "sampling": _sampling_identity(
            args, unified_273_flow=bool(runtime["unified_273_flow"])
        ),
        "model_assets": runtime["model_assets"],
        "asset_verification": runtime["asset_verification"],
    }
    del checkpoint
    return payload


def _preflight_path(args: argparse.Namespace) -> Path:
    if args.preflight_manifest:
        return Path(args.preflight_manifest).expanduser().resolve()
    return Path(args.output_dir).expanduser().resolve() / "preflight_manifest.json"


def load_and_validate_preflight(
    args: argparse.Namespace,
    *,
    checkpoint: dict[str, Any],
    runtime: dict[str, Any],
    dataset: Kimodo273TextDataset,
    plan: list[ContactCase],
) -> tuple[Path, dict[str, Any], str]:
    path = _preflight_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"Missing preflight manifest: {path}; run --preflight_only first")
    observed_sha = sha256_file(path)
    if args.profile == "production":
        if not args.preflight_sha256:
            raise RuntimeError("Shard launch requires --preflight_sha256")
        if observed_sha != args.preflight_sha256.lower():
            raise RuntimeError("Preflight manifest SHA256 mismatch")
    elif args.preflight_sha256 and observed_sha != args.preflight_sha256.lower():
        raise RuntimeError("Research preflight file differs from the requested manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PREFLIGHT_FORMAT or payload.get("status") != "passed":
        raise RuntimeError("Invalid v5 contact preflight manifest")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    expected_checkpoint_metadata = {
        **checkpoint_metadata(checkpoint),
        "next_global_step": int(checkpoint.get("next_global_step", -1)),
        "kind": runtime["checkpoint_kind"],
    }
    if args.profile == "production":
        checkpoint_sha = sha256_file(checkpoint_path)
        if payload.get("host") != socket.gethostname():
            raise RuntimeError("Preflight was produced on a different host")
        if payload.get("checkpoint") != {
            **_file_stat(checkpoint_path),
            "sha256": checkpoint_sha,
            "metadata": expected_checkpoint_metadata,
        }:
            raise RuntimeError("Checkpoint identity changed after preflight")
        if (
            args.checkpoint_sha256
            and payload["checkpoint"]["sha256"] != args.checkpoint_sha256.lower()
        ):
            raise RuntimeError("Requested checkpoint SHA256 differs from preflight")
        if payload.get("code") != evaluation_code_identity():
            raise RuntimeError("Evaluation code changed after preflight")
        if payload.get("environment") != _environment_identity():
            raise RuntimeError("Control evaluation environment changed after preflight")
    else:
        checkpoint_identity = payload.get("checkpoint")
        if not isinstance(checkpoint_identity, dict) or checkpoint_identity.get(
            "metadata"
        ) != expected_checkpoint_metadata:
            raise RuntimeError("Research checkpoint semantics differ from preflight")
    if payload.get("dataset") != _ordered_dataset_identity(
        dataset, split=args.split, caption_policy="first_full_motion"
    ):
        raise RuntimeError("Evaluation dataset order changed after preflight")
    if payload.get("plan") != _plan_identity(plan, args):
        raise RuntimeError("Evaluation case plan changed after preflight")
    if payload.get("sampling") != _sampling_identity(
        args, unified_273_flow=bool(runtime["unified_273_flow"])
    ):
        raise RuntimeError("Evaluation sampling contract changed after preflight")
    for name in (
        "checkpoint_kind",
        "unified_273_flow",
        "resolved_weight_source",
        "selected_weight_state_sha256",
        "normalizer_state_sha256",
        "model_config_sha256",
        "model_assets",
        "inference_state",
        "inference_state_sha256",
    ):
        expected = (
            payload["selected_weight"]["source"]
            if name == "resolved_weight_source"
            else payload["selected_weight"]["state_dict_sha256"]
            if name == "selected_weight_state_sha256"
            else payload["inference_state"].get(name)
            if name in {
                "checkpoint_kind",
                "unified_273_flow",
                "normalizer_state_sha256",
                "model_config_sha256",
            }
            else payload.get(name)
        )
        if runtime.get(name) != expected:
            raise RuntimeError(f"Control inference identity changed after preflight: {name}")
    if payload.get("model_assets") != runtime["model_assets"]:
        raise RuntimeError("Control model assets changed after preflight")
    if payload.get("asset_verification") != runtime["asset_verification"]:
        raise RuntimeError("Control asset verification changed after preflight")
    return path, payload, observed_sha


def stable_seed(seed: int, dataset_index: int, subtype: str) -> int:
    # Keep the successful v5 case/noise plan unchanged while versioning evidence.
    payload = f"{V5_CONTACT_PROTOCOL}:{seed}:{dataset_index}:{subtype}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def build_plan(
    num_items: int, *, seed: int, cases_per_subtype: int = 0
) -> list[ContactCase]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(num_items, generator=generator).tolist()
    assignments = [
        (index, V5_ALL_SUBTYPES[position % len(V5_ALL_SUBTYPES)])
        for position, index in enumerate(permutation)
    ]
    if cases_per_subtype > 0:
        counts = {name: 0 for name in V5_ALL_SUBTYPES}
        kept = []
        for index, subtype in assignments:
            if counts[subtype] >= cases_per_subtype:
                continue
            counts[subtype] += 1
            kept.append((index, subtype))
        assignments = kept
    return [
        ContactCase(
            dataset_index=index,
            subtype=subtype,
            text_regime=regime,
            sample_seed=stable_seed(seed, index, subtype),
        )
        for index, subtype in assignments
        for regime in TEXT_REGIMES
    ]


def _atomic_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _ensure_protocol(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("Existing v5 contact protocol differs from requested run")
        return
    _atomic_json(path, payload)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL {path}:{line_number}") from exc
    return records


def _constraint_to_device(
    value: CompiledKimodoContactConstraint, device: torch.device
) -> CompiledKimodoContactConstraint:
    base = value.base
    moved_base = CompiledKimodoConstraint(
        observed_motion=base.observed_motion.to(device),
        motion_mask=base.motion_mask.to(device),
        root_metric_frames=base.root_metric_frames.to(device),
        fullbody_metric_frames=base.fullbody_metric_frames.to(device),
        endpoint_position_metric_mask=base.endpoint_position_metric_mask.to(device),
        endpoint_rotation_metric_mask=base.endpoint_rotation_metric_mask.to(device),
        components=base.components,
    )
    return CompiledKimodoContactConstraint(
        observed_motion=value.observed_motion.to(device),
        motion_mask=value.motion_mask.to(device),
        contact_metric_mask=value.contact_metric_mask.to(device),
        base=moved_base,
        components=value.components,
    )


def _base_constraint_to_device(
    value: CompiledKimodoConstraint, device: torch.device
) -> CompiledKimodoConstraint:
    return CompiledKimodoConstraint(
        observed_motion=value.observed_motion.to(device),
        motion_mask=value.motion_mask.to(device),
        root_metric_frames=value.root_metric_frames.to(device),
        fullbody_metric_frames=value.fullbody_metric_frames.to(device),
        endpoint_position_metric_mask=value.endpoint_position_metric_mask.to(device),
        endpoint_rotation_metric_mask=value.endpoint_rotation_metric_mask.to(device),
        components=value.components,
    )


def _noise_stream_seed(sample_seed: int, stream: str) -> int:
    digest = hashlib.sha256(
        f"hy273-control-noise-v1:{sample_seed}:{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _initial_control_noise(
    case: ContactCase, frames: int, *, unified: bool = False
) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    if unified:
        generator = torch.Generator(device="cpu").manual_seed(
            _noise_stream_seed(case.sample_seed, "unified_273d")
        )
        state = torch.randn(
            1, frames, DIM_HY273, generator=generator, dtype=torch.float32
        )
        continuous = state[..., :CONT_DIM]
        contacts = state[..., CONTACT_SLICE]
        continuous_sha = tensor_sha256(continuous)
        contact_sha = tensor_sha256(contacts)
        unified_sha = tensor_sha256(state)
        return continuous, contacts, {
            "initial_noise_protocol": UNIFIED_INITIAL_NOISE_PROTOCOL,
            "initial_continuous_noise_sha256": continuous_sha,
            "initial_contact_noise_sha256": contact_sha,
            "initial_unified_noise_sha256": unified_sha,
            "initial_noise_sha256": unified_sha,
        }
    continuous_generator = torch.Generator(device="cpu").manual_seed(
        _noise_stream_seed(case.sample_seed, "continuous")
    )
    contact_generator = torch.Generator(device="cpu").manual_seed(
        _noise_stream_seed(case.sample_seed, "contacts")
    )
    continuous = torch.randn(
        1, frames, 269, generator=continuous_generator, dtype=torch.float32
    )
    contacts = torch.rand(
        1, frames, 4, generator=contact_generator, dtype=torch.float32
    )
    continuous_sha = tensor_sha256(continuous)
    contact_sha = tensor_sha256(contacts)
    return continuous, contacts, {
        "initial_noise_protocol": LEGACY_INITIAL_NOISE_PROTOCOL,
        "initial_continuous_noise_sha256": continuous_sha,
        "initial_contact_noise_sha256": contact_sha,
        "initial_noise_sha256": canonical_sha(
            {"continuous": continuous_sha, "contacts": contact_sha}
        ),
    }


def _constraint_payload_sha256(
    constraint: CompiledKimodoConstraint | CompiledKimodoContactConstraint,
    c_dir: torch.Tensor,
) -> str:
    base = constraint.base if isinstance(constraint, CompiledKimodoContactConstraint) else constraint
    tensors = {
        "observed_motion": constraint.observed_motion.contiguous(),
        "motion_mask": constraint.motion_mask.contiguous(),
        "root_metric_frames": base.root_metric_frames.contiguous(),
        "fullbody_metric_frames": base.fullbody_metric_frames.contiguous(),
        "endpoint_position_metric_mask": base.endpoint_position_metric_mask.contiguous(),
        "endpoint_rotation_metric_mask": base.endpoint_rotation_metric_mask.contiguous(),
        "c_dir": c_dir.contiguous(),
    }
    if isinstance(constraint, CompiledKimodoContactConstraint):
        tensors["contact_metric_mask"] = constraint.contact_metric_mask.contiguous()
    return canonical_sha(
        {
            "tensor_payload_sha256": combined_tensor_sha256(tensors),
            "components": constraint.components,
        }
    )


def _prepare_case(
    dataset: Kimodo273TextDataset,
    case: ContactCase,
    *,
    max_sparse_keyframes: int,
    target_asset_sha_cache: dict[str, str],
    unified_noise: bool,
) -> dict[str, Any]:
    item = dataset[case.dataset_index]
    transformed = apply_kimodo_training_transform(
        item["motion"].float().unsqueeze(0), random_heading=False, root_shift=True
    )
    target_cpu = transformed.motion[0].contiguous()
    is_contact_case = case.subtype in V5_CONTACT_SUBTYPES
    constraint_cpu = (
        compile_kimodo_contact_constraint(
            target_cpu,
            case.subtype,
            seed=case.sample_seed,
            max_sparse_keyframes=max_sparse_keyframes,
        )
        if is_contact_case
        else compile_kimodo_constraint(
            target_cpu,
            case.subtype,
            seed=case.sample_seed,
            max_sparse_keyframes=max_sparse_keyframes,
        )
    )
    c_dir_cpu = transformed.c_dir[0].contiguous()
    source_path = Path(dataset.records[case.dataset_index]["path"]).resolve()
    source_key = str(source_path)
    if source_key not in target_asset_sha_cache:
        target_asset_sha_cache[source_key] = sha256_file(source_path)
    initial_continuous_noise, initial_contact_noise, noise_evidence = (
        _initial_control_noise(
            case, int(target_cpu.shape[0]), unified=unified_noise
        )
    )
    evidence = {
        "motion_id": str(item["motion_id"]),
        "length": int(target_cpu.shape[0]),
        "text": str(item["text"]) if case.text_regime == "withtext" else "",
        "target_asset_path": source_key,
        "target_asset_sha256": target_asset_sha_cache[source_key],
        "target_tensor_sha256": tensor_sha256(target_cpu),
        "observed_motion_sha256": tensor_sha256(
            constraint_cpu.observed_motion.contiguous()
        ),
        "motion_mask_sha256": tensor_sha256(constraint_cpu.motion_mask.contiguous()),
        "c_dir_sha256": tensor_sha256(c_dir_cpu),
        "constraint_payload_sha256": _constraint_payload_sha256(
            constraint_cpu, c_dir_cpu
        ),
        "components": constraint_cpu.components,
        "contact_control_entries": int(
            constraint_cpu.contact_metric_mask.sum().item()
            if is_contact_case
            else 0
        ),
        "model_mask_fraction": float(
            constraint_cpu.motion_mask.float().mean().item()
        ),
        **noise_evidence,
    }
    return {
        "item": item,
        "target_cpu": target_cpu,
        "constraint_cpu": constraint_cpu,
        "c_dir_cpu": c_dir_cpu,
        "is_contact_case": is_contact_case,
        "initial_continuous_noise": initial_continuous_noise,
        "initial_contact_noise": initial_contact_noise,
        "evidence": evidence,
    }


def _validate_case_evidence(
    record: dict[str, Any],
    case: ContactCase,
    prepared: dict[str, Any],
    *,
    output_dir: Path | None = None,
    verify_output: bool = False,
) -> None:
    expected = prepared["evidence"]
    if any(record.get(name) != value for name, value in expected.items()):
        changed = [
            name for name, value in expected.items() if record.get(name) != value
        ]
        raise RuntimeError(
            f"Control case evidence changed for {case.key}: {changed}"
        )
    if verify_output:
        if output_dir is None:
            raise ValueError("output_dir is required when control output is verified")
        raw, exact = _load_case_output(record, case, prepared, output_dir)
        recomputed = _evaluate_case_outputs(raw, exact, prepared)
        if canonical_sha(record.get("metrics")) != canonical_sha(recomputed):
            raise RuntimeError(f"Stored control metrics changed for {case.key}")


def _case_output_path(output_dir: Path, case: ContactCase) -> Path:
    return output_dir / "case_outputs" / f"{case.key}.npz"


def _physical_exact_clamp(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the diagnostic overwrite after returning to physical K273 space."""

    if prediction.shape != observed.shape or mask.shape != prediction.shape:
        raise ValueError("Physical exact-clamp tensors must have identical shapes")
    if mask.dtype != torch.bool:
        raise TypeError("Physical exact-clamp mask must be bool")
    return torch.where(mask, observed, prediction)


def _write_case_output(
    output_dir: Path,
    case: ContactCase,
    raw_motion: torch.Tensor,
    exact_motion: torch.Tensor,
) -> tuple[Path, str]:
    path = _case_output_path(output_dir, case)
    raw = raw_motion.detach().cpu().float().contiguous().numpy()
    exact = exact_motion.detach().cpu().float().contiguous().numpy()
    atomic_write_npz(
        path,
        format=np.asarray(CONTROL_OUTPUT_FORMAT),
        case_key=np.asarray(case.key),
        length=np.asarray(raw.shape[0], dtype=np.int64),
        generated_raw=raw,
        diagnostic_exact_clamp=exact,
    )
    return path, sha256_file(path)


def _load_case_output(
    record: dict[str, Any],
    case: ContactCase,
    prepared: dict[str, Any],
    output_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_path = _case_output_path(output_dir, case).resolve()
    recorded_path = Path(str(record.get("output_path", ""))).expanduser().resolve()
    if recorded_path != expected_path or not recorded_path.is_file():
        raise RuntimeError(f"Control output path is missing or unexpected: {case.key}")
    expected_sha = str(record.get("output_sha256", ""))
    if len(expected_sha) != 64 or sha256_file(recorded_path) != expected_sha:
        raise RuntimeError(f"Control output SHA mismatch: {case.key}")
    with np.load(recorded_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "format",
            "case_key",
            "length",
            "generated_raw",
            "diagnostic_exact_clamp",
        }:
            raise RuntimeError(f"Unexpected control output schema: {case.key}")
        if str(archive["format"].item()) != CONTROL_OUTPUT_FORMAT:
            raise RuntimeError(f"Control output format mismatch: {case.key}")
        if str(archive["case_key"].item()) != case.key:
            raise RuntimeError(f"Control output case key mismatch: {case.key}")
        length = int(archive["length"].item())
        raw = np.asarray(archive["generated_raw"]).copy()
        exact = np.asarray(archive["diagnostic_exact_clamp"]).copy()
    expected_length = int(prepared["target_cpu"].shape[0])
    if (
        length != expected_length
        or raw.shape != (length, 273)
        or exact.shape != (length, 273)
        or raw.dtype != np.float32
        or exact.dtype != np.float32
        or not np.isfinite(raw).all()
        or not np.isfinite(exact).all()
    ):
        raise RuntimeError(f"Invalid control output payload: {case.key}")
    raw_tensor = torch.from_numpy(raw)
    exact_tensor = torch.from_numpy(exact)
    constraint = prepared["constraint_cpu"]
    if not torch.equal(
        exact_tensor[constraint.motion_mask],
        constraint.observed_motion[constraint.motion_mask],
    ):
        raise RuntimeError(f"Exact-clamped control output violates observations: {case.key}")
    return raw_tensor, exact_tensor


def _evaluate_case_outputs(
    raw_motion: torch.Tensor,
    exact_motion: torch.Tensor,
    prepared: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    target = prepared["target_cpu"]
    constraint = prepared["constraint_cpu"]
    evaluator = (
        evaluate_kimodo_contact_case
        if prepared["is_contact_case"]
        else evaluate_kimodo_constraint_case
    )
    return {
        "generated_raw": evaluator(raw_motion, target, constraint),
        "diagnostic_exact_clamp": evaluator(exact_motion, target, constraint),
        "ground_truth": evaluator(target, target, constraint),
    }


def _dataset(args: argparse.Namespace, split: str) -> Kimodo273TextDataset:
    return Kimodo273TextDataset(
        args.data_root,
        split=split,
        text_root=args.text_root or None,
        max_frames=int(args.max_frames),
        min_frames=2,
        random_crop=False,
        exclude_fallback_short_clips=False,
        deterministic_text=True,
        caption_policy="first_full_motion",
    )


def _load_model_runtime(
    checkpoint: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, torch.nn.Module, Any, bool]:
    state = checkpoint[runtime["resolved_weight_source"]]
    if state_dict_sha256(state) != runtime["selected_weight_state_sha256"]:
        raise RuntimeError("Selected control weight state changed before model load")
    if runtime["checkpoint_kind"] == ARCHIVED_CHECKPOINT_KIND:
        train_args = argparse.Namespace(**checkpoint["args"])
        if str(getattr(train_args, "prediction_type", "")) != "x0":
            raise RuntimeError("Archived control baseline requires x0 prediction")
        model = create_archived_model(train_args).to(device)
        normalizer = checkpoint_normalizer(
            checkpoint, train_args, device, str(Path(args.checkpoint).expanduser().resolve())
        )
        self_conditioning = bool(getattr(train_args, "self_conditioning", False))
    else:
        model = create_multitask_model_from_checkpoint(checkpoint).to(device)
        normalizer = multitask_normalizer_from_checkpoint(checkpoint, device)
        self_conditioning = False
    if bool(normalizer.normalize_contacts) != bool(runtime["unified_273_flow"]):
        raise RuntimeError("Control runtime contact protocol differs from checkpoint metadata")
    model.load_state_dict(state, strict=True)
    model.eval()
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("Frozen control evaluation requires float32 model weights")
    return runtime["checkpoint_kind"], model, normalizer, self_conditioning


def _sample_control_case(
    *,
    kind: str,
    model: torch.nn.Module,
    normalizer: Any,
    self_conditioning: bool,
    prepared: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Any:
    target = prepared["target_cpu"].to(device)
    constraint_cpu = prepared["constraint_cpu"]
    is_contact_case = bool(prepared["is_contact_case"])
    constraint = (
        _constraint_to_device(constraint_cpu, device)
        if is_contact_case
        else _base_constraint_to_device(constraint_cpu, device)
    )
    text = str(prepared["evidence"]["text"])
    if kind == ARCHIVED_CHECKPOINT_KIND:
        sampled = sample_ode(
            model,
            normalizer,
            torch.tensor([target.shape[0]], device=device),
            [text],
            constraint.observed_motion.unsqueeze(0),
            constraint.motion_mask.unsqueeze(0),
            prepared["c_dir_cpu"].unsqueeze(0).to(device),
            num_steps=args.num_steps,
            self_conditioning=self_conditioning,
            cfg_scale=args.cfg_scale,
            control_cfg_scale=args.control_cfg_scale,
            contact_init="random",
            contact_feedback="blend",
            cfg_apply_contacts=True,
            prediction_type="x0",
            velocity_t_eps=1e-4,
            return_details=True,
            initial_continuous_noise=prepared["initial_continuous_noise"],
            initial_contact_noise=prepared["initial_contact_noise"],
        )
        if not isinstance(sampled, ODESampleOutput):
            raise AssertionError("Expected detailed archived ODE output")
        return sampled
    lengths = torch.tensor([target.shape[0]], dtype=torch.long)
    condition = make_absent_condition(
        batch_size=1,
        target_frames=int(target.shape[0]),
        target_lengths=lengths,
        capability=CapabilityId.KIMODO_CONTROL,
    )
    condition = replace(
        condition,
        frame_gauge_dir=prepared["c_dir_cpu"].reshape(1, 2).float(),
    )
    condition.validate()
    unified_273_flow = bool(normalizer.normalize_contacts)
    noise_kwargs = (
        {
            "initial_unified_noise": torch.cat(
                [
                    prepared["initial_continuous_noise"],
                    prepared["initial_contact_noise"],
                ],
                dim=-1,
            )
        }
        if unified_273_flow
        else {
            "initial_continuous_noise": prepared["initial_continuous_noise"],
            "initial_contact_noise": prepared["initial_contact_noise"],
        }
    )
    return sample_hy273_multitask_ode(
        model,
        normalizer,
        condition,
        [text],
        constraint.observed_motion.unsqueeze(0),
        constraint.motion_mask.unsqueeze(0),
        num_steps=args.num_steps,
        text_cfg_scale=args.cfg_scale,
        control_cfg_scale=args.control_cfg_scale,
        **noise_kwargs,
    )


def run_shard(args: argparse.Namespace) -> None:
    _validate_production_profile(args)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must be in [0,num_shards)")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    runtime = _runtime_metadata(
        checkpoint, args, verify_assets=args.profile == "production"
    )
    dataset = _dataset(args, args.split)
    plan = build_plan(
        len(dataset), seed=args.seed, cases_per_subtype=args.cases_per_subtype
    )
    preflight_path, preflight, preflight_sha = load_and_validate_preflight(
        args,
        checkpoint=checkpoint,
        runtime=runtime,
        dataset=dataset,
        plan=plan,
    )
    if preflight.get("selected_weight") != {
        "source": runtime["resolved_weight_source"],
        "state_dict_sha256": runtime["selected_weight_state_sha256"],
    }:
        raise RuntimeError("Selected checkpoint weight tensor state changed after preflight")
    expected_keys = [case.key for case in plan]
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "checkpoint_size": preflight["checkpoint"]["size"],
        "checkpoint_kind": runtime["checkpoint_kind"],
        "unified_273_flow": bool(runtime["unified_273_flow"]),
        "requested_weight_source": args.weight_source,
        "weight_source": runtime["resolved_weight_source"],
        "selected_weight_state_sha256": runtime["selected_weight_state_sha256"],
        "normalizer_state_sha256": runtime["normalizer_state_sha256"],
        "model_config_sha256": runtime["model_config_sha256"],
        "model_asset_identity_sha256": runtime["inference_state"][
            "model_asset_identity_sha256"
        ],
        "inference_state_sha256": runtime["inference_state_sha256"],
        "profile": args.profile,
        "dataset_split": args.split,
        "dataset_size": len(dataset),
        "case_count": len(plan),
        "case_plan_sha256": canonical_sha(expected_keys),
        "subtypes": list(V5_ALL_SUBTYPES),
        "legacy_subtypes": list(KIMODO_CONTROL_SUBTYPES),
        "contact_subtypes": list(V5_CONTACT_SUBTYPES),
        "text_regimes": list(TEXT_REGIMES),
        "seed": int(args.seed),
        "ode_steps": int(args.num_steps),
        "text_cfg_scale": float(args.cfg_scale),
        "control_cfg_scale": float(args.control_cfg_scale),
        "contact_init": (
            "unified_273d_state"
            if runtime["unified_273_flow"]
            else "random"
        ),
        "contact_feedback": (
            "ode_273d"
            if runtime["unified_273_flow"]
            else "blend"
        ),
        "cfg_apply_contacts": True,
        "primary_output": "raw_pre_exact_clamp",
        "max_sparse_keyframes": int(args.max_sparse_keyframes),
        "initial_noise": (
            UNIFIED_INITIAL_NOISE_PROTOCOL
            if runtime["unified_273_flow"]
            else LEGACY_INITIAL_NOISE_PROTOCOL
        ),
        "num_shards": int(args.num_shards),
        "preflight_manifest": {
            "path": str(preflight_path),
            "sha256": preflight_sha,
            "code_sha256": preflight["code"]["sha256"],
            "dataset_order_sha256": preflight["dataset"]["ordered_records_sha256"],
            "environment_sha256": canonical_sha(preflight["environment"]),
            "model_asset_identity_sha256": runtime["inference_state"][
                "model_asset_identity_sha256"
            ],
        },
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    protocol_path = output_dir / "protocol_manifest.json"
    _ensure_protocol(protocol_path, protocol)
    protocol_sha = sha256_file(protocol_path)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    kind, model, normalizer, self_conditioning = _load_model_runtime(
        checkpoint, runtime, args, device
    )
    del checkpoint

    shard_cases = [
        case for position, case in enumerate(plan) if position % args.num_shards == args.shard_id
    ]
    shard_path = output_dir / "shards" / f"shard_{args.shard_id:02d}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and shard_path.exists():
        shard_path.unlink()
    records = _load_records(shard_path)
    by_key = {record["case_key"]: record for record in records}
    if len(by_key) != len(records):
        raise RuntimeError("Duplicate v5 contact case records")
    expected_by_key = {case.key: case for case in shard_cases}
    target_asset_sha_cache: dict[str, str] = {}
    for record in records:
        case = expected_by_key.get(str(record.get("case_key", "")))
        if case is None:
            raise RuntimeError("Existing shard contains a case outside its frozen assignment")
        if (
            record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("protocol_manifest_sha256") != protocol_sha
            or record.get("preflight_manifest_sha256") != preflight_sha
            or int(record.get("shard_id", -1)) != args.shard_id
            or int(record.get("dataset_index", -1)) != case.dataset_index
            or record.get("subtype") != case.subtype
            or record.get("text_regime") != case.text_regime
            or int(record.get("sample_seed", -1)) != case.sample_seed
        ):
            raise RuntimeError(f"Existing record provenance changed: {record.get('case_key')}")
        prepared = _prepare_case(
            dataset,
            case,
            max_sparse_keyframes=args.max_sparse_keyframes,
            target_asset_sha_cache=target_asset_sha_cache,
            unified_noise=bool(runtime["unified_273_flow"]),
        )
        if (
            record.get("weight_source") != runtime["resolved_weight_source"]
            or record.get("selected_weight_state_sha256")
            != runtime["selected_weight_state_sha256"]
            or record.get("inference_state_sha256")
            != runtime["inference_state_sha256"]
        ):
            raise RuntimeError(f"Existing record model identity changed: {case.key}")
        _validate_case_evidence(
            record,
            case,
            prepared,
            output_dir=output_dir,
            verify_output=True,
        )
    with shard_path.open("a", encoding="utf-8") as writer:
        started = time.perf_counter()
        new_count = 0
        for case in shard_cases:
            if case.key in by_key:
                continue
            case_started = time.perf_counter()
            prepared = _prepare_case(
                dataset,
                case,
                max_sparse_keyframes=args.max_sparse_keyframes,
                target_asset_sha_cache=target_asset_sha_cache,
                unified_noise=bool(runtime["unified_273_flow"]),
            )
            sampled = _sample_control_case(
                kind=kind,
                model=model,
                normalizer=normalizer,
                self_conditioning=self_conditioning,
                prepared=prepared,
                args=args,
                device=device,
            )
            raw_motion = sampled.raw_motion[0]
            constraint_cpu = prepared["constraint_cpu"]
            exact_motion = _physical_exact_clamp(
                raw_motion.detach().cpu().float(),
                constraint_cpu.observed_motion,
                constraint_cpu.motion_mask,
            )
            output_path, output_sha = _write_case_output(
                output_dir,
                case,
                raw_motion,
                exact_motion,
            )
            output_stub = {
                "output_path": str(output_path.resolve()),
                "output_sha256": output_sha,
            }
            raw_cpu, exact_cpu = _load_case_output(
                output_stub, case, prepared, output_dir
            )
            metrics = _evaluate_case_outputs(raw_cpu, exact_cpu, prepared)
            record = {
                "status": "ok",
                "protocol_version": PROTOCOL_VERSION,
                "protocol_manifest_sha256": protocol_sha,
                "case_key": case.key,
                "dataset_index": case.dataset_index,
                "subtype": case.subtype,
                "text_regime": case.text_regime,
                "sample_seed": case.sample_seed,
                **prepared["evidence"],
                **output_stub,
                "metrics": metrics,
                "elapsed_seconds": time.perf_counter() - case_started,
                "shard_id": args.shard_id,
                "weight_source": runtime["resolved_weight_source"],
                "selected_weight_state_sha256": runtime[
                    "selected_weight_state_sha256"
                ],
                "inference_state_sha256": runtime["inference_state_sha256"],
                "preflight_manifest_sha256": preflight_sha,
            }
            writer.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
            by_key[case.key] = record
            new_count += 1
            elapsed = time.perf_counter() - started
            done = len(by_key)
            rate = new_count / elapsed if elapsed > 0 else 0.0
            print(
                json.dumps(
                    {
                        "shard_id": args.shard_id,
                        "completed": done,
                        "total": len(shard_cases),
                        "cases_per_second": rate,
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


def aggregate(args: argparse.Namespace) -> None:
    _validate_production_profile(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    protocol_path = output_dir / "protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Aggregate requires the evidence-v2 control protocol")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if str(checkpoint_path) != protocol.get("checkpoint"):
        raise RuntimeError("Aggregate checkpoint path differs from the control protocol")
    if sha256_file(checkpoint_path) != protocol.get("checkpoint_sha256"):
        raise RuntimeError("Aggregate checkpoint content differs from the control protocol")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=False
    )
    runtime = _runtime_metadata(
        checkpoint, args, verify_assets=args.profile == "production"
    )
    dataset = _dataset(args, args.split)
    plan = build_plan(
        len(dataset), seed=args.seed, cases_per_subtype=args.cases_per_subtype
    )
    _, preflight, preflight_sha = load_and_validate_preflight(
        args,
        checkpoint=checkpoint,
        runtime=runtime,
        dataset=dataset,
        plan=plan,
    )
    if (
        runtime["resolved_weight_source"] != protocol.get("weight_source")
        or runtime["selected_weight_state_sha256"]
        != protocol.get("selected_weight_state_sha256")
        or runtime["inference_state_sha256"]
        != protocol.get("inference_state_sha256")
        or runtime["normalizer_state_sha256"]
        != protocol.get("normalizer_state_sha256")
        or runtime["model_config_sha256"] != protocol.get("model_config_sha256")
        or preflight.get("selected_weight")
        != {
            "source": runtime["resolved_weight_source"],
            "state_dict_sha256": runtime["selected_weight_state_sha256"],
        }
    ):
        raise RuntimeError("Aggregate control inference identity mismatch")
    del checkpoint
    records: list[dict[str, Any]] = []
    shard_evidence = []
    for shard_id in range(int(protocol["num_shards"])):
        shard_path = output_dir / "shards" / f"shard_{shard_id:02d}.jsonl"
        shard_records = _load_records(shard_path)
        records.extend(shard_records)
        shard_evidence.append(
            {"shard_id": shard_id, "rows": len(shard_records), "sha256": sha256_file(shard_path)}
        )
    by_key = {record["case_key"]: record for record in records}
    if len(by_key) != len(records):
        raise RuntimeError("Duplicate case keys across v5 contact shards")
    if len(records) != int(protocol["case_count"]):
        raise RuntimeError(
            f"Incomplete v5 contact run: {len(records)}/{protocol['case_count']}"
        )
    preflight_path = Path(protocol["preflight_manifest"]["path"]).resolve()
    if sha256_file(preflight_path) != protocol["preflight_manifest"]["sha256"]:
        raise RuntimeError("Control preflight changed before aggregate")
    plan_rows = preflight["plan"]["rows"]
    expected = {row["case_key"]: row for row in plan_rows}
    expected_shards = {
        row["case_key"]: position % int(protocol["num_shards"])
        for position, row in enumerate(plan_rows)
    }
    protocol_sha = sha256_file(protocol_path)
    required_hash_fields = (
        "target_asset_sha256",
        "target_tensor_sha256",
        "observed_motion_sha256",
        "motion_mask_sha256",
        "c_dir_sha256",
        "constraint_payload_sha256",
        "initial_continuous_noise_sha256",
        "initial_contact_noise_sha256",
        "initial_noise_sha256",
        "output_sha256",
    )
    target_asset_sha_cache: dict[str, str] = {}
    plan_case_by_key = {case.key: case for case in plan}
    for record in records:
        case = expected.get(record["case_key"])
        if case is None:
            raise RuntimeError(f"Aggregate found an unplanned case: {record['case_key']}")
        expected_shard = expected_shards[record["case_key"]]
        if (
            record.get("status") != "ok"
            or record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("protocol_manifest_sha256") != protocol_sha
            or record.get("preflight_manifest_sha256") != protocol["preflight_manifest"]["sha256"]
            or int(record.get("shard_id", -1)) != expected_shard
            or any(record.get(name) != case[name] for name in ("dataset_index", "subtype", "text_regime", "sample_seed"))
            or record.get("selected_weight_state_sha256") != protocol["selected_weight_state_sha256"]
            or record.get("inference_state_sha256") != protocol["inference_state_sha256"]
            or record.get("weight_source") != protocol["weight_source"]
        ):
            raise RuntimeError(f"Aggregate record provenance mismatch: {record['case_key']}")
        for name in required_hash_fields:
            value = str(record.get(name, ""))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise RuntimeError(f"Invalid {name} for {record['case_key']}")
        frozen_case = plan_case_by_key[record["case_key"]]
        prepared = _prepare_case(
            dataset,
            frozen_case,
            max_sparse_keyframes=args.max_sparse_keyframes,
            target_asset_sha_cache=target_asset_sha_cache,
            unified_noise=bool(protocol["unified_273_flow"]),
        )
        _validate_case_evidence(
            record,
            frozen_case,
            prepared,
            output_dir=output_dir,
            verify_output=True,
        )
    rows = []
    for regime in TEXT_REGIMES:
        for subtype in (*V5_ALL_SUBTYPES, "all"):
            selected = [
                record
                for record in records
                if record["text_regime"] == regime
                and (subtype == "all" or record["subtype"] == subtype)
            ]
            metrics: dict[str, dict[str, float]] = {}
            for pass_name in (
                "generated_raw",
                "diagnostic_exact_clamp",
                "ground_truth",
            ):
                names = sorted(
                    {
                        name
                        for record in selected
                        for name in record["metrics"][pass_name]
                        if not name.endswith("entries")
                    }
                )
                metrics[pass_name] = {
                    name: float(
                        torch.tensor(
                            [
                                record["metrics"][pass_name][name]
                                for record in selected
                                if name in record["metrics"][pass_name]
                            ],
                            dtype=torch.float64,
                        ).mean()
                    )
                    for name in names
                }
            rows.append(
                {
                    "text_regime": regime,
                    "subtype": subtype,
                    "case_count": len(selected),
                    **metrics,
                }
            )
    output_evidence = []
    for record in sorted(records, key=lambda row: str(row["output_path"])):
        path = Path(record["output_path"]).resolve()
        if sha256_file(path) != record["output_sha256"]:
            raise RuntimeError(f"Control output changed before aggregate: {path}")
        output_evidence.append(
            {
                "case_key": record["case_key"],
                "path": str(path),
                "sha256": record["output_sha256"],
            }
        )
    summary = {
        "format": CONTROL_SUMMARY_FORMAT,
        "status": "validated",
        "protocol": protocol,
        "protocol_manifest_sha256": protocol_sha,
        "case_count": len(records),
        "case_rows_sha256": canonical_sha(records),
        "shards": shard_evidence,
        "case_outputs": output_evidence,
        "rows": rows,
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(summary_path, summary)
    artifact_index = {
        "format": CONTROL_ARTIFACT_INDEX_FORMAT,
        "schema_version": 1,
        "status": "validated",
        "profile": args.profile,
        "case_count": len(records),
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "inference_state_sha256": protocol["inference_state_sha256"],
        "artifacts": {
            "preflight_manifest": protocol["preflight_manifest"],
            "protocol_manifest": {
                "path": str(protocol_path),
                "sha256": protocol_sha,
            },
            "summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "shards": shard_evidence,
            "case_outputs": output_evidence,
        },
    }
    artifact_index_path = output_dir / "artifact_index.json"
    _atomic_json(artifact_index_path, artifact_index)
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
    parser.add_argument(
        "--profile", choices=["research", "production", "smoke"], default="research"
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint_sha256", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--preflight_manifest", default="")
    parser.add_argument("--preflight_sha256", default="")
    parser.add_argument("--asset_verification_cache", default="")
    parser.add_argument(
        "--asset_verification_max_age", type=float, default=24 * 60 * 60
    )
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text_root", default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--weight_source", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--control_cfg_scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_sparse_keyframes", type=int, default=20)
    parser.add_argument("--cases_per_subtype", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight_only", action="store_true")
    mode.add_argument("--aggregate", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for preflight")
        payload = build_preflight_manifest(args)
        path = Path(args.output_dir).expanduser().resolve() / "preflight_manifest.json"
        _atomic_json(path, payload)
        print(
            json.dumps(
                {
                    "passed": True,
                    "preflight_manifest": str(path),
                    "preflight_sha256": sha256_file(path),
                    "checkpoint_sha256": payload["checkpoint"]["sha256"],
                    "case_count": payload["plan"]["case_count"],
                },
                sort_keys=True,
            )
        )
    elif args.aggregate:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for aggregate evidence validation")
        aggregate(args)
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --aggregate is used")
        run_shard(args)


if __name__ == "__main__":
    main()
