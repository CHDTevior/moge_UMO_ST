"""Full HumanML3D control benchmark using Kimodo's public evaluation semantics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from typing import Any, Iterable

import numpy as np
import torch

from data.kimodo273_datasets import Kimodo273TextDataset
from models.raw_motion.asset_integrity import sha256_file
from models.raw_motion.hy273_kimodo_benchmark import (
    KIMODO_CONTROL_SUBTYPES,
    TEXT_REGIMES,
    CompiledKimodoConstraint,
    aggregate_case_records,
    build_kimodo_case_plan,
    compile_kimodo_constraint,
    evaluate_kimodo_constraint_case,
    shard_kimodo_case_plan,
)
from models.raw_motion.hy273_normalizer import apply_kimodo_training_transform
from sample_hy273_raw import (
    ODESampleOutput,
    apply_checkpoint_path_override,
    checkpoint_normalizer,
    checkpoint_weight_state,
    sample_ode,
    verify_checkpoint_assets,
)
from train_hy273_raw_flow import create_model


DEFAULT_CHECKPOINT = (
    "/mnt/afs/mogeflow-control/checkpoints/t2m/"
    "hy273_redenoise_kimodo_complete_stage2_control_ddp8_20260713_0547/"
    "model/step_00400000.pt"
)
PROTOCOL_VERSION = "hy273_hml3d_kimodo_constraints_v4"
PREFLIGHT_FORMAT = "hy273_kimodo_eval_preflight_v2"
GPU_INVENTORY_FORMAT = "hy273_gpu_inventory_v1"
STABLE_GPU_IDENTITY_FORMAT = "hy273_stable_gpu_identity_v1"
GPU_LAUNCH_ATTESTATION_FORMAT = "hy273_gpu_launch_attestation_v1"
GPU_INVENTORY_MAX_AGE_SECONDS = 5 * 60
DEFAULT_CAPTION_POLICY = "first_full_motion"
DEFAULT_EXPECTED_TEST_ITEMS = 4042
DEFAULT_EXPECTED_FULL_CASES = 8084


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def repair_jsonl(path: Path) -> None:
    """Drop a partial final record left by an interrupted append."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        content = handle.read()
        last_newline = content.rfind(b"\n")
        handle.truncate(last_newline + 1 if last_newline >= 0 else 0)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed JSONL at {path}:{line_number}: {exc}"
                ) from exc
    return records


class DurableJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair_jsonl(path)
        self.handle = path.open("a", encoding="utf-8")

    def append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.handle.write(line + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def constraint_to_device(
    constraint: CompiledKimodoConstraint,
    device: torch.device,
) -> CompiledKimodoConstraint:
    return CompiledKimodoConstraint(
        observed_motion=constraint.observed_motion.to(device),
        motion_mask=constraint.motion_mask.to(device),
        root_metric_frames=constraint.root_metric_frames.to(device),
        fullbody_metric_frames=constraint.fullbody_metric_frames.to(device),
        endpoint_position_metric_mask=constraint.endpoint_position_metric_mask.to(device),
        endpoint_rotation_metric_mask=constraint.endpoint_rotation_metric_mask.to(device),
        components=constraint.components,
    )


def plan_digest(keys: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def json_rows_digest(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def evaluation_code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    paths = [
        root / "eval_hy273_kimodo_full_test.py",
        root / "sample_hy273_raw.py",
        root / "train_hy273_raw_flow.py",
        root / "data" / "kimodo273_datasets.py",
        root / "models" / "codeflow" / "dit_blocks.py",
        *sorted((root / "models" / "raw_motion").glob("*.py")),
        root / "scripts" / "launch" / "eval_hy273_kimodo_full_test_ddp8.sh",
        root / "tools" / "validate_gpu_inventory.py",
    ]
    entries = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Evaluation code dependency is missing: {resolved}")
        entries.append(
            {
                "path": str(resolved.relative_to(root)),
                "size": int(resolved.stat().st_size),
                "sha256": sha256_file(resolved),
            }
        )
    return {
        "files": entries,
        "sha256": json_rows_digest(entries),
    }


def checkpoint_train_args(
    checkpoint: dict[str, Any], args: argparse.Namespace
) -> argparse.Namespace:
    train_args = argparse.Namespace(**checkpoint.get("args", {}))
    apply_checkpoint_path_override(train_args, "data_root", args.data_root)
    apply_checkpoint_path_override(train_args, "text_root", args.text_root)
    if args.hytext_cache_dir:
        apply_checkpoint_path_override(
            train_args, "hytext_cache_dir", args.hytext_cache_dir
        )
    return train_args


def make_eval_dataset(
    train_args: argparse.Namespace, args: argparse.Namespace
) -> Kimodo273TextDataset:
    return Kimodo273TextDataset(
        train_args.data_root,
        split=args.split,
        text_root=train_args.text_root or None,
        max_frames=int(train_args.max_frames),
        min_frames=args.min_frames,
        random_crop=False,
        exclude_fallback_short_clips=False,
        deterministic_text=True,
        caption_policy=args.caption_policy,
    )


def make_eval_plan(
    dataset: Kimodo273TextDataset, args: argparse.Namespace
) -> tuple[list, tuple[str, ...], tuple[str, ...]]:
    subtypes = parse_csv(args.subtypes)
    regimes = parse_csv(args.text_regimes)
    plan = build_kimodo_case_plan(
        len(dataset),
        seed=args.seed,
        subtypes=subtypes,
        text_regimes=regimes,
        assignment=args.assignment,
        cases_per_subtype=args.cases_per_subtype,
    )
    if args.expected_dataset_size > 0 and len(dataset) != args.expected_dataset_size:
        raise RuntimeError(
            "Evaluation dataset size mismatch: "
            f"expected={args.expected_dataset_size}, actual={len(dataset)}"
        )
    if args.expected_case_count > 0 and len(plan) != args.expected_case_count:
        raise RuntimeError(
            "Evaluation case count mismatch: "
            f"expected={args.expected_case_count}, actual={len(plan)}"
        )
    return plan, subtypes, regimes


def evaluation_dataset_identity(
    dataset: Kimodo273TextDataset,
    *,
    split: str,
    caption_policy: str,
    hash_contents: bool,
) -> dict[str, Any]:
    split_path = dataset.data_root / "split_existing" / f"{split}.txt"
    split_ids = [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_ids = [str(record["id"]) for record in dataset.records]
    if split_ids != record_ids:
        missing = [motion_id for motion_id in split_ids if motion_id not in set(record_ids)]
        unexpected = [motion_id for motion_id in record_ids if motion_id not in set(split_ids)]
        raise RuntimeError(
            "Ordered split does not exactly match evaluation records: "
            f"split={len(split_ids)}, records={len(record_ids)}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    ordered_rows = [
        {
            "dataset_index": index,
            "motion_id": str(record["id"]),
            "rel_path": str(record["rel"]),
            "frames": int(record["frames"]),
        }
        for index, record in enumerate(dataset.records)
    ]
    caption_rows = []
    for index, record in enumerate(dataset.records):
        caption = dataset.caption_metadata(index)
        if caption_policy == "first_full_motion" and not caption[
            "caption_is_full_motion"
        ]:
            raise RuntimeError(
                "No full-motion caption found for deterministic Kimodo evaluation: "
                f"motion_id={record['id']}"
            )
        caption_rows.append(
            {
                "dataset_index": index,
                "motion_id": str(record["id"]),
                "caption_index": int(caption["caption_index"]),
                "caption_line_number": int(caption["caption_line_number"]),
                "text": str(caption["text"]),
            }
        )

    stat_rows = []
    content_rows = []
    total_bytes = 0
    for index, record in enumerate(dataset.records):
        files = (
            ("motion", Path(record["path"])),
            ("text", dataset.text_root / f"{record['id']}.txt"),
        )
        for kind, path in files:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Pinned test {kind} asset is missing: {resolved}"
                )
            stat = resolved.stat()
            row = {
                "dataset_index": index,
                "motion_id": str(record["id"]),
                "kind": kind,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
            stat_rows.append(row)
            total_bytes += int(stat.st_size)
            if hash_contents:
                content_rows.append(
                    {
                        **{key: value for key, value in row.items() if key != "mtime_ns"},
                        "sha256": sha256_file(resolved),
                    }
                )

    assets = {
        "file_count": len(stat_rows),
        "total_bytes": total_bytes,
        "stat_sha256": json_rows_digest(stat_rows),
    }
    if hash_contents:
        assets["content_sha256"] = json_rows_digest(content_rows)
    return {
        "caption_policy": caption_policy,
        "caption_selection_sha256": json_rows_digest(caption_rows),
        "data_root": str(dataset.data_root),
        "dataset_size": len(dataset),
        "ordered_records_sha256": json_rows_digest(ordered_rows),
        "split": split,
        "split_path": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path),
        "text_root": str(dataset.text_root),
        "assets": assets,
    }


def build_preflight_manifest(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    stat_before = checkpoint_path.stat()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    stat_after = checkpoint_path.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError(f"Checkpoint changed while hashing: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    train_args = checkpoint_train_args(checkpoint, args)
    dataset = make_eval_dataset(train_args, args)
    plan, subtypes, regimes = make_eval_plan(dataset, args)
    expected_keys = [case.key for case in plan]
    dataset_identity = evaluation_dataset_identity(
        dataset,
        split=args.split,
        caption_policy=args.caption_policy,
        hash_contents=True,
    )
    payload = {
        "format": PREFLIGHT_FORMAT,
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint": {
            "path": str(checkpoint_path),
            "size": int(stat_after.st_size),
            "mtime_ns": int(stat_after.st_mtime_ns),
            "sha256": checkpoint_sha256,
            "metadata": checkpoint_metadata(checkpoint),
        },
        "code": evaluation_code_identity(),
        "dataset": dataset_identity,
        "plan": {
            "assignment": args.assignment,
            "case_count": len(plan),
            "cases_per_subtype": args.cases_per_subtype,
            "expected_case_keys_sha256": plan_digest(expected_keys),
            "seed": args.seed,
            "subtypes": list(subtypes),
            "text_regimes": list(regimes),
        },
    }
    del checkpoint
    return payload


def ensure_json_manifest(
    path: Path, payload: dict[str, Any], *, label: str
) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = None
        for _ in range(100):
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                break
            except (json.JSONDecodeError, OSError):
                # Another shard won O_EXCL but has not completed its fsynced write.
                time.sleep(0.05)
        if existing is None:
            raise RuntimeError(f"Timed out waiting for {label}: {path}")
        if existing != payload:
            raise RuntimeError(
                f"Output directory contains a different {label}: {path}"
            )
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def ensure_protocol_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    ensure_json_manifest(
        output_dir / "protocol_manifest.json",
        payload,
        label="benchmark protocol",
    )


def seed_sampling(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def validate_metric_finiteness(metrics: dict[str, float], context: str) -> None:
    nonfinite = {
        name: value
        for name, value in metrics.items()
        if not math.isfinite(float(value))
    }
    if nonfinite:
        raise RuntimeError(f"Non-finite metrics for {context}: {nonfinite}")


def record_provenance(
    protocol: dict[str, Any], protocol_manifest_sha256: str
) -> dict[str, Any]:
    preflight = protocol["preflight_manifest"]
    return {
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "evaluation_code_sha256": preflight["code_sha256"],
        "dataset_assets_sha256": preflight["dataset_assets_sha256"],
        "dataset_order_sha256": preflight["dataset_order_sha256"],
        "plan_sha256": protocol["plan_sha256"],
    }


def validate_success_record(
    record: dict[str, Any],
    *,
    case: Any,
    expected_shard_id: int,
    protocol: dict[str, Any],
    protocol_manifest_sha256: str,
) -> None:
    key = str(record.get("case_key", ""))
    if record.get("status") != "ok":
        raise RuntimeError(f"Cannot resume non-success benchmark record: {key}")

    expected_fields = {
        "protocol_version": protocol["protocol_version"],
        "weight_source": protocol["weight_source"],
        "case_key": case.key,
        "dataset_index": int(case.dataset_index),
        "subtype": case.subtype,
        "family": case.family,
        "text_regime": case.text_regime,
        "sample_seed": int(case.sample_seed),
        "shard_id": int(expected_shard_id),
        **record_provenance(protocol, protocol_manifest_sha256),
    }
    mismatches = {
        name: {"expected": expected, "actual": record.get(name)}
        for name, expected in expected_fields.items()
        if record.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Benchmark record provenance mismatch for {key}: {mismatches}"
        )
    validate_record_launch_provenance(record, protocol)

    if protocol.get("caption_policy") == "first_full_motion" and not bool(
        record.get("caption_is_full_motion", False)
    ):
        raise RuntimeError(f"Benchmark record does not use a full-motion caption: {key}")
    if case.text_regime == "withtext":
        if record.get("text") != record.get("source_caption"):
            raise RuntimeError(f"withtext benchmark record lost its source caption: {key}")
    elif record.get("text") != "":
        raise RuntimeError(f"notext benchmark record contains text: {key}")

    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"Benchmark record has no metric payload: {key}")
    for pass_name in ("generated_raw", "ground_truth", "diagnostic_exact_clamp"):
        pass_metrics = metrics.get(pass_name)
        if not isinstance(pass_metrics, dict) or not pass_metrics:
            raise RuntimeError(
                f"Benchmark record is missing metric pass {pass_name}: {key}"
            )
        validate_metric_finiteness(pass_metrics, f"{key}/{pass_name}")


def protocol_case_expectations(
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    plan = build_kimodo_case_plan(
        int(protocol["dataset_size"]),
        seed=int(protocol["seed"]),
        subtypes=tuple(protocol["subtypes"]),
        text_regimes=tuple(protocol["text_regimes"]),
        assignment=str(protocol["assignment"]),
        cases_per_subtype=int(protocol["cases_per_subtype"]),
    )
    expected_keys = [case.key for case in plan]
    if expected_keys != list(protocol["expected_case_keys"]):
        raise RuntimeError("Protocol case list cannot be reconstructed exactly")
    if plan_digest(expected_keys) != protocol["plan_sha256"]:
        raise RuntimeError("Protocol case plan SHA256 mismatch")

    cases_by_key = {case.key: case for case in plan}
    shard_by_key: dict[str, int] = {}
    num_shards = int(protocol["num_shards"])
    for shard_id in range(num_shards):
        for case in shard_kimodo_case_plan(
            plan, shard_id=shard_id, num_shards=num_shards
        ):
            if case.key in shard_by_key:
                raise RuntimeError(f"Case assigned to multiple shards: {case.key}")
            shard_by_key[case.key] = shard_id
    if set(shard_by_key) != set(cases_by_key):
        raise RuntimeError("Protocol shard assignment does not cover the case plan")
    return cases_by_key, shard_by_key


def checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(checkpoint.get("step", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "has_ema": "ema" in checkpoint,
        "has_normalizer": "normalizer" in checkpoint,
    }


def stable_gpu_inventory_identity(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("format") != GPU_INVENTORY_FORMAT or not payload.get("passed"):
        raise RuntimeError("Invalid GPU inventory payload")
    devices = payload.get("devices")
    if not isinstance(devices, list) or not devices:
        raise RuntimeError("GPU inventory has no devices")

    canonical_devices = []
    for device in devices:
        canonical_devices.append(
            {
                "uuid": str(device.get("uuid", "")),
                "pci_bus_id": str(device.get("pci_bus_id", "")).lower(),
                "name": str(device.get("name", "")),
                "memory_total_mib": int(device.get("memory_total_mib", -1)),
            }
        )
    uuids = [device["uuid"] for device in canonical_devices]
    buses = [device["pci_bus_id"] for device in canonical_devices]
    signatures = {
        (device["name"], device["memory_total_mib"])
        for device in canonical_devices
    }
    if (
        not all(uuids)
        or not all(buses)
        or len(set(uuids)) != len(uuids)
        or len(set(buses)) != len(buses)
        or len(signatures) != 1
    ):
        raise RuntimeError("GPU inventory does not identify distinct homogeneous devices")
    if not str(payload.get("host", "")):
        raise RuntimeError("GPU inventory has no host identity")
    if int(payload.get("physical_device_count", -1)) != len(canonical_devices):
        raise RuntimeError("GPU inventory physical-device count is inconsistent")
    name, memory_total_mib = next(iter(signatures))
    if payload.get("homogeneous_signature") != {
        "name": name,
        "memory_total_mib": memory_total_mib,
    }:
        raise RuntimeError("GPU inventory homogeneous signature is inconsistent")

    return {
        "format": STABLE_GPU_IDENTITY_FORMAT,
        "host": str(payload.get("host", "")),
        "physical_device_count": len(canonical_devices),
        "devices": sorted(canonical_devices, key=lambda row: row["uuid"]),
    }


def validate_gpu_inventory_idleness(payload: dict[str, Any]) -> None:
    thresholds = payload.get("idle_thresholds")
    devices = payload.get("devices")
    if not isinstance(thresholds, dict) or not isinstance(devices, list):
        raise RuntimeError("GPU inventory is missing idleness evidence")
    max_memory = int(thresholds.get("max_memory_used_mib", -1))
    max_utilization = int(thresholds.get("max_utilization_percent", -1))
    if max_memory < 0 or max_utilization < 0:
        raise RuntimeError("GPU inventory has invalid idleness thresholds")
    if thresholds.get("require_no_compute_pids") is not True:
        raise RuntimeError("GPU inventory does not require an empty compute PID list")
    busy = [
        device
        for device in devices
        if int(device.get("memory_used_mib", max_memory + 1)) > max_memory
        or int(device.get("utilization_percent", max_utilization + 1))
        > max_utilization
        or bool(device.get("compute_pids"))
    ]
    if busy:
        raise RuntimeError(f"GPU launch attestation contains busy devices: {busy}")


def load_gpu_inventory_identity(
    path: Path,
    *,
    expected_num_shards: int,
    require_fresh: bool,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GPU inventory manifest not found: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("phase") != "before_launch":
        raise RuntimeError(
            f"GPU inventory was not captured immediately before launch: {path}"
        )
    if payload.get("host") != socket.gethostname():
        raise RuntimeError(
            f"GPU inventory host mismatch: {payload.get('host')} != {socket.gethostname()}"
        )
    stable = stable_gpu_inventory_identity(payload)
    if stable["physical_device_count"] != int(expected_num_shards):
        raise RuntimeError(
            "GPU inventory count does not match evaluation shards: "
            f"{stable['physical_device_count']} != {expected_num_shards}"
        )
    validate_gpu_inventory_idleness(payload)
    checked_unix = float(payload.get("checked_unix", 0.0))
    if require_fresh:
        age_seconds = time.time() - checked_unix
        if age_seconds < -5.0 or age_seconds > GPU_INVENTORY_MAX_AGE_SECONDS:
            raise RuntimeError(
                f"GPU inventory is not fresh enough for launch: age={age_seconds:.1f}s"
            )
    return {
        "stable": stable,
        "attestation": {
            "format": GPU_LAUNCH_ATTESTATION_FORMAT,
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "checked_unix": checked_unix,
        },
    }


def gpu_inventory_identity(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gpu_inventory_manifest:
        raise RuntimeError("CUDA benchmark shards require --gpu_inventory_manifest")
    return load_gpu_inventory_identity(
        Path(args.gpu_inventory_manifest),
        expected_num_shards=int(args.num_shards),
        require_fresh=True,
    )


def validate_stable_gpu_inventory(protocol: dict[str, Any]) -> None:
    identity = protocol.get("gpu_inventory")
    if not isinstance(identity, dict):
        raise RuntimeError("Benchmark protocol has no stable GPU identity")
    if identity.get("format") != STABLE_GPU_IDENTITY_FORMAT:
        raise RuntimeError("Benchmark protocol has an invalid stable GPU identity")
    devices = identity.get("devices")
    expected_count = int(protocol.get("num_shards", -1))
    if (
        not str(identity.get("host", ""))
        or int(identity.get("physical_device_count", -1)) != expected_count
        or not isinstance(devices, list)
        or len(devices) != expected_count
    ):
        raise RuntimeError("Stable GPU identity does not match benchmark shards")
    uuids = [str(device.get("uuid", "")) for device in devices]
    buses = [str(device.get("pci_bus_id", "")).lower() for device in devices]
    signatures = {
        (
            str(device.get("name", "")),
            int(device.get("memory_total_mib", -1)),
        )
        for device in devices
    }
    if (
        not all(uuids)
        or not all(buses)
        or len(set(uuids)) != len(devices)
        or len(set(buses)) != len(devices)
        or len(signatures) != 1
        or devices != sorted(devices, key=lambda row: str(row.get("uuid", "")))
    ):
        raise RuntimeError("Stable GPU identity contains duplicate or incomplete devices")


def record_launch_provenance(identity: dict[str, Any]) -> dict[str, Any]:
    attestation = identity.get("attestation")
    if not isinstance(attestation, dict):
        raise RuntimeError("Current launch has no GPU attestation")
    return {"gpu_launch_attestation": dict(attestation)}


def validate_record_launch_provenance(
    record: dict[str, Any], protocol: dict[str, Any]
) -> None:
    attestation = record.get("gpu_launch_attestation")
    if not isinstance(attestation, dict):
        raise RuntimeError(
            f"Benchmark record has no GPU launch attestation: {record.get('case_key', '')}"
        )
    if attestation.get("format") != GPU_LAUNCH_ATTESTATION_FORMAT:
        raise RuntimeError("Benchmark record has an invalid GPU launch attestation")
    loaded = load_gpu_inventory_identity(
        Path(str(attestation.get("path", ""))),
        expected_num_shards=int(protocol["num_shards"]),
        require_fresh=False,
    )
    if loaded["attestation"] != attestation:
        raise RuntimeError(
            "Benchmark record GPU launch attestation changed: "
            f"{record.get('case_key', '')}"
        )
    if loaded["stable"] != protocol.get("gpu_inventory"):
        raise RuntimeError(
            "Benchmark record was generated on GPUs outside the pinned identity: "
            f"{record.get('case_key', '')}"
        )


def asset_attestation_path(
    train_args: argparse.Namespace,
    requested_path: str = "",
) -> Path:
    if requested_path:
        return Path(requested_path).expanduser().resolve()
    manifest_hash = str(getattr(train_args, "asset_manifest_sha256", ""))
    suffix = manifest_hash[:24] if manifest_hash else "unversioned"
    return Path(f"/dev/shm/hy273_asset_verification_{suffix}.json")


def asset_attestation_payload(train_args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(
        str(getattr(train_args, "asset_manifest_path", ""))
    ).expanduser().resolve()
    expected_hash = str(getattr(train_args, "asset_manifest_sha256", "")).lower()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Asset manifest not found: {manifest_path}")
    actual_hash = sha256_file(manifest_path)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "Asset manifest hash changed before attestation: "
            f"expected={expected_hash}, actual={actual_hash}"
        )
    stat = manifest_path.stat()
    return {
        "format": "hy273_asset_verification_attestation_v1",
        "status": "ok",
        "host": socket.gethostname(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_hash,
        "manifest_size": int(stat.st_size),
        "manifest_mtime_ns": int(stat.st_mtime_ns),
        "verified_unix": time.time(),
        "verifier_pid": os.getpid(),
    }


def valid_asset_attestation(
    train_args: argparse.Namespace,
    path: Path,
    max_age_seconds: float,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if time.time() - float(payload.get("verified_unix", 0.0)) > max_age_seconds:
        return False
    if payload.get("status") == "error":
        raise RuntimeError(
            f"Coordinated asset verification failed: {payload.get('error')}"
        )
    manifest_path = Path(
        str(getattr(train_args, "asset_manifest_path", ""))
    ).expanduser().resolve()
    expected_hash = str(getattr(train_args, "asset_manifest_sha256", "")).lower()
    if payload.get("format") != "hy273_asset_verification_attestation_v1":
        return False
    if payload.get("host") != socket.gethostname():
        return False
    if payload.get("manifest_path") != str(manifest_path):
        return False
    if expected_hash and payload.get("manifest_sha256") != expected_hash:
        return False
    try:
        stat = manifest_path.stat()
    except OSError:
        return False
    if int(payload.get("manifest_size", -1)) != int(stat.st_size):
        return False
    if int(payload.get("manifest_mtime_ns", -1)) != int(stat.st_mtime_ns):
        return False
    # The small manifest itself is re-hashed by every worker. Its pinned hashes
    # cover the 42k assets fully checked by the attesting process.
    return sha256_file(manifest_path) == payload.get("manifest_sha256")


def verify_checkpoint_assets_coordinated(
    train_args: argparse.Namespace,
    *,
    requested_cache_path: str = "",
    max_age_seconds: float = 24 * 60 * 60,
    wait_timeout_seconds: float = 60 * 60,
) -> tuple[Path, bool]:
    """Fully verify once, then share a short-lived host-local attestation."""
    cache_path = asset_attestation_path(train_args, requested_cache_path)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if valid_asset_attestation(train_args, cache_path, max_age_seconds):
        return cache_path, True

    start = time.monotonic()
    while True:
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            if valid_asset_attestation(train_args, cache_path, max_age_seconds):
                return cache_path, True
            if time.monotonic() - start > wait_timeout_seconds:
                raise RuntimeError(
                    f"Timed out waiting for coordinated asset verification: {lock_path}"
                )
            try:
                lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
                owner_pid = int(lock_payload.get("pid", -1))
                owner_alive = owner_pid > 0 and Path(f"/proc/{owner_pid}").exists()
                if not owner_alive and time.time() - lock_path.stat().st_mtime > 30.0:
                    lock_path.unlink(missing_ok=True)
                    continue
            except (json.JSONDecodeError, OSError, ValueError):
                pass
            time.sleep(1.0)
            continue

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"pid": os.getpid(), "host": socket.gethostname(), "started": time.time()}
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        try:
            verify_checkpoint_assets(train_args)
            payload = asset_attestation_payload(train_args)
            atomic_write_json(cache_path, payload)
            return cache_path, False
        except Exception as exc:
            atomic_write_json(
                cache_path,
                {
                    "format": "hy273_asset_verification_attestation_v1",
                    "status": "error",
                    "host": socket.gethostname(),
                    "error": repr(exc),
                    "verified_unix": time.time(),
                },
            )
            raise
        finally:
            lock_path.unlink(missing_ok=True)


def load_and_validate_preflight(
    args: argparse.Namespace,
    *,
    checkpoint: dict[str, Any],
    dataset: Kimodo273TextDataset,
    plan: list,
) -> tuple[Path, dict[str, Any]]:
    path = (
        Path(args.preflight_manifest).expanduser().resolve()
        if args.preflight_manifest
        else Path(args.output_dir).expanduser().resolve() / "preflight_manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing evaluation preflight manifest: {path}. Run --preflight_only first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != PREFLIGHT_FORMAT:
        raise RuntimeError(f"Unsupported preflight manifest format: {payload.get('format')!r}")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "Preflight protocol version mismatch: "
            f"expected={PROTOCOL_VERSION}, actual={payload.get('protocol_version')}"
        )

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_stat = checkpoint_path.stat()
    expected_checkpoint = payload["checkpoint"]
    observed_checkpoint = {
        "path": str(checkpoint_path),
        "size": int(checkpoint_stat.st_size),
        "mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "metadata": checkpoint_metadata(checkpoint),
    }
    for key, value in observed_checkpoint.items():
        if expected_checkpoint.get(key) != value:
            raise RuntimeError(
                f"Checkpoint changed after preflight for {key}: "
                f"expected={expected_checkpoint.get(key)!r}, actual={value!r}"
            )

    current_code = evaluation_code_identity()
    if current_code != payload.get("code"):
        raise RuntimeError("Evaluation code changed after preflight")

    current_dataset = evaluation_dataset_identity(
        dataset,
        split=args.split,
        caption_policy=args.caption_policy,
        hash_contents=False,
    )
    expected_dataset = payload["dataset"]
    for key in (
        "caption_policy",
        "caption_selection_sha256",
        "data_root",
        "dataset_size",
        "ordered_records_sha256",
        "split",
        "split_path",
        "split_sha256",
        "text_root",
    ):
        if current_dataset.get(key) != expected_dataset.get(key):
            raise RuntimeError(f"Test dataset changed after preflight for {key}")
    for key in ("file_count", "total_bytes", "stat_sha256"):
        if current_dataset["assets"].get(key) != expected_dataset["assets"].get(key):
            raise RuntimeError(f"Test assets changed after preflight for {key}")
    if not expected_dataset["assets"].get("content_sha256"):
        raise RuntimeError("Preflight manifest is missing test asset content hashes")

    expected_plan = payload["plan"]
    observed_plan = {
        "assignment": args.assignment,
        "case_count": len(plan),
        "cases_per_subtype": args.cases_per_subtype,
        "expected_case_keys_sha256": plan_digest(case.key for case in plan),
        "seed": args.seed,
        "subtypes": list(parse_csv(args.subtypes)),
        "text_regimes": list(parse_csv(args.text_regimes)),
    }
    if observed_plan != expected_plan:
        raise RuntimeError("Evaluation case plan changed after preflight")
    return path, payload


def run_shard(args: argparse.Namespace) -> None:
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"Expected 0 <= shard_id < num_shards, got {args.shard_id}/{args.num_shards}"
        )
    if args.batch_size != 1:
        raise ValueError(
            "Kimodo exact per-case reproducibility requires --batch_size 1"
        )
    if args.num_steps < 1:
        raise ValueError("num_steps must be positive")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    train_args = checkpoint_train_args(checkpoint, args)
    metadata = checkpoint_metadata(checkpoint)

    dataset = make_eval_dataset(train_args, args)
    plan, subtypes, regimes = make_eval_plan(dataset, args)
    preflight_path, preflight = load_and_validate_preflight(
        args,
        checkpoint=checkpoint,
        dataset=dataset,
        plan=plan,
    )
    state_dict, resolved_weight_source = checkpoint_weight_state(
        checkpoint, args.weight_source, args.checkpoint
    )
    launch_gpu_identity = gpu_inventory_identity(args)
    expected_keys = [case.key for case in plan]
    counts = Counter((case.text_regime, case.subtype) for case in plan)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "adaptation": (
            "HumanML3D has no Kimodo content/repetition benchmark metadata; "
            "held-out clips are deterministically assigned to public Kimodo leaf types."
        ),
        "assignment": args.assignment,
        "asset_verification": {
            "cache": str(
                asset_attestation_path(train_args, args.asset_verification_cache)
            ),
            "max_age_seconds": args.asset_verification_max_age,
            "policy": "one full SHA256 pass, shared host-local attestation",
        },
        "batch_size": args.batch_size,
        "cases_per_subtype": args.cases_per_subtype,
        "cfg_scale_text": args.cfg_scale,
        "cfg_scale_control": args.control_cfg_scale,
        "caption_policy": args.caption_policy,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": preflight["checkpoint"]["sha256"],
        "checkpoint_size": preflight["checkpoint"]["size"],
        "checkpoint_metadata": metadata,
        "contact_feedback": args.contact_feedback,
        "contact_init": args.contact_init,
        "data_root": str(Path(train_args.data_root).resolve()),
        "dataset_size": len(dataset),
        "decode_for_metrics": "rotation_fk",
        "exact_clamp_role": "diagnostic_only",
        "expected_case_count": len(plan),
        "expected_case_keys": expected_keys,
        "expected_counts": {
            f"{regime}/{subtype}": count
            for (regime, subtype), count in sorted(counts.items())
        },
        "gpu_inventory": launch_gpu_identity["stable"],
        "gpu_inventory_policy": {
            "fresh_attestation_per_launch": True,
            "launch_attestation_format": GPU_LAUNCH_ATTESTATION_FORMAT,
            "max_age_seconds_at_worker_start": GPU_INVENTORY_MAX_AGE_SECONDS,
            "record_binds_attestation": True,
        },
        "max_frames": int(train_args.max_frames),
        "min_frames": args.min_frames,
        "max_sparse_keyframes": args.max_sparse_keyframes,
        "num_shards": args.num_shards,
        "num_steps": args.num_steps,
        "plan_sha256": plan_digest(expected_keys),
        "postprocess": False,
        "preflight_manifest": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "code_sha256": preflight["code"]["sha256"],
            "dataset_assets_sha256": preflight["dataset"]["assets"][
                "content_sha256"
            ],
            "dataset_order_sha256": preflight["dataset"][
                "ordered_records_sha256"
            ],
            "split_sha256": preflight["dataset"]["split_sha256"],
        },
        "primary_output": "raw_pre_exact_clamp",
        "seed": args.seed,
        "split": args.split,
        "subtypes": list(subtypes),
        "text_regimes": list(regimes),
        "requested_weight_source": args.weight_source,
        "weight_source": resolved_weight_source,
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_protocol_manifest(output_dir, protocol)
    protocol_path = output_dir / "protocol_manifest.json"
    protocol_manifest_sha256 = sha256_file(protocol_path)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    attestation_path, reused_attestation = verify_checkpoint_assets_coordinated(
        train_args,
        requested_cache_path=args.asset_verification_cache,
        max_age_seconds=args.asset_verification_max_age,
    )
    print(
        json.dumps(
            {
                "asset_attestation": str(attestation_path),
                "asset_attestation_reused": reused_attestation,
            }
        ),
        flush=True,
    )
    model = create_model(train_args).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    normalizer = checkpoint_normalizer(
        checkpoint, train_args, device, args.checkpoint
    )
    prediction_type = str(getattr(train_args, "prediction_type", "x0"))
    if prediction_type != "x0":
        raise RuntimeError(
            f"Final Kimodo-like checkpoint must use x0 prediction, got {prediction_type}"
        )
    del state_dict
    del checkpoint

    shard_path = output_dir / "shards" / f"shard_{args.shard_id:02d}.jsonl"
    progress_path = output_dir / "progress" / f"shard_{args.shard_id:02d}.json"
    if args.overwrite and shard_path.exists():
        shard_path.unlink()
    shard_cases = shard_kimodo_case_plan(
        plan,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    shard_case_by_key = {case.key: case for case in shard_cases}
    existing_records = deduplicate_records(load_jsonl(shard_path))
    for record in existing_records:
        key = str(record.get("case_key", ""))
        if key not in shard_case_by_key:
            raise RuntimeError(
                f"Shard {args.shard_id} contains an unexpected case record: {key}"
            )
        validate_success_record(
            record,
            case=shard_case_by_key[key],
            expected_shard_id=args.shard_id,
            protocol=protocol,
            protocol_manifest_sha256=protocol_manifest_sha256,
        )
    completed = {str(record["case_key"]) for record in existing_records}
    writer = DurableJsonlWriter(shard_path)
    shard_start = time.perf_counter()
    newly_completed = 0
    try:
        for position, case in enumerate(shard_cases, start=1):
            if case.key in completed:
                continue
            case_start = time.perf_counter()
            item = dataset[case.dataset_index]
            source_motion = item["motion"].float()
            transform = apply_kimodo_training_transform(
                source_motion.unsqueeze(0),
                random_heading=False,
                root_shift=True,
            )
            target_cpu = transform.motion[0].contiguous()
            constraint_cpu = compile_kimodo_constraint(
                target_cpu,
                case.subtype,
                seed=case.sample_seed,
                max_sparse_keyframes=args.max_sparse_keyframes,
            )
            target = target_cpu.unsqueeze(0).to(device)
            constraint = constraint_to_device(constraint_cpu, device)
            length = int(target.shape[1])
            lengths = torch.tensor([length], dtype=torch.long, device=device)
            text = str(item["text"]) if case.text_regime == "withtext" else ""
            seed_sampling(case.sample_seed, device)
            sampled = sample_ode(
                model,
                normalizer,
                lengths,
                [text],
                constraint.observed_motion.unsqueeze(0),
                constraint.motion_mask.unsqueeze(0),
                transform.c_dir.to(device),
                num_steps=args.num_steps,
                self_conditioning=bool(
                    getattr(train_args, "self_conditioning", False)
                ),
                cfg_scale=args.cfg_scale,
                control_cfg_scale=args.control_cfg_scale,
                contact_init=args.contact_init,
                contact_feedback=args.contact_feedback,
                cfg_apply_contacts=True,
                prediction_type=prediction_type,
                velocity_t_eps=1e-4,
                return_details=True,
            )
            if not isinstance(sampled, ODESampleOutput):
                raise AssertionError("Expected detailed ODE sampling output")
            raw_metrics = evaluate_kimodo_constraint_case(
                sampled.raw_motion[0], target[0], constraint
            )
            exact_metrics = evaluate_kimodo_constraint_case(
                sampled.exact_clamped_motion[0], target[0], constraint
            )
            gt_metrics = evaluate_kimodo_constraint_case(
                target[0], target[0], constraint
            )
            validate_metric_finiteness(raw_metrics, f"{case.key}/raw")
            validate_metric_finiteness(exact_metrics, f"{case.key}/exact")
            validate_metric_finiteness(gt_metrics, f"{case.key}/gt")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - case_start
            record = {
                "status": "ok",
                "protocol_version": PROTOCOL_VERSION,
                "case_key": case.key,
                "dataset_index": case.dataset_index,
                "motion_id": item["motion_id"],
                "rel_path": item["rel_path"],
                "crop_start": int(item.get("crop_start", 0)),
                "caption_index": int(item.get("caption_index", 0)),
                "caption_line_number": int(item.get("caption_line_number", 0)),
                "caption_from_tag": item.get("caption_from_tag"),
                "caption_to_tag": item.get("caption_to_tag"),
                "caption_is_full_motion": bool(
                    item.get("caption_is_full_motion", False)
                ),
                "length": length,
                "subtype": case.subtype,
                "family": case.family,
                "text_regime": case.text_regime,
                "sample_seed": case.sample_seed,
                "text": text,
                "source_caption": str(item["text"]),
                "constraint": {
                    "components": constraint_cpu.components,
                    "model_mask_fraction": float(
                        constraint_cpu.motion_mask.float().mean().item()
                    ),
                    "root_metric_frames": int(
                        constraint_cpu.root_metric_frames.sum().item()
                    ),
                    "fullbody_metric_frames": int(
                        constraint_cpu.fullbody_metric_frames.sum().item()
                    ),
                    "endpoint_position_targets": int(
                        constraint_cpu.endpoint_position_metric_mask.sum().item()
                    ),
                    "endpoint_rotation_targets": int(
                        constraint_cpu.endpoint_rotation_metric_mask.sum().item()
                    ),
                },
                "metrics": {
                    "generated_raw": raw_metrics,
                    "ground_truth": gt_metrics,
                    "diagnostic_exact_clamp": exact_metrics,
                },
                "elapsed_seconds": elapsed,
                "shard_id": args.shard_id,
                "weight_source": resolved_weight_source,
                **record_provenance(protocol, protocol_manifest_sha256),
                **record_launch_provenance(launch_gpu_identity),
            }
            writer.append(record)
            completed.add(case.key)
            newly_completed += 1
            total_done = sum(case_item.key in completed for case_item in shard_cases)
            total_elapsed = time.perf_counter() - shard_start
            rate = newly_completed / total_elapsed if total_elapsed > 0 else 0.0
            remaining = len(shard_cases) - total_done
            progress = {
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "completed": total_done,
                "total": len(shard_cases),
                "newly_completed": newly_completed,
                "remaining": remaining,
                "cases_per_second": rate,
                "eta_seconds": remaining / rate if rate > 0 else None,
                "last_case_key": case.key,
                "last_case_seconds": elapsed,
                "updated_unix": time.time(),
                "done": remaining == 0,
            }
            atomic_write_json(progress_path, progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
    finally:
        writer.close()

    final_records = deduplicate_records(load_jsonl(shard_path))
    for record in final_records:
        key = str(record.get("case_key", ""))
        if key not in shard_case_by_key:
            raise RuntimeError(
                f"Shard {args.shard_id} contains an unexpected final record: {key}"
            )
        validate_success_record(
            record,
            case=shard_case_by_key[key],
            expected_shard_id=args.shard_id,
            protocol=protocol,
            protocol_manifest_sha256=protocol_manifest_sha256,
        )
    atomic_write_json(
        output_dir / "shards" / f"shard_{args.shard_id:02d}_summary.json",
        {
            "shard_id": args.shard_id,
            "expected": len(shard_cases),
            "records": len(final_records),
            "protocol_manifest_sha256": protocol_manifest_sha256,
            "unique_success": len(
                {
                    record["case_key"]
                    for record in final_records
                    if record.get("status") == "ok"
                }
            ),
            "complete": all(case.key in completed for case in shard_cases),
        },
    )


def deduplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["case_key"])
        if key in by_key and by_key[key] != record:
            raise RuntimeError(f"Conflicting duplicate benchmark record: {key}")
        by_key[key] = record
    return [by_key[key] for key in sorted(by_key)]


def metric_cell(metrics: dict[str, Any], name: str, scale: float = 1.0) -> str:
    value = metrics.get(name)
    return "-" if value is None else f"{float(value) * scale:.3f}"


def metric_denominator_cell(metrics: dict[str, Any]) -> str:
    names = (
        "constraint_fullbody_keyframe",
        "constraint_end_effector",
        "constraint_root2d_err",
    )
    return "/".join(str(int(metrics.get(f"{name}__count", 0))) for name in names)


def render_markdown(summary: dict[str, Any], protocol: dict[str, Any]) -> str:
    lines = [
        "# HY273 HumanML3D Kimodo 控制全量评估",
        "",
        f"- 协议：`{protocol['protocol_version']}`",
        f"- 检查点：`{protocol['checkpoint']}`",
        f"- 检查点 SHA256：`{protocol['checkpoint_sha256']}`",
        f"- test 动作数：{protocol['dataset_size']}",
        f"- 生成 case：{summary['num_success']} / {protocol['expected_case_count']}",
        f"- 采样：EMA, ODE{protocol['num_steps']}, text CFG={protocol['cfg_scale_text']}, control CFG={protocol['cfg_scale_control']}",
        f"- 文本策略：`{protocol['caption_policy']}`（确定性优先完整动作 caption）",
        "- 主结果：raw pre-exact-clamp；所有动作质量指标使用 global rotation FK 解码。",
        "- 本次只评估 constrained-generation；不加载与 SMPLX22 不兼容的 SOMA TMR。",
        "- 说明：HumanML3D 没有 Kimodo content/repetition metadata，本结果不能与 SOMA/BONES-SEED 公布数值直接横比。",
        "",
        "## Raw Generated",
        "",
        "| Text | Level | Group | N | N(FB/EE/Root) | FB cm | EE cm | EE rot deg | Root cm | Root P95 cm | Root acc | Skate(pred) cm/s | Skate(height) cm/s | Skate ratio | Contact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"all": 0, "family": 1, "subtype": 2}
    rows = sorted(
        summary["rows"],
        key=lambda row: (
            row["text_regime"],
            order[row["level"]],
            row["name"],
        ),
    )
    for row in rows:
        metrics = row["generated_raw"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["text_regime"],
                    row["level"],
                    row["name"],
                    str(row["num_cases"]),
                    metric_denominator_cell(metrics),
                    metric_cell(metrics, "constraint_fullbody_keyframe", 100.0),
                    metric_cell(metrics, "constraint_end_effector", 100.0),
                    metric_cell(metrics, "constraint_end_effector_rotation_deg"),
                    metric_cell(metrics, "constraint_root2d_err", 100.0),
                    metric_cell(metrics, "constraint_root2d_err_p95", 100.0),
                    metric_cell(metrics, "constraint_root2d_acc"),
                    metric_cell(metrics, "foot_skate_from_pred_contacts", 100.0),
                    metric_cell(metrics, "foot_skate_from_height", 100.0),
                    metric_cell(metrics, "foot_skate_ratio"),
                    metric_cell(metrics, "foot_contact_consistency"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Ground Truth",
            "",
            "GT 使用同一批动作、同一约束和同一套 Kimodo 指标，是动作质量与表示误差基线。详细数值见 `summary.json`。",
            "",
            "## Exact Clamp Diagnostic",
            "",
            "`diagnostic_exact_clamp` 只用于判断最终硬覆盖的数值效果，不计作模型 learned control 成绩。详细数值见 `summary.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_output(output_dir: Path, allow_incomplete: bool) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    protocol_path = output_dir / "protocol_manifest.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing protocol manifest: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            "Protocol version mismatch during aggregation: "
            f"expected={PROTOCOL_VERSION}, actual={protocol.get('protocol_version')}"
        )
    validate_stable_gpu_inventory(protocol)
    protocol_manifest_sha256 = sha256_file(protocol_path)
    cases_by_key, shard_by_key = protocol_case_expectations(protocol)
    records: list[dict[str, Any]] = []
    for shard_path in sorted((output_dir / "shards").glob("shard_*.jsonl")):
        records.extend(load_jsonl(shard_path))
    records = deduplicate_records(records)
    expected = set(protocol["expected_case_keys"])
    record_keys = {str(record.get("case_key", "")) for record in records}
    unexpected = sorted(record_keys - expected)
    if unexpected:
        raise RuntimeError(
            f"Found {len(unexpected)} unexpected case keys: {unexpected[:3]}"
        )
    for record in records:
        if record.get("status") != "ok":
            continue
        key = str(record["case_key"])
        validate_success_record(
            record,
            case=cases_by_key[key],
            expected_shard_id=shard_by_key[key],
            protocol=protocol,
            protocol_manifest_sha256=protocol_manifest_sha256,
        )
    actual = {
        record["case_key"]
        for record in records
        if record.get("status") == "ok"
    }
    missing = sorted(expected - actual)
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Benchmark is incomplete: {len(actual)}/{len(expected)} cases; "
            f"first missing={missing[:3]}"
        )
    summary = aggregate_case_records(records)
    summary.update(
        {
            "protocol_version": protocol["protocol_version"],
            "expected_case_count": len(expected),
            "actual_case_count": len(actual),
            "complete": not missing,
            "missing_case_count": len(missing),
            "missing_case_examples": missing[:20],
            "plan_sha256": protocol["plan_sha256"],
            "protocol_manifest_sha256": protocol_manifest_sha256,
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_text(
        output_dir / "REPORT.md",
        render_markdown(summary, protocol),
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "actual": len(actual),
                "expected": len(expected),
                "complete": not missing,
                "summary": str(output_dir / "summary.json"),
                "report": str(output_dir / "REPORT.md"),
            },
            indent=2,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--text_root", default="")
    parser.add_argument("--hytext_cache_dir", default="")
    parser.add_argument("--preflight_manifest", default="")
    parser.add_argument("--asset_verification_cache", default="")
    parser.add_argument("--gpu_inventory_manifest", default="")
    parser.add_argument(
        "--asset_verification_max_age",
        type=float,
        default=24 * 60 * 60,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--caption_policy",
        choices=["first", "first_full_motion"],
        default=DEFAULT_CAPTION_POLICY,
    )
    parser.add_argument(
        "--expected_dataset_size",
        type=int,
        default=DEFAULT_EXPECTED_TEST_ITEMS,
    )
    parser.add_argument(
        "--expected_case_count",
        type=int,
        default=DEFAULT_EXPECTED_FULL_CASES,
    )
    parser.add_argument("--subtypes", default=",".join(KIMODO_CONTROL_SUBTYPES))
    parser.add_argument("--text_regimes", default=",".join(TEXT_REGIMES))
    parser.add_argument(
        "--assignment",
        choices=["balanced_partition", "cartesian"],
        default="balanced_partition",
    )
    parser.add_argument("--cases_per_subtype", type=int, default=0)
    parser.add_argument(
        "--min_frames",
        type=int,
        default=2,
        help="Full-test evaluation includes short clips; metrics require at least two frames.",
    )
    parser.add_argument("--max_sparse_keyframes", type=int, default=20)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--control_cfg_scale", type=float, default=2.0)
    parser.add_argument(
        "--contact_init", choices=["random", "zeros", "half"], default="random"
    )
    parser.add_argument(
        "--contact_feedback", choices=["blend", "prob", "fixed"], default="blend"
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--weight_source", choices=["ema", "model", "auto"], default="ema"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.preflight_only:
        output_dir = Path(args.output_dir).expanduser().resolve()
        path = output_dir / "preflight_manifest.json"
        payload = build_preflight_manifest(args)
        ensure_json_manifest(path, payload, label="evaluation preflight")
        print(
            json.dumps(
                {
                    "preflight_manifest": str(path),
                    "checkpoint_sha256": payload["checkpoint"]["sha256"],
                    "dataset_size": payload["dataset"]["dataset_size"],
                    "case_count": payload["plan"]["case_count"],
                    "test_assets_sha256": payload["dataset"]["assets"][
                        "content_sha256"
                    ],
                    "passed": True,
                },
                indent=2,
            )
        )
    elif args.aggregate_only:
        aggregate_output(Path(args.output_dir), args.allow_incomplete)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
