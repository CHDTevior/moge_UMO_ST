"""Regenerate representative Kimodo-control cases from the formal test records.

The formal evaluator intentionally stores metrics rather than generated arrays.
This tool selects transparent 25/50/75-percentile examples per public subtype,
then replays each case with its recorded dataset index, caption, and sample seed.
It is separate from the frozen formal-evaluation identity.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.kimodo273_datasets import Kimodo273TextDataset
from eval_hy273_kimodo_full_test import constraint_to_device, seed_sampling
from models.raw_motion.hy273_kimodo_benchmark import (
    KIMODO_CONTROL_SUBTYPES,
    compile_kimodo_constraint,
    evaluate_kimodo_constraint_case,
)
from models.raw_motion.hy273_normalizer import apply_kimodo_training_transform
from sample_hy273_raw import (
    ODESampleOutput,
    checkpoint_normalizer,
    checkpoint_weight_state,
    sample_ode,
)
from train_hy273_raw_flow import create_model


EXPECTED_CHECKPOINT_SHA256 = (
    "d5f00ec15888e1dc3ca9f8c38c8ef436ec6524397ae0257a01fc48ce3542b2f4"
)
EXPECTED_EVALUATION_CODE_SHA256 = (
    "81dac4705be3072ac3b4cedab72a53b95ff45a120ea258c3af20246a39d35ff3"
)
EXPECTED_PROTOCOL_VERSION = "hy273_hml3d_kimodo_constraints_v4"
SELECTION_FORMAT = "hy273_kimodo_gallery_selection_v2"
CONTROL_METRICS = (
    "constraint_root2d_err",
    "constraint_fullbody_keyframe",
    "constraint_end_effector",
    "constraint_end_effector_rotation_deg",
)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL at {path}:{line_number}") from exc
    return records


def load_formal_records(eval_dir: Path) -> list[dict[str, Any]]:
    shard_paths = sorted((eval_dir / "shards").glob("shard_*.jsonl"))
    if len(shard_paths) != 8:
        raise RuntimeError(f"Expected eight formal shards, found {len(shard_paths)}")
    records = [record for path in shard_paths for record in load_jsonl(path)]
    if len(records) != 8084:
        raise RuntimeError(f"Expected 8084 formal records, found {len(records)}")
    if any(record.get("status") != "ok" for record in records):
        raise RuntimeError("Formal records contain failed cases")
    keys = [str(record["case_key"]) for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Formal records contain duplicate case keys")
    for record in records:
        if record.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol drift in {record['case_key']}")
        if record.get("evaluation_code_sha256") != EXPECTED_EVALUATION_CODE_SHA256:
            raise RuntimeError(f"Evaluator identity drift in {record['case_key']}")
        if record.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"Checkpoint identity drift in {record['case_key']}")
        if record.get("weight_source") != "ema":
            raise RuntimeError(f"Non-EMA formal case: {record['case_key']}")
    return records


def percentile_ranks(
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
        for _, key in values[start:end]:
            output[key] = float(rank)
        start = end
    return output


def select_representative_cases(
    records: list[dict[str, Any]], quantiles: tuple[float, ...]
) -> list[dict[str, Any]]:
    with_text = [
        record for record in records if record.get("text_regime") == "withtext"
    ]
    if len(with_text) != 4042:
        raise RuntimeError(f"Expected 4042 with-text records, found {len(with_text)}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in with_text:
        grouped[str(record["subtype"])].append(record)
    if set(grouped) != set(KIMODO_CONTROL_SUBTYPES):
        raise RuntimeError("Formal records do not cover all Kimodo subtypes")

    selected: list[dict[str, Any]] = []
    for subtype in KIMODO_CONTROL_SUBTYPES:
        rows = grouped[subtype]
        metric_ranks = {
            name: percentile_ranks(rows, name)
            for name in CONTROL_METRICS
        }
        metric_ranks = {name: ranks for name, ranks in metric_ranks.items() if ranks}
        if not metric_ranks:
            raise RuntimeError(f"No control metrics available for {subtype}")
        scored: list[tuple[float, dict[str, Any], dict[str, float]]] = []
        for row in rows:
            key = str(row["case_key"])
            ranks = {
                name: values[key]
                for name, values in metric_ranks.items()
                if key in values
            }
            if not ranks:
                raise RuntimeError(f"No percentile score for {key}")
            score = sum(ranks.values()) / len(ranks)
            scored.append((float(score), row, ranks))
        scored.sort(key=lambda item: (item[0], str(item[1]["case_key"])))

        used: set[str] = set()
        for quantile in quantiles:
            candidates = [item for item in scored if str(item[1]["case_key"]) not in used]
            score, record, component_ranks = min(
                candidates,
                key=lambda item: (
                    abs(item[0] - quantile),
                    str(item[1]["case_key"]),
                ),
            )
            used.add(str(record["case_key"]))
            metrics = {
                name: float(record["metrics"]["generated_raw"][name])
                for name in CONTROL_METRICS
                if name in record["metrics"]["generated_raw"]
            }
            selected.append(
                {
                    "selection_index": len(selected),
                    "target_quantile": float(quantile),
                    "composite_percentile": float(score),
                    "metric_percentiles": component_ranks,
                    "selection_metrics": metrics,
                    "formal_record": record,
                }
            )
    return selected


def parse_quantiles(value: str) -> tuple[float, ...]:
    quantiles = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not quantiles or any(not 0.0 <= item <= 1.0 for item in quantiles):
        raise ValueError(f"Invalid quantiles: {value!r}")
    return quantiles


def prepare_selection(args: argparse.Namespace) -> dict[str, Any]:
    eval_dir = Path(args.formal_eval_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    protocol_path = eval_dir / "protocol_manifest.json"
    preflight_path = eval_dir / "preflight_manifest.json"
    summary_path = eval_dir / "summary.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_protocol = {
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "cfg_scale_text": 2.0,
        "cfg_scale_control": 2.0,
        "num_steps": 32,
        "seed": 3407,
        "weight_source": "ema",
        "expected_case_count": 8084,
        "dataset_size": 4042,
        "split": "test",
        "caption_policy": "first_full_motion",
        "max_frames": 300,
        "min_frames": 2,
        "max_sparse_keyframes": 20,
        "contact_init": "random",
        "contact_feedback": "blend",
    }
    for name, expected in expected_protocol.items():
        if protocol.get(name) != expected:
            raise RuntimeError(
                f"Formal protocol mismatch for {name}: "
                f"expected={expected!r}, actual={protocol.get(name)!r}"
            )
    if not summary.get("complete") or summary.get("num_success") != 8084:
        raise RuntimeError("Formal summary is incomplete")
    preflight_sha256 = sha256_file(preflight_path)
    protocol_preflight = protocol.get("preflight_manifest", {})
    if preflight_sha256 != protocol_preflight.get("sha256"):
        raise RuntimeError("Formal preflight manifest identity drift")
    dataset_identity = preflight.get("dataset", {})
    expected_dataset_identity = {
        "data_root": protocol.get("data_root"),
        "dataset_size": protocol.get("dataset_size"),
        "split": protocol.get("split"),
        "caption_policy": protocol.get("caption_policy"),
    }
    observed_dataset_identity = {
        name: dataset_identity.get(name) for name in expected_dataset_identity
    }
    if observed_dataset_identity != expected_dataset_identity:
        raise RuntimeError(
            "Formal protocol/preflight dataset identity mismatch: "
            f"expected={expected_dataset_identity!r}, "
            f"observed={observed_dataset_identity!r}"
        )
    if not dataset_identity.get("text_root"):
        raise RuntimeError("Formal preflight does not record text_root")

    checkpoint_stat_before = checkpoint_path.stat()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_stat_after = checkpoint_path.stat()
    stable_stat_before = (
        int(checkpoint_stat_before.st_dev),
        int(checkpoint_stat_before.st_ino),
        int(checkpoint_stat_before.st_size),
        int(checkpoint_stat_before.st_mtime_ns),
    )
    stable_stat_after = (
        int(checkpoint_stat_after.st_dev),
        int(checkpoint_stat_after.st_ino),
        int(checkpoint_stat_after.st_size),
        int(checkpoint_stat_after.st_mtime_ns),
    )
    if stable_stat_before != stable_stat_after:
        raise RuntimeError("Checkpoint changed while hashing")
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA mismatch: expected={EXPECTED_CHECKPOINT_SHA256}, "
            f"actual={checkpoint_sha256}"
        )
    quantiles = parse_quantiles(args.quantiles)
    records = load_formal_records(eval_dir)
    cases = select_representative_cases(records, quantiles)
    expected_count = len(KIMODO_CONTROL_SUBTYPES) * len(quantiles)
    if len(cases) != expected_count:
        raise RuntimeError(f"Expected {expected_count} selected cases, found {len(cases)}")
    selection = {
        "format": SELECTION_FORMAT,
        "selection_policy": (
            "with-text cases; within-subtype percentile rank per available control "
            "metric; equal-weight mean of metric percentiles; nearest unique case "
            "to each requested quantile"
        ),
        "created_unix": time.time(),
        "formal_eval_dir": str(eval_dir),
        "formal_protocol_sha256": sha256_file(protocol_path),
        "formal_preflight_sha256": preflight_sha256,
        "formal_summary_sha256": sha256_file(summary_path),
        "protocol": {
            **expected_protocol,
            # The formal evaluator fixes this internally rather than exposing a CLI flag.
            "cfg_apply_contacts": True,
        },
        "dataset": {
            "data_root": str(dataset_identity["data_root"]),
            "text_root": str(dataset_identity["text_root"]),
            "ordered_records_sha256": str(
                dataset_identity["ordered_records_sha256"]
            ),
            "caption_selection_sha256": str(
                dataset_identity["caption_selection_sha256"]
            ),
            "assets_content_sha256": str(
                dataset_identity["assets"]["content_sha256"]
            ),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "size": int(checkpoint_stat_after.st_size),
            "mtime_ns": int(checkpoint_stat_after.st_mtime_ns),
        },
        "quantiles": list(quantiles),
        "num_cases": len(cases),
        "cases": cases,
    }
    validate_selection_dataset(selection)
    return selection


def make_dataset(selection: dict[str, Any]) -> Kimodo273TextDataset:
    protocol = selection["protocol"]
    dataset_identity = selection["dataset"]
    return Kimodo273TextDataset(
        dataset_identity["data_root"],
        split=str(protocol["split"]),
        text_root=dataset_identity["text_root"],
        max_frames=int(protocol["max_frames"]),
        min_frames=int(protocol["min_frames"]),
        random_crop=False,
        exclude_fallback_short_clips=False,
        deterministic_text=True,
        caption_policy=str(protocol["caption_policy"]),
    )


def assert_item_matches_record(item: dict[str, Any], record: dict[str, Any]) -> None:
    expected = {
        "motion_id": str(record["motion_id"]),
        "rel_path": str(record["rel_path"]),
        "crop_start": int(record["crop_start"]),
        "caption_index": int(record["caption_index"]),
        "caption_line_number": int(record["caption_line_number"]),
        "text": str(record["source_caption"]),
    }
    observed = {
        "motion_id": str(item["motion_id"]),
        "rel_path": str(item["rel_path"]),
        "crop_start": int(item.get("crop_start", 0)),
        "caption_index": int(item.get("caption_index", 0)),
        "caption_line_number": int(item.get("caption_line_number", 0)),
        "text": str(item["text"]),
    }
    if observed != expected:
        raise RuntimeError(
            f"Dataset/caption drift for {record['case_key']}: "
            f"expected={expected!r}, observed={observed!r}"
        )


def validate_selection_dataset(
    selection: dict[str, Any],
) -> Kimodo273TextDataset:
    dataset = make_dataset(selection)
    expected_size = int(selection["protocol"]["dataset_size"])
    if len(dataset) != expected_size:
        raise RuntimeError(
            f"Gallery dataset size drift: expected={expected_size}, actual={len(dataset)}"
        )
    for selected in selection["cases"]:
        record = selected["formal_record"]
        index = int(record["dataset_index"])
        if index < 0 or index >= len(dataset):
            raise RuntimeError(
                f"Dataset index out of range for {record['case_key']}: {index}"
            )
        assert_item_matches_record(dataset[index], record)
    return dataset


def metric_differences(
    regenerated: dict[str, float], formal: dict[str, float]
) -> dict[str, float]:
    if set(regenerated) != set(formal):
        raise RuntimeError(
            f"Regenerated metric keys differ: {sorted(regenerated)} vs {sorted(formal)}"
        )
    return {
        name: abs(float(regenerated[name]) - float(formal[name]))
        for name in regenerated
    }


def case_directory(output_dir: Path, selection: dict[str, Any]) -> Path:
    record = selection["formal_record"]
    quantile = int(round(float(selection["target_quantile"]) * 100))
    return output_dir / "cases" / (
        f"{int(selection['selection_index']):02d}_"
        f"{record['subtype']}_q{quantile:02d}_{record['motion_id']}"
    )


def run_worker(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    selection_path = output_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("format") != SELECTION_FORMAT:
        raise RuntimeError("Unsupported gallery selection manifest")
    protocol = selection["protocol"]
    checkpoint_info = selection["checkpoint"]
    checkpoint_path = Path(checkpoint_info["path"])
    stat = checkpoint_path.stat()
    if int(stat.st_size) != int(checkpoint_info["size"]):
        raise RuntimeError("Checkpoint size changed after gallery preflight")
    if int(stat.st_mtime_ns) != int(checkpoint_info["mtime_ns"]):
        raise RuntimeError("Checkpoint mtime changed after gallery preflight")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = validate_selection_dataset(selection)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    train_args = argparse.Namespace(**checkpoint.get("args", {}))
    state_dict, weight_source = checkpoint_weight_state(
        checkpoint, "ema", str(checkpoint_path)
    )
    if weight_source != "ema":
        raise RuntimeError("Gallery requires EMA weights")
    prediction_type = str(getattr(train_args, "prediction_type", ""))
    if prediction_type != "x0":
        raise RuntimeError(f"Gallery requires x0 prediction, got {prediction_type!r}")
    model = create_model(train_args).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    normalizer = checkpoint_normalizer(
        checkpoint, train_args, device, str(checkpoint_path)
    )
    del state_dict
    del checkpoint

    assigned = [
        case
        for case in selection["cases"]
        if int(case["selection_index"]) % int(args.num_workers)
        == int(args.worker_rank)
    ]
    completed = 0
    for selected in assigned:
        out_dir = case_directory(output_dir, selected)
        metadata_path = out_dir / "metadata.json"
        if metadata_path.is_file() and not args.overwrite:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("case_key") == selected["formal_record"]["case_key"]:
                print(f"GALLERY_SKIP {existing['case_key']}", flush=True)
                completed += 1
                continue
            raise RuntimeError(f"Conflicting existing output: {out_dir}")

        record = selected["formal_record"]
        item = dataset[int(record["dataset_index"])]
        assert_item_matches_record(item, record)
        source_motion = item["motion"].float()
        transform = apply_kimodo_training_transform(
            source_motion.unsqueeze(0), random_heading=False, root_shift=True
        )
        target_cpu = transform.motion[0].contiguous()
        if int(target_cpu.shape[0]) != int(record["length"]):
            raise RuntimeError(f"Length drift for {record['case_key']}")
        constraint_cpu = compile_kimodo_constraint(
            target_cpu,
            str(record["subtype"]),
            seed=int(record["sample_seed"]),
            max_sparse_keyframes=int(protocol["max_sparse_keyframes"]),
        )
        target = target_cpu.unsqueeze(0).to(device)
        constraint = constraint_to_device(constraint_cpu, device)
        length = int(target.shape[1])
        lengths = torch.tensor([length], dtype=torch.long, device=device)
        seed_sampling(int(record["sample_seed"]), device)
        started = time.perf_counter()
        sampled = sample_ode(
            model,
            normalizer,
            lengths,
            [str(record["text"])],
            constraint.observed_motion.unsqueeze(0),
            constraint.motion_mask.unsqueeze(0),
            transform.c_dir.to(device),
            num_steps=int(protocol["num_steps"]),
            self_conditioning=bool(getattr(train_args, "self_conditioning", False)),
            cfg_scale=float(protocol["cfg_scale_text"]),
            control_cfg_scale=float(protocol["cfg_scale_control"]),
            contact_init=str(protocol["contact_init"]),
            contact_feedback=str(protocol["contact_feedback"]),
            cfg_apply_contacts=bool(protocol["cfg_apply_contacts"]),
            prediction_type=prediction_type,
            velocity_t_eps=1e-4,
            return_details=True,
        )
        if not isinstance(sampled, ODESampleOutput):
            raise AssertionError("Expected detailed ODE output")
        raw_metrics = evaluate_kimodo_constraint_case(
            sampled.raw_motion[0], target[0], constraint
        )
        exact_metrics = evaluate_kimodo_constraint_case(
            sampled.exact_clamped_motion[0], target[0], constraint
        )
        differences = metric_differences(
            raw_metrics, record["metrics"]["generated_raw"]
        )
        max_difference = max(differences.values(), default=0.0)
        if max_difference > float(args.metric_tolerance):
            raise RuntimeError(
                f"Formal replay metric drift for {record['case_key']}: "
                f"max_abs={max_difference} tolerance={args.metric_tolerance}"
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "generated_raw.npy", sampled.raw_motion[0].cpu().numpy())
        np.save(
            out_dir / "generated_exact_clamped.npy",
            sampled.exact_clamped_motion[0].cpu().numpy(),
        )
        np.save(out_dir / "target.npy", target_cpu.numpy())
        np.save(out_dir / "observed.npy", constraint_cpu.observed_motion.numpy())
        np.save(out_dir / "mask.npy", constraint_cpu.motion_mask.numpy())
        np.save(
            out_dir / "root_metric_frames.npy",
            constraint_cpu.root_metric_frames.numpy(),
        )
        np.save(
            out_dir / "fullbody_metric_frames.npy",
            constraint_cpu.fullbody_metric_frames.numpy(),
        )
        np.save(
            out_dir / "endpoint_position_metric_mask.npy",
            constraint_cpu.endpoint_position_metric_mask.numpy(),
        )
        np.save(
            out_dir / "endpoint_rotation_metric_mask.npy",
            constraint_cpu.endpoint_rotation_metric_mask.numpy(),
        )
        atomic_json(
            metadata_path,
            {
                "format": "hy273_kimodo_gallery_case_v1",
                "case_key": record["case_key"],
                "selection_index": selected["selection_index"],
                "target_quantile": selected["target_quantile"],
                "composite_percentile": selected["composite_percentile"],
                "metric_percentiles": selected["metric_percentiles"],
                "subtype": record["subtype"],
                "family": record["family"],
                "dataset_index": record["dataset_index"],
                "motion_id": record["motion_id"],
                "length": length,
                "caption": record["text"],
                "sample_seed": record["sample_seed"],
                "constraint_components": constraint_cpu.components,
                "formal_metrics": record["metrics"]["generated_raw"],
                "regenerated_metrics": raw_metrics,
                "diagnostic_exact_metrics": exact_metrics,
                "formal_metric_abs_diff": differences,
                "formal_metric_max_abs_diff": max_difference,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "evaluation_code_sha256": EXPECTED_EVALUATION_CODE_SHA256,
                "protocol": protocol,
                "physical_gpu": args.physical_gpu,
                "worker_rank": args.worker_rank,
                "elapsed_seconds": elapsed,
            },
        )
        completed += 1
        print(
            f"GALLERY_DONE {record['case_key']} q={selected['target_quantile']:.2f} "
            f"replay_max_abs={max_difference:.3g} seconds={elapsed:.2f}",
            flush=True,
        )
    atomic_json(
        output_dir / "workers" / f"worker_{int(args.worker_rank):02d}.json",
        {
            "worker_rank": int(args.worker_rank),
            "physical_gpu": str(args.physical_gpu),
            "assigned": len(assigned),
            "completed": completed,
            "status": "complete",
        },
    )


def finalize_manifest(output_dir: Path, selection: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for selected in selection["cases"]:
        case_dir = case_directory(output_dir, selected)
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing generated case metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "case_dir": str(case_dir.relative_to(output_dir)),
                "metadata": str(metadata_path.relative_to(output_dir)),
                **metadata,
            }
        )
    payload = {
        "format": "hy273_kimodo_gallery_manifest_v1",
        "num_cases": len(rows),
        "selection_manifest": "selection.json",
        "cases": rows,
    }
    atomic_json(output_dir / "gallery_manifest.json", payload)
    return payload


def launch(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "selection.json"
    if selection_path.exists() and not args.overwrite_selection:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        selection = prepare_selection(args)
        atomic_json(selection_path, selection)
    if args.select_only:
        print(json.dumps({"selection": str(selection_path), "cases": selection["num_cases"]}))
        return

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError(f"Invalid GPU list: {args.gpus!r}")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, Any, Path]] = []
    for rank, gpu in enumerate(gpus):
        log_path = log_dir / f"worker_{rank:02d}.log"
        handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output_dir",
            str(output_dir),
            "--checkpoint",
            str(args.checkpoint),
            "--worker_rank",
            str(rank),
            "--num_workers",
            str(len(gpus)),
            "--device",
            "cuda:0",
            "--physical_gpu",
            gpu,
            "--metric_tolerance",
            str(args.metric_tolerance),
        ]
        if args.overwrite:
            command.append("--overwrite")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle, log_path))

    failures: list[str] = []
    for process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code != 0:
            failures.append(f"{log_path}: exit={return_code}")
    if failures:
        raise RuntimeError("Gallery workers failed: " + "; ".join(failures))

    checkpoint_path = Path(selection["checkpoint"]["path"])
    checkpoint_sha256_after = sha256_file(checkpoint_path)
    if checkpoint_sha256_after != selection["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint changed during gallery generation")
    manifest = finalize_manifest(output_dir, selection)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selection": str(selection_path),
                "gallery_manifest": str(output_dir / "gallery_manifest.json"),
                "num_cases": manifest["num_cases"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal_eval_dir",
        default=(
            "/mnt/afs/mogeflow-control/eval_runs/"
            "hy273_step400k_hml3d_kimodo_full_v4_20260714_041717"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/mnt/afs/mogeflow_kimodo_like_local_artifacts/checkpoints/"
            "stage2/step_00400000.pt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "/mnt/afs/mogeflow-control/generation/"
            "hy273_step400k_kimodo_gallery_39"
        ),
    )
    parser.add_argument("--quantiles", default="0.25,0.50,0.75")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--metric_tolerance", type=float, default=1e-4)
    parser.add_argument("--select_only", action="store_true")
    parser.add_argument("--overwrite_selection", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker_rank", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical_gpu", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.worker_rank >= 0:
        if args.num_workers < 1 or args.worker_rank >= args.num_workers:
            raise ValueError("Invalid worker rank/world size")
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
