#!/usr/bin/env python
"""DDP8 exact-resume gate at the frozen 250K and 400K phase boundaries."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_multitask_manifest_dataset import HY273MultitaskManifestDataset
from common.fixed_bucket_ddp import FixedBucketGradientSynchronizer
from models.raw_motion.hy273_multitask_condition import TaskId, TrainStream
from models.raw_motion.hy273_multitask_losses import compute_hy273_multitask_loss
from models.raw_motion.hy273_normalizer import HY273Normalizer
from models.raw_motion.hy273_slices import CONTACT_SLICE, CONT_DIM
from tools.gate_hy273_multitask_capabilities import (
    CAPABILITIES,
    _materialize_batch,
    _prepare,
)
from train_hy273_multitask import (
    _context_optimizer_steps,
    apply_optimizer_phase,
    assert_and_mask_context_gradients,
    cfg_get,
    create_model,
    initialize_ema,
    load_config,
    optimizer_groups,
    seed_model_initialization,
    update_ema,
    validate_assets,
    validate_frozen_contract,
)


FORMAT = "hy273_multitask_phase_resume_gate_v1"
ROUTES = ("hml", "edit", "hml")
BOUNDARIES = (250_000, 400_000)
TRACE_PARAMETER_NAMES = (
    "root_input_proj.weight",
    "body_input_proj.weight",
    "direction_embed.2.weight",
)


def _selected_parameters(
    model: torch.nn.Module,
) -> dict[str, torch.nn.Parameter]:
    named = dict(model.named_parameters())
    missing = set(TRACE_PARAMETER_NAMES) - set(named)
    if missing:
        raise RuntimeError(f"Missing trace parameters: {sorted(missing)}")
    return {name: named[name] for name in TRACE_PARAMETER_NAMES}


def _selected_gradient_state(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad
        for name, parameter in _selected_parameters(model).items()
    }


def _selected_parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(_selected_parameters(model))


def _selected_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    return {
        name: optimizer.state.get(parameter, {})
        for name, parameter in _selected_parameters(model).items()
    }


def _trace_hash(value: Any, enabled: bool, rank: int) -> str | None:
    return _state_sha(value) if enabled and rank == 0 else None


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            digest.update(b"tensor\0")
            digest.update(_tensor_sha(item).encode("ascii"))
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda part: str(part)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            digest.update(str(len(item)).encode("ascii"))
            for child in item:
                visit(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))
            digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def _state_inventory(value: Any) -> dict[str, int]:
    tensors = 0
    elements = 0
    bytes_total = 0

    def visit(item: Any) -> None:
        nonlocal tensors, elements, bytes_total
        if torch.is_tensor(item):
            tensors += 1
            elements += item.numel()
            bytes_total += item.numel() * item.element_size()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {
        "tensor_count": tensors,
        "element_count": elements,
        "tensor_bytes": bytes_total,
    }


def _assert_exact(actual: Any, expected: Any, path: str = "state") -> None:
    if torch.is_tensor(expected):
        if not torch.is_tensor(actual):
            raise RuntimeError(f"{path}: expected tensor, got {type(actual).__name__}")
        if actual.dtype != expected.dtype or tuple(actual.shape) != tuple(expected.shape):
            raise RuntimeError(
                f"{path}: tensor metadata mismatch actual={actual.dtype}/{tuple(actual.shape)} "
                f"expected={expected.dtype}/{tuple(expected.shape)}"
            )
        actual_cpu = actual.detach().cpu()
        if not torch.equal(actual_cpu, expected):
            difference = (
                (actual_cpu.float() - expected.float()).abs().max().item()
                if actual_cpu.numel()
                else 0.0
            )
            raise RuntimeError(f"{path}: tensor mismatch max_abs={difference}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RuntimeError(f"{path}: mapping keys differ")
        for key in expected:
            _assert_exact(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise RuntimeError(f"{path}: sequence structure differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_exact(actual_item, expected_item, f"{path}[{index}]")
        return
    if actual != expected:
        raise RuntimeError(f"{path}: value mismatch actual={actual!r} expected={expected!r}")


def _optimizer_hparams(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    return [
        {
            "group_name": str(group["group_name"]),
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in optimizer.param_groups
    ]


def _capture_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    *,
    context_update_count: int,
    next_route_index: int,
) -> dict[str, Any]:
    return _to_cpu(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema,
            "context_update_count": int(context_update_count),
            "next_route_index": int(next_route_index),
            "optimizer_hparams": _optimizer_hparams(optimizer),
        }
    )


def _rank_probe(model: torch.nn.Module) -> torch.Tensor:
    parameters = tuple(model.parameters())
    selected = (parameters[0], parameters[len(parameters) // 2], parameters[-1])
    values = []
    for parameter in selected:
        flat = parameter.detach().float().view(-1)
        values.extend((flat.sum(), flat.square().sum()))
    return torch.stack(values).double()


def _assert_rank_sync(model: torch.nn.Module) -> None:
    value = _rank_probe(model)
    low, high = value.clone(), value.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if not torch.equal(low, high):
        raise RuntimeError("DDP ranks diverged in the phase/resume gate")


def _snapshot_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: dict[str, torch.Tensor],
    *,
    boundary: int,
    next_route_index: int,
    context_update_count: int,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "boundary": int(boundary),
        "next_route_index": int(next_route_index),
        "context_update_count": int(context_update_count),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
    }


def _write_snapshot(path: Path, payload: dict[str, Any], rank: int) -> None:
    if rank == 0:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
    dist.barrier()


def _load_snapshot(
    path: Path,
    *,
    boundary: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, int]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if checkpoint.get("format") != FORMAT or int(checkpoint.get("boundary", -1)) != boundary:
        raise RuntimeError("Phase/resume snapshot identity mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    apply_optimizer_phase(optimizer, boundary + int(checkpoint["next_route_index"]))
    ema = {key: value.to(device=device) for key, value in checkpoint["ema"].items()}
    random.setstate(checkpoint["rng"]["python"])
    np.random.set_state(checkpoint["rng"]["numpy"])
    torch.set_rng_state(checkpoint["rng"]["torch_cpu"])
    torch.cuda.set_rng_state_all(checkpoint["rng"]["torch_cuda"])
    return (
        ema,
        int(checkpoint["context_update_count"]),
        int(checkpoint["next_route_index"]),
    )


def _run_updates(
    *,
    boundary: int,
    route_start: int,
    route_stop: int,
    config: dict[str, Any],
    weights: Any,
    normalizer: HY273Normalizer,
    prepared: dict[str, dict[str, Any]],
    device: torch.device,
    local_rank: int,
    run_seed: int,
    bucket_cap_mb: float,
    gradient_sync_mode: str,
    prime_hml: bool,
    trace_state: bool,
    resume_path: Path | None = None,
    save_path: Path | None = None,
    capture_final: bool = True,
    expected_loaded_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    seed_model_initialization(run_seed + boundary)
    model = create_model(config).to(device)
    groups, _ = optimizer_groups(model, boundary)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    ema = initialize_ema(model)
    context_updates = 0
    actual_start = int(route_start)
    if resume_path is not None:
        ema, context_updates, actual_start = _load_snapshot(
            resume_path,
            boundary=boundary,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        if actual_start != route_start:
            raise RuntimeError(
                f"Resume route mismatch: checkpoint={actual_start}, requested={route_start}"
            )
        if dist.get_rank() == 0 and expected_loaded_state is not None:
            loaded_state = _capture_state(
                model,
                optimizer,
                ema,
                context_update_count=context_updates,
                next_route_index=actual_start,
            )
            _assert_exact(loaded_state, expected_loaded_state, "snapshot_roundtrip")
            del loaded_state
    ddp_kwargs = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "broadcast_buffers": False,
        "find_unused_parameters": False,
        "static_graph": gradient_sync_mode == "ddp",
    }
    if bucket_cap_mb > 0:
        ddp_kwargs["bucket_cap_mb"] = float(bucket_cap_mb)
    ddp = DDP(model, **ddp_kwargs)
    ddp.train()
    fixed_synchronizer = None
    if gradient_sync_mode == "fixed_bucket":
        fixed_synchronizer = FixedBucketGradientSynchronizer(
            model,
            bucket_cap_mb=bucket_cap_mb if bucket_cap_mb > 0 else 25.0,
        )
    elif gradient_sync_mode != "ddp":
        raise ValueError(f"Unknown gradient_sync_mode={gradient_sync_mode!r}")

    def forward_loss(batch: dict[str, Any], step: int):
        condition = batch["condition"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = ddp(
                batch["flow_state"]["model_in"],
                t=batch["timesteps"],
                c_dir=condition.frame_gauge_dir,
                text=batch["texts"],
                length_mask=condition.target_valid,
                x_self_cond=None,
                text_drop_prob=0.0,
                condition=condition,
            )
            bundle = compute_hy273_multitask_loss(
                x0_hat_cont=prediction[..., :CONT_DIM],
                contact_logits=prediction[..., CONTACT_SLICE],
                z_cont_imputed=batch["flow_state"]["z_cont_imp"],
                x0_target_norm=batch["x0_norm"],
                x0_target_physical=batch["target_physical"],
                hard_observed_norm=batch["observed_norm"],
                hard_mask=batch["hard_mask"],
                target_valid=condition.target_valid,
                timesteps=batch["timesteps"],
                normalizer=normalizer,
                global_step=step,
                weights=weights,
            )
        return prediction, bundle, condition

    if prime_hml:
        optimizer.zero_grad(set_to_none=True)
        sync_context = ddp.no_sync() if fixed_synchronizer is not None else nullcontext()
        with sync_context:
            prime_prediction, prime_bundle, prime_condition = forward_loss(
                prepared["hml"], boundary
            )
            prime_bundle.total.backward()
        if fixed_synchronizer is not None:
            fixed_synchronizer.synchronize()
        assert_and_mask_context_gradients(
            model,
            context_active=False,
            global_step=boundary,
            optimizer=optimizer,
        )
        optimizer.zero_grad(set_to_none=True)
        del prime_prediction, prime_bundle, prime_condition

    reports = []
    for route_index in range(actual_start, route_stop):
        route = ROUTES[route_index]
        step = boundary + route_index
        apply_optimizer_phase(optimizer, step)
        batch = prepared[route]
        optimizer.zero_grad(set_to_none=True)
        sync_context = ddp.no_sync() if fixed_synchronizer is not None else nullcontext()
        with sync_context:
            prediction, bundle, condition = forward_loss(batch, step)
            bundle.total.backward()
        if fixed_synchronizer is not None:
            fixed_synchronizer.synchronize()
        rank = dist.get_rank()
        gradient_after_backward_sha = _trace_hash(
            _selected_gradient_state(model), trace_state, rank
        )
        context_active = bool(
            condition.source_present.any().item()
            or (condition.task_id == int(TaskId.EDIT)).any().item()
        )
        assert_and_mask_context_gradients(
            model,
            context_active=context_active,
            global_step=step,
            optimizer=optimizer,
        )
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg_get(config, "training.gradient_clip"))
            ).item()
        )
        if not np.isfinite(grad_norm):
            raise RuntimeError("Non-finite gradient in phase/resume gate")
        gradient_after_clip_sha = _trace_hash(
            _selected_gradient_state(model), trace_state, rank
        )
        optimizer.step()
        model_after_step_sha = _trace_hash(
            _selected_parameter_state(model), trace_state, rank
        )
        optimizer_after_step_sha = _trace_hash(
            _selected_optimizer_state(model, optimizer), trace_state, rank
        )
        if source_present:
            context_updates += 1
        if step % int(cfg_get(config, "training.ema_every")) == 0:
            update_ema(ema, model, float(cfg_get(config, "training.ema_decay")))
        context_steps = sorted(set(_context_optimizer_steps(optimizer, model).values()))
        ddp_logging = ddp._get_ddp_logging_data()
        reports.append(
            {
                "route_index": route_index,
                "route": route,
                "step": step,
                "source_present": source_present,
                "loss": float(bundle.total.detach().float().item()),
                "grad_norm_preclip": grad_norm,
                "trace": {
                    "gradient_after_backward_sha256": gradient_after_backward_sha,
                    "gradient_after_clip_sha256": gradient_after_clip_sha,
                    "model_after_step_sha256": model_after_step_sha,
                    "optimizer_after_step_sha256": optimizer_after_step_sha,
                },
                "context_update_count": context_updates,
                "context_adam_steps": context_steps,
                "optimizer_hparams": _optimizer_hparams(optimizer),
                "uids": list(batch["uids"]),
                "ddp": {
                    key: ddp_logging.get(key)
                    for key in (
                        "bucket_cap_bytes",
                        "bucket_sizes",
                        "has_rebuilt_buckets",
                        "rebuilt_bucket_sizes",
                        "rebuilt_per_bucket_param_indices",
                        "prev_iteration_grad_ready_order_indices",
                    )
                },
                "fixed_gradient_sync": (
                    fixed_synchronizer.manifest
                    if fixed_synchronizer is not None
                    else None
                ),
            }
        )
    _assert_rank_sync(model)
    if save_path is not None:
        _write_snapshot(
            save_path,
            _snapshot_payload(
                model,
                optimizer,
                ema,
                boundary=boundary,
                next_route_index=route_stop,
                context_update_count=context_updates,
            ),
            dist.get_rank(),
        )
    captured = None
    if capture_final and dist.get_rank() == 0:
        captured = _capture_state(
            model,
            optimizer,
            ema,
            context_update_count=context_updates,
            next_route_index=route_stop,
        )
    del bundle, prediction, fixed_synchronizer, ddp, ema, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return captured, reports


def _prepare_batches(
    config: dict[str, Any], normalizer: HY273Normalizer, device: torch.device, rank: int
) -> dict[str, dict[str, Any]]:
    datasets = {
        stream: HY273MultitaskManifestDataset(
            cfg_get(config, "data.train_manifest"), stream
        )
        for stream in (TrainStream.HML_MIXED, TrainStream.MOTION_EDIT)
    }
    specs = {spec.name: spec for spec in CAPABILITIES}
    run_seed = int(cfg_get(config, "training.seed"))
    output = {}
    for route, spec_name, ordinal in (
        ("hml", "t2m", rank),
        ("edit", "edit", 8 + rank),
    ):
        dataset = datasets[specs[spec_name].stream]
        batch = _materialize_batch(
            dataset,
            specs[spec_name],
            sample_count=1,
            run_seed=run_seed,
            global_ordinal=ordinal,
        )
        output[route] = _prepare(
            batch,
            device=device,
            normalizer=normalizer,
            config=config,
            manifest_sha256=dataset.manifest_sha256,
            run_seed=run_seed,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/hy273_multitask_stage_b2_joint_adapt.yaml"
    )
    parser.add_argument(
        "--snapshot_dir", default="/dev/shm/hy273_multitask_phase_resume_gate"
    )
    parser.add_argument("--boundaries", default="250000,400000")
    parser.add_argument("--splits", default="1,2")
    parser.add_argument("--repeat_uninterrupted", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--bucket_cap_mb", type=float, default=0.0)
    parser.add_argument(
        "--gradient_sync_mode",
        choices=("ddp", "fixed_bucket"),
        default="ddp",
    )
    parser.add_argument("--prime_hml", action="store_true")
    parser.add_argument("--trace_state", action="store_true")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"Phase/resume gate requires 8 ranks, got {world_size}")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    boundaries = tuple(int(value) for value in args.boundaries.split(",") if value)
    splits = tuple(int(value) for value in args.splits.split(",") if value)
    if not boundaries or any(value not in BOUNDARIES for value in boundaries):
        raise ValueError(f"boundaries must be a nonempty subset of {BOUNDARIES}")
    if not splits or any(value not in (1, 2) for value in splits):
        raise ValueError("splits must be a nonempty subset of 1,2")
    config, config_path = load_config(args.config)
    weights = validate_frozen_contract(config)
    assets = validate_assets(config)
    run_seed = int(cfg_get(config, "training.seed"))
    stats_root = Path(cfg_get(config, "data.stats_root"))
    normalizer = HY273Normalizer.from_data_root(
        stats_root / "full",
        stats_dir=stats_root / "full",
        variance_eps=float(cfg_get(config, "model.stats_variance_eps")),
    ).to(device)
    prepared = _prepare_batches(config, normalizer, device, rank)
    snapshot_dir = Path(args.snapshot_dir).expanduser().resolve()
    if rank == 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    started = time.perf_counter()
    boundary_reports = []
    try:
        for boundary in boundaries:
            expected, uninterrupted_reports = _run_updates(
                boundary=boundary,
                route_start=0,
                route_stop=len(ROUTES),
                config=config,
                weights=weights,
                normalizer=normalizer,
                prepared=prepared,
                device=device,
                local_rank=local_rank,
                run_seed=run_seed,
                bucket_cap_mb=args.bucket_cap_mb,
                gradient_sync_mode=args.gradient_sync_mode,
                prime_hml=args.prime_hml,
                trace_state=args.trace_state,
            )
            if rank == 0 and expected is None:
                raise AssertionError("Rank 0 did not capture uninterrupted state")
            repeated_exact = None
            if args.repeat_uninterrupted:
                repeated, repeated_reports = _run_updates(
                    boundary=boundary,
                    route_start=0,
                    route_stop=len(ROUTES),
                    config=config,
                    weights=weights,
                    normalizer=normalizer,
                    prepared=prepared,
                    device=device,
                    local_rank=local_rank,
                    run_seed=run_seed,
                    bucket_cap_mb=args.bucket_cap_mb,
                    gradient_sync_mode=args.gradient_sync_mode,
                    prime_hml=args.prime_hml,
                    trace_state=args.trace_state,
                )
                if rank == 0:
                    if expected is None or repeated is None:
                        raise AssertionError("Rank 0 did not capture repeated state")
                    _assert_exact(repeated, expected, "repeat_uninterrupted")
                    repeated_exact = {
                        "exact_match": True,
                        "reports": repeated_reports,
                        "state_inventory": _state_inventory(repeated),
                    }
                    del repeated
            split_reports = []
            for split_after in splits:
                snapshot = snapshot_dir / f"boundary_{boundary}_split_{split_after}.pt"
                prefix_state, prefix_reports = _run_updates(
                    boundary=boundary,
                    route_start=0,
                    route_stop=split_after,
                    config=config,
                    weights=weights,
                    normalizer=normalizer,
                    prepared=prepared,
                    device=device,
                    local_rank=local_rank,
                    run_seed=run_seed,
                    bucket_cap_mb=args.bucket_cap_mb,
                    gradient_sync_mode=args.gradient_sync_mode,
                    prime_hml=args.prime_hml,
                    trace_state=args.trace_state,
                    save_path=snapshot,
                    capture_final=True,
                )
                resumed, suffix_reports = _run_updates(
                    boundary=boundary,
                    route_start=split_after,
                    route_stop=len(ROUTES),
                    config=config,
                    weights=weights,
                    normalizer=normalizer,
                    prepared=prepared,
                    device=device,
                    local_rank=local_rank,
                    run_seed=run_seed,
                    bucket_cap_mb=args.bucket_cap_mb,
                    gradient_sync_mode=args.gradient_sync_mode,
                    prime_hml=args.prime_hml,
                    trace_state=args.trace_state,
                    resume_path=snapshot,
                    expected_loaded_state=prefix_state,
                )
                if rank == 0:
                    if resumed is None or expected is None or prefix_state is None:
                        raise AssertionError("Rank 0 did not capture resumed state")
                    if args.trace_state:
                        expected_suffix = uninterrupted_reports[split_after:]
                        comparison = []
                        for expected_report, resumed_report in zip(
                            expected_suffix, suffix_reports
                        ):
                            comparison.append(
                                {
                                    "route_index": resumed_report["route_index"],
                                    "route": resumed_report["route"],
                                    "trace_equal": {
                                        key: resumed_report["trace"][key]
                                        == expected_report["trace"][key]
                                        for key in resumed_report["trace"]
                                    },
                                }
                            )
                        print(
                            json.dumps(
                                {
                                    "event": "phase_resume_trace_comparison",
                                    "boundary": boundary,
                                    "split_after": split_after,
                                    "comparison": comparison,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    _assert_exact(resumed, expected)
                    split_reports.append(
                        {
                            "split_after_route_count": split_after,
                            "snapshot_size": snapshot.stat().st_size,
                            "snapshot_sha256": _file_sha(snapshot),
                            "state_inventory": _state_inventory(resumed),
                            "prefix_reports": prefix_reports,
                            "suffix_reports": suffix_reports,
                            "exact_match": True,
                        }
                    )
                    snapshot.unlink()
                    del prefix_state
                dist.barrier()
            if rank == 0:
                assert expected is not None
                boundary_reports.append(
                    {
                        "boundary": boundary,
                        "routes": ROUTES,
                        "uninterrupted_reports": uninterrupted_reports,
                        "uninterrupted_state_inventory": _state_inventory(expected),
                        "repeated_uninterrupted": repeated_exact,
                        "resume_splits": split_reports,
                    }
                )
                del expected
            gc.collect()
    finally:
        if rank == 0:
            for leftover in snapshot_dir.glob("boundary_*_split_*.pt"):
                leftover.unlink()
        dist.barrier()

    if rank == 0:
        payload = {
            "format": FORMAT,
            "passed": True,
            "world_size": world_size,
            "deterministic_algorithms": bool(args.deterministic),
            "bucket_cap_mb": float(args.bucket_cap_mb),
            "gradient_sync_mode": args.gradient_sync_mode,
            "prime_hml": bool(args.prime_hml),
            "trace_state": bool(args.trace_state),
            "config_path": str(config_path),
            "asset_identity": assets,
            "boundaries": boundary_reports,
            "elapsed_seconds": time.perf_counter() - started,
        }
        output = Path(args.output_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
