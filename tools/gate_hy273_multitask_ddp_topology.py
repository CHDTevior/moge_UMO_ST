#!/usr/bin/env python
"""Real DDP static-graph gate alternating source-absent/present routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.hy273_multitask_manifest_dataset import HY273MultitaskManifestDataset
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
    load_config,
    optimizer_groups,
    seed_model_initialization,
    tensor_group_norm,
    validate_assets,
    validate_frozen_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hy273_multitask_stage_b2_joint_adapt.yaml")
    parser.add_argument(
        "--sequence", choices=["hml-edit-hml", "edit-hml-edit"], required=True
    )
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"Topology gate requires 8 ranks, got {world_size}")
    config, _ = load_config(args.config)
    weights = validate_frozen_contract(config)
    assets = validate_assets(config)
    seed = int(cfg_get(config, "training.seed"))
    seed_model_initialization(seed)
    datasets = {
        stream: HY273MultitaskManifestDataset(
            cfg_get(config, "data.train_manifest"), stream
        )
        for stream in (TrainStream.HML_MIXED, TrainStream.MOTION_EDIT)
    }
    manifest_sha = datasets[TrainStream.HML_MIXED].manifest_sha256
    if manifest_sha != datasets[TrainStream.MOTION_EDIT].manifest_sha256:
        raise RuntimeError("Manifest SHA differs across topology-gate streams")
    specs = {spec.name: spec for spec in CAPABILITIES}
    batches = {
        "hml": _materialize_batch(
            datasets[TrainStream.HML_MIXED],
            specs["t2m"],
            sample_count=1,
            run_seed=seed,
            global_ordinal=rank,
        ),
        "edit": _materialize_batch(
            datasets[TrainStream.MOTION_EDIT],
            specs["edit"],
            sample_count=1,
            run_seed=seed,
            global_ordinal=world_size + rank,
        ),
    }
    stats_root = Path(cfg_get(config, "data.stats_root"))
    normalizer = HY273Normalizer.from_data_root(
        stats_root / "full",
        stats_dir=stats_root / "full",
        variance_eps=float(cfg_get(config, "model.stats_variance_eps")),
    ).to(device)
    model = create_model(config).to(device)
    groups, _ = optimizer_groups(model, 250_000)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    ddp = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,
    )
    ddp.train()
    route_order = args.sequence.split("-")
    reports = []
    context_updates = 0
    started = time.perf_counter()
    for offset, route in enumerate(route_order):
        step = 250_000 + offset
        apply_optimizer_phase(optimizer, step)
        prepared = _prepare(
            batches[route],
            device=device,
            normalizer=normalizer,
            config=config,
            manifest_sha256=manifest_sha,
            run_seed=seed,
        )
        condition = prepared["condition"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = ddp(
                prepared["flow_state"]["model_in"],
                t=prepared["timesteps"],
                c_dir=condition.frame_gauge_dir,
                text=prepared["texts"],
                length_mask=condition.target_valid,
                x_self_cond=None,
                text_drop_prob=0.0,
                condition=condition,
            )
            bundle = compute_hy273_multitask_loss(
                x0_hat_cont=prediction[..., :CONT_DIM],
                contact_logits=prediction[..., CONTACT_SLICE],
                z_cont_imputed=prepared["flow_state"]["z_cont_imp"],
                x0_target_norm=prepared["x0_norm"],
                x0_target_physical=prepared["target_physical"],
                hard_observed_norm=prepared["observed_norm"],
                hard_mask=prepared["hard_mask"],
                target_valid=condition.target_valid,
                timesteps=prepared["timesteps"],
                normalizer=normalizer,
                global_step=step,
                weights=weights,
            )
        bundle.total.backward()
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
        context_grad = tensor_group_norm(
            (*model.context_weight_parameters(), *model.context_bias_parameters())
        )
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        )
        optimizer.step()
        if source_present:
            context_updates += 1
        context_steps = sorted(set(_context_optimizer_steps(optimizer, model).values()))
        values = torch.tensor(
            [float(bundle.total.detach()), grad_norm, context_grad],
            device=device,
            dtype=torch.float64,
        )
        low, high = values.clone(), values.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("Non-finite DDP topology-gate metrics")
        reports.append(
            {
                "route": route,
                "step": step,
                "source_present": source_present,
                "loss_rank_min": float(low[0]),
                "loss_rank_max": float(high[0]),
                "grad_norm_rank_min": float(low[1]),
                "grad_norm_rank_max": float(high[1]),
                "context_grad_rank_min": float(low[2]),
                "context_grad_rank_max": float(high[2]),
                "context_adam_steps": context_steps,
            }
        )
    expected_updates = sum(route == "edit" for route in route_order)
    if context_updates != expected_updates:
        raise RuntimeError("Context update accounting mismatch")
    if set(_context_optimizer_steps(optimizer, model).values()) != {expected_updates}:
        raise RuntimeError("Context Adam state does not match source-present updates")
    dist.barrier()
    if rank == 0:
        output = {
            "format": "hy273_multitask_ddp_topology_gate_v1",
            "passed": True,
            "sequence": args.sequence,
            "world_size": world_size,
            "asset_identity": assets,
            "context_updates": context_updates,
            "elapsed_seconds": time.perf_counter() - started,
            "reports": reports,
        }
        path = Path(args.output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
