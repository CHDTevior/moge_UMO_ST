from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from common.fixed_bucket_ddp import FixedBucketGradientSynchronizer
from models.raw_motion.hy273_multitask_losses import RatioLossTerm
from train_hy273_multitask import (
    apply_preserved_ratio_partition,
    create_local_ratio_process_group,
)


SUBRANK_DENOMINATORS = (0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 4.0, 1.0)
SUBRANK_NUMERATOR_COEFFICIENTS = (0.0, 2.0, 0.0, 0.0, 3.0, 5.0, 7.0, 11.0)


def _distributed_ratio_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_file: str,
    expanded: bool,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        synchronizer = FixedBucketGradientSynchronizer(
            model, bucket_cap_mb=0.001
        )

        if expanded:
            coefficient = SUBRANK_NUMERATOR_COEFFICIENTS[rank]
            denominator = SUBRANK_DENOMINATORS[rank]
            ratio_group_size = 2
        else:
            coefficient = sum(
                SUBRANK_NUMERATOR_COEFFICIENTS[2 * rank : 2 * rank + 2]
            )
            denominator = sum(
                SUBRANK_DENOMINATORS[2 * rank : 2 * rank + 2]
            )
            ratio_group_size = 1

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            prediction = model(torch.tensor([[coefficient]], dtype=torch.float32))
            numerator = prediction.sum()
        term = RatioLossTerm(
            name="synthetic",
            group="representation",
            numerator=numerator,
            denominator=prediction.new_tensor(denominator),
            weight=0.7,
        )
        bundle = type("Bundle", (), {})()
        bundle.terms = {"synthetic": term}
        bundle.total = term.weighted
        ratio_group = create_local_ratio_process_group(
            world_size=world_size,
            rank=rank,
            group_size=ratio_group_size,
        )
        apply_preserved_ratio_partition(
            bundle,
            process_group=ratio_group,
            group_size=ratio_group_size,
        )
        bundle.total.backward()
        synchronizer.synchronize()
        gradient = model.weight.grad.detach().cpu()
        gathered = [torch.zeros_like(gradient) for _ in range(world_size)]
        dist.all_gather(gathered, gradient)
        if rank == 0:
            torch.save(torch.stack(gathered), result_file)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_distributed_case(
    tmp_path: Path, *, world_size: int, expanded: bool
) -> torch.Tensor:
    label = "expanded" if expanded else "parent"
    init_file = tmp_path / f"{label}_init"
    result_file = tmp_path / f"{label}_grad.pt"
    mp.spawn(
        _distributed_ratio_worker,
        args=(
            world_size,
            str(init_file),
            str(result_file),
            expanded,
        ),
        nprocs=world_size,
        join=True,
    )
    return torch.load(result_file, map_location="cpu", weights_only=True)


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed is unavailable",
)
def test_pair_ratio_collective_and_world_gradient_average_match_parent(
    tmp_path: Path,
) -> None:
    parent = _run_distributed_case(tmp_path, world_size=4, expanded=False)
    expanded = _run_distributed_case(tmp_path, world_size=8, expanded=True)
    torch.testing.assert_close(
        parent,
        parent[0].expand_as(parent),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        expanded,
        expanded[0].expand_as(expanded),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        expanded[0],
        parent[0],
        rtol=0.0,
        atol=0.0,
    )
