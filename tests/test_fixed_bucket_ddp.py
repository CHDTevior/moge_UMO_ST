from __future__ import annotations

import torch
from torch import nn

from common.fixed_bucket_ddp import FixedBucketGradientSynchronizer


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))


def test_fixed_bucket_manifest_is_stable_for_the_same_parameter_order() -> None:
    left = FixedBucketGradientSynchronizer(_model(), bucket_cap_mb=0.00005)
    right = FixedBucketGradientSynchronizer(_model(), bucket_cap_mb=0.00005)
    assert left.manifest == right.manifest
    assert left.bucket_count > 1


def test_fixed_bucket_sync_packs_averages_and_unpacks(monkeypatch) -> None:
    model = _model()
    synchronizer = FixedBucketGradientSynchronizer(model, bucket_cap_mb=0.00005)
    expected = {}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        parameter.grad = torch.full_like(parameter, float(index + 2))
        expected[name] = torch.full_like(parameter, float(index + 3) / 2.0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor, op) -> None:
        del op
        tensor.add_(1.0)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    synchronizer.synchronize()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, expected[name], rtol=0.0, atol=0.0)
