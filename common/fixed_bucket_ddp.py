"""Restart-stable synchronous gradient averaging for distributed training."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
import torch.distributed as dist


class FixedBucketGradientSynchronizer:
    """Average dense gradients in a fixed parameter and bucket order.

    Native DDP may rebuild buckets from the first backward graph after a
    restart. Mixed task graphs can therefore choose a different NCCL message
    layout and lose bitwise replay. This synchronizer packs gradients only
    after backward, using a construction-time plan that cannot depend on the
    first task observed by a process.
    """

    FORMAT = "fixed_bucket_gradient_sync_v1"

    def __init__(self, model: torch.nn.Module, bucket_cap_mb: float) -> None:
        if not float(bucket_cap_mb) > 0:
            raise ValueError("fixed bucket capacity must be positive")
        cap_bytes = max(1, int(float(bucket_cap_mb) * 1024 * 1024))
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not named:
            raise RuntimeError("fixed gradient synchronizer received no parameters")

        self._buckets: list[tuple[tuple[str, torch.nn.Parameter], ...]] = []
        current: list[tuple[str, torch.nn.Parameter]] = []
        current_bytes = 0
        current_dtype: torch.dtype | None = None
        current_device_type: str | None = None
        for row in named:
            _, parameter = row
            parameter_bytes = parameter.numel() * parameter.element_size()
            device_type = parameter.device.type
            incompatible = current and (
                parameter.dtype != current_dtype or device_type != current_device_type
            )
            over_capacity = current and current_bytes + parameter_bytes > cap_bytes
            if incompatible or over_capacity:
                self._buckets.append(tuple(current))
                current = []
                current_bytes = 0
            if not current:
                current_dtype = parameter.dtype
                current_device_type = device_type
            current.append(row)
            current_bytes += parameter_bytes
        if current:
            self._buckets.append(tuple(current))

        bucket_rows = []
        for bucket in self._buckets:
            first = bucket[0][1]
            bucket_rows.append(
                {
                    "elements": sum(parameter.numel() for _, parameter in bucket),
                    "bytes": sum(
                        parameter.numel() * parameter.element_size()
                        for _, parameter in bucket
                    ),
                    "parameter_count": len(bucket),
                    "dtype": str(first.dtype),
                    "device_type": first.device.type,
                }
            )
        manifest: dict[str, Any] = {
            "format": self.FORMAT,
            "bucket_cap_bytes": cap_bytes,
            "parameter_count": len(named),
            "parameter_elements": sum(parameter.numel() for _, parameter in named),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for _, parameter in named
            ),
            "ordered_parameter_name_sha256": hashlib.sha256(
                "\n".join(name for name, _ in named).encode("utf-8")
            ).hexdigest(),
            "buckets": bucket_rows,
        }
        encoded = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        self.manifest = manifest

        if dist.is_available() and dist.is_initialized():
            gathered: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, manifest["manifest_sha256"])
            if len(set(gathered)) != 1:
                raise RuntimeError(
                    f"fixed gradient bucket plans differ across ranks: {gathered}"
                )

        max_elements: dict[tuple[torch.device, torch.dtype], int] = {}
        for bucket in self._buckets:
            first = bucket[0][1]
            key = (first.device, first.dtype)
            elements = sum(parameter.numel() for _, parameter in bucket)
            max_elements[key] = max(max_elements.get(key, 0), elements)
        self._buffers = {
            key: torch.empty(elements, device=key[0], dtype=key[1])
            for key, elements in max_elements.items()
        }

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @torch.no_grad()
    def synchronize(self) -> None:
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        for bucket in self._buckets:
            first = bucket[0][1]
            count = sum(parameter.numel() for _, parameter in bucket)
            flat = self._buffers[(first.device, first.dtype)][:count]
            offset = 0
            for name, parameter in bucket:
                gradient = parameter.grad
                if gradient is None:
                    raise RuntimeError(
                        f"fixed gradient synchronizer found grad=None for {name}"
                    )
                if gradient.is_sparse:
                    raise RuntimeError("fixed gradient synchronizer requires dense gradients")
                if gradient.device != first.device or gradient.dtype != first.dtype:
                    raise RuntimeError("fixed gradient bucket mixes devices or dtypes")
                elements = gradient.numel()
                flat[offset : offset + elements].copy_(gradient.reshape(-1))
                offset += elements
            if world_size > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.mul_(1.0 / float(world_size))
            offset = 0
            for _, parameter in bucket:
                elements = parameter.numel()
                parameter.grad.copy_(flat[offset : offset + elements].view_as(parameter))
                offset += elements
