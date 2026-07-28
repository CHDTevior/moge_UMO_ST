from __future__ import annotations

import subprocess

import pytest

from tools.validate_gpu_inventory import collect_inventory


def fake_runner(devices: dict[str, tuple[str, ...]], pids: dict[str, str] | None = None):
    pids = pids or {}

    def run(command, *, check, capture_output, text):
        assert check is False
        assert capture_output is True
        assert text is True
        selected = next(item.split("=", 1)[1] for item in command if item.startswith("--id="))
        if any(item.startswith("--query-gpu=") for item in command):
            row = devices[selected]
            stdout = ", ".join(row) + "\n"
        else:
            stdout = pids.get(selected, "")
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return run


def device(
    index: int,
    uuid: str,
    bus: str,
    *,
    name: str = "NVIDIA A100-SXM4-80GB",
    used: int = 0,
    total: int = 81920,
    utilization: int = 0,
) -> tuple[str, ...]:
    return (
        str(index),
        uuid,
        bus,
        name,
        str(used),
        str(total),
        str(utilization),
    )


def test_gpu_inventory_accepts_distinct_homogeneous_idle_devices() -> None:
    devices = {
        "0": device(0, "GPU-a", "0000:01:00.0"),
        "GPU-b": device(1, "GPU-b", "0000:02:00.0"),
    }
    inventory = collect_inventory(
        ["0", "GPU-b"], phase="before_launch", runner=fake_runner(devices)
    )
    assert inventory["passed"] is True
    assert inventory["physical_device_count"] == 2
    assert [row["uuid"] for row in inventory["devices"]] == ["GPU-a", "GPU-b"]


def test_gpu_inventory_rejects_selector_aliases_for_same_physical_gpu() -> None:
    same = device(0, "GPU-a", "0000:01:00.0")
    devices = {"0": same, "GPU-a": same}
    with pytest.raises(RuntimeError, match="same physical UUID"):
        collect_inventory(
            ["0", "GPU-a"], phase="before_launch", runner=fake_runner(devices)
        )


def test_gpu_inventory_rejects_memory_mismatch_or_busy_device() -> None:
    mismatch = {
        "0": device(0, "GPU-a", "0000:01:00.0"),
        "1": device(1, "GPU-b", "0000:02:00.0", total=40960),
    }
    with pytest.raises(RuntimeError, match="not homogeneous"):
        collect_inventory(
            ["0", "1"], phase="before_launch", runner=fake_runner(mismatch)
        )

    busy = {
        "0": device(0, "GPU-a", "0000:01:00.0"),
        "1": device(1, "GPU-b", "0000:02:00.0"),
    }
    with pytest.raises(RuntimeError, match="not idle"):
        collect_inventory(
            ["0", "1"],
            phase="before_launch",
            runner=fake_runner(busy, pids={"1": "12345\n"}),
        )
