#!/usr/bin/env python3
"""Fail-closed physical GPU inventory and idleness validation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Callable, Sequence


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run_csv(command: list[str], runner: Runner) -> list[list[str]]:
    result = runner(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"GPU query failed ({result.returncode}): {' '.join(command)}: "
            f"{result.stderr.strip()}"
        )
    return [
        [field.strip() for field in row]
        for row in csv.reader(result.stdout.splitlines())
        if row and any(field.strip() for field in row)
    ]


def _parse_int(value: str, *, field: str, selected_id: str) -> int:
    try:
        return int(float(value))
    except ValueError as exc:
        raise RuntimeError(
            f"GPU {selected_id} returned non-numeric {field}: {value!r}"
        ) from exc


def query_gpu(
    selected_id: str,
    *,
    nvidia_smi: str = "nvidia-smi",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    rows = _run_csv(
        [
            nvidia_smi,
            f"--id={selected_id}",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        runner,
    )
    if len(rows) != 1 or len(rows[0]) != 7:
        raise RuntimeError(
            f"GPU selector {selected_id!r} resolved to {len(rows)} rows; expected one physical GPU"
        )
    index, uuid, bus_id, name, memory_used, memory_total, utilization = rows[0]
    if not uuid or not bus_id:
        raise RuntimeError(f"GPU selector {selected_id!r} has no canonical UUID/bus ID")

    process_rows = _run_csv(
        [
            nvidia_smi,
            f"--id={selected_id}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        runner,
    )
    compute_pids: list[int] = []
    for row in process_rows:
        value = row[0].strip()
        if value.isdigit():
            compute_pids.append(int(value))
        elif value.lower() not in {"", "no running processes found"}:
            raise RuntimeError(
                f"GPU {selected_id} returned an unrecognized process row: {row}"
            )

    return {
        "selected_id": selected_id,
        "index": _parse_int(index, field="index", selected_id=selected_id),
        "uuid": uuid,
        "pci_bus_id": bus_id.lower(),
        "name": name,
        "memory_used_mib": _parse_int(
            memory_used, field="memory.used", selected_id=selected_id
        ),
        "memory_total_mib": _parse_int(
            memory_total, field="memory.total", selected_id=selected_id
        ),
        "utilization_percent": _parse_int(
            utilization, field="utilization.gpu", selected_id=selected_id
        ),
        "compute_pids": sorted(set(compute_pids)),
    }


def collect_inventory(
    selectors: Sequence[str],
    *,
    phase: str,
    max_memory_used_mib: int = 499,
    max_utilization_percent: int = 5,
    nvidia_smi: str = "nvidia-smi",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    normalized = [selector.strip() for selector in selectors if selector.strip()]
    if not normalized:
        raise RuntimeError("No GPU selectors were provided")
    devices = [
        query_gpu(selector, nvidia_smi=nvidia_smi, runner=runner)
        for selector in normalized
    ]

    uuids = [device["uuid"] for device in devices]
    buses = [device["pci_bus_id"] for device in devices]
    if len(set(uuids)) != len(uuids):
        raise RuntimeError(
            f"GPU selectors alias the same physical UUID: selectors={normalized}, uuids={uuids}"
        )
    if len(set(buses)) != len(buses):
        raise RuntimeError(
            f"GPU selectors alias the same PCI device: selectors={normalized}, buses={buses}"
        )

    signatures = {
        (device["name"], int(device["memory_total_mib"])) for device in devices
    }
    if len(signatures) != 1:
        raise RuntimeError(
            "Selected GPUs are not homogeneous by model and total memory: "
            f"{sorted(signatures)}"
        )

    busy = [
        device
        for device in devices
        if int(device["memory_used_mib"]) > max_memory_used_mib
        or int(device["utilization_percent"]) > max_utilization_percent
        or bool(device["compute_pids"])
    ]
    if busy:
        raise RuntimeError(f"Selected GPUs are not idle at phase {phase!r}: {busy}")

    name, memory_total_mib = next(iter(signatures))
    return {
        "format": "hy273_gpu_inventory_v1",
        "phase": phase,
        "host": socket.gethostname(),
        "checked_unix": time.time(),
        "pid": os.getpid(),
        "requested_selectors": normalized,
        "physical_device_count": len(devices),
        "homogeneous_signature": {
            "name": name,
            "memory_total_mib": memory_total_mib,
        },
        "idle_thresholds": {
            "max_memory_used_mib": max_memory_used_mib,
            "max_utilization_percent": max_utilization_percent,
            "require_no_compute_pids": True,
        },
        "devices": devices,
        "passed": True,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-list", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-memory-used-mib", type=int, default=499)
    parser.add_argument("--max-utilization-percent", type=int, default=5)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = collect_inventory(
        args.gpu_list.split(","),
        phase=args.phase,
        max_memory_used_mib=args.max_memory_used_mib,
        max_utilization_percent=args.max_utilization_percent,
        nvidia_smi=args.nvidia_smi,
    )
    output = Path(args.output).expanduser().resolve()
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "passed": True,
                "phase": args.phase,
                "physical_device_count": payload["physical_device_count"],
                "signature": payload["homogeneous_signature"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
