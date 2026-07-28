"""Crash-durable writers for production evaluation evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def fsync_directory(path: str | Path) -> None:
    """Persist directory-entry updates after an atomic rename."""

    directory = Path(path).expanduser().resolve()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_replace(source: str | Path, target: str | Path) -> None:
    destination = Path(target).expanduser().resolve()
    os.replace(Path(source), destination)
    fsync_directory(destination.parent)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        durable_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        durable_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
