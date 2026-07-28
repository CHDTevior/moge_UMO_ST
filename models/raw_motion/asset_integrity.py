"""Content-addressed training asset manifests for reproducible HY273 runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_asset_entry(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Training asset not found: {resolved}")
    return {
        "path": str(resolved),
        "size": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def write_asset_manifest(paths: Iterable[str | Path], output: str | Path) -> dict[str, Any]:
    unique = sorted({str(Path(path).expanduser().resolve()) for path in paths})
    manifest = {
        "format": "hy273_training_assets_v1",
        "files": [build_asset_entry(path) for path in unique],
    }
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_asset_manifest(
    manifest_path: str | Path,
    expected_manifest_sha256: str = "",
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training asset manifest not found: {path}")
    if expected_manifest_sha256:
        actual = sha256_file(path)
        expected = str(expected_manifest_sha256).strip().lower()
        if actual != expected:
            raise RuntimeError(
                f"Training asset manifest SHA256 mismatch: expected={expected}, actual={actual}, path={path}"
            )
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "hy273_training_assets_v1":
        raise RuntimeError(f"Unsupported training asset manifest format: {manifest.get('format')!r}")
    for entry in manifest.get("files", []):
        asset = Path(entry["path"])
        if not asset.is_file():
            raise FileNotFoundError(f"Pinned training asset is missing: {asset}")
        actual_size = int(asset.stat().st_size)
        if actual_size != int(entry["size"]):
            raise RuntimeError(
                f"Pinned training asset size mismatch: expected={entry['size']}, actual={actual_size}, path={asset}"
            )
        actual_sha = sha256_file(asset)
        if actual_sha != str(entry["sha256"]).lower():
            raise RuntimeError(
                f"Pinned training asset SHA256 mismatch: expected={entry['sha256']}, actual={actual_sha}, path={asset}"
            )
    return manifest
