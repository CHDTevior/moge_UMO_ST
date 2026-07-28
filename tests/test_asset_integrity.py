from pathlib import Path

import pytest

from models.raw_motion.asset_integrity import (
    sha256_file,
    verify_asset_manifest,
    write_asset_manifest,
)


def test_asset_manifest_detects_content_tampering(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest = tmp_path / "assets.json"
    write_asset_manifest([second, first], manifest)
    manifest_sha = sha256_file(manifest)
    verified = verify_asset_manifest(manifest, manifest_sha)
    assert len(verified["files"]) == 2

    first.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_asset_manifest(manifest, manifest_sha)
