from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.raw_motion.hy273_normalizer import apply_yaw_rotation
from models.raw_motion.hy273_t2m_eval import (
    PARENTS,
    hy273_to_motionstreamer272,
)


CONVERTER_ROOT = Path("/mnt/afs/UMO_debug/motion272_to_hymotion")
SOURCE_272 = Path(
    "/mnt/afs/MotionMillion/272-dim-HumanML3D/motion_data/000000.npy"
)


def _matrix_to_kimodo_cont6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def _physical_k273_from_motionstreamer272(motion: np.ndarray) -> np.ndarray:
    if str(CONVERTER_ROOT) not in sys.path:
        sys.path.insert(0, str(CONVERTER_ROOT))
    from motion272_to_hymotion.geometry import recover_272

    recovered = recover_272(motion)
    local = recovered.rotations_matrix
    global_rot = local.copy()
    for joint in range(1, 22):
        global_rot[:, joint] = global_rot[:, int(PARENTS[joint])] @ local[:, joint]
    positions = recovered.positions
    root = recovered.translation
    k273 = np.zeros((len(motion), 273), dtype=np.float32)
    k273[:, :3] = root
    k273[:, 3] = 1.0
    local_pos = positions.copy()
    local_pos[..., 0] -= root[:, None, 0]
    local_pos[..., 2] -= root[:, None, 2]
    k273[:, 5:71] = local_pos.reshape(len(motion), 66)
    k273[:, 71:203] = _matrix_to_kimodo_cont6d(global_rot).reshape(
        len(motion), 132
    )
    return k273


@pytest.mark.skipif(not SOURCE_272.is_file(), reason="MotionStreamer272 audit asset absent")
def test_physical_motionstreamer_roundtrip_is_exact() -> None:
    source = np.load(SOURCE_272).astype(np.float32)
    k273 = _physical_k273_from_motionstreamer272(source)
    reconstructed = hy273_to_motionstreamer272(k273)
    np.testing.assert_allclose(reconstructed, source, rtol=2e-5, atol=3e-6)


@pytest.mark.skipif(not SOURCE_272.is_file(), reason="MotionStreamer272 audit asset absent")
def test_bridge_is_invariant_to_global_yaw_and_horizontal_translation() -> None:
    source = np.load(SOURCE_272).astype(np.float32)
    k273 = _physical_k273_from_motionstreamer272(source)
    transformed = torch.from_numpy(k273).unsqueeze(0)
    transformed = apply_yaw_rotation(transformed, torch.tensor([1.234]))
    transformed[..., 0] += 12.5
    transformed[..., 2] -= 7.25
    baseline = hy273_to_motionstreamer272(k273)
    candidate = hy273_to_motionstreamer272(transformed[0].numpy())
    np.testing.assert_allclose(candidate, baseline, rtol=2e-5, atol=4e-6)


def test_bridge_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Expected"):
        hy273_to_motionstreamer272(np.zeros((5, 272), dtype=np.float32))
    bad = np.zeros((5, 273), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        hy273_to_motionstreamer272(bad)
