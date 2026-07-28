"""Frozen HY273-to-MotionStreamer272 bridge for T2M evaluation."""

from __future__ import annotations

import numpy as np


HY273_TO_MS272_PROTOCOL = "hy273_physical_to_motionstreamer272_v1"
DIM_HY273 = 273
DIM_MS272 = 272
NUM_JOINTS = 22
PARENTS = np.asarray(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=np.int64,
)


def _normalize(value: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), eps)


def kimodo_cont6d_to_matrix(value: np.ndarray) -> np.ndarray:
    """Decode Kimodo column-6D rotations into matrices."""

    value = np.asarray(value, dtype=np.float64)
    if value.shape[-1] != 6:
        raise ValueError(f"Expected Kimodo cont6d, got {value.shape}")
    x = _normalize(value[..., :3])
    y_raw = value[..., 3:]
    y = _normalize(y_raw - np.sum(x * y_raw, axis=-1, keepdims=True) * x)
    z = np.cross(x, y, axis=-1)
    return np.stack((x, y, z), axis=-1)


def matrix_to_motionstreamer_6d(matrix: np.ndarray) -> np.ndarray:
    """Encode MotionStreamer row-6D rotations (first two matrix rows)."""

    matrix = np.asarray(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices, got {matrix.shape}")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def reconstruct_hy273_global_positions(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float64)
    if motion.ndim != 2 or motion.shape[1] != DIM_HY273:
        raise ValueError(f"Expected [T,{DIM_HY273}], got {motion.shape}")
    positions = motion[:, 5:71].reshape(len(motion), NUM_JOINTS, 3).copy()
    positions[..., 0] += motion[:, None, 0]
    positions[..., 2] += motion[:, None, 2]
    return positions


def global_to_local_rotations(global_rotations: np.ndarray) -> np.ndarray:
    global_rotations = np.asarray(global_rotations, dtype=np.float64)
    if global_rotations.ndim != 4 or global_rotations.shape[1:] != (
        NUM_JOINTS,
        3,
        3,
    ):
        raise ValueError(f"Expected [T,{NUM_JOINTS},3,3], got {global_rotations.shape}")
    local = global_rotations.copy()
    for joint in range(1, NUM_JOINTS):
        parent = int(PARENTS[joint])
        local[:, joint] = (
            np.swapaxes(global_rotations[:, parent], -1, -2)
            @ global_rotations[:, joint]
        )
    return local


def world_to_heading_frame(root_rotation: np.ndarray) -> np.ndarray:
    """Return MotionStreamer's world-to-horizontal-heading matrices."""

    root_rotation = np.asarray(root_rotation, dtype=np.float64)
    forward = root_rotation @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    yaw = np.arctan2(forward[..., 0], forward[..., 2])
    cos = np.cos(yaw)
    sin = np.sin(yaw)
    local_to_world = np.zeros((*yaw.shape, 3, 3), dtype=np.float64)
    local_to_world[..., 0, 0] = cos
    local_to_world[..., 0, 2] = sin
    local_to_world[..., 1, 1] = 1.0
    local_to_world[..., 2, 0] = -sin
    local_to_world[..., 2, 2] = cos
    return np.swapaxes(local_to_world, -1, -2)


def hy273_to_motionstreamer272(motion: np.ndarray) -> np.ndarray:
    """Convert one raw physical HY273 sequence to raw MotionStreamer272.

    The bridge reconstructs physical positions and local skeletal rotations,
    removes the initial global yaw, then applies MotionStreamer's exact
    heading/local-root decomposition. It never interpolates or normalizes K273.
    """

    motion = np.asarray(motion)
    if motion.ndim != 2 or motion.shape[1] != DIM_HY273:
        raise ValueError(f"Expected [T,{DIM_HY273}], got {motion.shape}")
    if not len(motion) or not np.isfinite(motion).all():
        raise ValueError("HY273 input must be a non-empty finite sequence")
    motion64 = motion.astype(np.float64, copy=False)
    frames = len(motion64)
    global_positions = reconstruct_hy273_global_positions(motion64)
    root_positions = global_positions[:, 0]
    global_rotations = kimodo_cont6d_to_matrix(
        motion64[:, 71:203].reshape(frames, NUM_JOINTS, 6)
    )
    local_rotations = global_to_local_rotations(global_rotations)
    world_to_heading = world_to_heading_frame(global_rotations[:, 0])

    # Canonicalize the first global yaw while retaining all relative heading.
    relative_heading = world_to_heading @ world_to_heading[0].T
    heading_delta = np.empty_like(relative_heading)
    heading_delta[0] = np.eye(3, dtype=np.float64)
    if frames > 1:
        heading_delta[1:] = (
            relative_heading[1:]
            @ np.swapaxes(relative_heading[:-1], -1, -2)
        )

    horizontal_root = root_positions.copy()
    horizontal_root[:, 1] = 0.0
    centered_positions = global_positions - horizontal_root[:, None]
    local_positions = np.einsum(
        "tij,tkj->tki", world_to_heading, centered_positions
    )

    root_velocity_world = np.zeros((frames, 3), dtype=np.float64)
    if frames > 1:
        root_velocity_world[1:] = root_positions[1:] - root_positions[:-1]
    root_velocity_local = np.zeros_like(root_velocity_world)
    if frames > 1:
        root_velocity_local[1:] = np.einsum(
            "tij,tj->ti", world_to_heading[:-1], root_velocity_world[1:]
        )

    local_position_velocity = np.zeros_like(local_positions)
    if frames > 1:
        local_position_velocity[1:] = local_positions[1:] - local_positions[:-1]

    stored_rotations = local_rotations.copy()
    stored_rotations[:, 0] = world_to_heading @ global_rotations[:, 0]

    output = np.zeros((frames, DIM_MS272), dtype=np.float32)
    output[:, 0:2] = root_velocity_local[:, [0, 2]]
    output[:, 2:8] = matrix_to_motionstreamer_6d(heading_delta)
    output[:, 8:74] = local_positions.reshape(frames, 66)
    output[:, 74:140] = local_position_velocity.reshape(frames, 66)
    output[:, 140:272] = matrix_to_motionstreamer_6d(stored_rotations).reshape(
        frames, 132
    )
    if not np.isfinite(output).all():
        raise RuntimeError("HY273-to-MotionStreamer272 conversion produced non-finite data")
    return output
