"""Track and quaternion math for controller_v6."""
# ruff: noqa: TC002

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def gate_rpy_from_quat(gate_quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert gate quaternions in [qx, qy, qz, qw] order to XYZ Euler angles."""
    return Rotation.from_quat(np.asarray(gate_quat, dtype=np.float64)).as_euler("xyz")


def horizontal_unit(vec: NDArray[np.float64], eps: float = 1e-9) -> NDArray[np.float64]:
    """Project a vector onto the XY plane and normalize it with a stable fallback."""
    out = np.array([vec[0], vec[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(out))
    if norm < eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return out / norm


def gate_axis(
    rpy: NDArray[np.float64], local_axis: int = 0, eps: float = 1e-9
) -> NDArray[np.float64]:
    """Return a gate-local axis projected onto the ground plane."""
    matrix = Rotation.from_euler("xyz", rpy).as_matrix()
    return horizontal_unit(matrix[:, local_axis], eps)


def gate_entry_exit(
    gate_position: NDArray[np.float64],
    gate_rpy: NDArray[np.float64],
    entry_distance: float = 0.25,
    exit_distance: float = 0.30,
    eps: float = 1e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create entry and exit points relative to the gate along its local x-axis."""
    axis = gate_axis(gate_rpy, local_axis=0, eps=eps)
    center = np.asarray(gate_position, dtype=np.float64)
    return center - entry_distance * axis, center + exit_distance * axis


def body_z_from_quat(quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytic body-z axis for [qx, qy, qz, qw] quaternion order."""
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    return np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dtype=np.float64,
    )
