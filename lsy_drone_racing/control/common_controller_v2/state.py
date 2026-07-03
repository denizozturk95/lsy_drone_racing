"""Observation parsing routines for the controller_v6 controller."""
# ruff: noqa: TC002

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DroneObservation:
    """Single-drone observation frame using [qx, qy, qz, qw] quaternion order."""

    target_gate: int
    gate_pos: NDArray[np.float64]
    gate_quat: NDArray[np.float64]
    obstacles_pos: NDArray[np.float64]
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    quat: NDArray[np.float64]


def scalar_gate_index(value: object) -> int:
    """Convert the stored target-gate field into a single integer."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"target_gate must contain exactly one value, got shape {arr.shape}")
    return int(arr.reshape(()))


def as_vector(value: object, length: int, name: str) -> NDArray[np.float64]:
    """Return a fixed-length floating-point vector."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}")
    return arr


def parse_observation(obs: dict[str, NDArray[np.floating]]) -> DroneObservation:
    """Parse and validate the observation keys required by the controller."""
    gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
    gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
    obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64).reshape(-1, 3)
    if gate_pos.ndim != 2 or gate_pos.shape[1] != 3:
        raise ValueError(f"gates_pos must have shape (n, 3), got {gate_pos.shape}")
    if gate_quat.ndim != 2 or gate_quat.shape[1] != 4:
        raise ValueError(f"gates_quat must have shape (n, 4), got {gate_quat.shape}")
    if len(gate_pos) < 1 or len(gate_pos) != len(gate_quat):
        raise ValueError("controller_v6 expects at least one gate with matching quaternions")

    return DroneObservation(
        target_gate=scalar_gate_index(obs["target_gate"]),
        gate_pos=gate_pos.copy(),
        gate_quat=gate_quat.copy(),
        obstacles_pos=obstacles_pos.copy(),
        pos=as_vector(obs["pos"], 3, "pos"),
        vel=as_vector(obs["vel"], 3, "vel"),
        quat=as_vector(obs["quat"], 4, "quat"),
    )
