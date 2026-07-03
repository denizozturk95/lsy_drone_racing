"""Attitude action generation for controller_v6."""
# ruff: noqa: TC001, TC002

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.common_controller_v2.feedback import CascadedPid
from lsy_drone_racing.control.common_controller_v2.geometry import body_z_from_quat
from lsy_drone_racing.control.common_controller_v2.settings import CommandSettings
from lsy_drone_racing.control.common_controller_v2.trajectory import ReferenceCurve


@dataclass(frozen=True)
class TrackingSample:
    """Reference values and feedback terms used for diagnostics."""

    pos_ref: NDArray[np.float64]
    vel_ref: NDArray[np.float64]
    accel_ref: NDArray[np.float64]
    feedback_force: NDArray[np.float64]
    thrust_vector: NDArray[np.float64]


def attitude_action(
    reference: ReferenceCurve,
    t_eval: float,
    pos: NDArray[np.float64],
    vel: NDArray[np.float64],
    quat: NDArray[np.float64],
    feedback: CascadedPid,
    dt: float,
    mass: float,
    gravity: float,
    settings: CommandSettings,
) -> tuple[NDArray[np.float32], TrackingSample]:
    """Track a position reference and return [roll, pitch, yaw, collective_thrust]."""
    ref_pos = np.asarray(reference(t_eval), dtype=np.float64).reshape(3)
    ref_vel = np.asarray(reference.derivative(1)(t_eval), dtype=np.float64).reshape(3)
    ref_acc = np.asarray(reference.derivative(2)(t_eval), dtype=np.float64).reshape(3)
    ref_acc = _limit_lateral(ref_acc, settings.lateral_accel_limit)

    force = feedback.update(ref_pos - pos, ref_vel - vel, dt)
    thrust_vector = force + settings.feedforward_scale * mass * ref_acc
    thrust_vector[2] += mass * gravity
    action = _vector_to_attitude(thrust_vector, quat, settings)
    return action, TrackingSample(ref_pos, ref_vel, ref_acc, force, thrust_vector)


def _limit_lateral(accel: NDArray[np.float64], limit: float) -> NDArray[np.float64]:
    clipped = np.asarray(accel, dtype=np.float64).reshape(3).copy()
    horizontal = float(np.linalg.norm(clipped[:2]))
    if horizontal > limit > 0.0:
        clipped[:2] *= limit / horizontal
    return clipped


def _vector_to_attitude(
    thrust_vector: NDArray[np.float64], quat: NDArray[np.float64], settings: CommandSettings
) -> NDArray[np.float32]:
    current_body_z = body_z_from_quat(quat)
    collective = float(np.dot(thrust_vector, current_body_z))
    norm = float(np.linalg.norm(thrust_vector))
    if norm < settings.norm_eps:
        z_des = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_des = thrust_vector / norm

    # Yaw policy: align the desired body x-axis with the world x-axis as much as possible.
    world_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_des = np.cross(z_des, world_x)
    y_norm = float(np.linalg.norm(y_des))
    if y_norm < settings.norm_eps:
        world_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_des = np.cross(z_des, world_x)
        y_norm = float(np.linalg.norm(y_des))
    y_des /= y_norm + settings.norm_eps
    x_des = np.cross(y_des, z_des)
    matrix = np.column_stack((x_des, y_des, z_des))
    euler = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
    action = np.array([euler[0], euler[1], euler[2], collective], dtype=np.float64)
    if settings.clip_actions:
        action[:3] = np.clip(action[:3], -settings.euler_limit, settings.euler_limit)
        action[3] = np.clip(action[3], settings.thrust_min, settings.thrust_max)
    return action.astype(np.float32)
