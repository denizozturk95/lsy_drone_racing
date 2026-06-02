"""cascaded position velocity feedback."""
# ruff: noqa: TC001, TC002

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from lsy_drone_racing.control.online_planner.settings import FeedbackProfile, FeedbackSettings


@dataclass(frozen=True)
class CascadeGains:
    """resolved gains for the outer position and inner velocity loops."""

    outer_kp: NDArray[np.float64]
    outer_ki: NDArray[np.float64]
    outer_i_limit: NDArray[np.float64]
    outer_output_limit: NDArray[np.float64]
    inner_kp: NDArray[np.float64]
    inner_ki: NDArray[np.float64]
    inner_kd: NDArray[np.float64]
    inner_i_limit: NDArray[np.float64]
    output_limit: NDArray[np.float64]
    derivative_tau: NDArray[np.float64]


def resolve_gains(profile: FeedbackProfile, settings: FeedbackSettings) -> CascadeGains:
    """convert the documented pid table into cascaded gains."""
    inner_kp = np.maximum(1.04 * profile.kd, settings.eps)
    return CascadeGains(
        outer_kp=0.98 * profile.kp / inner_kp,
        outer_ki=0.90 * profile.ki / inner_kp,
        outer_i_limit=profile.outer_i_limit,
        outer_output_limit=settings.outer_clamp,
        inner_kp=inner_kp,
        inner_ki=0.010 * inner_kp,
        inner_kd=0.012 * inner_kp,
        inner_i_limit=settings.inner_i_limit,
        output_limit=settings.output_clamp,
        derivative_tau=settings.derivative_tau,
    )


class CascadedPid:
    """stateful cascaded controller with anti-windup and derivative filtering."""

    def __init__(self, settings: FeedbackSettings):
        """create a cascaded controller from a single resolved gain profile."""
        self._settings = settings
        self._gains = resolve_gains(settings.profile, settings)
        self._outer_integral = np.zeros(3, dtype=np.float64)
        self._inner_integral = np.zeros(3, dtype=np.float64)
        self._previous_velocity_error: NDArray[np.float64] | None = None
        self._filtered_derivative = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        """clear integrators and derivative memory."""
        self._outer_integral.fill(0.0)
        self._inner_integral.fill(0.0)
        self._previous_velocity_error = None
        self._filtered_derivative.fill(0.0)

    def update(
        self, pos_error: NDArray[np.float64], vel_error: NDArray[np.float64], dt: float
    ) -> NDArray[np.float64]:
        """return a clipped force-like feedback command."""
        pos_error = np.asarray(pos_error, dtype=np.float64).reshape(3)
        vel_error = np.asarray(vel_error, dtype=np.float64).reshape(3)
        dt = max(float(dt), self._settings.eps)
        gains = self._gains

        previous_outer = self._outer_integral.copy()
        self._outer_integral = np.clip(
            self._outer_integral + pos_error * dt, -gains.outer_i_limit, gains.outer_i_limit
        )
        raw_outer = gains.outer_kp * pos_error + gains.outer_ki * self._outer_integral
        saturated_outer = np.clip(raw_outer, -gains.outer_output_limit, gains.outer_output_limit)
        outer_winding = (np.abs(raw_outer) > gains.outer_output_limit) & (
            np.sign(raw_outer) == np.sign(pos_error)
        )
        self._outer_integral[outer_winding] = previous_outer[outer_winding]
        velocity_request = np.where(
            outer_winding,
            np.clip(raw_outer, -gains.outer_output_limit, gains.outer_output_limit),
            saturated_outer,
        )

        velocity_error = vel_error + velocity_request
        if self._previous_velocity_error is None:
            raw_derivative = np.zeros(3, dtype=np.float64)
        else:
            raw_derivative = (velocity_error - self._previous_velocity_error) / dt
        self._previous_velocity_error = velocity_error.copy()
        alpha = dt / (np.maximum(gains.derivative_tau, 0.0) + dt)
        self._filtered_derivative += alpha * (raw_derivative - self._filtered_derivative)

        previous_inner = self._inner_integral.copy()
        self._inner_integral = np.clip(
            self._inner_integral + velocity_error * dt, -gains.inner_i_limit, gains.inner_i_limit
        )
        raw_command = (
            gains.inner_kp * velocity_error
            + gains.inner_ki * self._inner_integral
            + gains.inner_kd * self._filtered_derivative
        )
        limited = np.clip(raw_command, -gains.output_limit, gains.output_limit)
        inner_winding = (np.abs(raw_command) > gains.output_limit) & (
            np.sign(raw_command) == np.sign(velocity_error)
        )
        self._inner_integral[inner_winding] = previous_inner[inner_winding]
        return limited
