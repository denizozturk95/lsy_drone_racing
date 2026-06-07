"""Parametrized v8 cheap-lever experiment on the deploy-faithful scenario.

Runs GateSearchV7 with ONE reliability lever applied at RUNTIME (no file edits), measures
finish rate + the reversal-gate crossing center-miss + lap time over a seeded batch. Levers are
applied by monkeypatch so several can run as independent parallel processes without colliding:

  LEVER=baseline                          # unmodified v7 reference
  LEVER=prebias  GATE=2 BIAS=-0.08        # shift the gate the CONTROLLER sees by BIAS m along
                                          #   the gate +y (lateral) axis; env still scores real gate
  LEVER=ffscale  SCALE=0.8                # raise CommandSettings.feedforward_scale (default 0.6)
  LEVER=phaseadv LEAD=0.03                # evaluate the feedforward (vel/acc) LEAD s ahead
  LEVER=combo    GATE=2 BIAS=-0.08 SCALE=0.8 LEAD=0.03   # stack all three

Run:
    cd /Users/denizozturk/IdeaProjects/lsy_drone_racing && \
        SCIPY_ARRAY_API=1 N_RUNS=120 LEVER=prebias GATE=2 BIAS=-0.08 \
        .pixi/envs/default/bin/python scripts/v8_experiment.py
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202  (diagnostic script)

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import gate_search_v7
from lsy_drone_racing.control.online_planner import attitude as attitude_mod
from lsy_drone_racing.utils import load_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

N_RUNS = int(os.environ.get("N_RUNS", "120"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
CONFIG_FILE = "level2_deploy.toml"

LEVER = os.environ.get("LEVER", "baseline")
GATE = int(os.environ.get("GATE", "2"))
BIAS = float(os.environ.get("BIAS", "0.0"))
SCALE = float(os.environ.get("SCALE", "0.6"))
LEAD = float(os.environ.get("LEAD", "0.0"))

_ORIG_ATTITUDE = gate_search_v7.attitude_action


def _phase_advanced_attitude(lead):
    """Build an attitude_action that leads the feedforward (vel/acc) terms by `lead` seconds."""

    def fn(reference, t_eval, pos, vel, quat, feedback, dt, mass, gravity, settings):
        t_ff = t_eval + lead
        ref_pos = np.asarray(reference(t_eval), dtype=np.float64).reshape(3)
        ref_vel = np.asarray(reference.derivative(1)(t_ff), dtype=np.float64).reshape(3)
        ref_acc = np.asarray(reference.derivative(2)(t_ff), dtype=np.float64).reshape(3)
        ref_acc = attitude_mod._limit_lateral(ref_acc, settings.lateral_accel_limit)
        force = feedback.update(ref_pos - pos, ref_vel - vel, dt)
        thrust_vector = force + settings.feedforward_scale * mass * ref_acc
        thrust_vector[2] += mass * gravity
        action = attitude_mod._vector_to_attitude(thrust_vector, quat, settings)
        return action, attitude_mod.TrackingSample(ref_pos, ref_vel, ref_acc, force, thrust_vector)

    return fn


def apply_lever():
    """Install the global (module-level) part of the lever. Returns a per-controller hook."""
    if LEVER in ("phaseadv", "combo") and LEAD != 0.0:
        gate_search_v7.attitude_action = _phase_advanced_attitude(LEAD)
    else:
        gate_search_v7.attitude_action = _ORIG_ATTITUDE

    def per_controller(ctrl, gate_quat):
        if LEVER in ("ffscale", "combo") and SCALE != 0.6:
            ctrl._settings = dataclasses.replace(
                ctrl._settings,
                command=dataclasses.replace(ctrl._settings.command, feedforward_scale=SCALE),
            )
        bias_vec = None
        if LEVER in ("prebias", "combo") and BIAS != 0.0:
            lateral = Rotation.from_quat(gate_quat[GATE]).as_matrix()[:, 1]
            bias_vec = BIAS * lateral
        return bias_vec

    return per_controller


def run_episode(env, config, per_controller):
    """Run one episode; return finish/time + reversal-gate crossing center-miss."""
    obs, info = env.reset()
    ctrl = gate_search_v7.GateSearchV7(obs, info, config)
    n_gates = len(np.asarray(obs["gates_pos"]))
    gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
    gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64).reshape(n_gates, 4)
    bias_vec = per_controller(ctrl, gate_quat)
    rot = Rotation.from_quat(gate_quat[GATE]).as_matrix()
    normal = rot[:, 0]

    best_along = np.inf
    miss_at_cross = np.nan
    i = 0
    while True:
        step_obs = obs
        if bias_vec is not None:
            step_obs = dict(obs)
            gp = gate_pos.copy()
            gp[GATE] = gp[GATE] + bias_vec
            step_obs["gates_pos"] = gp
        action = ctrl.compute_control(step_obs, info)

        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        if tg == GATE:
            rel = pos - gate_pos[GATE]
            along = abs(float(rel @ normal))
            if along < best_along:
                best_along = along
                miss_at_cross = float(np.linalg.norm(rel - (rel @ normal) * normal))

        obs, reward, terminated, truncated, info = env.step(action)
        finished_ctrl = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated or finished_ctrl:
            break
        i += 1

    finished = int(np.asarray(obs["target_gate"]).reshape(())) == -1
    ctrl.episode_callback()
    ctrl.episode_reset()
    return {"finished": finished, "time": i / config.env.freq if finished else None,
            "gate_miss": miss_at_cross}


def main():
    """Run the configured lever over N seeds and print finish rate + gate-miss + lap time."""
    config = load_config(Path(__file__).parents[1] / "config" / CONFIG_FILE)
    config.sim.render = False
    per_controller = apply_lever()

    env = gymnasium.make(
        config.env.id, freq=config.env.freq, sim_config=config.sim,
        sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
        track=config.env.track, disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"), seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    results = []
    for k in range(N_RUNS):
        # per-run seed for reproducibility/comparability across levers
        config.env.seed = SEED_OFFSET + k
        r = run_episode(env, config, per_controller)
        results.append(r)
    env.close()
    gate_search_v7.attitude_action = _ORIG_ATTITUDE

    fins = [r for r in results if r["finished"]]
    misses = np.array([r["gate_miss"] for r in results if np.isfinite(r["gate_miss"])])
    times = np.array([r["time"] for r in fins])
    desc = f"LEVER={LEVER} GATE={GATE} BIAS={BIAS} SCALE={SCALE} LEAD={LEAD}"
    print("=" * 84)
    print(desc)
    print(f"  finish: {len(fins)}/{len(results)} ({100 * len(fins) / len(results):.1f}%)"
          f"  seeds {SEED_OFFSET}..{SEED_OFFSET + len(results) - 1}")
    if len(times):
        print(f"  lap: mean {times.mean():.2f}s  min {times.min():.2f}s  max {times.max():.2f}s")
    if len(misses):
        print(f"  gate {GATE} center-miss: mean {misses.mean():.4f}  max {misses.max():.4f}"
              f"  (collision half-width ~0.20m)")


if __name__ == "__main__":
    main()
