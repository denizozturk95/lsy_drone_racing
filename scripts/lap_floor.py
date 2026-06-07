"""Lap-time-floor decomposition for GateSearchV7 on the deploy-faithful scenario.

Bounds the speed-side ROI of any future tracking work (v9/v10): reports how the ~10.7s lap
splits across the four gate segments and the final HOME descent, with each segment's path
length and average speed. Segments where average speed is far below the v_cruise cap (the
U-turn) are where time concentrates and where tracking improvements *could* buy time; segments
already near the cap are near-irreducible.

Run:
    cd /Users/denizozturk/IdeaProjects/lsy_drone_racing && \
        SCIPY_ARRAY_API=1 N_RUNS=10 .pixi/envs/default/bin/python scripts/lap_floor.py
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202  (diagnostic script)

from __future__ import annotations

import logging
import os
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control import gate_search_v7
from lsy_drone_racing.utils import load_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

N_RUNS = int(os.environ.get("N_RUNS", "10"))
CONFIG_FILE = "level2_deploy.toml"


def run_episode(env, config):
    """Return per-target-gate (steps, path length, max speed) for one finished episode."""
    obs, info = env.reset()
    ctrl = gate_search_v7.GateSearchV7(obs, info, config)
    n_gates = len(np.asarray(obs["gates_pos"]))
    steps = np.zeros(n_gates + 1)  # index n_gates = HOME (target == -1)
    path = np.zeros(n_gates + 1)
    vmax = np.zeros(n_gates + 1)
    prev_pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]

    i = 0
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        finished_ctrl = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        vel = float(np.linalg.norm(np.asarray(obs["vel"], dtype=np.float64).reshape(-1)[:3]))
        idx = n_gates if tg == -1 else tg
        steps[idx] += 1
        path[idx] += float(np.linalg.norm(pos - prev_pos))
        vmax[idx] = max(vmax[idx], vel)
        prev_pos = pos
        if terminated or truncated or finished_ctrl:
            break
        i += 1

    finished = int(np.asarray(obs["target_gate"]).reshape(())) == -1
    ctrl.episode_callback()
    ctrl.episode_reset()
    return {"finished": finished, "time": i / config.env.freq, "steps": steps,
            "path": path, "vmax": vmax, "n_gates": n_gates}


def main():
    """Run N finished episodes and print the per-segment lap-time decomposition."""
    config = load_config(Path(__file__).parents[1] / "config" / CONFIG_FILE)
    config.sim.render = False
    env = gymnasium.make(
        config.env.id, freq=config.env.freq, sim_config=config.sim,
        sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
        track=config.env.track, disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"), seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    runs = []
    for k in range(N_RUNS):
        config.env.seed = k
        r = run_episode(env, config)
        if r["finished"]:
            runs.append(r)
    env.close()

    n_gates = runs[0]["n_gates"]
    freq = config.env.freq
    steps = np.mean([r["steps"] for r in runs], axis=0)
    path = np.mean([r["path"] for r in runs], axis=0)
    vmax = np.mean([r["vmax"] for r in runs], axis=0)
    times = steps / freq
    total = times.sum()

    print("=" * 78)
    print(f"LAP-FLOOR over {len(runs)} finished runs   total {total:.2f}s")
    print(f"{'segment':<14}{'time(s)':>9}{'%lap':>7}{'path(m)':>9}{'avg v':>8}{'max v':>8}")
    labels = [f"-> gate {g}" for g in range(n_gates)] + ["-> HOME"]
    for idx in range(n_gates + 1):
        if steps[idx] < 0.5:
            continue
        avg_v = path[idx] / max(times[idx], 1e-9)
        print(f"{labels[idx]:<14}{times[idx]:>9.2f}{100*times[idx]/total:>7.1f}"
              f"{path[idx]:>9.2f}{avg_v:>8.2f}{vmax[idx]:>8.2f}")
    race = times[:n_gates].sum()
    print(f"\n  race (gates only, excl HOME): {race:.2f}s")
    print(f"  v_cruise cap = {gate_search_v7._V_CRUISE} (peri) / {gate_search_v7._V_CRUISE_INTER}"
          f" (inter), VMAX = {gate_search_v7._VMAX}")
    print("  segments with avg v << cap = where time-allocation slows = the recoverable budget.")


if __name__ == "__main__":
    main()
