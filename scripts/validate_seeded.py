"""Seeded validation harness for known-track controllers on the deploy-faithful scenario.

Unlike eval_deploy.py (which constructs the env with ONE seed and cannot pin per-run seeds),
this pins ``config.env.seed`` per run, so results are reproducible and directly comparable across
controllers and against the experiment harness. Reports finish rate, lap-time stats, and per-gate
crossing center-miss (the gate-clip margin against the ~0.20 m collision half-width).

This is the harness the TRUE-100% bar is measured on: e.g. 0 fails over 300 seeded runs.

    cd /Users/denizozturk/IdeaProjects/lsy_drone_racing && \
        SCIPY_ARRAY_API=1 CONTROLLER=gate_search_v8 N_RUNS=300 SEED_OFFSET=1000 \
        .pixi/envs/default/bin/python scripts/validate_seeded.py
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202  (diagnostic script)

from __future__ import annotations

import importlib
import logging
import os
import re
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation

from lsy_drone_racing.utils import load_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONTROLLER = os.environ.get("CONTROLLER", "gate_search_v8")
N_RUNS = int(os.environ.get("N_RUNS", "300"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "1000"))
CONFIG_FILE = os.environ.get("CONFIG_FILE", "level2_deploy.toml")


def _controller_cls(module_name):
    """Import lsy_drone_racing.control.<module_name> and return its GateSearch* class."""
    mod = importlib.import_module(f"lsy_drone_racing.control.{module_name}")
    # class name convention: GateSearchV<n> from gate_search_v<n>
    m = re.match(r"gate_search_v(\d+)", module_name)
    cls_name = f"GateSearchV{m.group(1)}" if m else None
    if cls_name and hasattr(mod, cls_name):
        return getattr(mod, cls_name)
    # fallback: first class whose name starts with GateSearch
    for name in dir(mod):
        if name.startswith("GateSearch"):
            return getattr(mod, name)
    raise RuntimeError(f"no GateSearch* class in {module_name}")


def run_episode(env, controller_cls, config):
    """Run one episode; return finish/time + per-gate crossing center-miss."""
    obs, info = env.reset()
    ctrl = controller_cls(obs, info, config)
    n_gates = len(np.asarray(obs["gates_pos"]))
    gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
    gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64).reshape(n_gates, 4)
    normals = [Rotation.from_quat(gate_quat[g]).as_matrix()[:, 0] for g in range(n_gates)]

    best_along = np.full(n_gates, np.inf)
    miss = np.full(n_gates, np.nan)
    i = 0
    while True:
        action = ctrl.compute_control(obs, info)
        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        if 0 <= tg < n_gates:
            rel = pos - gate_pos[tg]
            along = abs(float(rel @ normals[tg]))
            if along < best_along[tg]:
                best_along[tg] = along
                miss[tg] = float(np.linalg.norm(rel - (rel @ normals[tg]) * normals[tg]))
        obs, reward, terminated, truncated, info = env.step(action)
        finished_ctrl = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated or finished_ctrl:
            break
        i += 1

    finished = int(np.asarray(obs["target_gate"]).reshape(())) == -1
    failed_gate = -1 if finished else tg
    ctrl.episode_callback()
    ctrl.episode_reset()
    return {"finished": finished, "time": i / config.env.freq if finished else None,
            "miss": miss, "failed_gate": failed_gate, "n_gates": n_gates}


def main():
    """Run CONTROLLER over N seeded runs and print finish rate, lap stats, per-gate miss."""
    config = load_config(Path(__file__).parents[1] / "config" / CONFIG_FILE)
    config.sim.render = False
    controller_cls = _controller_cls(CONTROLLER)

    env = gymnasium.make(
        config.env.id, freq=config.env.freq, sim_config=config.sim,
        sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
        track=config.env.track, disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"), seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    results = []
    for k in range(N_RUNS):
        config.env.seed = SEED_OFFSET + k
        r = run_episode(env, controller_cls, config)
        results.append(r)
    env.close()

    n_gates = results[0]["n_gates"]
    fins = [r for r in results if r["finished"]]
    fails = [r for r in results if not r["finished"]]
    times = np.array([r["time"] for r in fins])

    print("=" * 84)
    last = SEED_OFFSET + N_RUNS - 1
    print(f"VALIDATE {CONTROLLER}  config={CONFIG_FILE}  seeds {SEED_OFFSET}..{last}")
    print(f"  finish: {len(fins)}/{N_RUNS} ({100 * len(fins) / N_RUNS:.2f}%)")
    if len(times):
        print(f"  lap: mean {times.mean():.3f}s  min {times.min():.3f}s  max {times.max():.3f}s")
    if fails:
        from collections import Counter
        fail_seeds = [SEED_OFFSET + i for i, r in enumerate(results) if not r["finished"]]
        print(f"  FAIL seeds: {fail_seeds}")
        print(f"  failing-gate histogram: {dict(Counter(r['failed_gate'] for r in fails))}")
    print("  per-gate crossing center-miss (m), collision half-width ~0.20:")
    for g in range(n_gates):
        vals = np.array([r["miss"][g] for r in results if np.isfinite(r["miss"][g])])
        if len(vals):
            print(f"    gate {g}: mean {vals.mean():.4f}  max {vals.max():.4f}")


if __name__ == "__main__":
    main()
