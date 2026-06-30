"""Ensemble selector for the fixed-seed final: run each controller, report finish rate, pick best.

The level3 ensemble (v17 + v24, optionally + gate_search_v11) wins ~47-48% held-out by being
ANTI-CORRELATED — each controller wins different track layouts. The final is a FIXED seed, so the
deployment move is: run every candidate on that seed (>=20 runs) and submit the one that finishes
most. This script does exactly that.

    python scripts/ensemble_select.py --config level3.toml --seed 1234 --n_runs 20

Run it under the acados env (absolute ACADOS_SOURCE_DIR + DYLD_LIBRARY_PATH + TERA_PATH) since v17/
v24 need the MPCC solver. Pin threads (OMP/OPENBLAS/VECLIB/MKL=1) for a reproducible comparison.
"""

from __future__ import annotations

from pathlib import Path

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = ["controller_v17.py", "controller_v24.py", "gate_search_v11.py"]


def _finish_rate(controller_cls, cfg, seed: int, n_runs: int) -> int:
    """Return the number of finished runs for one controller on a fixed seed."""
    env = JaxToNumpy(
        gymnasium.make(
            cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim,
            sensor_range=cfg.env.sensor_range, control_mode=cfg.env.control_mode,
            track=cfg.env.track, disturbances=cfg.env.get("disturbances"),
            randomizations=cfg.env.get("randomizations"), seed=int(seed),
        )
    )
    finished = 0
    for _ in range(n_runs):
        obs, info = env.reset()
        controller = controller_cls(obs, info, cfg)
        while True:
            action = controller.compute_control(obs, info)
            obs, reward, term, trunc, done = env.step(action)
            if controller.step_callback(action, obs, reward, term, trunc, done) or term or trunc:
                break
        finished += int(obs["target_gate"] == -1)
    env.close()
    return finished


def main(config: str = "level3.toml", seed: int = 0, n_runs: int = 20,
         controllers: tuple[str, ...] = tuple(_DEFAULT)) -> None:
    """Evaluate each controller on the fixed seed and print the winner to submit."""
    results = {}
    for name in controllers:
        cfg = load_config(_ROOT / "config" / config)
        cfg.sim.render = False
        cfg.env.seed = int(seed)
        cls = load_controller(_ROOT / "lsy_drone_racing" / "control" / name)
        results[name] = _finish_rate(cls, cfg, seed, n_runs)
        print(f"  {name:24s} {results[name]:3d}/{n_runs} = {100 * results[name] // n_runs}%", flush=True)
    best = max(results, key=results.get)
    print(f"\nSUBMIT: {best}  ({results[best]}/{n_runs} = {100 * results[best] // n_runs}%)")


if __name__ == "__main__":
    fire.Fire(main)
