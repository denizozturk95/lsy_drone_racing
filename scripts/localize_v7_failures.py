"""Instrumented localization of GateSearchV7 failures on the deploy-faithful scenario.

Runs config/level2_deploy.toml across many randomized-start seeds and, per gate, logs the
drone's closest approach to the gate-frame CENTER (3-D) and to nearby obstacle POLES (XY, since
poles are vertical columns). On a FAILED run it reports which gate index was the target when
the episode terminated, plus the closest-approach margins on the segment leading into that gate.

Run:
    cd /Users/denizozturk/IdeaProjects/lsy_drone_racing && \
        SCIPY_ARRAY_API=1 .pixi/envs/default/bin/python scripts/localize_v7_failures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control import gate_search_v7
from lsy_drone_racing.utils import load_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os

N_RUNS = int(os.environ.get("N_RUNS", "30"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
CONFIG_FILE = "level2_deploy.toml"


def run_episode(env, controller_cls, config, seed):
    """Run one episode, tracking per-gate closest approach to gate center and obstacle poles."""
    obs, info = env.reset()
    controller = controller_cls(obs, info, config)

    n_gates = len(np.asarray(obs["gates_pos"]))
    n_obs = len(np.asarray(obs["obstacles_pos"]).reshape(-1, 3))

    # Static track poses (deploy-faithful => true and constant from t=0)
    gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)  # (n_gates, 3)
    obs_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64).reshape(-1, 3)  # (n_obs, 3)

    # min 3-D distance from drone to each gate center, only while that gate is the active target
    min_gate_center = np.full(n_gates, np.inf)
    # min XY distance from drone to each obstacle pole, only while approaching the active gate
    # tracked per (target_gate, obstacle) so we can report what was near the failing gate
    min_obs_xy_per_target = np.full((n_gates, n_obs), np.inf)
    # max speed reached while each gate was the active target
    max_speed_per_target = np.zeros(n_gates)

    i = 0
    last_target = int(np.asarray(obs["target_gate"]).reshape(()))
    while True:
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        controller_finished = controller.step_callback(
            action, obs, reward, terminated, truncated, info
        )

        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        vel = np.asarray(obs["vel"], dtype=np.float64).reshape(-1)[:3]
        if 0 <= tg < n_gates:
            d_center = float(np.linalg.norm(pos - gate_pos[tg]))
            min_gate_center[tg] = min(min_gate_center[tg], d_center)
            dxy = np.linalg.norm(obs_pos[:, :2] - pos[:2], axis=1)
            min_obs_xy_per_target[tg] = np.minimum(min_obs_xy_per_target[tg], dxy)
            spd = float(np.linalg.norm(vel))
            max_speed_per_target[tg] = max(max_speed_per_target[tg], spd)
            last_target = tg

        if terminated or truncated or controller_finished:
            break
        i += 1

    curr_time = i / config.env.freq
    final_target = int(np.asarray(obs["target_gate"]).reshape(()))
    finished = final_target == -1
    # gate that was the target when the episode ended (the one we failed to pass on a fail)
    failed_gate = -1 if finished else last_target

    controller.episode_callback()
    controller.episode_reset()

    return {
        "seed": seed,
        "time": curr_time if finished else None,
        "finished": finished,
        "failed_gate": failed_gate,
        "min_gate_center": min_gate_center,
        "min_obs_xy_per_target": min_obs_xy_per_target,
        "max_speed_per_target": max_speed_per_target,
        "n_gates": n_gates,
        "n_obs": n_obs,
        "gate_pos": gate_pos,
        "obs_pos": obs_pos,
    }


def main():
    config = load_config(Path(__file__).parents[1] / "config" / CONFIG_FILE)
    config.sim.render = False
    controller_cls = gate_search_v7.GateSearchV7

    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    results = []
    for k in range(N_RUNS):
        s = SEED_OFFSET + k
        r = run_episode(env, controller_cls, config, seed=s)
        results.append(r)
        status = f"FINISH {r['time']:.2f}s" if r["finished"] else f"FAIL @ gate {r['failed_gate']}"
        logger.info(f"run {s:2d}: {status}")
    env.close()

    n_gates = results[0]["n_gates"]
    gate_pos = results[0]["gate_pos"]
    obs_pos = results[0]["obs_pos"]

    fails = [r for r in results if not r["finished"]]
    finishes = [r for r in results if r["finished"]]
    n = len(results)

    print("\n" + "=" * 78)
    print(f"SUMMARY: {len(finishes)}/{n} finished ({100*len(finishes)/n:.1f}%)")
    if finishes:
        ts = [r["time"] for r in finishes]
        print(f"  finish time: mean {np.mean(ts):.2f}s  min {np.min(ts):.2f}s  max {np.max(ts):.2f}s")

    print("\nTRACK gate centers:")
    for gi in range(n_gates):
        print(f"  gate {gi}: {np.round(gate_pos[gi], 3)}")
    print("obstacle poles (xy):")
    for oi in range(len(obs_pos)):
        print(f"  obs {oi}: {np.round(obs_pos[oi][:2], 3)}")

    print("\nPER-GATE closest approach to GATE CENTER (3-D, m), across ALL runs:")
    print("  (min over all runs = worst-case tightest pass; large center distance != clip,")
    print("   but the gate inner half-opening is ~0.2m, frame outer half-width ~0.36m)")
    for gi in range(n_gates):
        vals = np.array([r["min_gate_center"][gi] for r in results if np.isfinite(r["min_gate_center"][gi])])
        if len(vals):
            print(f"  gate {gi}: min {vals.min():.3f}  mean {vals.mean():.3f}  max {vals.max():.3f}")

    print("\nPER-GATE max speed while that gate was target (m/s):")
    for gi in range(n_gates):
        vals = np.array([r["max_speed_per_target"][gi] for r in results if r["max_speed_per_target"][gi] > 0])
        if len(vals):
            print(f"  gate {gi}: min {vals.min():.3f}  mean {vals.mean():.3f}  max {vals.max():.3f}")

    print("\nPER-GATE closest obstacle-pole approach (XY, m) while that gate was target:")
    for gi in range(n_gates):
        # nearest pole over all runs for this target gate
        per_run_min = []
        nearest_obs_idx = []
        for r in results:
            row = r["min_obs_xy_per_target"][gi]
            if np.all(~np.isfinite(row)):
                continue
            per_run_min.append(np.min(row))
            nearest_obs_idx.append(int(np.argmin(row)))
        if per_run_min:
            per_run_min = np.array(per_run_min)
            # which obstacle is most often the closest
            oi = max(set(nearest_obs_idx), key=nearest_obs_idx.count)
            print(f"  gate {gi}: nearest-pole min {per_run_min.min():.3f}  mean {per_run_min.mean():.3f}"
                  f"  (usually obs {oi} @ xy {np.round(obs_pos[oi][:2],2)})")

    if fails:
        print("\n" + "=" * 78)
        print(f"FAILED RUNS ({len(fails)}):")
        from collections import Counter
        fg = Counter(r["failed_gate"] for r in fails)
        print(f"  failing-gate histogram: {dict(fg)}")
        for r in fails:
            gi = r["failed_gate"]
            print(f"\n  seed {r['seed']}: terminated targeting gate {gi}")
            if 0 <= gi < n_gates:
                print(f"    closest to gate {gi} center: {r['min_gate_center'][gi]:.3f} m")
                row = r["min_obs_xy_per_target"][gi]
                if np.any(np.isfinite(row)):
                    oi = int(np.argmin(row))
                    print(f"    closest obstacle pole (xy): obs {oi} @ {r['min_obs_xy_per_target'][gi][oi]:.3f} m"
                          f" (pole xy {np.round(obs_pos[oi][:2],2)})")
                print(f"    max speed while targeting gate {gi}: {r['max_speed_per_target'][gi]:.3f} m/s")
    else:
        print("\nNo failures observed in this batch.")


if __name__ == "__main__":
    main()
