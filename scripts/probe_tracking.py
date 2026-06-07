"""Determinism + lateral-bias probe for GateSearchV7 on the deploy-faithful scenario.

Answers two load-bearing questions before any v8/v9 work:

1. DETERMINISM (ILC/pre-bias premise): is the gate-crossing lateral error repeatable across
   random start/mass/dynamics seeds? -> low STD means the lag is geometry-driven, not noise.
2. DIRECTION (cheapest v8 lever): which signed side of each gate center does the drone actually
   cross? -> a consistent offset can be cancelled by pre-biasing the gate waypoint.

For each step it captures the controller's reference position (curve(t_eval)) via a monkeypatch
on attitude_action, plus the actual pos and active target gate. At each gate's CROSSING instant
(closest approach to the gate plane along the gate normal) it records, in the gate's local frame:
  - signed LATERAL offset of the drone from the gate center  (the pre-bias signal)
  - signed lateral TRACKING error (ref_pos - pos)            (the feedforward/phase-advance signal)
  - vertical offset
Aggregated per gate across seeds as MEAN +/- STD.

Run:
    cd /Users/denizozturk/IdeaProjects/lsy_drone_racing && \
        SCIPY_ARRAY_API=1 N_RUNS=80 .pixi/envs/default/bin/python scripts/probe_tracking.py
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202  (diagnostic script)

from __future__ import annotations

import logging
import os
from pathlib import Path

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import gate_search_v7
from lsy_drone_racing.utils import load_config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

N_RUNS = int(os.environ.get("N_RUNS", "80"))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", "0"))
CONFIG_FILE = "level2_deploy.toml"

# Per-step reference positions captured by the attitude_action monkeypatch (one entry per step).
_REF_LOG: list[np.ndarray] = []
_ORIG_ATTITUDE = gate_search_v7.attitude_action


def _patched_attitude(curve, t_eval, pos, *args, **kwargs):
    """Record the reference position the controller is tracking this step, then delegate."""
    try:
        _REF_LOG.append(np.asarray(curve(t_eval), dtype=np.float64).reshape(3))
    except Exception:
        _REF_LOG.append(np.full(3, np.nan))
    return _ORIG_ATTITUDE(curve, t_eval, pos, *args, **kwargs)


def run_episode(env, controller_cls, config, seed):
    """Run one episode, capturing per-step (target_gate, pos, ref_pos) for crossing analysis."""
    _REF_LOG.clear()
    obs, info = env.reset()
    controller = controller_cls(obs, info, config)

    n_gates = len(np.asarray(obs["gates_pos"]))
    gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
    gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64).reshape(n_gates, 4)

    steps_tg: list[int] = []
    steps_pos: list[np.ndarray] = []

    i = 0
    while True:
        action = controller.compute_control(obs, info)
        # target gate + pos at the moment compute_control acted (aligns with _REF_LOG[i])
        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        steps_tg.append(tg)
        steps_pos.append(pos)

        obs, reward, terminated, truncated, info = env.step(action)
        finished_ctrl = controller.step_callback(action, obs, reward, terminated, truncated, info)
        if terminated or truncated or finished_ctrl:
            break
        i += 1

    final_target = int(np.asarray(obs["target_gate"]).reshape(()))
    finished = final_target == -1
    controller.episode_callback()
    controller.episode_reset()

    # Align reference log with per-step records (drop any tail mismatch defensively).
    n = min(len(steps_tg), len(_REF_LOG))
    steps_tg = np.array(steps_tg[:n])
    steps_pos = np.array(steps_pos[:n])
    steps_ref = np.array(_REF_LOG[:n])

    # For each gate, find the crossing step = min |(pos - center) . normal| while it is the target,
    # then record local-frame offsets there.
    per_gate = {}
    for gi in range(n_gates):
        mask = steps_tg == gi
        if not np.any(mask):
            continue
        rot = Rotation.from_quat(gate_quat[gi]).as_matrix()
        normal = rot[:, 0]   # gate-local +x = travel/normal axis
        lateral = rot[:, 1]  # gate-local +y = lateral axis
        p = steps_pos[mask]
        r = steps_ref[mask]
        rel = p - gate_pos[gi]
        along = np.abs(rel @ normal)
        j = int(np.argmin(along))  # crossing instant
        rel_cross = rel[j]
        err_cross = r[j] - p[j]  # reference minus actual (tracking error)
        per_gate[gi] = {
            "lat_offset": float(rel_cross @ lateral),       # drone vs center, signed lateral
            "vert_offset": float(rel_cross[2]),
            "center_miss_xy": float(np.linalg.norm(rel_cross - (rel_cross @ normal) * normal)),
            "lat_track_err": float(err_cross @ lateral),    # ref vs drone, signed lateral
            "norm_track_err": float(err_cross @ normal),
        }

    failed_gate = -1 if finished else int(steps_tg[-1])
    return {"seed": seed, "finished": finished, "failed_gate": failed_gate,
            "per_gate": per_gate, "n_gates": n_gates, "gate_pos": gate_pos}


def _stats(vals):
    a = np.array(vals, dtype=np.float64)
    return a.mean(), a.std(), a.min(), a.max()


def main():
    """Run the probe over N seeds and print per-gate crossing offsets (mean +/- STD)."""
    config = load_config(Path(__file__).parents[1] / "config" / CONFIG_FILE)
    config.sim.render = False
    gate_search_v7.attitude_action = _patched_attitude  # install probe

    env = gymnasium.make(
        config.env.id, freq=config.env.freq, sim_config=config.sim,
        sensor_range=config.env.sensor_range, control_mode=config.env.control_mode,
        track=config.env.track, disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"), seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    results = []
    for k in range(N_RUNS):
        s = SEED_OFFSET + k
        r = run_episode(env, gate_search_v7.GateSearchV7, config, seed=s)
        results.append(r)
        tag = "FIN" if r["finished"] else f"FAIL@g{r['failed_gate']}"
        logger.info(f"run {s:2d}: {tag}")
    env.close()
    gate_search_v7.attitude_action = _ORIG_ATTITUDE  # restore

    n_gates = results[0]["n_gates"]
    n = len(results)
    fails = [r for r in results if not r["finished"]]

    print("\n" + "=" * 84)
    print(f"PROBE: {n - len(fails)}/{n} finished over seeds {SEED_OFFSET}..{SEED_OFFSET + n - 1}")
    if fails:
        from collections import Counter
        print(f"  failing-gate histogram: {dict(Counter(r['failed_gate'] for r in fails))}")

    print("\nPER-GATE CROSSING ANALYSIS (gate-local frame; sign matters for pre-bias direction)")
    print("  lat_offset  = signed lateral offset of DRONE from gate center  (mean +/- STD)")
    print("  lat_trk_err = signed lateral (ref - drone) tracking error      (mean +/- STD)")
    print("  STD is the determinism signal: small STD => repeatable => pre-bias/ILC valid.\n")
    for gi in range(n_gates):
        lo = [r["per_gate"][gi]["lat_offset"] for r in results if gi in r["per_gate"]]
        cm = [r["per_gate"][gi]["center_miss_xy"] for r in results if gi in r["per_gate"]]
        te = [r["per_gate"][gi]["lat_track_err"] for r in results if gi in r["per_gate"]]
        vo = [r["per_gate"][gi]["vert_offset"] for r in results if gi in r["per_gate"]]
        if not lo:
            continue
        lm, ls, lmn, lmx = _stats(lo)
        cmm, cms, _, cmx = _stats(cm)
        tm, ts, _, _ = _stats(te)
        vm, vs, _, _ = _stats(vo)
        print(f"  gate {gi}:")
        print(f"    lat_offset   mean {lm:+.4f}  STD {ls:.4f}  (min {lmn:+.4f}, max {lmx:+.4f})")
        print(f"    center_miss  mean {cmm:.4f}  STD {cms:.4f}  (max {cmx:.4f})  [half-ap ~0.20]")
        print(f"    lat_trk_err  mean {tm:+.4f}  STD {ts:.4f}")
        print(f"    vert_offset  mean {vm:+.4f}  STD {vs:.4f}")
        bias = -lm
        print(f"    => suggested lateral pre-bias for gate {gi}: {bias:+.4f} m along gate +y axis")


if __name__ == "__main__":
    main()
