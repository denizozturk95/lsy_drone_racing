r"""Ablate GateSearchV10 factors on Level 3 to find what raises the finish rate.

Crash decomposition showed Level-3 failures are obstacle-pole hits in the gate corridor + en-route
crashes on the cross-arena search->navigate transit, not pure altitude clips. This toggles the
high-leverage factors one at a time (single-change from the current v10) over a seed batch, reusing
one env, and reports finish % + crash breakdown so the best config can be picked in one run.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/v10_l3_ablation.py --n_runs 12
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, E501, T201, C901

from __future__ import annotations

import ctypes
import dataclasses
import os
import sys
from ctypes import wintypes
from pathlib import Path


def _activate_env_dlls():
    if os.name != "nt":
        return
    root = os.path.dirname(os.path.abspath(sys.executable))
    dirs = [root, *(os.path.join(root, *p) for p in (("Library", "bin"), ("Scripts",), ("DLLs",)))]
    existing = [d for d in dirs if os.path.isdir(d)]
    os.environ["PATH"] = os.pathsep.join(existing) + os.pathsep + os.environ.get("PATH", "")
    for d in existing:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass


def _patch_mujoco():
    if os.name != "nt":
        return
    import mujoco
    orig = mujoco.MjSpec.from_file

    def short(p):
        p = os.path.abspath(str(p))
        if p.isascii():
            return p
        fn = ctypes.windll.kernel32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        buf = ctypes.create_unicode_buffer(1024)
        return buf.value if fn(p, buf, 1024) else p

    mujoco.MjSpec.from_file = staticmethod(lambda p, *a, **k: orig(short(p), *a, **k))


_activate_env_dlls()
_patch_mujoco()

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402

from lsy_drone_racing.control import gate_search_v10 as v10mod  # noqa: E402
from lsy_drone_racing.control.gate_search_v10 import GateSearchV10  # noqa: E402
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402

# Single-change toggles from the current v10. nav=planner overrides; globals=module-global patches.
ABLATIONS = {
    "v10_base   ": dict(),
    "robs_0.18  ": dict(nav=dict(r_obs=0.18)),
    "robs_0.22  ": dict(nav=dict(r_obs=0.22)),
    "vcruise_0.8": dict(nav=dict(v_cruise=0.8, v_cruise_inter=0.8)),
    "handoff_0.5": dict(globals=dict(_NAV_START_VEL_SCALE=0.5)),
    "interleaved": dict(globals=dict(_DISCOVER_ALL_FIRST=False)),
}


def _run(env, cfg, n_gates, spec, seed):
    saved = {}
    for k, v in spec.get("globals", {}).items():
        saved[k] = getattr(v10mod, k)
        setattr(v10mod, k, v)
    try:
        obs, info = env.reset(seed=seed)
        ctrl = GateSearchV10(obs, info, cfg)
        nav = spec.get("nav", {})
        if nav:
            planner = dataclasses.replace(ctrl._settings.planner, **nav)
            ctrl._settings = dataclasses.replace(ctrl._settings, planner=planner)
            ctrl._references = ReferenceManager(
                planner, ctrl._settings.runtime.replan_gate_delta_m,
                ctrl._settings.runtime.replan_obstacle_delta_m,
            )
        finish_tick, maxtg, tick = None, -1, 0
        terminated = False
        last = np.zeros(3)
        while True:
            tg = int(np.asarray(obs["target_gate"]).reshape(()))
            p = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
            if not (p[0] == -1 and p[1] == -1):
                last = p.copy()
            if tg == -1 and finish_tick is None:
                finish_tick = tick
            if tg != -1:
                maxtg = max(maxtg, tg)
            a = ctrl.compute_control(obs, info)
            obs, r, terminated, tr, info = env.step(a)
            if ctrl.step_callback(a, obs, r, terminated, tr, info) or terminated or tr:
                break
            tick += 1
        finished = finish_tick is not None
        mode = str(getattr(ctrl, "_mode", "?"))
        reason = ("finish" if finished else
                  ("crash:" + ("grnd" if last[2] < 0.15 else mode[:4].lower()) if terminated else "timeout"))
        return {"finished": finished, "lap": finish_tick / cfg.env.freq if finished else None,
                "gates": n_gates if finished else max(maxtg, 0), "reason": reason}
    finally:
        for k, v in saved.items():
            setattr(v10mod, k, v)


def main(config: str = "level3.toml", n_runs: int = 12, seed_offset: int = 0):
    """Run each ablation over n_runs Level-3 seeds and print a comparison table."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    n_gates = len(cfg.env.track.gates)
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    ))
    print("=" * 84)
    print(f"config={config}  n_runs={n_runs}  (single-change ablations from current v10)")
    print(f"{'ablation':<12} {'finish':>8}  {'lap(mean)':>9}  {'meanGates':>9}   reasons")
    for name, spec in ABLATIONS.items():
        rows = [_run(env, cfg, n_gates, spec, seed_offset + k) for k in range(n_runs)]
        fins = [r for r in rows if r["finished"]]
        laps = [r["lap"] for r in fins]
        lap_s = f"{np.mean(laps):.1f}s" if laps else "  -  "
        reasons = {}
        for r in rows:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        rstr = " ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"{name:<12} {len(fins):>3}/{len(rows):<3}  {lap_s:>9}  "
              f"{np.mean([r['gates'] for r in rows]):>9.2f}   {rstr}")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
