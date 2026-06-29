r"""Isolate the effect of NAVIGATE speed on GateSearchV10 (reliable base): finish % and lap time.

Overrides only the planner cruise/peak speeds, leaving v10's proven discover-all search intact, so
the speed↔reliability trade-off of faster navigation is measured cleanly.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/nav_speed_sweep.py --n_runs 15
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

from lsy_drone_racing.control.gate_search_v10 import GateSearchV10  # noqa: E402
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402

# (v_cruise, v_cruise_inter, max_speed) — peri-gate / inter-gate / cap
SETTINGS = {
    "v10_base 1.0/1.0/1.4": (1.0, 1.0, 1.4),
    "fast     1.2/2.0/2.6": (1.2, 2.0, 2.6),
    "faster   1.3/2.4/3.0": (1.3, 2.4, 3.0),
    "fastest  1.5/2.8/3.5": (1.5, 2.8, 3.5),
}


def _run(env, cfg, n_gates, speeds, seed, dt):
    obs, info = env.reset(seed=seed)
    ctrl = GateSearchV10(obs, info, cfg)
    vc, vci, vmax = speeds
    planner = dataclasses.replace(ctrl._settings.planner, v_cruise=vc, v_cruise_inter=vci, max_speed=vmax)
    ctrl._settings = dataclasses.replace(ctrl._settings, planner=planner)
    ctrl._references = ReferenceManager(
        planner, ctrl._settings.runtime.replan_gate_delta_m, ctrl._settings.runtime.replan_obstacle_delta_m,
    )
    finish_tick, maxtg, tick = None, -1, 0
    nav_ticks = 0
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
        if getattr(ctrl, "_mode", "") == "NAVIGATE":
            nav_ticks += 1
        obs, r, terminated, tr, info = env.step(a)
        if ctrl.step_callback(a, obs, r, terminated, tr, info) or terminated or tr:
            break
        tick += 1
    finished = finish_tick is not None
    mode = str(getattr(ctrl, "_mode", "?"))
    reason = ("finish" if finished else
              ("crash:" + ("grnd" if last[2] < 0.15 else mode[:4].lower()) if terminated else "timeout"))
    return {"finished": finished, "lap": finish_tick * dt if finished else None,
            "gates": n_gates if finished else max(maxtg, 0), "reason": reason,
            "nav": nav_ticks * dt}


def main(config: str = "level3.toml", n_runs: int = 15, seed_offset: int = 0):
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    n_gates = len(cfg.env.track.gates)
    dt = 1.0 / cfg.env.freq
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    ))
    print("=" * 90)
    print(f"NAVIGATE speed sweep on GateSearchV10  config={config}  n_runs={n_runs}")
    print(f"{'setting':<22} {'finish':>8} {'lap(mean)':>10} {'navT(fin)':>10}  reasons")
    for name, speeds in SETTINGS.items():
        rows = [_run(env, cfg, n_gates, speeds, seed_offset + k, dt) for k in range(n_runs)]
        fins = [r for r in rows if r["finished"]]
        laps = [r["lap"] for r in fins]
        navs = [r["nav"] for r in fins]
        lap_s = f"{np.mean(laps):.1f}s" if laps else "  -  "
        nav_s = f"{np.mean(navs):.1f}s" if navs else "  -  "
        reasons = {}
        for r in rows:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        rstr = " ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"{name:<22} {len(fins):>3}/{len(rows):<3} {lap_s:>10} {nav_s:>10}  {rstr}")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
