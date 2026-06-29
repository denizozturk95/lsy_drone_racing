r"""Find the real NAVIGATE-speed levers: ablate the slowdown heuristics, not just the speed cap.

nav_anatomy showed 85% of navigate is < 1 m/s even on level flight — the cruise cap is rarely
binding; peri_gate_radius, turn_slowdown and t_min_seg force the clamped spline slow. This sweeps
those on GateSearchV10 (reliable global search) and reports finish %, lap, and navigate time.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/nav_timing_ablation.py --n_runs 15
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, E501, C901, T201

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

import lsy_drone_racing.control.online_planner.timing as timing  # noqa: E402
from lsy_drone_racing.control.gate_search_v10 import GateSearchV10  # noqa: E402
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402

_ORIG_TURN = timing.turn_slowdown

# Tracking-precision sweep at a fixed fast-decoupled speed (decoupleC). Raise feedforward so the
# drone anticipates the curve (less lag/corner-cut at speed) and vary lookahead. The hypothesis:
# better tracking lets the drone cross gates fast WITHOUT the crash:navi penalty.
_SPD = dict(v_cruise=1.2, v_cruise_inter=2.6, max_speed=3.2, peri_gate_radius=0.50)
ABLATIONS = {
    "base                 ": dict(),
    "decoupleC (ff0.6)    ": dict(planner=dict(**_SPD)),
    "C+ff0.8              ": dict(planner=dict(**_SPD), command=dict(feedforward_scale=0.8)),
    "C+ff1.0              ": dict(planner=dict(**_SPD), command=dict(feedforward_scale=1.0)),
    "C+ff0.9+look0.12     ": dict(planner=dict(**_SPD), command=dict(feedforward_scale=0.9), runtime=dict(lookahead_s=0.12)),
    "C+ff1.0+look0.10     ": dict(planner=dict(**_SPD), command=dict(feedforward_scale=1.0), runtime=dict(lookahead_s=0.10)),
    "ff1.0 only (v10 spd) ": dict(command=dict(feedforward_scale=1.0)),
}


def _run(env, cfg, n_gates, spec, seed, dt):
    # Patch turn_slowdown gain if requested
    turn = spec.get("turn")
    if turn is not None:
        def _patched(waypoints, segment_times, min_sharpness=0.4, slow_gain=0.6, _g=turn):
            return _ORIG_TURN(waypoints, segment_times, min_sharpness=0.4, slow_gain=_g)
        timing.turn_slowdown = _patched
    else:
        timing.turn_slowdown = _ORIG_TURN
    try:
        obs, info = env.reset(seed=seed)
        ctrl = GateSearchV10(obs, info, cfg)
        pk = spec.get("planner", {})
        ck = spec.get("command", {})
        rk = spec.get("runtime", {})
        if ck:
            ctrl._settings = dataclasses.replace(
                ctrl._settings, command=dataclasses.replace(ctrl._settings.command, **ck))
        if rk:
            ctrl._settings = dataclasses.replace(
                ctrl._settings, runtime=dataclasses.replace(ctrl._settings.runtime, **rk))
        if pk:
            planner = dataclasses.replace(ctrl._settings.planner, **pk)
            ctrl._settings = dataclasses.replace(ctrl._settings, planner=planner)
            ctrl._references = ReferenceManager(
                planner, ctrl._settings.runtime.replan_gate_delta_m, ctrl._settings.runtime.replan_obstacle_delta_m,
            )
        finish_tick, maxtg, tick, nav_ticks = None, -1, 0, 0
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
                "gates": n_gates if finished else max(maxtg, 0), "reason": reason, "nav": nav_ticks * dt}
    finally:
        timing.turn_slowdown = _ORIG_TURN


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
    print("=" * 96)
    print(f"NAVIGATE timing ablation on GateSearchV10  config={config}  n_runs={n_runs}")
    print(f"{'ablation':<22} {'finish':>7} {'lap':>7} {'navT(fin)':>10}  reasons")
    for name, spec in ABLATIONS.items():
        rows = [_run(env, cfg, n_gates, spec, seed_offset + k, dt) for k in range(n_runs)]
        fins = [r for r in rows if r["finished"]]
        laps = [r["lap"] for r in fins]
        navs = [r["nav"] for r in fins]
        lap_s = f"{np.mean(laps):.1f}s" if laps else "  -  "
        nav_s = f"{np.mean(navs):.1f}s" if navs else "  -  "
        reasons = {}
        for r in rows:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        rstr = " ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"{name:<22} {len(fins):>2}/{len(rows):<2} {lap_s:>7} {nav_s:>10}  {rstr}")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
