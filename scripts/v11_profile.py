r"""Profile where GateSearchV11 spends its lap: ticks per mode, search episodes, global fallback.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/v11_profile.py --n_runs 10
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, E501, T201, C901

from __future__ import annotations

import ctypes
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

from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402


def main(config: str = "level3.toml", controller: str = "gate_search_v11.py", n_runs: int = 10,
         seed_offset: int = 0):
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    n_gates = len(cfg.env.track.gates)
    ctrl_cls = load_controller(Path(__file__).parents[1] / "lsy_drone_racing/control" / controller)
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    ))
    dt = 1.0 / cfg.env.freq
    print("=" * 96)
    print(f"profile {controller}  n={n_runs}")
    print(f"{'seed':>4} {'fin':>3} {'lap':>6} {'TO':>5} {'SRCH':>6} {'NAV':>5} {'#srch':>5} {'glob':>4} {'gates':>5}")
    for k in range(n_runs):
        seed = seed_offset + k
        obs, info = env.reset(seed=seed)
        ctrl = ctrl_cls(obs, info, cfg)
        modes = {}
        prev_mode = None
        n_search_eps = 0
        glob_ticks = 0
        finish_tick, maxtg, tick = None, -1, 0
        while True:
            tg = int(np.asarray(obs["target_gate"]).reshape(()))
            if tg == -1 and finish_tick is None:
                finish_tick = tick
            if tg != -1:
                maxtg = max(maxtg, tg)
            a = ctrl.compute_control(obs, info)
            m = getattr(ctrl, "_mode", "?")
            modes[m] = modes.get(m, 0) + 1
            if m == "SEARCH":
                if prev_mode != "SEARCH":
                    n_search_eps += 1
                if getattr(ctrl, "_search_is_global", False):
                    glob_ticks += 1
            prev_mode = m
            obs, r, terminated, tr, info = env.step(a)
            if ctrl.step_callback(a, obs, r, terminated, tr, info) or terminated or tr:
                break
            tick += 1
        finished = finish_tick is not None
        lap = finish_tick * dt if finished else tick * dt
        gates = n_gates if finished else max(maxtg, 0)
        print(f"{seed:>4} {'Y' if finished else 'n':>3} {lap:>6.1f} "
              f"{modes.get('TAKEOFF', 0)*dt:>5.1f} {modes.get('SEARCH', 0)*dt:>6.1f} "
              f"{modes.get('NAVIGATE', 0)*dt:>5.1f} {n_search_eps:>5} {glob_ticks*dt:>4.1f} {gates:>5}")
        ctrl.episode_callback()
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
