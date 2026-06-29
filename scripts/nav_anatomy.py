r"""Anatomy of the NAVIGATE phase for GateSearchV10: where does the slow ~14 s go?

For each NAVIGATE tick, record speed |v|, vertical speed, and height, then report how navigate time
splits between descending vs level flight and how much is spent slow (<1.0 m/s). Pinpoints whether
the descent-rate cap, peri-gate zones, or turns are the dominant time sink before reworking them.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/nav_anatomy.py --n_runs 12
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


def main(config: str = "level3.toml", controller: str = "gate_search_v10.py", n_runs: int = 12,
         seed_offset: int = 0):
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    ctrl_cls = load_controller(Path(__file__).parents[1] / "lsy_drone_racing/control" / controller)
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    ))
    dt = 1.0 / cfg.env.freq
    # Aggregate over finishing runs only (so navigate covers the whole course).
    agg_nav_t = agg_path = agg_slow_t = agg_desc_t = agg_desc_path = agg_level_path = 0.0
    desc_speeds, level_speeds = [], []
    n_fin = 0
    print(f"anatomy {controller}  n={n_runs}")
    for k in range(n_runs):
        seed = seed_offset + k
        obs, info = env.reset(seed=seed)
        ctrl = ctrl_cls(obs, info, cfg)
        finish_tick, tick = None, 0
        prev_p = None
        nav_t = path = slow_t = desc_t = desc_path = level_path = 0.0
        while True:
            tg = int(np.asarray(obs["target_gate"]).reshape(()))
            if tg == -1 and finish_tick is None:
                finish_tick = tick
            a = ctrl.compute_control(obs, info)
            mode = getattr(ctrl, "_mode", "")
            p = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
            v = np.asarray(obs["vel"], dtype=np.float64).reshape(-1)[:3]
            if mode == "NAVIGATE" and prev_p is not None:
                step_d = float(np.linalg.norm(p - prev_p))
                spd = float(np.linalg.norm(v))
                nav_t += dt
                path += step_d
                if spd < 1.0:
                    slow_t += dt
                if v[2] < -0.15:  # descending
                    desc_t += dt
                    desc_path += step_d
                    desc_speeds.append(spd)
                else:
                    level_path += step_d
                    level_speeds.append(spd)
            prev_p = p
            obs, r, terminated, tr, info = env.step(a)
            if ctrl.step_callback(a, obs, r, terminated, tr, info) or terminated or tr:
                break
            tick += 1
        if finish_tick is not None:
            n_fin += 1
            agg_nav_t += nav_t; agg_path += path; agg_slow_t += slow_t
            agg_desc_t += desc_t; agg_desc_path += desc_path; agg_level_path += level_path
        ctrl.episode_callback()
    env.close()
    if n_fin == 0:
        print("no finishers")
        return
    print(f"finishers: {n_fin}/{n_runs}  (averages per finishing run)")
    print(f"  navigate time      : {agg_nav_t/n_fin:6.2f} s")
    print(f"  navigate path      : {agg_path/n_fin:6.2f} m   avg speed {agg_path/agg_nav_t:5.2f} m/s")
    print(f"  time spent <1.0 m/s: {agg_slow_t/n_fin:6.2f} s   ({100*agg_slow_t/agg_nav_t:4.1f}% of nav)")
    print(f"  descending time    : {agg_desc_t/n_fin:6.2f} s   ({100*agg_desc_t/agg_nav_t:4.1f}% of nav)")
    print(f"  descending path    : {agg_desc_path/n_fin:6.2f} m   avg desc speed {np.mean(desc_speeds):5.2f} m/s")
    print(f"  level path         : {agg_level_path/n_fin:6.2f} m   avg level speed {np.mean(level_speeds):5.2f} m/s")


if __name__ == "__main__":
    fire.Fire(main)
