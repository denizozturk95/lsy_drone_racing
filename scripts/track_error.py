r"""Measure NAVIGATE tracking error for a controller: |pos - reference|, per axis, and near gates.

Tells us whether the inner-loop tracker lags the reference (and on which axis) at speed — the
identified wall for fast laps. Reads the controller's reference plan + progress from outside.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/track_error.py --controller gate_search_v12.py --n_runs 10
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


def main(controller: str = "gate_search_v12.py", config: str = "level3.toml", n_runs: int = 10,
         seed_offset: int = 0):
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    dt = 1.0 / cfg.env.freq
    ctrl_cls = load_controller(Path(__file__).parents[1] / "lsy_drone_racing/control" / controller)
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    ))
    err_all, err_gate, spd_all = [], [], []
    for k in range(n_runs):
        obs, info = env.reset(seed=seed_offset + k)
        ctrl = ctrl_cls(obs, info, cfg)
        while True:
            p = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
            v = np.asarray(obs["vel"], dtype=np.float64).reshape(-1)[:3]
            gp = np.asarray(obs["gates_pos"], dtype=np.float64)
            tg = int(np.asarray(obs["target_gate"]).reshape(()))
            action = ctrl.compute_control(obs, info)
            if getattr(ctrl, "_mode", "") == "NAVIGATE":
                plan = ctrl._references.plan
                if plan is not None:
                    clock_t = (ctrl._tick - ctrl._plan_start_tick) * dt
                    look = ctrl._settings.runtime.lookahead_s
                    t_eval = float(np.clip(min(clock_t, ctrl._progress_t + look), 0.0, plan.t_total))
                    ref = np.asarray(plan.curve(t_eval), dtype=np.float64).reshape(3)
                    e = p - ref
                    err_all.append(np.abs(e))
                    spd_all.append(float(np.linalg.norm(v)))
                    if tg < len(gp) and float(np.linalg.norm(p[:2] - gp[tg][:2])) < 0.5:
                        err_gate.append(np.abs(e))
            obs, r, term, tr, info = env.step(action)
            if ctrl.step_callback(action, obs, r, term, tr, info) or term or tr:
                break
        ctrl.episode_callback()
    env.close()
    err_all = np.array(err_all); err_gate = np.array(err_gate); spd_all = np.array(spd_all)
    print(f"controller={controller}  navigate ticks={len(err_all)}  gate-zone ticks={len(err_gate)}")
    print(f"  mean speed {spd_all.mean():.2f}  p95 speed {np.percentile(spd_all,95):.2f}")
    def line(tag, arr):
        if not len(arr):
            print(f"  {tag}: (none)"); return
        m = arr.mean(0); p95 = np.percentile(arr, 95, 0); mx = arr.max(0)
        print(f"  {tag}: |err| mean x{m[0]:.3f} y{m[1]:.3f} z{m[2]:.3f} | p95 x{p95[0]:.3f} y{p95[1]:.3f} z{p95[2]:.3f} | max x{mx[0]:.2f} y{mx[1]:.2f} z{mx[2]:.2f}")
    line("ALL navigate ", err_all)
    line("GATE zone(<0.5m)", err_gate)


if __name__ == "__main__":
    fire.Fire(main)
