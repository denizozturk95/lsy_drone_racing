r"""Locate WHERE navigate crashes happen: nearest gate, lateral/vertical offset, speed at crash.

Compares two controllers on the same seeds and prints, per non-finishing run, the crash mode so we
can tell frame-clip (at a gate plane) from path/transit crashes — the navigate-rework blocker.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/crash_locate.py --n_runs 15
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


def _run(env, cfg, ctrl_cls, seed, dt):
    obs, info = env.reset(seed=seed)
    ctrl = ctrl_cls(obs, info, cfg)
    finish_tick, tick = None, 0
    last = np.zeros(3); last_v = np.zeros(3); last_tg = 0
    last_gp = np.zeros((1, 3)); last_op = np.zeros((1, 3)); last_ov = np.zeros(1, dtype=bool)
    terminated = False
    while True:
        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        p = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        v = np.asarray(obs["vel"], dtype=np.float64).reshape(-1)[:3]
        gp = np.asarray(obs["gates_pos"], dtype=np.float64)
        op = np.asarray(obs["obstacles_pos"], dtype=np.float64)
        ov = np.asarray(obs["obstacles_visited"], dtype=bool)
        if not (p[0] == -1 and p[1] == -1):
            last, last_v, last_tg = p.copy(), v.copy(), tg
            last_gp, last_op, last_ov = gp.copy(), op.copy(), ov.copy()
        if tg == -1 and finish_tick is None:
            finish_tick = tick
        a = ctrl.compute_control(obs, info)
        obs, r, terminated, tr, info = env.step(a)
        if ctrl.step_callback(a, obs, r, terminated, tr, info) or terminated or tr:
            break
        tick += 1
    ctrl.episode_callback()
    if finish_tick is not None:
        return f"seed{seed:2d} FINISH {finish_tick*dt:5.1f}s"
    if not terminated:
        return f"seed{seed:2d} timeout/other tg={last_tg}"
    spd = float(np.linalg.norm(last_v))
    # nearest gate (any) and nearest obstacle (any) in XY at the crash point
    gdists = np.linalg.norm(last_gp[:, :2] - last[:2], axis=1)
    gi = int(np.argmin(gdists)); gd = float(gdists[gi])
    odists = np.linalg.norm(last_op[:, :2] - last[:2], axis=1)
    oi = int(np.argmin(odists)); od = float(odists[oi]); oknown = bool(last_ov[oi])
    if last[2] < 0.2:
        kind = "GROUND"
    elif gd < 0.45:
        kind = f"GATE-FRAME g{gi}"
    elif od < 0.25:
        kind = f"OBSTACLE o{oi}{'(known)' if oknown else '(UNSEEN)'}"
    else:
        kind = "open-air?"
    return (f"seed{seed:2d} CRASH {kind:18s} tg={last_tg} spd={spd:.2f} "
            f"nearGate g{gi}={gd:.2f} nearObs o{oi}={od:.2f}{'k' if oknown else 'u'} pos={np.round(last,2)}")


def main(controllers: str = "gate_search_v10.py,gate_search_v12.py", config: str = "level3.toml",
         n_runs: int = 15, seed_offset: int = 0):
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    dt = 1.0 / cfg.env.freq
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    ))
    for cname in controllers.split(","):
        ctrl_cls = load_controller(Path(__file__).parents[1] / "lsy_drone_racing/control" / cname.strip())
        print("=" * 90)
        print(cname.strip())
        for k in range(n_runs):
            print("  " + _run(env, cfg, ctrl_cls, seed_offset + k, dt))
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
