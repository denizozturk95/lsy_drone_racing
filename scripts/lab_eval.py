r"""Headless lab-track evaluation: finish rate, lap time, and arena-bound safety margin.

Purpose
-------
Reproduce, in sim, the real-lab failure where the drone's (too-wide) trajectory leaves the
arena and the run is aborted. For each seed it reports the worst-case approach to every
``safety_limits`` plane, so a trajectory change can be judged on BOTH speed and how much
in-bounds clearance it buys -- before spending a lab slot.

Windows note
------------
MuJoCo's C++ ``MjSpec.from_file`` opens model files through the ANSI codepage and fails on a
project path containing non-ASCII characters (this repo lives under ``...\\Masaüstü\\...``).
We monkeypatch ``MjSpec.from_file`` to pass MuJoCo the Windows 8.3 short path (pure ASCII).
No-op on non-Windows and on already-ASCII paths.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/lab_eval.py \
        --config real_track.toml --controller gate_search_v8.py --n_runs 20
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, E501, T201  (diagnostic script)

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path


def _activate_env_dlls():
    r"""Put the pixi env's conda DLL dirs on the search path so numpy can load LAPACK.

    This env's numpy is conda-installed and links a shared BLAS/LAPACK in ``Library\\bin``;
    running ``python.exe`` directly (no ``pixi run`` -- the workspace doesn't list win-64) skips
    activation, so ``np.linalg`` hard-crashes (delay-load 0xc06d007f). Replicate activation by
    prepending the env's bin dirs to PATH and registering them as DLL directories. No-op off
    Windows. Must run before numpy is imported (i.e. before the MuJoCo patch, which imports numpy).
    """
    if os.name != "nt":
        return
    root = os.path.dirname(os.path.abspath(sys.executable))
    dirs = [root, *(os.path.join(root, *p) for p in (
        ("Library", "bin"), ("Library", "mingw-w64", "bin"), ("Scripts",), ("DLLs",),
    ))]
    existing = [d for d in dirs if os.path.isdir(d)]
    os.environ["PATH"] = os.pathsep.join(existing) + os.pathsep + os.environ.get("PATH", "")
    for d in existing:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass


_activate_env_dlls()


def _install_mujoco_shortpath_patch():
    """Make MuJoCo open model files via their ASCII 8.3 short path on Windows."""
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

    def patched(path, *a, **k):
        return orig(short(path), *a, **k)

    mujoco.MjSpec.from_file = staticmethod(patched)


_install_mujoco_shortpath_patch()

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402

from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402


def _run_one(env, cfg, ctrl_cls, low, high, seed):
    # Record the *input* observation each tick (always a valid in-flight state). The post-step obs
    # on the terminating step is a [-1,-1,-1] terminal sentinel, not a real position, so recording
    # after step() would spuriously flag out-of-bounds. Lap time is measured to the moment the last
    # gate is passed (target_gate -> -1), excluding the subsequent HOME descent.
    obs, info = env.reset(seed=seed)
    ctrl = ctrl_cls(obs, info, cfg)
    tick, finish_tick, worst_margin, oob = 0, None, np.inf, False
    pos_log = []
    while True:
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        if int(np.asarray(obs["target_gate"]).reshape(())) == -1 and finish_tick is None:
            finish_tick = tick
        pos_log.append(pos.copy())
        # Lateral (x,y) + ceiling margin; the z-floor is excluded because the drone rests on the
        # ground (z~0) at start/landing, which is not a "left the zone" abort.
        margin = float(min(
            pos[0] - low[0], high[0] - pos[0],
            pos[1] - low[1], high[1] - pos[1],
            high[2] - pos[2],
        ))
        worst_margin = min(worst_margin, margin)
        oob = oob or margin < 0.0
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        if ctrl.step_callback(action, obs, reward, terminated, truncated, info):
            break
        if terminated or truncated:
            break
        tick += 1
    finished = finish_tick is not None
    ctrl.episode_callback()
    ctrl.episode_reset()
    p = np.asarray(pos_log)
    return {
        "seed": seed,
        "finished": finished,
        "time": finish_tick / cfg.env.freq if finished else None,
        "worst_margin": worst_margin,
        "oob": oob,
        "x_abs_max": float(np.max(np.abs(p[:, 0]))),
        "y_abs_max": float(np.max(np.abs(p[:, 1]))),
        "z_max": float(np.max(p[:, 2])),
    }


def evaluate(
    config: str = "real_track.toml",
    controller: str | None = None,
    n_runs: int = 20,
    seed_offset: int = 0,
    deploy_faithful: bool = False,
):
    """Run a seed batch headless and print finish/lap/bound-margin stats.

    deploy_faithful=True drops the gate/obstacle *position* randomizations, leaving only drone
    start/mass/inertia + action/dynamics noise -- i.e. the real deploy scenario where the track is
    Vicon-measured and known from t=0 (nominal == true). Use it to judge next week's lab run; the
    default (with gate/obstacle randomization + sensor_range) is the harder online/Level-3 scenario.
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    low = np.asarray(cfg.env.track.safety_limits.pos_limit_low, dtype=np.float64)
    high = np.asarray(cfg.env.track.safety_limits.pos_limit_high, dtype=np.float64)
    ctrl_path = (
        Path(__file__).parents[1]
        / "lsy_drone_racing/control"
        / (controller or cfg.controller.file)
    )
    ctrl_cls = load_controller(ctrl_path)
    randomizations = cfg.env.get("randomizations")
    if deploy_faithful and randomizations is not None:
        randomizations = {
            k: v for k, v in randomizations.items()
            if k not in ("gate_pos", "gate_rpy", "obstacle_pos")
        }
    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=randomizations,
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)
    rows = [_run_one(env, cfg, ctrl_cls, low, high, seed_offset + k) for k in range(n_runs)]
    env.close()

    fins = [r for r in rows if r["finished"]]
    times = np.array([r["time"] for r in fins], dtype=np.float64)
    margins = np.array([r["worst_margin"] for r in rows], dtype=np.float64)
    n_oob = sum(r["oob"] for r in rows)
    print("=" * 78)
    print(f"config={config}  controller={controller or cfg.controller.file}  n={n_runs}")
    print(f"  finish:        {len(fins)}/{len(rows)} ({100 * len(fins) / len(rows):.1f}%)")
    if len(times):
        print(f"  lap time (s):  mean {times.mean():.2f}  min {times.min():.2f}  max {times.max():.2f}")
    print(f"  out-of-bounds: {n_oob}/{len(rows)} runs crossed a lateral/ceiling safety plane")
    print(f"  lat+ceil margin: min {margins.min():.3f} m  mean {margins.mean():.3f} m  (>=0 stays in)")
    print(f"  peak |x|={max(r['x_abs_max'] for r in rows):.3f} (lim {high[0]:.2f})  "
          f"|y|={max(r['y_abs_max'] for r in rows):.3f} (lim {high[1]:.2f})  "
          f"z={max(r['z_max'] for r in rows):.3f} (lim {high[2]:.2f})")
    worst = sorted(rows, key=lambda r: r["worst_margin"])[:5]
    print("  tightest runs (seed: margin, |y|max, finished, time):")
    for r in worst:
        t = f"{r['time']:.2f}s" if r["time"] is not None else "DNF"
        print(f"    seed {r['seed']:>3}: margin {r['worst_margin']:+.3f}  "
              f"|y|max {r['y_abs_max']:.3f}  {'OK' if r['finished'] else 'FAIL'}  {t}")
    return rows


if __name__ == "__main__":
    fire.Fire(evaluate)
