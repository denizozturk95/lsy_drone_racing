r"""Level-3 evaluation: finish rate, lap time, and failure-mode breakdown over random tracks.

Level 3 (level3.toml, randomize=true) hides each gate's true position until the drone is within
sensor_range (0.7 m XY); obs reports the [0,0,h] nominal until then. A controller must search,
discover, and navigate. This measures finish %, lap time (to last gate), mean gates passed, and
WHY runs fail -- crash (env terminated), timeout (truncated / controller 30 s), or out-of-bounds --
so robustness work can target the real failure mode. Each seed is an independent random track.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/level3_eval.py \
        --controller gate_search_v2.py --n_runs 30
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


def _run_one(env, cfg, ctrl_cls, low, high, n_gates, seed):
    obs, info = env.reset(seed=seed)
    ctrl = ctrl_cls(obs, info, cfg)
    tick, finish_tick, maxtg = 0, None, -1
    ymax = xmax = 0.0
    oob = False
    terminated = truncated = ctrl_done = False
    last_pos = np.zeros(3)
    while True:
        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        p = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        if not (p[0] == -1 and p[1] == -1):
            last_pos = p.copy()
            ymax, xmax = max(ymax, abs(float(p[1]))), max(xmax, abs(float(p[0])))
            if min(p[0] - low[0], high[0] - p[0], p[1] - low[1], high[1] - p[1]) < 0.0:
                oob = True
        if tg == -1 and finish_tick is None:
            finish_tick = tick
        if tg != -1:
            maxtg = max(maxtg, tg)
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        ctrl_done = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if ctrl_done or terminated or truncated:
            break
        tick += 1
    finished = finish_tick is not None
    mode = str(getattr(ctrl, "_mode", "?"))
    # target_gate == number of gates already passed; maxtg is the highest target reached.
    gates_passed = n_gates if finished else max(maxtg, 0)
    ctrl.episode_callback()
    ctrl.episode_reset()
    # Failure reason (only meaningful if not finished); crashes are sub-classified by location.
    if finished:
        reason = "finish"
    elif terminated:
        if last_pos[2] < 0.15:
            reason = "crash:ground"
        elif abs(last_pos[0]) > 2.8 or abs(last_pos[1]) > 2.8 or last_pos[2] > 2.4:
            reason = "crash:wall"
        else:
            reason = f"crash:{mode[:4].lower()}"  # in-bounds collision, tagged by mode (sear/navi)
    elif truncated or ctrl_done:
        reason = "timeout"
    else:
        reason = "other"
    return {
        "seed": seed, "finished": finished,
        "time": finish_tick / cfg.env.freq if finished else None,
        "gates_passed": gates_passed, "reason": reason, "oob": oob,
        "ymax": ymax, "xmax": xmax,
    }


def evaluate(config: str = "level3.toml", controller: str | None = None, n_runs: int = 30,
             seed_offset: int = 0):
    """Run a Level-3 seed batch and print finish %, lap, gates-passed, and failure breakdown."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    low = np.asarray(cfg.env.track.safety_limits.pos_limit_low, dtype=np.float64)
    high = np.asarray(cfg.env.track.safety_limits.pos_limit_high, dtype=np.float64)
    n_gates = len(cfg.env.track.gates)
    ctrl_path = Path(__file__).parents[1] / "lsy_drone_racing/control" / (controller or cfg.controller.file)
    ctrl_cls = load_controller(ctrl_path)
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    ))
    rows = [_run_one(env, cfg, ctrl_cls, low, high, n_gates, seed_offset + k) for k in range(n_runs)]
    env.close()

    fins = [r for r in rows if r["finished"]]
    times = np.array([r["time"] for r in fins], dtype=np.float64)
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    print("=" * 78)
    print(f"config={config}  controller={controller or cfg.controller.file}  n={n_runs}")
    print(f"  FINISH: {len(fins)}/{len(rows)} ({100*len(fins)/len(rows):.1f}%)")
    if len(times):
        print(f"  lap (finishers): mean {times.mean():.2f}s  min {times.min():.2f}s  max {times.max():.2f}s")
    print(f"  gates passed (all runs): mean {np.mean([r['gates_passed'] for r in rows]):.2f} / {n_gates}")
    print("  end reason: " + "  ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
    print(f"  out-of-bounds (crossed safety plane): {sum(r['oob'] for r in rows)}/{len(rows)}  "
          f"(peak |x|={max(r['xmax'] for r in rows):.2f}/{high[0]:.1f}, |y|={max(r['ymax'] for r in rows):.2f}/{high[1]:.1f})")
    return rows


if __name__ == "__main__":
    fire.Fire(evaluate)
