r"""Ablate GateSearchV9's changes vs GateSearchV8 to find which one regresses finishing.

v9 = v8 + {trimmed d_pre/d_post, narrower reversal swing, arena geofence, gate-2 bias dropped}.
On level2_deploy.toml v8 finishes 100% but v9 0%. This runs each factor in isolation (over a seed
batch, reusing one env) by overriding the controller's planner settings / bias at runtime, so we
can see which change causes the crash and back off just that one. Reports finish %, where failures
die (mean max target gate reached), lateral bound behaviour, and lap time.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/v9_ablation.py \
        --config level2_deploy.toml --n_runs 8
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

from lsy_drone_racing.control.gate_search_v9 import GateSearchV9  # noqa: E402
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager  # noqa: E402
from lsy_drone_racing.utils import load_config  # noqa: E402

# Speed sweep relative to the v9 default (v_cruise 1.4 / inter 1.6 / max 1.9). Slower peri-gate
# speed should track the gate-2 U-turn more cleanly -> fewer crashes, at a small time cost.
ABLATIONS = {
    "v9_base   ": dict(),
    "slow_1.2  ": dict(v_cruise=1.2, v_cruise_inter=1.5, max_speed=1.7),
    "slow_1.1  ": dict(v_cruise=1.1, v_cruise_inter=1.4, max_speed=1.6),
    "slow_1.0  ": dict(v_cruise=1.0, v_cruise_inter=1.3, max_speed=1.5),
    "v8_equiv  ": dict(d_pre=0.60, d_post=0.40, reversal_swing_m=0.55, reversal_apex_m=0.10, geofence_margin=0.0, _bias={2: -0.073}),
}


def _run(env, cfg, n_gates, low, high, overrides, seed):
    obs, info = env.reset(seed=seed)
    ctrl = GateSearchV9(obs, info, cfg)
    bias = overrides.pop("_bias", {})
    planner = dataclasses.replace(ctrl._settings.planner, **overrides)
    ctrl._settings = dataclasses.replace(ctrl._settings, planner=planner)
    ctrl._gate_bias = dict(bias)
    ctrl._references = ReferenceManager(
        planner,
        ctrl._settings.runtime.replan_gate_delta_m,
        ctrl._settings.runtime.replan_obstacle_delta_m,
    )
    tick, finish_tick, maxtg, ymax, oob = 0, None, 0, 0.0, False
    while True:
        tg = int(np.asarray(obs["target_gate"]).reshape(()))
        p = np.asarray(obs["pos"], dtype=np.float64).reshape(-1)[:3]
        if not (p[0] == -1 and p[1] == -1):
            ymax = max(ymax, abs(float(p[1])))
            lat = min(p[0] - low[0], high[0] - p[0], p[1] - low[1], high[1] - p[1])
            oob = oob or lat < 0.0
        if tg == -1 and finish_tick is None:
            finish_tick = tick
        if 0 <= tg < n_gates:
            maxtg = max(maxtg, tg)
        a = ctrl.compute_control(obs, info)
        obs, r, term, trunc, info = env.step(a)
        if ctrl.step_callback(a, obs, r, term, trunc, info) or term or trunc:
            break
        tick += 1
    return {
        "finished": finish_tick is not None,
        "lap": finish_tick / cfg.env.freq if finish_tick is not None else None,
        "maxtg": maxtg,
        "ymax": ymax,
        "oob": oob,
    }


def main(config: str = "level2_deploy.toml", n_runs: int = 8, seed_offset: int = 0,
         deploy_faithful: bool = False):
    """Run every ablation over n_runs seeds on `config` and print a comparison table."""
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False
    low = np.asarray(cfg.env.track.safety_limits.pos_limit_low, dtype=np.float64)
    high = np.asarray(cfg.env.track.safety_limits.pos_limit_high, dtype=np.float64)
    n_gates = len(cfg.env.track.gates)
    randomizations = cfg.env.get("randomizations")
    if deploy_faithful and randomizations is not None:
        randomizations = {
            k: v for k, v in randomizations.items()
            if k not in ("gate_pos", "gate_rpy", "obstacle_pos")
        }
    env = JaxToNumpy(gymnasium.make(
        cfg.env.id, freq=cfg.env.freq, sim_config=cfg.sim, sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode, track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"), randomizations=randomizations,
        seed=cfg.env.seed,
    ))
    print("=" * 84)
    print(f"config={config}  n_runs={n_runs}  (gates={n_gates})")
    print(f"{'ablation':<12} {'finish':>8}  {'lap(mean)':>9}  {'meanMaxGate':>11}  {'|y|max':>7}  {'OOB':>4}")
    for name, ovr in ABLATIONS.items():
        rows = [_run(env, cfg, n_gates, low, high, dict(ovr), seed_offset + k) for k in range(n_runs)]
        fins = [r for r in rows if r["finished"]]
        laps = [r["lap"] for r in fins]
        lap_s = f"{np.mean(laps):.2f}s" if laps else "  -  "
        print(f"{name:<12} {len(fins):>3}/{len(rows):<3}  {lap_s:>9}  "
              f"{np.mean([r['maxtg'] for r in rows]):>11.2f}  "
              f"{max(r['ymax'] for r in rows):>7.3f}  {sum(r['oob'] for r in rows):>4}")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
