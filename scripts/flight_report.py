"""Render the 4-panel flight report for a controller on a track.

Flies one episode (default: controller_v10_5_max on final.toml), logs the flown
trajectory + measured speed, and produces the same four figures we made for
controller_v27:

    fig_3d.png       flown racing line in 3D, colored by speed
    fig_topdown.png  top-down racing line + gates, obstacles, crossing arrows
    fig_speed.png    speed profile vs time with gate-crossing markers
    fig_montage.png  2x3 MuJoCo renders through the flight (green = live plan)

    $ python scripts/flight_report.py
    $ python scripts/flight_report.py --controller controller_v27.py --config final.toml

Everything is the *flown* trajectory (measured pos/vel), so it works for any
controller without reaching into its internal planner.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import gymnasium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)

CMAP = "turbo"  # dark-blue slow -> red fast, like the v27 figures


def _fly(config, controller_file, n_montage: int, mont_w: int, mont_h: int):
    """Run one episode, return the logged trajectory, gate crossings and montage frames."""
    root = Path(__file__).parents[1]
    controller_cls = load_controller(root / "lsy_drone_racing/control" / controller_file)
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset()
    u = env.unwrapped
    gates = np.asarray(u.data.gates_pos[0])  # (n, 3) ground-truth poses
    obstacles = np.asarray(u.data.obstacles_pos[0])  # (n, 3)
    ctrl = controller_cls(obs, info, config)

    ts, pos, spd = [], [], []
    crossings = {}  # gate index -> (time, velocity) at the moment it is passed
    frames = []  # (time, rgb) sampled for the montage
    freq = config.env.freq
    prev_target = int(obs["target_gate"])
    n_gates = gates.shape[0]

    i = 0
    while True:
        t = i / freq
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)

        ts.append(t)
        pos.append(np.asarray(obs["pos"], dtype=float))
        spd.append(float(np.linalg.norm(obs["vel"])))

        target = int(obs["target_gate"])
        if target != prev_target:  # a gate was just crossed
            crossed = prev_target if target != -1 else n_gates - 1
            crossings[crossed] = (t, np.asarray(obs["vel"], dtype=float))
            prev_target = target

        # Sample montage frames ~every 0.3 s, with the controller's live plan drawn.
        if i % max(1, round(freq * 0.3)) == 0:
            ctrl.render_callback(u.sim)
            rgb = u.sim.render(mode="rgb_array", camera=u.settings.camera,
                               cam_config=u.settings.cam_config, width=mont_w, height=mont_h)
            frames.append((t, np.asarray(rgb)))

        if terminated or truncated or finished:
            break
        i += 1

    env.close()
    ts, pos, spd = np.array(ts), np.array(pos), np.array(spd)
    valid = pos[:, 2] > -0.05  # drop the terminal sentinel obs (pos = [-1, -1, -1])
    ts, pos, spd = ts[valid], pos[valid], spd[valid]
    traj = dict(
        t=ts, pos=pos, spd=spd,
        gates=gates, obstacles=obstacles, crossings=crossings,
        lap=ts[-1], finished=len(crossings) == n_gates,
    )
    # Pick n_montage evenly spaced frames.
    idx = np.linspace(0, len(frames) - 1, n_montage).round().astype(int)
    traj["montage"] = [frames[k] for k in idx]
    return traj


def _colored_line_2d(ax, x, y, c, norm):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP, norm=norm, linewidth=2.6)
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    return lc


def _plot_topdown(traj, title, path):
    x, y = traj["pos"][:, 0], traj["pos"][:, 1]
    norm = plt.Normalize(0, traj["spd"].max())
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    lc = _colored_line_2d(ax, x, y, traj["spd"], norm)
    fig.colorbar(lc, ax=ax, label="speed (m/s)")

    for j, o in enumerate(traj["obstacles"]):
        ax.add_patch(plt.Circle((o[0], o[1]), 0.3, fill=False, ls=":", color="0.5"))
        ax.plot(o[0], o[1], "o", color="firebrick", ms=7)
        ax.annotate(f"obs{j + 1}", (o[0], o[1]), fontsize=8, color="0.4",
                    ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")

    for j, g in enumerate(traj["gates"]):
        vhat = _crossing_dir(traj, j)
        perp = np.array([-vhat[1], vhat[0]]) * 0.25  # gate bar spans the opening
        ax.plot([g[0] - perp[0], g[0] + perp[0]], [g[1] - perp[1], g[1] + perp[1]],
                color="seagreen", lw=5, solid_capstyle="round")
        ax.annotate("", xy=(g[0] + vhat[0] * 0.45, g[1] + vhat[1] * 0.45), xytext=(g[0], g[1]),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4))
        ax.annotate(f"G{j + 1}  {g[2]:.2f}m", (g[0], g[1]), fontsize=9, fontweight="bold",
                    color="seagreen", ha="center", va="top", xytext=(0, -8),
                    textcoords="offset points")

    ax.plot(x[0], y[0], "*", color="black", ms=16)
    ax.annotate("START", (x[0], y[0]), fontsize=9, fontweight="bold",
                xytext=(8, 8), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=11)
    ax.margins(0.12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _crossing_dir(traj, gate_idx):
    """Unit heading at the gate crossing, falling back to the gate's nearest path tangent."""
    cr = traj["crossings"].get(gate_idx)
    if cr is not None and np.linalg.norm(cr[1][:2]) > 1e-3:
        v = cr[1][:2]
        return v / np.linalg.norm(v)
    k = int(np.argmin(np.linalg.norm(traj["pos"][:, :2] - traj["gates"][gate_idx, :2], axis=1)))
    k = min(k, len(traj["pos"]) - 2)
    v = traj["pos"][k + 1, :2] - traj["pos"][k, :2]
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _plot_3d(traj, title, path):
    p, sp = traj["pos"], traj["spd"]
    z = p[:, 2]
    k = int(np.argmax(z > 0.2)) if (z > 0.2).any() else 0  # drop the ground-level takeoff climb
    p, sp = p[k:], sp[k:]
    norm = plt.Normalize(0, traj["spd"].max())
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    segs = np.stack([p[:-1], p[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=CMAP, norm=norm, linewidth=2.4)
    lc.set_array(sp[:-1])
    ax.add_collection3d(lc)
    fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.1, label="speed (m/s)")

    for o in traj["obstacles"]:  # obstacles as vertical poles
        ax.plot([o[0], o[0]], [o[1], o[1]], [0, 1.5], color="firebrick", lw=3)
    for j, g in enumerate(traj["gates"]):  # gates as vertical squares in the crossing plane
        vhat = _crossing_dir(traj, j)
        lat = np.array([-vhat[1], vhat[0], 0.0]) * 0.22
        up = np.array([0.0, 0.0, 0.22])
        c = g.copy()
        loop = np.array([c + lat + up, c - lat + up, c - lat - up, c + lat - up, c + lat + up])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="seagreen", lw=2)
        ax.text(g[0], g[1], g[2] + 0.28, f"G{j + 1}", color="seagreen", fontweight="bold")

    start = traj["pos"][0]
    ax.scatter(*start, color="black", marker="*", s=120)
    ax.text(start[0], start[1], start[2] + 0.1, "START", fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_zlim(0, max(1.6, p[:, 2].max() + 0.2))
    ax.view_init(elev=22, azim=-61)
    ax.set_title(title, fontsize=11)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_speed(traj, title, path):
    t, s = traj["t"], traj["spd"]
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.fill_between(t, s, alpha=0.18, color="steelblue")
    ax.plot(t, s, color="steelblue", lw=2.2)
    ax.axhline(s.max(), ls=":", color="firebrick", lw=1.2)
    for j, (ct, _) in sorted(traj["crossings"].items(), key=lambda kv: kv[1][0]):
        ax.axvline(ct, ls="--", color="seagreen", lw=1, alpha=0.8)
        ax.text(ct, s.max() * 0.5, f"G{j + 1}", color="seagreen", fontweight="bold",
                ha="center", backgroundcolor="white", fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speed (m/s)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, s.max() * 1.12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_montage(traj, title, path, cols=3):
    frames = traj["montage"]
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.3 * rows), constrained_layout=True)
    for ax, (t, rgb) in zip(np.ravel(axes), frames):
        ax.imshow(rgb)
        ax.text(0.02, 0.94, f"t={t:.1f}s", transform=ax.transAxes, color="yellow",
                fontsize=11, fontweight="bold", va="top")
        ax.axis("off")
    for ax in np.ravel(axes)[len(frames):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(
    config: str = "final.toml",
    controller: str = "controller_v10_5_max.py",
    out: str = "report/figs_v10_5_max",
    montage_frames: int = 6,
    montage_width: int = 960,
    montage_height: int = 540,
) -> str:
    """Fly one episode and write the four flight-report figures.

    Args:
        config: Config file in config/ (track, seed, physics).
        controller: Controller file in lsy_drone_racing/control/.
        out: Output directory for the PNG figures.
        montage_frames: Number of MuJoCo snapshots in the montage.
        montage_width: Montage render width in pixels.
        montage_height: Montage render height in pixels.

    Returns:
        The output directory path.
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    traj = _fly(cfg, controller, montage_frames, montage_width, montage_height)
    out_dir = Path(__file__).parents[1] / out
    out_dir.mkdir(parents=True, exist_ok=True)

    name = controller.replace(".py", "")
    lap = traj["lap"]
    peak = traj["spd"].max()
    ng, no = len(traj["gates"]), len(traj["obstacles"])
    end = "finished" if traj["finished"] else "CRASHED"

    _plot_3d(traj, f"{config} - {name} flown racing line (3D)",
             out_dir / "fig_3d.png")
    _plot_topdown(traj, f"{config} - {name} flown racing line\n"
                        f"top-down | lap {lap:.2f}s ({end}) | {ng} gates, {no} obstacles",
                  out_dir / "fig_topdown.png")
    _plot_speed(traj, f"Flown speed profile | lap {lap:.2f}s | peak {peak:.2f} m/s",
                out_dir / "fig_speed.png")
    _plot_montage(traj, f"{name} flying the track (MuJoCo render, green = live plan)",
                  out_dir / "fig_montage.png")

    logger.info("Wrote 4 figures to %s (lap %.2fs, %s, peak %.2f m/s)",
                out_dir, lap, end, peak)
    return str(out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(report)
