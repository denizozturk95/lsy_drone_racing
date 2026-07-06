"""Render a controller's flight to numbered PNG frames.

Runs one episode like scripts/sim.py, but instead of opening the GUI it grabs an
offscreen RGB frame every few steps and writes them as frame_XXXX.png. Defaults
reproduce the report_v2 flight: controller_v10_5_max on the fixed final.toml track.

    $ python scripts/flight_frames.py                       # -> flight_frames_level3/
    $ python scripts/flight_frames.py --config level2.toml --fps 20

Stitch to a video with:  ffmpeg -r 10 -i flight_frames_level3/frame_%04d.png flight.mp4
"""

from __future__ import annotations

import logging
from pathlib import Path

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from PIL import Image

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)


def _capture(env, width: int, height: int):
    """Sync MuJoCo and return one offscreen RGB frame (mirrors race_core.render)."""
    u = env.unwrapped
    if not u.data.sim_data.core.mjx_synced:
        u.data, u.sim.mjx_data = u._render_sync(u.data, u.sim.mjx_data)
    return u.sim.render(
        mode="rgb_array",
        camera=u.settings.camera,
        cam_config=u.settings.cam_config,
        width=width,
        height=height,
    )


def render_frames(
    config: str = "final.toml",
    controller: str = "controller_v10_5_max.py",
    out: str = "flight_frames_level3",
    fps: float = 10.0,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Fly one episode and dump PNG frames.

    Args:
        config: Config file in config/ (defines track, seed, physics).
        controller: Controller file in lsy_drone_racing/control/.
        out: Output directory for the PNG frames.
        fps: Frames saved per simulated second (env runs at config.env.freq).
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        The output directory path.
    """
    config = load_config(Path(__file__).parents[1] / "config" / config)
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control" / controller
    controller_cls = load_controller(control_path)

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

    out_dir = Path(__file__).parents[1] / out
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):  # start clean so old frames don't linger
        old.unlink()

    stride = max(1, round(config.env.freq / fps))  # save every `stride` env steps

    obs, info = env.reset()
    ctrl = controller_cls(obs, info, config)
    i = frames = 0
    while True:
        action = ctrl.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
        if i % stride == 0:
            rgb = _capture(env, width, height)
            Image.fromarray(rgb).save(out_dir / f"frame_{frames:04d}.png")
            frames += 1
        if terminated or truncated or finished:
            break
        i += 1

    env.close()
    gates = obs["target_gate"]
    logger.info(
        "Wrote %d frames to %s (flight time %.2fs, gate %s)",
        frames, out_dir, i / config.env.freq, "DONE" if gates == -1 else gates,
    )
    return str(out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(render_frames)
