"""Deployment-faithful evaluation.

Same shape as `evaluate.py` (runs N episodes, reports average time + success rate), but on
`level2_deploy.toml` — the measured-track scenario the real `save_track_as_config.py` +
`deploy.py` flow produces (no gate/obstacle placement perturbation). Use this to judge the
controller you actually fly; `evaluate.py` grades the harder *unmeasured*-track scenario.

    python scripts/eval_deploy.py                       # uses config's controller.file
    python scripts/eval_deploy.py --controller gate_search_v6.py --n_runs 20
"""

import logging
from pathlib import Path

import fire
import numpy as np
from sim import simulate

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main(controller: str | None = None, n_runs: int = 20, config_file: str = "level2_deploy.toml"):
    """Run the deployment-faithful config N times and report success rate + average time."""
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    ctrl = controller or config.controller.file
    ep_times = simulate(config=config_file, controller=ctrl, n_runs=n_runs, render=False)

    n_failed = len([x for x in ep_times if x is None])
    success_rate = 1 - n_failed / n_runs
    finished = [x for x in ep_times if x is not None]
    avg = float(np.mean(finished)) if finished else float("nan")
    logger.info(f"Controller: {ctrl}  config: {config_file}")
    logger.info(f"Success Rate: {success_rate * 100:.1f}%  ({n_runs - n_failed}/{n_runs})")
    logger.info(f"Average Time (successful): {avg:.3f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("lsy_drone_racing").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
