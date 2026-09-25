"""
Run motion plans from config in the viewer

    python source/run_plan.py                      every env of ik.n_envs, epoch 0
    python source/run_plan.py --env 5 --episode 0  replay one attempt from a recording (its env and epoch
                                                   numbers from episodes.json) as a single-env scene

The layout and task of an attempt are a pure function of (seed, env, episode), so a replay reproduces
it exactly, provided ik.n_envs and record.workers match the session (they set the round-robin pair order).
"""

import argparse
import copy

import numpy as np

from config import CONFIG, env_config, instruction
from stacking_scene import build_scene, init_genesis, reset_envs, robot_dofs
from utils.episode import EpisodeMonitor
from utils.ik import to_numpy
from utils.motion import PlanRunner


def main(show_viewer=None, verbose=True, env=None, episode=0):
    """show_viewer overrides config["viewer"]["show"] when given. env=None runs all envs of ik.n_envs."""
    config = copy.deepcopy(CONFIG)
    if show_viewer is None:
        show_viewer = config["viewer"]["show"]
    env_ids = list(range(config["ik"]["n_envs"])) if env is None else [env]
    n_envs = len(env_ids)

    init_genesis(config)
    scene, robot, table, objects, cameras = build_scene(config, show_viewer=show_viewer)
    scene.build(n_envs=n_envs, env_spacing=config["ik"]["env_spacing"])
    dofs_idx = robot_dofs(robot, config)
    ee_link = robot.get_link(config["ik"]["ee_link"])

    env_configs = [env_config(config, i, episode) for i in env_ids]
    reset_envs(robot, objects, env_configs)
    if verbose:
        for i, c in zip(env_ids, env_configs):
            objs = {n: (round(s["pos"][0], 3), round(s["pos"][1], 3), round(s["yaw"], 1)) for n, s in c["scene"]["objects"].items()}
            print(f"env {i} episode {episode}: \"{instruction(c['task'])}\"  objects (x, y, yaw): {objs}")

    runner = PlanRunner(scene, robot, ee_link, env_configs, dofs_idx, config, verbose=verbose)
    monitor = EpisodeMonitor(objects, env_configs, runner, config)
    while not monitor.all_terminated():
        runner.step()
        monitor.update()
    for _ in range(int(round(config["sim"]["hold_after_plan"] / config["sim"]["dt"]))):
        scene.step()

    if verbose:
        for k, r in enumerate(monitor.results):
            status = "success" if r.success else f"FAILED ({r.reason})"
            print(f"env {env_ids[k]}: {status} at t = {r.end_step * runner.dt:.2f}s")
        for name, ent in objects.items():
            pos = to_numpy(ent.get_pos())
            print(f"  {name}: " + "  ".join(f"env{env_ids[k]} {np.round(pos[k], 3).tolist()}" for k in range(n_envs)))

    return scene, robot, objects, cameras, runner, monitor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run motion plans in the viewer, or replay one recorded attempt.")
    parser.add_argument("--env", type=int, help="replay this env number only (from episodes.json)")
    parser.add_argument("--episode", type=int, default=0, help="epoch number of the attempt (default 0)")
    parser.add_argument("--headless", action="store_true", help="run without the viewer")
    args = parser.parse_args()
    main(show_viewer=False if args.headless else None, env=args.env, episode=args.episode)
