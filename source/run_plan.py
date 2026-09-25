"""
Run every environment's motion plan from config["plan"] on the stacking scene.

Run from anywhere:  python source/run_plan.py
"""

import numpy as np
import genesis as gs

from config import CONFIG
from stacking_scene import build_scene, init_genesis, make_env_configs, apply_env_configs, save_camera_previews
from utils.ik import to_numpy
from utils.motion import PlanRunner


def main(show_viewer=True, verbose=True):
    config = CONFIG
    n_envs = config["ik"]["n_envs"]

    init_genesis(config)
    scene, robot, table, objects, cameras = build_scene(config, show_viewer=show_viewer)
    scene.build(n_envs=n_envs, env_spacing=config["ik"]["env_spacing"])

    env_configs = make_env_configs(config, n_envs)
    dofs_idx = apply_env_configs(robot, objects, env_configs)
    ee_link = robot.get_link(config["ik"]["ee_link"])

    runner = PlanRunner(scene, robot, ee_link, env_configs, dofs_idx, config, verbose=verbose)
    runner.run()

    for _ in range(int(round(config["sim"]["hold_after_plan"] / config["sim"]["dt"]))):
        scene.step()

    if verbose:
        print(f"plans finished at t = {runner.time:.2f}s")
        for name, ent in objects.items():
            pos = to_numpy(ent.get_pos())
            print(f"  {name}: " + "  ".join(f"env{i} {np.round(pos[i], 3).tolist()}" for i in range(n_envs)))

    for path in save_camera_previews(cameras, config["preview"]["dir"]):
        print(f"saved {path}")
    if config["preview"]["record_video"]:
        scene.stop_recording()
        print(f"videos written to {config['preview']['video_dir']}")

    return scene, robot, objects, cameras, runner


if __name__ == "__main__":
    main()
