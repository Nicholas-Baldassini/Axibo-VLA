"""
Record demonstrations into a LeRobot v3 dataset session.

Runs record.epochs rounds. In every round each env gets a fresh config from env_config (random
task pair and object placement), is reset, executes its plan, and is judged by the episode monitor:
success when the stack holds still for success.hold_time after the arm retreats, failure otherwise.
Frames stream into one writer per env at record.fps while the simulation runs; a successful
attempt is saved as an episode the moment it terminates and a failed one is discarded. The per-env
datasets are merged into the session dataset at the end.

Run from anywhere:  python source/record.py
"""

import copy
import time

from config import CONFIG, SOURCE_DIR, env_config, instruction
from stacking_scene import build_scene, init_genesis, reset_envs, robot_dofs
from utils.episode import EpisodeMonitor
from utils.motion import PlanRunner
from utils.recorder import Recorder, Session


def project_version():
    path = SOURCE_DIR.parent / "VERSION"
    return path.read_text().strip() if path.exists() else "unknown"


def main(show_viewer=False, verbose=True):
    config = copy.deepcopy(CONFIG)
    config["preview"]["record_video"] = False        # the dataset's videos replace the preview recorders
    n_envs = config["ik"]["n_envs"]
    epochs = config["record"]["epochs"]
    session = Session(config, project_version(), SOURCE_DIR)
    t0 = time.time()

    init_genesis(config)
    scene, robot, table, objects, cameras = build_scene(config, show_viewer=show_viewer)
    scene.build(n_envs=n_envs, env_spacing=config["ik"]["env_spacing"])
    dofs_idx = robot_dofs(robot, config)
    ee_link = robot.get_link(config["ik"]["ee_link"])
    recorder = Recorder(robot, objects, cameras, dofs_idx, config, session)

    attempts = []
    for epoch in range(epochs):
        t_round = time.time()
        env_configs = [env_config(config, i, epoch) for i in range(n_envs)]
        instructions = [instruction(c["task"]) for c in env_configs]
        reset_envs(robot, objects, env_configs)
        runner = PlanRunner(scene, robot, ee_link, env_configs, dofs_idx, config, verbose=False, hook=recorder.hook)
        recorder.start_round(runner, instructions)
        monitor = EpisodeMonitor(objects, env_configs, runner, config)

        handled = set()
        while not monitor.all_terminated():
            runner.step()
            monitor.update()
            for i, r in enumerate(monitor.results):
                if r.terminated and i not in handled:
                    handled.add(i)
                    local = recorder.end_episode(i, r.success)
                    attempts.append({
                        "epoch": epoch, "env": i, "task": env_configs[i]["task"], "instruction": instructions[i],
                        "success": r.success, "reason": r.reason, "frames": int(recorder.n_frames[i]),
                        "sim_time_s": round(r.end_step * runner.dt, 2), "local_episode": local,
                        "episode_index": None, "config": env_configs[i],
                    })
                    if verbose:
                        status = "success" if r.success else f"FAILED ({r.reason})"
                        print(f"epoch {epoch} env {i}: {status:24s} {recorder.n_frames[i]:4d} frames  \"{instructions[i]}\"")
        if verbose:
            print(f"epoch {epoch}: {runner.time:.1f}s of sim in {time.time() - t_round:.1f}s wall")

    t_fin = time.time()
    offsets = recorder.finalize()
    for a in attempts:
        if a["local_episode"] is not None:
            a["episode_index"] = offsets[a["env"]] + a["local_episode"]
    metadata = session.write(config, attempts, time.time() - t0)

    if verbose:
        reasons = ", ".join(f"{k}: {v}" for k, v in metadata["failure_reasons"].items()) or "none"
        print(f"merged dataset in {time.time() - t_fin:.1f}s")
        print(f"session: {session.dir}")
        print(f"{metadata['episodes_attempted']} episodes attempted, {metadata['episodes_succeeded']} succeeded, "
              f"{metadata['episodes_failed']} failed (reasons: {reasons})")
        print(f"{metadata['frames']} frames written at {metadata['fps']} fps, wall time {metadata['wall_time_s']}s")
    return session, attempts


if __name__ == "__main__":
    main()
