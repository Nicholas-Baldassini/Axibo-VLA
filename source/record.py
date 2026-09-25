"""
Record demonstrations into a LeRobot v3 dataset session.

Runs record.epochs rounds. In every round each env gets a fresh config from env_config (random
task pair and object placement), is reset, executes its plan, and is judged by the episode monitor:
success when the stack holds still for success.hold_time after the arm retreats, failure otherwise.
Frames stream into one writer per env at record.fps while the simulation runs; a successful
attempt is saved as an episode the moment it terminates and a failed one is discarded.

record.workers > 1 runs that many copies of the simulation in separate processes, each with
ik.n_envs envs and globally unique env numbers (so every env draws different tasks and layouts),
and merges all of their per-env datasets into the session dataset at the end.

Run from anywhere:  python source/record.py
"""

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

from config import CONFIG, SOURCE_DIR, env_config, instruction
from stacking_scene import build_scene, init_genesis, reset_envs, robot_dofs
from utils.episode import EpisodeMonitor
from utils.motion import PlanRunner
from utils.recorder import Recorder, Session, merge_session, to_jsonable


def project_version():
    path = SOURCE_DIR.parent / "VERSION"
    return path.read_text().strip() if path.exists() else "unknown"


def run_rounds(config, session, env_offset=0, show_viewer=False, verbose=True, tag=""):
    """Simulate record.epochs rounds of ik.n_envs envs, streaming successes into per-env datasets.
    Returns (attempts, episode_counts) with envs numbered from env_offset."""
    n_envs = config["ik"]["n_envs"]
    epochs = config["record"]["epochs"]

    init_genesis(config)
    scene, robot, table, objects, cameras = build_scene(config, show_viewer=show_viewer)
    scene.build(n_envs=n_envs, env_spacing=config["ik"]["env_spacing"])
    dofs_idx = robot_dofs(robot, config)
    ee_link = robot.get_link(config["ik"]["ee_link"])
    recorder = Recorder(robot, objects, cameras, dofs_idx, config, session, env_offset=env_offset)

    attempts = []
    for epoch in range(epochs):
        t_round = time.time()
        env_configs = [env_config(config, env_offset + i, epoch) for i in range(n_envs)]
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
                        "epoch": epoch, "env": env_offset + i, "task": env_configs[i]["task"], "instruction": instructions[i],
                        "success": r.success, "reason": r.reason, "frames": int(recorder.n_frames[i]),
                        "sim_time_s": round(r.end_step * runner.dt, 2), "local_episode": local,
                        "episode_index": None, "config": env_configs[i],
                    })
                    if verbose:
                        status = "success" if r.success else f"FAILED ({r.reason})"
                        print(f"{tag}epoch {epoch} env {env_offset + i}: {status:24s} {recorder.n_frames[i]:4d} frames  \"{instructions[i]}\"", flush=True)
        if verbose:
            print(f"{tag}epoch {epoch}: {runner.time:.1f}s of sim in {time.time() - t_round:.1f}s wall", flush=True)

    return attempts, recorder.close()


def worker_main(worker, session_dir, verbose):
    """One recorder process of a multi-worker session: run its rounds, then hand results to the parent."""
    config = copy.deepcopy(CONFIG)
    config["preview"]["record_video"] = False
    session = Session(config, project_version(), SOURCE_DIR, existing=session_dir)
    env_offset = worker * config["ik"]["n_envs"]
    attempts, counts = run_rounds(config, session, env_offset, show_viewer=False, verbose=verbose, tag=f"[w{worker}] ")
    (session.dir / f"worker{worker}.json").write_text(json.dumps({
        "attempts": to_jsonable(attempts), "episode_counts": {str(k): v for k, v in counts.items()},
    }))


def main(show_viewer=None, verbose=True):
    """show_viewer overrides config["viewer"]["show"] when given."""
    config = copy.deepcopy(CONFIG)
    config["preview"]["record_video"] = False        # the dataset's videos replace the preview recorders
    if show_viewer is None:
        show_viewer = config["viewer"]["show"]
    workers = config["record"].get("workers", 1)
    session = Session(config, project_version(), SOURCE_DIR)
    t0 = time.time()

    if workers <= 1:
        attempts, counts = run_rounds(config, session, 0, show_viewer=show_viewer, verbose=verbose)
    else:
        if show_viewer and verbose:
            print(f"{workers} workers: running headless (the viewer needs a single process)")
        procs = [
            subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(w),
                              "--session", str(session.dir)] + ([] if verbose else ["--quiet"]))
            for w in range(workers)
        ]
        failed = [w for w, p in enumerate(procs) if p.wait() != 0]
        if failed:
            raise RuntimeError(f"worker(s) {failed} exited with an error; see their output above")
        attempts, counts = [], {}
        for w in range(workers):
            data = json.loads((session.dir / f"worker{w}.json").read_text())
            attempts += data["attempts"]
            counts.update({int(k): v for k, v in data["episode_counts"].items()})
            (session.dir / f"worker{w}.json").unlink()
        attempts.sort(key=lambda a: (a["epoch"], a["env"]))

    t_fin = time.time()
    offsets = merge_session(session, config["record"]["repo_id"], counts)
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
    parser = argparse.ArgumentParser(description="Record demonstrations into a LeRobot dataset session.")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)      # internal: spawned by main()
    parser.add_argument("--session", help=argparse.SUPPRESS)
    parser.add_argument("--quiet", action="store_true", help="print only the final summary")
    args = parser.parse_args()
    if args.worker is not None:
        worker_main(args.worker, args.session, verbose=not args.quiet)
    else:
        main(verbose=not args.quiet)
