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


def _hms(seconds):
    m, s = divmod(int(max(seconds, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Progress:
    """One status line, rewritten in place, over the per-episode lines.

    Only a terminal gets the live line; redirected output and multi-worker runs fall back to a
    line per epoch, since several workers sharing one carriage return is unreadable.
    """

    def __init__(self, epochs, n_envs, enabled=True, every=0.25):
        self.epochs, self.n_envs, self.every = epochs, n_envs, every
        self.live = enabled and sys.stdout.isatty()
        self.t0 = time.time()
        self.last = 0.0
        self.width = 0
        self.saved = 0
        self.steps_done = 0     # steps of finished epochs; runner.step_count restarts each epoch

    def end_epoch(self, runner):
        self.steps_done += runner.step_count

    def _fraction(self, epoch, runner, monitor):
        """How far this epoch is, averaged over envs: a terminated env is done, a running one is
        as far as its plan has got. Capped below 1 while running so the bar cannot sit at full."""
        total = 0.0
        for i in range(self.n_envs):
            if monitor.results[i].terminated:
                total += 1.0
            else:
                total += 0.9 * runner.action_idx[i] / max(1, len(runner.plans[i]))
        return total / max(1, self.n_envs)

    def update(self, epoch, runner, monitor, force=False):
        if not self.live:
            return
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        frac = self._fraction(epoch, runner, monitor)
        overall = (epoch + frac) / self.epochs
        elapsed = now - self.t0
        eta = elapsed * (1 - overall) / overall if overall > 1e-6 else 0
        done = sum(r.terminated for r in monitor.results)
        # Steps per second of one env, the same quantity Genesis called "FPS per env"; times the
        # env count it is the total FPS Genesis reported.
        sps = (self.steps_done + runner.step_count) / elapsed if elapsed > 0 else 0
        bar = "#" * int(round(20 * frac)) + "." * (20 - int(round(20 * frac)))
        line = (f"epoch {epoch + 1}/{self.epochs} [{bar}] {100 * frac:3.0f}%  "
                f"done {done:>3}/{self.n_envs}  saved {self.saved:>4}  "
                f"sim {runner.time:5.1f}s  {sps * self.n_envs:5.0f} FPS ({sps:4.1f}/env)  "
                f"elapsed {_hms(elapsed)}  eta {_hms(eta)}")
        self.width = max(self.width, len(line))
        print("\r" + line.ljust(self.width), end="", flush=True)

    def message(self, text):
        """Print a line above the status line without leaving fragments behind."""
        if self.live:
            print("\r" + " " * self.width + "\r", end="")
        print(text, flush=True)


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

    # The live line only makes sense for a single process writing to a terminal.
    progress = Progress(epochs, n_envs, enabled=verbose and not tag)

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
                    progress.saved += bool(r.success)
                    attempts.append({
                        "epoch": epoch, "env": env_offset + i, "task": env_configs[i]["task"], "instruction": instructions[i],
                        "success": r.success, "reason": r.reason, "frames": int(recorder.n_frames[i]),
                        "sim_time_s": round(r.end_step * runner.dt, 2), "local_episode": local,
                        "episode_index": None, "config": env_configs[i],
                    })
                    if verbose:
                        # Only failures are worth a line each; successes are counted on the status
                        # line, and one line per success buries them over a few hundred episodes.
                        if not r.success:
                            progress.message(f"{tag}epoch {epoch + 1} env {env_offset + i}: FAILED ({r.reason})"
                                             f"  \"{instructions[i]}\"")
                        elif not progress.live:
                            print(f"{tag}epoch {epoch + 1} env {env_offset + i}: success "
                                  f"{recorder.n_frames[i]:4d} frames  \"{instructions[i]}\"", flush=True)
            progress.update(epoch, runner, monitor)
        progress.update(epoch, runner, monitor, force=True)
        progress.end_epoch(runner)
        if verbose:
            wall = time.time() - t_round
            ok = sum(1 for a in attempts if a["epoch"] == epoch and a["success"])
            progress.message(f"{tag}epoch {epoch + 1}/{epochs}: {ok}/{n_envs} saved, "
                             f"{runner.time:.1f}s of sim in {wall:.1f}s wall "
                             f"({n_envs / wall * 60:.1f} episodes/min)")

    if verbose:
        progress.message(f"{tag}finalizing videos and per-env datasets...")
    return attempts, recorder.close()


def worker_main(worker, session_dir, verbose):
    """One recorder process of a multi-worker session: run its rounds, then hand results to the parent."""
    config = copy.deepcopy(CONFIG)
    session = Session(config, project_version(), SOURCE_DIR, existing=session_dir)
    env_offset = worker * config["ik"]["n_envs"]
    attempts, counts = run_rounds(config, session, env_offset, show_viewer=False, verbose=verbose, tag=f"[w{worker}] ")
    (session.dir / f"worker{worker}.json").write_text(json.dumps({
        "attempts": to_jsonable(attempts), "episode_counts": {str(k): v for k, v in counts.items()},
    }))


def main(show_viewer=None, verbose=True):
    """show_viewer overrides config["viewer"]["show"] when given."""
    config = copy.deepcopy(CONFIG)
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
