"""
One rollout with the policy's compute time on the clock.

The simulator runs on simulated time t; a prediction launched at t with wall-clock latency L lands
at t + L (real-time mode) or at t (lockstep mode, the task 2 protocol, where the simulator waits).
Between launch and landing the arm holds its last command, which is the stall the speed
optimisations are meant to remove. Predictions run on engine threads, one CUDA stream each; the
controller advances the physics one step at a time and, before each step, waits on the wall clock
only as long as a pending prediction could still land inside that step, so the simulated outcome
depends on the measured latencies and nothing else about wall time.

Schedules:
    sync        one prediction at a time: observe, wait, apply, execute one control period, observe
    pipelined   a prediction launched every control period on the newest frame while an engine is free

What is recorded, per rollout: every launch and landing (simulated time, latency, staleness), the
joint positions and tool position at every physics step, the control-period ticks, the Scorer's
outcome, and a video of the front camera sampled at a fixed simulated rate (so a stall is visible
as the arm standing still, with a WAITING tag while it does). metrics() turns the traces into the
smoothness numbers:
    stall_s / stall_frac     simulated time with no action to execute while a prediction was pending
    action_hz                actions applied per simulated second
    jerk_rms, accel_rms      joint-space, rad/s^3 and rad/s^2, finite differences at the physics rate,
                             over the steps where the arm is moving (a held arm has no jerk, and
                             would make a stalling run look smooth); jerk_p99 is the peak
    dv_boundary / dv_within  joint-speed change over one physics step at the steps where a new action
                             landed, against all other steps (p95 of each)
    duty                     fraction of physics steps with the tool moving faster than 1 cm/s
    stops_per_s              how often the tool comes to rest and starts again: stop-and-go
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from evaluate import Scorer

STILL_SPEED = 0.01     # m/s: below this the tool counts as stationary (duty cycle)


class Done:
    """A finished prediction with a future's interface, for the single-engine case, where the
    prediction runs on the controller's own thread (compiled CUDA graphs are bound to the thread
    that recorded them, and nothing else is going on meanwhile anyway)."""

    def __init__(self, value):
        self.value = value

    def result(self, timeout=None):
        return self.value


class Controller:
    def __init__(self, sim, instruction, predictor_factory, compose, crit, specs, timing="realtime", schedule="sync",
                 engines=1, control_hz=10.0, horizon_actions=100, max_sim_s=60.0, filter_factory=None,
                 video_fps=30.0, video_camera="front", label="", latency=None, hold="pose"):
        self.sim, self.instruction, self.compose = sim, instruction, compose
        # latency: None puts each prediction's measured wall-clock latency on the simulated clock;
        # a number (seconds) puts that fixed latency on it instead (the prediction still runs and
        # its real latency is recorded), which makes a run reproducible on any machine and lets
        # success be measured against the stall length directly.
        self.latency = latency
        # hold: what the arm does between an action landing and the next one.
        #   pose      the command jumps to the action's target at once and holds there: the stall
        #   stretch   the command moves to the target at constant speed over the expected interval
        #             between landings, so the arm keeps moving at a steady pace
        #   velocity  like stretch, but it keeps going at that speed past the target until the next
        #             action lands (capped at 1.5 times the action)
        #   freeze    the arm is held at its measured joints from the moment the frame is taken until
        #             the action lands, then the command is restored and the action applied to it.
        #             The policy learned the controller's tracking lag (the arm trails the command
        #             in every recording); settling onto the command during a wait removes that lag
        #             and shifts every decision by it, freezing keeps it.
        self.hold = hold
        # The video is sampled on the simulated clock, not at the policy's cadence: a stall shows as
        # the arm standing still for as many frames as it stood still.
        self.video_every = max(1, int(round((1.0 / video_fps) / float(sim.cfg["sim"]["dt"]))))
        self.video_camera, self.label = video_camera, label
        self.crit, self.specs = crit, specs
        self.timing, self.schedule, self.engines = timing, schedule, max(1, engines)
        self.T = 1.0 / control_hz
        self.dt = float(sim.cfg["sim"]["dt"])
        self.steps_per_control = max(1, int(round(self.T / self.dt)))
        self.horizon_actions, self.max_sim_s = horizon_actions, max_sim_s
        self.predictor_factory = predictor_factory
        self.filter_factory = filter_factory
        self._local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=self.engines)
        self._lock = threading.Lock()
        self._predictors = []

    def _predict(self, images, text):
        """Runs on an engine thread: its own predictor and CUDA stream; returns (action, latency_s).
        The latency covers everything the policy does with a camera frame: composing the view,
        preprocessing, prefill, decoding, un-normalizing."""
        if not hasattr(self._local, "predictor"):
            self._local.predictor = self.predictor_factory()
            # several engines: each on its own stream so their kernels can overlap; one engine: the default stream
            self._local.stream = torch.cuda.Stream() if torch.cuda.is_available() and self.engines > 1 else None
            with self._lock:
                self._predictors.append(self._local.predictor)
        predictor = self._local.predictor
        t0 = time.perf_counter()
        canvas = predictor.compose(images)
        t1 = time.perf_counter()
        if self._local.stream is not None:
            with torch.cuda.stream(self._local.stream):
                action = predictor.predict(canvas, text)
            self._local.stream.synchronize()
        else:
            action = predictor.predict(canvas, text)
        t2 = time.perf_counter()
        self._local.last_ms = {"compose": round((t1 - t0) * 1000, 1), **getattr(predictor, "last_ms", {"predict": round((t2 - t1) * 1000, 1)})}
        return np.asarray(action, dtype=float), t2 - t0

    def run(self, layout, start_yaw="home"):
        sim, T, dt = self.sim, self.T, self.dt
        sim.episode = layout - 1
        sim.reset()
        if start_yaw == "aligned":
            sim.align_yaw(self.instruction["source"])
        for p in self._predictors:
            if hasattr(p, "reset"):
                p.reset()
        filt = self.filter_factory() if self.filter_factory else None
        scorer = Scorer(sim, self.instruction["source"], self.instruction["destination"], self.crit, self.specs)

        t = 0.0
        pending = []                    # [launch_t, future, wall clock at launch]
        next_launch = 0.0               # simulated time of the next allowed launch
        exec_until = 0.0                # the last applied action is being executed until this time
        segment = None                  # hold stretch/velocity: the action being traced out, see _advance
        frozen = None                   # hold freeze: the command set aside while the arm is held still
        gaps = []                       # simulated time between consecutive landings, for the expected interval
        applied = predictions = 0
        stall_s = 0.0
        events = []                     # dicts: launch_t, land_t, latency_s, staleness_s
        q_trace, tool_trace, t_trace, landed_step = [], [], [], []
        frames = []                     # the video: the camera at a fixed simulated rate, annotated
        step = 0
        wall0 = time.perf_counter()
        outcome_reached = False
        while applied < self.horizon_actions and t < self.max_sim_s and not outcome_reached:
            # ---- launch
            free = len(pending) < (1 if self.schedule == "sync" else self.engines)
            if free and t + 1e-9 >= next_launch:
                obs = sim.observation()
                if self.hold == "freeze" and self.timing != "lockstep" and not pending:
                    frozen = (sim.cmd_pos.copy(), sim.cmd_quat.copy())
                    q_now = sim._to_array(sim.robot.get_dofs_position(sim.arm_dofs))
                    sim.robot.control_dofs_position(q_now, sim.arm_dofs)   # the arm stays put, lag and all; the fingers keep closing
                wall = time.perf_counter()
                if self.engines == 1:
                    future = Done(self._predict(obs["images"], self.instruction["text"]))
                else:
                    future = self.pool.submit(self._predict, obs["images"], self.instruction["text"])
                pending.append([t, future, wall])
                predictions += 1
                next_launch = t + T if self.schedule == "pipelined" else np.inf

            # ---- which pending predictions land within this physics step
            landing = []
            for launch_t, future, wall_launch in pending:
                if self.timing == "lockstep":
                    action, latency = future.result()
                    landing.append((launch_t, launch_t, action, latency))
                    continue
                if self.latency is not None:                       # a fixed latency on the clock
                    if launch_t + self.latency <= t + dt + 1e-9:
                        action, latency = future.result()
                        landing.append((launch_t + self.latency, launch_t, action, latency))
                    continue
                # It lands in this step iff its latency is at most (t + dt - launch_t). With one
                # prediction at a time nothing else happens meanwhile, so wait for it outright and
                # let the physics catch up afterwards (the GPU is then the policy's alone, as on a
                # robot); with several in flight, wait only for what is left of that budget, then it
                # is either here or not.
                if self.schedule == "sync":
                    action, latency = future.result()
                else:
                    budget = (t + dt - launch_t) - (time.perf_counter() - wall_launch)
                    try:
                        action, latency = future.result(timeout=max(0.0, budget))
                    except TimeoutError:
                        continue
                land_t = launch_t + latency
                if land_t <= t + dt + 1e-9:
                    landing.append((land_t, launch_t, action, latency))
            for land_t, launch_t, action, latency in sorted(landing, key=lambda x: x[0]):
                pending = [p for p in pending if p[0] != launch_t]
                if filt is not None:
                    action = filt(action)
                if events:
                    gaps.append(max(t, land_t) - events[-1]["land_t"])
                if self.hold == "freeze":
                    if frozen is not None:
                        sim.cmd_pos, sim.cmd_quat = frozen       # the command the policy assumed, not where the arm stopped
                        frozen = None
                    sim.command(action)
                elif self.hold == "pose":
                    sim.command(action)
                else:
                    segment = self._segment(action, t, gaps, T, latency)
                    self._advance(segment, t)
                applied += 1
                exec_until = max(t, land_t) + T
                events.append({"launch_t": round(launch_t, 4), "land_t": round(land_t, 4), "latency_s": round(latency, 4),
                               "staleness_s": round(max(t, land_t) - launch_t, 4), "step": step,
                               "action": [round(float(v), 5) for v in action],
                               "ms": getattr(self._local, "last_ms", None) if self.engines == 1 else None})
                landed_step.append(step)
                if self.schedule == "sync":
                    next_launch = exec_until

            # ---- one physics step
            if segment is not None:
                self._advance(segment, t)
                busy = segment["frac"] < 1.0
            else:
                busy = t < exec_until - 1e-9
            waiting = bool(pending) and not busy
            if waiting:
                stall_s += dt            # waiting for a prediction with nothing left to execute
            if step % self.video_every == 0:
                frames.append(self._frame(t, applied, waiting, events[-1]["latency_s"] if events else None))
            sim.scene.step()
            t += dt
            step += 1
            q, tool = sim.joints()
            q_trace.append(q); tool_trace.append(tool); t_trace.append(t)
            if step % self.steps_per_control == 0:
                if scorer.update(applied, t, tool, sim.cmd_grip):
                    outcome_reached = True

        for _, future, _ in pending:     # drain, so the engines are idle before the next rollout
            try:
                future.result()
            except Exception:
                pass
        record = {"layout": layout, "source": self.instruction["source"], "destination": self.instruction["destination"],
                  "instruction": self.instruction["text"], **scorer.finish(),
                  "actions": applied, "predictions": predictions, "sim_s": round(t, 3), "wall_s": round(time.perf_counter() - wall0, 1),
                  "stall_s": round(stall_s, 3), "timing": self.timing, "schedule": self.schedule, "engines": self.engines,
                  "hold": self.hold, "latency_model": "measured" if self.latency is None else f"fixed {self.latency * 1000:.0f} ms"}
        traces = {"t": np.array(t_trace), "q": np.array(q_trace), "tool": np.array(tool_trace),
                  "landed_step": np.array(landed_step, dtype=int), "events": events}
        frames.append(self._frame(t, applied, False, events[-1]["latency_s"] if events else None, final=scorer.outcome))
        record.update(metrics(traces, self.dt))
        record["latencies_s"] = [e["latency_s"] for e in events]
        lat = np.array(record["latencies_s"]) * 1000
        stale = np.array([e["staleness_s"] for e in events]) * 1000
        breakdown = [e["ms"] for e in events if e.get("ms")]
        record.update({
            "latency_breakdown_ms": {k: round(float(np.median([b[k] for b in breakdown])), 1) for k in breakdown[0]} if breakdown else None,
            "latency_ms_p50": round(float(np.median(lat)), 1) if len(lat) else None,
            "latency_ms_p99": round(float(np.percentile(lat, 99)), 1) if len(lat) else None,
            "latency_ms_max": round(float(lat.max()), 1) if len(lat) else None,
            "staleness_ms_p50": round(float(np.median(stale)), 1) if len(stale) else None,   # age of the frame an action came from, when applied
            "sim_s_to_success": round(t, 3) if record["success"] else None,
            "actions_to_success": applied if record["success"] else None,
        })
        return record, traces, frames

    def _segment(self, action, t, gaps, T, latency):
        """The action as a motion to trace out from the current command: its translation and
        rotation, spread over the expected time until the next landing (the median of the last
        landing gaps; before any, one control period plus the latency just seen for a single
        engine, one period when several are in flight)."""
        from scipy.spatial.transform import Rotation

        if len(gaps) >= 3:
            duration = float(np.median(gaps[-5:]))
        else:
            duration = T + (self.latency if self.latency is not None else latency) if self.schedule == "sync" else T
        sim = self.sim
        grip = action[6]
        if sim.gripper_mode == "threshold":
            grip = 1.0 if grip >= 0.5 else 0.0
        return {"t0": t, "duration": max(duration, T), "frac": 0.0,
                "pos0": sim.cmd_pos.copy(), "delta": np.asarray(action[:3], dtype=float),
                "rot0": Rotation.from_quat(sim.cmd_quat[[1, 2, 3, 0]]), "rotvec": np.asarray(action[3:6], dtype=float), "grip": grip}

    def _advance(self, seg, t):
        """Move the command along the segment to where it should be at time t."""
        from scipy.spatial.transform import Rotation

        frac = (t - seg["t0"]) / seg["duration"]
        frac = min(frac, 1.0) if self.hold == "stretch" else min(frac, 1.5)
        seg["frac"] = frac
        pos = seg["pos0"] + frac * seg["delta"]
        quat = (seg["rot0"] * Rotation.from_rotvec(frac * seg["rotvec"])).as_quat()[[3, 0, 1, 2]]
        self.sim.command_pose(pos, quat, seg["grip"])

    def _frame(self, t, applied, waiting, latency, final=None):
        """The camera image with the clock, the action count, the last latency and, while the arm
        waits for a prediction with nothing to execute, a red WAITING tag."""
        from PIL import Image, ImageDraw

        image = Image.fromarray(np.ascontiguousarray(self.sim.observation()["images"][self.video_camera])).convert("RGB")
        draw = ImageDraw.Draw(image)
        lines = [f"{self.label}  {self.timing}", f"t = {t:5.2f} s   actions {applied:3d}" + (f"   last latency {latency * 1000:4.0f} ms" if latency else "")]
        if final:
            lines.append(final)
        draw.rectangle([0, 0, 330, 14 * len(lines) + 6], fill=(0, 0, 0))
        for k, line in enumerate(lines):
            draw.text((4, 3 + 14 * k), line, fill=(255, 255, 255))
        if waiting:
            draw.rectangle([image.width - 110, 0, image.width, 20], fill=(200, 0, 0))
            draw.text((image.width - 100, 4), "WAITING", fill=(255, 255, 255))
        return np.asarray(image)

    def close(self):
        self.pool.shutdown(wait=True)


def metrics(traces, dt):
    """Smoothness numbers from one rollout's traces (see the module docstring)."""
    q, tool, landed = traces["q"], traces["tool"], traces["landed_step"]
    n = len(q)
    if n < 4:
        return {}
    v = np.diff(q, axis=0) / dt                      # (n-1, 8) joint velocities
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    speed = np.linalg.norm(v, axis=1)                # joint-space speed per step
    dv = np.abs(np.diff(speed))                      # change of joint speed over one physics step
    boundary = np.zeros(len(dv), dtype=bool)
    for s in landed:                                 # the step at which an action landed: its effect shows over the next steps
        boundary[max(0, s - 1):min(len(dv), s + 2)] = True
    tool_speed = np.linalg.norm(np.diff(tool, axis=0), axis=1) / dt
    moving = tool_speed > STILL_SPEED                # (n-1,)
    moving_a, moving_j = moving[1:], moving[2:]      # aligned with a and j
    starts = int(np.sum(moving[1:] & ~moving[:-1]))  # rest -> motion transitions
    out = {
        "sim_s": round(float(traces["t"][-1]), 3),
        "action_hz": round(len(landed) / max(float(traces["t"][-1]), 1e-6), 2),
        "jerk_rms": round(float(np.sqrt(np.mean(j[moving_j] ** 2))) if moving_j.any() else 0.0, 1),
        "jerk_p99": round(float(np.percentile(np.abs(j[moving_j]), 99)) if moving_j.any() else 0.0, 1),
        "accel_rms": round(float(np.sqrt(np.mean(a[moving_a] ** 2))) if moving_a.any() else 0.0, 2),
        "speed_mean": round(float(speed.mean()), 3),
        "dv_boundary_p95": round(float(np.percentile(dv[boundary], 95)) if boundary.any() else 0.0, 3),
        "dv_within_p95": round(float(np.percentile(dv[~boundary], 95)) if (~boundary).any() else 0.0, 3),
        "duty": round(float(np.mean(moving)), 3),
        "stops_per_s": round(starts / max(float(traces["t"][-1]), 1e-6), 2),
        "tool_path_m": round(float(np.sum(np.linalg.norm(np.diff(tool, axis=0), axis=1))), 3),
    }
    return out
