"""
Episode termination: decides per env whether a stacking attempt succeeded or failed.

The check starts when an env's arm has retreated, i.e. when its plan reaches its final action
(the return home) or finishes. From then on, every sim step:
    - the source object must be resting on the destination, the destination must be upright and
      at its resting height, and every other object must be where it spawned and upright;
      otherwise the episode FAILS immediately ("unstacked", "destination moved", "disturbed").
    - the scene must hold still: no object may drift more than success.still_pos_tol or rotate
      more than success.still_rot_tol_deg over a window of success.hold_time seconds. Drift
      restarts the window. Completing the window is SUCCESS.
    - if the plan has ended and success.timeout seconds pass without a completed window, the
      episode FAILS ("not settled").
Tolerances are in config["success"].
"""

import numpy as np

from utils.frames import table_to_world, object_height
from utils.ik import to_numpy, quat_to_mat


def tilt_deg(quat):
    """Angle in degrees between the local z axis and world z, for (n, 4) quaternions."""
    R = quat_to_mat(np.asarray(quat, dtype=float))
    return np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))


def rot_angle_deg(q_a, q_b):
    """Angle in degrees between two batches of (n, 4) quaternions."""
    dot = np.abs(np.sum(np.asarray(q_a) * np.asarray(q_b), axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, -1.0, 1.0)))


class Result:
    __slots__ = ("success", "reason", "end_step")

    def __init__(self):
        self.success = None      # True / False, None while running
        self.reason = None       # "stacked" or the failure reason
        self.end_step = None     # runner.step_count when the episode terminated

    @property
    def terminated(self):
        return self.success is not None


class EpisodeMonitor:
    def __init__(self, objects, env_configs, runner, config):
        self.objects = objects
        self.names = list(objects)
        self.env_configs = env_configs
        self.runner = runner
        self.crit = config["success"]
        self.n = len(env_configs)
        self.dt = runner.dt
        self.results = [Result() for _ in range(self.n)]
        self.plan_end_step = [None] * self.n
        self.window_start = [None] * self.n
        self.window_ref = [None] * self.n          # (pos (O, 3), quat (O, 4)) at the window start
        # Where every object should rest if untouched, world frame, per env.
        self.spawn = np.array([
            [table_to_world(c, (*c["scene"]["objects"][o]["pos"], object_height(c["scene"]["objects"][o]) / 2))
             for o in self.names]
            for c in env_configs
        ])

    # ---- helpers ----

    def all_terminated(self):
        return all(r.terminated for r in self.results)

    def _poses(self):
        pos = np.stack([to_numpy(self.objects[o].get_pos()) for o in self.names], axis=1).astype(float)   # (n, O, 3)
        quat = np.stack([to_numpy(self.objects[o].get_quat()) for o in self.names], axis=1).astype(float) # (n, O, 4)
        return pos, quat

    def _retreated(self, i):
        plan = self.runner.plans[i]
        return self.runner.done[i] or self.runner.action_idx[i] >= max(len(plan) - 1, 0)

    def _stack_state(self, i, pos, quat):
        """None if the stack is as required, else a failure reason."""
        c = self.crit
        cfg = self.env_configs[i]
        specs = cfg["scene"]["objects"]
        s, d = self.names.index(cfg["task"]["source"]), self.names.index(cfg["task"]["destination"])
        tilt = tilt_deg(quat[i])

        delta = pos[i, s] - pos[i, d]
        dz_expected = object_height(specs[self.names[d]]) / 2 + object_height(specs[self.names[s]]) / 2
        if abs(delta[0]) > c["stack_xy_tol"] or abs(delta[1]) > c["stack_xy_tol"] or abs(delta[2] - dz_expected) > c["stack_z_tol"]:
            return "unstacked"
        if tilt[d] > c["upright_tol_deg"] or abs(pos[i, d, 2] - self.spawn[i, d, 2]) > c["rest_z_tol"]:
            return "destination moved"
        for o in range(len(self.names)):
            if o in (s, d):
                continue
            if np.linalg.norm(pos[i, o] - self.spawn[i, o]) > c["disturb_tol"] or tilt[o] > c["upright_tol_deg"]:
                return "disturbed"
        return None

    def _terminate(self, i, success, reason):
        r = self.results[i]
        r.success, r.reason, r.end_step = success, reason, self.runner.step_count

    # ---- per step ----

    def update(self):
        """Call after every runner.step()."""
        step = self.runner.step_count
        c = self.crit
        pos, quat = self._poses()

        for i in range(self.n):
            r = self.results[i]
            if r.terminated:
                continue
            if self.runner.done[i] and self.plan_end_step[i] is None:
                self.plan_end_step[i] = step
            if not self._retreated(i):
                continue

            problem = self._stack_state(i, pos, quat)
            if problem is not None:
                self._terminate(i, False, problem)
                continue

            ref = self.window_ref[i]
            drifted = ref is not None and (
                np.linalg.norm(pos[i] - ref[0], axis=-1).max() > c["still_pos_tol"]
                or rot_angle_deg(quat[i], ref[1]).max() > c["still_rot_tol_deg"]
            )
            if ref is None or drifted:
                self.window_start[i] = step
                self.window_ref[i] = (pos[i].copy(), quat[i].copy())
            elif (step - self.window_start[i]) * self.dt >= c["hold_time"]:
                self._terminate(i, True, "stacked")
                continue

            if self.plan_end_step[i] is not None and (step - self.plan_end_step[i]) * self.dt > c["timeout"]:
                self._terminate(i, False, "not settled")
