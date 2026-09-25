"""
Motion plans: an ordered list of duration-based actions, one plan per environment.

Actions (build them with the helpers below and put the list in config["plan"]):
    home(duration)                  joint-space move to control.home_qpos
    idle(duration)                  hold the current target
    gripper(open, duration)         open=1.0 fully open, 0.0 closed
    ik(target, duration, ...)       task-space move of the tool center point (TCP) to `target`

Targets
    A target is either an (x, y, z) tuple in table coords, or a reference to an object:
        RedCylinder = Obj("red_cylinder")
        RedCylinder()                      the object's center
        RedCylinder(delta=(0, 0, 0.1))     10 cm above its center
        on_top(BlueCube, RedCylinder)      where the cylinder's center must be to rest on the cube
        offset(target, (dx, dy, dz))       shift any target
    The TCP is the point between the finger pads (config ik.tcp_offset), so ik(RedCylinder())
    puts the gripper centered on the cylinder, ready to close on it.
    resolve_plan(plan, scene) turns object references into table coords using the object
    positions in `scene`; env_config calls it last, after any randomization.

ik options:
    path      "line"    straight segments through start, via points, goal (default)
              "spline"  natural cubic spline through start, via points, goal
              "arc"     spline through start, a via point raised `lift` above the higher end, goal
    via       list of extra targets between start and goal
    profile   "minjerk" (default), "smoothstep", or "trapezoid"
    v_max, a_max   speed and acceleration limits for "trapezoid"; duration then becomes a lower
                   bound and may be omitted, the move takes as long as the limits require
    quat      goal orientation (w, x, y, z), defaults to ik.target_quat; orientation is slerped
    gripper   optional 0..1 gripper opening applied during the move

pick_and_place(pick, place) builds a whole grasp sequence from two targets.

Every duration is in seconds of sim time. A path P(s) is arc-length parametrized on s in [0, 1]
and a profile gives s as a function of normalized time, so geometry and timing are independent.
PlanRunner steps all envs together; each env follows its own plan, so plans may differ in
content and length between envs. An env that has finished simply holds its last target.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from utils.frames import table_to_world, object_center, object_height
from utils.ik import to_numpy, solve_ik, tcp_pos_from_link


# --------------------------------- targets ---------------------------------

class Target(NamedTuple):
    """A point defined relative to an object in config["scene"]["objects"]."""
    object: str
    delta: tuple = (0.0, 0.0, 0.0)
    carried: Optional[str] = None     # set by on_top: the object being placed on `object`
    clearance: float = 0.0


class Obj:
    """Handle for object-relative targets: Obj("red_cylinder")(delta=(0, 0, 0.1))."""

    def __init__(self, name):
        self.name = name

    def __call__(self, delta=(0.0, 0.0, 0.0)):
        return Target(self.name, tuple(float(v) for v in delta))

    def __repr__(self):
        return f"Obj({self.name!r})"


def _name(x):
    if isinstance(x, Obj):
        return x.name
    if isinstance(x, Target):
        return x.object
    return str(x)


def on_top(base, carried, clearance=0.005, delta=(0.0, 0.0, 0.0)):
    """Where `carried`'s center must be for it to rest on top of `base`."""
    return Target(_name(base), tuple(float(v) for v in delta), _name(carried), float(clearance))


def offset(target, d):
    """Shift a target (tuple or Target) by (dx, dy, dz)."""
    if isinstance(target, Target):
        return target._replace(delta=tuple(float(a + b) for a, b in zip(target.delta, d)))
    return tuple(float(a + b) for a, b in zip(target, d))


def resolve_target(target, scene):
    """Target or tuple -> (x, y, z) in table coords, using the object positions in `scene`."""
    if isinstance(target, Target):
        x, y, z = object_center(scene, target.object)
        if target.carried is not None:
            z += object_height(scene["objects"][target.object]) / 2 \
                + object_height(scene["objects"][target.carried]) / 2 + target.clearance
        dx, dy, dz = target.delta
        return (x + dx, y + dy, z + dz)
    if len(target) != 3:
        raise ValueError(f"target must be (x, y, z) or a Target, got {target!r}")
    return tuple(float(v) for v in target)


def resolve_plan(plan, scene):
    """Copy of `plan` with every ik target and via point resolved to table coords."""
    out = []
    for action in plan:
        if action["type"] == "ik":
            action = dict(action)
            action["target"] = resolve_target(action["target"], scene)
            action["via"] = [resolve_target(p, scene) for p in action["via"]]
        out.append(action)
    return out


def describe_target(target):
    if isinstance(target, Target):
        s = f"on_top({target.object}, {target.carried})" if target.carried else target.object
        if any(target.delta):
            s += f"+{tuple(round(v, 3) for v in target.delta)}"
        return s
    return str(tuple(round(v, 3) for v in target))


# ------------------------------ action builders ------------------------------

def home(duration):
    return {"type": "home", "duration": float(duration)}


def idle(duration):
    return {"type": "idle", "duration": float(duration)}


def gripper(open, duration):
    return {"type": "gripper", "open": float(open), "duration": float(duration)}


def ik(target, duration=None, via=None, path="line", profile="minjerk", v_max=None, a_max=None,
       lift=0.10, quat=None, gripper=None):
    if not isinstance(target, Target):
        target = tuple(float(v) for v in target)
        if len(target) != 3:
            raise ValueError(f"target must be (x, y, z) or a Target, got {target!r}")
    if path not in PATHS:
        raise ValueError(f"unknown path {path!r}, expected one of {sorted(PATHS)}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}, expected one of {sorted(PROFILES)}")
    if profile == "trapezoid" and (v_max is None or a_max is None):
        raise ValueError("profile='trapezoid' needs v_max and a_max")
    if profile != "trapezoid" and duration is None:
        raise ValueError(f"profile={profile!r} needs a duration")
    return {
        "type": "ik",
        "target": target,
        "duration": None if duration is None else float(duration),
        "via": [p if isinstance(p, Target) else tuple(float(v) for v in p) for p in (via or [])],
        "path": path,
        "profile": profile,
        "v_max": v_max,
        "a_max": a_max,
        "lift": float(lift),
        "quat": quat,
        "gripper": gripper,
    }


def pick_and_place(pick, place, hover=0.10, lift=0.10, open=1.0, closed=0.0, approach_time=1.5, descend_time=1.0,
                   grip_time=0.5, transfer_time=2.0, retreat_time=1.0, profile="minjerk", quat=None):
    """Grasp at `pick` and release at `place`. Both are targets for the TCP, e.g. pick=RedCylinder().

    Approach and transfer moves use an "arc" path raised `lift` above the hover height so the tool
    lifts over anything in between; descend and retreat are vertical lines.
    """
    up = (0.0, 0.0, hover)
    move = dict(profile=profile, quat=quat)
    return [
        ik(offset(pick, up), approach_time, path="arc", lift=lift, gripper=open, **move),
        ik(pick, descend_time, path="line", **move),
        gripper(closed, grip_time),
        ik(offset(pick, up), retreat_time, path="line", **move),
        ik(offset(place, up), transfer_time, path="arc", lift=lift, **move),
        ik(place, descend_time, path="line", **move),
        gripper(open, grip_time),
        ik(offset(place, up), retreat_time, path="line", **move),
    ]


def describe(action):
    kind = action["type"]
    if kind == "ik":
        parts = [f"ik -> {describe_target(action['target'])}", action["path"], action["profile"]]
        if action["via"]:
            parts.append(f"via {len(action['via'])}")
        if action["quat"] is not None:
            parts.append(f"quat={tuple(round(v, 3) for v in action['quat'])}")
        if action["gripper"] is not None:
            parts.append(f"gripper={action['gripper']:.2f}")
        if action["duration"] is not None:
            parts.append(f"over {action['duration']:.2f}s")
        else:
            parts.append(f"v_max={action['v_max']} a_max={action['a_max']}")
        return " ".join(parts)
    if kind == "gripper":
        return f"gripper open={action['open']:.2f} over {action['duration']:.2f}s"
    return f"{kind} over {action['duration']:.2f}s"


# ---------------------------------- paths ----------------------------------
# A path builder takes world-frame start, goal and via points and returns (P, length) where
# P(s) for s in [0, 1] is arc-length parametrized.

def _dedupe(pts):
    keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-9]
    return pts[keep]


def _constant_path(goal):
    goal = np.asarray(goal, dtype=float)
    return (lambda s: goal.copy()), 0.0


def line_path(start, goal, via=(), **_):
    pts = _dedupe(np.asarray([start, *via, goal], dtype=float))
    if len(pts) < 2:
        return _constant_path(goal)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    length = cum[-1]

    def P(s):
        d = s * length
        return np.array([np.interp(d, cum, pts[:, k]) for k in range(3)])

    return P, length


def spline_path(start, goal, via=(), samples=400, **_):
    pts = _dedupe(np.asarray([start, *via, goal], dtype=float))
    if len(pts) < 3:
        return line_path(start, goal, via)
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    spline = CubicSpline(u, pts, bc_type="natural")
    uu = np.linspace(0.0, u[-1], samples)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(spline(uu), axis=0), axis=1))])
    length = arc[-1]

    def P(s):
        return spline(np.interp(s * length, arc, uu))

    return P, length


def arc_path(start, goal, via=(), lift=0.10, **_):
    """Spline through start, a point raised `lift` above the higher end, and goal. Ignores `via`."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    mid = (start + goal) / 2
    mid[2] = max(start[2], goal[2]) + lift
    return spline_path(start, goal, [mid])


PATHS = {"line": line_path, "spline": spline_path, "arc": arc_path}


# --------------------------------- profiles ---------------------------------
# A profile maps normalized time tau in [0, 1] to path progress s in [0, 1].

def smoothstep(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau * tau * (3.0 - 2.0 * tau)


def minjerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau * tau)


def trapezoid(length, v_max, a_max, duration=None):
    """Trapezoidal (or triangular) velocity profile. Returns (s_of_tau, actual_duration)."""
    if length < 1e-9:
        return smoothstep, (duration or 0.0)
    if length >= v_max * v_max / a_max:
        t_min = length / v_max + v_max / a_max          # reaches cruise speed
    else:
        t_min = 2.0 * np.sqrt(length / a_max)           # triangular
    T = max(t_min, duration or 0.0)
    # Cruise speed v that covers `length` in time T with accel a_max: length = v * (T - v / a_max).
    disc = max(T * T - 4.0 * length / a_max, 0.0)
    v = (T - np.sqrt(disc)) * a_max / 2.0
    f = min((v / a_max) / T, 0.5)                       # fraction of T spent accelerating
    vn = 1.0 / (1.0 - f)                                # normalized cruise speed

    def s_of_tau(tau):
        tau = np.clip(tau, 0.0, 1.0)
        if tau < f:
            return vn * tau * tau / (2.0 * f)
        if tau <= 1.0 - f:
            return vn * (tau - f / 2.0)
        return 1.0 - vn * (1.0 - tau) ** 2 / (2.0 * f)

    return s_of_tau, T


def make_profile(action, length):
    """Return (s_of_tau, duration) for an ik action once its path length is known."""
    if action["profile"] == "trapezoid":
        return trapezoid(length, action["v_max"], action["a_max"], action["duration"])
    return PROFILES[action["profile"]], action["duration"]


PROFILES = {"smoothstep": smoothstep, "minjerk": minjerk, "trapezoid": trapezoid}


def quat_slerp(q0, q1, s):
    """Interpolate (w, x, y, z) quaternions."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if np.allclose(q0, q1) or np.allclose(q0, -q1):
        return q1.copy()
    rots = Rotation.from_quat([q0[[1, 2, 3, 0]], q1[[1, 2, 3, 0]]])   # scipy wants (x, y, z, w)
    q = Slerp([0.0, 1.0], rots)(float(np.clip(s, 0.0, 1.0))).as_quat()
    return q[[3, 0, 1, 2]]


# --------------------------------- runner ---------------------------------

class PlanRunner:
    """Drives every env through its own config["plan"], one sim step at a time."""

    def __init__(self, scene, robot, ee_link, env_configs, dofs_idx, config, verbose=True, hook=None):
        self.scene = scene
        self.robot = robot
        self.ee_link = ee_link
        self.env_configs = env_configs
        self.ik_cfg = config["ik"]
        self.dt = config["sim"]["dt"]
        self.verbose = verbose
        self.hook = hook   # called as hook(runner) each step after the command is issued, before the sim steps

        n_arm = len(config["control"]["arm_joints"])
        self.dofs = list(dofs_idx)              # entity DOF indices, arm joints then gripper joints
        self.arm_dofs = self.dofs[:n_arm]
        self.arm_cols = slice(0, n_arm)         # columns of target_q holding the arm
        self.grip_cols = slice(n_arm, None)     # columns of target_q holding the gripper

        self.n = len(env_configs)
        self.plans = [list(c.get("plan", [])) for c in env_configs]
        self.step_count = 0

        # Commanded state. tcp_cmd / quat_cmd are the last task-space targets sent to IK.
        self.target_q = to_numpy(robot.get_dofs_position(self.dofs)).astype(float).copy()
        self.start_q = self.target_q.copy()
        self.goal_q = self.target_q.copy()
        self.tcp_cmd = tcp_pos_from_link(ee_link, self.ik_cfg["tcp_offset"])
        # Start from the wrist's actual orientation so the first task-space move slerps toward its
        # target instead of demanding it instantly (which whipped the wrist at the start of a plan).
        self.quat_cmd = to_numpy(ee_link.get_quat()).astype(float).copy()
        self.start_quat = self.quat_cmd.copy()
        self.goal_quat = self.quat_cmd.copy()
        self.path_fn = [None] * self.n
        self.profile_fn = [None] * self.n

        self.action_idx = np.zeros(self.n, dtype=int)
        self.t = np.zeros(self.n, dtype=int)          # steps elapsed in the current action
        self.steps = np.ones(self.n, dtype=int)       # steps the current action lasts
        self.in_ik = np.zeros(self.n, dtype=bool)
        self.done = np.zeros(self.n, dtype=bool)

        for i in range(self.n):
            self._begin(i)

    @property
    def time(self):
        return self.step_count * self.dt

    def all_done(self):
        return bool(self.done.all())

    def current_action(self, i):
        plan = self.plans[i]
        return plan[self.action_idx[i]] if self.action_idx[i] < len(plan) else None

    def _gripper_qpos(self, i, open_frac):
        ctrl = self.env_configs[i]["control"]
        opened = np.asarray(ctrl["gripper_open_qpos"], dtype=float)
        closed = np.asarray(ctrl["gripper_closed_qpos"], dtype=float)
        return closed + float(open_frac) * (opened - closed)

    def _begin(self, i):
        """Start env i's next action: freeze the start state and set the goal."""
        plan = self.plans[i]
        if self.action_idx[i] >= len(plan):
            # Plan finished: hold whatever was last commanded (an ik action only writes the arm
            # columns of target_q, so start_q/goal_q must be refreshed here or the arm snaps back).
            self.done[i] = True
            self.in_ik[i] = False
            self.start_q[i] = self.target_q[i]
            self.goal_q[i] = self.target_q[i]
            return

        action = plan[self.action_idx[i]]
        kind = action["type"]
        cfg = self.env_configs[i]
        duration = action["duration"]
        self.t[i] = 0
        self.start_q[i] = self.target_q[i]
        self.goal_q[i] = self.target_q[i]
        self.in_ik[i] = False

        if kind == "home":
            self.goal_q[i] = cfg["control"]["home_qpos"]
        elif kind == "idle":
            pass
        elif kind == "gripper":
            self.goal_q[i, self.grip_cols] = self._gripper_qpos(i, action["open"])
        elif kind == "ik":
            self.in_ik[i] = True
            scene_cfg = cfg["scene"]
            start = tcp_pos_from_link(self.ee_link, self.ik_cfg["tcp_offset"])[i]
            goal = np.asarray(table_to_world(cfg, resolve_target(action["target"], scene_cfg)), dtype=float)
            via = [table_to_world(cfg, resolve_target(p, scene_cfg)) for p in action["via"]]
            self.path_fn[i], length = PATHS[action["path"]](start, goal, via, lift=action["lift"])
            self.profile_fn[i], duration = make_profile(action, length)
            self.start_quat[i] = self.quat_cmd[i]
            quat = action["quat"] if action["quat"] is not None else cfg["ik"]["target_quat"]
            self.goal_quat[i] = np.asarray(quat, dtype=float)
            if action["gripper"] is not None:
                self.goal_q[i, self.grip_cols] = self._gripper_qpos(i, action["gripper"])
        else:
            raise ValueError(f"env {i}: unknown action type {kind!r}")

        self.steps[i] = max(1, int(round(duration / self.dt)))
        if self.verbose:
            print(f"[{self.time:6.2f}s] env {i}: {describe(action)}")

    def step(self):
        """Advance every env by one sim step."""
        tau = np.clip((self.t + 1) / self.steps, 0.0, 1.0)   # normalized time at the end of this step
        s_joint = smoothstep(tau)
        s_joint[self.done] = 1.0

        # Joint-space interpolation covers home, gripper, and the gripper part of ik actions.
        self.target_q = self.start_q + (self.goal_q - self.start_q) * s_joint[:, None]

        # Task-space moves: evaluate each env's path and profile, then one batched IK call.
        if self.in_ik.any():
            for i in np.flatnonzero(self.in_ik):
                s = self.profile_fn[i](tau[i])
                self.tcp_cmd[i] = self.path_fn[i](s)
                self.quat_cmd[i] = quat_slerp(self.start_quat[i], self.goal_quat[i], s)
            q = to_numpy(solve_ik(self.robot, self.ee_link, self.tcp_cmd, self.quat_cmd, self.ik_cfg))
            self.target_q[self.in_ik, self.arm_cols] = q[self.in_ik][:, self.arm_dofs]

        self.robot.control_dofs_position(self.target_q, self.dofs)
        if self.hook is not None:
            self.hook(self)          # observation of this step and the command just issued, paired
        self.scene.step()
        self.step_count += 1
        self.t += 1

        for i in np.flatnonzero((self.t >= self.steps) & ~self.done):
            self.action_idx[i] += 1
            self._begin(i)

    def run(self):
        while not self.all_done():
            self.step()
