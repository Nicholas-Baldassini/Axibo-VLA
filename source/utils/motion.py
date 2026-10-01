"""
Motion plans: an ordered list of duration-based actions, one plan per environment.

Actions (build them with the helpers below and put the list in config["plan"]):
    home(duration)                  joint-space move to control.home_qpos
    ik(target, speed, ...)          task-space move of the tool center point (TCP) to `target`,
                                    optionally setting the gripper on its first step

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
    speed     m/s toward the next waypoint, constant from the first step to the last: the tool
              moves at this speed the whole way and the next action starts on the following step
    via       list of extra targets between start and goal; the path is straight segments
    quat      goal orientation (w, x, y, z), defaults to ik.target_quat; orientation is slerped
    gripper   optional 0..1 gripper opening, set in full on the move's first step
    noise     m, standard deviation of the disturbance injected into the executed command during
              this move (0 = none); see PlanRunner
    record    False marks a move the recorder must not store (the randomized start move)

pick_and_place(pick, place) builds a whole grasp sequence from two targets.

Moves are executed by pursuit, not by a precomputed schedule: at every sim step the command
moves from wherever it is toward the current waypoint at the move's speed, so a command that has
been pushed off the path heads back to the path's next waypoint. That is what makes the
disturbance useful. With `noise`, a smooth random push (an offset that glides at a bounded rate
toward a new random target every half second) is added to the executed command, the arm follows
the disturbed command, and the recorded label is the clean pursuit rate from the disturbed
position toward the waypoint (the idea of DART, Laskey et al. 2017): the demonstrations then
contain states off the nominal path, at every height and offset, each labelled with the move
that returns to the target. Measured on run 9, a policy trained on undisturbed straight paths
reproduced them to 0.3 mm open loop but could not recover from its own 1-3 cm deviations,
because no demonstration frame had ever shown one.

Engine 0.6.0 (2026-09-29) removed the arc and spline paths and the min-jerk, smoothstep and
trapezoid timing profiles. Every move is now straight segments at one constant speed, so every
recorded step of a move has the same size and no move eases into or out of a target. The
trapezoid's deceleration into the place point (carry acceleration 0.3 m/s^2) produced 12 frames of
shrinking actions, 12 -> 2 mm per step, before the release label; the policy trained on them
stopped 0.5-0.9 cm above the release height and never opened (task2.md, run 8). Slow stretches
that physically need care (the last centimetres of a descent, the first of a lift) are separate
moves at a lower constant speed, still 5 mm per control step, not a ramp toward zero.

Every duration is in seconds of sim time. PlanRunner steps all envs together; each env follows
its own plan, so plans may differ in content and length between envs. An env that has finished
simply holds its last target.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from utils.frames import table_to_world, object_center, object_height, grasp_yaw
from utils.ik import to_numpy, solve_ik, tcp_pos_from_link, quat_mul, yaw_quat


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


def grasp_quat(spec, base_quat):
    """Gripper orientation for grasping `spec`: `base_quat` (gripper pointing down, fingers closing
    along x) turned about the world z axis so the fingers close across the object's faces."""
    return tuple(float(v) for v in quat_mul(yaw_quat(grasp_yaw(spec)), np.asarray(base_quat, dtype=float)))


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


def ik(target, speed, via=None, quat=None, gripper=None, noise=0.0, yaw_noise_deg=0.0, record=True):
    """Move the TCP to `target` through `via` in straight segments at a constant `speed` (m/s),
    turning toward `quat` on the way, with pushes of half-width `noise` (m) on the executed
    position and `yaw_noise_deg` on the executed wrist yaw."""
    if not isinstance(target, Target):
        target = tuple(float(v) for v in target)
        if len(target) != 3:
            raise ValueError(f"target must be (x, y, z) or a Target, got {target!r}")
    if not speed or speed <= 0:
        raise ValueError(f"ik needs a positive speed in m/s, got {speed!r}")
    return {
        "type": "ik",
        "target": target,
        "speed": float(speed),
        "via": [p if isinstance(p, Target) else tuple(float(v) for v in p) for p in (via or [])],
        "quat": quat,
        "gripper": gripper,
        "noise": float(noise or 0.0),
        "yaw_noise_deg": float(yaw_noise_deg or 0.0),
        "record": bool(record),
    }


NOISE_PHASES = ("transit", "descent", "final", "lift", "retreat")


def pick_and_place(pick, place, hover=0.10, open=1.0, closed=0.0, quat=None,
                   pick_offset=(0.0, 0.0), place_offset=(0.0, 0.0),
                   speed=0.15, slow=0.05, final=0.03, lift=0.02, noise=None, yaw_noise=None):
    """Grasp at `pick` and release at `place`. Both are targets for the TCP, e.g. pick=RedCylinder().

    Eight moves, every one at a constant speed, each starting on the step after the previous,
    with the disturbance level of `noise` (m, by phase name) applied to the executed command:
        1. to a hover point above `pick`                          speed   transit
        2. diagonally down to `final` above `pick`                speed   descent
        3. straight down onto `pick`                              slow    final
        4. close, and straight up by `lift` while the fingers close  slow  lift
        5. up to hover height, across to a hover point above `place`  speed  transit
        6. diagonally down to `final` above `place`               speed   descent
        7. straight down onto `place`                             slow    final
        8. open, and straight up to hover height                  speed   retreat

    The gripper commands are part of the move that follows them: the runner sets the finger
    target on the move's first step, so the tool never spends a frame standing still. The slow
    moves exist for the physics, not for the labels: the fingers stop 0.5 cm above the table at
    the grasp, and the object must not be yanked while the fingers are still closing. At `slow`
    0.05 m/s they still command 5 mm per 10 Hz control step, and nothing anywhere in the plan
    commands less. Every move ends at full speed, so the frame whose gripper label flips is
    preceded by moving frames, never by a standstill (the trapezoid's deceleration into the place
    is what run 8's policy stalled on).

    pick_offset and place_offset shift the two hover points sideways, in metres, so every descent
    starts with a visible gap to close. The disturbance (see the module docstring) puts the tool
    off the path at every height as well, with the label pointing back; the phases where the
    fingers are around an object ("final", "lift") should carry little or none, since the pursuit
    of a disturbance-free move removes whatever offset was inherited within a few frames.
    """
    noise = dict.fromkeys(NOISE_PHASES, 0.0) | dict(noise or {})
    yaw = dict.fromkeys(NOISE_PHASES, 0.0) | dict(yaw_noise or {})
    up = (0.0, 0.0, hover)
    above = (0.0, 0.0, final)
    pick_hover = offset(pick, (pick_offset[0], pick_offset[1], hover))
    place_hover = offset(place, (place_offset[0], place_offset[1], hover))
    move = lambda target, spd, phase, **kw: ik(target, spd, quat=quat, noise=noise[phase], yaw_noise_deg=yaw[phase], **kw)
    return [
        move(pick_hover, speed, "transit", gripper=open),
        move(offset(pick, above), speed, "descent"),
        move(pick, slow, "final"),
        move(offset(pick, (0.0, 0.0, lift)), slow, "lift", gripper=closed),
        move(place_hover, speed, "transit", via=[offset(pick, up)]),
        move(offset(place, above), speed, "descent"),
        move(place, slow, "final"),
        move(offset(place, up), speed, "retreat", gripper=open),
    ]


def describe(action):
    kind = action["type"]
    if kind == "ik":
        parts = [f"ik -> {describe_target(action['target'])}", f"{action['speed']:.2f} m/s"]
        if action["via"]:
            parts.append(f"via {len(action['via'])}")
        if action["quat"] is not None:
            parts.append(f"quat={tuple(round(v, 3) for v in action['quat'])}")
        if action["gripper"] is not None:
            parts.append(f"gripper={action['gripper']:.2f}")
        if action.get("noise"):
            parts.append(f"noise={100 * action['noise']:.1f}cm")
        if action.get("yaw_noise_deg"):
            parts.append(f"yaw_noise={action['yaw_noise_deg']:.0f}deg")
        if not action.get("record", True):
            parts.append("unrecorded")
        return " ".join(parts)
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




def rotvec_between(q_from, q_to):
    """Rotation vector, in q_from's own frame, that turns (w, x, y, z) quaternion q_from into q_to:
    the quantity train/actions.py records as a rotation label and train/infer.py applies by
    right-multiplying the current command."""
    r_from = Rotation.from_quat(np.asarray(q_from, dtype=float)[[1, 2, 3, 0]])   # scipy wants (x, y, z, w)
    r_to = Rotation.from_quat(np.asarray(q_to, dtype=float)[[1, 2, 3, 0]])
    return (r_from.inv() * r_to).as_rotvec()


def rotvec_quat(rotvec):
    """(w, x, y, z) quaternion of a rotation vector."""
    q = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_quat()
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
        # Disturbance settings (config randomization.path_noise): how often a new push target is
        # drawn, how fast the offset moves toward it, and how much of it goes into z relative to
        # x and y. Each env draws from its own generator seeded by env_config, so a recording is
        # reproducible.
        noise_cfg = config.get("randomization", {}).get("path_noise", {})
        self.noise_interval = float(noise_cfg.get("interval", 0.5))
        self.noise_rate = float(noise_cfg.get("rate", 0.05))
        self.noise_axes = np.array([1.0, 1.0, float(noise_cfg.get("z_factor", 0.5))])
        # The wrist orientation is pursued the same way: turned toward the move's goal orientation
        # at yaw_speed, and pushed at up to yaw_push_rate toward a random yaw offset.
        self.yaw_speed = np.deg2rad(float(noise_cfg.get("yaw_speed_deg", 120.0)))
        self.yaw_push_rate = np.deg2rad(float(noise_cfg.get("yaw_push_rate_deg", 45.0)))
        self.yaw_settle = float(noise_cfg.get("yaw_settle_s", 0.1))     # one control step at 10 Hz
        self.rngs = [np.random.default_rng(c.get("noise_seed", 0)) for c in env_configs]
        self.noise_target = [None] * len(env_configs)
        self.noise_clock = np.zeros(len(env_configs))
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
        # Pursuit state of the current task-space move, per env.
        self.waypoints = [[] for _ in range(self.n)]  # world points still to visit, in order
        self.speed = np.zeros(self.n)
        self.sigma = np.zeros(self.n)                  # push half-width of the current move, m
        self.yaw_sigma = np.zeros(self.n)              # push half-width on the wrist yaw, rad
        self.noise = np.zeros((self.n, 3))             # the current push offset in the executed command
        self.yaw_noise = np.zeros(self.n)              # the current push offset in the executed yaw, rad
        self.yaw_target = np.zeros(self.n)
        # The undisturbed pursuit rates this step, the label: m/s in table coords (x, y, z) and
        # rad/s as a rotation vector in the tool frame (the frame train/infer.py applies it in).
        self.clean_rate = np.zeros((self.n, 6))
        self.finish = np.zeros(self.n, dtype=bool)     # the move reached its last waypoint this step

        self.action_idx = np.zeros(self.n, dtype=int)
        self.t = np.zeros(self.n, dtype=int)          # steps elapsed in the current action
        self.steps = np.ones(self.n, dtype=int)       # steps the current action lasts (joint-space moves)
        self.in_ik = np.zeros(self.n, dtype=bool)
        self.cmd_current = np.zeros(self.n, dtype=bool)   # tcp_cmd reflects the arm's latest task-space target
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
        duration = action.get("duration")
        self.t[i] = 0
        self.start_q[i] = self.target_q[i]
        self.goal_q[i] = self.target_q[i]
        self.in_ik[i] = False

        if kind == "home":
            self.goal_q[i] = cfg["control"]["home_qpos"]
            self.cmd_current[i] = False       # a joint-space move leaves tcp_cmd behind
        elif kind == "ik":
            self.in_ik[i] = True
            scene_cfg = cfg["scene"]
            # A move continues from the last commanded tool pose, not from where the arm is: under
            # PD control the arm trails the command by up to a centimetre at speed, and starting
            # the next segment from the arm would step the command backwards by that lag at every
            # move boundary. Only after a joint-space move (home) is the command stale, and then
            # the arm's actual pose is the start.
            if self.cmd_current[i]:
                start = self.tcp_cmd[i].copy()
            else:
                start = tcp_pos_from_link(self.ee_link, self.ik_cfg["tcp_offset"])[i]
                self.tcp_cmd[i] = start
                self.noise[i] = 0.0
                self.cmd_current[i] = True
            goal = np.asarray(table_to_world(cfg, resolve_target(action["target"], scene_cfg)), dtype=float)
            via = [table_to_world(cfg, resolve_target(p, scene_cfg)) for p in action["via"]]
            pts = _dedupe(np.asarray([start, *via, goal], dtype=float))
            self.waypoints[i] = [p.copy() for p in pts[1:]] or [goal.copy()]
            self.speed[i] = action["speed"]
            self.sigma[i] = action.get("noise", 0.0)
            self.yaw_sigma[i] = np.deg2rad(action.get("yaw_noise_deg", 0.0))
            self.finish[i] = False
            duration = self.dt            # pursuit ends when the last waypoint is reached, not by time
            self.start_quat[i] = self.quat_cmd[i]
            quat = action["quat"] if action["quat"] is not None else cfg["ik"]["target_quat"]
            self.goal_quat[i] = np.asarray(quat, dtype=float)
            if action["gripper"] is not None:
                # The finger target is set in full on the move's first step, not interpolated over
                # the move: the recorded gripper label is this target, and it must flip on one
                # frame while the tool keeps moving.
                self.goal_q[i, self.grip_cols] = self._gripper_qpos(i, action["gripper"])
                self.start_q[i, self.grip_cols] = self.goal_q[i, self.grip_cols]
        else:
            raise ValueError(f"env {i}: unknown action type {kind!r}")

        self.steps[i] = max(1, int(round(duration / self.dt)))
        if self.verbose:
            print(f"[{self.time:6.2f}s] env {i}: {describe(action)}")

    def _pursue(self, i):
        """One sim step of env i's task-space move: the clean pursuit rate from the executed command
        toward the current waypoint (the label), the clean step along the polyline, and the
        disturbance increment on top of it. Sets finish when the last waypoint is reached."""
        x = self.tcp_cmd[i]
        wps = self.waypoints[i]
        d = wps[0] - x
        dist = float(np.linalg.norm(d))
        self.clean_rate[i, :3] = (d / dist * self.speed[i]) if dist > 1e-9 else 0.0
        pos = x.copy()
        budget = self.speed[i] * self.dt
        while budget > 0.0:
            d = wps[0] - pos
            dist = float(np.linalg.norm(d))
            # A waypoint within the budget (plus rounding) counts as reached on this step; without
            # the margin a move whose length divides exactly into steps landed a rounding error
            # short of its goal and spent one extra sub-step with a zero rate (measured).
            if dist <= budget + 1e-6:
                pos = wps[0].copy()
                budget -= dist
                if len(wps) > 1:
                    wps.pop(0)
                else:
                    self.finish[i] = True
                    break
            else:
                pos = pos + d / dist * budget
                budget = 0.0
        # Smooth random pushes: the offset moves at a bounded rate toward a target drawn every
        # `interval` seconds from a box of half-width sigma (z scaled by z_factor). The arm can
        # follow a push like that, and a held object stays held, where white or OU noise at the
        # physics rate shook the command by millimetres every step (measured: 5 of 6 stacks lost).
        # With sigma 0 the target is the origin, so a move without disturbance sheds the offset it
        # inherited at the same bounded rate.
        n = self.noise[i]
        self.noise_clock[i] += self.dt
        if self.noise_clock[i] >= self.noise_interval or self.noise_target[i] is None:
            self.noise_clock[i] = 0.0
            self.noise_target[i] = self.rngs[i].uniform(-1.0, 1.0, size=3) * self.sigma[i] * self.noise_axes
        if self.sigma[i] == 0.0:
            self.noise_target[i] = np.zeros(3)
        d = self.noise_target[i] - n
        dist = float(np.linalg.norm(d))
        step = min(dist, self.noise_rate * self.dt)
        n_new = n + (d / dist * step if dist > 1e-9 else 0.0)
        self.tcp_cmd[i] = pos + (n_new - n)
        self.noise[i] = n_new

        # Orientation, pursued the same way: the clean rate turns the executed orientation toward
        # the move's goal at yaw_speed (a rotation vector in the tool frame, the label), the
        # executed orientation takes that turn, and a push offset about the tool's own z axis (the
        # wrist yaw when the gripper points down) glides toward a random yaw target drawn together
        # with the position push. Since engine 0.6.0's random start yaw, every recorded approach
        # contains the turn to the grasp yaw; run 10, whose start move had already aligned the
        # wrist so the rotation labels were all zero, gripped every cube diagonally and lost 17
        # of 20 cube placements against 1 of 10 cylinders, and scored 30/30 once the wrist was
        # aligned for it.
        q_exec = self.quat_cmd[i]
        rel = rotvec_between(q_exec, self.goal_quat[i])
        ang = float(np.linalg.norm(rel))
        axis = rel / ang if ang > 1e-6 else np.zeros(3)
        # The rate that reaches the goal within yaw_settle seconds, capped at yaw_speed: far from
        # the goal a constant turn, within the last degrees a turn that shrinks with the error.
        # A constant-rate label would have the policy step 12 degrees past a 3 degree error and
        # oscillate around the grasp yaw; a label proportional to the error converges.
        rate = min(ang / self.yaw_settle, self.yaw_speed) if ang > 1e-6 else 0.0
        self.clean_rate[i, 3:] = axis * rate
        turn = min(ang, rate * self.dt)
        q_clean = quat_mul(q_exec, rotvec_quat(axis * turn)) if turn > 0.0 else q_exec
        yn = self.yaw_noise[i]
        if self.yaw_sigma[i] == 0.0:
            self.yaw_target[i] = 0.0
        elif self.noise_clock[i] == 0.0:            # a new push target was drawn this step
            self.yaw_target[i] = self.rngs[i].uniform(-1.0, 1.0) * self.yaw_sigma[i]
        yd = self.yaw_target[i] - yn
        yn_new = yn + float(np.clip(yd, -self.yaw_push_rate * self.dt, self.yaw_push_rate * self.dt))
        self.quat_cmd[i] = quat_mul(q_clean, rotvec_quat(np.array([0.0, 0.0, yn_new - yn])))
        self.yaw_noise[i] = yn_new

    def step(self):
        """Advance every env by one sim step."""
        tau = np.clip((self.t + 1) / self.steps, 0.0, 1.0)   # normalized time at the end of this step
        s_joint = tau            # joint-space moves (home, gripper) interpolate linearly; none is recorded
        s_joint[self.done] = 1.0

        # Joint-space interpolation covers home, gripper, and the gripper part of ik actions.
        self.target_q = self.start_q + (self.goal_q - self.start_q) * s_joint[:, None]

        # Task-space moves: one pursuit step per env, then one batched IK call.
        if self.in_ik.any():
            for i in np.flatnonzero(self.in_ik):
                self._pursue(i)
            q = to_numpy(solve_ik(self.robot, self.ee_link, self.tcp_cmd, self.quat_cmd, self.ik_cfg))
            self.target_q[self.in_ik, self.arm_cols] = q[self.in_ik][:, self.arm_dofs]
        self.clean_rate[~self.in_ik] = 0.0

        self.robot.control_dofs_position(self.target_q, self.dofs)
        if self.hook is not None:
            self.hook(self)          # observation of this step and the command just issued, paired
        self.scene.step()
        self.step_count += 1
        self.t += 1

        # A joint-space action ends by time; a task-space move ends when its pursuit reached the
        # last waypoint on this step.
        ended = np.where(self.in_ik, self.finish, self.t >= self.steps) & ~self.done
        for i in np.flatnonzero(ended):
            self.action_idx[i] += 1
            self._begin(i)

    def run(self):
        while not self.all_done():
            self.step()
