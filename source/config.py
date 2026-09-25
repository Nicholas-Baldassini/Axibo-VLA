"""
Configuration for the language-conditioned stacking scene.

Coordinate conventions
----------------------
Each Genesis environment is its own simulation with its own frame; env_spacing only
spreads them out in the viewer. The table center sits at every env's origin, so table
(x, y) == env (x, y).

Every position below is in TABLE COORDINATES unless stated otherwise:
    x, y  measured from the table center, meters
    z     measured up from the table surface, meters
Wrist-camera pos/lookat are in the wrist link's own frame.

Everything is collected into the single CONFIG dict at the bottom, and
env_config(CONFIG, i) produces the config for environment i.
"""

import copy
from pathlib import Path

import numpy as np

from utils.motion import home, idle, ik, gripper, pick_and_place, Obj, on_top, resolve_plan

SOURCE_DIR = Path(__file__).resolve().parent
PREVIEW_DIR = SOURCE_DIR.parent / "previews"

# The one random seed. Seeds Genesis, numpy, torch and python's random at init, and every
# per-episode draw in env_config derives from it together with the env and episode numbers.
SEED = 0

TABLE_HEIGHT = 0.75  # z of the table surface above the ground plane
TABLE_SIZE = (0.8, 1.0, TABLE_HEIGHT)  # (x depth, y width, z height)

# Colors, RGBA in 0..1.
RED = (0.85, 0.10, 0.10, 1.0)
BLUE = (0.10, 0.25, 0.90, 1.0)

SCENE = {
    # One fixed box standing on the ground plane, centered on the env origin.
    "table": {
        "size": TABLE_SIZE,
        "pos": (0.0, 0.0, TABLE_HEIGHT / 2),   # world frame: box center, bottom face on the ground
        "color": (0.60, 0.45, 0.30, 1.0),
        "friction": 1.0,
    },
    # Base link bolted to the table surface, 10 cm in from the -x edge, facing +x.
    "robot": {
        "urdf": str(SOURCE_DIR / "assets" / "piper_x" / "piper_x_d435.urdf"),
        "pos": (-0.3, 0.0, 0.0),
        "euler": (0.0, 0.0, 0.0),              # degrees, scipy extrinsic x-y-z
    },
    # The three task objects. "pos" is (x, y) on the table surface; each rests on it.
    "objects": {
        "red_cube":     {"shape": "box",      "size": 0.04,                   "color": RED,  "pos": (0.0, -0.12)},
        "red_cylinder": {"shape": "cylinder", "radius": 0.02, "height": 0.04, "color": RED,  "pos": (0.0,  0.00)},
        "blue_cube":    {"shape": "box",      "size": 0.04,                   "color": BLUE, "pos": (0.0,  0.12)},
    },
    "object_density": 1000.0,  # kg/m^3, a 4 cm cube weighs 64 g
    "object_friction": 1.0,
    "spawn_gap": 0.002,        # clearance above the table so nothing spawns interpenetrating
}

# Handles for object-relative targets: RedCylinder() is its center, RedCylinder(delta=(0, 0, 0.1))
# is 10 cm above it. Resolved to table coords per env by env_config, so they follow randomization.
RedCube = Obj("red_cube")
RedCylinder = Obj("red_cylinder")
BlueCube = Obj("blue_cube")

# Each camera: "res" (width, height) px; either "fov" (vertical, degrees) or "intrinsics"
# {"fx", "fy", "cx", "cy"} in pixels for that res, from which the render FOV is derived; "near"/"far"
# clip planes in m; extrinsics as pos/lookat/up in table coords, or in the link frame when
# "attach_link" is set. The full camera model (K matrix, FOVs, extrinsics) is stored with every
# recorded session. See utils/cameras.py for what a real camera can and cannot be matched on.
CAMERAS = {
    # Bird's eye view, straight down on the table center. "up" = +x puts the far edge at the top of the image.
    "overhead": {
        "res": (640, 480),
        "fov": 60,
        "near": 0.01,
        "far": 10.0,
        "pos": (0.0, 0.0, 0.85),
        "lookat": (0.0, 0.0, 0.0),
        "up": (1.0, 0.0, 0.0),
    },
    # Opposite the robot, past the +x edge, looking back at the table center and pitched down.
    "front": {
        "res": (640, 480),
        "fov": 60,
        "near": 0.01,
        "far": 10.0,
        "pos": (0.7, 0.0, 0.45),
        "lookat": (0.0, 0.0, 0.12),
        "up": (0.0, 0.0, 1.0),
    },
    # Wrist camera: a RealSense D435 color stream at 640x480. The intrinsics below are the nominal
    # values for that mode (42.5 deg vertical FOV); replace them with your unit's calibration from
    # `rs-enumerate-devices -c`. pos/lookat/up are in the Link6 frame and reproduce the D435 mount
    # in the URDF: 6.9 cm to the side of the wrist, looking along the gripper toward the fingertips.
    "wrist": {
        "res": (640, 480),
        "intrinsics": {"fx": 617.0, "fy": 617.0, "cx": 320.0, "cy": 240.0},
        "near": 0.01,
        "far": 10.0,
        "attach_link": "Link6",
        "pos": (0.029, -0.069, 0.022),
        "lookat": (0.03, 0.03, 0.30),
        "up": (0.0, -0.94, 0.34),
    },
}

VIEWER = {
    "res": (1280, 960),
    "camera_pos": (1.3, -1.1, TABLE_HEIGHT + 0.7),     # world frame
    "camera_lookat": (0.0, 0.0, TABLE_HEIGHT + 0.1),   # world frame
    "camera_fov": 40,
    "world_frame_size": 0.3,       # length of the origin axes drawn in the viewer
    "draw_camera_frustums": True,  # draw each camera's frustum in the viewer
}

PREVIEW = {
    "dir": str(PREVIEW_DIR),                         # one PNG per camera, envs side by side
    "record_video": True,                            # one MP4 per camera per env, written while the sim steps
    "video_dir": str(SOURCE_DIR.parent / "videos"),
    "video_hz": 30,                                  # frames captured per second of sim time; also the video fps
}

CONTROL = {
    # Joints in the order used for every control call.
    "arm_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    "gripper_joints": ("joint7", "joint8"),
    "kp": np.array([300, 300, 300, 100, 100, 100, 50, 50], dtype=float),          # position gains
    "kv": np.array([30, 30, 30, 10, 10, 10, 5, 5], dtype=float),                   # velocity gains
    "force_limit": np.array([100, 100, 100, 100, 100, 100, 20, 20], dtype=float),  # from the URDF effort limits
    # Pose held after build: arm raised in front of the robot, gripper open. joint2/joint3 unfold the arm.
    "home_qpos": np.array([0.0, 1.0, -1.2, 0.0, 0.6, 0.0, 0.04, -0.04]),
    # Gripper joint positions (joint7, joint8) for gripper(open=1.0) and gripper(open=0.0).
    "gripper_open_qpos": (0.05, -0.05),
    "gripper_closed_qpos": (0.0, 0.0),
}

IK = {
    "n_envs": 4,
    "env_spacing": (1.5, 1.5),           # viewer-only grid spacing between environments
    "ee_link": "Link6",                  # wrist link the gripper hangs off; the URDF has no tool-center link
    "tcp_offset": (0.0, 0.0, 0.1308),    # center of the finger pads in the ee_link frame, measured
                                         # from the URDF meshes: python -m utils.gripper
    "target_pos": (0.0, 0.0, 0.12),      # tool-center target, table coords: 12 cm above the red cylinder
    "target_quat": (0.0, 1.0, 0.0, 0.0),  # (w, x, y, z). Gripper pointing straight down, fingers closing along x.
    "rot_mask": (True, True, True),      # constrain full orientation. (False, False, True) frees the yaw.
    "apply": "control",                  # "control": PD-drive the arm to the solution. "set": teleport the joints.
    "settle_steps": 300,                 # sim steps to give the arm to reach the target
}

SIM = {
    "dt": 1 / 90,             # 90 Hz physics: an integer multiple of the 30 Hz recording rate
    "n_steps": 2700,          # stacking_scene.py: 30 s of sim time
    "hold_after_plan": 1.0,   # run_plan.py / record.py: seconds to keep simulating after the last plan finishes
}

# The language-conditioned task: pick `source` and place it on `destination` (two different objects).
# source/destination here are the defaults; with randomization.task on, env_config draws a pair per episode.
TASK = {
    "source": "red_cylinder",
    "destination": "blue_cube",
    "instruction": "put the {source} on the {destination}",   # object names are humanized: "red cylinder"
    # Ordered (source, destination) pairs never used for data collection. Three objects give six
    # ordered pairs; hold one out entirely and use it as the generalization test.
    "holdout_pairs": [("red_cube", "blue_cube")],
}

# Per-episode randomization, drawn in env_config from (SEED, env, episode), so a session is
# reproducible and every env and epoch differs.
RANDOMIZATION = {
    "task": {
        "randomize": True,            # pick a random allowed (source, destination) pair each episode
    },
    "placement": {
        "randomize": True,            # random object positions on the table each episode
        # Table coords where object centers may land. Kept within 0.40 m of the robot base at
        # (-0.3, 0): further out the vertical-gripper IK is no longer accurate at hover height.
        "area": {"x": (-0.15, 0.04), "y": (-0.20, 0.20)},
        # Minimum gap between object footprints, m (edge to edge). 0.05 keeps neighbors outside the
        # open gripper, whose fingers reach 7.4 cm from the tool center along the closing axis.
        "margin": 0.05,
        "max_tries": 500,             # rejection-sampling attempts per object before giving up
    },
}

# Episode termination. After the arm's retreat move, the source must sit on the destination and the
# whole scene must hold still for `hold_time`; then the episode is a success. If the source is ever
# not on the destination after the retreat, or the scene has not settled `timeout` seconds after the
# plan ends, the episode is a failure and is left out of the dataset.
SUCCESS = {
    "hold_time": 2.0,            # s the stack must hold still after the arm retreats
    "timeout": 5.0,              # s after the plan ends to reach a stable stack, else failure
    "stack_xy_tol": 0.015,       # m: source center over the destination center, per axis
    "stack_z_tol": 0.010,        # m: source center at its resting height on the destination
    "upright_tol_deg": 10.0,     # max tilt of the destination and of untouched objects
    "rest_z_tol": 0.005,         # m: destination stays at its resting height on the table
    "disturb_tol": 0.010,        # m: untouched objects stay within this of where they spawned
    "still_pos_tol": 0.002,      # m: max drift of any object over the hold window
    "still_rot_tol_deg": 2.0,    # max rotation drift of any object over the hold window
}


def instruction(task):
    """The episode's language instruction, e.g. 'put the red cylinder on the blue cube'."""
    return task["instruction"].format(
        source=task["source"].replace("_", " "),
        destination=task["destination"].replace("_", " "),
    )


def make_plan(task, hover=0.10):
    """Motion plan for `task`, executed in order by run_plan.py and record.py.

    Targets are object references, resolved to table coords per env by env_config as its last
    step, after any randomization. ik(Obj(name)()) centers the gripper on that object. See
    utils/motion.py for the ik options (path shape, timing profile, via points, orientation).
    """
    src, dst = Obj(task["source"]), Obj(task["destination"])
    return [
        home(0.5),
        idle(0.5),
        # lift keeps the arc apex at hover + 0.02: with the gripper vertical the arm cannot reach much
        # higher than 0.12 above the table, so a taller arc would sag through an infeasible waypoint.
        *pick_and_place(pick=src(), place=on_top(dst, src), hover=hover, lift=0.02),
        home(1.0),   # return home after the retreat; the success check starts when this action begins
    ]


# What record.py stores per frame, on top of the cameras. Every quantity is in table coords.
# "delta" observations are the change since the previous frame; "delta" actions are the command
# relative to the current observation. Turn any of them off to skip it.
RECORD = {
    "epochs": 4,                                      # episodes per env; a session records n_envs * epochs attempts
    "fps": 30,                                        # frames per second of sim time; the sim runs at 1/sim.dt
    "root": str(SOURCE_DIR.parent / "datasets"),      # one session directory per recording run goes here
    "repo_id": "axibo/piper_stack",                   # LeRobot dataset name; also the session directory prefix
    "robot_type": "piper_x",
    "vcodec": "libsvtav1",                            # dataset video codec; "h264" or "hevc_videotoolbox" also work
    "cameras": ["overhead", "front", "wrist"],        # observation.images.<name>
    "observation": {
        "joint_pos": True, "joint_pos_delta": True,   # observation.state is joint_pos, the LeRobot convention
        "joint_vel": True, "joint_vel_delta": True,
        "tcp_pose": True, "tcp_pose_delta": True,     # xyz + quat(w,x,y,z) of the tool center, and gripper opening
        "link_poses": True, "link_poses_delta": True, # FK of every robot link: xyz + quat each
        "objects": True,                              # each object's center xyz + quat
    },
    "action": {
        "tcp_pose": True, "tcp_pose_delta": True,     # commanded tool pose + gripper; `action` is the absolute one
        "joint_pos": True, "joint_pos_delta": True,   # commanded joint targets
        "joint_vel": True, "joint_vel_delta": True,   # commanded joint velocity (rate of the joint targets)
        "link_poses": True, "link_poses_delta": True, # FK of the commanded joint targets
    },
}

CONFIG = {
    "seed": SEED,
    "scene": SCENE,
    "cameras": CAMERAS,
    "viewer": VIEWER,
    "preview": PREVIEW,
    "control": CONTROL,
    "ik": IK,
    "sim": SIM,
    "task": TASK,
    "randomization": RANDOMIZATION,
    "success": SUCCESS,
    "plan": make_plan(TASK),
    "record": RECORD,
}


# ------------------------------ per-episode randomization ------------------------------

def allowed_pairs(task, object_names):
    """All ordered (source, destination) pairs of distinct objects minus the held-out ones."""
    holdout = {tuple(p) for p in task.get("holdout_pairs", [])}
    return [(s, d) for s in object_names for d in object_names if s != d and (s, d) not in holdout]


def footprint_radius(spec):
    """Radius of the circle enclosing an object's footprint on the table."""
    return spec["size"] / np.sqrt(2) if spec["shape"] == "box" else spec["radius"]


def sample_placement(scene, rng, placement):
    """Random (x, y) per object inside placement.area with at least placement.margin between footprints."""
    (x0, x1), (y0, y1) = placement["area"]["x"], placement["area"]["y"]
    placed = {}
    for name, spec in scene["objects"].items():
        r = footprint_radius(spec)
        for _ in range(placement["max_tries"]):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if all(np.hypot(x - px, y - py) >= r + pr + placement["margin"] for px, py, pr in placed.values()):
                placed[name] = (float(x), float(y), r)
                break
        else:
            raise ValueError(f"could not place {name!r} without overlap; enlarge placement.area or reduce margin")
    return {name: (x, y) for name, (x, y, _) in placed.items()}


def env_config(config, env_idx, episode_idx=0):
    """Return the config for environment env_idx in episode (epoch) episode_idx.

    Draws the task pair and the object placement from randomization, then resolves the plan's
    object references into table coords for this layout. Deterministic in (seed, env, episode).

    Genesis batched environments share geometry, so only state-like fields can differ between
    envs: object positions, control.home_qpos, the task and the whole plan. Sizes, shapes,
    colors, the table, cameras and n_envs come from the base config for every env.
    """
    cfg = copy.deepcopy(config)
    rand = cfg["randomization"]
    rng = np.random.default_rng([cfg["seed"], env_idx, episode_idx])

    if rand["task"]["randomize"]:
        pairs = allowed_pairs(cfg["task"], list(cfg["scene"]["objects"]))
        cfg["task"]["source"], cfg["task"]["destination"] = pairs[rng.integers(len(pairs))]

    if rand["placement"]["randomize"]:
        for name, xy in sample_placement(cfg["scene"], rng, rand["placement"]).items():
            cfg["scene"]["objects"][name]["pos"] = xy

    # Last step: build the plan for this task and turn object references into table coords.
    cfg["plan"] = resolve_plan(make_plan(cfg["task"]), cfg["scene"])
    return cfg
