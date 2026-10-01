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

from utils.motion import home, ik, pick_and_place, Obj, on_top, offset, resolve_plan, grasp_quat
from utils.ik import quat_mul, yaw_quat

SOURCE_DIR = Path(__file__).resolve().parent

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
    # The three task objects. "pos" is (x, y) on the table surface; each rests on it. "yaw" is the
    # rotation about z in degrees. Both are redrawn per episode by env_config.
    # 5 cm cubes and a 5 cm cylinder since engine 0.6.0 (4 cm through 0.5.1): a cube stays on a
    # cube until its centre passes the base's edge, so the physical placement tolerance goes from
    # 2 to 2.5 cm, the fingers tolerate 5 mm more lateral error per side, and the objects cover
    # 56% more of the 224x224 policy input. The success test's tolerances are unchanged. Cubes
    # stay cubes (one "size"), and the cylinder keeps the cubes' height so every stack geometry
    # is the same as before, 5 cm taller objects aside.
    "objects": {
        "red_cube":     {"shape": "box",      "size": 0.05,                    "color": RED,  "pos": (0.0, -0.12), "yaw": 0.0},
        "red_cylinder": {"shape": "cylinder", "radius": 0.025, "height": 0.05, "color": RED,  "pos": (0.0,  0.00), "yaw": 0.0},
        "blue_cube":    {"shape": "box",      "size": 0.05,                    "color": BLUE, "pos": (0.0,  0.12), "yaw": 0.0},
    },
    "object_density": 1000.0,  # kg/m^3, a 5 cm cube weighs 125 g
    "object_friction": 1.2,    # was 1.0 through engine 0.4.0; raised slightly for engine 0.5.0 against sliding on landing
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
    # Opposite the robot, past the +x edge, looking back at the workspace and pitched down. This is
    # the policy's input, so it was placed by measurement (2026-09-27, tool at hover over an object):
    #   pos (0.7, 0, 0.45) lookat (0, 0, 0.12), the first session:  a 4 cm tool move changes the
    #     image by 1.22/255 in x and 1.87/255 in y; a 4 cm cube covers ~456 px at 640x480.
    #   pos (0.45, 0, 0.55) lookat (-0.08, 0, 0.02), this one:      2.02/255 in x, 2.77/255 in y;
    #     ~821 px per cube. The whole placement area and the arm stay in frame.
    # x is the camera's depth axis, the direction the first policies drifted along unseen, so the
    # closer, steeper view is what makes that motion visible. An oblique view from the side scored
    # higher still (2.26 / 3.25) but skews the workspace and hides more behind the arm.
    #   Engine 0.5.0 (2026-09-28): pos (0.40, 0, 0.50), fov 45, aimed at (-0.08, 0, 0.18). Measured
    #     against the 0.4.0 camera: closer and 45 deg gives 3.71 / 3.90 per 4 cm and ~1987 px per
    #     cube (1.85x the sensitivity, 2.4x the pixels). The aim point is 18 cm up, not on the
    #     table: aimed at the table, the narrower lens lost the gripper entirely above 25 cm, so
    #     the policy could not see its own tool for the upper half of every approach (the 0.4.0
    #     camera lost it above ~40 cm). The first 0.5.0 session (piper_stack_2026-09-28_12-02-35)
    #     was recorded with that mistake; the run trained on it missed grasps by 2-7 cm. Aimed at
    #     18 cm the gripper is in frame from 45 cm down, all four placement-area corners stay in
    #     frame, and the lower third of the frame is no longer empty table.
    "front": {
        "res": (640, 480),
        "fov": 45,
        "near": 0.01,
        "far": 10.0,
        "pos": (0.40, 0.0, 0.50),
        "lookat": (-0.08, 0.0, 0.18),
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
    "show": False,                  # open the interactive viewer window; False runs headless (faster)
    "res": (1280, 960),
    "camera_pos": (1.3, -1.1, TABLE_HEIGHT + 0.7),     # world frame
    "camera_lookat": (0.0, 0.0, TABLE_HEIGHT + 0.1),   # world frame
    "camera_fov": 40,
    "world_frame_size": 0.3,       # length of the origin axes drawn in the viewer
    "draw_camera_frustums": True,  # draw each camera's frustum in the viewer
}

CONTROL = {
    # Joints in the order used for every control call.
    "arm_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    "gripper_joints": ("joint7", "joint8"),
    "kp": np.array([300, 300, 300, 100, 100, 100, 50, 50], dtype=float),          # position gains
    "kv": np.array([30, 30, 30, 10, 10, 10, 5, 5], dtype=float),                   # velocity gains
    "force_limit": np.array([100, 100, 100, 100, 100, 100, 20, 20], dtype=float),  # from the URDF effort limits
    # Pose held after build and returned to at the end of every plan: gripper pointing down about
    # 20 cm above the table centre (TCP (-0.08, 0, 0.20)), open, on the same wrist branch the
    # pick-and-place IK uses (joint4 +90, joint6 -90). Engine 0.5.1: the previous home
    # [0, 1.0, -1.2, 0, 0.6, 0] held the arm vertical with the TCP 55-60 cm up, above the front
    # camera's frame, so the first 1-1.5 s of every demonstration showed no gripper at all and the
    # camera had to keep headroom for a pose that appears nowhere else in the task. Straight-down
    # IK is only accurate up to about 20 cm this close to the base (2 cm error at 20 cm, 7 cm at
    # 25 cm, measured), which sets the height.
    "home_qpos": np.array([0.0, 1.2, -1.18, 1.57, 0.0, -1.57, 0.04, -0.04]),
    # Gripper joint positions (joint7, joint8) for gripper(open=1.0) and gripper(open=0.0).
    "gripper_open_qpos": (0.05, -0.05),
    "gripper_closed_qpos": (0.0, 0.0),
}

IK = {
    # Envs simulated in parallel. On CUDA this is bounded by what the batch renderer can allocate,
    # not by speed: Madrona takes a large fixed pool plus roughly 20 MB per env, and it aborts in
    # makeDedicatedImage when it runs out rather than raising something catchable. Measured on an
    # 11 GB RTX 2080 Ti already driving a display (~2.2 GB used before Genesis starts), with three
    # 640x480 cameras: 32 envs peaks at 8.6 GB, 40 at 9.0 GB, 48 at 9.3 GB, and 64 fails outright.
    # 40 leaves about 2.3 GB of headroom and is a multiple of the five allowed task pairs, so every
    # epoch draws each pair the same number of times. Re-measure after changing cameras or res.
    "n_envs": 40,
    "env_spacing": (1.5, 1.5),           # viewer-only grid spacing between environments
    "ee_link": "Link6",                  # wrist link the gripper hangs off; the URDF has no tool-center link
    "tcp_offset": (0.0, 0.0, 0.1308),    # center of the finger pads in the ee_link frame, measured
                                         # measured from the URDF meshes
    "target_pos": (0.0, 0.0, 0.12),      # tool-center target, table coords: 12 cm above the red cylinder
    "target_quat": (0.0, 1.0, 0.0, 0.0),  # (w, x, y, z). Gripper pointing straight down, fingers closing along x.
    "rot_mask": (True, True, True),      # constrain full orientation. (False, False, True) frees the yaw.
    "apply": "control",                  # "control": PD-drive the arm to the solution. "set": teleport the joints.
    "settle_steps": 300,                 # sim steps to give the arm to reach the target
}

SIM = {
    "dt": 1 / 90,             # 90 Hz physics: an integer multiple of the 30 Hz recording rate
    "n_steps": 2700,          # stacking_scene.py: 30 s of sim time
    "hold_after_plan": 1.0,   # record.py: seconds to keep simulating after the last plan finishes
    # Compute backend: "auto" is Genesis's own pick (CUDA, then AMD, then Metal, then CPU), "cuda"
    # and "cpu" force one. The backend also decides how the cameras render: on CUDA every env and
    # camera goes through the Madrona batch renderer in a single GPU pass, anywhere else each env is
    # rasterized on the CPU one at a time.
    #
    # It does not decide recording throughput. A recording run is limited by the LeRobot writer, not
    # by the simulator, so the two backends finish a round in the same wall time (measured at 16
    # envs, one epoch: 9.4 episodes/min on CUDA against 9.3 on CPU). What CUDA buys is headroom:
    # rendering drops from the largest cost in the loop to about 2% of it, so envs, cameras and
    # resolution become nearly free to raise, and the writer is then the only thing to beat.
    "backend": "auto",
}

# The language-conditioned task: pick `source` and place it on `destination` (two different objects).
# source/destination here are the defaults; with randomization.task on, env_config draws a pair per episode.
TASK = {
    "source": "red_cylinder",
    "destination": "blue_cube",
    "instruction": "put the {source} on the {destination}",   # object names are humanized: "red cylinder"
    # Ordered (source, destination) pairs never used for data collection. Three objects give six
    # ordered pairs; hold one out entirely and use it as the generalization test. "blue cube on red
    # cube" is held out: cube-on-cube is the most stable placement, so a failure on it is a language
    # grounding failure rather than a physics one, and the red cube (the only object that needs both
    # colour and shape to identify) stays in training as a grasp target in both of its source pairs.
    "holdout_pairs": [("blue_cube", "red_cube")],
}

# Per-episode randomization, drawn in env_config from (SEED, env, episode), so a session is
# reproducible and every env and epoch differs.
RANDOMIZATION = {
    "task": {
        "randomize": True,            # assign the allowed (source, destination) pairs round-robin by global
                                      # episode number, so attempts are balanced exactly across envs, workers and epochs
    },
    "orientation": {
        "randomize": True,            # random yaw per object each episode; the grasp turns to match the picked object
        "yaw_range": (-180.0, 180.0), # degrees
    },
    # Sideways offset of the hover points above the source and the destination, drawn per episode
    # for each. This is what puts corrections into the demonstrations: with the hover displaced,
    # the descent has to close a visible gap to the object, so the recorded actions depend on where
    # the tool is relative to its target instead of only on the phase of the plan. Measured on the
    # first session, recorded without this, the tool sat within 0.8 mm (std) of the same spot
    # relative to the object at every descent, and the policies trained on it ignored 4 cm
    # displacements at hover entirely. 0.03 m is about the error those policies made.
    "approach": {
        "randomize": True,
        # m; (dx, dy) uniform in a square of this half-width. 0.03 in engine 0.3.0 taught
        # corrections of up to 3 cm; the policy then landed 5-8 cm short on objects at the edge of
        # the placement area, where its own approach error exceeds what it ever saw corrected.
        "max_offset": 0.05,
    },
    # Start pose: control.home_qpos with a small per-episode perturbation of the arm joints, so the
    # first frames of the demonstrations are not one identical picture and a policy does not tie
    # its opening move to a single pixel-exact arm configuration. The gripper joints are left alone.
    "start": {
        "randomize": True,
        "joint_range_deg": 5.0,       # each arm joint: uniform in +-this, degrees
        # Engine 0.6.0: the plan then moves the tool, unrecorded, to a random point in this box
        # (table coords, m) before the recorded approach begins, so demonstrations start from
        # anywhere the policy might find itself rather than from one home pose. The box is inside
        # the front camera's frame at these heights (coverage measured 2026-09-28) and within the
        # reach of the straight-down IK (accurate to 2 cm up to about 20 cm near the base).
        "tool_box": {"x": (-0.15, 0.05), "y": (-0.18, 0.18), "z": (0.12, 0.20)},
        # The wrist yaw the start move leaves, uniform in +-this about the base orientation
        # (degrees); the recorded approach then turns to the grasp yaw, which is the turn the
        # policy has to learn. A cube's grasp yaw is within +-45 deg of the base, so this range
        # covers every turn the pick can need.
        "yaw_range_deg": 45.0,
    },
    # Disturbance of the executed command during the plan's moves (utils/motion.py, PlanRunner):
    # smooth random pushes. Every `interval` seconds a new push target is drawn uniformly within
    # +-sigma (m, per phase; z scaled by z_factor) and the offset glides toward it at `rate` m/s.
    # The recorded label is the clean pursuit rate back toward the waypoint, so every disturbed
    # frame teaches a correction. Phases with the fingers around an object carry none; the pursuit
    # of those moves removes an inherited offset at the same rate (2 cm in 0.4 s).
    "path_noise": {
        "enabled": True,
        "sigma": {"transit": 0.02, "descent": 0.012, "final": 0.0, "lift": 0.0, "retreat": 0.01},
        "interval": 0.5,
        "rate": 0.05,
        "z_factor": 0.5,
        # The wrist yaw, the same way: the executed yaw is pushed within +-yaw_sigma_deg (per
        # phase) at up to yaw_push_rate_deg per second, while the clean turn toward the grasp yaw
        # runs at yaw_speed_deg per second and is the recorded rotation label. Run 10, recorded
        # with the wrist already aligned before the first frame, never learned to turn it and
        # gripped every cube diagonally (task2.md).
        "yaw_sigma_deg": {"transit": 10.0, "descent": 8.0, "final": 0.0, "lift": 0.0, "retreat": 5.0},
        "yaw_speed_deg": 120.0,
        "yaw_push_rate_deg": 45.0,
        "yaw_settle_s": 0.1,     # within the last 12 deg the turn rate shrinks with the error (one 10 Hz step)
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


def make_plan(task, scene, hover=0.10, pick_offset=(0.0, 0.0), place_offset=(0.0, 0.0), start=None, start_yaw=0.0,
              noise=None, yaw_noise=None):
    """Motion plan for `task` on the object layout in `scene`, executed in order by record.py.

    Targets are object references, resolved to table coords per env by env_config as its last
    step, after any randomization. ik(Obj(name)()) centers the gripper on that object. The gripper
    orientation is IK["target_quat"] (pointing down) turned about z to close across the picked
    object's faces given its yaw, and is held for the whole pick-and-place so the wrist never
    twists mid-carry. See utils/motion.py for the ik options (path shape, timing profile, via points).

    pick_offset / place_offset displace the hover points sideways (see randomization.approach).
    The opening home and idle settle the arm before it moves; the recorder does not store those
    frames (see Recorder.capture), so a recorded episode starts with the first frame that moves.
    """
    src, dst = Obj(task["source"]), Obj(task["destination"])
    quat = grasp_quat(scene["objects"][task["source"]], IK["target_quat"])
    # Grasp 5 mm below the object's centre (engine 0.5.0), a margin against the policy's measured
    # habit of closing 1.5-2.5 cm above where the demonstrations do; the demonstrations themselves
    # still grasp exactly where aimed. Scripted plan on 10 layouts: 5 mm low stacks 10/10 even with
    # the grasp shifted 1 cm sideways; 1 cm above centre stacks 5/10, 2 cm above 0/10. The early
    # close itself is addressed by the gripper label (train/config/openvla.py, lookahead False).
    grasp = offset(src(), (0.0, 0.0, -0.005))
    # The randomized start (randomization.start.tool_box, yaw_range_deg): an unrecorded move from
    # home to the drawn point with the wrist at the drawn yaw, so the recorded approach begins
    # there, from the first frame, at speed, and turning toward the grasp yaw.
    start_quat = tuple(float(v) for v in quat_mul(yaw_quat(start_yaw), np.asarray(IK["target_quat"], dtype=float)))
    start_move = [ik(start, 0.15, quat=start_quat, gripper=1.0, record=False)] if start is not None else []
    return [
        home(0.5),
        *start_move,
        # Engine 0.6.0: eight straight-line moves at two constant speeds (see pick_and_place).
        # 0.15 m/s everywhere the tool travels, 15 mm per 10 Hz control step, the carry speed of
        # 0.5.0 (the policy corrects about 25% of its lateral error per step on a descent, so the
        # slower approach gives the pick descent the same number of correcting steps the place
        # had); 0.05 m/s, 5 mm per step, for the last 3 cm onto an object and the first 2 cm of
        # the lift. No ramps: the trapezoid profile of 0.4.0-0.5.1 decelerated into the place
        # over 12 frames of shrinking actions and the policy trained on it stalled just above the
        # release height (task2.md, run 8).
        # The place is a 1.5 cm drop. Scripted, friction 1.2, 10 layouts: at the nominal grasp
        # 1.0 / 1.5 / 2.0 cm all stack 10/10; grasped 1 cm too high they stack 7 / 8 / 9. A
        # set-down (3 mm) pushes a high-grasped object into the destination, 0/10, so the drop's
        # clearance is kept as the margin for the grasp error the policy makes.
        *pick_and_place(pick=grasp, place=on_top(dst, src, clearance=0.015), hover=hover, quat=quat,
                        pick_offset=pick_offset, place_offset=place_offset, speed=0.15, slow=0.05,
                        noise=noise, yaw_noise=yaw_noise),
        home(1.0),   # return home after the retreat; the success check starts when this action begins
    ]


# What record.py stores per frame, on top of the cameras. Every quantity is in table coords.
# "delta" observations are the change since the previous frame; "delta" actions are the command
# relative to the current observation. Turn any of them off to skip it.
RECORD = {
    # Episodes per env; a session records workers * n_envs * epochs attempts. 40 envs x 15 epochs
    # = 600 attempts, which at the ~97% success rate of this plan lands around 580 saved
    # demonstrations. Failed attempts are discarded, so ask for a few more than you need.
    "epochs": 17,
    # Recorder processes run in parallel, each simulating ik.n_envs envs and merged into one dataset
    # at the end. On a CPU backend camera rendering is single-threaded per process and is the
    # bottleneck, so more envs in one process stop helping around 16 while extra processes scale
    # nearly linearly (measured: 10.4 episodes/min alone, 9.6 each with two running), at about
    # 2.5 GB of RAM per process at 16 envs.
    #
    # On CUDA that no longer holds. The batch renderer draws every env at once, so rendering stops
    # being the limit (measured at 32 envs: 1.6 ms of physics and 18 ms of rendering per step, but
    # 19 ms just to copy the frames back to host, and roughly twice that again inside the LeRobot
    # writer encoding them). What is left is CPU-bound and per-env: the h264 encode in add_frame and
    # the dataset finalization in save_episode together dominate a recording run. Extra workers
    # still help, but they now compete for GPU memory instead of RAM: the batch renderer needs about
    # 8.7 GB for 64 envs x 3 cameras at 640x480, so a single 11 GB card fits one worker at that size
    # and needs a smaller ik.n_envs to fit more.
    "workers": 1,
    "fps": 30,                                        # frames per second of sim time; the sim runs at 1/sim.dt
    "root": str(SOURCE_DIR.parent / "datasets"),      # one session directory per recording run goes here
    "repo_id": "axibo/piper_stack",                   # LeRobot dataset name; also the session directory prefix
    "robot_type": "piper_x",
    # Dataset video codec. One encoder stays live per camera per env, so the codec must be cheap per
    # instance: h264 at 1 thread is one thread and ~20 MB each (64 envs: 195 threads, 3.6 GB), while
    # libsvtav1 spawns ~17 threads per instance and hits macOS's 4096-thread cap around 64 envs.
    "vcodec": "h264",
    "encoder_threads": 1,
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
        "gripper_target": True,                       # the plan's gripper decision, 0/1 (action.gripper_target)
        "clean_rate": True,                           # the undisturbed pursuit rate toward the next waypoint, m/s (action.clean_rate)
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
    "control": CONTROL,
    "ik": IK,
    "sim": SIM,
    "task": TASK,
    "randomization": RANDOMIZATION,
    "success": SUCCESS,
    "plan": make_plan(TASK, SCENE),
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
        # Round-robin by global attempt number: exact balance across envs, workers and epochs.
        pairs = allowed_pairs(cfg["task"], list(cfg["scene"]["objects"]))
        total_envs = cfg["ik"]["n_envs"] * cfg["record"].get("workers", 1)
        cfg["task"]["source"], cfg["task"]["destination"] = pairs[(episode_idx * total_envs + env_idx) % len(pairs)]

    if rand["placement"]["randomize"]:
        for name, xy in sample_placement(cfg["scene"], rng, rand["placement"]).items():
            cfg["scene"]["objects"][name]["pos"] = xy

    if rand["orientation"]["randomize"]:
        lo, hi = rand["orientation"]["yaw_range"]
        for spec in cfg["scene"]["objects"].values():
            spec["yaw"] = float(rng.uniform(lo, hi))

    start_point = None
    if rand.get("start", {}).get("randomize"):
        n_arm = len(cfg["control"]["arm_joints"])
        home_qpos = np.array(cfg["control"]["home_qpos"], dtype=float)
        home_qpos[:n_arm] += np.deg2rad(rng.uniform(-rand["start"]["joint_range_deg"], rand["start"]["joint_range_deg"], size=n_arm))
        cfg["control"]["home_qpos"] = home_qpos
        box = rand["start"].get("tool_box")
        if box:
            start_point = tuple(float(rng.uniform(*box[k])) for k in ("x", "y", "z"))
    start_yaw = 0.0
    if start_point is not None and rand["start"].get("yaw_range_deg"):
        start_yaw = float(np.deg2rad(rng.uniform(-rand["start"]["yaw_range_deg"], rand["start"]["yaw_range_deg"])))
    cfg["start_point"] = start_point
    cfg["start_yaw"] = start_yaw

    offsets = {"pick_offset": (0.0, 0.0), "place_offset": (0.0, 0.0)}
    if rand.get("approach", {}).get("randomize"):
        r = rand["approach"]["max_offset"]
        offsets = {k: tuple(float(v) for v in rng.uniform(-r, r, size=2)) for k in offsets}
    cfg["approach_offsets"] = offsets   # kept in the session's episode records, so a dataset can be checked

    noise = rand.get("path_noise", {})
    sigma = dict(noise.get("sigma", {})) if noise.get("enabled") else {}
    yaw_sigma = dict(noise.get("yaw_sigma_deg", {})) if noise.get("enabled") else {}
    cfg["noise_seed"] = int(rng.integers(2**31 - 1))   # the runner's disturbance generator for this env

    # Last step: build the plan for this task and layout, and turn object references into table coords.
    cfg["plan"] = resolve_plan(make_plan(cfg["task"], cfg["scene"], start=start_point, start_yaw=start_yaw,
                                         noise=sigma, yaw_noise=yaw_sigma, **offsets), cfg["scene"])
    return cfg
