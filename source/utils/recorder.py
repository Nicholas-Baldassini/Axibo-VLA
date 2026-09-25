"""
Record demonstrations into a LeRobot v3 dataset, one episode per successful env attempt.

A recording session is a directory  <record.root>/<dataset name>_<YYYYmmdd_HHMMSS>/  holding
    metadata.json           date, project version, library versions, platform, fps, attempted /
                            succeeded / failed episode counts, failure reasons, success tolerances
    config.json             the base config
    episodes.json           every attempt: epoch, env, task, instruction, success, reason, frames,
                            dataset episode index (successes only), and the resolved env config
    requirements.txt        copy of the repo's requirements.txt
    requirements.lock.txt   pip freeze of the recording environment
    dataset/                the LeRobotDataset (meta/, data/, videos/)

How frames flow. Every env has its own LeRobotDataset writer with streaming video encoding, so a
captured frame goes straight into that env's episode and its videos are encoded in the background
while the simulation keeps stepping; nothing is buffered. When an env's attempt terminates, its
episode is saved (success) or discarded (failure). At the end the per-env datasets are aggregated
into dataset/ by remuxing, without re-encoding, and removed.

Features. All vectors are float32; every cartesian quantity is in table coords (x, y from the table
center, z above the surface); quaternions are (w, x, y, z) with their sign kept continuous within an
episode (no q / -q jumps); rotation deltas are rotation vectors.
    observation.images.<camera>     (H, W, 3) video
    observation.state               joint positions, arm joints then gripper joints (LeRobot's name for it)
    observation.joint_pos_delta     change since the previous frame (zeros on the first frame)
    observation.joint_vel           joint velocities;  observation.joint_vel_delta  change since the previous frame
    observation.tcp_pose            [x, y, z, qw, qx, qy, qz, gripper]  tool center pose, gripper opening 0..1
    observation.tcp_pose_delta      [dx, dy, dz, rx, ry, rz, dgripper]
    observation.link_poses          [x, y, z, qw, qx, qy, qz] per robot link;  observation.link_poses_delta  6 per link
    observation.objects             [x, y, z, qw, qx, qy, qz] per object
    action                          commanded [x, y, z, qw, qx, qy, qz, gripper]  (absolute tool pose)
    action.tcp_pose_delta           the command relative to the current observation
    action.joint_pos                commanded joint targets;  action.joint_pos_delta  targets minus observed
    action.joint_vel                rate of the joint targets;  action.joint_vel_delta  that minus observed velocity
    action.link_poses               link poses reached at the next frame (Genesis exposes FK only on the simulated
                                    state);  action.link_poses_delta  next minus current
    success                         always 1.0, since only successful attempts are written
"""

import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from genesis.utils.misc import tensor_to_array
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.cameras import describe as describe_camera
from utils.frames import table_top_z
from utils.ik import to_numpy, tcp_pos_from_link
from utils.motion import Target, Obj

POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
POSE_DELTA_NAMES = ["dx", "dy", "dz", "rx", "ry", "rz"]


# ------------------------------ small helpers ------------------------------

def canonical(q):
    """Quaternions (..., 4) with their largest-magnitude component positive, so q and -q never both
    appear for one rotation. (A w >= 0 rule would flip around orientations with w ~ 0, such as the
    gripper pointing straight down.)"""
    q = np.asarray(q, dtype=float)
    lead = np.take_along_axis(q, np.abs(q).argmax(axis=-1, keepdims=True), axis=-1)
    return np.where(lead < 0, -q, q)


def continuous(q, prev):
    """Quaternions (..., 4) with the sign that agrees with `prev` (dot >= 0), so a stream never jumps
    between q and -q within an episode. The first frame (prev None) uses the fixed canonical rule."""
    q = np.asarray(q, dtype=float)
    if prev is None:
        return canonical(q)
    return np.where(np.sum(q * prev, axis=-1, keepdims=True) < 0, -q, q)


def rot_delta(q_from, q_to):
    """Rotation vectors taking q_from to q_to, batched (n, 4) (w, x, y, z) -> (n, 3)."""
    r_from = Rotation.from_quat(np.asarray(q_from)[:, [1, 2, 3, 0]])
    r_to = Rotation.from_quat(np.asarray(q_to)[:, [1, 2, 3, 0]])
    return (r_from.inv() * r_to).as_rotvec().astype(np.float32)


def pose_delta(pos_from, quat_from, pos_to, quat_to):
    return np.concatenate([np.asarray(pos_to) - np.asarray(pos_from), rot_delta(quat_from, quat_to)], axis=-1)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, Target):
        return {"_target": x._asdict()}
    if isinstance(x, Obj):
        return {"_obj": x.name}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return str(x)


def _cpu_name():
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def hardware_info():
    """The machine a session was recorded on."""
    import torch
    try:
        import genesis as gs
        b = getattr(gs, "backend", None)
        backend = None if b is None else (getattr(b, "name", None) or str(b))   # e.g. "cpu", "cuda", "metal"
    except Exception:
        backend = None
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        gpu = "Apple GPU (Metal)"
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        ram_gb = None
    return {
        "cpu": _cpu_name(),
        "cores": os.cpu_count(),
        "ram_gb": ram_gb,
        "gpu": gpu,
        "genesis_backend": backend,
        "machine": platform.machine(),
        "os": platform.platform(),
        "hostname": platform.node(),
    }


# --------------------------------- session ---------------------------------

class Session:
    """The directory for one recording run and its metadata files."""

    def __init__(self, config, project_version, source_dir):
        rec = config["record"]
        self.created = dt.datetime.now()
        base = f"{rec['repo_id'].split('/')[-1]}_{self.created:%Y-%m-%d_%H-%M-%S}"
        root = Path(rec["root"])
        self.name = base
        for n in range(2, 1000):            # two runs in the same second: never overwrite
            if not (root / self.name).exists():
                break
            self.name = f"{base}_{n}"
        self.dir = root / self.name
        self.dir.mkdir(parents=True, exist_ok=False)
        self.dataset_dir = self.dir / "dataset"
        self.project_version = project_version
        self.source_dir = Path(source_dir)
        (self.dir / "config.json").write_text(json.dumps(to_jsonable(config), indent=2))

    def write(self, config, attempts, wall_time_s):
        """attempts: list of dicts with epoch, env, task, instruction, success, reason, frames, episode_index, config."""
        import genesis, lerobot, torch

        (self.dir / "episodes.json").write_text(json.dumps(to_jsonable(attempts), indent=2))

        req = self.source_dir.parent / "requirements.txt"
        if req.exists():
            shutil.copy(req, self.dir / "requirements.txt")
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        (self.dir / "requirements.lock.txt").write_text(freeze.stdout)

        succeeded = [a for a in attempts if a["success"]]
        failed = [a for a in attempts if not a["success"]]
        metadata = {
            "session": self.name,
            "created_at": self.created.isoformat(timespec="seconds"),
            "project_version": self.project_version,
            "seed": config["seed"],
            "hardware": hardware_info(),
            "dataset_format": "lerobot_v3",
            "repo_id": config["record"]["repo_id"],
            "fps": config["record"]["fps"],
            "sim_dt": config["sim"]["dt"],
            "n_envs": config["ik"]["n_envs"],
            "epochs": config["record"]["epochs"],
            "episodes_attempted": len(attempts),
            "episodes_succeeded": len(succeeded),
            "episodes_failed": len(failed),
            "failure_reasons": dict(Counter(a["reason"] for a in failed)),
            "success_rate": len(succeeded) / len(attempts) if attempts else None,
            "frames": int(sum(a["frames"] for a in succeeded)),
            "instructions": dict(Counter(a["instruction"] for a in succeeded)),
            "holdout_pairs": [list(p) for p in config["task"].get("holdout_pairs", [])],
            "success_criteria": config["success"],
            "randomization": to_jsonable(config["randomization"]),
            "cameras": {name: describe_camera(config["cameras"][name]) for name in config["record"]["cameras"]},
            "wall_time_s": round(wall_time_s, 1),
            "versions": {
                "python": platform.python_version(),
                "genesis": genesis.__version__,
                "lerobot": lerobot.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
            "platform": platform.platform(),
        }
        (self.dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return metadata


# --------------------------------- recorder ---------------------------------

class Recorder:
    """Streams observations, commands and camera frames from a PlanRunner into one dataset per env."""

    def __init__(self, robot, objects, cameras, dofs_idx, config, session):
        self.robot = robot
        self.objects = objects
        self.cameras = cameras
        self.config = config
        rec = config["record"]
        self.fps = rec["fps"]
        self.n = config["ik"]["n_envs"]

        steps_per_frame = (1.0 / config["sim"]["dt"]) / self.fps
        if abs(steps_per_frame - round(steps_per_frame)) > 1e-6:
            raise ValueError(f"1/sim.dt = {1 / config['sim']['dt']:.3f} Hz must be an integer multiple of record.fps = {self.fps}")
        self.every = int(round(steps_per_frame))

        ctrl = config["control"]
        self.dofs = list(dofs_idx)
        self.joint_names = list(ctrl["arm_joints"]) + list(ctrl["gripper_joints"])
        self.n_arm = len(ctrl["arm_joints"])
        self.grip_open = np.asarray(ctrl["gripper_open_qpos"], dtype=float)
        self.grip_closed = np.asarray(ctrl["gripper_closed_qpos"], dtype=float)
        self.links = list(robot.links)
        self.link_names = [l.name for l in self.links]
        self.object_names = list(config["scene"]["objects"])
        self.cam_names = list(rec["cameras"])
        self.ee_link = robot.get_link(config["ik"]["ee_link"])
        self.tcp_offset = config["ik"]["tcp_offset"]
        self.table_offset = np.array([*config["scene"]["table"]["pos"][:2], table_top_z(config)], dtype=float)

        self.obs_on = {k for k, v in rec["observation"].items() if v}
        self.act_on = {k for k, v in rec["action"].items() if v}
        self.features = self._build_features()
        self.numeric = [n for n in self.features if not n.startswith("observation.images.") and n != "success"]
        self.lookahead = [n for n in ("action.link_poses", "action.link_poses_delta") if n in self.features]

        # One writer per env, each streaming its videos as frames arrive; merged at finalize().
        self.session = session
        self.repo_id = rec["repo_id"]
        self.env_dirs = [session.dir / f"dataset_env{i}" for i in range(self.n)]
        self.datasets = [
            LeRobotDataset.create(
                f"{self.repo_id}_env{i}", fps=self.fps, features=self.features, root=self.env_dirs[i],
                robot_type=rec.get("robot_type", "piper_x"), use_videos=True, vcodec=rec.get("vcodec", "libsvtav1"),
                streaming_encoding=True,
            )
            for i in range(self.n)
        ]
        self.local_episodes = [0] * self.n     # episodes saved so far per env
        self.runner = None

    # ---- feature spec ----

    def _build_features(self):
        J = self.joint_names
        L, O = self.link_names, self.object_names
        pose = lambda names: [f"{n}.{c}" for n in names for c in POSE_NAMES]
        pose_d = lambda names: [f"{n}.{c}" for n in names for c in POSE_DELTA_NAMES]
        tcp = POSE_NAMES + ["gripper"]
        tcp_d = POSE_DELTA_NAMES + ["dgripper"]

        obs = {
            "joint_pos": ("observation.state", J),
            "joint_pos_delta": ("observation.joint_pos_delta", [f"d{j}" for j in J]),
            "joint_vel": ("observation.joint_vel", [f"v{j}" for j in J]),
            "joint_vel_delta": ("observation.joint_vel_delta", [f"dv{j}" for j in J]),
            "tcp_pose": ("observation.tcp_pose", tcp),
            "tcp_pose_delta": ("observation.tcp_pose_delta", tcp_d),
            "link_poses": ("observation.link_poses", pose(L)),
            "link_poses_delta": ("observation.link_poses_delta", pose_d(L)),
            "objects": ("observation.objects", pose(O)),
        }
        act = {
            "tcp_pose": ("action", tcp),
            "tcp_pose_delta": ("action.tcp_pose_delta", tcp_d),
            "joint_pos": ("action.joint_pos", J),
            "joint_pos_delta": ("action.joint_pos_delta", [f"d{j}" for j in J]),
            "joint_vel": ("action.joint_vel", [f"v{j}" for j in J]),
            "joint_vel_delta": ("action.joint_vel_delta", [f"dv{j}" for j in J]),
            "link_poses": ("action.link_poses", pose(L)),
            "link_poses_delta": ("action.link_poses_delta", pose_d(L)),
        }
        features = {}
        for cam in self.cam_names:
            w, h = self.config["cameras"][cam]["res"]
            features[f"observation.images.{cam}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
        for key, (name, names) in obs.items():
            if key in self.obs_on:
                features[name] = {"dtype": "float32", "shape": (len(names),), "names": names}
        for key, (name, names) in act.items():
            if key in self.act_on:
                features[name] = {"dtype": "float32", "shape": (len(names),), "names": names}
        features["success"] = {"dtype": "float32", "shape": (1,), "names": ["success"]}
        return features

    # ---- rounds ----

    def start_round(self, runner, instructions):
        """Begin a round: every env starts a new episode driven by `runner`."""
        self.runner = runner
        self.instructions = list(instructions)
        self.active = np.ones(self.n, dtype=bool)   # still recording
        self.n_frames = np.zeros(self.n, dtype=int)
        self.pending = [None] * self.n               # last frame per env, waiting for the next observation
        self.prev = None
        self.prev_target_q = None
        self.prev_quat_cmd = None

    def _push(self, i, frame, next_obs_rows=None):
        """Complete a pending frame's look-ahead actions and hand it to env i's writer."""
        if self.lookahead:
            if next_obs_rows is not None:
                frame["action.link_poses"] = next_obs_rows["observation.link_poses"][i]
                frame["action.link_poses_delta"] = next_obs_rows["observation.link_poses_delta"][i]
            else:   # last frame of the episode: the links stay where they are
                frame["action.link_poses"] = frame["observation.link_poses"]
                frame["action.link_poses_delta"] = np.zeros_like(frame["observation.link_poses_delta"])
        self.datasets[i].add_frame(frame)

    def end_episode(self, i, success):
        """Stop recording env i: save its episode if the attempt succeeded, else discard it.
        Returns the env-local episode index of a saved episode, else None."""
        if self.pending[i] is not None:
            self._push(i, self.pending[i])
            self.pending[i] = None
        self.active[i] = False
        if success:
            self.datasets[i].save_episode()
            self.local_episodes[i] += 1
            return self.local_episodes[i] - 1
        self.datasets[i].clear_episode_buffer()
        return None

    def finalize(self):
        """Flush every env's writer, merge them into the session dataset, and remove the per-env copies.
        Returns {env: offset} such that final episode index = offset + env-local index."""
        for ds in self.datasets:
            ds.finalize()
        with_episodes = [i for i in range(self.n) if self.local_episodes[i] > 0]
        offsets = {}
        if with_episodes:
            aggregate_datasets(
                [f"{self.repo_id}_env{i}" for i in with_episodes], self.repo_id,
                roots=[self.env_dirs[i] for i in with_episodes], aggr_root=self.session.dataset_dir,
            )
            total = 0
            for i in with_episodes:
                offsets[i] = total
                total += self.local_episodes[i]
        for d in self.env_dirs:
            shutil.rmtree(d, ignore_errors=True)
        return offsets

    # ---- capture ----

    def hook(self, runner):
        """PlanRunner hook: runs after the command for this step is issued, before the sim steps."""
        if runner.step_count % self.every == 0 and self.active.any():
            self.capture(runner)

    def _to_table(self, pos_world):
        return np.asarray(pos_world, dtype=float) - self.table_offset

    def _gripper_opening(self, q):
        """Normalized 0..1 opening from the first gripper joint of (n, n_dofs) joint positions."""
        j = self.n_arm
        return np.clip((q[:, j] - self.grip_closed[0]) / (self.grip_open[0] - self.grip_closed[0]), 0.0, 1.0)[:, None]

    def _observe(self):
        p = self.prev
        q = to_numpy(self.robot.get_dofs_position(self.dofs)).astype(float)
        qd = to_numpy(self.robot.get_dofs_velocity(self.dofs)).astype(float)
        link_pos = np.stack([self._to_table(to_numpy(l.get_pos())) for l in self.links], axis=1)      # (n, L, 3)
        link_quat = np.stack([to_numpy(l.get_quat()) for l in self.links], axis=1)                    # (n, L, 4)
        link_quat = continuous(link_quat, None if p is None else p["link_quat"])
        tcp_pos = self._to_table(tcp_pos_from_link(self.ee_link, self.tcp_offset))
        tcp_quat = continuous(to_numpy(self.ee_link.get_quat()), None if p is None else p["tcp_quat"])
        grip = self._gripper_opening(q)
        obj_pos = np.stack([self._to_table(to_numpy(self.objects[o].get_pos())) for o in self.object_names], axis=1)
        obj_quat = np.stack([to_numpy(self.objects[o].get_quat()) for o in self.object_names], axis=1)
        obj_quat = continuous(obj_quat, None if p is None else p["obj_quat"])
        return {"q": q, "qd": qd, "link_pos": link_pos, "link_quat": link_quat,
                "tcp_pos": tcp_pos, "tcp_quat": tcp_quat, "grip": grip, "obj_pos": obj_pos, "obj_quat": obj_quat}

    def _link_pose_delta(self, a, b):
        n, L = a["link_pos"].shape[:2]
        d = pose_delta(a["link_pos"].reshape(-1, 3), a["link_quat"].reshape(-1, 4),
                       b["link_pos"].reshape(-1, 3), b["link_quat"].reshape(-1, 4))
        return d.reshape(n, L * 6)

    def capture(self, runner):
        o = self._observe()
        p = self.prev if self.prev is not None else o
        n = self.n
        rows = {}

        # Observations.
        rows["observation.state"] = o["q"]
        rows["observation.joint_pos_delta"] = o["q"] - p["q"]
        rows["observation.joint_vel"] = o["qd"]
        rows["observation.joint_vel_delta"] = o["qd"] - p["qd"]
        rows["observation.tcp_pose"] = np.concatenate([o["tcp_pos"], o["tcp_quat"], o["grip"]], axis=1)
        rows["observation.tcp_pose_delta"] = np.concatenate(
            [pose_delta(p["tcp_pos"], p["tcp_quat"], o["tcp_pos"], o["tcp_quat"]), o["grip"] - p["grip"]], axis=1)
        rows["observation.link_poses"] = np.concatenate([o["link_pos"], o["link_quat"]], axis=2).reshape(n, -1)
        rows["observation.link_poses_delta"] = self._link_pose_delta(p, o)
        rows["observation.objects"] = np.concatenate([o["obj_pos"], o["obj_quat"]], axis=2).reshape(n, -1)

        # Commands issued this step.
        tq = np.asarray(runner.target_q, dtype=float)
        ptq = self.prev_target_q if self.prev_target_q is not None else tq
        tcp_cmd = self._to_table(runner.tcp_cmd)
        quat_cmd = continuous(runner.quat_cmd, self.prev_quat_cmd)
        grip_cmd = self._gripper_opening(tq)
        rows["action"] = np.concatenate([tcp_cmd, quat_cmd, grip_cmd], axis=1)
        rows["action.tcp_pose_delta"] = np.concatenate(
            [pose_delta(o["tcp_pos"], o["tcp_quat"], tcp_cmd, quat_cmd), grip_cmd - o["grip"]], axis=1)
        rows["action.joint_pos"] = tq
        rows["action.joint_pos_delta"] = tq - o["q"]
        rows["action.joint_vel"] = (tq - ptq) * self.fps
        rows["action.joint_vel_delta"] = rows["action.joint_vel"] - o["qd"]
        rows = {k: v.astype(np.float32) for k, v in rows.items()}

        images = {cam: tensor_to_array(self.cameras[cam].read().rgb) for cam in self.cam_names}   # (n, H, W, 3)

        for i in np.flatnonzero(self.active):
            frame = {name: rows[name][i] for name in self.numeric if name not in self.lookahead}
            frame["success"] = np.array([1.0], dtype=np.float32)
            frame["task"] = self.instructions[i]
            for cam in self.cam_names:
                frame[f"observation.images.{cam}"] = np.ascontiguousarray(images[cam][i])
            if self.pending[i] is not None:
                self._push(i, self.pending[i], next_obs_rows=rows)   # its look-ahead actions are now known
            self.pending[i] = frame
            self.n_frames[i] += 1

        self.prev = o
        self.prev_target_q = tq.copy()
        self.prev_quat_cmd = quat_cmd.copy()
