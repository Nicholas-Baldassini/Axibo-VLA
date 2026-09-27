"""
Run a fine-tuned OpenVLA policy in the simulator from a small GUI, for qualitative testing.

    python train/infer.py train/runs/openvla_2026-09-26_00-01-05
    python train/infer.py train/runs/openvla_2026-09-26_00-01-05/step_001000

The run directory is the only argument it needs. Everything else is read from what the run saved:

    run/config.json              the training config, so the policy is rebuilt exactly as trained
                                 (base checkpoint, 4-bit or not, dtype, which camera it sees, how
                                 the action vector is laid out and at what stride), plus the path
                                 of the recording session it learned from
    <checkpoint>/adapter/        the LoRA weights and the processor
    .../dataset_statistics.json  the q01/q99 bounds that un-normalize a prediction
    session/config.json          the scene it was recorded in: the objects, the task phrasing and
                                 the pair that was held out of training

The window shows what the policy sees, what it is doing and what it is asking for: the camera
images (the one the model reads is framed), the joint and tool traces, and the predicted action
over time. Type an instruction in the box, or click one of the task buttons; the held-out pair is
marked so a deliberate out-of-distribution test is one click away.

Inference follows OpenVLA's own predict_action: the image goes through the checkpoint's processor,
the instruction into the prompt the model was trained on, seven tokens are decoded greedily, mapped
back to bin centres and un-normalized with the training statistics. The action is a delta on the
commanded tool pose plus an absolute gripper opening, applied at 1/stride of the recording rate.
"""

import setuptools  # noqa: F401  (imported first on purpose, see below)
# This is the only module that loads both Genesis and peft. Genesis imports the standard library's
# distutils, and peft later pulls in setuptools, whose _distutils_hack then asserts that distutils
# came from setuptools and dies. Importing setuptools before anything else lets the hack install
# its override first, which costs nothing and keeps the two stacks in one process.

import argparse
import copy
import json
import os
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(ROOT / "source"))

from actions import ActionTokenizer, unnormalize_actions   # noqa: E402

HISTORY = 300          # control steps kept in the traces
SETTLE_STEPS = 60      # sim steps after a reset, so the arm is at rest before the first action


# --------------------------------- the run ---------------------------------

def resolve_checkpoint(path):
    """(run_dir, adapter_dir) from a run directory or one of its checkpoints.

    A run directory holds config.json and one checkpoint per save; "final" is preferred, otherwise
    the highest step. A checkpoint directory is accepted directly so an earlier save can be tried.
    """
    p = Path(path).expanduser().resolve()
    if (p / "adapter").is_dir():
        return p.parent, p / "adapter"
    if not (p / "config.json").exists():
        sys.exit(f"{p} is not a training run: no config.json. Pass train/runs/<run> or one of its checkpoints.")
    if (p / "final" / "adapter").is_dir():
        return p, p / "final" / "adapter"
    steps = sorted(p.glob("step_*/adapter"))
    if not steps:
        sys.exit(f"{p} has no saved checkpoint yet (no final/ or step_*/). Let training save one first.")
    return p, steps[-1]


def load_run(path):
    """Everything the GUI and the policy need, read from the run and the session behind it."""
    run_dir, adapter = resolve_checkpoint(path)
    run = json.loads((run_dir / "config.json").read_text())

    # The base checkpoint is also recorded as an absolute path. When it is not there, the same
    # checkpoint under this machine's train/checkpoints/ (where train/download.py puts it) stands in.
    checkpoint = Path(run["config"]["model"]["checkpoint"])
    if not checkpoint.exists():
        local = TRAIN_DIR / "checkpoints" / checkpoint.name
        if not local.exists():
            sys.exit(f"base checkpoint not found at {checkpoint} or {local}.\nRun train/download.py first.")
        run["config"]["model"]["checkpoint"] = str(local)

    stats_file = adapter / "dataset_statistics.json"
    if not stats_file.exists():
        stats_file = run_dir / "dataset_statistics.json"
    stats = json.loads(stats_file.read_text())
    # The run file nests the statistics under the dataset key; a checkpoint saves them bare.
    if run["norm_stats_key"] in stats:
        stats = stats[run["norm_stats_key"]]["action"]

    # The run records the session it trained on as an absolute path, which does not survive being
    # copied to another machine. A session_config.json beside the run or the checkpoint stands in
    # for it: the scene, the task phrasing and the held-out pair are all inference needs from the
    # recording, so the videos do not have to come along.
    session = Path(run["session"])
    # The session path is the recording machine's; its last component is the session's name, which
    # train/download.py keeps, so datasets/<name>/config.json on this machine is the same file.
    candidates = [session / "config.json", run_dir / "session_config.json", adapter / "session_config.json",
                  ROOT / "datasets" / session.name / "config.json"]
    session_cfg_path = next((c for c in candidates if c.exists()), None)
    if session_cfg_path is None:
        sys.exit(f"cannot find the recording config this run trained on.\nLooked for: "
                 + "\n           ".join(str(c) for c in candidates)
                 + f"\nCopy the session's config.json to {run_dir / 'session_config.json'}.")
    session_cfg = json.loads(session_cfg_path.read_text())
    return {"run_dir": run_dir, "adapter": adapter, "run": run, "stats": stats,
            "session": session, "session_cfg": session_cfg}


def task_buttons(session_cfg):
    """[(instruction, held_out)] for every ordered pair of distinct objects, in a stable order."""
    from config import instruction

    names = list(session_cfg["scene"]["objects"])
    holdout = {tuple(p) for p in session_cfg["task"].get("holdout_pairs", [])}
    template = session_cfg["task"]["instruction"]
    out = []
    for source in names:
        for destination in names:
            if source == destination:
                continue
            task = {"source": source, "destination": destination, "instruction": template}
            out.append((instruction(task), (source, destination) in holdout))
    return out


# --------------------------------- the policy ---------------------------------

class Policy:
    """The fine-tuned model: an image and an instruction in, one action vector out."""

    def __init__(self, run, adapter, stats):
        import torch
        from peft import PeftModel
        from model import load_base, load_processor, resolve_device, resolve_dtype

        self.torch = torch
        model_cfg = dict(run["config"]["model"])
        model_cfg["gradient_checkpointing"] = False    # inference only
        self.device = resolve_device(model_cfg["device"])
        self.dtype = resolve_dtype(model_cfg["dtype"], self.device)

        # Loaded with the CPU as torch's default device: after gs.init(backend=gs.cuda) the default
        # is cuda:0, and the processor calls .numpy() on tensors it builds while loading.
        with torch.device("cpu"):
            self.processor = load_processor(str(adapter))
        self.tokenizer = self.processor.tokenizer
        base = load_base(model_cfg, self.device, self.dtype)
        self.model = PeftModel.from_pretrained(base, str(adapter)).eval()

        # What the vision tower actually wants, which is neither the configured compute dtype nor
        # simply the first parameter's: 4-bit weights are stored as uint8, and on the way to
        # quantizing, prepare_model_for_kbit_training upcasts every remaining float parameter to
        # float32. So the image has to follow the first genuinely floating-point parameter.
        self.input_dtype = next((p.dtype for p in self.model.parameters() if p.is_floating_point()),
                                self.dtype)

        self.stats = stats
        self.names = list(run["action_names"])
        self.action_tokenizer = ActionTokenizer(self.tokenizer.vocab_size, bins=256)
        self.image_key = run["config"]["observation"]["image"]
        self.camera = self.image_key.split("observation.images.", 1)[-1]
        self.stride = int(run["config"]["action"]["stride"])

    def predict(self, image, instruction):
        """(D,) action for one RGB frame (H, W, 3) uint8 and an instruction."""
        from PIL import Image
        from data import prompt_ids

        torch = self.torch
        # Preprocess on the CPU explicitly: once Genesis runs on CUDA, torch's default device is
        # cuda:0, and the processor's numpy-based steps cannot take a CUDA tensor.
        with torch.device("cpu"):
            pixel_values = self.processor.image_processor(
                Image.fromarray(image), return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, self.input_dtype)
        input_ids = torch.tensor([prompt_ids(self.tokenizer, instruction)], device=self.device)

        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids, pixel_values=pixel_values,
                max_new_tokens=len(self.names), do_sample=False,
            )
        tokens = generated[0, -len(self.names):].cpu().numpy()
        normalized = self.action_tokenizer.decode_token_ids_to_actions(tokens)
        return np.asarray(unnormalize_actions(normalized, self.stats), dtype=float)


# --------------------------------- the simulator ---------------------------------

class Simulator:
    """One environment of the recorded scene, driven a few sim steps at a time by tool-pose deltas.

    The policy owns the GPU, so this runs on the CPU rasterizer by default: the batch renderer
    reserves several GB up front even for a single environment, which does not fit beside the model
    on an 11 GB card.
    """

    def __init__(self, session_cfg, backend="cpu"):
        import genesis as gs
        from config import CONFIG, env_config
        from stacking_scene import build_scene, init_genesis, reset_envs, robot_dofs
        from utils.frames import table_top_z
        from utils.ik import move_tcp, tcp_pos_from_link
        from genesis.utils.misc import tensor_to_array

        self._env_config, self._reset_envs = env_config, reset_envs
        self._move_tcp, self._tcp_pos_from_link = move_tcp, tcp_pos_from_link
        self._to_array = tensor_to_array

        # Madrona reserves a 4 GB device-side malloc heap by default (gs_madrona src/mw/cuda_exec.cpp,
        # setCudaHeapSize), which is most of what the batch renderer costs here. Measured on this
        # scene, one env, three 640x480 cameras: 5.2 GB at the default, 1.4 GB at 256 MB, and the
        # rendered frames are pixel-identical. That is what lets the renderer share a 12 GB card
        # with the policy. A value already set in the shell wins.
        os.environ.setdefault("MADRONA_MWGPU_DEVICE_HEAP_SIZE", str(256 * 1024 * 1024))

        cfg = copy.deepcopy(CONFIG)
        cfg["sim"]["backend"] = backend
        cfg["ik"]["n_envs"] = 1
        cfg["viewer"]["draw_camera_frustums"] = False
        self.cfg = cfg
        self.warn = scene_mismatch(cfg, session_cfg)

        init_genesis(cfg)
        self.scene, self.robot, _, self.objects, self.cameras = build_scene(cfg, show_viewer=False)
        self.scene.build(n_envs=1, env_spacing=cfg["ik"]["env_spacing"])
        self.dofs = robot_dofs(self.robot, cfg)
        self.n_arm = len(cfg["control"]["arm_joints"])
        self.arm_dofs = self.dofs[:self.n_arm]
        self.grip_dofs = self.dofs[self.n_arm:]
        self.ee_link = self.robot.get_link(cfg["ik"]["ee_link"])
        self.joint_names = list(cfg["control"]["arm_joints"]) + list(cfg["control"]["gripper_joints"])
        self.table_offset = np.array([*cfg["scene"]["table"]["pos"][:2], table_top_z(cfg)], dtype=float)
        self.grip_open = np.asarray(cfg["control"]["gripper_open_qpos"], dtype=float)
        self.grip_closed = np.asarray(cfg["control"]["gripper_closed_qpos"], dtype=float)
        self.episode = 0

    # -- state --

    def reset(self):
        """A fresh random layout, the arm at home, and the commanded pose re-seeded from the arm."""
        self.episode += 1
        env_cfg = self._env_config(self.cfg, self.episode, 0)
        self._reset_envs(self.robot, self.objects, [env_cfg])
        for _ in range(SETTLE_STEPS):
            self.scene.step()
        # The recorded action is a delta on the *commanded* pose, so the command is tracked here the
        # way PlanRunner tracked it, starting from where the arm actually is.
        self.cmd_pos = np.asarray(self._tcp_pos_from_link(self.ee_link, self.cfg["ik"]["tcp_offset"]))[0]
        self.cmd_quat = self._to_array(self.ee_link.get_quat())[0].astype(float)
        self.cmd_grip = float(self.gripper_opening())
        return env_cfg

    def gripper_opening(self):
        q = self._to_array(self.robot.get_dofs_position(self.dofs))[0][self.n_arm:]
        span = self.grip_open - self.grip_closed
        return float(np.clip(np.mean((q - self.grip_closed) / np.where(span == 0, 1, span)), 0.0, 1.0))

    def observation(self):
        """Joint positions, tool pose in table coords, gripper opening, and every camera frame."""
        q = self._to_array(self.robot.get_dofs_position(self.dofs))[0]
        tcp = np.asarray(self._tcp_pos_from_link(self.ee_link, self.cfg["ik"]["tcp_offset"]))[0]
        images = {name: self._to_array(cam.read().rgb)[0] for name, cam in self.cameras.items()}
        return {"joints": q.astype(float), "tcp": tcp - self.table_offset,
                "gripper": self.gripper_opening(), "images": images}

    # -- control --

    def apply(self, action, sim_steps):
        """Integrate one predicted action onto the commanded pose and run the sim forward."""
        from scipy.spatial.transform import Rotation

        self.cmd_pos = self.cmd_pos + np.asarray(action[:3], dtype=float)
        # build_action_table produced the rotation as (r_now.inv() * r_ahead).as_rotvec(), so the
        # delta is applied on the right of the current command.
        now = Rotation.from_quat(self.cmd_quat[[1, 2, 3, 0]])
        turned = now * Rotation.from_rotvec(np.asarray(action[3:6], dtype=float))
        self.cmd_quat = turned.as_quat()[[3, 0, 1, 2]]
        self.cmd_grip = float(np.clip(action[6], 0.0, 1.0))

        self._move_tcp(self.robot, self.ee_link, self.arm_dofs,
                       self.cmd_pos[None, :], self.cmd_quat[None, :], self.cfg["ik"])
        grip_q = self.grip_closed + self.cmd_grip * (self.grip_open - self.grip_closed)
        self.robot.control_dofs_position(np.tile(grip_q, (1, 1)), self.grip_dofs)
        for _ in range(sim_steps):
            self.scene.step()


def scene_mismatch(live, session_cfg):
    """A warning when source/config.py no longer matches the scene the policy was trained in."""
    differences = []
    for name, spec in session_cfg["scene"]["objects"].items():
        current = live["scene"]["objects"].get(name)
        if current is None:
            differences.append(f"object {name!r} is gone")
        elif current["shape"] != spec["shape"]:
            differences.append(f"object {name!r} is a {current['shape']}, was a {spec['shape']}")
    for cam, spec in session_cfg["cameras"].items():
        current = live["cameras"].get(cam)
        if current is None:
            differences.append(f"camera {cam!r} is gone")
        elif tuple(current["res"]) != tuple(spec["res"]):
            differences.append(f"camera {cam!r} is {tuple(current['res'])}, was {tuple(spec['res'])}")
    return differences


# --------------------------------- the rollout thread ---------------------------------

class Rollout:
    """Owns the simulator and the policy on one thread; the GUI only reads the state it publishes.

    Genesis is not safe to drive from several threads and a prediction takes long enough to freeze
    an interactive window, so everything happens here and the figure redraws from a snapshot.
    """

    def __init__(self, sim, policy, fps):
        self.sim, self.policy = sim, policy
        self.sim_steps = max(1, int(round(policy.stride * (1.0 / sim.cfg["sim"]["dt"]) / fps)))
        self.hz = (1.0 / sim.cfg["sim"]["dt"]) / self.sim_steps
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self._request = None          # ("run", instruction) | ("reset", None)
        self.instruction = None
        self.running = False
        self.status = "ready"
        self.step = 0
        self.images = {}
        self.joints = deque(maxlen=HISTORY)
        self.tcp = deque(maxlen=HISTORY)
        self.grip = deque(maxlen=HISTORY)
        self.actions = deque(maxlen=HISTORY)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    # -- called from the GUI thread --

    def run_instruction(self, text):
        with self.lock:
            self._request = ("run", text)

    def reset(self):
        with self.lock:
            self._request = ("reset", None)

    def pause(self):
        with self.lock:
            self.running = False
            self.status = "paused"

    def snapshot(self):
        with self.lock:
            return {
                "images": dict(self.images), "status": self.status, "step": self.step,
                "instruction": self.instruction, "running": self.running,
                "joints": np.array(self.joints) if self.joints else np.zeros((0, 8)),
                "tcp": np.array(self.tcp) if self.tcp else np.zeros((0, 3)),
                "grip": np.array(self.grip) if self.grip else np.zeros((0,)),
                "actions": np.array(self.actions) if self.actions else np.zeros((0, len(self.policy.names))),
            }

    # -- the thread --

    def start(self):
        self.thread.start()

    def _publish(self, obs, action=None):
        with self.lock:
            self.images = obs["images"]
            self.joints.append(obs["joints"])
            self.tcp.append(obs["tcp"])
            self.grip.append(obs["gripper"])
            if action is not None:
                self.actions.append(action)

    def _do_reset(self):
        self.sim.reset()
        with self.lock:
            self.joints.clear(); self.tcp.clear(); self.grip.clear(); self.actions.clear()
            self.step = 0
        self._publish(self.sim.observation())

    def _loop(self):
        self._do_reset()
        with self.lock:
            self.status = "ready - type an instruction or pick a task"
        while not self.stop.is_set():
            with self.lock:
                request, self._request = self._request, None
            if request is not None:
                kind, value = request
                self._do_reset()
                with self.lock:
                    if kind == "run":
                        self.instruction, self.running = value, True
                        self.status = "running"
                    else:
                        # A new layout stops the policy and keeps the instruction, so the same
                        # command can be retried on fresh object positions with one click.
                        self.running = False
                        self.status = f"new layout {self.sim.episode} - press Run or pick a task"
                continue

            with self.lock:
                running, instruction = self.running, self.instruction
            if not running or not instruction:
                self.stop.wait(0.05)
                continue

            obs = self.sim.observation()
            try:
                action = self.policy.predict(obs["images"][self.policy.camera], instruction)
            except Exception as exc:                      # a failed prediction should not kill the GUI
                with self.lock:
                    self.running, self.status = False, f"prediction failed: {exc}"
                continue
            self.sim.apply(action, self.sim_steps)
            with self.lock:
                self.step += 1
            self._publish(self.sim.observation(), action)


# --------------------------------- the window ---------------------------------

class Dashboard:
    """Cameras, traces and the controls, redrawn from the rollout's snapshot."""

    def __init__(self, rollout, tasks, camera_order, warnings):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

        self.plt, self.rollout, self.tasks = plt, rollout, tasks
        self.model_cam = rollout.policy.camera
        self.cameras = [c for c in camera_order if c in rollout.sim.cameras]

        self.fig = plt.figure(figsize=(16.5, 9.5))
        try:
            self.fig.canvas.manager.set_window_title("OpenVLA rollout")
        except AttributeError:      # headless backends have no window manager
            pass
        grid = self.fig.add_gridspec(4, 6, left=0.04, right=0.98, top=0.94, bottom=0.26,
                                     hspace=0.55, wspace=0.45)

        # Only the camera the policy reads is shown: what the window displays is what the model sees.
        self.image_axes = {}
        self.image_artists = {}
        ax = self.fig.add_subplot(grid[0:2, 0:3])
        self.image_axes[self.model_cam] = ax
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{self.model_cam}  <- policy input", fontsize=10, color="tab:blue")
        for side in ax.spines.values():
            side.set_color("tab:blue")
            side.set_linewidth(2.0)
        self.image_artists[self.model_cam] = ax.imshow(np.zeros((2, 2, 3), np.uint8), interpolation="nearest")

        self.ax_status = self.fig.add_subplot(grid[2:4, 0:3])
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9)

        self.ax_joints = self.fig.add_subplot(grid[0, 3:6])
        self.ax_tcp = self.fig.add_subplot(grid[1, 3:6])
        self.ax_act = self.fig.add_subplot(grid[2, 3:6])
        self.ax_grip = self.fig.add_subplot(grid[3, 3:6])
        for ax, title in ((self.ax_joints, "Joint positions (rad)"),
                          (self.ax_tcp, "Tool position, table coords (m)"),
                          (self.ax_act, "Predicted action: translation (m) and rotation (rad)"),
                          (self.ax_grip, "Gripper: commanded vs observed")):
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

        names = rollout.policy.names
        # The gripper dimension is plotted against the observed opening rather than with the
        # deltas, which are on a completely different scale.
        self.grip_dim = names.index("gripper") if "gripper" in names else len(names) - 1
        delta_names = [n for i, n in enumerate(names) if i != self.grip_dim]
        self.joint_lines = [self.ax_joints.plot([], [], lw=1, label=n)[0] for n in rollout.sim.joint_names]
        self.tcp_lines = [self.ax_tcp.plot([], [], lw=1, label=n)[0] for n in ("x", "y", "z")]
        self.act_lines = [self.ax_act.plot([], [], lw=1, label=n)[0] for n in delta_names]
        self.grip_lines = [self.ax_grip.plot([], [], lw=1, label=n)[0] for n in ("commanded", "observed")]
        for ax in (self.ax_joints, self.ax_tcp, self.ax_act, self.ax_grip):
            ax.legend(fontsize=6, ncol=4, loc="upper left")

        self._build_controls(Button, TextBox, warnings)

    def _build_controls(self, Button, TextBox, warnings):
        # Six task buttons on one row, the held-out pair marked, then the free-text box.
        self.buttons = []
        n = len(self.tasks)
        width, gap, left = 0.145, 0.012, 0.04
        for i, (text, held_out) in enumerate(self.tasks):
            ax = self.fig.add_axes([left + i * (width + gap), 0.135, width, 0.055])
            label = text.replace("put the ", "").replace(" on the ", " -> ")
            button = Button(ax, ("X  " if held_out else "") + label, hovercolor="0.85")
            button.label.set_fontsize(7.5)
            if held_out:
                button.label.set_color("tab:red")
                for side in ax.spines.values():
                    side.set_color("tab:red")
            button.on_clicked(lambda _event, t=text: self._send(t))
            self.buttons.append(button)
        self.fig.text(left, 0.202, "Tasks   (X = held out of training, so this one is the out-of-distribution test)",
                      fontsize=8, color="0.3")

        box_ax = self.fig.add_axes([left + 0.075, 0.055, 0.60, 0.05])
        self.text_box = TextBox(box_ax, "instruction  ", initial="")
        self.text_box.label.set_fontsize(9)
        self.text_box.on_submit(self._send)

        run_ax = self.fig.add_axes([left + 0.69, 0.055, 0.08, 0.05])
        self.run_button = Button(run_ax, "Run", hovercolor="0.85")
        self.run_button.on_clicked(lambda _e: self._send(self.text_box.text))

        pause_ax = self.fig.add_axes([left + 0.78, 0.055, 0.08, 0.05])
        self.pause_button = Button(pause_ax, "Pause", hovercolor="0.85")
        self.pause_button.on_clicked(lambda _e: self.rollout.pause())

        reset_ax = self.fig.add_axes([left + 0.87, 0.055, 0.10, 0.05])
        self.reset_button = Button(reset_ax, "New layout", hovercolor="0.85")
        self.reset_button.label.set_fontsize(8.5)
        self.reset_button.on_clicked(lambda _e: self.rollout.reset())

        if warnings:
            self.fig.text(left, 0.012, "source/config.py no longer matches the recorded scene: "
                                       + "; ".join(warnings), fontsize=8, color="tab:red")

    def _send(self, text):
        text = (text or "").strip()
        if not text:
            return
        if self.text_box.text != text:
            self.text_box.set_val(text)
        self.rollout.run_instruction(text)

    def _update(self, _frame):
        state = self.rollout.snapshot()
        for cam, artist in self.image_artists.items():
            image = state["images"].get(cam)
            if image is not None:
                artist.set_data(image)
                artist.set_extent((0, image.shape[1], image.shape[0], 0))

        def redraw(ax, lines, data):
            if len(data) == 0:
                return
            x = np.arange(len(data))
            for i, line in enumerate(lines[:data.shape[1]]):
                line.set_data(x, data[:, i])
            ax.relim(); ax.autoscale_view()
            ax.set_xlim(0, max(len(data), 10))

        actions = state["actions"]
        redraw(self.ax_joints, self.joint_lines, state["joints"])
        redraw(self.ax_tcp, self.tcp_lines, state["tcp"])
        redraw(self.ax_act, self.act_lines, actions[:, :len(self.act_lines)])
        if len(actions):
            self.grip_lines[0].set_data(np.arange(len(actions)), actions[:, self.grip_dim])
        if len(state["grip"]):
            self.grip_lines[1].set_data(np.arange(len(state["grip"])), state["grip"])
        self.ax_grip.relim(); self.ax_grip.autoscale_view()
        self.ax_grip.set_xlim(0, max(len(state["grip"]), 10))

        self.status_text.set_text(self._status(state))
        return []

    def _status(self, state):
        r = self.rollout
        action = state["actions"][-1] if len(state["actions"]) else None
        lines = [
            f"status       {state['status']}",
            f"instruction  {state['instruction'] or '-'}",
            f"step         {state['step']}   control {r.hz:.1f} Hz  ({r.sim_steps} sim steps per action)",
            f"policy sees  {self.model_cam}   stride {r.policy.stride}",
            "",
            "last action",
        ]
        if action is not None:
            for name, value in zip(r.policy.names, action):
                lines.append(f"   {name:<9s} {value:+.4f}")
        else:
            lines.append("   (none yet)")
        return "\n".join(lines)

    def show(self):
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self._update, interval=200,
                                  blit=False, cache_frame_data=False)
        self.plt.show()


# --------------------------------- entry point ---------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a fine-tuned OpenVLA policy in the simulator.")
    parser.add_argument("run", help="a training run directory (train/runs/<run>) or one of its checkpoints")
    parser.add_argument("--backend", default=None,
                        help="Genesis backend for the simulator. The default is whatever rendered "
                             "the training data, because the policy is sensitive to which renderer "
                             "produced its input; see the note in main().")
    args = parser.parse_args()

    bundle = load_run(args.run)
    run, session_cfg = bundle["run"], bundle["session_cfg"]
    print(f"run       {bundle['run_dir']}")
    print(f"checkpoint{'':1s} {bundle['adapter']}")
    print(f"session   {bundle['session']}  ({run['dataset']['episodes']} episodes, {run['dataset']['frames']} frames)")
    print(f"policy    sees {run['config']['observation']['image']}, predicts {run['action_names']}")

    # Render with whatever rendered the training data. The CUDA batch renderer and the CPU
    # rasterizer do not produce the same image, and the policy is sensitive to the difference: on
    # one measured frame they differ by 12.6/255 on average, and the actions predicted from them
    # disagree completely (dz -0.0033 against +0.0082, drot_y -0.0005 against -0.1224), which is
    # enough to send a rollout somewhere the demonstrations never went. The cost is that the batch
    # renderer wants about 5.7 GB beside the policy's 5.7 GB, so an 11 GB card cannot hold both.
    recorded_on = (json.loads((bundle["session"] / "metadata.json").read_text())
                   .get("hardware", {}).get("genesis_backend")
                   if (bundle["session"] / "metadata.json").exists() else None)
    # Without a metadata.json to say what recorded the data, cuda is the default: measured on this
    # checkpoint, the CPU rasterizer's frame of the same layout roughly doubles every predicted
    # rotation and the y translation against the batch renderer's frame.
    backend = args.backend or recorded_on or "cuda"
    if args.backend and recorded_on and args.backend != recorded_on:
        print(f"  warning: rendering with {args.backend} but the data was recorded with "
              f"{recorded_on}; the policy sees a different image than it was trained on")
    print(f"renderer  {backend}" + (f"  (data recorded with {recorded_on})" if recorded_on else ""))
    # The policy is built before the simulator on purpose: gs.init(backend=gs.cuda) switches
    # torch's default device to cuda:0 (measured: torch.get_default_device() reads cuda:0 after
    # it), and the checkpoint's image processor calls .numpy() on tensors it creates while loading,
    # which fails for a CUDA tensor. Loading the policy first keeps that code on the CPU.
    policy = Policy(run, bundle["adapter"], bundle["stats"])
    sim = Simulator(session_cfg, backend=backend)
    for line in sim.warn:
        print(f"  warning: {line}")

    rollout = Rollout(sim, policy, fps=session_cfg["record"]["fps"])
    rollout.start()
    dashboard = Dashboard(rollout, task_buttons(session_cfg),
                          camera_order=list(session_cfg["cameras"]), warnings=sim.warn)
    try:
        dashboard.show()
    finally:
        # Let the rollout finish the sim step it is in. Tearing the process down underneath Genesis
        # while it is stepping aborts in its C++ destructors instead of exiting cleanly.
        rollout.stop.set()
        rollout.thread.join(timeout=10)


if __name__ == "__main__":
    main()
