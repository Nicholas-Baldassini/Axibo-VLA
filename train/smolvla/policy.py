"""
A fine-tuned SmolVLA in the simulator, behind the same interface as infer.py's OpenVLA Policy, so
Rollout, the dashboard and evaluate.py drive either one through step(obs, instruction).

Inference is lerobot's: from the three cameras, the joint positions and the instruction the model
predicts a chunk of chunk_size future tool poses (x y z in table coordinates, quaternion wxyz,
gripper); the first n_action_steps of them are executed one per control step at the recording rate
(stride 1, 30 Hz) before the next prediction. SmolVLA's actions are absolute poses where OpenVLA's
are deltas, so each pose is turned into the delta from the pose the simulator currently commands and
handed to Simulator.apply unchanged: the workspace floor, the gripper threshold and the traces all
work the same, and the delta lands exactly on the predicted pose.
"""

from collections import deque

import numpy as np
import torch

import observe


class SmolVLA:
    names = ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"]   # what step() returns: Simulator.apply's action layout
    camera = "policy input"                                     # the dashboard's name for the composed view below
    stride = 1                                                  # one action per recorded frame

    def __init__(self, run, checkpoint, action_steps=None):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(self.device).eval()
        cfg = self.policy.config
        cfg.device = self.device
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}})
        self.chunk_size = cfg.chunk_size
        # Executing fewer of the chunk's actions before the next prediction closes the loop more often
        # (every n_action_steps frames) at the cost of one prediction per n_action_steps steps.
        self.n_action_steps = min(cfg.chunk_size, action_steps or cfg.n_action_steps)

        cameras = run["config"]["observation"]["cameras"]        # dataset camera key -> the key the policy expects
        self.keys = {old.split(".")[-1]: new for old, new in cameras.items()}   # simulator camera name -> policy key
        self.view = observe.spec({"image": list(cameras)})       # the cameras side by side, for display and videos
        self.cameras = observe.camera_names(self.view)
        self.queue = deque()

    def reset(self):
        """Drop the rest of the current chunk: the scene changed under it."""
        self.queue.clear()
        self.policy.reset()

    def compose(self, images):
        """The three camera frames side by side (uint8 H, W, 3): what the policy is given, for display."""
        return observe.compose(images, self.view)

    def step(self, obs, instruction):
        """The next action for Simulator.apply from one simulator observation; predicts a new chunk when the last is used up."""
        if not self.queue:
            self.queue.extend(self.predict_chunk(obs, instruction)[: self.n_action_steps])
        return self.delta(obs["cmd"], self.queue.popleft())

    def predict_chunk(self, obs, instruction):
        """(chunk_size, 8) absolute poses + gripper from the cameras, joints and instruction."""
        batch = {key: torch.from_numpy(np.ascontiguousarray(obs["images"][cam])).permute(2, 0, 1).float() / 255.0
                 for cam, key in self.keys.items()}
        batch["observation.state"] = torch.as_tensor(np.asarray(obs["joints"], dtype=np.float32))
        batch["task"] = instruction
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(self.pre(batch))     # (1, chunk_size, 8), normalized
            chunk = self.post(chunk[0])                                   # un-normalized, on the CPU
        return [np.asarray(a, dtype=float) for a in chunk.numpy()]

    @staticmethod
    def delta(cmd, pose):
        """Simulator.apply's action that moves the commanded pose `cmd` (table pos, quat wxyz) onto `pose`."""
        from scipy.spatial.transform import Rotation

        cmd_pos, cmd_quat = cmd
        now = Rotation.from_quat(np.asarray(cmd_quat, dtype=float)[[1, 2, 3, 0]])
        target = Rotation.from_quat(np.asarray(pose[3:7], dtype=float)[[1, 2, 3, 0]])
        return np.concatenate([np.asarray(pose[:3], dtype=float) - cmd_pos, (now.inv() * target).as_rotvec(), [pose[7]]])
