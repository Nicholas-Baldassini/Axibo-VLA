"""
Training samples for SmolVLA straight out of a recorded LeRobot session, no format conversion.

The session on disk is not touched. What SmolVLA needs that OpenVLA does not, and why it is done
here rather than in the dataset:

  Camera names.  The pretrained checkpoint (lerobot/smolvla_base) names its three image inputs
      observation.images.camera1/2/3, and lerobot refuses a dataset whose camera names differ. The
      session records observation.images.front/wrist/overhead, so each frame's cameras are handed
      over under the pretrained names, per config["observation"]["cameras"]. OpenVLA never saw this
      problem: it takes one image, composed by observe.py from whichever cameras the config lists.

  The label column.  lerobot's policies read their target from the one key literally named
      "action", and the normalizer looks its statistics up under that name. The OpenVLA runs build
      their label from action.clean_rate (the plan's undisturbed pursuit rate, a per-step delta) and
      action.gripper_target; SmolVLA cannot be pointed at those from its own config. The column is
      chosen in config["action"]["key"] and handed over as "action". The default is the recorded
      "action" column itself: the absolute tool pose and gripper the arm was commanded, which is the
      natural target for a policy that predicts a 1.67 s chunk of future poses, and the plain SmolVLA
      recipe. The disturbances in it (the engine's random pushes) are zero-mean, so regression
      averages them out; the clean-rate label stays an option for a later run.

  Action chunks.  SmolVLA regresses the next chunk_size actions at once, so each sample carries the
      label of the next chunk_size frames (lerobot's delta_timestamps), padded past the episode end
      with the last frame and flagged in action_is_pad. OpenVLA reads one action per frame.

  Proprioception.  SmolVLA conditions on a state vector; OpenVLA has none. The 8 joint positions
      (observation.state) go in unchanged.

Images are the dataset's own float tensors in [0, 1], (3, 480, 640); SmolVLA letterboxes them to
512x512 itself. The instruction goes in verbatim as "task", as the OpenVLA samples do.
"""

import json

import numpy as np
from torch.utils.data import Dataset

from data import find_session


class SmolVLASamples(Dataset):
    """Frames of one recorded session as (cameras, state, instruction, action chunk) samples."""

    def __init__(self, config, datasets_root):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.processor.rename_processor import rename_stats

        data_cfg, obs_cfg, act_cfg = config["data"], config["observation"], config["action"]
        self.session = find_session(data_cfg["session"], datasets_root)
        meta_path = self.session / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.repo_id = self.metadata.get("repo_id", "local/dataset")

        self.cameras = dict(obs_cfg["cameras"])          # dataset name -> name the policy expects
        self.state_key, self.action_key = obs_cfg["state"], act_cfg["key"]
        self.chunk_size = int(act_cfg["chunk_size"])

        meta = LeRobotDataset(self.repo_id, root=self.session / "dataset", episodes=data_cfg["episodes"],
                              video_backend=data_cfg["video_backend"]).meta
        for key in self.cameras:
            if key not in meta.video_keys:
                raise SystemExit(f"observation.cameras {key!r} is not a camera of this dataset: {meta.video_keys}")
        for key in (self.state_key, self.action_key):
            if key not in meta.features:
                raise SystemExit(f"{key!r} is not a column of this dataset: {list(meta.features)}")
        self.fps = meta.fps
        # The label of frame t is the action at frames t .. t + chunk_size - 1.
        delta_timestamps = {self.action_key: [i / self.fps for i in range(self.chunk_size)]}
        self.dataset = LeRobotDataset(
            self.repo_id, root=self.session / "dataset", episodes=data_cfg["episodes"],
            delta_timestamps=delta_timestamps, video_backend=data_cfg["video_backend"],
        )
        self.n_episodes = self.dataset.num_episodes

        # What the policy sees, under the names it sees them by, and the statistics that normalize them.
        self.input_features = {
            **{new: PolicyFeature(type=FeatureType.VISUAL, shape=self._chw(meta.features[old]["shape"]))
               for old, new in self.cameras.items()},
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=tuple(meta.features[self.state_key]["shape"])),
        }
        self.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(meta.features[self.action_key]["shape"])),
        }
        rename = {**self.cameras, self.state_key: "observation.state", self.action_key: "action"}
        self.stats = {k: v for k, v in rename_stats(meta.stats, rename).items()
                      if k in self.input_features or k in self.output_features}
        self.action_names = list(meta.features[self.action_key].get("names") or
                                 [f"{self.action_key}[{i}]" for i in range(self.output_features["action"].shape[0])])

    @staticmethod
    def _chw(shape):
        h, w, c = shape
        return (c, h, w)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        sample = {new: item[old] for old, new in self.cameras.items()}
        sample["observation.state"] = item[self.state_key]
        sample["action"] = item[self.action_key]
        sample["action_is_pad"] = item[f"{self.action_key}_is_pad"]
        sample["task"] = item["task"]
        return sample

    def describe_sample(self, i=0):
        """A readable dump of one sample: instruction, cameras, state, the first and last action of the chunk."""
        sample = self[i]
        item = self.dataset[i]
        chunk = sample["action"].numpy()
        return "\n".join([
            f"sample {i}: \"{sample['task']}\"  (episode {int(item['episode_index'])}, frame {int(item['frame_index'])})",
            "  cameras    " + "  ".join(f"{new} {tuple(sample[new].shape)}" for new in self.cameras.values()),
            "  state      " + "  ".join(f"{v:+.3f}" for v in sample["observation.state"].numpy()),
            f"  action chunk {chunk.shape}, {int(sample['action_is_pad'].sum())} padded frames",
            "    first    " + "  ".join(f"{n} {v:+.4f}" for n, v in zip(self.action_names, chunk[0])),
            "    last     " + "  ".join(f"{n} {v:+.4f}" for n, v in zip(self.action_names, chunk[-1])),
        ])

    def stats_json(self):
        """The normalization statistics as plain lists, for dataset_statistics.json."""
        return {k: {s: np.asarray(v).tolist() for s, v in d.items()} for k, d in self.stats.items()}
