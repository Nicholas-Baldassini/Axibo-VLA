"""
Training samples for OpenVLA straight out of a recorded LeRobot session, no format conversion.

Each sample is one frame: the configured camera image, that frame's instruction, and the action
vector built from the recorded columns (actions.py). The token sequence is OpenVLA's own:

    <s>In: What action should the robot take to {instruction}?\nOut: {action tokens}</s>

with the recorded instruction dropped into the template verbatim, e.g. "put the red cylinder on the
blue cube". The loss is on the action tokens and the end-of-sequence token only. Inference must
build the prompt with prompt_ids() below; the checkpoint's predict_action does the same thing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset

from actions import action_statistics, build_action_table, episode_ranges, normalize_actions
from model import load_processor

IGNORE_INDEX = -100


def find_session(path, datasets_root):
    """A session directory or its dataset/ subdirectory; None = the most recently written session."""
    if path is None:
        sessions = [p for p in Path(datasets_root).glob("*") if (p / "dataset" / "meta" / "info.json").exists()]
        if not sessions:
            sys.exit(f"no finished recording sessions under {datasets_root}; set data.session")
        return max(sessions, key=lambda p: p.stat().st_mtime)
    p = Path(path).expanduser().resolve()
    if p.name == "dataset" and (p / "meta" / "info.json").exists():
        return p.parent
    if (p / "dataset" / "meta" / "info.json").exists():
        return p
    sys.exit(f"{p} is not a finished recording session: no dataset/meta/info.json")


PROMPT = "In: What action should the robot take to {instruction}?\nOut:" # Taken straight from the openVLA Paper
EMPTY_TOKEN = 29871   # Llama's lone "▁": OpenVLA's training text tokenized to it between "Out:" and the action tokens


def prompt_ids(tokenizer, instruction):
    """Token ids of the prompt: <s>, the template with the instruction, ending in the empty token that
    preceded every action token during OpenVLA's pretraining (predict_action inserts the same one)."""
    ids = tokenizer(PROMPT.format(instruction=instruction), add_special_tokens=True).input_ids
    if ids[-1] != EMPTY_TOKEN:
        ids.append(EMPTY_TOKEN)
    return ids


def encode_sample(tokenizer, action_tokenizer, instruction, action_normalized):
    """input_ids and labels (lists) for one frame: the prompt, then one token per action dimension, then </s>."""
    prompt = prompt_ids(tokenizer, instruction)
    targets = action_tokenizer.token_ids(action_normalized).tolist() + [tokenizer.eos_token_id]
    return prompt + targets, [IGNORE_INDEX] * len(prompt) + targets


def collate(batch, pad_token_id):
    """Right-pad a list of samples into one batch; padding is masked out of attention and the loss."""
    length = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
    for k, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[k, :n] = b["input_ids"]
        labels[k, :n] = b["labels"]
        attention_mask[k, :n] = 1
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class OpenVLASamples(Dataset):
    """Frames of one recorded session as (image, instruction, action tokens) samples."""

    def __init__(self, config, processor, action_tokenizer, datasets_root):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        data_cfg = config["data"]
        self.session = find_session(data_cfg["session"], datasets_root)
        meta_path = self.session / "metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.repo_id = self.metadata.get("repo_id", "local/dataset")
        self.dataset = LeRobotDataset(
            self.repo_id, root=self.session / "dataset",
            episodes=data_cfg["episodes"], video_backend=data_cfg["video_backend"],
        )
        self.image_key = config["observation"]["image"]
        if self.image_key not in self.dataset.meta.video_keys:
            sys.exit(f"observation.image {self.image_key!r} is not a camera of this dataset: {self.dataset.meta.video_keys}")
        self.checkpoint = config["model"]["checkpoint"]
        self.processor, self.tokenizer, self.action_tokenizer = processor, processor.tokenizer, action_tokenizer

        # Actions for every frame, computed once from the tabular columns.
        hf = self.dataset.hf_dataset
        keys = {comp["key"] for comp in config["action"]["components"]}
        for key in keys:
            if key not in self.dataset.meta.features:
                sys.exit(f"action component key {key!r} is not in this dataset: {list(self.dataset.meta.features)}")
        columns = {k: np.asarray(hf[k], dtype=np.float32) for k in keys}
        names = {k: self._names(k, columns[k].shape[1]) for k in keys}
        ranges = episode_ranges(np.asarray(hf["episode_index"]))
        self.actions, self.action_names, mask = build_action_table(columns, names, ranges, config["action"])
        self.stats = action_statistics(self.actions, mask)
        self.normalized = normalize_actions(self.actions, self.stats)
        self.n_episodes, self.fps = len(ranges), self.dataset.meta.fps
        self.image_shape = tuple(self.dataset.meta.features[self.image_key]["shape"])

    def _names(self, key, width):
        names = self.dataset.meta.features[key].get("names")
        return list(names) if names else [f"{key}[{i}]" for i in range(width)]

    def __getstate__(self):
        # DataLoader workers get a copy without the processor: its dynamically loaded class does not
        # pickle, so each worker reloads it from the checkpoint on first use.
        state = self.__dict__.copy()
        state["processor"] = state["tokenizer"] = None
        return state

    def _ensure_processor(self):
        if self.processor is None:
            self.processor = load_processor(self.checkpoint)
            self.tokenizer = self.processor.tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        self._ensure_processor()
        item = self.dataset[i]
        image = TVF.to_pil_image(item[self.image_key])
        pixel_values = self.processor.image_processor(image, return_tensors="pt")["pixel_values"][0]
        input_ids, labels = encode_sample(self.tokenizer, self.action_tokenizer, item["task"], self.normalized[i])
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def describe_sample(self, i=0):
        """A readable dump of one sample: instruction, action, normalized action, token ids."""
        item = self.dataset[i]
        input_ids, _ = encode_sample(self.tokenizer, self.action_tokenizer, item["task"], self.normalized[i])
        n = len(self.action_names)
        return "\n".join([
            f"sample {i}: \"{item['task']}\"  (episode {int(item['episode_index'])}, frame {int(item['frame_index'])})",
            "  action     " + "  ".join(f"{name} {v:+.4f}" for name, v in zip(self.action_names, self.actions[i])),
            "  normalized " + "  ".join(f"{v:+.3f}" for v in self.normalized[i]),
            f"  tokens: prompt {input_ids[:-(n + 1)]} = {self.tokenizer.decode(input_ids[:-(n + 1)])!r}",
            f"          action {input_ids[-(n + 1):-1]}  eos {input_ids[-1]}   (loss on these {n + 1})",
        ])
