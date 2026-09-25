"""
Fine-tuning configuration: OpenVLA on a recorded session.

Everything a training run reads is in the single CONFIG dict at the bottom: which session and
episodes, which camera the model sees, how the action vector is built from the recorded columns,
the checkpoint, LoRA, the optimizer and schedule, saving, logging and rollout previews.

    python train/finetune.py --config openvla

Another file in this folder can import CONFIG and override parts of it (see smoke.py).
"""

from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parents[1]
ROOT = TRAIN_DIR.parent

SEED = 0

DATA = {
    # A recording session directory (datasets/piper_stack_<date>_<time>) or its dataset/ subdirectory.
    # None picks the most recently written session under datasets/.
    "session": None,
    "episodes": None,          # episode indices to train on, None = every episode in the session
    "video_backend": "pyav",   # frame decoder for the dataset videos ("torchcodec" needs an FFmpeg build)
    "num_workers": 4,          # DataLoader processes decoding frames in parallel; 0 decodes in the main process
}

# OpenVLA sees one RGB image and the instruction, nothing else: the image is resized to 224x224 by the
# checkpoint's processor, and the instruction is the dataset's per-frame task string. Proprioception is
# not an input of this architecture (OpenVLA-OFT adds it), so observation.state stays unused here.
OBSERVATION = {
    "image": "observation.images.front",
}

# The action vector the model predicts, one token per dimension. Components are concatenated in
# order. Each takes the "dims" columns of a recorded key at the current frame t and at t + stride
# (clamped to the episode's last frame) and produces:
#     "delta": None           the value at t + stride (an absolute command, e.g. the gripper opening)
#     "delta": "translation"  value[t + stride] - value[t]
#     "delta": "rotation"     the rotation from quat[t] to quat[t + stride] as a rotation vector (4 dims in, 3 out)
# "normalize" maps a dimension to [-1, 1] by its 1st..99th percentiles before binning (OpenVLA's
# convention); leave it off for a dimension that already lives in [-1, 1] such as the gripper.
# The default is OpenVLA's own 7-D layout: tool translation, tool rotation, gripper.
ACTION = {
    # OpenVLA was pretrained on 3-10 Hz data; a 30 Hz delta is too small to survive 256-bin
    # quantization, so the action reaches 3 frames ahead (10 Hz) and the policy will run at that rate.
    "stride": 3,
    "components": [
        {"key": "action", "dims": (0, 1, 2), "delta": "translation", "normalize": True},   # tool center xyz, table coords
        {"key": "action", "dims": (3, 4, 5, 6), "delta": "rotation", "normalize": True},  # tool quaternion (w, x, y, z)
        {"key": "action", "dims": (7,), "delta": None, "normalize": False},               # gripper opening 0..1
    ],
}

MODEL = {
    # Hugging Face id or local directory. python train/download.py fetches openvla/openvla-7b (15 GB) here.
    "checkpoint": str(TRAIN_DIR / "checkpoints" / "openvla-7b"),
    "device": "auto",                  # "cuda" | "mps" | "cpu" | "auto" (the first available, in that order)
    "dtype": "auto",                   # "bfloat16" | "float32" | "auto" (bfloat16 on cuda, float32 elsewhere)
    "load_in_4bit": False,             # QLoRA: 4-bit base weights through bitsandbytes, CUDA only. What fits a 16 GB GPU.
    "gradient_checkpointing": False,   # recompute activations in the backward pass to save memory
    # Random-init miniature of the architecture (same code path, tokenizer and processor, ~30M
    # parameters) to exercise the pipeline where the 7B model cannot run. It learns nothing useful.
    "tiny": False,
}

# LoRA on every linear layer (vision backbone, projector and language model), as in OpenVLA's own
# fine-tuning script; rank 32 with alpha 16 are its defaults.
LORA = {
    "rank": 32,
    "alpha": 16,
    "dropout": 0.0,
}

OPTIM = {
    "lr": 5e-4,                    # OpenVLA's LoRA default at batch 16; scale it down with the batch
    "weight_decay": 0.0,
    "batch_size": 16,
    "grad_accumulation_steps": 1,  # effective batch = batch_size * grad_accumulation_steps
    "max_steps": 5000,             # optimizer steps
}

SAVE = {
    "root": str(TRAIN_DIR / "runs"),   # one directory per run: <config>_<date>_<time>
    "every_steps": 500,                # LoRA adapter checkpoint every N optimizer steps, plus one at the end
    "merged": False,                   # also write the base model with the adapter merged in at the end
                                       # (a full copy of the checkpoint, 15 GB for the 7B; loads with from_pretrained)
}

LOG = {
    "every_steps": 10,   # print and append to train_log.jsonl every N optimizer steps
}

# Rollout preview videos: run the current policy in the simulator for a few episodes at every
# checkpoint and save the camera videos next to it. Needs the inference loop, which comes with the
# evaluation step; until then every_steps = 0 keeps it off.
PREVIEW = {
    "every_steps": 0,
    "episodes": 2,
    "cameras": ["front", "wrist"],
}

CONFIG = {
    "seed": SEED,
    "data": DATA,
    "observation": OBSERVATION,
    "action": ACTION,
    "model": MODEL,
    "lora": LORA,
    "optim": OPTIM,
    "save": SAVE,
    "log": LOG,
    "preview": PREVIEW,
}
