"""
Fine-tuning configuration: SmolVLA on a recorded session.

Same shape as openvla.py: one CONFIG dict, read by train/smolvla/finetune.py.

    python train/smolvla/finetune.py --config smolvla

The run reads the same session as the OpenVLA runs and changes nothing in it; train/smolvla/data.py
explains the renaming SmolVLA needs. Values here are SmolVLA's own fine-tuning defaults (lerobot
0.4.4 presets, the paper's recipe) except the batch size, which is what the 12 GB card holds.
"""

from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parents[1]

SEED = 0

DATA = {
    "session": None,           # a session directory under datasets/; None = the most recent one
    "episodes": None,          # a list of episode indices, or None for all
    "video_backend": "pyav",
    "num_workers": 8,
}

OBSERVATION = {
    # SmolVLA takes every camera separately (no composing, unlike OpenVLA's single canvas), but the
    # pretrained checkpoint names its three image inputs camera1..3; these are the session's cameras
    # in those slots. Top-down, wrist, extra is the order its pretraining data used.
    "cameras": {
        "observation.images.front": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
        "observation.images.overhead": "observation.images.camera3",
    },
    # Proprioception: SmolVLA reads one state vector (OpenVLA has none). The 8 joint positions.
    "state": "observation.state",
}

ACTION = {
    # The recorded column used as the label; SmolVLA regresses a chunk of future actions from it.
    # "action" is the executed tool-pose command: x y z (table coords, m), qw qx qy qz, gripper 0..1.
    # The OpenVLA runs train on action.clean_rate deltas instead; see data.py for why that is not
    # the default here.
    "key": "action",
    "chunk_size": 50,          # frames of future actions predicted per inference (1.67 s at 30 fps)
    "n_action_steps": 50,      # how many of them inference executes before predicting again
}

MODEL = {
    "checkpoint": "lerobot/smolvla_base",   # Hugging Face id or local directory (450M, pretrained on SO-100 data)
    "device": "auto",
    "use_amp": False,          # bf16 autocast; SmolVLA's reference training is fp32 and the card showed no speedup
}

# lerobot's SmolVLA presets: AdamW 1e-4, betas (0.9, 0.95), weight decay 1e-10, grad clip 10, cosine
# decay to 2.5e-6 with 1000 warmup steps over 30k steps (scaled down to max_steps when shorter).
OPTIM = {
    "lr": 1e-4,
    "betas": (0.9, 0.95),
    "eps": 1e-8,
    "weight_decay": 1e-10,
    "grad_clip": 10.0,
    "warmup_steps": 1000,
    "decay_steps": 30000,
    "decay_lr": 2.5e-6,
    # The blog's fine-tune is batch 64 for 20k steps. Batch 16 is what fits beside 12 GB in fp32
    # (7 GB, 0.84 s/step measured on the RTX 4070), so 20k steps is about 4.7 h and 2.5 passes
    # over the 129k frames; the OpenVLA run it is compared with saw 184k samples.
    "batch_size": 16,
    "grad_accumulation_steps": 1,
    "max_steps": 20000,
}

SAVE = {
    "root": str(TRAIN_DIR / "runs"),
    "every_steps": 2500,
}

LOG = {
    "every_steps": 20,
}

CONFIG = {
    "seed": SEED,
    "data": DATA,
    "observation": OBSERVATION,
    "action": ACTION,
    "model": MODEL,
    "optim": OPTIM,
    "save": SAVE,
    "log": LOG,
}
