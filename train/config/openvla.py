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
    # Frames to keep from each episode, applied as episode[start:stop]; (None, None) keeps all.
    # The first session (piper_stack_2026-09-26_00-01-05) stored the plan's opening home and idle
    # actions, an unchanging image paired with a zero action that teaches a history-free policy to
    # stand still, and needed (27, None) here to cut them. The recorder now leaves those frames out
    # at the source (utils/recorder.py, first_move), so sessions recorded after 2026-09-27 keep
    # everything. Set (27, None) again only to train on the first session.
    "frames": (None, None),
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
    # What the 256 bins span per normalized dimension. OpenVLA's "q99" maps the 1st..99th
    # percentiles and clips the rest, which on this data caps the demonstrations' fast sideways
    # moves: the plan covers its approach in about five control steps, so the per-step dy reaches
    # 0.064 m against a q99 of 0.037. Measured over all 670 episodes, that clipping loses a mean of
    # 1.3 cm and up to 6 cm of approach in y, and the first policy landed short and stalled.
    # "minmax" spans the full range; the data is a synthetic plan with no outliers to guard against,
    # and a bin is still 0.4 mm of dy.
    "bounds": "minmax",
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
    # "bfloat16" | "float16" | "float32" | "auto" (bfloat16 on cuda, float32 elsewhere).
    # float16 on a pre-Ampere card. bfloat16 needs sm_80 for native tensor cores; below that torch
    # still accepts it but bitsandbytes has no fused 4-bit kernel for it and falls back to
    # dequantizing each weight before the matmul, which is both slow and memory-hungry (measured on
    # an sm_75 RTX 2080 Ti: 5.22 ms per 4096x4096 4-bit matmul in bfloat16 against 0.71 ms in
    # float16). The LoRA parameters stay float32, so no gradient scaler is needed.
    "dtype": "float16",
    # QLoRA: 4-bit base weights through bitsandbytes, CUDA only. Required below about 16 GB of VRAM:
    # the 7B checkpoint is ~15 GB in bfloat16, so an 11 GB card runs out partway through .to(device).
    "load_in_4bit": True,
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
    "dropout": 0.01,
}

OPTIM = {
    # OpenVLA's LoRA default is 5e-4 at batch 16; this is that scaled down to the effective batch
    # of 4 below. The first run left it at 5e-4 and the loss reached 0.70 before spiking at step
    # 2320, degrading to 1.5, and going nan at 4220.
    "lr": 1e-4,
    # Maximum gradient norm per optimizer step. A step whose gradient is not finite is dropped
    # rather than applied, since one of those turns the whole adapter to nan permanently.
    "grad_clip": 1.0,
    "weight_decay": 0.0,
    # Effective batch = batch_size * grad_accumulation_steps, and only the effective batch matters
    # for the gradient, so trade one for the other to fit the GPU. 4-bit weights and LoRA rest at
    # 5.7 GB, which leaves about 4 GB for activations on an 11 GB card: batch_size 4 runs out in the
    # LoRA forward, batch_size 1 fits with room to spare. Raise batch_size and drop the
    # accumulation on a larger card.
    # Measured on a 12 GB RTX 4070 at the same effective batch of 4: batch_size 1 runs 0.41 step/s
    # at 9.6 GB peak, batch_size 2 runs 0.48 step/s at 10.8 GB, batch_size 4 runs out of memory.
    "batch_size": 2,
    "grad_accumulation_steps": 2,
    # The first run stopped at 5000 steps, 20,000 samples of the 214,909 frames, with the loss
    # still falling and 76% token accuracy; closed loop it landed 2-11 cm from the object. 20,000
    # steps is 80,000 samples, about 11.6 hours at 0.48 step/s, and a checkpoint every 1000 steps
    # means the run can be stopped early and the newest step_* directory used.
    "max_steps": 20000,            # optimizer steps
}

SAVE = {
    "root": str(TRAIN_DIR / "runs"),   # one directory per run: <config>_<date>_<time>
    "every_steps": 1000,               # LoRA adapter checkpoint every N optimizer steps, plus one at the end
    "merged": False,                   # also write the base model with the adapter merged in at the end
                                       # (a full copy of the checkpoint, 15 GB for the 7B; loads with from_pretrained)
}

LOG = {
    "every_steps": 20,   # print and append to train_log.jsonl every N optimizer steps
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
