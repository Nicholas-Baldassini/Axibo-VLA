"""
Speed evaluation: which optimisations to test, on which rollouts, under which clock.

    python train/speed_eval/speed_eval.py --config speed_eval

OPTIMS lists the stacks to run; each entry is a tuple of module names from train/speed_eval/
(with or without the _infer.py suffix). The empty tuple, or "none", is the vanilla inference of
train/infer.py (generate(), one action at a time, the simulator waiting) and is the baseline.
COMBOS lists (source, destination, rollouts); rollout i of every pair, in every stack, uses layout
SEEDS[i], so each stack sees exactly the same scenes.
"""

from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parents[1]

RUN = str(TRAIN_DIR / "runs" / "openvla_80gb_2026-09-30_17-20-27" / "step_010000")   # run 11, the 44/60 checkpoint

OPTIMS = [
    (),                                          # vanilla: generate(), whatever precision infer.py picks for the card
    ("b4_infer",),
    ("b16_infer",),
    ("torch_compile_infer"),
    ("smoothing_infer"),
    ("draft_verify_infer"),
    ("pipeline_infer"),
    ("b16_infer", "torch_compile_infer"),
    ("b16_infer", "draft_verify_infer"),
    ("b16_infer", "torch_compile_infer", "draft_verify_infer"),
    ("b16_infer", "torch_compile_infer", "draft_verify_infer", "smoothing_infer"),
    ("b16_infer", "draft_verify_infer", "pipeline_infer"),
]

COMBOS = [
    ("red_cube", "red_cylinder", 3),
    ("red_cylinder", "blue_cube", 3),
    ("blue_cube", "red_cube", 3),                # the held-out pair
]

SEEDS = list(range(1, 11))                       # layout numbers: rollout i of a pair uses SEEDS[i]

# "realtime": a prediction launched at simulated time t lands at t + its measured latency, and the
# arm holds until then. "lockstep": it lands at t (the simulator waits), the protocol of task 2.
TIMING = "realtime"

HORIZON = {
    "actions": 100,        # policy actions per rollout (the task 2 protocol's 100 steps at 10 Hz)
    "sim_s": 40.0,         # and never more simulated seconds than this
}

CONTROL = {
    "backend": "cuda",     # the renderer the data was recorded with; see infer.py
    "z_min": 0.013,
    "gripper": "threshold",
    "start_yaw": "home",
    # None: each prediction's measured latency goes on the simulated clock. A number of seconds:
    # that fixed latency instead (reproducible anywhere; success against stall length).
    "latency": None,
    # What the arm does between landings when no stack says otherwise: "pose" jumps to the target
    # and holds (the stall), "stretch"/"velocity" keep it moving (stretch_infer.py).
    "hold": "pose",
}

VIDEO = {
    "fps": 30,             # frames per simulated second: a stall shows as the arm standing still, tagged WAITING
    "camera": "front",
}

CHECK = {
    "frames": 20,          # recorded frames on which every stack's decoding is compared with generate() before its rollouts
}

SETTINGS = {
    "b16": {"merge_lora": True},
    "torch_compile": {"mode": "reduce-overhead", "vision": False},   # vision: the timm backbone does not compile, see torch_compile_infer.py
    "smoothing": {"alpha": 0.5},
    "pipeline": {"engines": 3},
    "stretch": {"mode": "stretch"},     # or "velocity"
    "threads": {"threads": 4},          # torch CPU threads; see threads_infer.py
}

OUT = str(TRAIN_DIR / "evals" / "speed_eval")

CONFIG = {
    "run": RUN, "optims": OPTIMS, "combos": COMBOS, "seeds": SEEDS, "timing": TIMING, "horizon": HORIZON,
    "control": CONTROL, "video": VIDEO, "check": CHECK, "settings": SETTINGS, "out": OUT,
}
