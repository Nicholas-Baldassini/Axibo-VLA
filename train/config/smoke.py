"""
Pipeline test that runs on a laptop: the real OpenVLA tokenizer, processor and data path driving a
random-init miniature of the model for a few steps. Checks that samples, tokens, losses, metrics and
checkpoints all come out; the loss should fall as the miniature memorizes the data.

    python train/finetune.py --config smoke
"""

import copy

from config.openvla import CONFIG as BASE

CONFIG = copy.deepcopy(BASE)
CONFIG["model"].update({"tiny": True, "dtype": "float32"})
CONFIG["data"]["num_workers"] = 0
CONFIG["optim"].update({"batch_size": 4, "grad_accumulation_steps": 1, "max_steps": 40, "lr": 1e-3})
CONFIG["save"].update({"every_steps": 20, "merged": True})
CONFIG["log"]["every_steps"] = 5
