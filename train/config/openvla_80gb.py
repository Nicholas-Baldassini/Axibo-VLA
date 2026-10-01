

import copy

from config.openvla import CONFIG as BASE

CONFIG = copy.deepcopy(BASE)
CONFIG["model"].update({"load_in_4bit": False, "dtype": "bfloat16", "gradient_checkpointing": False})
CONFIG["optim"].update({"batch_size": 16, "grad_accumulation_steps": 1, "lr": 3e-4, "max_steps": 15000,
                        "decay_after": 10000, "decay_factor": 0.1})
CONFIG["data"]["num_workers"] = 16       # ~10+ frames/s to decode; the instance has 32 cores
CONFIG["save"]["every_steps"] = 2500
