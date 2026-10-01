

import copy

from config.openvla import CONFIG as BASE

CONFIG = copy.deepcopy(BASE)
CONFIG["model"].update({"load_in_4bit": False, "dtype": "bfloat16", "gradient_checkpointing": True})
CONFIG["optim"].update({"batch_size": 8, "grad_accumulation_steps": 2, "lr": 3e-4, "max_steps": 15000,
                        "decay_after": 10000, "decay_factor": 0.1})
CONFIG["data"]["num_workers"] = 8
CONFIG["save"]["every_steps"] = 1000
