"""
4-bit (NF4, bitsandbytes) weights with float16 compute: the load infer.py uses on a card under
24 GB, made explicit so it can be measured against bf16 on the same card.
"""

from speed_eval.base import Optimisation


class FourBit(Optimisation):
    name = "b4"
    conflicts = ("b16",)
    needs_cuda = True

    def model_config(self, model_cfg):
        return {**model_cfg, "load_in_4bit": True, "dtype": "float16"}


OPTIMISATION = FourBit
