"""
bf16 weights, as the run was trained, instead of the 4-bit load infer.py falls back to on a small card.

The 4-bit bitsandbytes kernels dequantize every weight before each matmul; in bf16 the 7B's
weights stream straight through the tensor cores, which is where the decode time goes. Needs about
16 GB of free GPU memory (15 GB of weights plus the renderer). With merge_lora (default on) the LoRA
adapter is folded into the base weights first, which removes the adapter's extra matmuls from every
layer; the merge is a bf16 rounding of W + BA, and the exactness check reports what it changes.
"""

import torch

from speed_eval.base import Optimisation


class Bf16(Optimisation):
    name = "b16"
    conflicts = ("b4",)
    needs_cuda = True

    def model_config(self, model_cfg):
        free_gb = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0.0
        if free_gb < 16:
            raise SystemExit(f"b16_infer needs about 16 GB of free GPU memory for the bf16 checkpoint; {free_gb:.1f} GB free")
        return {**model_cfg, "load_in_4bit": False, "dtype": "bfloat16"}

    def prepare(self, policy):
        if self.settings.get("merge_lora", True) and hasattr(policy.model, "merge_and_unload"):
            policy.model = policy.model.merge_and_unload().eval()
            policy.input_dtype = next(p.dtype for p in policy.model.parameters() if p.is_floating_point())


OPTIMISATION = Bf16
