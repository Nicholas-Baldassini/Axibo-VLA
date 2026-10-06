"""
torch.compile on the decode step, with a static KV cache so the shapes never change.

Each of the 7 action tokens is one forward of the 7B over a single token: hundreds of small kernels
launched from Python, most of the step at batch size 1. With fixed cache buffers the step can be
captured as a CUDA graph (mode "reduce-overhead") and replayed. The prefill keeps its length (the
prompt varies with the instruction) and is left uncompiled. The vision tower and projector have a
fixed shape and can be compiled too with `vision` on, but the checkpoint's timm backbone trips
dynamo (its get_intermediate_layers builds a set from a range), so that is off by default. The
first calls compile and are excluded from the latency statistics. Output is the same greedy
argmax; the exactness check confirms it.

Cannot stack with pipeline_infer: the static cache is one buffer per layer inside the model, so two
engines would overwrite each other's cache.
"""

import time

import torch

from speed_eval.base import Optimisation


class TorchCompile(Optimisation):
    name = "torch_compile"
    conflicts = ("pipeline",)
    needs_cuda = True

    def decoder_options(self):
        return {"static_cache": True}

    def prepare_decoder(self, decoder):
        mode = self.settings.get("mode", "reduce-overhead")
        # One graph per step length, 1..7 tokens, is more than dynamo's default of 8 recompiles
        # per function allows before it gives up and runs eagerly.
        dynamo = torch._dynamo.config
        for name in ("recompile_limit", "cache_size_limit"):          # the name depends on the torch version
            if getattr(dynamo, name, None) is not None:
                setattr(dynamo, name, max(getattr(dynamo, name), 4 * decoder.n_tokens))
        decoder.step_fn = torch.compile(decoder._step, mode=mode, dynamic=False)
        if self.settings.get("vision", False):
            vla = decoder.vla
            decoder.embed_patches = torch.compile(lambda px: vla.projector(vla.vision_backbone(px)), mode=mode, dynamic=False)
        # One graph per step length (1 token, or 2..7 when a draft is verified): record them all
        # now, so compilation never lands inside a timed prediction.
        t0 = time.perf_counter()
        decoder.warmup()
        print(f"   torch.compile: {decoder.n_tokens} step shapes compiled and warmed in {time.perf_counter() - t0:.0f} s", flush=True)


OPTIMISATION = TorchCompile
