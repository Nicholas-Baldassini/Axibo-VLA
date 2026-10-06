"""
What an optimisation module looks like. Each file in this directory named *_infer.py defines one
subclass and exposes it as OPTIMISATION; speed_eval.py stacks the ones a config names and asks
each for its part:

    model_config(cfg)      change how the checkpoint is loaded (precision, quantization)
    prepare(policy)        act on the loaded model once (merge the adapter, compile)
    decoder_options()      how the action tokens are decoded: {"static_cache", "draft"}
    prepare_decoder(dec)   act on each Decoder built for an engine (compile its step)
    controller_options()   how predictions are scheduled and applied: {"engines", "filter"}

Every method has a no-op default, so a module overrides only what it changes. `conflicts` lists
the names of optimisations that cannot be stacked with it, and `settings` is the dict from the
config's SETTINGS under the module's key (the file name without _infer).
"""


class Optimisation:
    name = "base"
    conflicts = ()
    needs_cuda = False

    def __init__(self, settings=None):
        self.settings = dict(settings or {})

    def model_config(self, model_cfg):
        return model_cfg

    def prepare(self, policy):
        pass

    def decoder_options(self):
        return {}

    def prepare_decoder(self, decoder):
        pass

    def controller_options(self):
        return {}

    def describe(self):
        return self.name + (f" {self.settings}" if self.settings else "")
