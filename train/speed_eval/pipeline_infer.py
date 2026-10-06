"""
Pipelined parallel inference: several engines, a new prediction launched every control period on
the newest frame, each applied when it returns.

With one engine the loop is observe, wait a whole prediction, apply, move: the arm sees a new
action every latency plus one control period. With N engines (threads, each on its own CUDA stream,
each with its own decoder over the shared weights) up to N predictions are in flight, so actions can
arrive every control period while each one is still based on a frame one latency old. What the GPU
gives back when it is shared N ways is what the latency and action-rate numbers measure; the policy
itself is unchanged.
"""

from speed_eval.base import Optimisation


class Pipeline(Optimisation):
    name = "pipeline"
    conflicts = ("torch_compile",)

    def controller_options(self):
        return {"engines": int(self.settings.get("engines", 3)), "schedule": "pipelined"}


OPTIMISATION = Pipeline
