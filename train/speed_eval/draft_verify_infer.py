"""
Draft-and-verify decoding: the previous action's 7 tokens as a speculative draft.

At 10 Hz the demonstrations move at constant speed in fixed steps, so consecutive actions share most
of their tokens. The draft is fed in one forward pass and its tokens are accepted up to the first
one that differs from the greedy argmax, which is replaced by that argmax (decoding.py). The result
is exactly what token-by-token greedy decoding produces; what changes is the number of forward
passes per action: 1 after the prefill when the whole draft holds, up to 7 when nothing does. Each
engine keeps its own draft (its last action).
"""

from speed_eval.base import Optimisation


class DraftVerify(Optimisation):
    name = "draft_verify"

    def decoder_options(self):
        return {"draft": True}


OPTIMISATION = DraftVerify
