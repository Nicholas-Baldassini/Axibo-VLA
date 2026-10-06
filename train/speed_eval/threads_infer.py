"""
Pin torch's CPU thread pool to a few threads.

The only CPU work per prediction is the image processor (crop, resize to 224, normalize) and the
prompt's tokenization, a few milliseconds of single-threaded work. torch sizes its intra-op pool to
every core it sees, and on a shared or containerized host (the rented 5090: 24 cores visible, fewer
granted) the pool's spin-wait costs more than the work: measured there, the image processor took
113 ms with the default 24 threads and 0.5 ms with 1, and the host-to-device copy 11 ms against 0.1.
Nothing about the model or its outputs changes.
"""

import torch

from speed_eval.base import Optimisation


class Threads(Optimisation):
    name = "threads"

    def prepare(self, policy):
        n = int(self.settings.get("threads", 4))
        torch.set_num_threads(n)
        print(f"   torch CPU threads: {n}", flush=True)


OPTIMISATION = Threads
