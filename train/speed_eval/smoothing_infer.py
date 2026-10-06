"""
Action-level smoothing: an exponential moving average over the applied translation and rotation.

    a_t = alpha * predicted_t + (1 - alpha) * a_{t-1}        (alpha = 1 is no smoothing)

The gripper is a decision, not a motion, and passes through untouched. This is the one
optimisation here that changes what the arm is told to do: it trades a lag of about (1 - alpha) /
alpha control steps for smaller step-to-step changes, so its place in the report is the success
rate next to the smoothness numbers. The filter restarts at every rollout.
"""

import numpy as np

from speed_eval.base import Optimisation


class Smoothing(Optimisation):
    name = "smoothing"

    def controller_options(self):
        alpha = float(self.settings.get("alpha", 0.5))
        dims = slice(0, 6)

        class Filter:
            def __init__(self):
                self.prev = None

            def __call__(self, action):
                action = np.array(action, dtype=float)
                if self.prev is not None:
                    action[dims] = alpha * action[dims] + (1 - alpha) * self.prev[dims]
                self.prev = action
                return action

        return {"filter": Filter}


OPTIMISATION = Smoothing
