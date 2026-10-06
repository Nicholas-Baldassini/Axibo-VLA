"""
Execute each action over the time until the next one, instead of in 100 ms followed by a hold.

The policy's action is the displacement for the next control period. Applied all at once, the arm
moves for about 100 ms and then stands still until the next prediction lands: the stall. Here the
command is moved along the action at constant speed over the expected interval between landings
(the median of the recent ones), so the arm is always in motion and arrives at the action's target
as the next action is due. The policy, its inputs and its outputs are untouched; what changes is the
trajectory the controller traces between them, which is also why the arm's state at every
observation again resembles the recordings (mid-motion, not settled).

mode "velocity" keeps the speed past the target when the next action is late, up to 1.5 actions.
"""

from speed_eval.base import Optimisation


class Stretch(Optimisation):
    name = "stretch"

    def controller_options(self):
        return {"hold": self.settings.get("mode", "stretch")}


OPTIMISATION = Stretch
