"""
Freeze the arm where it is while a prediction is pending, instead of letting it settle onto the
last command.

Every recording was made with the same position controller at 10 Hz, so in every training frame the
arm trails its command by the controller's tracking lag, a few centimetres at the demonstrations'
speed. The policy learned to decide from that picture: it closes when the fingers still look a
little above the grasp depth, because the command is already there. When the arm is allowed to
converge during a wait the lag is gone, every decision lands a lag too early (the close comes
several actions sooner and too high, the grasp is marginal, the release follows), and no amount of
decoding speed short of zero latency restores it.

Freezing holds the measured arm joints from the moment the frame is rendered until the action
lands (the fingers keep following their command), then restores the command the policy assumed
and applies the action to it. The arm still stops while the policy thinks (this is the stall,
unchanged); the hope was that the state at every observation would be the one lockstep produces.

Measured: it is not enough. The lag the policy relies on builds up over several consecutive
actions of continuous motion (up to 5 cm on the final descent, where the arm cannot follow the
policy's 15 mm steps); a pause preserves one action's worth of it but not the accumulation, so the
close still lands too high. Holding the fingers as well blocks the grasp entirely (9/9 missed).
Kept as the documented negative result.
"""

from speed_eval.base import Optimisation


class Freeze(Optimisation):
    name = "freeze"
    conflicts = ("stretch",)

    def controller_options(self):
        return {"hold": "freeze"}


OPTIMISATION = Freeze
