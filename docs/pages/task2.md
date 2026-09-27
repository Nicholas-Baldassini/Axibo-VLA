# Task 2

<p class="placeholder">Working notes, 2026-09-27. Not the write-up yet: a record of what was tried, what was measured, and what changed, with code references, to be turned into the finished section once the next run is evaluated.</p>

## Runs so far

| run | data | steps / samples | training metrics at the end | closed loop |
|---|---|---|---|---|
| `vlaBlockStack` (run 1) | session `piper_stack_2026-09-26_00-01-05`, engine 0.2.0, front camera | 5,000 / 20,000 (under 0.1 epoch) | loss 0.66, token accuracy 0.76, normalized L1 0.022 | approaches the right object, lands 2 to 11 cm off, half-closes the gripper, then predicts zero forever |
| `openvla_2026-09-27_00-58-03` (run 2), stopped at step 16,000 | same session, min/max bounds, batch 2 x 2 | 16,000 / 64,000 | loss 0.51, accuracy 0.81, L1 0.009 | arrives within 1.5 to 3 cm of the object at hover, then drifts about 10 cm in x while emitting a 1.5 mm action, descends onto empty table |

Both were fine-tuned from `openvla/openvla-7b` with LoRA rank 32 on 4-bit weights, one image and the instruction, seven action tokens per step at 10 Hz (`train/config/openvla.py`).

## Issues found

1. **The policy fits the data but not the loop.** Open-loop on recorded frames, run 1 matches the recorded targets closely, including closing at the grasp and opening at the place; replaying recorded states in the live simulator gives the same error as on decoded video frames. So the checkpoint, the renderer match and the frame pipeline were ruled out one by one; the failure is what happens after the policy's own first small error.
2. **Renderer mismatch (fixed).** The policy is sensitive to which renderer drew its input: the CPU rasterizer's frame of the same layout differs by 12/255 on average from the Madrona batch renderer's, and the predicted rotations and dy roughly double on it. Inference now defaults to CUDA (`train/infer.py:626`) and fits beside the model on a 12 GB card by trimming Madrona's device heap from its 4 GB default (`train/infer.py:222`; measured 5.2 GB to 1.4 GB, frames pixel-identical).
3. **Speed cap from the normalization (fixed).** OpenVLA's 1st to 99th percentile bounds clipped the demonstrations' fast sideways moves (per-step dy up to 0.064 m against a q99 of 0.037). Over 670 episodes that lost a mean of 1.3 cm and up to 6 cm of approach. Bounds are now the dataset min and max (`train/actions.py:61`, `train/config/openvla.py:63`), with the file layout unchanged so inference reads it as before.
4. **Under-training (partly fixed).** Run 1 saw under a tenth of an epoch. Run 2 used batch 2 with accumulation 2 (measured 0.48 step/s at 10.8 GB on the RTX 4070; batch 4 runs out of memory) for 20,000 steps (`train/config/openvla.py:115-121`). Training metrics improved steadily and closed-loop approach accuracy improved with them, but the run still failed.
5. **No corrections in the data (the measured root cause).** A controlled test on run 2: drive the policy to hover over the target, displace the tool by -4 to +4 cm in x or y, read its prediction. The prediction does not change with the offset, on either axis. In the dataset, the tool-to-object offset at the start of every descent has a standard deviation of 0.8 mm across all 670 episodes: the scripted plan always arrives at the same relative spot, so there is nothing to learn about correcting. Behaviour cloning under covariate shift; the fix belongs in the data engine.
6. **A dwell in the plan (the freeze).** The min-jerk profile stops at every waypoint, so each demonstration holds a static picture at hover height, gripper open, over the object, labelled "do nothing" for about three control steps, then "descend". A history-free model cannot tell those apart. Run 1, loosely fitted, fell through to descending; run 2, better fitted, resolved the ambiguity toward staying, and the front camera's near-blindness to depth turned its residual output into an unseen drift.
7. **Camera.** For the original front camera, a 4 cm tool move in x changes the image by 1.2/255 on average (x is its depth axis) and 1.9/255 in y, with a cube covering about 456 pixels. Measured as secondary to item 5, since the displacement test failed on y too, but worth fixing while re-recording.

## Data engine changes (engine 0.3.0)

Sessions recorded from 2026-09-27 on are engine `0.3.0` (`VERSION`), not interchangeable with `0.2.0` sessions: different camera pose, different plan, no opening frames.

- **Hover offsets.** The hover points above the source and the destination are displaced sideways per episode, uniform up to 3 cm on each axis (`source/config.py:214`, drawn in `env_config` at `source/config.py:426`; applied in `source/utils/motion.py:178`). The descent runs diagonally to a point 3 cm straight above the object, then vertically (`source/utils/motion.py:205`). Trial of 9 episodes: offset at the descent spreads with 3 cm standard deviation on both axes.
- **No resting states.** Approach and carry are each one continuous spline through their hover points (`source/utils/motion.py:211`, `:216`; `arc_path` now accepts via points, `:287`). Only the grasp and release are stationary, where the gripper label is what changes. Trial: the slowest commanded step above the object is 15 mm, against 1.2 mm before.
- **One cruise speed.** Trapezoid profile at 0.3 m/s with 1.5 m/s^2 ramps (`source/config.py:289`). Median step along the approach 28 to 30 mm; first six labels of an episode 9, 14, 19, 24, 28, 30 mm instead of 0.5 mm.
- **Start pose perturbation.** Each arm joint of the home pose is perturbed by up to 4 degrees per episode (`source/config.py:221`, `:419`); about 2.5 cm standard deviation at the tool.
- **Opening frames not stored.** The recorder skips frames until an env's first task-space move (`source/utils/recorder.py:382`, `:493`), replacing the 27-frame cut that training applied to 0.2.0 sessions (`train/config/openvla.py:31`).
- **Front camera moved** to (0.45, 0, 0.55) looking at (-0.08, 0, 0.02) (`source/config.py:96-104`): 2.0/255 in x, 2.8/255 in y per 4 cm, about 821 pixels per cube, whole placement area in frame.

Trial recordings with the full set: 10 attempts, 9 successes, twice; the one failure is the same layout both times.

## Held in reserve

- **Trajectory noise (DART):** execute a noisy command, label with the clean plan, so corrections exist along the whole trajectory rather than only in the descents. Needs the recorder to store the clean command beside the executed one and the loader to build deltas across the two. Gated on the displacement test after the next run.
- **Two camera views** through the same OpenVLA checkpoint, OpenVLA-OFT style: each view through the shared vision encoder and projector, patch tokens concatenated. No new pretrained weights. Gated on whether x precision is still the limit.

## Evaluation harness to build from

The rollout probes used above are the seed of the protocol the brief asks for: per-pair success with seed list and trial count, failure taxonomy, the reversal test on a fixed layout, and the held-out pair (blue cube on red cube). Layouts are reproducible by reset number, so the fixed-scene reversal test is two instructions on the same reset.
