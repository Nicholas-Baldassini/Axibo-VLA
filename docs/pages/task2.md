# Task 2

Same as Task 1, everything is set by a single config file [here](https://github.com/Nicholas-Baldassini/AxiboVLAtest/tree/main/train/config) 

## Overview

`openvla/openvla-7b` with LoRA on every linear layer, batch 16, decaying lr and image augmentation. 

I chose to stich the front camera and wrist camera into a single 224x224 image to give better sensitivty and gradient signals during grasp. I found without the wrist camera mounted, the policy sometimes stalls nearing a grasp, possibly due to occlusion or because only a few pixels change on the front camera during that action. This is still on vanilla openvla. This was a risky decision knowing openvla pretrained checkpoints likely werent trained with this type of image stiching. 

Image observation preview, 
![model input](images/task2/model_input__success_layout1.gif)

Some test rollouts of the final policy:

Red cube on red cylinder
![red cube on red cylinder](images/Demo1RedCubeCylinder.gif)

Blue cube on red cylinder
![blue cube on red cylinder](images/BlueCubeRedCylinder.gif)

Instruction worded differently from the training template
![different wording](images/DifferentWording.gif)

The action is the target end effector at 10 Hz. So translation in table coordinates, rotation as a rotation vector in the EE frame, and the gripper as a binary command. Gripper gave me a lot of problems until I made the action label binary.

**Dataset** Session [`piper_stack_2026-09-30_11-40-27`](https://huggingface.co/datasets/Boominator/vlaTestStorage/tree/main/piper_stack_2026-09-30_11-40-27) on [Hugging Face](https://huggingface.co/datasets/Boominator/vlaTestStorage) (private, request access), engine version 0.7.0: 648 demonstrations over the five trained pairs, 129k frames at 30 fps, recorded in 40 parallel environments, takes about 1hr on a 4080.


## Evaluation

`train/evaluate.py`, run on an RTX 5090 in bf16.

| item | setting |
|---|---|
| ordered pairs | all 6 combinations |
| layouts per pair | 10 trials per combination |
| trials | 60 rollouts, one per pair and layout |
| start state | objects at random position & rotation, arm at the home pose, wrist at the home yaw|
| max horizon | 100 steps at 10 Hz |
| success | same as the data engine |
| workspace floor | There is a hard clip for the gripper. If the position is clipped at min height during inference. |
| reversal test | the three pairs and their reverses on the same ten layouts |

**Failure taxonomy.** 
1. Never reached
2. Wrong object grasped
3. Missed grasp
4. Knocked destination
5. Stacked on wrong
6. Grasped and dropped
7. Disturbed third
8. Not settled.

## Results

Checkpoint [`openvla_80gb_2026-09-30_17-20-27/step_010000`](https://huggingface.co/Boominator/vlaBlockStack/tree/main/openvla_80gb_2026-09-30_17-20-27/step_010000) on [Hugging Face](https://huggingface.co/Boominator/vlaBlockStack) (private, request access), evaluation folder `train/evals/v11_final_suite`.

| ordered pair | success | failures |
|---|---|---|
| red cube on red cylinder | 6/10 | knocked destination 2, grasped and dropped 1, not settled 1 |
| red cube on blue cube | 7/10 | knocked destination 1, grasped and dropped 1, not settled 1 |
| red cylinder on red cube | 8/10 | not settled 2 |
| red cylinder on blue cube | 9/10 | grasped and dropped 1 |
| **blue cube on red cube (held out)** | **7/10** | not settled 2, grasped and dropped 1 |
| blue cube on red cylinder | 7/10 | not settled 2, grasped and dropped 1 |
| **all** | **44/60 (73%)** | |

The held-out pair, never seen in training performs just as well. 

**Reversal test**, the same layouts but instruction reversed:

| pair | forward | reverse |
|---|---|---|
| red cube / red cylinder | 6/10 | 8/10 |
| blue cube / red cube | 7/10 | 7/10 |
| blue cube / red cylinder | 7/10 | 9/10 |

Good enough, perhaps a larger trial count would show more statistically significant results. 

## Failure cases

The first rollout of some failure cases in evaluation is recorded and shown here.

**Not settled**

![not settled](images/task2/not_settled__red_cube_on_red_cylinder__layout8.gif)

**Grasped and dropped**

![grasped and dropped](images/task2/grasped_and_dropped__red_cube_on_red_cylinder__layout2.gif)

**Knocked destination** 

![knocked destination](images/task2/knocked_destination__red_cube_on_red_cylinder__layout6.gif)

## Data Engine Evolution

I ran 11 different training runs before getting to a 73% success rate. Each run exposed a potential flaw in the data engine which I iterated on. Here is a summary. 


| engine | what changed | why |
|---|---|---|
| 0.1.0 | The original, show in Task 1 |
| 0.2.0 | Removed jerk in scripted motion plan and fixed home pose | Policy was freezing nearing objects |
| 0.3.0 | Remove static frames at the start and end of an episode, made scripted motion plan more smooth with splines, remove static frames between waypoints | Policy was still freezing while hovering over object, this fixed it. |
| 0.4.0 | Added a binary gripper action column and trained on that instead of continuous scalar 0-1 | The gripper wasnt closing over an object, just staying open or slightly moving |
| 0.5.0 | Moved front camera closer to scene, increased friction in objects | low success rate at grasping |
| 0.5.1 | Moved front camera again | Objects sometimes occluded by gripper causing ambigous actions |
| 0.6.0 | Removed trapezoid timing for scripted motion plan, instead gripper moves at constant speeds in straight lines, made objects bigger, randomzied start position to increase robustness | Policy was perfectly predicting training set in open loop, but would freeze with small deviations in closed loop control|
| 0.7.0 | Randomized the wrist rotation as well in the training set, not just gripper position | This fixed everything, made it more robust |