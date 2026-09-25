# Task 1

[github link](https://github.com/Nicholas-Baldassini/AxiboVLAtest)

## Pipeline

Everything is configured in one file, `source/config.py`: the scene, the cameras, the robot gains, the motion plan, the randomization, the success criteria and what gets recorded.

```
python source/record.py
```

![Data Engine](images/engine.gif)
4 envs at 30fps on my mac pro m3. 



It runs `record.epochs` rounds. In every round each of the `n_envs` parallel environments gets its own config from `env_config(config, env, episode)`, which assigns the task pair round robin and draws every object's position and yaw from the session seed together with the env and episode numbers. The plan for that task is generated from object-relative targets, `pick_and_place(pick=Obj(source)(), place=on_top(destination, source))`, with the gripper turned to close across the picked object's faces, and resolved against the drawn layout as the last step.

**Dataset.** <mark>[[link](https://huggingface.co/datasets/Baldinator/vlaPickandPlaceDataset)]</mark>
Please request access, it is a private dataset. (Full dataset not yet uploaded)

## Held-out pair

I chose the held out pair to be **put the blue cube on the red cube**. because in my opinion

- **Cube on cube is the most stable placement.** Cube on cube seems the most stable placement. So it should be the easiest task for a policy to complete, if it happens to perform worse on the held out task. 

I considered also that blue is a single colour and cylinder is a single shape. red cube is the only object that is not identified by a single descriptive i.e red nor cube is unique. 

## Observations

Every quantity below is recorded at 30 Hz, the physics runs at 90 Hz. All vectors are float32. Every Cartesian quantity is in table coordinates. Quaternions for orientation since euler angles could be ambigous.

| Key | Dims | What it is |
|---|---|---|
| `observation.images.overhead` | 480×640×3 | bird's eye view |
| `observation.images.front` | 480×640×3 | camera opposite the robot |
| `observation.images.wrist` | 480×640×3 | camera on the wrist |
| `observation.state` | 8 | joint positions: six arm joints, two gripper joints  |
| `observation.joint_pos_delta` | 8 | delta joint positions |
| `observation.joint_vel` | 8 | joint velocities |
| `observation.joint_vel_delta` | 8 | delta joint velocities |
| `observation.tcp_pose` | 8 | tool centre xyz, quaternion, gripper opening 0..1 vector |
| `observation.tcp_pose_delta` | 7 | tcp delta |
| `observation.link_poses` | 63 | xyz and quaternion of all nine robot links |
| `observation.link_poses_delta` | 54 | deltas |
| `observation.objects` | 21 | SE3 of each object |
| `success` | 1 | always 1.0, since only successful attempts are written |

I decided to record more than just the images because I might as well, it is free. If I ever wanted to train something state based in the future, we have the data already saved, no need to recollect. A lot of the data can also be computed after the fact (FK for example) but it is nice to have here. Likewise, we can choose which camera angle, or all of them, when training without recollecting. 

So during training, just choose a subset here to train. Also sometimes you can combine the state of a system with the image input, joint readings on a real robot is normally free as well. 

**Camera Intrinsics.** I also added camera intrinsics and resolution into the config. If this were ever used for sim to real, we can adjust the focal length etc. Could of also added depth if desired, RGBD instead of just RGB. 

## Actions

The same logic for actions, collect more than necessary because its free why not. 

| Key | Dims | What it is |
|---|---|---|
| `action` | 8 | tool pose & gripper |
| `action.tcp_pose_delta` | 7 | tool pose delta |
| `action.joint_pos` | 8 | joints |
| `action.joint_pos_delta` | 8 | joint delta |
| `action.joint_vel` | 8 | joint vel |
| `action.joint_vel_delta` | 8 | delta |
| `action.link_poses` | 63 | each link pose |
| `action.link_poses_delta` | 54 | deltas |

`action` if desired, can transform coordinate frame to be in robot coordinates, but for now saved in world frame. 

The task description in plain text, 'put the {source} on the {destination}' is also saved per episode. 

## Randomization and episode success

Every object's is randomly placed on the table with 5cm buffer between objects. Object rotation kept constant.

The monitor starts judging when the arm has retreated, and applies these tolerances every step:

| Criterion | Tolerance |
|---|---|
| source over the destination | within 1.5 cm in x and y |
| source at its resting height on the destination | within 1 cm |
| destination upright and at rest height | tilt under 10°, height within 5 mm |
| third object undisturbed | within 1 cm of where it spawned, tilt under 10° |
| scene still | no object drifts more than 2 mm or rotates more than 2° over a 2 s window |
| settle timeout | 5 s after the plan ends |


## What a session saves

```
datasets/piper_stack_<date>_<time>/
├── metadata.json           date, seed, hardware, versions, counts, success rate, tolerances, camera models
├── config.json             the full config the session ran with
├── episodes.json           every attempt: epoch, env, task, instruction, outcome, frames, episode index, resolved config
├── requirements.txt        the repo's pinned requirements
├── requirements.lock.txt   pip freeze of the recording environment
└── dataset/                the LeRobot v3 dataset: meta/, data/ (parquet), videos/ (AV1 mp4)
```

- [config.py](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/source/config.py), the configuration every session runs from
- [metadata.json](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/docs/data/metadata.json) from a recorded session
- [episodes.json](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/docs/data/episodes.json) from a recorded session
- [requirements.txt](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/requirements.txt), the pinned requirements
- [requirements.lock.txt](https://github.com/Nicholas-Baldassini/AxiboVLAtest/blob/main/docs/data/requirements.lock.txt), the exact environment of the recording

## Dataset summary
A sanity check visualizer to review a recorded dataset. Purely for debugging. 
This is mainly to view the non-image based data, if it will be used. 

```
python source/visualization/dataset_summary.py datasets/<session>
```

![Dataset summary](images/summary.png)

## Dataset visualization

```
python source/visualization/visualize_dataset.py datasets/<session>
```

Also for debugging

![Dataset visualization](images/viewer.gif)
