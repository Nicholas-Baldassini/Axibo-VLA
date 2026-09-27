# Moving this project to another machine

Nothing large is in git: `.gitignore` excludes `datasets/`, `train/runs/` and `train/checkpoints/`.
Four things have to travel, and they travel differently.

| what | size | how | needed for |
|---|---|---|---|
| the code | small | git | everything |
| base OpenVLA checkpoint | 15 GB | `train/download.py`, from `openvla/openvla-7b` | training and inference |
| a training run (adapters, config, log) | ~0.9 GB per checkpoint | your Hugging Face model repo | inference |
| the recording session (dataset) | 1.5 GB | your Hugging Face dataset repo | training, and the open-loop checks |

Inference does **not** need the dataset. It needs the base checkpoint, one adapter, and the
recording's `config.json`, which is uploaded beside the run as `session_config.json`.

---

## 1. Before you leave: commit the code

The clone is useless without it. `train/infer.py` in particular is new.

```bash
git add -A && git commit -m "recording on GPU, trimmed training, inference GUI" && git push
```

## 2. Upload the training run

Everything in the run directory: every checkpoint, the resolved config, the training log.

```bash
cd train/runs/openvla_2026-09-26_16-35-56
hf upload Boominator/vlaBlockStack .
```

Then the recording's config, which inference needs and the run only references by an absolute
path that will not exist on the other machine:

```bash
hf upload Boominator/vlaBlockStack \
  ~/Documents/Axib/AxiboVLAtest/datasets/piper_stack_2026-09-26_00-01-05/config.json \
  session_config.json
```

The repo then looks like this, which is the layout `train/infer.py` expects:

```
config.json              the training config, and which session it trained on
session_config.json      the scene, the six task pairs, the held-out pair
train_log.jsonl          every logged step
dataset_statistics.json  action normalization for the run
step_001000/ ... step_004000/ , final/
    adapter/             LoRA weights, processor, dataset_statistics.json
    metrics.json         loss, accuracy, grad_norm at that step
```

## 3. Upload the dataset, if you want to train there too

From the session's `dataset/` subdirectory, so `meta/`, `data/` and `videos/` land at the repo
root and the result loads as a LeRobot dataset:

```bash
cd ~/Documents/Axib/AxiboVLAtest/datasets/piper_stack_2026-09-26_00-01-05
hf upload Baldinator/vlaPickandPlaceDataset dataset . --repo-type=dataset
for f in metadata.json config.json episodes.json requirements.lock.txt; do
  hf upload Baldinator/vlaPickandPlaceDataset "$f" "session/$f" --repo-type=dataset
done
```

---

## 4. On the new machine

```bash
git clone <this repo> && cd AxiboVLAtest
conda create -n axibEnv python=3.11 -y && conda activate axibEnv
pip install -r requirements.txt

python train/download.py                     # base openvla-7b, 15 GB

hf download Boominator/vlaBlockStack --local-dir train/runs/vlaBlockStack
```

That is enough to run the policy:

```bash
python train/infer.py train/runs/vlaBlockStack/final
```

### The dataset, for training or the open-loop checks

The loader wants a session directory with the LeRobot dataset under `dataset/`, so the two halves
of the repo are put back where they came from:

```bash
SESSION=datasets/piper_stack_2026-09-26_00-01-05
mkdir -p $SESSION/dataset
hf download Baldinator/vlaPickandPlaceDataset --repo-type=dataset --local-dir /tmp/vla-dataset
mv /tmp/vla-dataset/meta /tmp/vla-dataset/data /tmp/vla-dataset/videos $SESSION/dataset/
mv /tmp/vla-dataset/session/* $SESSION/
```

Check it landed right, and train:

```bash
ls $SESSION/dataset/meta/info.json $SESSION/metadata.json     # both must exist
python train/finetune.py --config openvla
```

`data.session: None` picks the most recently written session under `datasets/`, so nothing needs
editing if this is the only one.

---

## What the machine has to be

**Rendering.** A policy is sensitive to which renderer produced its input. This dataset was
recorded with the Madrona batch renderer on CUDA; evaluating the same policy against the CPU
rasterizer instead changes the predicted actions completely, even though the two images differ by
only about 12 of 255 grey levels on average. `train/infer.py` defaults to the backend recorded in
the session's `metadata.json` and warns if `--backend` disagrees with it. Madrona is Linux x86-64
and CUDA only, so a faithful rollout needs that; a Mac or a CPU-only box can train and can run the
open-loop checks against recorded frames, but its rollouts are not measuring this policy.

**Memory.** Inference holds two things on the GPU at once: the batch renderer, about 5.7 GB even
for one environment, and the 4-bit policy, about 5.7 GB. An 11 GB card cannot do both, which is
why the rollouts here were run against the wrong renderer to begin with. 16 GB is the floor and
24 GB is comfortable.

Training is lighter, since it renders nothing: 4-bit weights plus LoRA rest at 5.7 GB and the
whole run fits in about 9.6 GB at `batch_size: 1`. On a larger card raise `optim.batch_size` and
drop `optim.grad_accumulation_steps` to match; only their product matters to the gradient.

**Recording** needs the most GPU memory of the three. `ik.n_envs` is set for an 11 GB card; the
note on that key in `source/config.py` has the measured numbers and how to re-derive them.
