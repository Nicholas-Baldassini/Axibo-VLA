# Training

Fine-tuning OpenVLA on sessions recorded by the data engine in `source/`.

```
train/
├── config/            one config per run type: openvla.py the base settings (12 GB card, 4-bit), openvla_80gb.py
│                      and openvla_5090.py the bf16 runs for those cards, smolvla.py the SmolVLA run
├── smolvla/           the same fine-tune with SmolVLA instead of OpenVLA: data.py hands the session over under the
│                      names SmolVLA expects (and says why), finetune.py is the loop, policy.py runs a checkpoint in the
│                      simulator behind the same interface as the OpenVLA policy; its own conda env (requirements.txt)
├── speed_eval/        inference speed and smoothness (task 3): speed_eval.py runs the seeded rollouts of config/speed_eval.py
│                      under every optimisation stack it lists, one *_infer.py per optimisation (b16, b4, torch_compile,
│                      draft_verify, smoothing, pipeline, stretch, freeze), decoding.py the decoder they share, controller.py
│                      the clock that puts the policy's latency into the simulation, check_decoder.py the token-exactness
│                      check, report.py one table across evaluation folders (real time against the lockstep ceiling, the
│                      fixed-latency sweep)
├── checkpoints/       downloaded base models (not committed)
├── runs/              one directory per training run (not committed)
├── evals/             one folder per evaluation: protocol, seeds, per-rollout records, summary, videos
├── download.py        one command for a new machine: base checkpoint, a run, a session, from Hugging Face
├── upload.py          a run to Hugging Face, as a version beside earlier ones (sessions: source/upload.py)
├── upload_watch.sh    on a rented machine: upload every Nth checkpoint while training, prune the rest
├── infer.py           the policy in the simulator, from a small GUI
├── evaluate.py        the evaluation protocol: per-pair success over fixed layouts, taxonomy, videos
├── finetune.py        the training loop
├── data.py            LeRobot session -> (image, instruction, action tokens) samples
├── observe.py         the policy image: cameras tiled into one frame, augmentation, centre crop
├── actions.py         action vector from the recorded columns, normalization, action tokens
└── model.py           model loading (bf16 or 4-bit) and LoRA
```

```
python train/download.py                                 # the base checkpoint, once
python train/finetune.py --config openvla_80gb           # the fine-tune, on an 80 GB card (openvla_5090 on a 5090)
python train/evaluate.py train/runs/<run>/step_010000    # the evaluation suite, bf16 on a card that runs Madrona
python train/infer.py train/runs/<run>/step_010000       # the policy in the simulator, interactively
```

The SmolVLA run (450M parameters, trains on the 12 GB card in about 5 hours, same session, same run layout):

```
conda create -n smolvlaEnv python=3.11 && conda activate smolvlaEnv
pip install -r train/smolvla/requirements.txt
python train/smolvla/finetune.py --config smolvla
python train/evaluate.py train/runs/<smolvla run>/final --horizon 300     # 30 Hz control, so 300 steps = the 10 s of OpenVLA's 100
python train/infer.py train/runs/<smolvla run>/final --action-steps 10    # re-plan every 10 frames instead of the whole 50-frame chunk
```

evaluate.py and infer.py read the run's config.json to tell the two apart; a SmolVLA checkpoint is the directory
holding model.safetensors, and `--action-steps` sets how much of each predicted chunk is executed before the
next prediction.

The speed evaluation (task 3) runs the same seeded rollouts under each optimisation stack of
`train/config/speed_eval.py` and reports latency p50/p99, stall, jerk and success side by side; see
the docstrings in `train/speed_eval/`:

```
python train/speed_eval/speed_eval.py --config speed_eval                      # every stack in the config
python train/speed_eval/speed_eval.py --optims none,b16_infer+draft_verify_infer --combos "red_cube>red_cylinder:3" --timing lockstep
python train/speed_eval/check_decoder.py train/runs/<run>/step_010000          # the decoder against generate(), on its own
python train/speed_eval/speed_eval.py --optims none --latency 100               # a fixed latency on the clock instead of the measured one
python train/speed_eval/report.py train/evals/speed_eval                        # every evaluation folder in one table
```

Evaluation needs the Madrona renderer, which works on RTX 5090 hosts and not on the H100 hosts tried; `--trials`, `--horizon` and `--start-yaw` set the protocol, and every evaluation writes its folder under `train/evals/`.

## Moving between machines

Nothing large is in git. One command fetches the base checkpoint, a fine-tuned run and a recorded
session from Hugging Face and puts each where the code expects it:

```
python train/download.py --run Boominator/vlaBlockStack --dataset Boominator/vlaTestStorage
python train/infer.py train/runs/<run>             # the policy in the simulator
python train/finetune.py --config openvla          # a new fine-tune on the session
```

Repos hold versions side by side, one folder per run or session, named after the directory on
disk; the newest is taken unless `--version` names one, and `--list` shows what is there. Going the
other way, `python train/upload.py <run dir> --repo ...` pushes a run with the recording's config
beside it, and `python source/upload.py <session dir> --repo ...` pushes a session. Both compare
every file with the repo first and print what is new, already there, or would be overwritten.

## The checkpoint

The base model is `openvla/openvla-7b` from Hugging Face: <https://huggingface.co/openvla/openvla-7b>.
It is the 7B model released with the OpenVLA paper (Kim et al., 2024, <https://github.com/openvla/openvla>),
pretrained on 970k robot episodes from Open X-Embodiment. Inside it is a Prismatic VLM: a fused
DINOv2 + SigLIP vision backbone at 224x224, a two-layer MLP projector, and a Llama 2 7B language model.
Actions come out as text tokens: each action dimension is discretized into 256 bins that map onto
the last 256 entries of the Llama vocabulary. The weights and model code are MIT licensed; the Llama 2
backbone carries Meta's Llama 2 license.

How it gets onto a machine:

```
python train/download.py
```

That runs `snapshot_download` on the repo into `train/checkpoints/openvla-7b/`. The full checkpoint is
15.1 GB: three safetensors shards, `config.json` (which also carries the normalization statistics of
every pretraining dataset), the tokenizer, the image processor config, and three Python files
(`modeling_prismatic.py`, `configuration_prismatic.py`, `processing_prismatic.py`) that define the
model. Those files are why `transformers`, `tokenizers` and `timm` are pinned in `requirements.txt`
to the versions the checkpoint was written for; the model refuses newer `timm` outright.

The script checks free space before downloading and refuses without a 5 GB margin; rerun it after
clearing space and it fetches only what is missing.

## Why LoRA

Full fine-tuning of a 7B model means gradients and two Adam moments for every weight, about 16 bytes
per parameter on top of the weights themselves, well over 100 GB across several GPUs. LoRA (low-rank
adaptation) freezes the base model and trains small rank-32 matrices added to every linear layer,
around 1.4% of the parameters. OpenVLA's authors fine-tune this way themselves and report it matching
full fine-tuning on their tasks. What it buys us:

- **It fits.** OpenVLA's numbers: about 27 GB at a small batch, 72 GB at batch 16, on a single GPU.
  With the base weights loaded in 4-bit (`model.load_in_4bit`) the 16 GB RTX 4080 can run it at batch 1.
- **Small artifacts.** A run saves an adapter of a few hundred MB instead of a 15 GB copy of the model,
  so checkpoints are cheap to keep and to move between machines. `save.merged` writes the full merged
  model when a self-contained checkpoint is wanted.
- **The base stays intact.** Adapters for different datasets or action spaces can share one download,
  and a few thousand frames of stacking data cannot corrupt the pretrained weights.

The same loop trains everything else unchanged if LoRA is ever dropped: the adapter is the only
difference between `model.py`'s output and the base model.

## What a run does

Every frame of the session becomes one sample: the configured camera image resized to 224x224 by the
checkpoint's processor, the recorded instruction dropped verbatim into OpenVLA's pretraining prompt
("In: What action should the robot take to {instruction}?\nOut:"), and the action vector built
per `config/openvla.py` from the recorded columns (by default OpenVLA's 7-D layout: tool translation,
tool rotation, gripper, taken 3 frames ahead so the policy runs at 10 Hz). Actions are normalized by
their 1st..99th percentiles, binned, and become the tokens the model must predict after the prompt.
The loss is next-token cross-entropy on those tokens, and the log also tracks how many action tokens
are predicted exactly and the mean error of the decoded actions.

A run directory holds `config.json` (the config plus what it resolved to: session, engine version,
model), `dataset_statistics.json` (the normalization needed to turn predictions back into actions),
`train_log.jsonl`, and a checkpoint directory every `save.every_steps` plus `final/`.
