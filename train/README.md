# Training

Fine-tuning OpenVLA on sessions recorded by the data engine in `source/`.

```
train/
├── config/            one config per run type; openvla.py is the real run, smoke.py a laptop-sized pipeline test
├── checkpoints/       downloaded base models (not committed)
├── runs/              one directory per training run (not committed)
├── download.py        one command for a new machine: base checkpoint, a run, a session, from Hugging Face
├── upload.py          a run to Hugging Face, as a version beside earlier ones (sessions: source/upload.py)
├── infer.py           the policy in the simulator, from a small GUI
├── finetune.py        the training loop
├── data.py            LeRobot session -> (image, instruction, action tokens) samples
├── actions.py         action vector from the recorded columns, normalization, action tokens
└── model.py           model loading, tiny stand-in, LoRA
```

```
python train/download.py                   # the base checkpoint, once
python train/finetune.py --config smoke    # pipeline test, runs on a laptop in a minute
python train/finetune.py --config openvla  # the real fine-tune, on the GPU machine
```

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

On the Mac only the small files (about 2.5 MB, everything except the weight shards) were fetched,
because the disk had 16 GB free. The script checks free space before downloading and refuses without
a 5 GB margin; rerun it on the Linux machine, or after clearing space, and it fetches only the shards.
The small files are enough for the smoke config: it builds a random-init miniature of the same
architecture from the checkpoint's own model code, tokenizer and processor.

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
