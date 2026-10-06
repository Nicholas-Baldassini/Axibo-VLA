"""
Fine-tune SmolVLA on a recorded session.

    python train/smolvla/finetune.py --config smolvla      the run described by train/config/smolvla.py

Needs lerobot's SmolVLA extras, which pin a newer transformers than the OpenVLA checkpoint accepts,
so it runs in its own environment (requirements.txt here), not in the OpenVLA one.

The loop is lerobot's SmolVLA fine-tune (flow-matching loss from policy.forward, its AdamW and
cosine schedule, the frozen VLM and trained action expert of the pretrained config) written in the
shape of train/finetune.py, so a run looks the same on disk:

    train/runs/<config>_<date>_<time>/
        config.json               the training config plus what it resolved to (session, dataset version, model)
        dataset_statistics.json   the mean/std that normalize state and actions
        train_log.jsonl           one record per logged step
        step_<n>/, final/         the policy (config.json, model.safetensors), its pre/post-processors, metrics.json

A checkpoint loads with SmolVLAPolicy.from_pretrained(<dir>) and make_pre_post_processors(cfg, pretrained_path=<dir>).
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader

TRAIN_DIR = Path(__file__).resolve().parents[1]
ROOT = TRAIN_DIR.parent
sys.path[0] = str(TRAIN_DIR)      # train/ in place of this directory, so `data` is train/data.py and this package is `smolvla`

from config import load_config                        # noqa: E402
from finetune import make_run_dir, seed_everything    # noqa: E402
from model import resolve_device                      # noqa: E402
from smolvla.data import SmolVLASamples               # noqa: E402


def load_policy(model_cfg, action_cfg, data, device):
    """The pretrained SmolVLA with its inputs and outputs re-declared for this session."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(model_cfg["checkpoint"])     # the checkpoint's config.json says type: smolvla
    cfg.input_features, cfg.output_features = data.input_features, data.output_features
    cfg.chunk_size, cfg.n_action_steps = action_cfg["chunk_size"], action_cfg["n_action_steps"]
    cfg.device = str(device)
    policy = SmolVLAPolicy.from_pretrained(model_cfg["checkpoint"], config=cfg).to(device)
    pre, post = make_pre_post_processors(cfg, dataset_stats=data.stats)
    return policy, pre, post


def save_checkpoint(policy, pre, post, run_dir, name, metrics=None):
    out = run_dir / name
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)
    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    return out


def main(config_name):
    config = load_config(config_name)
    seed_everything(config["seed"])
    model_cfg, optim_cfg, save_cfg, log_cfg = config["model"], config["optim"], config["save"], config["log"]
    device = resolve_device(model_cfg["device"])

    # ---- data
    data = SmolVLASamples(config, ROOT / "datasets")
    print(f"session: {data.session}  (dataset {data.repo_id}, engine version {data.metadata.get('project_version', '?')})")
    print(f"  {data.n_episodes} episodes, {len(data)} frames at {data.fps} fps; cameras "
          + ", ".join(f"{old.split('.')[-1]} -> {new.split('.')[-1]}" for old, new in data.cameras.items()))
    print(f"  action ({len(data.action_names)}): {' '.join(data.action_names)} from {config['action']['key']!r}, "
          f"chunks of {data.chunk_size} frames = {data.chunk_size / data.fps:.2f} s")
    for key in ("mean", "std"):
        print(f"    {key}  " + "  ".join(f"{v:+.4f}" for v in np.asarray(data.stats['action'][key]).ravel()))
    print(data.describe_sample(0))

    # ---- model
    policy, pre, post = load_policy(model_cfg, config["action"], data, device)
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    pcfg = policy.config
    print(f"model: {model_cfg['checkpoint']} on {device}; {total / 1e6:.0f}M parameters, {trainable / 1e6:.0f}M trainable "
          f"(vision encoder frozen: {pcfg.freeze_vision_encoder}, expert only: {pcfg.train_expert_only}); "
          f"{pcfg.num_vlm_layers} VLM layers, images letterboxed to {pcfg.resize_imgs_with_padding}")

    # ---- run directory
    run_dir = make_run_dir(save_cfg, config_name)
    resolved = {
        "config_name": config_name,
        "config": config,
        "session": str(data.session),
        "dataset": {"repo_id": data.repo_id, "engine_version": data.metadata.get("project_version"), "episodes": data.n_episodes, "frames": len(data)},
        "action_names": data.action_names,
        "model": {"checkpoint": model_cfg["checkpoint"], "type": "smolvla", "device": str(device), "parameters": total, "trainable": trainable},
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=1, default=str))
    (run_dir / "dataset_statistics.json").write_text(json.dumps(data.stats_json(), indent=1))
    print(f"run: {run_dir}")

    # ---- training
    from lerobot.optim.optimizers import AdamWConfig
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    loader = DataLoader(
        data, batch_size=optim_cfg["batch_size"], shuffle=True, drop_last=False,
        num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0,
        pin_memory=device.type == "cuda",
    )
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    max_steps, log_every, save_every = optim_cfg["max_steps"], log_cfg["every_steps"], save_cfg["every_steps"]
    optimizer = AdamWConfig(lr=optim_cfg["lr"], betas=tuple(optim_cfg["betas"]), eps=optim_cfg["eps"],
                            weight_decay=optim_cfg["weight_decay"], grad_clip_norm=optim_cfg["grad_clip"]).build(trainable_params)
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=optim_cfg["warmup_steps"], num_decay_steps=optim_cfg["decay_steps"],
        peak_lr=optim_cfg["lr"], decay_lr=optim_cfg["decay_lr"]).build(optimizer, max_steps)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if model_cfg["use_amp"] and device.type == "cuda" else nullcontext()
    accumulation = optim_cfg["grad_accumulation_steps"]

    policy.train()
    log_file = open(run_dir / "train_log.jsonl", "a")
    window = {"loss": [], "grad_norm": []}
    step, micro, t_start, t_log, samples = 0, 0, time.time(), time.time(), 0
    skipped = 0           # optimizer steps dropped because the gradient was not finite
    last_record = None    # the most recent log record, stored alongside each checkpoint
    while step < max_steps:
        for batch in loader:
            batch = pre(batch)             # tokenizes the instruction, normalizes state and actions, moves to the device
            with autocast:
                loss, _ = policy.forward(batch)
            (loss / accumulation).backward()
            micro += 1
            samples += len(batch["action"])
            window["loss"].append(loss.item())
            if micro % accumulation:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg["grad_clip"])
            window["grad_norm"].append(float(grad_norm))
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % log_every == 0 or step == max_steps:
                now = time.time()
                rate = log_every / max(now - t_log, 1e-6)
                record = {
                    "step": step,
                    "loss": float(np.mean(window["loss"])),
                    "loss_max": float(np.max(window["loss"])),
                    "grad_norm": float(np.mean(window["grad_norm"])),
                    "grad_norm_max": float(np.max(window["grad_norm"])),
                    "step_per_s": round(rate, 3),
                    "lr": optimizer.param_groups[0]["lr"],
                    "samples": samples,
                    "elapsed_s": round(now - t_start, 1),
                    "skipped_steps": skipped,
                }
                last_record = record
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                eta = timedelta(seconds=int((max_steps - step) / max(rate, 1e-6)))
                print(f"step {step}/{max_steps}  loss {record['loss']:.4f}  grad {record['grad_norm']:.2f}  "
                      f"lr {record['lr']:.1e}  {rate:.2f} step/s  eta {eta}" + (f"  skipped {skipped}" if skipped else ""))
                window = {"loss": [], "grad_norm": []}
                t_log = now

            if save_every and step % save_every == 0 and step < max_steps:
                print(f"saved {save_checkpoint(policy, pre, post, run_dir, f'step_{step:06d}', metrics=last_record)}")
            if step >= max_steps:
                break

    print(f"saved {save_checkpoint(policy, pre, post, run_dir, 'final', metrics=last_record)}")
    log_file.close()
    print(f"done: {step} steps, {samples} samples, {timedelta(seconds=int(time.time() - t_start))}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on a recorded session.")
    parser.add_argument("--config", default="smolvla", help="name of a file in train/config/ (default: smolvla)")
    main(parser.parse_args().config)
