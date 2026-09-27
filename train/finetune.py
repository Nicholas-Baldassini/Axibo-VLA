"""
Fine-tune OpenVLA on a recorded session with LoRA.

    python train/finetune.py --config openvla     the run described by train/config/openvla.py
    python train/finetune.py --config smoke       pipeline test with a tiny random model (runs on a laptop)

The loop follows OpenVLA's own vla-scripts/finetune.py: next-token loss on the action tokens, and the
action token accuracy and L1 error logged alongside it. Data comes straight from the LeRobot session
(data.py), no RLDS conversion.

A run writes train/runs/<config>_<date>_<time>/
    config.json               the training config plus what it resolved to (session, dataset version, model)
    dataset_statistics.json   per-dimension action statistics, needed to un-normalize predictions
    train_log.jsonl           one record per logged step
    step_<n>/, final/         LoRA adapter + processor + statistics; final/merged/ when save.merged
"""

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(TRAIN_DIR))

from actions import ActionTokenizer                                    # noqa: E402
from config import load_config                                         # noqa: E402
from data import OpenVLASamples, collate                               # noqa: E402
from model import add_lora, count_parameters, load_base, load_processor, resolve_device, resolve_dtype  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(save_cfg, config_name):
    root = Path(save_cfg["root"])
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir, n = root / f"{config_name}_{stamp}", 1
    while run_dir.exists():
        n += 1
        run_dir = root / f"{config_name}_{stamp}_{n}"
    run_dir.mkdir()
    return run_dir


def action_metrics(logits, labels, n_patches, action_tokenizer, n_dims):
    """Action token accuracy and L1 error (normalized units) of a batch, as OpenVLA logs them.

    The model inserts the image patches after the first text token, so the logit predicting text
    token j (j >= 1) sits at position n_patches + j - 1.

    The L1 also comes back per action dimension. A single mean hides which part of the action is
    being learned: translation, rotation and the gripper are on different scales and converge at
    different rates, and a policy can look healthy on average while one dimension is not learned.
    """
    preds = logits[:, n_patches:-1].argmax(dim=-1)
    truth = labels[:, 1:].to(preds.device)
    mask = truth > action_tokenizer.action_token_begin_idx
    total = int(mask.sum())
    if total == 0:
        return float("nan"), float("nan"), np.full(n_dims, np.nan)
    accuracy = float(((preds == truth) & mask).sum()) / total
    p = action_tokenizer.decode_token_ids_to_actions(preds[mask].cpu().numpy())
    t = action_tokenizer.decode_token_ids_to_actions(truth[mask].cpu().numpy())
    error = np.abs(p - t)
    # Every sample contributes exactly n_dims action tokens, in order, so this reshape lines the
    # errors up by dimension. A truncated batch would not divide evenly; report only the mean then.
    per_dim = error.reshape(-1, n_dims).mean(axis=0) if total % n_dims == 0 else np.full(n_dims, np.nan)
    return accuracy, float(error.mean()), per_dim


def save_checkpoint(model, processor, stats, run_dir, name, merged=False, metrics=None):
    out = run_dir / name
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    processor.save_pretrained(adapter)
    (adapter / "dataset_statistics.json").write_text(json.dumps(stats, indent=1))
    # How this checkpoint was doing when it was written. dataset_statistics.json next to it holds
    # the action normalization, not the training metrics, so without this a checkpoint on its own
    # says nothing about which step it is or whether it was any good, and picking the best one
    # means going back to the run's train_log.jsonl.
    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if merged:
        # merge_and_unload folds the adapter into the base weights in place, so this is only done last.
        base = model.merge_and_unload()
        base.save_pretrained(out / "merged")
        processor.save_pretrained(out / "merged")
    return out


def main(config_name):
    config = load_config(config_name)
    seed_everything(config["seed"])
    model_cfg, optim_cfg, save_cfg, log_cfg = config["model"], config["optim"], config["save"], config["log"]
    device = resolve_device(model_cfg["device"])
    dtype = resolve_dtype(model_cfg["dtype"], device)
    if config["preview"]["every_steps"]:
        print("preview.every_steps is set, but rollout previews are not wired up yet (they come with the evaluation code); ignoring")

    # ---- data
    processor = load_processor(model_cfg["checkpoint"])
    tokenizer = processor.tokenizer
    action_tokenizer = ActionTokenizer(tokenizer.vocab_size, bins=256)
    data = OpenVLASamples(config, processor, action_tokenizer, ROOT / "datasets")
    stats = data.stats
    h, w = data.image_shape[:2]
    print(f"session: {data.session}  (dataset {data.repo_id}, engine version {data.metadata.get('project_version', '?')})")
    print(f"  {data.n_episodes} episodes, {len(data)} frames at {data.fps} fps; image {data.image_key} {w}x{h} -> 224x224")
    stride = config["action"]["stride"]
    print(f"  action ({len(data.action_names)}): {' '.join(data.action_names)}; stride {stride} frames = {data.fps / stride:.1f} Hz")
    for key in ("q01", "q99"):
        print(f"    {key}  " + "  ".join(f"{v:+.4f}" for v in stats[key]))
    print("    normalized: " + " ".join("yes" if m else "no" for m in stats["mask"]))
    print(data.describe_sample(0))

    # ---- model
    base = load_base(model_cfg, device, dtype)
    n_patches = base.vision_backbone.featurizer.patch_embed.num_patches
    stats_key = data.repo_id
    base.norm_stats = {stats_key: {"action": stats}}          # what predict_action un-normalizes with
    base.config.norm_stats = base.norm_stats
    model = add_lora(base, config["lora"])
    total, trainable = count_parameters(model)
    kind = "tiny random init" if model_cfg["tiny"] else ("4-bit" if model_cfg["load_in_4bit"] else str(dtype).split(".")[-1])
    print(f"model: {Path(model_cfg['checkpoint']).name} ({kind}) on {device}; {total / 1e6:.1f}M parameters, "
          f"{trainable / 1e6:.2f}M trainable (LoRA rank {config['lora']['rank']}); {n_patches} image patches")

    # ---- run directory
    run_dir = make_run_dir(save_cfg, config_name)
    resolved = {
        "config_name": config_name,
        "config": config,
        "session": str(data.session),
        "dataset": {"repo_id": data.repo_id, "engine_version": data.metadata.get("project_version"), "episodes": data.n_episodes, "frames": len(data)},
        "action_names": data.action_names,
        "norm_stats_key": stats_key,
        "model": {"checkpoint": model_cfg["checkpoint"], "device": str(device), "dtype": str(dtype), "parameters": total, "trainable": trainable},
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=1, default=str))
    (run_dir / "dataset_statistics.json").write_text(json.dumps({stats_key: {"action": stats, "num_transitions": len(data), "num_trajectories": data.n_episodes}}, indent=1))
    print(f"run: {run_dir}")

    # ---- training
    loader = DataLoader(
        data, batch_size=optim_cfg["batch_size"], shuffle=True, drop_last=False,
        num_workers=config["data"]["num_workers"], persistent_workers=config["data"]["num_workers"] > 0,
        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"],
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and dtype == torch.bfloat16 else nullcontext()
    accumulation = optim_cfg["grad_accumulation_steps"]
    max_steps, log_every, save_every = optim_cfg["max_steps"], log_cfg["every_steps"], save_cfg["every_steps"]
    model_dtype = next(model.parameters()).dtype


    # TRAIN LOOP ------


    model.train()
    log_file = open(run_dir / "train_log.jsonl", "a")
    window = {"loss": [], "accuracy": [], "l1": [], "l1_dims": [], "grad_norm": []}
    step, micro, t_start, t_log, samples = 0, 0, time.time(), time.time(), 0
    skipped = 0     # optimizer steps dropped because the gradient was not finite
    last_record = None    # the most recent log record, stored alongside each checkpoint
    while step < max_steps:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, dtype=model_dtype)
            input_ids, attention_mask, labels = (batch[k].to(device) for k in ("input_ids", "attention_mask", "labels"))
            with autocast:
                output = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
            (output.loss / accumulation).backward()
            micro += 1
            samples += len(input_ids)

            accuracy, l1, l1_dims = action_metrics(output.logits.detach(), labels, n_patches,
                                                   action_tokenizer, len(data.action_names))
            window["loss"].append(output.loss.item())
            window["accuracy"].append(accuracy)
            window["l1"].append(l1)
            window["l1_dims"].append(l1_dims)
            if micro % accumulation:
                continue

            # The 4-bit matmuls compute in float16 and nothing scales the loss, so a single bad
            # batch can hand back an inf or nan gradient; without this guard the optimizer writes
            # it into the adapter and every step afterwards is nan. Clipping also keeps an ordinary
            # loss spike from moving the adapter far enough to destabilize the run.
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, optim_cfg["grad_clip"])
            window["grad_norm"].append(float(grad_norm))
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % log_every == 0 or step == max_steps:
                now = time.time()
                rate = log_every / max(now - t_log, 1e-6)
                record = {
                    "step": step,
                    "loss": float(np.mean(window["loss"])),
                    # The worst step in the window, not just its mean: a mean over 20 steps hides
                    # the spike that precedes a divergence, which is how the first run's collapse
                    # went unnoticed until the loss was already nan.
                    "loss_max": float(np.max(window["loss"])),
                    "action_accuracy": float(np.nanmean(window["accuracy"])),
                    "action_l1": float(np.nanmean(window["l1"])),
                    "action_l1_per_dim": dict(zip(data.action_names,
                                                  np.nanmean(window["l1_dims"], axis=0).round(5).tolist())),
                    # The earliest warning there is: the gradient norm climbs well before the loss
                    # does, and its maximum is what the clip is there to catch.
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
                print(f"step {step}/{max_steps}  loss {record['loss']:.4f}  acc {record['action_accuracy']:.3f}  "
                      f"l1 {record['action_l1']:.3f}  {rate:.2f} step/s  eta {eta}"
                      + (f"  skipped {skipped}" if skipped else ""))
                window = {"loss": [], "accuracy": [], "l1": [], "l1_dims": [], "grad_norm": []}
                t_log = now

            if save_every and step % save_every == 0 and step < max_steps:
                print(f"saved {save_checkpoint(model, processor, stats, run_dir, f'step_{step:06d}', metrics=last_record)}")
            if step >= max_steps:
                break

    merged = save_cfg["merged"] and not model_cfg["load_in_4bit"]
    if save_cfg["merged"] and not merged:
        print("save.merged skipped: a 4-bit base model cannot be merged; load the adapter on the full checkpoint instead")
    print(f"saved {save_checkpoint(model, processor, stats, run_dir, 'final', merged=merged, metrics=last_record)}")
    log_file.close()
    print(f"done: {step} steps, {samples} samples, {timedelta(seconds=int(time.time() - t_start))}")
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA on a recorded session.")
    parser.add_argument("--config", default="openvla", help="name of a file in train/config/ (default: openvla)")
    main(parser.parse_args().config)
