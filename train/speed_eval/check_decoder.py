"""
Exactness and speed of the Decoder (decoding.py) against the checkpoint's own generate(), on
recorded frames. Every decoder variant must produce the same 7 token ids as generate() on every
frame; this is the test speed_eval.py runs before trusting an optimisation, and it can be run on
its own:

    python train/speed_eval/check_decoder.py <run or checkpoint> [--frames 40] [--modes greedy,draft,greedy-static,draft-static]

Frames are taken in order along one episode (so the draft, the previous action, is what it would
be at inference), then at random, and the report gives per mode: agreement with generate(),
forward passes per action, and the latency median/p99.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1]
sys.path[0:0] = [str(TRAIN_DIR), str(TRAIN_DIR.parent / "source")]

import infer                                   # noqa: E402
import observe                                 # noqa: E402
from data import prompt_ids                    # noqa: E402
from speed_eval.decoding import Decoder        # noqa: E402


def local_session(session):
    """The run records the session where it trained; on this machine it is datasets/<name> (train/download.py)."""
    session = Path(session)
    local = TRAIN_DIR.parent / "datasets" / session.name
    if (session / "dataset" / "meta" / "info.json").exists():
        return session
    if (local / "dataset" / "meta" / "info.json").exists():
        return local
    sys.exit(f"the run's session is not on this machine: {session} (nor {local})")


def frames_of(policy, session, n):
    """[(canvas, instruction)] for the first n frames of episode 0, stepping 3 (the control stride), then random frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out = []
    with torch.device("cpu"):        # once Genesis is up, torch's default device is cuda:0 and the dataset's tensors would land there
        ds = LeRobotDataset("piper_stack", root=str(session / "dataset"), video_backend="pyav")
        ep = ds.hf_dataset["episode_index"]
        ep = ep.cpu().numpy() if hasattr(ep, "cpu") else np.asarray(ep)
        first = np.flatnonzero(ep == 0)
        rows = list(first[: n // 2 * 3: 3]) + list(np.random.default_rng(0).choice(len(ds), n - min(n // 2, len(first[::3])), replace=False))
        for row in rows:
            item = ds[int(row)]
            images = {c: (item[f"observation.images.{c}"].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in policy.cameras}
            out.append((policy.compose(images), item["task"]))
    return out


def inputs(policy, canvas, instruction):
    with torch.device("cpu"):
        pixel_values = policy.processor.image_processor(observe.eval_view(canvas, policy.view), return_tensors="pt")["pixel_values"]
    return (torch.tensor([prompt_ids(policy.tokenizer, instruction)], device=policy.device),
            pixel_values.to(policy.device, policy.input_dtype))


def reference_tokens(policy, input_ids, pixel_values):
    n = len(policy.names)
    with torch.inference_mode():
        generated = policy.model.generate(input_ids=input_ids, pixel_values=pixel_values, max_new_tokens=n, do_sample=False)
    return generated[0, -n:].cpu().numpy()


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


def references(policy, frames, log=print):
    """generate()'s tokens on every frame, and its latency: what every decoder is held to."""
    ref, ref_ms = [], []
    for canvas, text in frames:
        ids, px = inputs(policy, canvas, text)
        toks, ms = timed(lambda: reference_tokens(policy, ids, px))
        ref.append(toks); ref_ms.append(ms)
    report = {"ms_p50": float(np.median(ref_ms[2:])), "ms_p99": float(np.percentile(ref_ms[2:], 99)), "passes": len(policy.names)}
    log(f"  generate(): {report['ms_p50']:.0f} ms median, {report['ms_p99']:.0f} ms p99 over {len(frames)} frames")
    return ref, report


def check(policy, frames, ref, name, tokens_fn, forward_calls=None, log=print):
    """One decoding (tokens_fn(input_ids, pixel_values) -> token ids) against the reference tokens:
    agreement, forward passes per action (forward_calls() after each call) and latency."""
    agree, ms_all, passes = [], [], []
    for k, (canvas, text) in enumerate(frames):
        ids, px = inputs(policy, canvas, text)
        toks, ms = timed(lambda: tokens_fn(ids, px))
        agree.append(bool(np.array_equal(np.asarray(toks), ref[k])))
        if k >= 2:                       # the first calls include compilation / warm-up
            ms_all.append(ms)
            passes.append(forward_calls() if forward_calls else len(policy.names))
    report = {"frames": len(frames), "agreement": float(np.mean(agree)), "disagreements": [k for k, a in enumerate(agree) if not a],
              "ms_p50": float(np.median(ms_all)), "ms_p99": float(np.percentile(ms_all, 99)), "passes": float(np.mean(passes))}
    log(f"  {name:14s}: agreement {report['agreement'] * 100:.1f}%  {report['ms_p50']:.0f} ms median, {report['ms_p99']:.0f} ms p99, "
        f"{report['passes']:.2f} forward passes per action" + (f"  DISAGREES on frames {report['disagreements']}" if report["disagreements"] else ""))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("run")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--modes", default="greedy,draft,greedy-static,draft-static")
    args = parser.parse_args()
    bundle = infer.load_run(args.run)
    policy = infer.Policy(bundle["run"], bundle["adapter"], bundle["stats"])
    frames = frames_of(policy, local_session(bundle["session"]), args.frames)
    print(f"{len(frames)} frames; model {bundle['run']['config']['model']['dtype']}, 4-bit {bundle['run']['config']['model']['load_in_4bit']}")
    ref, _ = references(policy, frames)
    for mode in args.modes.split(","):
        dec = Decoder(policy, static_cache=mode.endswith("-static"))
        dec.draft = mode.startswith("draft")
        check(policy, frames, ref, mode, dec.tokens, lambda: dec.forward_calls)
        dec.close()


if __name__ == "__main__":
    main()
