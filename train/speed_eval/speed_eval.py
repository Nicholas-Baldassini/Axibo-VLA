"""
Speed evaluation of the OpenVLA policy: the same seeded rollouts under every optimisation stack
the config lists, with the policy's compute time on the simulator's clock.

    python train/speed_eval/speed_eval.py --config speed_eval
    python train/speed_eval/speed_eval.py --config speed_eval --timing lockstep --optims none,b4_infer

For each stack: the checkpoint is loaded the way the stack asks, the stack's decoding is checked
token for token against generate() on recorded frames (check_decoder.py), then every (source,
destination) pair of the config is run for its number of rollouts on the fixed layouts, through
controller.py. One folder per evaluation under train/evals/speed_eval/<time>/:

    config.json              the config and the machine
    <stack>/results.json     one record per rollout: outcome, latencies, smoothness metrics
    <stack>/check.json       the exactness check
    <stack>/traces/*.npz     joint and tool traces of every rollout, for plots
    <stack>/videos/          the policy's view of the first rollout
    summary.md, summary.json the table across stacks: success, latency p50/p99, action rate, stall, jerk
"""

import argparse
import importlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

TRAIN_DIR = Path(__file__).resolve().parents[1]
ROOT = TRAIN_DIR.parent
sys.path[0] = str(TRAIN_DIR)
sys.path.insert(1, str(ROOT / "source"))

import infer                                                    # noqa: E402
import observe                                                  # noqa: E402
from data import prompt_ids                                     # noqa: E402
from actions import unnormalize_actions                         # noqa: E402
from speed_eval import check_decoder                            # noqa: E402
from speed_eval.controller import Controller                    # noqa: E402
from speed_eval.decoding import Decoder                         # noqa: E402


def load_config(name):
    """train/config/<name>.py's CONFIG, by path: `config` as a module name is source/config.py once infer is imported."""
    spec = importlib.util.spec_from_file_location(f"speed_eval_config_{name}", TRAIN_DIR / "config" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return json.loads(json.dumps(module.CONFIG))        # a plain copy, lists for tuples


# ---- the stacks -----------------------------------------------------------------------------------

def load_stack(names, settings):
    """[Optimisation] for a config entry such as ("b16_infer", "torch_compile_infer"); () or "none" is vanilla."""
    if isinstance(names, str):
        names = () if names.lower() == "none" else (names,)
    stack = []
    for name in names:
        module = name[:-3] if name.endswith(".py") else name
        if not module.endswith("_infer"):
            module += "_infer"
        cls = importlib.import_module(f"speed_eval.{module}").OPTIMISATION
        stack.append(cls(settings.get(cls.name, {})))
    names_in = {o.name for o in stack}
    for o in stack:
        clash = names_in & set(o.conflicts)
        if clash:
            sys.exit(f"{o.name} cannot be stacked with {sorted(clash)}")
    return stack


def stack_name(stack):
    return "+".join(o.name for o in stack) or "none"


class DecodedPolicy:
    """infer.Policy's predict() with its generate() replaced by a Decoder; one per engine."""

    def __init__(self, policy, decoder):
        self.policy, self.decoder = policy, decoder
        self.names, self.cameras, self.view, self.stride = policy.names, policy.cameras, policy.view, policy.stride

    def compose(self, images):
        return self.policy.compose(images)

    def reset(self):
        self.decoder.last_tokens = None

    def predict(self, canvas, instruction):
        p, torch_ = self.policy, self.policy.torch
        t0 = time.perf_counter()
        with torch_.device("cpu"):
            pixel_values = p.processor.image_processor(observe.eval_view(canvas, p.view), return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(p.device, p.input_dtype)
        input_ids = torch_.tensor([prompt_ids(p.tokenizer, instruction)], device=p.device)
        t1 = time.perf_counter()
        tokens = self.decoder.tokens(input_ids, pixel_values)
        t2 = time.perf_counter()
        action = np.asarray(unnormalize_actions(p.action_tokenizer.decode_token_ids_to_actions(tokens), p.stats), dtype=float)
        # where the time went, for the log: preprocessing (crop, resize, normalize, tokenize), the model, the rest
        self.last_ms = {"preprocess": round((t1 - t0) * 1000, 1), "model": round((t2 - t1) * 1000, 1), "post": round((time.perf_counter() - t2) * 1000, 1),
                        "passes": self.decoder.forward_calls}
        return action


def build(bundle, stack):
    """(policy, predictor_factory, controller options) for one stack."""
    run = json.loads(json.dumps(bundle["run"]))           # a copy: each stack sets its own model config
    model_cfg = dict(run["config"]["model"])
    for o in stack:
        model_cfg = o.model_config(model_cfg)
    run["config"]["model"] = model_cfg
    policy = infer.Policy(run, bundle["adapter"], bundle["stats"])
    for o in stack:
        o.prepare(policy)
    dec_opts, ctrl_opts = {}, {}
    for o in stack:
        dec_opts.update(o.decoder_options())
        ctrl_opts.update(o.controller_options())
    instances = []                                        # every DecodedPolicy built, to close their caches afterwards
    if not dec_opts:
        factory = lambda: policy                          # generate() itself (vanilla, or only the precision changed)
    else:
        def factory():
            if instances and ctrl_opts.get("engines", 1) == 1:
                return instances[0]                       # one decoder serves the check and the single engine in turn
            dec = Decoder(policy, static_cache=dec_opts.get("static_cache", False))
            dec.draft = dec_opts.get("draft", False)
            for o in stack:
                o.prepare_decoder(dec)
            instances.append(DecodedPolicy(policy, dec))
            return instances[-1]
    return policy, factory, ctrl_opts, model_cfg, instances


def tokens_fn(probe, policy):
    """(tokens(input_ids, pixel_values), forward_calls()) of a predictor: its decoder's, or generate()'s."""
    if isinstance(probe, DecodedPolicy):
        return probe.decoder.tokens, lambda: probe.decoder.forward_calls
    return (lambda ids, px: check_decoder.reference_tokens(policy, ids, px)), None


# ---- the evaluation --------------------------------------------------------------------------------

def percentile(xs, p):
    return round(float(np.percentile(xs, p)) * 1000, 1) if len(xs) else None


def median_of(records, key):
    xs = [r[key] for r in records if r.get(key) is not None]
    return round(float(np.median(xs)), 3) if xs else None


def summarize(name, records, check):
    lat = [l for r in records for l in r["latencies_s"]]
    outcomes = Counter(r["outcome"] for r in records)
    by_pair = {}
    for r in records:
        k = f"{r['source']}>{r['destination']}"
        by_pair.setdefault(k, {"success": 0, "rollouts": 0, "held_out": r.get("held_out", False), "outcomes": Counter()})
        by_pair[k]["success"] += int(r["success"]); by_pair[k]["rollouts"] += 1; by_pair[k]["outcomes"][r["outcome"]] += 1
    for v in by_pair.values():
        v["outcomes"] = dict(v["outcomes"])
    grasp = [np.hypot(*r["grasp"]["tool_to_source_xy_m"]) * 100 for r in records if r.get("grasp")]
    release = [np.hypot(*r["release"]["tool_to_dest_xy_m"]) * 100 for r in records if r.get("release")]
    return {
        "stack": name, "rollouts": len(records), "success": sum(r["success"] for r in records),
        "by_pair": by_pair,
        "reached": sum(r["reached_source"] for r in records), "lifted_source": sum(r["source"] in r["lifted"] for r in records),
        "grasp_offset_cm_median": round(float(np.median(grasp)), 2) if grasp else None,      # tool to source at the first close
        "release_offset_cm_median": round(float(np.median(release)), 2) if release else None,  # tool to destination at the release
        "sim_s_to_success_median": median_of(records, "sim_s_to_success"),
        "actions_to_success_median": median_of(records, "actions_to_success"),
        "staleness_ms_p50": median_of(records, "staleness_ms_p50"),
        "latency_ms_max": max((r["latency_ms_max"] for r in records if r.get("latency_ms_max") is not None), default=None),
        "latency_ms_p50": percentile(lat, 50), "latency_ms_p99": percentile(lat, 99), "predictions": len(lat),
        "action_hz": round(float(np.mean([r["action_hz"] for r in records])), 2),
        "stall_frac": round(float(np.sum([r["stall_s"] for r in records]) / max(np.sum([r["sim_s"] for r in records]), 1e-6)), 3),
        "jerk_rms": round(float(np.mean([r["jerk_rms"] for r in records])), 1),
        "jerk_p99": round(float(np.mean([r["jerk_p99"] for r in records])), 1),
        "accel_rms": round(float(np.mean([r["accel_rms"] for r in records])), 2),
        "stops_per_s": round(float(np.mean([r["stops_per_s"] for r in records])), 2),
        "dv_boundary_p95": round(float(np.mean([r["dv_boundary_p95"] for r in records])), 3),
        "dv_within_p95": round(float(np.mean([r["dv_within_p95"] for r in records])), 3),
        "duty": round(float(np.mean([r["duty"] for r in records])), 3),
        "sim_s_mean": round(float(np.mean([r["sim_s"] for r in records])), 2),
        "outcomes": dict(outcomes.most_common()),
        "check_agreement": check.get("agreement"), "check_passes": check.get("passes"),
    }


def table(rows):
    cols = [("stack", "stack"), ("success", "success"), ("latency_ms_p50", "p50 ms"), ("latency_ms_p99", "p99 ms"),
            ("action_hz", "actions/s"), ("stall_frac", "stall"), ("duty", "duty"), ("stops_per_s", "stops/s"),
            ("jerk_rms", "jerk rms"), ("jerk_p99", "jerk p99"), ("dv_boundary_p95", "dv boundary"), ("dv_within_p95", "dv within"),
            ("sim_s_mean", "sim s"), ("check_agreement", "tokens = vanilla")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            if key == "success":
                v = f"{r['success']}/{r['rollouts']}"
            elif key == "check_agreement" and v is not None:
                v = f"{v * 100:.0f}%"
            cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison(rows):
    """The second table: what each stack did to the task, next to the vanilla row."""
    base = next((r for r in rows if r["stack"] == "none"), rows[0] if rows else None)
    pairs = sorted({p for r in rows for p in r.get("by_pair", {})})
    head = ["stack", "success", "vs vanilla", "speedup p50"] + pairs + ["reached", "lifted", "grasp off cm", "release off cm",
                                                                        "sim s to success", "actions to success", "staleness ms", "outcomes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        d_success = r["success"] - base["success"] if base else 0
        speedup = (base["latency_ms_p50"] / r["latency_ms_p50"]) if base and r.get("latency_ms_p50") and base.get("latency_ms_p50") else None
        cells = [r["stack"], f"{r['success']}/{r['rollouts']}", f"{d_success:+d}", f"{speedup:.2f}x" if speedup else ""]
        cells += [f"{r['by_pair'][p]['success']}/{r['by_pair'][p]['rollouts']}" if p in r.get("by_pair", {}) else "" for p in pairs]
        cells += [str(r.get(k, "")) if r.get(k) is not None else "" for k in
                  ("reached", "lifted_source", "grasp_offset_cm_median", "release_offset_cm_median", "sim_s_to_success_median",
                   "actions_to_success_median", "staleness_ms_p50")]
        cells.append(", ".join(f"{k} {v}" for k, v in r.get("outcomes", {}).items()))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(out, cfg, rows, total):
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    (out / "summary.md").write_text(
        f"# Speed evaluation, {cfg['timing']} timing, {total} rollouts per stack\n\n"
        "Latency is wall-clock per prediction (compose, preprocess, prefill, decode, un-normalize); stall is the share of "
        "simulated time with nothing to execute while a prediction was pending; duty the share of physics steps with the tool "
        "moving; jerk is joint-space RMS over moving steps (rad/s^3); dv is the p95 joint-speed change over one physics step "
        "at action landings vs elsewhere; 'tokens = vanilla' is the agreement of the stack's action tokens with the vanilla "
        "policy's on recorded frames.\n\n" + table(rows) + "\n\n## Task outcome per stack\n\n" + comparison(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Speed evaluation of the policy under optimisation stacks, on seeded rollouts.")
    parser.add_argument("--config", default="speed_eval", help="name of a file in train/config/ (default: speed_eval)")
    parser.add_argument("--optims", help="comma-separated stacks to run instead of the config's, '+' joins a stack: none,b4_infer,b16_infer+torch_compile_infer")
    parser.add_argument("--timing", choices=("realtime", "lockstep"))
    parser.add_argument("--combos", help='instead of the config\'s: "red_cube>red_cylinder:2,blue_cube>red_cube:1"')
    parser.add_argument("--check-frames", type=int, help="recorded frames for the exactness check (0 skips it)")
    parser.add_argument("--horizon", type=int, help="policy actions per rollout instead of the config's")
    parser.add_argument("--latency", type=float, help="fixed prediction latency in ms on the simulated clock instead of the measured one")
    parser.add_argument("--out")
    parser.add_argument("--stack", type=int, help="(internal) run only this index of the config's optims, in a child process")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.optims:
        cfg["optims"] = [tuple(s.split("+")) if s != "none" else () for s in args.optims.split(",")]
    if args.timing:
        cfg["timing"] = args.timing
    if args.combos:
        cfg["combos"] = [(p.split(">")[0], p.split(">")[1].split(":")[0], int(p.split(":")[1])) for p in args.combos.split(",")]
    if args.check_frames is not None:
        cfg["check"]["frames"] = args.check_frames
    if args.horizon:
        cfg["horizon"]["actions"] = args.horizon
    if args.latency is not None:
        cfg["control"]["latency"] = args.latency / 1000.0

    out = Path(args.out) if args.out else Path(cfg["out"]) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{cfg['timing']}"
    out.mkdir(parents=True, exist_ok=True)
    total = sum(n for _, _, n in cfg["combos"])

    if args.stack is None:
        # ---- the parent never touches the GPU: one child for the reference, one per stack (a 7B,
        # its compiled graphs and the renderer are released for certain when a process ends),
        # then the summary from what the children wrote.
        print(f"evaluation folder {out}\n{len(cfg['optims'])} stacks x {total} rollouts, timing {cfg['timing']}, "
              f"horizon {cfg['horizon']['actions']} actions / {cfg['horizon']['sim_s']} s", flush=True)
        for k in range(-1, len(cfg["optims"])):
            cmd = [sys.executable, __file__, "--config", args.config, "--stack", str(k), "--out", str(out)]
            for flag, value in (("--optims", args.optims), ("--timing", args.timing), ("--combos", args.combos)):
                if value:
                    cmd += [flag, str(value)]
            if args.check_frames is not None:
                cmd += ["--check-frames", str(args.check_frames)]
            if args.horizon:
                cmd += ["--horizon", str(args.horizon)]
            if args.latency is not None:
                cmd += ["--latency", str(args.latency)]
            code = subprocess.call(cmd)
            if code:
                print(f"   {'reference' if k < 0 else f'stack {k}'} failed with exit code {code}; see above", flush=True)
                if k < 0:
                    sys.exit(code)
            if k >= 0:        # the summary so far, so a long run can be read while it is going
                done = [json.loads((out / stack_name(load_stack(n, cfg["settings"])) / "summary.json").read_text())
                        for n in cfg["optims"][:k + 1] if (out / stack_name(load_stack(n, cfg["settings"])) / "summary.json").exists()]
                write_summary(out, cfg, done, total)
        rows = [json.loads((out / stack_name(load_stack(n, cfg["settings"])) / "summary.json").read_text())
                for n in cfg["optims"] if (out / stack_name(load_stack(n, cfg["settings"])) / "summary.json").exists()]
        write_summary(out, cfg, rows, total)
        print("\n" + table(rows) + "\n\n" + comparison(rows))
        print(f"\nwritten {out}")
        return

    bundle = infer.load_run(cfg["run"])
    session_cfg = bundle["session_cfg"]
    from config import instruction as make_instruction
    names = list(session_cfg["scene"]["objects"])
    for s, d, n in cfg["combos"]:
        if s not in names or d not in names:
            sys.exit(f"unknown object in combo {(s, d, n)}; objects are {names}")
        if n > len(cfg["seeds"]):
            sys.exit(f"combo {(s, d, n)} asks for more rollouts than there are seeds ({len(cfg['seeds'])})")
    template = session_cfg["task"]["instruction"]
    holdout = {tuple(p) for p in session_cfg["task"].get("holdout_pairs", [])}

    if args.stack < 0:
        # ---- the reference child: the vanilla policy's tokens on recorded frames, what every stack is held to
        (out / "config.json").write_text(json.dumps({
            "config": cfg, "created": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "checkpoint": str(bundle["checkpoint"]), "torch": torch.__version__,
        }, indent=1, default=str))
        print(f"checkpoint {bundle['checkpoint']}", flush=True)
        if cfg["check"]["frames"]:
            vanilla = infer.Policy(bundle["run"], bundle["adapter"], bundle["stats"])
            frames = check_decoder.frames_of(vanilla, check_decoder.local_session(bundle["session"]), cfg["check"]["frames"])
            print(f"\nreference: vanilla generate() on {len(frames)} recorded frames", flush=True)
            ref, ref_report = check_decoder.references(vanilla, frames, log=lambda s: print(" " + s, flush=True))
            np.savez(out / "reference.npz", tokens=np.array(ref), ms_p50=ref_report["ms_p50"], ms_p99=ref_report["ms_p99"])
        return

    # ---- a stack child
    stack = load_stack(cfg["optims"][args.stack], cfg["settings"])
    name = stack_name(stack)
    folder = out / name
    (folder / "traces").mkdir(parents=True, exist_ok=True)
    print(f"\n== {name}: " + (", ".join(o.describe() for o in stack) or "vanilla infer.py"), flush=True)
    policy, factory, ctrl_opts, model_cfg, instances = build(bundle, stack)
    print(f"   model {'4-bit' if model_cfg['load_in_4bit'] else model_cfg['dtype']}, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    # The simulator comes after the model (infer.py explains why) and before the decoders: Genesis
    # changes torch's global state (the default device), and a step compiled before that is
    # recompiled at its first call afterwards, in the middle of a timed rollout.
    c = cfg["control"]
    sim = infer.Simulator(session_cfg, backend=c["backend"], z_min=None if c["z_min"] < 0 else c["z_min"], gripper=c["gripper"])

    # ---- exactness: this stack's tokens against the vanilla reference, on the same frames
    check = {}
    if (out / "reference.npz").exists():
        reference = np.load(out / "reference.npz")
        frames = check_decoder.frames_of(policy, check_decoder.local_session(bundle["session"]), len(reference["tokens"]))
        probe = factory()
        fn, calls = tokens_fn(probe, policy)
        if hasattr(probe, "reset"):
            probe.reset()
        check = check_decoder.check(policy, frames, list(reference["tokens"]), name, fn, calls, log=lambda s: print("  " + s, flush=True))
        check["vanilla_ms_p50"] = float(reference["ms_p50"])
        del probe, fn, calls
    (folder / "check.json").write_text(json.dumps(check, indent=1))

    # ---- the rollouts
    control_hz = session_cfg["record"]["fps"] / policy.stride
    records = []
    for s, d, n in cfg["combos"]:
        instr = {"source": s, "destination": d, "text": make_instruction({"source": s, "destination": d, "instruction": template})}
        video = cfg.get("video", {"fps": 30, "camera": "front"})
        ctrl = Controller(sim, instr, factory, policy.compose, sim.cfg["success"], sim.cfg["scene"]["objects"], timing=cfg["timing"],
                          schedule=ctrl_opts.get("schedule", "sync"), engines=ctrl_opts.get("engines", 1),
                          control_hz=control_hz, horizon_actions=cfg["horizon"]["actions"], max_sim_s=cfg["horizon"]["sim_s"],
                          filter_factory=ctrl_opts.get("filter"), video_fps=video["fps"], video_camera=video["camera"], label=name,
                          latency=cfg["control"].get("latency"), hold=ctrl_opts.get("hold", cfg["control"].get("hold", "pose")))
        for i in range(n):
            layout = cfg["seeds"][i]
            rec, traces, frames = ctrl.run(layout, start_yaw=cfg["control"]["start_yaw"])
            rec["held_out"] = (s, d) in holdout
            records.append(rec)
            np.savez_compressed(folder / "traces" / f"{s}_on_{d}__layout{layout}.npz", t=traces["t"], q=traces["q"], tool=traces["tool"],
                                landed_step=traces["landed_step"], events=json.dumps(traces["events"]))
            # every rollout on video, on the simulated clock: the same scene under every stack, side by side
            from evaluate import write_video
            write_video(folder / "videos" / f"{s}_on_{d}__layout{layout}.mp4", frames, fps=video["fps"], gif=False)
            rec["video"] = f"{name}/videos/{s}_on_{d}__layout{layout}.mp4"
            (folder / "results.json").write_text(json.dumps(records, indent=1))
            lat = np.array(rec["latencies_s"]) * 1000
            print(f"   {instr['text']:<40s} layout {layout:2d}  {rec['outcome']:<20s} {rec['actions']:3d} actions in {rec['sim_s']:5.1f} s"
                  f"  latency p50 {np.median(lat) if len(lat) else float('nan'):4.0f} ms  max {lat.max() if len(lat) else float('nan'):5.0f}  "
                  f"stall {rec['stall_s']:5.1f} s  jerk {rec['jerk_rms']:7.0f}  "
                  f"dv boundary/within {rec['dv_boundary_p95']:.2f}/{rec['dv_within_p95']:.2f}", flush=True)
        ctrl.close()
    row = summarize(name, records, check)
    (folder / "summary.json").write_text(json.dumps(row, indent=1))
    print("   " + json.dumps({k: v for k, v in row.items() if k not in ("outcomes",)}), flush=True)
    for inst in instances:
        inst.decoder.close()


if __name__ == "__main__":
    main()
