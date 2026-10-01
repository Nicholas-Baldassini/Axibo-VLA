"""
Evaluate a fine-tuned policy in the simulator under a declared protocol.

    python train/evaluate.py train/runs/<run>/step_010000
    python train/evaluate.py train/runs/<run> --trials 20 --pairs "red_cylinder>blue_cube,blue_cube>red_cylinder" --viewer

Every ordered pair (the five trained on plus the held-out one) is run once on each of TRIALS
layouts. A layout is the simulator's reset number, so the seed list is the list of those numbers
and any layout is reproducible from its number alone: object positions, yaws and the start pose.
The pair is set by the instruction, so the reversal test is two rows of the table on the same
layouts. One folder per evaluation under train/evals/:

    metadata.json    the checkpoint and everything it depends on, the protocol, the seed list
    seeds.json       the layout numbers, in order
    results.json     one record per rollout: outcome, events, timings
    summary.md       the table: success per ordered pair, taxonomy, reversal pairs, held-out pair
    videos/          the front-camera view of the first failure of each type (and the first success)

Success is the recorder's own criterion (config.success, utils/episode.py): the source resting on
the destination within stack_xy_tol / stack_z_tol, the destination upright and at its rest height,
the third object undisturbed, and the whole scene still for hold_time. Here it is checked from the
moment the policy has released a grasped object; a rollout ends at the first completed hold.

Failure taxonomy, from events observed during the rollout, first match wins:
    never reached           the tool never came within REACH_XY of the source below REACH_Z
    wrong object grasped    an object other than the source was lifted
    missed grasp            reached the source, never lifted it
    knocked destination     the destination moved or tilted beyond the criterion
    stacked on wrong        the source ended resting on the third object
    grasped and dropped     the source was lifted and ended on the table
    disturbed third         the third object moved
    not settled             the stack looked right but never held still, or still in hand at the end
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(ROOT / "source"))

from infer import Z_MIN, Policy, Rollout, Simulator, load_run   # noqa: E402

# ---- protocol -----------------------------------------------------------------------------------
TRIALS = 10          # layouts per ordered pair; the seed list is 1..TRIALS unless --layouts is given
HORIZON = 300        # control steps per rollout: 30 s at 10 Hz; a demonstration takes about 10 s
REACH_XY = 0.04      # m: tool within this of the source in x and y, below REACH_Z, counts as reached
REACH_Z = 0.05       # m above the table
LIFT_Z = 0.02        # m above its rest height: the object counts as lifted
VIDEO_FPS = 10       # one frame per control step
# --------------------------------------------------------------------------------------------------


def all_pairs(names):
    return [(s, d) for s in names for d in names if s != d]


def parse_pairs(text, names):
    if text in (None, "all"):
        return all_pairs(names)
    out = []
    for item in text.split(","):
        s, d = item.split(">")
        if s not in names or d not in names:
            sys.exit(f"unknown object in pair {item!r}; objects are {names}")
        out.append((s, d))
    return out


def parse_layouts(text):
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def write_video(path, frames):
    """frames: list of (H, W, 3) uint8, one per control step, to an h264 mp4 and a GIF beside it
    (the GIF at half size, for embedding in the write-up)."""
    import av
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=VIDEO_FPS)
        stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        stream.options = {"crf": "23"}
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    h, w = frames[0].shape[:2]
    gif = [Image.fromarray(f).resize((w // 2, h // 2), Image.BILINEAR) for f in frames]
    gif[0].save(str(path.with_suffix(".gif")), save_all=True, append_images=gif[1:],
                duration=int(1000 / VIDEO_FPS), loop=0, optimize=True)


def rollout(sim, policy, instruction, layout, horizon, sim_steps, crit, specs, start_yaw="home"):
    """One episode on one layout. Returns (record, front-camera frames)."""
    from utils.frames import object_height
    from utils.episode import rot_angle_deg, tilt_deg

    sim.episode = layout - 1
    sim.reset()
    if start_yaw == "aligned":
        sim.align_yaw(instruction["source"])
    names = list(sim.objects)
    src, dst = instruction["source"], instruction["destination"]
    s, d = names.index(src), names.index(dst)
    third = [o for o in range(len(names)) if o not in (s, d)]
    height = {o: object_height(specs[o]) for o in names}
    poses = lambda: (np.stack([np.asarray(sim._to_array(e.get_pos()))[0] for e in sim.objects.values()]),
                     np.stack([np.asarray(sim._to_array(e.get_quat()))[0] for e in sim.objects.values()]))
    spawn_pos, _ = poses()
    dt_step = sim_steps * sim.cfg["sim"]["dt"]

    frames = []
    reached = False
    lifted = set()               # objects seen lifted while the gripper was commanded closed
    released_after_grasp = False
    was_closed = False
    grasp_event, release_event = None, None      # where the tool was when the gripper closed on / let go of the source
    hold_start, hold_ref = None, None
    outcome, ended_at = None, horizon
    t0 = time.time()
    for step in range(horizon):
        obs = sim.observation()
        canvas = policy.compose(obs["images"])      # exactly what the policy is given (before the centre crop)
        frames.append(canvas)
        action = policy.predict(canvas, instruction["text"])
        sim.apply(action, sim_steps)
        pos, quat = poses()
        tool = obs["tcp"]
        if not reached and tool[2] < REACH_Z and np.all(np.abs(tool[:2] - (pos[s][:2] - sim.table_offset[:2])) < REACH_XY):
            reached = True
        tool_xy = tool[:2]
        if sim.cmd_grip < 0.5:
            if not was_closed and grasp_event is None:
                grasp_event = {"step": step, "tool_to_source_xy_m": [round(float(v), 4) for v in tool_xy - (pos[s][:2] - sim.table_offset[:2])],
                               "tool_z_m": round(float(tool[2]), 4)}
            for o in range(len(names)):
                if pos[o][2] - spawn_pos[o][2] > LIFT_Z:
                    lifted.add(o)
            was_closed = True
        elif was_closed:
            was_closed = False
            if s in lifted and release_event is None:
                release_event = {"step": step, "tool_to_dest_xy_m": [round(float(v), 4) for v in tool_xy - (pos[d][:2] - sim.table_offset[:2])],
                                 "tool_z_m": round(float(tool[2]), 4),
                                 "place_z_m": round(float(spawn_pos[d][2] - sim.table_offset[2] + (height[names[d]] + height[names[s]]) / 2), 4)}
            if lifted:
                released_after_grasp = True

        if released_after_grasp:      # the recorder's stacking test, from the first release on
            delta = pos[s] - pos[d]
            dz_expected = height[names[d]] / 2 + height[names[s]] / 2
            stacked = (abs(delta[0]) <= crit["stack_xy_tol"] and abs(delta[1]) <= crit["stack_xy_tol"]
                       and abs(delta[2] - dz_expected) <= crit["stack_z_tol"])
            tilt = tilt_deg(quat)
            dest_ok = tilt[d] <= crit["upright_tol_deg"] and abs(pos[d][2] - spawn_pos[d][2]) <= crit["rest_z_tol"]
            third_ok = all(np.linalg.norm(pos[o] - spawn_pos[o]) <= crit["disturb_tol"] and tilt[o] <= crit["upright_tol_deg"] for o in third)
            if stacked and dest_ok and third_ok:
                still = hold_ref is not None and (np.linalg.norm(pos - hold_ref[0], axis=-1).max() <= crit["still_pos_tol"]
                                                  and rot_angle_deg(quat, hold_ref[1]).max() <= crit["still_rot_tol_deg"])
                if not still:
                    hold_start, hold_ref = step, (pos.copy(), quat.copy())
                elif (step - hold_start) * dt_step >= crit["hold_time"]:
                    outcome, ended_at = "success", step + 1
                    break
            else:
                hold_start, hold_ref = None, None

    pos, quat = poses()
    tilt = tilt_deg(quat)
    if outcome is None:
        on_table = lambda o: abs(pos[o][2] - spawn_pos[o][2]) <= crit["stack_z_tol"] + 0.01
        resting_on = lambda o, base: (np.all(np.abs(pos[o][:2] - pos[base][:2]) <= 0.03)
                                      and abs(pos[o][2] - pos[base][2] - (height[names[base]] + height[names[o]]) / 2) <= 0.015)
        dest_moved = tilt[d] > crit["upright_tol_deg"] or np.linalg.norm(pos[d] - spawn_pos[d]) > crit["disturb_tol"]
        third_moved = any(np.linalg.norm(pos[o] - spawn_pos[o]) > crit["disturb_tol"] or tilt[o] > crit["upright_tol_deg"] for o in third)
        if not reached and s not in lifted:
            outcome = "never reached"
        elif lifted and s not in lifted:
            outcome = "wrong object grasped"
        elif s not in lifted:
            outcome = "missed grasp"
        elif dest_moved:
            outcome = "knocked destination"
        elif any(resting_on(s, o) for o in third):
            outcome = "stacked on wrong"
        elif on_table(s):
            outcome = "grasped and dropped"
        elif third_moved:
            outcome = "disturbed third"
        else:
            outcome = "not settled"
    record = {
        "layout": layout, "source": src, "destination": dst, "instruction": instruction["text"],
        "outcome": outcome, "success": outcome == "success", "steps": ended_at,
        "reached_source": reached, "lifted": sorted(names[o] for o in lifted),
        "grasp": grasp_event, "release": release_event,
        "final_source_to_dest_xy_m": [round(float(v), 4) for v in (pos[s][:2] - pos[d][:2])],
        "wall_s": round(time.time() - t0, 1),
    }
    return record, frames


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy per ordered pair over a fixed list of layouts.")
    parser.add_argument("run", help="a training run directory or one of its checkpoints")
    parser.add_argument("--trials", type=int, default=TRIALS, help=f"layouts per pair, seeds 1..N (default {TRIALS})")
    parser.add_argument("--layouts", help="explicit reset numbers instead of 1..trials, e.g. 1-10 or 1,4,28,49")
    parser.add_argument("--pairs", default="all", help='"all" or "src>dst,src>dst,..." with object names')
    parser.add_argument("--horizon", type=int, default=HORIZON, help=f"control steps per rollout (default {HORIZON})")
    parser.add_argument("--viewer", action="store_true", help="open the Genesis viewer window while evaluating")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--z-min", type=float, default=Z_MIN)
    parser.add_argument("--gripper", choices=("threshold", "raw"), default="threshold")
    parser.add_argument("--out", help="evaluation folder (default: train/evals/<time>_<run>_<checkpoint>)")
    parser.add_argument("--start-yaw", choices=("home", "aligned"), default="home",
                        help="wrist yaw when the policy takes over: the home pose's (default), or turned to the "
                             "source object's grasp yaw first, the state every engine 0.6.0 demonstration starts "
                             "in (Simulator.align_yaw; uses the simulator's object yaw, recorded in the metadata)")
    args = parser.parse_args()

    bundle = load_run(args.run)
    run, session_cfg = bundle["run"], bundle["session_cfg"]
    from config import instruction as make_instruction
    names = list(session_cfg["scene"]["objects"])
    pairs = parse_pairs(args.pairs, names)
    layouts = parse_layouts(args.layouts) if args.layouts else list(range(1, args.trials + 1))
    holdout = {tuple(p) for p in session_cfg["task"].get("holdout_pairs", [])}
    template = session_cfg["task"]["instruction"]

    checkpoint = bundle["adapter"].parent
    out = Path(args.out) if args.out else TRAIN_DIR / "evals" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{bundle['run_dir'].name}_{checkpoint.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seeds.json").write_text(json.dumps({"layouts": layouts, "meaning": "simulator reset numbers; env_config(CONFIG, n, 0)"}, indent=1))

    policy = Policy(run, bundle["adapter"], bundle["stats"])
    sim = Simulator(session_cfg, backend=args.backend, z_min=None if args.z_min < 0 else args.z_min,
                    gripper=args.gripper, show_viewer=args.viewer)
    r = Rollout(sim, policy, fps=session_cfg["record"]["fps"])
    crit, specs = sim.cfg["success"], sim.cfg["scene"]["objects"]
    protocol = {"trials": len(layouts), "layouts": layouts, "pairs": pairs, "horizon": args.horizon,
                "control_hz": round(r.hz, 2), "z_min": args.z_min, "gripper": args.gripper, "backend": args.backend,
                "start_yaw": args.start_yaw,
                "criteria": crit, "reach_xy": REACH_XY, "reach_z": REACH_Z, "lift_z": LIFT_Z}
    metrics_file = checkpoint / "metrics.json"
    metadata = {
        "created": datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(), "git_commit": git_commit(),
        "checkpoint": str(checkpoint), "adapter": str(bundle["adapter"]), "run": str(bundle["run_dir"]),
        "training_metrics_at_checkpoint": json.loads(metrics_file.read_text()) if metrics_file.exists() else None,
        "run_config": {"model": run["config"]["model"], "action": run["config"]["action"], "optim": run["config"]["optim"],
                       "lora": run["config"]["lora"], "observation": run["config"]["observation"]},
        "dataset": run.get("dataset"), "session": str(bundle["session"]),
        "inference_model": run["config"]["model"],      # after load_run's fit-to-card substitution, if any
        "holdout_pairs": sorted(holdout), "protocol": protocol,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=1, default=str))
    print(f"evaluation folder {out}\ncheckpoint {checkpoint}\n{len(pairs)} pairs x {len(layouts)} layouts (seeds {layouts}), "
          f"horizon {args.horizon} steps at {r.hz:.0f} Hz, z floor {args.z_min}, gripper {args.gripper}", flush=True)

    records, saved_videos = [], {}
    for s, d in pairs:
        instr = {"source": s, "destination": d, "text": make_instruction({"source": s, "destination": d, "instruction": template})}
        for layout in layouts:
            rec, frames = rollout(sim, policy, instr, layout, args.horizon, r.sim_steps, crit, specs, start_yaw=args.start_yaw)
            rec["held_out"] = (s, d) in holdout
            if rec["outcome"] not in saved_videos:          # first of each outcome, successes included
                name = f"{rec['outcome'].replace(' ', '_')}__{s}_on_{d}__layout{layout}.mp4"
                write_video(out / "videos" / name, frames)
                saved_videos[rec["outcome"]] = name
                rec["video"] = f"videos/{name}"
            records.append(rec)
            (out / "results.json").write_text(json.dumps(records, indent=1))
            print(f"  {instr['text']:<42s} layout {layout:3d}  {rec['outcome']:<22s} {rec['steps']:4d} steps  "
                  f"lifted {rec['lifted'] or '-'}  src-dst xy {rec['final_source_to_dest_xy_m']}", flush=True)

    # ---- summary
    by_pair = defaultdict(list)
    for rec in records:
        by_pair[(rec["source"], rec["destination"])].append(rec)
    lines = [f"# Evaluation of {checkpoint.name} ({bundle['run_dir'].name})", "",
             f"{len(pairs)} ordered pairs x {len(layouts)} layouts = {len(records)} rollouts; seeds {layouts}; "
             f"horizon {args.horizon} steps; z floor {args.z_min}; gripper {args.gripper}.",
             f"Success: source resting on destination (xy tol {crit['stack_xy_tol']} m, z tol {crit['stack_z_tol']} m), "
             f"destination upright and at rest, third object within {crit['disturb_tol']} m of its spawn, scene still {crit['hold_time']} s.",
             "", "| ordered pair | success | failures |", "|---|---|---|"]
    for (s, d), recs in by_pair.items():
        n_ok = sum(x["success"] for x in recs)
        tax = ", ".join(f"{k} {v}" for k, v in Counter(x["outcome"] for x in recs if not x["success"]).most_common())
        lines.append(f"| {s} > {d}{' (held out)' if (s, d) in holdout else ''} | {n_ok}/{len(recs)} | {tax} |")
    lines.append(f"| **all** | **{sum(x['success'] for x in records)}/{len(records)}** | |")
    rev = [((a, b), (b, a)) for a, b in pairs if (b, a) in by_pair and (a, b) < (b, a)]
    if rev:
        lines += ["", "Reversal pairs (same layouts, opposite instruction):"]
        for p, q in rev:
            lines.append(f"- {p[0]} > {p[1]}: {sum(x['success'] for x in by_pair[p])}/{len(by_pair[p])};  "
                         f"{q[0]} > {q[1]}: {sum(x['success'] for x in by_pair[q])}/{len(by_pair[q])}")
    lines += ["", "Failure taxonomy over all rollouts: " + ", ".join(f"{k} {v}" for k, v in Counter(x["outcome"] for x in records).most_common()),
              "", "Videos (first of each outcome): " + ", ".join(f"{k}: {v}" for k, v in saved_videos.items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[5:]))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
