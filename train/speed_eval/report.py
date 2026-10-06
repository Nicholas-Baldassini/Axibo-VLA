"""
One table across evaluation folders: every stack's real-time result next to its lockstep ceiling,
and the stall dose-response from the fixed-latency sweeps.

    python train/speed_eval/report.py train/evals/speed_eval            # every folder under it
    python train/speed_eval/report.py <folder> <folder> ... [--out report.md]

A folder is one speed_eval.py run (config.json, summary.json, <stack>/results.json). Folders are
grouped by their timing and latency model: "lockstep", "realtime" (measured latency) and
"fixed <n> ms"; a stack's row shows its success under each, its latency and smoothness from the
real-time run, and its token agreement with vanilla.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(folder):
    folder = Path(folder)
    cfg = json.loads((folder / "config.json").read_text())["config"] if (folder / "config.json").exists() else {}
    timing = cfg.get("timing", "?")
    latency = cfg.get("control", {}).get("latency")
    mode = "lockstep" if timing == "lockstep" else (f"fixed {latency * 1000:.0f} ms" if latency is not None else "realtime")
    stacks = {}
    for results in folder.glob("*/results.json"):
        records = json.loads(results.read_text())
        if not records:
            continue
        name = results.parent.name
        summary = json.loads((results.parent / "summary.json").read_text()) if (results.parent / "summary.json").exists() else {}
        check = json.loads((results.parent / "check.json").read_text()) if (results.parent / "check.json").exists() else {}
        lat = [l for r in records for l in r["latencies_s"]]
        stacks[name] = {
            "success": sum(r["success"] for r in records), "rollouts": len(records),
            "p50": np.median(lat) * 1000 if lat else None, "p99": np.percentile(lat, 99) * 1000 if lat else None,
            "action_hz": np.mean([r["action_hz"] for r in records]), "stall": summary.get("stall_frac"),
            "jerk": np.mean([r["jerk_rms"] for r in records]), "stops": np.mean([r.get("stops_per_s", 0) for r in records]),
            "staleness": summary.get("staleness_ms_p50"), "agreement": check.get("agreement"),
            "breakdown": next((r["latency_breakdown_ms"] for r in records if r.get("latency_breakdown_ms")), None),
            "outcomes": summary.get("outcomes"), "folder": str(folder),
        }
    return mode, stacks


def fmt(v, spec="{:.0f}"):
    return "" if v is None else spec.format(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--out")
    args = parser.parse_args()
    folders = []
    for f in args.folders:
        p = Path(f)
        folders += [p] if (p / "summary.json").exists() else sorted(q for q in p.glob("*") if (q / "summary.json").exists())
    by_mode = defaultdict(dict)        # mode -> stack -> row (a later folder of the same mode and stack replaces an earlier one)
    for folder in folders:
        mode, stacks = load(folder)
        for name, row in stacks.items():
            by_mode[mode][name] = row

    lines = ["# Speed evaluation report", ""]
    # ---- the main table: real time next to lockstep
    rt, ls = by_mode.get("realtime", {}), by_mode.get("lockstep", {})
    names = list(dict.fromkeys(list(rt) + list(ls)))
    if names:
        lines += ["## Stacks: real time against the lockstep ceiling", "",
                  "| stack | lockstep success | real-time success | p50 ms | p99 ms | actions/s | stall | stops/s | jerk rms | staleness ms | tokens = vanilla | time split ms |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for n in names:
            r, l = rt.get(n, {}), ls.get(n, {})
            split = r.get("breakdown") or {}
            lines.append(f"| {n} | {fmt(l.get('success'), '{}')}/{fmt(l.get('rollouts'), '{}')} | {fmt(r.get('success'), '{}')}/{fmt(r.get('rollouts'), '{}')} | "
                         f"{fmt(r.get('p50'))} | {fmt(r.get('p99'))} | {fmt(r.get('action_hz'), '{:.1f}')} | {fmt(r.get('stall'), '{:.0%}')} | "
                         f"{fmt(r.get('stops'), '{:.1f}')} | {fmt(r.get('jerk'))} | {fmt(r.get('staleness'))} | "
                         f"{fmt((r.get('agreement') if r else l.get('agreement')), '{:.0%}')} | "
                         + ", ".join(f"{k} {v}" for k, v in split.items() if k != "passes") + " |")
        lines.append("")
    # ---- the sweep
    fixed = sorted((m for m in by_mode if m.startswith("fixed")), key=lambda m: float(m.split()[1]))
    if fixed:
        stack_names = list(dict.fromkeys(n for m in fixed for n in by_mode[m]))
        lines += ["## Success against the latency on the clock (fixed latency, same seeds)", "",
                  "| hold / stack | " + " | ".join(fixed) + " |", "|---|" + "---|" * len(fixed)]
        for n in stack_names:
            cells = []
            for m in fixed:
                r = by_mode[m].get(n)
                cells.append(f"{r['success']}/{r['rollouts']} (stall {r['stall']:.0%})" if r and r.get("stall") is not None else (f"{r['success']}/{r['rollouts']}" if r else ""))
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    # ---- outcomes
    lines += ["## Failure taxonomy", ""]
    for mode in ["lockstep", "realtime"] + fixed:
        for n, r in by_mode.get(mode, {}).items():
            if r.get("outcomes"):
                lines.append(f"- {mode}, {n}: " + ", ".join(f"{k} {v}" for k, v in r["outcomes"].items()))
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"written {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
