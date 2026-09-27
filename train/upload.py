"""
Upload a training run to a Hugging Face model repo, as one version among others:

    python train/upload.py train/runs/openvla_2026-09-27_00-58-03 --repo Boominator/vlaBlockStack
    python train/upload.py --repo Boominator/vlaBlockStack                the newest run under train/runs/
    python train/upload.py ... --dry-run                                  only report what would happen

The run goes up under a folder named after it, mirroring the directory on disk (config.json,
dataset_statistics.json, train_log.jsonl, step_*/ and final/), plus the recording's config.json as
<run>/session_config.json, which inference needs on a machine that has no copy of the session. It
is found through the session path the run's config.json records, or given with --session.

Every file is compared with what the repo holds before anything is sent, and the result printed:
new, already there and identical, or different. Different files are not replaced unless
--overwrite is given. Needs `hf auth login`.
"""

import argparse
import json
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent
sys.path.insert(0, str(ROOT / "source"))

from utils.hub import list_files, push   # noqa: E402


def newest_run(runs_root):
    runs = sorted(p for p in Path(runs_root).iterdir() if (p / "config.json").exists() and p.name != "vlaBlockStack")
    if not runs:
        sys.exit(f"no training run under {runs_root}")
    return runs[-1]


def main():
    parser = argparse.ArgumentParser(description="Upload a training run to a Hugging Face model repo.")
    parser.add_argument("run", nargs="?", help="run directory (default: the newest under train/runs/)")
    parser.add_argument("--repo", required=True, help="model repo id, e.g. Boominator/vlaBlockStack")
    parser.add_argument("--session", help="session directory whose config.json goes up as session_config.json "
                                          "(default: the one the run's config.json records)")
    parser.add_argument("--only", help="comma-separated checkpoints to include, e.g. final,step_016000 "
                                       "(default: every one; each is about 0.9 GB)")
    parser.add_argument("--overwrite", action="store_true", help="replace files that differ from the repo's")
    parser.add_argument("--dry-run", action="store_true", help="compare and report only")
    parser.add_argument("--public", action="store_true", help="if the repo has to be created, make it public")
    args = parser.parse_args()

    run = Path(args.run).resolve() if args.run else newest_run(TRAIN_DIR / "runs")
    if not (run / "config.json").exists():
        sys.exit(f"{run} is not a training run: no config.json")
    run_cfg = json.loads((run / "config.json").read_text())
    files = list_files(run)
    checkpoints = sorted(p.name for p in run.iterdir() if p.is_dir() and (p / "adapter").is_dir())
    if args.only:
        keep = set(args.only.split(","))
        unknown = keep - set(checkpoints)
        if unknown:
            sys.exit(f"no checkpoint(s) {', '.join(sorted(unknown))} in {run}; have {', '.join(checkpoints)}")
        files = [f for f in files if f.split("/")[0] not in set(checkpoints) - keep]
        checkpoints = sorted(keep)

    # The recording's config travels with the run, under the name inference looks for.
    session = Path(args.session) if args.session else Path(run_cfg.get("session", ""))
    session_cfg = session / "config.json"
    staged = None
    if session_cfg.exists() and not (run / "session_config.json").exists():
        staged = run / "session_config.json"
        staged.write_bytes(session_cfg.read_bytes())
        files = sorted(files + ["session_config.json"])
        print(f"session_config.json <- {session_cfg}")
    elif not session_cfg.exists() and not (run / "session_config.json").exists():
        # Refused rather than warned: the first upload went up without it, and inference on the
        # next machine was blocked until the recording's config was reconstructed by hand.
        sys.exit(f"no session config found at {session_cfg}. Pass --session <session dir> (its config.json is "
                 f"uploaded as session_config.json); inference elsewhere cannot run without it.")
    else:
        print(f"session_config.json already in {run}")

    print(f"run {run.name}: {len(files)} files, checkpoints {', '.join(checkpoints) or 'none'}, "
          f"{run_cfg.get('dataset', {}).get('episodes', '?')} episodes of engine "
          f"{run_cfg.get('dataset', {}).get('engine_version', '?')} -> {args.repo}/{run.name}/")
    try:
        push(run, files, args.repo, "model", prefix=run.name, marker="config.json",
             message=f"run {run.name}: {', '.join(checkpoints)}",
             overwrite=args.overwrite, dry_run=args.dry_run, private=not args.public)
    finally:
        if staged is not None and args.dry_run:
            staged.unlink()     # a dry run leaves the run directory as it was
    if not args.dry_run:
        print(f"download elsewhere: python train/download.py --run {args.repo} --version {run.name}")


if __name__ == "__main__":
    main()
