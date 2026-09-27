"""
Upload a recorded session to a Hugging Face dataset repo, as one version among others:

    python source/upload.py datasets/piper_stack_2026-09-27_12-00-00 --repo Boominator/vlaTestStorage
    python source/upload.py --repo Boominator/vlaTestStorage            the newest session under datasets/
    python source/upload.py ... --dry-run                               only report what would happen

The session goes up under a folder named after it, mirroring the directory on disk:

    <session>/dataset/{meta,data,videos}     the LeRobot dataset
    <session>/metadata.json, config.json, episodes.json, requirements.txt, requirements.lock.txt

so a repo can hold every session ever recorded, and train/download.py can pick one by name or take
the newest. Before uploading, every file is compared with what the repo already holds and the
result printed: new, already there and identical, or different. Different files are not replaced
unless --overwrite is given. Needs `hf auth login`.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.hub import list_files, push   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def newest_session(datasets_root):
    sessions = sorted(p for p in Path(datasets_root).iterdir() if (p / "dataset" / "meta" / "info.json").exists())
    if not sessions:
        sys.exit(f"no recorded session under {datasets_root}")
    return sessions[-1]


def main():
    parser = argparse.ArgumentParser(description="Upload a recorded session to a Hugging Face dataset repo.")
    parser.add_argument("session", nargs="?", help="session directory (default: the newest under datasets/)")
    parser.add_argument("--repo", required=True, help="dataset repo id, e.g. Boominator/vlaTestStorage")
    parser.add_argument("--overwrite", action="store_true", help="replace files that differ from the repo's")
    parser.add_argument("--dry-run", action="store_true", help="compare and report only")
    parser.add_argument("--public", action="store_true", help="if the repo has to be created, make it public")
    args = parser.parse_args()

    session = Path(args.session).resolve() if args.session else newest_session(ROOT / "datasets")
    if not (session / "dataset" / "meta" / "info.json").exists():
        sys.exit(f"{session} has no dataset/meta/info.json: not a recorded session")
    # The session files are part of the version, not optional: without config.json a run trained
    # on this session cannot be run elsewhere, and without metadata.json infer.py cannot tell which
    # renderer recorded it. The first upload left them out and both had to be rebuilt by hand.
    missing = [n for n in ("metadata.json", "config.json", "episodes.json") if not (session / n).exists()]
    if missing:
        sys.exit(f"{session} is missing {', '.join(missing)}; a session is uploaded whole or not at all")
    meta = json.loads((session / "metadata.json").read_text())
    files = list_files(session)
    print(f"session {session.name}: {len(files)} files, engine {meta.get('project_version', '?')}, "
          f"{meta.get('episodes_succeeded', '?')} episodes, {meta.get('frames', '?')} frames -> {args.repo}/{session.name}/")
    push(session, files, args.repo, "dataset", prefix=session.name, marker="dataset/meta/info.json",
         message=f"session {session.name}: engine {meta.get('project_version', '?')}, {meta.get('episodes_succeeded', '?')} episodes",
         overwrite=args.overwrite, dry_run=args.dry_run, private=not args.public)
    if not args.dry_run:
        print(f"download elsewhere: python train/download.py --no-base --dataset {args.repo} --version {session.name}")


if __name__ == "__main__":
    main()
