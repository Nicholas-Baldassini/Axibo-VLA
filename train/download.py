"""
Put everything a machine needs for inference or training in place, from Hugging Face:

    python train/download.py                                        base checkpoint only (openvla/openvla-7b, 15 GB)
    python train/download.py --run Boominator/vlaBlockStack         + a fine-tuned run   -> train/runs/<run>/
    python train/download.py --dataset Boominator/vlaTestStorage    + a recorded session -> datasets/<session>/
    python train/download.py --run ... --dataset ...                all three at once

Afterwards, with nothing else:

    python train/infer.py train/runs/<run>                the policy in the simulator
    python train/finetune.py --config openvla             a new fine-tune on the downloaded session

A repo holds versions side by side, one folder per run or session, named after the directory they
came from (train/upload.py and source/upload.py write that layout). The newest version by name is
taken unless --version names one; --list shows what a repo holds. A repo whose files sit at the
root (the first uploads, before versions) is read as a single version named after the repo.

Where things land:

    train/checkpoints/<name>/        the base model, exactly the Hugging Face repo
    train/runs/<run>/                config.json, dataset_statistics.json, train_log.jsonl, step_*/, final/,
                                     session_config.json (the recording's config, which inference needs)
    datasets/<session>/dataset/      the LeRobot dataset (meta/, data/, videos/)
    datasets/<session>/              metadata.json, config.json, episodes.json, requirements*.txt

Re-running only fetches what is missing; nothing is deleted. Private repos need `hf auth login`.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent
MARGIN_GB = 5.0   # free space to leave on the disk after a download


def _files(repo, repo_type):
    return HfApi().repo_info(repo, repo_type=repo_type, files_metadata=True).siblings


def _check_space(dest, missing):
    free = shutil.disk_usage(dest).free
    if free < missing + MARGIN_GB * 1e9:
        sys.exit(f"not enough disk space at {dest}: need {missing / 1e9:.1f} GB plus a {MARGIN_GB:.0f} GB margin, "
                 f"have {free / 1e9:.1f} GB")


def versions(files, marker, root_marker=None):
    """{version: [files]}. A top-level folder is a version when it holds `marker`; the root is the
    version '' when it holds `root_marker` (default: `marker`), and owns every file that is not
    inside a version folder."""
    names = {f.rfilename for f in files}
    folders = {f.rfilename.split("/")[0] for f in files if "/" in f.rfilename}
    found = {v: [f for f in files if f.rfilename.startswith(v + "/")] for v in folders if f"{v}/{marker}" in names}
    if (root_marker or marker) in names:
        found[""] = [f for f in files if f.rfilename.split("/")[0] not in found]
    return found


def choose(repo, repo_type, marker, version, root_marker=None):
    files = _files(repo, repo_type)
    found = versions(files, marker, root_marker)
    if not found:
        sys.exit(f"{repo} holds no version with {marker} one folder down or {root_marker or marker} at the root")
    names = sorted(found)
    if version == "root":
        version = ""
    if version is None:
        version = names[-1]
    if version not in found:
        sys.exit(f"{repo} has no version {version!r}; available: {', '.join(n or 'root' for n in names)}")
    print(f"{repo}: versions " + ", ".join((n or "root (legacy layout)") + (" <- chosen" if n == version else "") for n in names)
          + ("" if len(names) == 1 else "   (pick another with --version)"))
    return version, found[version], sorted(v for v in found if v)


def _fetch(repo, repo_type, version, files, dest, rel, ignore=()):
    """Bring `files` of one version into dest, each at dest / rel(file). Skips files already there."""
    total = sum(f.size or 0 for f in files)
    present = sum((dest / rel(f)).stat().st_size for f in files if (dest / rel(f)).exists())
    print(f"  {total / 1e9:.2f} GB in {len(files)} files, {present / 1e9:.2f} GB already in {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    _check_space(dest, total - present)
    if version:
        tmp = Path(snapshot_download(repo, repo_type=repo_type, allow_patterns=[f"{version}/*"]))
        for f in files:
            target = dest / rel(f)
            if not target.exists() or target.stat().st_size != (f.size or 0):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(tmp / f.rfilename, target)
    else:
        snapshot_download(repo, repo_type=repo_type, local_dir=dest, ignore_patterns=[".cache/*", *ignore])


def download_base(repo):
    dest = TRAIN_DIR / "checkpoints" / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    files = _files(repo, "model")
    total = sum(f.size or 0 for f in files)
    present = sum((dest / f.rfilename).stat().st_size for f in files if (dest / f.rfilename).exists())
    print(f"base checkpoint {repo}: {total / 1e9:.1f} GB in {len(files)} files, {present / 1e9:.1f} GB already in {dest}")
    _check_space(dest, total - present)
    snapshot_download(repo, local_dir=dest)
    print(f"  ready: {dest}")
    return dest


def download_run(repo, version=None):
    version, files, folders = choose(repo, "model", "config.json", version)
    dest = TRAIN_DIR / "runs" / (version or repo.split("/")[-1])
    strip = lambda f: f.rfilename[len(version) + 1:] if version else f.rfilename
    _fetch(repo, "model", version, files, dest, strip, ignore=[f"{v}/*" for v in folders])
    if not (dest / "session_config.json").exists():
        # A run uploaded before train/upload.py existed has no session config of its own. The
        # session it trained on is named in its config.json; if that session is on this machine
        # (downloaded with --dataset, in the same command or earlier), its config.json serves.
        session_name = Path(json.loads((dest / "config.json").read_text()).get("session", "")).name
        local_cfg = ROOT / "datasets" / session_name / "config.json"
        if session_name and local_cfg.exists():
            shutil.copy(local_cfg, dest / "session_config.json")
            print(f"  session_config.json <- {local_cfg} (the run carried none)")
        else:
            print(f"  WARNING: this run carries no session_config.json and its session {session_name!r} is not under "
                  f"datasets/. Inference will refuse to start. Download the session (--dataset <repo>) or copy the "
                  f"recording's config.json to {dest / 'session_config.json'}.")
    print(f"  ready: {dest}")
    return dest


def download_dataset(repo, version=None):
    # A version folder mirrors a session directory (dataset/meta/...); the legacy root layout has
    # meta/, data/, videos/ at the root and the session files under session/.
    version, files, folders = choose(repo, "dataset", "dataset/meta/info.json", version, root_marker="meta/info.json")
    dest = ROOT / "datasets" / (version or repo.split("/")[-1])
    if version:
        _fetch(repo, "dataset", version, files, dest, lambda f: f.rfilename[len(version) + 1:])
    else:
        data = [f for f in files if not f.rfilename.startswith("session/")]
        _fetch(repo, "dataset", "", data, dest / "dataset", lambda f: f.rfilename,
               ignore=["session/*", *(f"{v}/*" for v in folders)])
        session = [f for f in files if f.rfilename.startswith("session/")]
        if session:
            tmp = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=["session/*"]))
            for f in session:
                shutil.copy(tmp / f.rfilename, dest / Path(f.rfilename).name)
    missing = [n for n in ("metadata.json", "config.json") if not (dest / n).exists()]
    if missing:
        print(f"  warning: no {' or '.join(missing)} came with this dataset. Training works without them; inference on "
              f"a run trained here needs config.json (or a session_config.json beside the run).")
    print(f"  ready: {dest}")
    return dest


def show(repo, repo_type, marker, root_marker=None):
    files = _files(repo, repo_type)
    found = versions(files, marker, root_marker)
    print(f"{repo} ({repo_type}): {sum(f.size or 0 for f in files) / 1e9:.2f} GB, {len(files)} files")
    for v in sorted(found):
        fs = found[v]
        print(f"  {v or 'root (legacy layout)'}: {len(fs)} files, {sum(f.size or 0 for f in fs) / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download the base checkpoint, a fine-tuned run and a recorded session.")
    parser.add_argument("--base", default="openvla/openvla-7b", help="base model repo (default: openvla/openvla-7b)")
    parser.add_argument("--no-base", action="store_true", help="skip the base checkpoint")
    parser.add_argument("--run", help="fine-tuned run repo, e.g. Boominator/vlaBlockStack")
    parser.add_argument("--dataset", help="dataset repo, e.g. Boominator/vlaTestStorage")
    parser.add_argument("--version", help="which run or session folder to take (default: the newest by name)")
    parser.add_argument("--list", action="store_true", help="show the versions in --run / --dataset and stop")
    args = parser.parse_args()

    if args.list:
        if args.run:
            show(args.run, "model", "config.json")
        if args.dataset:
            show(args.dataset, "dataset", "dataset/meta/info.json", root_marker="meta/info.json")
        return

    done = []
    if not args.no_base:
        done.append(download_base(args.base))
    if args.dataset:      # before the run, so a run without its own session config can take the session's
        done.append(download_dataset(args.dataset, args.version))
    run_dir = None
    if args.run:
        run_dir = download_run(args.run, args.version)
        done.append(run_dir)
    print("done:")
    for d in done:
        print(f"  {d.relative_to(ROOT)}")
    if run_dir is not None:
        print(f"inference:  python train/infer.py {run_dir.relative_to(ROOT)}")
    if args.dataset:
        print("training:   python train/finetune.py --config openvla")


if __name__ == "__main__":
    main()
