"""
Download an OpenVLA checkpoint into train/checkpoints/<name>, after checking the free disk space.

    python train/download.py                                            openvla/openvla-7b (15 GB)
    python train/download.py --repo openvla/openvla-7b-finetuned-libero-spatial

Re-running only fetches what is missing.
"""

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

TRAIN_DIR = Path(__file__).resolve().parent
MARGIN_GB = 5.0   # free space to leave on the disk after the download


def main(repo):
    dest = TRAIN_DIR / "checkpoints" / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    files = HfApi().model_info(repo, files_metadata=True).siblings
    total = sum(f.size or 0 for f in files)
    present = sum((dest / f.rfilename).stat().st_size for f in files if (dest / f.rfilename).exists())
    missing = total - present
    free = shutil.disk_usage(dest).free
    print(f"{repo}: {total / 1e9:.1f} GB in {len(files)} files, {present / 1e9:.1f} GB already in {dest}")
    print(f"to download: {missing / 1e9:.1f} GB; free disk: {free / 1e9:.1f} GB")
    if free < missing + MARGIN_GB * 1e9:
        sys.exit(f"not enough disk space: need {missing / 1e9:.1f} GB plus a {MARGIN_GB:.0f} GB margin, have {free / 1e9:.1f} GB")
    snapshot_download(repo, local_dir=dest)
    print(f"checkpoint ready: {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an OpenVLA checkpoint.")
    parser.add_argument("--repo", default="openvla/openvla-7b", help="Hugging Face model id")
    main(parser.parse_args().repo)
