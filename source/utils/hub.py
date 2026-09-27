"""
Hugging Face uploads that say what they are doing: which files are new, which are already there
and identical, which would be overwritten. Shared by source/upload.py (sessions) and
train/upload.py (runs).

A repo holds several versions side by side, one folder each, named after the session or run
directory (those names carry their timestamps). Uploading a version that is already there changes
nothing unless --overwrite is given; uploading a new one adds a folder and leaves the others alone.
"""

import hashlib
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path):
    """The id git gives a file's content, which is what the Hub reports for non-LFS files."""
    size = Path(path).stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_files(repo, repo_type):
    """{path: (size, lfs sha256 or None, git blob sha1)} for every file in the repo, {} if it does not exist."""
    api = HfApi()
    if not api.repo_exists(repo, repo_type=repo_type):
        return None
    out = {}
    for entry in api.list_repo_tree(repo, repo_type=repo_type, recursive=True, expand=True):
        if not hasattr(entry, "size"):        # a folder
            continue
        lfs = getattr(entry, "lfs", None)
        out[entry.path] = (entry.size, lfs.sha256 if lfs else None, getattr(entry, "blob_id", None))
    return out


def plan(local_root, files, prefix, remote):
    """Status of every local file against the repo: 'new', 'same' or 'changed', with sizes."""
    rows = []
    for rel in files:
        path = Path(local_root) / rel
        target = f"{prefix}/{rel}" if prefix else rel
        size = path.stat().st_size
        if remote is None or target not in remote:
            status = "new"
        else:
            r_size, r_sha256, r_blob = remote[target]
            if r_size != size:
                status = "changed"
            elif r_sha256 is not None:
                status = "same" if _sha256(path) == r_sha256 else "changed"
            else:
                status = "same" if _git_blob_sha1(path) == r_blob else "changed"
        rows.append((rel, target, status, size))
    return rows


def describe(rows, remote, prefix, marker):
    """Print what the repo already holds and what this upload would do. A top-level folder is a
    version when it holds `marker` (e.g. config.json for a run, dataset/meta/info.json for a
    session); `marker` at the root itself means the legacy single-version layout."""
    gb = lambda n: f"{n / 1e9:.2f} GB"
    if remote is None:
        print("  repo: does not exist yet, will be created (private)")
    else:
        folders = {p.split("/")[0] for p in remote if "/" in p}
        versions = sorted(v for v in folders if f"{v}/{marker}" in remote)
        legacy = marker in remote or marker.split("/", 1)[-1] in remote
        print(f"  repo: {len(remote)} files, {gb(sum(v[0] for v in remote.values()))}; versions present: "
              f"{', '.join(versions) if versions else 'none'}" + ("; plus one at the root (legacy layout)" if legacy else ""))
        if prefix in versions:
            print(f"  version {prefix!r} is already in the repo")
    counts = {s: [0, 0] for s in ("new", "same", "changed")}
    for _, _, status, size in rows:
        counts[status][0] += 1
        counts[status][1] += size
    print(f"  this upload: {counts['new'][0]} new files ({gb(counts['new'][1])}), "
          f"{counts['same'][0]} already there and identical ({gb(counts['same'][1])}), "
          f"{counts['changed'][0]} would overwrite ({gb(counts['changed'][1])})")
    for rel, target, status, size in rows:
        if status == "changed":
            print(f"    overwrite: {target}")


def push(local_root, files, repo, repo_type, prefix, marker, message, overwrite=False, dry_run=False, private=True):
    """Compare, print, and commit the new (and, with overwrite, the changed) files in one commit."""
    remote = remote_files(repo, repo_type)
    rows = plan(local_root, files, prefix, remote)
    describe(rows, remote, prefix, marker)
    todo = [r for r in rows if r[2] == "new" or (overwrite and r[2] == "changed")]
    blocked = [r for r in rows if r[2] == "changed" and not overwrite]
    if blocked:
        print(f"  {len(blocked)} file(s) differ from the repo and are left alone; pass --overwrite to replace them")
    if dry_run:
        print("  dry run: nothing uploaded")
        return None
    if not todo:
        print("  nothing to upload")
        return None
    api = HfApi()
    if remote is None:
        api.create_repo(repo, repo_type=repo_type, private=private)
    ops = [CommitOperationAdd(path_in_repo=target, path_or_fileobj=str(Path(local_root) / rel)) for rel, target, _, _ in todo]
    info = api.create_commit(repo_id=repo, repo_type=repo_type, operations=ops, commit_message=message)
    print(f"  uploaded {len(todo)} files in one commit: {info.commit_url}")
    return info


def list_files(root, skip_dirs=(".cache",)):
    """Relative paths of every file under root, skipping cache directories, sorted."""
    root = Path(root)
    out = []
    for p in root.rglob("*"):
        if p.is_file() and not any(part in skip_dirs for part in p.relative_to(root).parts):
            out.append(str(p.relative_to(root)))
    return sorted(out)
