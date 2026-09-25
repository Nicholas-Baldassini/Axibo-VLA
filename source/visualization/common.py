"""Shared loading for the visualization tools: a recorded session directory and its LeRobot dataset."""

import json
import sys
from pathlib import Path

import numpy as np


def find_session(path):
    """Accept a session directory or its dataset/ subdirectory; exit with a message otherwise."""
    p = Path(path).expanduser().resolve()
    if p.name == "dataset" and (p / "meta" / "info.json").exists():
        return p.parent
    if (p / "dataset" / "meta" / "info.json").exists():
        return p
    sys.exit(f"{p} is not a finished recording session: no dataset/meta/info.json found")


def short_camera_name(key):
    return key.split("observation.images.", 1)[-1]


class SessionData:
    """A recording session: metadata, per-attempt log, and the dataset's tabular columns in memory."""

    def __init__(self, path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dir = find_session(path)
        self.name = self.dir.name
        self.metadata = self._read_json("metadata.json") or {}
        self.config = self._read_json("config.json") or {}
        self.attempts = self._read_json("episodes.json") or []
        self.by_episode = {a["episode_index"]: a for a in self.attempts if a.get("episode_index") is not None}

        self.dataset = LeRobotDataset(self.metadata.get("repo_id", "local/dataset"), root=self.dir / "dataset")
        self.fps = self.dataset.meta.fps
        self.features = self.dataset.meta.features
        self.n_episodes = self.dataset.meta.total_episodes
        self.cameras = [k for k in self.features if k.startswith("observation.images.")]
        self.numeric = [k for k, f in self.features.items() if f["dtype"] == "float32" and k != "timestamp"]
        hf = self.dataset.hf_dataset
        self.columns = {k: np.asarray(hf[k], dtype=np.float32) for k in self.numeric}

    def _read_json(self, name):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def names(self, key):
        names = self.features[key].get("names")
        return list(names) if names else [f"{key}[{i}]" for i in range(self.features[key]["shape"][0])]

    def episode(self, e):
        rec = self.dataset.meta.episodes[e]
        return {
            "index": e,
            "start": int(rec["dataset_from_index"]),
            "end": int(rec["dataset_to_index"]),
            "length": int(rec["length"]),
            "task": rec["tasks"][0] if len(rec["tasks"]) else "",
            "record": rec,
            "attempt": self.by_episode.get(e),
        }

    def episode_columns(self, e):
        ep = self.episode(e)
        return {k: v[ep["start"]:ep["end"]] for k, v in self.columns.items()}

    def video_segment(self, e, camera):
        """(mp4 path, from_timestamp, to_timestamp) of episode e for a camera key."""
        rec = self.dataset.meta.episodes[e]
        path = self.dir / "dataset" / self.dataset.meta.video_path.format(
            video_key=camera,
            chunk_index=int(rec[f"videos/{camera}/chunk_index"]),
            file_index=int(rec[f"videos/{camera}/file_index"]),
        )
        return path, float(rec[f"videos/{camera}/from_timestamp"]), float(rec[f"videos/{camera}/to_timestamp"])


def decode_segment(path, t_from, t_to, scale=1.0, expected=None):
    """Decode the frames of one video file between two timestamps, sequentially, as (T, H, W, 3) uint8."""
    import av
    import cv2

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(t_from / stream.time_base), stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t < t_from - 1e-4:
                continue
            if t >= t_to - 1e-4:
                break
            img = frame.to_ndarray(format="rgb24")
            if scale != 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(img)
    if not frames:
        raise RuntimeError(f"no frames decoded from {path} in [{t_from}, {t_to})")
    arr = np.stack(frames)
    if expected is not None and len(arr) != expected:
        if len(arr) > expected:
            arr = arr[:expected]
        else:
            arr = np.concatenate([arr, np.repeat(arr[-1:], expected - len(arr), axis=0)])
    return arr


def resample(signal, n=100):
    """Resample a (T, d) episode signal onto n points of normalized time."""
    signal = np.asarray(signal)
    T = len(signal)
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, n)
    if signal.ndim == 1:
        return np.interp(dst, src, signal)
    return np.stack([np.interp(dst, src, signal[:, j]) for j in range(signal.shape[1])], axis=1)
