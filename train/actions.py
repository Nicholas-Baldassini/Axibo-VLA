"""
Action vectors for OpenVLA: built from recorded columns per config, normalized to [-1, 1] and
mapped onto the 256 action tokens at the end of the Llama vocabulary, the way OpenVLA's own
training code does it (prismatic/vla/action_tokenizer.py and its bounds-q99 normalization).
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_delta(q_from, q_to):
    """Rotation vectors taking q_from to q_to, (n, 4) quaternions (w, x, y, z) -> (n, 3)."""
    r_from = Rotation.from_quat(np.asarray(q_from)[:, [1, 2, 3, 0]])
    r_to = Rotation.from_quat(np.asarray(q_to)[:, [1, 2, 3, 0]])
    return (r_from.inv() * r_to).as_rotvec()


def episode_ranges(episode_index):
    """[start, end) row ranges of the contiguous episodes in an episode_index column."""
    ep = np.asarray(episode_index)
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], len(ep)]
    return list(zip(starts.tolist(), ends.tolist()))


def build_action_table(columns, feature_names, ranges, action_cfg):
    """(N, D) float32 action per frame per action_cfg, with the D names and the per-dimension normalize mask.

    columns: {key: (N, k) array} for every key the components use. feature_names: {key: [k names]}.
    ranges: the episodes' [start, end) rows, so a frame's target never crosses into the next episode.
    """
    stride = int(action_cfg["stride"])
    n = len(next(iter(columns.values())))
    target = np.arange(n)
    for start, end in ranges:
        target[start:end] = np.minimum(np.arange(start, end) + stride, end - 1)

    parts, names, normalize = [], [], []
    for comp in action_cfg["components"]:
        key, dims = comp["key"], list(comp["dims"])
        col = columns[key]
        now, ahead = col[:, dims], col[target][:, dims]
        col_names = [feature_names[key][d] for d in dims]
        delta = comp.get("delta")
        if delta is None:
            values, part_names = ahead, col_names
        elif delta == "translation":
            values, part_names = ahead - now, [f"d{c}" for c in col_names]
        elif delta == "rotation":
            if len(dims) != 4:
                raise ValueError(f"'rotation' needs the 4 quaternion columns (w, x, y, z), got dims {dims}")
            values, part_names = rotation_delta(now, ahead), ["drot_x", "drot_y", "drot_z"]
        else:
            raise ValueError(f"unknown delta {delta!r} in action component {comp}")
        parts.append(np.asarray(values, dtype=np.float32))
        names += part_names
        normalize += [bool(comp.get("normalize", True))] * values.shape[1]
    return np.concatenate(parts, axis=1), names, np.array(normalize, dtype=bool)


def action_statistics(table, normalize_mask, bounds="q99"):
    """Per-dimension statistics in OpenVLA's dataset_statistics layout (plain lists, JSON-ready).

    bounds picks what the q01/q99 slots hold, which is what normalize_actions maps onto [-1, 1]:
        "q99"     the 1st and 99th percentiles, OpenVLA's convention, which clips the outer 2%
        "minmax"  the dataset minimum and maximum, so nothing is clipped
    The slots keep their q01/q99 names either way, so the file loads wherever OpenVLA's does.
    """
    if bounds == "q99":
        low, high = np.quantile(table, 0.01, axis=0), np.quantile(table, 0.99, axis=0)
    elif bounds == "minmax":
        low, high = table.min(axis=0), table.max(axis=0)
    else:
        raise ValueError(f"unknown action bounds {bounds!r}; use \"q99\" or \"minmax\"")
    return {
        "mean": table.mean(axis=0).tolist(),
        "std": table.std(axis=0).tolist(),
        "max": table.max(axis=0).tolist(),
        "min": table.min(axis=0).tolist(),
        "q01": low.tolist(),
        "q99": high.tolist(),
        "bounds": bounds,
        "mask": [bool(m) for m in normalize_mask],   # True: the dimension is normalized by q01..q99
    }


def normalize_actions(actions, stats):
    """Masked dimensions go from [q01, q99] to [-1, 1] (OpenVLA's bounds-q99 normalization); everything is clipped."""
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats["mask"], dtype=bool)
    scaled = 2.0 * (actions - low) / np.maximum(high - low, 1e-8) - 1.0
    return np.clip(np.where(mask, scaled, actions), -1.0, 1.0).astype(np.float32)


def unnormalize_actions(normalized, stats):
    """Inverse of normalize_actions; what the checkpoint's predict_action applies at inference."""
    low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats["mask"], dtype=bool)
    return np.where(mask, 0.5 * (normalized + 1.0) * (high - low) + low, normalized)


class ActionTokenizer:
    """Continuous actions in [-1, 1] <-> the last `bins` tokens of the tokenizer's vocabulary.

    OpenVLA's ActionTokenizer: bin index b (1..bins) becomes token id vocab_size - b.
    """

    def __init__(self, vocab_size, bins=256, min_action=-1.0, max_action=1.0):
        self.vocab_size, self.n_bins = int(vocab_size), bins   # tokenizer.vocab_size, 32000 for Llama 2
        self.min_action, self.max_action = min_action, max_action
        self.bins = np.linspace(min_action, max_action, bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.action_token_begin_idx = self.vocab_size - (bins + 1)

    def token_ids(self, action):
        """One token id per dimension of an action (D,)."""
        clipped = np.clip(np.asarray(action, dtype=np.float64), self.min_action, self.max_action)
        return (self.vocab_size - np.digitize(clipped, self.bins)).astype(np.int64)

    def decode_token_ids_to_actions(self, action_token_ids):
        """Token ids -> continuous actions in [-1, 1] (bin centers)."""
        discretized = self.vocab_size - np.asarray(action_token_ids)
        discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized]
