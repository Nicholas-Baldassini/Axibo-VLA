"""
Summary of a recorded session: outcome counts, task-pair coverage, where objects spawned, the
average signals over normalized episode time, episode lengths, and a 2-D PCA projection of every
observation state vector (no images). Shown in a window and saved as summary.png in the session.

    python source/visualization/dataset_summary.py datasets/piper_stack_20260924_182109
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SessionData, resample  # noqa: E402

N_TIME = 100   # points of normalized episode time for the averaged signals


def humanize(name):
    return name.replace("_", " ")


def object_names(s):
    return list(s.config.get("scene", {}).get("objects", {}))


# --------------------------------- panels ---------------------------------

def panel_outcomes(ax, s):
    m = s.metadata
    succeeded, failed = m.get("episodes_succeeded", s.n_episodes), m.get("episodes_failed", 0)
    reasons = m.get("failure_reasons", {})
    labels = ["succeeded"] + [f"failed:\n{r}" for r in reasons] if reasons else ["succeeded", "failed"]
    values = [succeeded] + (list(reasons.values()) if reasons else [failed])
    colors = ["tab:green"] + ["tab:red"] * (len(values) - 1)
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fontsize=8)
    total = succeeded + failed
    ax.set_title(f"Outcomes: {succeeded}/{total} succeeded" + (f" ({100 * succeeded / total:.0f}%)" if total else ""), fontsize=10)
    ax.tick_params(labelsize=8)


def panel_pairs(ax, s):
    objs = object_names(s)
    pairs = [(a, b) for a in objs for b in objs if a != b]
    holdout = {tuple(p) for p in s.metadata.get("holdout_pairs", [])}
    recorded = Counter((a["task"]["source"], a["task"]["destination"]) for a in s.attempts if a.get("success"))
    attempted = Counter((a["task"]["source"], a["task"]["destination"]) for a in s.attempts)
    x = np.arange(len(pairs))
    ax.bar(x - 0.2, [attempted[p] for p in pairs], width=0.4, color="lightgray", label="attempted")
    bars = ax.bar(x + 0.2, [recorded[p] for p in pairs], width=0.4, color="tab:blue", label="recorded")
    ax.bar_label(bars, fontsize=7)
    for i, p in enumerate(pairs):
        if p in holdout:
            ax.text(i, 0.3, "held out", ha="center", va="bottom", fontsize=7, color="tab:red", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{humanize(a)} → {humanize(b)}" for a, b in pairs], fontsize=6.5, rotation=30, ha="right")
    ax.set_title("Task pairs (source → destination)", fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(axis="y", labelsize=8)


def panel_spawns(ax, s):
    scene = s.config.get("scene", {})
    table = scene.get("table", {})
    if table:
        sx, sy = table["size"][:2]
        tx, ty = table["pos"][:2]
        ax.add_patch(Rectangle((tx - sx / 2, ty - sy / 2), sx, sy, fill=False, color="saddlebrown", lw=1.5, label="table"))
    area = s.config.get("randomization", {}).get("placement", {}).get("area")
    if area:
        ax.add_patch(Rectangle((area["x"][0], area["y"][0]), area["x"][1] - area["x"][0], area["y"][1] - area["y"][0],
                               fill=False, color="gray", ls="--", lw=1, label="spawn area"))
    if "robot" in scene:
        ax.plot(*scene["robot"]["pos"][:2], marker="s", color="k", ms=8, ls="none", label="robot base")
    for name, spec in scene.get("objects", {}).items():
        color = spec.get("color", (0.5, 0.5, 0.5, 1))[:3]
        marker = "s" if spec.get("shape") == "box" else "o"     # cubes as squares, cylinders as circles
        ok = [a["config"]["scene"]["objects"][name]["pos"] for a in s.attempts if a.get("success")]
        bad = [a["config"]["scene"]["objects"][name]["pos"] for a in s.attempts if not a.get("success")]
        if ok:
            ax.scatter(*np.array(ok).T, s=32, color=color, marker=marker, edgecolor="k", lw=0.4, label=humanize(name))
        if bad:
            ax.scatter(*np.array(bad).T, s=40, color=color, marker="x", label=f"{humanize(name)} (failed)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_title(f"Object spawn positions, {len(s.attempts)} attempts", fontsize=10)
    ax.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def averaged(s, column, sel=slice(None)):
    """(mean, std) over episodes of a column resampled onto normalized time: arrays (N_TIME, d)."""
    stacks = [resample(s.episode_columns(e)[column][:, sel], N_TIME) for e in range(s.n_episodes)]
    arr = np.stack(stacks)
    return arr.mean(axis=0), arr.std(axis=0)


def panel_mean_lines(ax, s, column, sel, title, labels=None, ylabel=""):
    if column not in s.columns:
        ax.set_axis_off()
        return
    mean, std = averaged(s, column, sel)
    tau = np.linspace(0, 1, N_TIME)
    names = labels or s.names(column)[sel]
    for j in range(mean.shape[1]):
        ax.plot(tau, mean[:, j], lw=1.2, label=names[j])
        ax.fill_between(tau, mean[:, j] - std[:, j], mean[:, j] + std[:, j], alpha=0.15)
    ax.set_title(f"{title} (mean ± std over {s.n_episodes} episodes)", fontsize=10)
    ax.set_xlabel("normalized episode time", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=6, ncol=min(4, mean.shape[1]))
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def panel_task_objects(ax, s):
    """Height of the picked object and of the destination over normalized time, averaged over episodes."""
    if "observation.objects" not in s.columns or not s.by_episode:
        ax.set_axis_off()
        return
    names = s.names("observation.objects")
    z_index = {n.split(".")[0]: i for i, n in enumerate(names) if n.endswith(".z")}
    src, dst = [], []
    for e in range(s.n_episodes):
        a = s.by_episode.get(e)
        if not a:
            continue
        objs = s.episode_columns(e)["observation.objects"]
        src.append(resample(objs[:, z_index[a["task"]["source"]]], N_TIME))
        dst.append(resample(objs[:, z_index[a["task"]["destination"]]], N_TIME))
    tau = np.linspace(0, 1, N_TIME)
    for arr, label, color in ((np.array(src), "picked object", "tab:red"), (np.array(dst), "destination", "tab:blue")):
        ax.plot(tau, arr.mean(0), color=color, lw=1.4, label=label)
        ax.fill_between(tau, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=color, alpha=0.15)
    ax.set_title("Object height above the table (mean ± std)", fontsize=10)
    ax.set_xlabel("normalized episode time", fontsize=8)
    ax.set_ylabel("z (m)", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def panel_pca(ax, fig, s):
    keys = [k for k in s.numeric if k.startswith("observation.") and not k.startswith("observation.images.")]
    X = np.concatenate([s.columns[k] for k in keys], axis=1).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    pcs = X @ vt[:2].T
    explained = sv[:2] ** 2 / (sv ** 2).sum()
    progress = np.concatenate([np.linspace(0, 1, s.episode(e)["length"]) for e in range(s.n_episodes)])
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=progress, cmap="viridis", s=4, alpha=0.7)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("episode progress", fontsize=8)
    ax.set_title(f"PCA of the {X.shape[1]}-D observation state, {len(X)} frames\n"
                 f"PC1 {100 * explained[0]:.0f}%, PC2 {100 * explained[1]:.0f}% of variance", fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)


def panel_lengths(ax, s):
    lengths = np.array([s.episode(e)["length"] for e in range(s.n_episodes)]) / s.fps
    ax.hist(lengths, bins=min(10, max(3, s.n_episodes)), color="tab:purple", alpha=0.8)
    ax.set_title(f"Episode duration (s), mean {lengths.mean():.1f}", fontsize=10)
    ax.set_xlabel("seconds", fontsize=8)
    ax.tick_params(labelsize=7)


def panel_info(ax, s):
    m = s.metadata
    v = m.get("versions", {})
    hw = m.get("hardware", {})
    frames = m.get("frames", sum(s.episode(e)["length"] for e in range(s.n_episodes)))
    rate = m.get("success_rate")
    lines = [
        f"session     {s.name}",
        f"created     {m.get('created_at', '?')}",
        f"seed        {m.get('seed', '?')}     project v{m.get('project_version', '?')}",
        f"versions    genesis {v.get('genesis', '?')}, lerobot {v.get('lerobot', '?')}, torch {v.get('torch', '?')}",
        f"hardware    {hw.get('cpu', '?')}, {hw.get('cores', '?')} cores, {hw.get('ram_gb', '?')} GB",
        f"gpu         {hw.get('gpu') or 'none'}   genesis backend {hw.get('genesis_backend', '?')}",
        f"dataset     {m.get('repo_id', 'local')}",
        f"episodes    {s.n_episodes} recorded of {m.get('episodes_attempted', '?')} attempted"
        + (f", success rate {rate:.2f}" if isinstance(rate, (int, float)) else ""),
        f"frames      {frames} at {s.fps} fps, {m.get('n_envs', '?')} envs x {m.get('epochs', '?')} epochs",
        f"held out    {', '.join(' → '.join(p) for p in m.get('holdout_pairs', [])) or 'none'}",
        f"cameras     {', '.join(m.get('cameras', {}).keys()) if isinstance(m.get('cameras'), dict) else ', '.join(m.get('cameras', []))}",
        f"features    {len(s.numeric)} numeric, {len(s.cameras)} video",
        f"wall time   {m.get('wall_time_s', '?')} s",
    ]
    ax.set_axis_off()
    ax.text(0, 1, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=7.5, transform=ax.transAxes)


# --------------------------------- figure ---------------------------------

def build_figure(s):
    fig = plt.figure(figsize=(13, 7.5))
    fig.suptitle(f"Dataset summary: {s.name}", fontsize=11)
    gs = fig.add_gridspec(3, 3, left=0.05, right=0.97, top=0.92, bottom=0.07, hspace=0.6, wspace=0.3)

    panel_info(fig.add_subplot(gs[0, 0]), s)
    panel_outcomes(fig.add_subplot(gs[0, 1]), s)
    panel_pairs(fig.add_subplot(gs[0, 2]), s)
    panel_spawns(fig.add_subplot(gs[1, 0]), s)
    panel_mean_lines(fig.add_subplot(gs[1, 1]), s, "observation.state", slice(None), "Joint positions", ylabel="rad")
    panel_mean_lines(fig.add_subplot(gs[1, 2]), s, "observation.tcp_pose", slice(0, 3), "Tool position", labels=["x", "y", "z"], ylabel="m")
    panel_task_objects(fig.add_subplot(gs[2, 0]), s)
    panel_pca(fig.add_subplot(gs[2, 1]), fig, s)
    panel_lengths(fig.add_subplot(gs[2, 2]), s)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Summary plots for a recorded session.")
    parser.add_argument("session", help="session directory, e.g. datasets/piper_stack_20260924_182109")
    args = parser.parse_args()

    s = SessionData(args.session)
    fig = build_figure(s)
    out = s.dir / "summary.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
