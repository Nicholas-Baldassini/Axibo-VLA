"""
Replay one demonstration from a recorded session.

Left: every camera video of the episode in a grid, played at the dataset's frame rate and looped.
Right: the session and episode info, and a few signal plots with a moving time cursor: joint
positions (observed solid, commanded dashed), joint velocities, tool position, gripper opening,
and the height of every object.

    python source/visualization/visualize_dataset.py datasets/piper_stack_20260924_182109

Keys:  left / right  previous / next demonstration      space  pause / resume      r  restart
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SessionData, decode_segment, short_camera_name  # noqa: E402

# ----------------------------------- settings -----------------------------------
START_EPISODE = 0        # demonstration shown first (dataset episode index)
PLAYBACK_SPEED = 1.0     # 1.0 = real time
DISPLAY_SCALE = 0.5      # decoded frames are shrunk by this factor for display
# ---------------------------------------------------------------------------------

# Which signals to plot. Each graph: title, list of (column, line style, slice of that column, legend prefix).
# A graph is skipped when none of its columns exist in the dataset.
GRAPHS = [
    ("Joint positions (rad)", [("observation.state", "-", slice(None), ""), ("action.joint_pos", "--", slice(None), None)]),
    ("Joint velocities (rad/s)", [("observation.joint_vel", "-", slice(None), "")]),
    ("Tool position (m)", [("observation.tcp_pose", "-", slice(0, 3), ""), ("action", "--", slice(0, 3), None)]),
    ("Gripper opening", [("observation.tcp_pose", "-", slice(7, 8), "observed"), ("action", "--", slice(7, 8), "commanded")]),
    ("Object height (m)", [("observation.objects", "-", slice(2, None, 7), "")]),
]

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class Viewer:
    def __init__(self, session, start_episode=0, speed=1.0, scale=0.5):
        self.s = session
        self.speed = speed
        self.scale = scale
        self.paused = False
        self.k = 0
        self.graphs = [(title, [c for c in cols if c[0] in self.s.columns]) for title, cols in GRAPHS]
        self.graphs = [(t, c) for t, c in self.graphs if c]
        self._build_figure()
        self.load_episode(start_episode % self.s.n_episodes)

    # ---- layout ----

    def _build_figure(self):
        cams = self.s.cameras
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title(f"{self.s.name}") if self.fig.canvas.manager else None
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], left=0.02, right=0.98, top=0.95, bottom=0.06, wspace=0.10)

        cols = 1 if len(cams) <= 2 else 2
        rows = max(1, math.ceil(len(cams) / cols))
        left = gs[0].subgridspec(rows, cols, hspace=0.12, wspace=0.03)
        self.images = {}
        for i, cam in enumerate(cams):
            ax = self.fig.add_subplot(left[i // cols, i % cols])
            ax.set_axis_off()
            ax.set_title(short_camera_name(cam), fontsize=10, pad=4)
            self.images[cam] = ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8), interpolation="nearest")

        n_graphs = len(self.graphs)
        right = gs[1].subgridspec(1 + n_graphs, 1, height_ratios=[1.6] + [1.0] * n_graphs, hspace=0.55)
        self.info_ax = self.fig.add_subplot(right[0])
        self.info_ax.set_axis_off()
        self.info_text = self.info_ax.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9, transform=self.info_ax.transAxes)
        self.graph_axes = []
        first = None
        for i in range(n_graphs):
            ax = self.fig.add_subplot(right[1 + i], sharex=first)
            first = first or ax
            self.graph_axes.append(ax)
        self.cursors = []

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ---- data ----

    def load_episode(self, e):
        self.e = e
        self.ep = self.s.episode(e)
        self.cols = self.s.episode_columns(e)
        T = self.ep["length"]
        print(f"loading episode {e} ({T} frames) ...", flush=True)
        self.frames = {}
        for cam in self.s.cameras:
            path, t0, t1 = self.s.video_segment(e, cam)
            self.frames[cam] = decode_segment(path, t0, t1, scale=self.scale, expected=T)
        self.t = np.arange(T) / self.s.fps
        self.k = 0
        self._draw_graphs()
        self.render_frame(0)

    def _draw_graphs(self):
        for line in self.cursors:
            line.remove()
        self.cursors = []
        for ax, (title, cols) in zip(self.graph_axes, self.graphs):
            ax.cla()
            for column, style, sel, prefix in cols:
                y = self.cols[column][:, sel]
                names = self.s.names(column)[sel]
                for j in range(y.shape[1]):
                    label = None if prefix is None else (prefix if prefix and y.shape[1] == 1 else names[j].split(".")[0] if column == "observation.objects" else names[j])
                    ax.plot(self.t, y[:, j], style, color=COLORS[j % len(COLORS)], lw=1.0, label=label)
            ax.set_title(title, fontsize=9, loc="left", pad=2)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=6, ncol=min(4, len(labels)), loc="upper right", framealpha=0.6)
            self.cursors.append(ax.axvline(0.0, color="k", lw=0.9))
        if self.graph_axes:
            self.graph_axes[-1].set_xlabel("time (s)", fontsize=8)
            self.graph_axes[-1].set_xlim(0, self.t[-1] if len(self.t) > 1 else 1)

    # ---- drawing ----

    def _info(self):
        a = self.ep["attempt"] or {}
        m = self.s.metadata
        result = ("success" if a.get("success") else "failed") + (f" ({a['reason']})" if a.get("reason") else "") if a else "n/a"
        lines = [
            f"session   {self.s.name}",
            f"dataset   {m.get('repo_id', 'local')}   {self.s.n_episodes} episodes, {self.s.fps} fps",
            f"episode   {self.e} / {self.s.n_episodes - 1}     env {a.get('env', '?')}   epoch {a.get('epoch', '?')}",
            f"task      {self.ep['task']}",
            f"result    {result}   {self.ep['length']} frames, {self.ep['length'] / self.s.fps:.1f} s",
            f"frame     {self.k} / {self.ep['length'] - 1}   t = {self.k / self.s.fps:5.2f} s" + ("   [paused]" if self.paused else ""),
            "",
            "keys      left/right episode   space pause   r restart",
        ]
        return "\n".join(lines)

    def render_frame(self, k):
        self.k = k
        for cam, im in self.images.items():
            im.set_data(self.frames[cam][k])
            h, w = self.frames[cam].shape[1:3]
            im.set_extent((0, w, h, 0))
        tk = self.t[k]
        for line in self.cursors:
            line.set_xdata([tk, tk])
        self.info_text.set_text(self._info())

    # ---- interaction ----

    def _tick(self, _):
        if not self.paused:
            self.render_frame((self.k + 1) % self.ep["length"])
        return []

    def _on_key(self, event):
        if event.key == "right":
            self.load_episode((self.e + 1) % self.s.n_episodes)
        elif event.key == "left":
            self.load_episode((self.e - 1) % self.s.n_episodes)
        elif event.key == " ":
            self.paused = not self.paused
            self.info_text.set_text(self._info())
        elif event.key == "r":
            self.render_frame(0)
        self.fig.canvas.draw_idle()

    def run(self):
        interval_ms = 1000.0 / (self.s.fps * self.speed)
        self.anim = FuncAnimation(self.fig, self._tick, interval=interval_ms, blit=False, cache_frame_data=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Replay demonstrations from a recorded session.")
    parser.add_argument("session", help="session directory, e.g. datasets/piper_stack_20260924_182109")
    args = parser.parse_args()

    # matplotlib binds left/right to toolbar navigation by default; free them for episode switching
    for key in ("keymap.back", "keymap.forward"):
        matplotlib.rcParams[key] = [k for k in matplotlib.rcParams[key] if k not in ("left", "right")]

    session = SessionData(args.session)
    Viewer(session, start_episode=START_EPISODE, speed=PLAYBACK_SPEED, scale=DISPLAY_SCALE).run()


if __name__ == "__main__":
    main()
