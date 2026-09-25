"""
Camera models: intrinsics <-> field of view, so a configured camera can replicate a real one.

A camera entry in config["cameras"] has
    res          (width, height) in pixels
    fov          vertical field of view in degrees, used when no intrinsics are given
    intrinsics   optional {"fx", "fy", "cx", "cy"} in pixels for `res`, e.g. from a RealSense's
                 rs-enumerate-devices -c or an OpenCV calibration; the render FOV is derived from fy
    near, far    clip planes in meters
    pos, lookat, up   extrinsics (table coords, or the link frame when attach_link is set)

The rasterizer is an ideal pinhole with square pixels and a centered principal point. Real cameras
that deviate from that (fx != fy, off-center cx/cy, lens distortion) can only be matched
approximately: undistort and center-crop real images to the pinhole model instead.
"""

import math
import warnings

import numpy as np


def intrinsics_matrix(cam):
    """3x3 K for a camera config, from its intrinsics if given, else from fov and resolution."""
    w, h = cam["res"]
    intr = cam.get("intrinsics")
    if intr:
        fx, fy = float(intr["fx"]), float(intr["fy"])
        cx, cy = float(intr.get("cx", w / 2)), float(intr.get("cy", h / 2))
    else:
        fx = fy = (h / 2) / math.tan(math.radians(cam["fov"]) / 2)
        cx, cy = w / 2, h / 2
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def vertical_fov(cam):
    """Vertical FOV in degrees for the renderer, derived from intrinsics when present."""
    intr = cam.get("intrinsics")
    if not intr:
        return float(cam["fov"])
    w, h = cam["res"]
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr.get("cx", w / 2)), float(intr.get("cy", h / 2))
    if abs(fx - fy) > 0.01 * fy:
        warnings.warn(f"camera: fx={fx:.1f} and fy={fy:.1f} differ; the renderer uses fy and square pixels, "
                      f"so the horizontal FOV will be off by {100 * abs(fx - fy) / fy:.1f}%")
    if abs(cx - w / 2) > 0.01 * w or abs(cy - h / 2) > 0.01 * h:
        warnings.warn(f"camera: principal point ({cx:.1f}, {cy:.1f}) is off-center for res {w}x{h}; "
                      f"the renderer uses a centered principal point")
    return math.degrees(2 * math.atan((h / 2) / fy))


def horizontal_fov(cam):
    w, _ = cam["res"]
    return math.degrees(2 * math.atan((w / 2) / intrinsics_matrix(cam)[0, 0]))


def describe(cam):
    """Everything about a camera worth storing next to a dataset, JSON-ready."""
    w, h = cam["res"]
    K = intrinsics_matrix(cam)
    return {
        "res": [int(w), int(h)],
        "model": "pinhole",
        "intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
        "K": K.tolist(),
        "fov_vertical_deg": vertical_fov(cam),
        "fov_horizontal_deg": horizontal_fov(cam),
        "near": cam.get("near"),
        "far": cam.get("far"),
        "attach_link": cam.get("attach_link"),
        "frame": "link" if "attach_link" in cam else "table",
        "pos": list(cam["pos"]),
        "lookat": list(cam["lookat"]),
        "up": list(cam["up"]),
    }
