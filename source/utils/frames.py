"""Coordinate helpers for the table frame used throughout config.py."""

import numpy as np


def table_top_z(config):
    table = config["scene"]["table"]
    return table["pos"][2] + table["size"][2] / 2


def table_to_world(config, pos):
    """Table coords (x, y from the table center, z above the surface) to the env's world frame."""
    tx, ty, _ = config["scene"]["table"]["pos"]
    x, y, z = pos
    return (tx + x, ty + y, table_top_z(config) + z)


def object_height(spec):
    return spec["size"] if spec["shape"] == "box" else spec["height"]


def object_center(scene, name):
    """Center of an object resting on the table, in table coords."""
    spec = scene["objects"][name]
    x, y = spec["pos"]
    return (float(x), float(y), object_height(spec) / 2)


def object_yaw(spec):
    """The object's yaw about z in radians (config "yaw" is in degrees, 0 when absent)."""
    return float(np.radians(spec.get("yaw", 0.0)))


def grasp_yaw(spec):
    """Wrist yaw that closes the fingers across the object's faces, in radians.

    A box is symmetric under 90 degree turns, so its yaw is wrapped into (-45, 45] degrees and
    the wrist turns as little as possible; a cylinder is axisymmetric and needs no turn."""
    if spec["shape"] != "box":
        return 0.0
    quarter = np.pi / 2
    return float((object_yaw(spec) + quarter / 2) % quarter - quarter / 2)
