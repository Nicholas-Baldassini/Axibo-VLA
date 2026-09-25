"""Coordinate helpers for the table frame used throughout config.py."""


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
