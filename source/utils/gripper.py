"""
Gripper geometry measured from the URDF, so the tool center point (TCP) sits between the finger pads.

The TCP is expressed in the wrist link frame (the link the finger joints hang off). Its z is the
center of the finger pads along the finger direction, and x, y are 0 because the fingers are
symmetric about the wrist axis.

    python -m utils.gripper      prints the measurements for the URDF in config.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _resolve_mesh(urdf_path, filename):
    urdf_path = Path(urdf_path).resolve()
    if filename.startswith("package://"):
        pkg, rest = filename[len("package://"):].split("/", 1)
        for parent in urdf_path.parents:
            if parent.name == pkg and (parent / rest).exists():
                return parent / rest
            if (parent / pkg / rest).exists():
                return parent / pkg / rest
        raise FileNotFoundError(f"cannot resolve {filename} relative to {urdf_path}")
    return (urdf_path.parent / filename).resolve()


def _floats(text, default):
    return np.array([float(v) for v in (text or default).split()], dtype=float)


def _origin(elem):
    """(R, t) of a URDF <origin> child, identity when absent. URDF rpy is extrinsic x-y-z."""
    o = None if elem is None else elem.find("origin")
    xyz = _floats(None if o is None else o.get("xyz"), "0 0 0")
    rpy = _floats(None if o is None else o.get("rpy"), "0 0 0")
    return Rotation.from_euler("xyz", rpy).as_matrix(), xyz


def finger_geometry(urdf_path, finger_joints=("joint7", "joint8")):
    """Collision-mesh vertices of each finger link expressed in the wrist (parent) link frame at q = 0."""
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    links = {l.get("name"): l for l in root.findall("link")}

    fingers = {}
    for name in finger_joints:
        j = joints[name]
        R_j, t_j = _origin(j)
        child = links[j.find("child").get("link")]
        collision = child.find("collision")
        mesh_el = collision.find("geometry").find("mesh")
        mesh = trimesh.load(_resolve_mesh(urdf_path, mesh_el.get("filename")), force="mesh")
        if mesh_el.get("scale"):
            mesh.apply_scale(_floats(mesh_el.get("scale"), "1 1 1"))
        R_c, t_c = _origin(collision)
        verts = (R_j @ (R_c @ mesh.vertices.T + t_c[:, None]) + t_j[:, None]).T
        limit = j.find("limit")
        fingers[name] = {
            "link": child.get("name"),
            "parent": j.find("parent").get("link"),
            "verts": verts,
            "axis": R_j @ _floats(j.find("axis").get("xyz"), "0 0 1"),   # sliding direction in the wrist frame
            "lower": float(limit.get("lower", 0.0)),
            "upper": float(limit.get("upper", 0.0)),
        }
    return fingers


def measure_gripper(urdf_path, finger_joints=("joint7", "joint8"), pad_tol=1e-3):
    """Where the finger pads are, in the wrist frame, plus the closed and open gap between them."""
    fingers = finger_geometry(urdf_path, finger_joints)
    (na, a), (nb, b) = fingers.items()

    # Closing direction: from finger a toward finger b, in the wrist x-y plane.
    d = b["verts"].mean(axis=0) - a["verts"].mean(axis=0)
    d[2] = 0.0
    d /= np.linalg.norm(d)

    pads = {}
    inner = {}
    for name, f, sign in ((na, a, 1.0), (nb, b, -1.0)):
        toward_other = sign * d
        proj = f["verts"] @ toward_other
        face = f["verts"][proj > proj.max() - pad_tol]        # the face closest to the other finger
        pads[name] = (float(face[:, 2].min()), float(face[:, 2].max()))
        inner[name] = float(proj.max())                        # inner face position along toward_other
        travel = f["axis"] @ (-toward_other) * np.array([f["lower"], f["upper"]])   # displacement away from the other finger
        f["q_closed"], f["q_open"] = (f["lower"], f["upper"]) if travel[0] <= travel[1] else (f["upper"], f["lower"])
        f["opening"] = float(travel.max() - travel.min())

    closed_gap = -(inner[na] + inner[nb])                      # distance between inner faces at q = 0
    pad_center_z = float(np.mean([np.mean(pads[na]), np.mean(pads[nb])]))
    z_all = np.concatenate([a["verts"][:, 2], b["verts"][:, 2]])
    return {
        "wrist_link": a["parent"],
        "tcp_offset": (0.0, 0.0, round(pad_center_z, 4)),
        "pad_z_range": {n: tuple(round(v, 4) for v in pads[n]) for n in (na, nb)},
        "finger_z_range": (round(float(z_all.min()), 4), round(float(z_all.max()), 4)),
        "closed_gap": round(closed_gap, 4),
        "open_gap": round(closed_gap + a["opening"] + b["opening"], 4),
        "gripper_closed_qpos": (a["q_closed"], b["q_closed"]),
        "gripper_open_qpos": (a["q_open"], b["q_open"]),
    }


if __name__ == "__main__":
    from config import CONFIG

    for k, v in measure_gripper(CONFIG["scene"]["robot"]["urdf"], CONFIG["control"]["gripper_joints"]).items():
        print(f"{k}: {v}")
