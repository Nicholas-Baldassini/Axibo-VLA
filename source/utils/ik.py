"""
Batched inverse kinematics helpers for the Piper X.

Targets are given for the tool center point (TCP), the spot between the fingertips.
The URDF has no link there, so a TCP target is converted to a pose for the wrist link
(config["ik"]["ee_link"]) using config["ik"]["tcp_offset"] before calling the solver.
"""

import numpy as np


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def quat_to_mat(q):
    """(n, 4) quaternions in (w, x, y, z) to (n, 3, 3) rotation matrices."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def tcp_to_link_pos(tcp_pos, tcp_quat, tcp_offset):
    """Where the wrist link must be for the TCP to sit at tcp_pos with orientation tcp_quat."""
    offset_world = quat_to_mat(tcp_quat) @ np.asarray(tcp_offset, dtype=float)
    return np.asarray(tcp_pos, dtype=float) - offset_world


def tcp_pos_from_link(ee_link, tcp_offset):
    """Current TCP position in every env, (n_envs, 3), from the wrist link's pose."""
    link_pos = to_numpy(ee_link.get_pos())
    link_quat = to_numpy(ee_link.get_quat())
    return link_pos + quat_to_mat(link_quat) @ np.asarray(tcp_offset, dtype=float)


def solve_ik(robot, ee_link, tcp_pos, tcp_quat, ik_cfg):
    """Batched IK. tcp_pos: (n_envs, 3) world, tcp_quat: (n_envs, 4). Returns qpos of shape (n_envs, n_dofs)."""
    link_pos = tcp_to_link_pos(tcp_pos, tcp_quat, ik_cfg["tcp_offset"])
    return robot.inverse_kinematics(
        link=ee_link,
        pos=link_pos,
        quat=np.asarray(tcp_quat, dtype=float),
        rot_mask=list(ik_cfg["rot_mask"]),
    )


def move_tcp(robot, ee_link, arm_dofs, tcp_pos, tcp_quat, ik_cfg):
    """Solve IK for every env and send the arm there in one go. Gripper joints are left untouched."""
    q = solve_ik(robot, ee_link, tcp_pos, tcp_quat, ik_cfg)
    q_arm = q[:, arm_dofs]
    if ik_cfg["apply"] == "set":
        robot.set_dofs_position(q_arm, arm_dofs)
    robot.control_dofs_position(q_arm, arm_dofs)
    return q
