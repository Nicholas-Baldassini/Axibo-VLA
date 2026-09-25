"""
Language-conditioned stacking scene for the Piper X arm.

The scene is built entirely in code from Genesis primitives; only the robot comes
from a URDF. All positions, sizes, colors, cameras and gains live in config.py.

Geometry (table, object shapes, colors, cameras) is shared by every environment and
comes from the base config. Per-environment state (object positions, robot home pose)
comes from env_config(config, i) and is applied after scene.build().

Run from anywhere:  python source/stacking_scene.py
"""

import random

import numpy as np
import torch
import genesis as gs

from config import CONFIG, env_config
from utils.cameras import vertical_fov
from utils.frames import table_to_world, object_yaw


# ------------------------------ coordinates ------------------------------

def object_spawn_pos(config, spec):
    """World position that puts an object resting on the table surface."""
    x, y = spec["pos"]
    half_height = spec["size"] / 2 if spec["shape"] == "box" else spec["height"] / 2
    return table_to_world(config, (x, y, half_height + config["scene"]["spawn_gap"]))


# ---------------------------------- init ----------------------------------

def init_genesis(config, backend=None):
    """Initialize Genesis with the config seed and seed python, numpy and torch the same way."""
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gs.init(backend=backend or gs.cpu, seed=seed)


# -------------------------------- entities --------------------------------

def add_object(scene, config, name, spec):
    pos = object_spawn_pos(config, spec)
    if spec["shape"] == "box":
        s = spec["size"]
        morph = gs.morphs.Box(pos=pos, size=(s, s, s))
    elif spec["shape"] == "cylinder":
        morph = gs.morphs.Cylinder(pos=pos, radius=spec["radius"], height=spec["height"])
    else:
        raise ValueError(f"unknown shape {spec['shape']!r} for object {name!r}")

    return scene.add_entity(
        morph,
        material=gs.materials.Rigid(rho=config["scene"]["object_density"], friction=config["scene"]["object_friction"]),
        surface=gs.surfaces.Default(color=spec["color"]),
        name=name,
    )


def add_cameras(scene, robot, config):
    """One RGB camera sensor per entry in config["cameras"]. Each read() returns (n_envs, H, W, 3)."""
    cameras = {}
    for name, cam in config["cameras"].items():
        kwargs = dict(
            res=cam["res"], fov=vertical_fov(cam), up=cam["up"],
            near=cam.get("near", 0.01), far=cam.get("far", 100.0),
            draw_debug=config["viewer"]["draw_camera_frustums"],
        )
        if "attach_link" in cam:
            # Attached camera: pos and lookat are in the link's frame and follow it.
            link = robot.get_link(cam["attach_link"])
            kwargs.update(entity_idx=robot.idx, link_idx_local=link.idx_local, pos=cam["pos"], lookat=cam["lookat"])
        else:
            kwargs.update(pos=table_to_world(config, cam["pos"]), lookat=table_to_world(config, cam["lookat"]))
        cameras[name] = scene.add_sensor(gs.sensors.RasterizerCameraOptions(**kwargs))
    return cameras


def build_scene(config, show_viewer=True):
    """Create the scene and add every entity and camera. Call scene.build(n_envs=...) afterwards."""
    viewer = config["viewer"]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config["sim"]["dt"]),
        viewer_options=gs.options.ViewerOptions(
            res=viewer["res"],
            camera_pos=viewer["camera_pos"],
            camera_lookat=viewer["camera_lookat"],
            camera_fov=viewer["camera_fov"],
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=viewer["world_frame_size"],
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    table_cfg = config["scene"]["table"]
    table = scene.add_entity(
        gs.morphs.Box(pos=table_cfg["pos"], size=table_cfg["size"], fixed=True),
        material=gs.materials.Rigid(friction=table_cfg["friction"]),
        surface=gs.surfaces.Default(color=table_cfg["color"]),
        name="table",
    )

    objects = {name: add_object(scene, config, name, spec) for name, spec in config["scene"]["objects"].items()}

    robot_cfg = config["scene"]["robot"]
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=robot_cfg["urdf"],
            pos=table_to_world(config, robot_cfg["pos"]),
            euler=robot_cfg["euler"],
            fixed=True,
        ),
        name="piper_x",
    )

    cameras = add_cameras(scene, robot, config)
    return scene, robot, table, objects, cameras


# ------------------------------ per-env state ------------------------------

def make_env_configs(config, n_envs):
    return [env_config(config, i) for i in range(n_envs)]


def robot_dofs(robot, config):
    """Local DOF indices for the arm joints followed by the gripper joints."""
    names = config["control"]["arm_joints"] + config["control"]["gripper_joints"]
    return [robot.get_joint(name).dofs_idx_local[0] for name in names]


def apply_env_configs(robot, objects, env_configs):
    """Set per-environment state: object poses, robot gains and home pose. Call after scene.build()."""
    for name, ent in objects.items():
        specs = [c["scene"]["objects"][name] for c in env_configs]
        ent.set_pos(np.array([object_spawn_pos(c, s) for c, s in zip(env_configs, specs)]))
        half = np.array([object_yaw(s) / 2 for s in specs])
        ent.set_quat(np.stack([np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)], axis=1))

    # Gains and force limits are shared by every env in Genesis, so they come from the first config.
    ctrl = env_configs[0]["control"]
    dofs_idx = robot_dofs(robot, env_configs[0])
    robot.set_dofs_kp(kp=ctrl["kp"], dofs_idx_local=dofs_idx)
    robot.set_dofs_kv(kv=ctrl["kv"], dofs_idx_local=dofs_idx)
    robot.set_dofs_force_range(lower=-ctrl["force_limit"], upper=ctrl["force_limit"], dofs_idx_local=dofs_idx)

    home = np.array([c["control"]["home_qpos"] for c in env_configs])
    robot.set_dofs_position(home, dofs_idx)       # teleport to home
    robot.control_dofs_position(home, dofs_idx)   # then hold it with the PD controller
    return dofs_idx


def zero_velocity(entity, n_envs):
    if hasattr(entity, "zero_all_dofs_velocity"):
        entity.zero_all_dofs_velocity()
    else:
        entity.set_dofs_velocity(np.zeros((n_envs, entity.n_dofs)))


def reset_envs(robot, objects, env_configs):
    """Put every env back to its config's start state: objects at their spawn poses and at rest,
    robot at home and at rest. Returns the robot DOF indices."""
    n = len(env_configs)
    dofs_idx = apply_env_configs(robot, objects, env_configs)
    for ent in objects.values():
        zero_velocity(ent, n)
    zero_velocity(robot, n)
    return dofs_idx


def main(show_viewer=None):
    """Build the scene with every env at its start state and let it run. show_viewer overrides config["viewer"]["show"]."""
    config = CONFIG
    n_envs = config["ik"]["n_envs"]
    if show_viewer is None:
        show_viewer = config["viewer"]["show"]

    init_genesis(config)
    scene, robot, table, objects, cameras = build_scene(config, show_viewer=show_viewer)
    scene.build(n_envs=n_envs, env_spacing=config["ik"]["env_spacing"])

    env_configs = make_env_configs(config, n_envs)
    dofs_idx = apply_env_configs(robot, objects, env_configs)
    for _ in range(config["sim"]["n_steps"]):
        scene.step()

    return scene, robot, table, objects, cameras, dofs_idx


if __name__ == "__main__":
    main()
