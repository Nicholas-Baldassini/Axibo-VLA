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
from functools import partial
from pathlib import Path

import numpy as np
import torch
import genesis as gs
from genesis.utils.misc import tensor_to_array

from config import CONFIG, env_config
from utils.cameras import vertical_fov
from utils.frames import table_to_world


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


def _read_env_rgb(camera, env_idx):
    return camera.read(envs_idx=env_idx).rgb


def add_video_recorders(scene, cameras, n_envs, config):
    """Record <video_dir>/<camera>_env<i>.mp4 for every camera and env while the sim steps.

    Must be called before scene.build(). Recording starts on build; call scene.stop_recording()
    when done so the files are flushed.
    """
    out_dir = Path(config["preview"]["video_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, cam in cameras.items():
        for i in range(n_envs):
            scene.add_recorder(
                data_func=partial(_read_env_rgb, cam, i),
                rec_options=gs.recorders.VideoFile(
                    filename=str(out_dir / f"{name}_env{i}.mp4"),
                    hz=config["preview"]["video_hz"],
                ),
            )
    return out_dir


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
    if config["preview"]["record_video"]:
        add_video_recorders(scene, cameras, config["ik"]["n_envs"], config)

    return scene, robot, table, objects, cameras


# ------------------------------ per-env state ------------------------------

def make_env_configs(config, n_envs):
    return [env_config(config, i) for i in range(n_envs)]


def robot_dofs(robot, config):
    """Local DOF indices for the arm joints followed by the gripper joints."""
    names = config["control"]["arm_joints"] + config["control"]["gripper_joints"]
    return [robot.get_joint(name).dofs_idx_local[0] for name in names]


def apply_env_configs(robot, objects, env_configs):
    """Set per-environment state: object positions, robot gains and home pose. Call after scene.build()."""
    for name, ent in objects.items():
        pos = np.array([object_spawn_pos(c, c["scene"]["objects"][name]) for c in env_configs])
        ent.set_pos(pos)

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
    """Put every env back to its config's start state: objects upright at their spawn positions and
    at rest, robot at home and at rest. Returns the robot DOF indices."""
    n = len(env_configs)
    dofs_idx = apply_env_configs(robot, objects, env_configs)
    for ent in objects.values():
        ent.set_quat(np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)))
        zero_velocity(ent, n)
    zero_velocity(robot, n)
    return dofs_idx


# -------------------------------- previews --------------------------------

def save_camera_previews(cameras, out_dir):
    """Write one PNG per camera with every env's frame side by side. Returns the file paths."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, cam in cameras.items():
        rgb = tensor_to_array(cam.read().rgb)
        if rgb.ndim == 3:                     # unbatched scene: add the env axis
            rgb = rgb[None]
        sheet = np.concatenate(list(rgb), axis=1)
        path = out_dir / f"{name}.png"
        plt.imsave(path, sheet)
        paths.append(path)
    return paths


def main(show_viewer=None):
    """show_viewer overrides config["viewer"]["show"] when given."""
    config = CONFIG
    n_envs = config["ik"]["n_envs"]
    if show_viewer is None:
        show_viewer = config["viewer"]["show"]

    init_genesis(config)
    scene, robot, table, objects, cameras = build_scene(config, show_viewer=show_viewer)
    scene.build(n_envs=n_envs, env_spacing=config["ik"]["env_spacing"])

    env_configs = make_env_configs(config, n_envs)
    dofs_idx = apply_env_configs(robot, objects, env_configs)

    for _ in range(50):
        scene.step()
    for path in save_camera_previews(cameras, config["preview"]["dir"]):
        print(f"saved {path}")

    for _ in range(config["sim"]["n_steps"] - 50):
        scene.step()

    if config["preview"]["record_video"]:
        scene.stop_recording()
        print(f"videos written to {config['preview']['video_dir']}")

    return scene, robot, table, objects, cameras, dofs_idx


if __name__ == "__main__":
    main()
