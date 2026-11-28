"""Per-sample domain randomization (decision D4), all draws from one seed.

Pure sampling is testable without Blender; the bpy layer applies a sample to the open greenhouse:

    blender -b assets/Harvix_Greenhouse.blend --python src/randomize.py -- --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_control import RAIL_LANES, apply_pose, clamp_pose, load_spec, place_root, preset  # noqa: E402

# Plants span y = 3.4 .. 116.6; keep the robot 3 m inside so the camera sees crop, not end walls.
Y_RANGE = (6.5, 113.5)
# Wrist angle that turns each arm camera to the crop wall on that side (measured in `neutral`).
WRIST_TO_WALL = {"+x": -math.pi / 2, "-x": math.pi / 2}
# Shoulder, elbow and wrist all yaw the arm about world Z. Measured: on the left arm they turn
# the same way, on the right arm the wrist turns opposite to shoulder and elbow.
WRIST_YAW_SIGN = {"left": 1, "right": -1}
# Working envelope: offsets around `neutral`, kept small so the camera stays on the crop wall.
ENVELOPE = {"wrist": 0.35, "shoulder": 0.4, "elbow": 0.4, "fingers": 0.35}
SUN_ELEVATION_DEG = (15.0, 75.0)
SUN_ENERGY = (1.5, 5.0)  # scene default 3.2
WORLD_STRENGTH = (0.6, 1.4)
EXPOSURE = (-0.7, 0.7)
# Scene fruit is mostly ripe (0.92-1.0); shifting down yields turning and unripe trusses too.
RIPE_SHIFT = (-0.6, 0.05)


def _u(rng: random.Random, lo_hi: tuple[float, float]) -> float:
    return rng.uniform(*lo_hi)


def sample_pose(rng: random.Random, spec: dict) -> tuple[dict, dict]:
    """Arm pose in the working envelope, both arms facing a randomly chosen crop wall."""
    pose = preset("neutral", spec)
    walls = {}
    for side in ("left", "right"):
        wall = rng.choice(sorted(WRIST_TO_WALL))
        walls[side] = wall
        lift = spec["joint_limits"][f"{side}_lift"]
        pose[f"{side}_lift"] = rng.uniform(lift["lower"], lift["upper"])
        offsets = {j: rng.uniform(-1, 1) * ENVELOPE[j] for j in ("shoulder", "elbow", "fingers")}
        for joint, off in offsets.items():
            pose[f"{side}_{joint}"] += off
        # The wrist cancels the shoulder and elbow yaw, so the camera stays on the chosen wall.
        arm_yaw = offsets["shoulder"] + offsets["elbow"]
        noise = rng.uniform(-1, 1) * ENVELOPE["wrist"]
        pose[f"{side}_wrist"] = WRIST_TO_WALL[wall] + noise - WRIST_YAW_SIGN[side] * arm_yaw
    pose, _ = clamp_pose(pose, spec)
    return pose, walls


def sample(seed: int, spec: dict) -> dict:
    """All random draws for one sample. Same seed, same sample."""
    rng = random.Random(seed)
    lane = rng.choice(RAIL_LANES)
    y = round(rng.uniform(*Y_RANGE), 4)
    pose, walls = sample_pose(rng, spec)
    return {
        "seed": seed,
        "robot": {"lane_x": lane, "y": y},
        "pose": pose,
        "walls": walls,
        "light": {
            "sun_elevation_deg": _u(rng, SUN_ELEVATION_DEG),
            "sun_azimuth_deg": rng.uniform(0.0, 360.0),
            "sun_energy": _u(rng, SUN_ENERGY),
            "world_rotation_deg": rng.uniform(0.0, 360.0),
            "world_strength": _u(rng, WORLD_STRENGTH),
            "exposure": _u(rng, EXPOSURE),
        },
        "ripe_shift": _u(rng, RIPE_SHIFT),
    }


def sun_direction(elevation_deg: float, azimuth_deg: float) -> tuple[float, float, float]:
    """Unit vector from the scene towards the sun (+Z up, azimuth from +X towards +Y)."""
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def apply_robot(params: dict, spec: dict, root_name: str = "H01") -> None:
    import bpy

    root = bpy.data.objects[root_name]
    place_root(root, lane_x=params["robot"]["lane_x"], y=params["robot"]["y"])
    apply_pose(root, params["pose"], spec)


def apply_light(light: dict) -> None:
    import bpy
    from mathutils import Vector

    sun = bpy.data.objects["HV_SUN"]
    sun.rotation_mode = "QUATERNION"
    # A sun light shines down its local -Z, so +Z points at the sun.
    d = sun_direction(light["sun_elevation_deg"], light["sun_azimuth_deg"])
    sun.rotation_quaternion = Vector(d).to_track_quat("Z", "Y")
    sun.data.energy = light["sun_energy"]

    scene = bpy.context.scene
    nodes = scene.world.node_tree.nodes
    mapping = next(n for n in nodes if n.type == "MAPPING")
    mapping.inputs["Rotation"].default_value[2] = math.radians(light["world_rotation_deg"])
    background = next(n for n in nodes if n.type == "BACKGROUND")
    background.inputs["Strength"].default_value = light["world_strength"]
    scene.view_settings.exposure = light["exposure"]


def apply_ripe_shift(shift: float, material: str = "CROP_V3_fruit") -> None:
    """Insert (once) clamp(ripe + shift) between the `ripe` attribute and the colour ramp."""
    import bpy

    tree = bpy.data.materials[material].node_tree
    node = tree.nodes.get("HX_ripe_shift")
    if node is None:
        link = next(
            lk
            for lk in tree.links
            if lk.from_node.type == "ATTRIBUTE"
            and lk.from_node.attribute_name == "ripe"
            and lk.to_node.type == "VALTORGB"
        )
        src, dst = link.from_socket, link.to_socket
        tree.links.remove(link)
        node = tree.nodes.new("ShaderNodeMath")
        node.name = "HX_ripe_shift"
        node.operation = "ADD"
        node.use_clamp = True
        tree.links.new(src, node.inputs[0])
        tree.links.new(node.outputs[0], dst)
    node.inputs[1].default_value = shift


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Apply one randomized sample to the greenhouse")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--render", type=Path, help="Optional PNG path to render one frame to")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import bpy

    from robot_camera import CameraSpec, add_camera, point_instancers_at

    args = parse_args(argv)
    spec = load_spec()
    params = sample(args.seed, spec)
    apply_robot(params, spec)
    apply_light(params["light"])
    apply_ripe_shift(params["ripe_shift"])
    cam = add_camera("H01", args.side, CameraSpec())
    point_instancers_at(cam)

    mw = cam.matrix_world
    params["camera"] = {
        "side": args.side,
        "world": [round(v, 4) for v in mw.translation],
        "forward": [round(-v, 4) for v in mw.to_3x3().col[2]],
    }
    print(f"[rand] {json.dumps(params)}", flush=True)

    if args.render:
        bpy.context.scene.render.filepath = str(args.render.resolve())
        bpy.ops.render.render(write_still=True)
        print(f"[rand] rendered {args.render.resolve()}", flush=True)


if __name__ == "__main__":
    main()
