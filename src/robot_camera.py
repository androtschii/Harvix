"""Robot-mounted render camera for Harvix (decision D2).

Pure camera math (intrinsics, projection, frame conversion) is testable without Blender; the bpy
layer adds a camera parented to the arm camera link and points the crop LOD/culling at it.

    blender -b assets/Harvix_Greenhouse.blend --python src/robot_camera.py -- --side left
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_control import apply_pose, load_spec, preset  # noqa: E402

# Camera rotation relative to H01_DESIGN_J_<side>_camera (XYZ euler). Measured in the `neutral`
# pose: with it the Blender camera (looking down its -Z) faces the arm's tool, image-up = world +Z.
MOUNT_EULER = {"left": (0.0, math.pi, 0.0), "right": (0.0, 0.0, math.pi)}
INSTANCERS = ("GH_fruit_instancer", "GH_plant_instancer")


@dataclass(frozen=True)
class CameraSpec:
    width: int = 640
    height: int = 480
    hfov_deg: float = 70.0
    clip_start: float = 0.05
    clip_end: float = 30.0

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Resolution must be positive, got {self.width}x{self.height}")
        if not 0 < self.hfov_deg < 180:
            raise ValueError(f"hfov_deg must be in (0, 180), got {self.hfov_deg}")
        if not 0 < self.clip_start < self.clip_end:
            raise ValueError(
                f"Need 0 < clip_start < clip_end, got {self.clip_start}, {self.clip_end}"
            )


def focal_px(spec: CameraSpec) -> tuple[float, float]:
    fx = (spec.width / 2) / math.tan(math.radians(spec.hfov_deg) / 2)
    return fx, fx  # square pixels


def intrinsics(spec: CameraSpec) -> list[list[float]]:
    fx, fy = focal_px(spec)
    return [[fx, 0.0, spec.width / 2], [0.0, fy, spec.height / 2], [0.0, 0.0, 1.0]]


def vfov_deg(spec: CameraSpec) -> float:
    return math.degrees(2 * math.atan((spec.height / 2) / focal_px(spec)[1]))


def blender_to_opencv(point) -> tuple[float, float, float]:
    """Blender camera frame (x right, y up, looks down -z) to OpenCV (x right, y down, z ahead)."""
    x, y, z = point
    return (x, -y, -z)


def project(K, point_cam) -> tuple[float, float] | None:
    """Project an OpenCV camera-frame point to pixel (u, v); None if it is behind the camera."""
    x, y, z = point_cam
    if z <= 0:
        return None
    return (K[0][0] * x / z + K[0][2], K[1][1] * y / z + K[1][2])


def add_camera(root_name: str, side: str, spec: CameraSpec):
    import bpy

    link = bpy.data.objects[f"{root_name}_DESIGN_J_{side}_camera"]
    name = f"HX_CAM_{side}"
    old = bpy.data.objects.get(name)
    if old is not None:
        data = old.data
        bpy.data.objects.remove(old)
        bpy.data.cameras.remove(data)

    data = bpy.data.cameras.new(name)
    data.sensor_fit = "HORIZONTAL"
    data.angle = math.radians(spec.hfov_deg)
    data.clip_start = spec.clip_start
    data.clip_end = spec.clip_end
    cam = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(cam)
    cam.parent = link
    cam.matrix_parent_inverse.identity()
    cam.location = (0.0, 0.0, 0.0)
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = MOUNT_EULER[side]

    scene = bpy.context.scene
    scene.camera = cam
    scene.render.resolution_x = spec.width
    scene.render.resolution_y = spec.height
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    return cam


def _set_input(mod, name: str, value) -> None:
    for item in mod.node_group.interface.items_tree:
        if getattr(item, "in_out", None) == "INPUT" and item.name == name:
            # Blender 5.3: modifier inputs are RNA structs; mod[...] IDProperty access fails.
            getattr(mod.properties.inputs, item.identifier).value = value
            return
    raise KeyError(f"Modifier {mod.name!r} has no input {name!r}")


def point_instancers_at(cam) -> None:
    """Make crop LOD and culling follow the robot camera instead of the scene's film camera."""
    import bpy

    for ob_name in INSTANCERS:
        ob = bpy.data.objects[ob_name]
        for mod in ob.modifiers:
            if mod.type == "NODES":
                _set_input(mod, "Camera", cam)
                _set_input(mod, "Use Camera", True)
        ob.update_tag()
    bpy.context.view_layer.update()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Mount a render camera on the H01 arm")
    p.add_argument("--root", default="H01")
    p.add_argument("--side", default="left", choices=sorted(MOUNT_EULER))
    p.add_argument("--preset", default="neutral")
    p.add_argument("--hfov", type=float, default=70.0)
    p.add_argument("--render", type=Path, help="Optional PNG path to render one frame to")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import bpy

    args = parse_args(argv)
    spec_robot = load_spec()
    root = bpy.data.objects[args.root]
    apply_pose(root, preset(args.preset, spec_robot), spec_robot)

    spec = CameraSpec(hfov_deg=args.hfov)
    cam = add_camera(args.root, args.side, spec)
    point_instancers_at(cam)

    K = intrinsics(spec)
    mw = cam.matrix_world
    tcp = bpy.data.objects[f"{args.root}_CUT_TCP_{args.side.capitalize()}"]
    tcp_cam = blender_to_opencv(mw.inverted() @ tcp.matrix_world.translation)
    uv = project(K, tcp_cam)
    payload = {
        "side": args.side,
        "K": [[round(v, 3) for v in row] for row in K],
        "hfov_deg": spec.hfov_deg,
        "vfov_deg": round(vfov_deg(spec), 3),
        "cam_world": [round(v, 4) for v in mw.translation],
        "cam_forward_world": [round(-v, 4) for v in mw.to_3x3().col[2]],
        "tcp_in_view": [round(v, 1) for v in uv] if uv else None,
    }
    print(f"[camera] {json.dumps(payload)}", flush=True)

    if args.render:
        bpy.context.scene.render.filepath = str(args.render.resolve())
        bpy.ops.render.render(write_still=True)
        print(f"[camera] rendered {args.render.resolve()}", flush=True)


if __name__ == "__main__":
    main()
