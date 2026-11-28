"""Robot kinematics and pose control for Harvix.

Provides pure specification validation, preset handling, lane math,
and Blender object property application functions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "assets" / "robot_spec.json"
RAIL_LANES = tuple(range(9, 92, 2))
TARGET_SUFFIXES = (
    "CUT_TCP_Left",
    "CUT_TCP_Right",
    "J_left_camera",
    "J_right_camera",
)


def load_spec(path: str | Path = DEFAULT_SPEC) -> dict:
    """Load the robot specification JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def joint_names(spec: dict) -> list[str]:
    """Return sorted joint names from joint_limits."""
    return sorted(spec["joint_limits"].keys())


def validate_pose(pose: dict, spec: dict) -> None:
    """Raise ValueError unless every joint exists and its value is a number within limits."""
    _, outside = clamp_pose(pose, spec)
    if outside:
        joint = outside[0]
        lo, hi = spec["joint_limits"][joint]["lower"], spec["joint_limits"][joint]["upper"]
        raise ValueError(f"Joint '{joint}' value {pose[joint]} is outside limits [{lo}, {hi}]")


def clamp_pose(pose: dict, spec: dict) -> tuple[dict, list[str]]:
    """Return a clamped copy of the pose and a list of joints that were clamped."""
    limits = spec["joint_limits"]
    clamped = {}
    clamped_joints = []
    for joint, val in pose.items():
        if joint not in limits:
            raise ValueError(f"Unknown joint '{joint}'. Known joints: {sorted(limits.keys())}")
        if isinstance(val, bool) or not isinstance(val, (int, float)) or math.isnan(val):
            raise ValueError(f"Joint '{joint}' value must be a valid number, got {val!r}")
        lower = limits[joint]["lower"]
        upper = limits[joint]["upper"]
        if val < lower:
            clamped[joint] = lower
            clamped_joints.append(joint)
        elif val > upper:
            clamped[joint] = upper
            clamped_joints.append(joint)
        else:
            clamped[joint] = val
    return clamped, clamped_joints


def preset(name: str, spec: dict) -> dict:
    """Return full joint pose with values from the given preset overlaying neutral."""
    poses = spec.get("poses", {})
    if name not in poses:
        available = sorted(poses.keys())
        raise KeyError(f"Unknown preset '{name}'. Available presets: {available}")
    full_pose = dict(poses["neutral"])
    full_pose.update(poses[name])
    return full_pose


def validate_lane(x: float) -> None:
    """Validate that x is within 1e-6 tolerance of a rail lane."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or math.isnan(x):
        raise ValueError(f"Lane coordinate must be a valid number, got {x!r}")
    if not any(abs(x - lane) <= 1e-6 for lane in RAIL_LANES):
        raise ValueError(f"Lane {x} is not one of the valid rail lanes {RAIL_LANES}")


def apply_pose(root, pose: dict, spec: dict, strict: bool = True) -> dict:
    """Apply joint pose properties onto the root object and update view layer."""
    import bpy

    if strict:
        validate_pose(pose, spec)
        target_pose = dict(pose)
    else:
        target_pose, _ = clamp_pose(pose, spec)

    prefix = spec.get("joint_property_prefix", "q_") if spec else "q_"
    for joint, val in target_pose.items():
        root[f"{prefix}{joint}"] = val

    # Writing an ID property from Python tags nothing, and tagging only the root still leaves
    # some joint drivers stale in about one run in four (seen on seed 5: right_fingers kept its
    # old value while the render, which rebuilds the scene, used the new one). Tag every part and
    # check the joints' driven values, so labels are never computed for a pose we don't render.
    stale = []
    for _ in range(3):
        root.update_tag()
        for ob in root.children_recursive:
            ob.update_tag()
        bpy.context.view_layer.update()
        stale = stale_joints(root, target_pose)
        if not stale:
            return target_pose
    raise RuntimeError(f"Joint drivers did not settle: {stale}")


def stale_joints(root, pose: dict, tol: float = 1e-6) -> list[str]:
    """Joints whose evaluated `hv_q` on the joint link differs from the requested value."""
    import bpy

    dg = bpy.context.evaluated_depsgraph_get()
    prefix = "" if root.name == "HV_ROOT" else f"{root.name}_"  # robot file vs greenhouse
    stale = []
    for joint, val in pose.items():
        link = bpy.data.objects.get(f"{prefix}DESIGN_J_{joint}")
        if link is None or "hv_q" not in link:
            continue
        if abs(link.evaluated_get(dg)["hv_q"] - val) > tol:
            stale.append(joint)
    return stale


def read_pose(root, spec: dict) -> dict:
    """Read current joint pose properties from the root object."""
    prefix = spec.get("joint_property_prefix", "q_") if spec else "q_"
    pose = {}
    for joint in joint_names(spec):
        prop = f"{prefix}{joint}"
        if prop in root:
            pose[joint] = root[prop]
    return pose


def place_root(root, lane_x: float | None = None, y: float | None = None) -> None:
    """Place the root object on the rail grid, only modifying x and/or y."""
    if lane_x is not None:
        validate_lane(lane_x)
        root.location.x = float(lane_x)
    if y is not None:
        if isinstance(y, bool) or not isinstance(y, (int, float)) or math.isnan(y):
            raise ValueError(f"Invalid y coordinate: {y!r}")
        root.location.y = float(y)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments passed after '--'."""
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Harvix robot control CLI")
    parser.add_argument("--root", default="H01", help="Name of root object")
    parser.add_argument("--preset", default="stow", help="Preset pose name")
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="Path to robot_spec.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Blender entrypoint for applying a preset pose and printing tool transforms."""
    import bpy

    args = parse_args(argv)
    spec = load_spec(args.spec)

    root = bpy.data.objects.get(args.root)
    if root is None:
        raise ValueError(f"Root object '{args.root}' not found in scene")

    targets = {}
    for child in root.children_recursive:
        for suffix in TARGET_SUFFIXES:
            if child.name.endswith(suffix):
                targets[suffix] = child

    missing = [s for s in TARGET_SUFFIXES if s not in targets]
    if missing:
        raise RuntimeError(
            f"Could not find child objects under root '{args.root}' with suffixes: {missing}"
        )

    bpy.context.view_layer.update()
    before = {
        s: [round(float(c), 4) for c in targets[s].matrix_world.translation]
        for s in TARGET_SUFFIXES
    }

    current = read_pose(root, spec)
    target_pose = preset(args.preset, spec)
    apply_pose(root, target_pose, spec)

    after = {
        s: [round(float(c), 4) for c in targets[s].matrix_world.translation]
        for s in TARGET_SUFFIXES
    }
    changed = any(abs(current.get(j, 0.0) - v) > 1e-6 for j, v in target_pose.items())
    if changed and before == after:
        raise RuntimeError("Pose changed but no tool point moved: drivers did not evaluate")

    payload = {
        "before": before,
        "after": after,
    }
    print(f"[robot] {json.dumps(payload)}")


if __name__ == "__main__":
    main()
