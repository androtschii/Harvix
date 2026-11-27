import json
from pathlib import Path

SPEC = json.loads((Path(__file__).parents[1] / "assets" / "robot_spec.json").read_text())


def test_limits_are_ordered():
    for name, lim in SPEC["joint_limits"].items():
        assert lim["lower"] <= lim["upper"], name


def test_presets_use_known_joints_within_limits():
    limits = SPEC["joint_limits"]
    for pose_name, pose in SPEC["poses"].items():
        for joint, value in pose.items():
            assert joint in limits, f"{pose_name}: unknown joint {joint}"
            lim = limits[joint]
            assert lim["lower"] <= value <= lim["upper"], f"{pose_name}.{joint}={value}"
