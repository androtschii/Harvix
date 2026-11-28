import json
import math

import pytest

from randomize import (
    ENVELOPE,
    RIPE_SHIFT,
    SUN_ELEVATION_DEG,
    WRIST_TO_WALL,
    WRIST_YAW_SIGN,
    Y_RANGE,
    parse_args,
    sample,
    sun_direction,
)
from robot_control import RAIL_LANES, load_spec, preset, validate_lane, validate_pose


@pytest.fixture(scope="module")
def spec():
    return load_spec()


def test_same_seed_same_sample(spec):
    assert sample(7, spec) == sample(7, spec)
    assert sample(7, spec) != sample(8, spec)


def test_sample_is_json_serializable(spec):
    s = sample(3, spec)
    assert json.loads(json.dumps(s)) == s


@pytest.mark.parametrize("seed", range(50))
def test_sample_within_bounds(seed, spec):
    s = sample(seed, spec)
    validate_lane(s["robot"]["lane_x"])
    assert s["robot"]["lane_x"] in RAIL_LANES
    assert Y_RANGE[0] <= s["robot"]["y"] <= Y_RANGE[1]
    validate_pose(s["pose"], spec)
    assert set(s["pose"]) == set(spec["joint_limits"])
    light = s["light"]
    assert SUN_ELEVATION_DEG[0] <= light["sun_elevation_deg"] <= SUN_ELEVATION_DEG[1]
    assert 0 <= light["sun_azimuth_deg"] <= 360
    assert RIPE_SHIFT[0] <= s["ripe_shift"] <= RIPE_SHIFT[1]


@pytest.mark.parametrize("seed", range(50))
def test_pose_faces_chosen_wall(seed, spec):
    s = sample(seed, spec)
    neutral = preset("neutral", spec)
    for side, wall in s["walls"].items():
        pose = s["pose"]
        arm_yaw = sum(pose[f"{side}_{j}"] - neutral[f"{side}_{j}"] for j in ("shoulder", "elbow"))
        total = pose[f"{side}_wrist"] - WRIST_TO_WALL[wall] + WRIST_YAW_SIGN[side] * arm_yaw
        assert abs(total) <= ENVELOPE["wrist"] + 1e-9
        for joint in ("shoulder", "elbow", "fingers"):
            assert abs(s["pose"][f"{side}_{joint}"] - neutral[f"{side}_{joint}"]) <= ENVELOPE[joint]
        # Joints outside the envelope keep their neutral value.
        assert s["pose"][f"{side}_tool"] == neutral[f"{side}_tool"]
    assert s["pose"]["rail_joint"] == neutral["rail_joint"]


def test_both_walls_and_many_lanes_occur(spec):
    samples = [sample(i, spec) for i in range(200)]
    assert {s["walls"]["left"] for s in samples} == set(WRIST_TO_WALL)
    assert len({s["robot"]["lane_x"] for s in samples}) > 30


def test_sun_direction():
    assert sun_direction(90, 0) == pytest.approx((0, 0, 1), abs=1e-12)
    assert sun_direction(0, 0) == pytest.approx((1, 0, 0))
    assert sun_direction(0, 90) == pytest.approx((0, 1, 0), abs=1e-12)
    x, y, z = sun_direction(30, 200)
    assert math.isclose(x * x + y * y + z * z, 1.0)
    assert z == pytest.approx(0.5)


def test_parse_args():
    args = parse_args([])
    assert (args.seed, args.side, args.render) == (0, "left", None)
    assert parse_args(["--seed", "12", "--side", "right"]).seed == 12
