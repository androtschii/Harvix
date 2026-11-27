import sys
import types

import pytest

from robot_control import (
    DEFAULT_SPEC,
    RAIL_LANES,
    apply_pose,
    clamp_pose,
    joint_names,
    load_spec,
    place_root,
    pose_to_props,
    preset,
    read_pose,
    snap_lane,
    validate_lane,
    validate_pose,
)


@pytest.fixture
def spec():
    return load_spec(DEFAULT_SPEC)


def test_load_spec(spec):
    assert "joint_limits" in spec
    assert "poses" in spec
    assert spec["joint_property_prefix"] == "q_"
    assert len(spec["joint_limits"]) == 21


def test_joint_names(spec):
    names = joint_names(spec)
    assert len(names) == 21
    assert names == sorted(names)
    assert names[0] == "left_elbow"
    assert "right_wrist" in names
    assert set(names) == set(spec["joint_limits"].keys())


def test_rail_lanes():
    assert isinstance(RAIL_LANES, tuple)
    assert len(RAIL_LANES) == 42
    assert RAIL_LANES[0] == 9
    assert RAIL_LANES[-1] == 91
    assert all(RAIL_LANES[i + 1] - RAIL_LANES[i] == 2 for i in range(len(RAIL_LANES) - 1))


def test_snap_lane():
    assert snap_lane(8) == 9
    assert snap_lane(8.0) == 9
    assert snap_lane(9.9) == 9
    assert snap_lane(10) == 9
    assert snap_lane(10.0) == 9
    assert snap_lane(10.1) == 11
    assert snap_lane(100) == 91
    assert snap_lane(100.0) == 91

    assert snap_lane(9) == 9
    assert snap_lane(91) == 91
    assert snap_lane(12.0) == 11
    assert snap_lane(14.0) == 13

    with pytest.raises(ValueError):
        snap_lane(float("nan"))
    with pytest.raises(ValueError):
        snap_lane("10")
    with pytest.raises(ValueError):
        snap_lane(True)


def test_validate_lane():
    validate_lane(9)
    validate_lane(9.0)
    validate_lane(91.0)
    validate_lane(25)
    validate_lane(9.0 + 1e-7)
    validate_lane(91.0 - 5e-7)

    with pytest.raises(ValueError, match="not one of the valid rail lanes"):
        validate_lane(10)
    with pytest.raises(ValueError, match="not one of the valid rail lanes"):
        validate_lane(8.0)
    with pytest.raises(ValueError, match="not one of the valid rail lanes"):
        validate_lane(92.0)
    with pytest.raises(ValueError, match="not one of the valid rail lanes"):
        validate_lane(9.001)

    with pytest.raises(ValueError, match="valid number"):
        validate_lane(float("nan"))
    with pytest.raises(ValueError, match="valid number"):
        validate_lane(True)
    with pytest.raises(ValueError, match="valid number"):
        validate_lane("9")


def test_validate_pose(spec):
    validate_pose({"left_elbow": 0.0}, spec)
    validate_pose({"left_elbow": -2.3, "left_shoulder": 1.66}, spec)

    # Unknown joint
    with pytest.raises(ValueError, match="Unknown joint 'unknown_joint'"):
        validate_pose({"unknown_joint": 0.0}, spec)

    # Out of limit: lower
    with pytest.raises(ValueError, match="outside limits"):
        validate_pose({"left_elbow": -2.31}, spec)

    # Out of limit: upper
    with pytest.raises(ValueError, match="outside limits"):
        validate_pose({"left_elbow": 2.31}, spec)

    # NaN value
    with pytest.raises(ValueError, match="must be a valid number"):
        validate_pose({"left_elbow": float("nan")}, spec)

    # Non-number values
    with pytest.raises(ValueError, match="must be a valid number"):
        validate_pose({"left_elbow": "0.0"}, spec)
    with pytest.raises(ValueError, match="must be a valid number"):
        validate_pose({"left_elbow": True}, spec)
    with pytest.raises(ValueError, match="must be a valid number"):
        validate_pose({"left_elbow": None}, spec)


def test_clamp_pose(spec):
    # Within limits: no clamping
    pose = {"left_elbow": 0.5, "right_lift": 0.0}
    clamped, modified = clamp_pose(pose, spec)
    assert clamped == pose
    assert modified == []
    assert clamped is not pose

    # Clamping below lower limit
    clamped, modified = clamp_pose({"left_elbow": -3.0}, spec)
    assert clamped["left_elbow"] == -2.3
    assert modified == ["left_elbow"]

    # Clamping above upper limit
    clamped, modified = clamp_pose({"left_elbow": 5.0}, spec)
    assert clamped["left_elbow"] == 2.3
    assert modified == ["left_elbow"]

    # Multiple joints clamping
    input_pose = {"left_elbow": -5.0, "right_elbow": 5.0, "left_shoulder": 0.0}
    clamped, modified = clamp_pose(input_pose, spec)
    assert clamped["left_elbow"] == -2.3
    assert clamped["right_elbow"] == 2.3
    assert clamped["left_shoulder"] == 0.0
    assert set(modified) == {"left_elbow", "right_elbow"}

    # Unknown joint raises ValueError
    with pytest.raises(ValueError, match="Unknown joint"):
        clamp_pose({"fake_joint": 1.0}, spec)

    # Invalid value raises ValueError
    with pytest.raises(ValueError, match="must be a valid number"):
        clamp_pose({"left_elbow": float("nan")}, spec)


def test_preset(spec):
    # Neutral preset has all 21 joints
    p_neutral = preset("neutral", spec)
    assert len(p_neutral) == 21
    assert set(p_neutral.keys()) == set(joint_names(spec))
    assert p_neutral == spec["poses"]["neutral"]

    # Stow preset has all 21 joints
    p_stow = preset("stow", spec)
    assert len(p_stow) == 21
    assert set(p_stow.keys()) == set(joint_names(spec))

    # Specified stow values match spec
    for joint, val in spec["poses"]["stow"].items():
        assert p_stow[joint] == val

    # Non-stow joints take values from neutral
    for joint in ("rail_joint", "left_tool", "right_tool"):
        assert joint not in spec["poses"]["stow"]
        assert p_stow[joint] == p_neutral[joint]

    # Unknown preset raises KeyError
    with pytest.raises(KeyError, match="Unknown preset 'invalid_preset'"):
        preset("invalid_preset", spec)


def test_pose_to_props(spec):
    pose = {"left_elbow": 0.5, "right_wrist": -1.0}

    # Default prefix "q_"
    props_default = pose_to_props(pose)
    assert props_default == {"q_left_elbow": 0.5, "q_right_wrist": -1.0}

    # Custom prefix parameter
    props_custom = pose_to_props(pose, prefix="custom_")
    assert props_custom == {"custom_left_elbow": 0.5, "custom_right_wrist": -1.0}

    # Prefix from spec dictionary
    props_from_spec = pose_to_props(pose, spec)
    assert props_from_spec == {"q_left_elbow": 0.5, "q_right_wrist": -1.0}

    # Prefix as string passed positionally
    props_pos_str = pose_to_props(pose, "p_")
    assert props_pos_str == {"p_left_elbow": 0.5, "p_right_wrist": -1.0}


def test_place_root():
    class MockLocation:
        def __init__(self, x=0.0, y=0.0, z=5.0):
            self.x = x
            self.y = y
            self.z = z

    class MockRoot:
        def __init__(self):
            self.location = MockLocation()
            self.rotation_euler = (0.0, 0.0, 1.57)

    root = MockRoot()
    place_root(root, lane_x=13, y=42.5)
    assert root.location.x == 13.0
    assert root.location.y == 42.5
    assert root.location.z == 5.0
    assert root.rotation_euler == (0.0, 0.0, 1.57)

    # lane_x only
    place_root(root, lane_x=15)
    assert root.location.x == 15.0
    assert root.location.y == 42.5

    # y only
    place_root(root, y=10.0)
    assert root.location.x == 15.0
    assert root.location.y == 10.0

    # Invalid lane
    with pytest.raises(ValueError):
        place_root(root, lane_x=10)

    # Invalid y
    with pytest.raises(ValueError):
        place_root(root, y=float("nan"))
    with pytest.raises(ValueError):
        place_root(root, y=True)


def test_read_pose(spec):
    root = {
        "q_left_elbow": 0.25,
        "q_right_elbow": -0.5,
        "hv_status": "green",
    }
    pose = read_pose(root, spec)
    assert pose == {"left_elbow": 0.25, "right_elbow": -0.5}
    assert "hv_status" not in pose


def test_apply_pose_with_mock_bpy(monkeypatch, spec):
    mock_bpy = types.ModuleType("bpy")
    updated = []
    mock_bpy.context = types.SimpleNamespace(
        view_layer=types.SimpleNamespace(update=lambda: updated.append(True))
    )
    monkeypatch.setitem(sys.modules, "bpy", mock_bpy)

    class Root(dict):
        tagged = 0

        def update_tag(self):
            self.tagged += 1

    root = Root()
    applied = apply_pose(root, {"left_elbow": 0.5}, spec, strict=True)
    assert applied == {"left_elbow": 0.5}
    assert root["q_left_elbow"] == 0.5
    assert root.tagged == 1
    assert len(updated) == 1

    # Non-strict clamping
    applied_clamped = apply_pose(root, {"left_elbow": 10.0}, spec, strict=False)
    assert applied_clamped["left_elbow"] == 2.3
    assert root["q_left_elbow"] == 2.3
    assert len(updated) == 2

    # Strict failure
    with pytest.raises(ValueError):
        apply_pose(root, {"left_elbow": 10.0}, spec, strict=True)
