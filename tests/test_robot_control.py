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
    preset,
    read_pose,
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


class _Tagged(dict):
    """Stands in for a Blender object: custom props as dict items plus update_tag()."""

    def __init__(self, name, **props):
        super().__init__(props)
        self.name = name
        self.tagged = 0
        self.children_recursive = []

    def update_tag(self):
        self.tagged += 1

    def evaluated_get(self, dg):
        return dg.get(self.name, self)


def _mock_bpy(monkeypatch, objects, evaluated=None):
    """Fake bpy: `evaluated` maps object name -> what the depsgraph returns for it."""
    mock_bpy = types.ModuleType("bpy")
    updated = []
    mock_bpy.context = types.SimpleNamespace(
        view_layer=types.SimpleNamespace(update=lambda: updated.append(True)),
        evaluated_depsgraph_get=lambda: evaluated or {},
    )
    mock_bpy.data = types.SimpleNamespace(objects=objects)
    monkeypatch.setitem(sys.modules, "bpy", mock_bpy)
    return updated


def test_apply_pose_with_mock_bpy(monkeypatch, spec):
    link = _Tagged("H01_DESIGN_J_left_elbow", hv_q=0.0)
    root = _Tagged("H01")
    root.children_recursive = [link]
    # The depsgraph reports the driven value that apply_pose asked for.
    evaluated = {}
    updated = _mock_bpy(monkeypatch, {link.name: link}, evaluated)
    evaluated[link.name] = _Tagged(link.name, hv_q=0.5)

    applied = apply_pose(root, {"left_elbow": 0.5}, spec, strict=True)
    assert applied == {"left_elbow": 0.5}
    assert root["q_left_elbow"] == 0.5
    assert (root.tagged, link.tagged) == (1, 1)  # every robot part is tagged, not only the root
    assert len(updated) == 1

    # Non-strict clamping
    evaluated[link.name] = _Tagged(link.name, hv_q=2.3)
    applied_clamped = apply_pose(root, {"left_elbow": 10.0}, spec, strict=False)
    assert applied_clamped["left_elbow"] == 2.3
    assert root["q_left_elbow"] == 2.3

    # Strict failure
    with pytest.raises(ValueError):
        apply_pose(root, {"left_elbow": 10.0}, spec, strict=True)


def test_apply_pose_raises_when_drivers_stay_stale(monkeypatch, spec):
    link = _Tagged("H01_DESIGN_J_left_elbow", hv_q=0.0)
    root = _Tagged("H01")
    root.children_recursive = [link]
    updated = _mock_bpy(monkeypatch, {link.name: link}, {link.name: link})  # hv_q never changes
    with pytest.raises(RuntimeError, match="left_elbow"):
        apply_pose(root, {"left_elbow": 0.5}, spec)
    assert len(updated) == 3  # retried before giving up
