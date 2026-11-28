import math

import pytest

from robot_camera import (
    MOUNT_EULER,
    CameraSpec,
    blender_to_opencv,
    focal_px,
    intrinsics,
    parse_args,
    project,
    vfov_deg,
)


def test_focal_and_intrinsics_default():
    spec = CameraSpec()
    fx, fy = focal_px(spec)
    assert fx == pytest.approx(457.01, abs=0.01)
    assert fy == fx
    K = intrinsics(spec)
    assert K == [[fx, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]]


def test_vfov_from_aspect():
    assert vfov_deg(CameraSpec()) == pytest.approx(55.4, abs=0.1)
    assert vfov_deg(CameraSpec(width=480, height=480, hfov_deg=60)) == pytest.approx(60.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": -1},
        {"hfov_deg": 0},
        {"hfov_deg": 180},
        {"clip_start": 0},
        {"clip_start": 5.0, "clip_end": 1.0},
    ],
)
def test_camera_spec_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        CameraSpec(**kwargs)


def test_project():
    K = intrinsics(CameraSpec())
    assert project(K, (0.0, 0.0, 2.0)) == (320.0, 240.0)
    # A point at the right edge of the horizontal FOV lands on u = width.
    u, v = project(K, (math.tan(math.radians(35)), 0.0, 1.0))
    assert u == pytest.approx(640.0)
    assert v == 240.0
    assert project(K, (0.0, 0.0, 0.0)) is None
    assert project(K, (0.1, 0.1, -1.0)) is None


def test_blender_to_opencv():
    assert blender_to_opencv((0.0, 0.0, -1.0)) == (0.0, 0.0, 1.0)  # straight ahead
    assert blender_to_opencv((1.0, 2.0, -3.0)) == (1.0, -2.0, 3.0)
    assert blender_to_opencv((0.0, 1.0, 0.0)) == (0.0, -1.0, 0.0)  # image up is -y in OpenCV


def test_mount_euler_sides():
    assert set(MOUNT_EULER) == {"left", "right"}
    assert MOUNT_EULER["left"] == (0.0, math.pi, 0.0)
    assert MOUNT_EULER["right"] == (0.0, 0.0, math.pi)


def test_parse_args_defaults():
    args = parse_args([])
    assert (args.root, args.side, args.preset, args.hfov, args.render) == (
        "H01",
        "left",
        "neutral",
        70.0,
        None,
    )
    with pytest.raises(SystemExit):
        parse_args(["--side", "middle"])
