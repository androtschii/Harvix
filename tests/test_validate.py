import json

import numpy as np
from PIL import Image

from annotate import id_to_srgb8, rle_encode, write_png16
from robot_control import load_spec, preset
from validate import (
    check_annotation,
    check_depth,
    check_meta,
    check_projection,
    mask_outline,
    summarize,
    validate,
)

H, W = 6, 8


def _mask():
    m = np.zeros((H, W), dtype=bool)
    m[1:3, 2:5] = True
    return m


def _ann(image_id=5, tid=1, ripeness=0.85, **over):
    m = _mask()
    a = {
        "id": image_id * 1000 + tid,
        "truss_id": tid,
        "image_id": image_id,
        "category_id": 1,
        "area": int(m.sum()),
        "bbox": [2, 1, 3, 2],
        "segmentation": rle_encode(m),
        "ripeness": ripeness,
        "ripe": ripeness >= 0.8,
        "cut_point_px": [3.0, 1.0],
        "center_cam": [0, 0, 1],
    }
    a.update(over)
    return a


IMG = {"id": 5, "width": W, "height": H}


def test_consistent_annotation_passes():
    ids = np.zeros((H, W), dtype=int)
    ids[_mask()] = 1
    assert check_annotation(_ann(), IMG, ids) == []


def test_annotation_problems_found():
    ids = np.zeros((H, W), dtype=int)
    assert any("mask.png" in p for p in check_annotation(_ann(), IMG, ids))
    assert any("area" in p for p in check_annotation(_ann(area=99), IMG, None))
    assert any("bbox" in p for p in check_annotation(_ann(bbox=[0, 0, 1, 1]), IMG, None))
    assert any("ripe flag" in p for p in check_annotation(_ann(ripe=False), IMG, None))
    assert any("outside" in p for p in check_annotation(_ann(ripeness=1.5, ripe=True), IMG, None))


def test_depth_cover():
    depth = np.zeros((H, W), dtype=np.uint16)
    assert check_depth(_ann(), depth)
    depth[_mask()] = 1200
    assert check_depth(_ann(), depth) == []


def test_check_meta():
    spec = load_spec()
    good = {"pose": preset("neutral", spec), "robot": {"lane_x": 13, "y": 50.0}}
    assert check_meta(5, good, spec) == []
    assert check_meta(5, None, spec)
    assert check_meta(5, {**good, "robot": {"lane_x": 12, "y": 0}}, spec)
    assert "bad metadata" in check_meta(5, {"robot": {"lane_x": 13}}, spec)[0]


def _meta(seed, side, ripe_shift=-0.1):
    return {
        "seed": seed,
        "camera": {"side": side},
        "robot": {"lane_x": 13 + 2 * seed, "y": 10.0 + seed},
        "light": {"sun_elevation_deg": 30.0 + seed},
        "ripe_shift": ripe_shift,
        "trusses": [{}] * 4,
    }


def test_summarize():
    coco = {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [_ann(1, 1, 0.9), _ann(1, 2, 0.3), _ann(1, 3, 0.85)],
    }
    s = summarize(coco, {1: _meta(1, "right"), 2: _meta(2, "left")})
    assert (s["images"], s["annotations"]) == (2, 3)
    assert s["trusses_per_image"] == {"min": 0, "mean": 1.5, "max": 3}
    assert s["ripe_share"] == round(2 / 3, 3)
    assert s["cut_point_visible_share"] == 0  # field absent in these annotations
    assert s["visible_of_realized"] == round(3 / 8, 3)
    assert s["sides"] == {"right": 1, "left": 1}
    assert s["lanes_used"] == 2


def test_validate_end_to_end(tmp_path):
    ds = tmp_path
    for d in ("images", "masks", "depth"):
        (ds / d).mkdir()
    Image.new("RGB", (W, H), (90, 120, 60)).save(ds / "images/000005.png")
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    mask[_mask()] = id_to_srgb8(1)
    Image.fromarray(mask).save(ds / "masks/000005.png")
    depth = np.zeros((H, W), dtype=np.uint16)
    depth[_mask()] = 900
    write_png16(ds / "depth/000005.png", depth)
    img = {
        **IMG,
        "file_name": "images/000005.png",
        "mask_file": "masks/000005.png",
        "depth_file": "depth/000005.png",
    }
    (ds / "annotations.json").write_text(json.dumps({"images": [img], "annotations": [_ann()]}))
    meta = {**_meta(5, "left"), "pose": preset("neutral", load_spec())}
    meta["camera"]["K"] = [[457.0, 0, 4.0], [0, 457.0, 3.0], [0, 0, 1]]
    (ds / "metadata.jsonl").write_text(json.dumps(meta) + "\n")

    stats, problems = validate(ds, overlays=1)
    assert problems == []
    assert stats["annotations"] == 1
    assert (ds / "report.md").read_text().startswith("# Dataset validation")
    assert (ds / "overlays" / "000005.png").exists()


def test_mask_outline():
    m = np.zeros((5, 5), dtype=bool)
    m[1:4, 1:4] = True
    o = mask_outline(m)
    assert o.sum() == 8 and not o[2, 2]
    assert not mask_outline(np.zeros((3, 3), dtype=bool)).any()


def test_check_projection_catches_camera_mismatch():
    K = [[100.0, 0, 50.0], [0, 100.0, 40.0], [0, 0, 1]]

    def ann(u, v, shift=0):
        # centre_cam that projects to (u, v); bbox centred at (u + shift, v)
        return {
            "center_cam": [(u - 50) / 100, (v - 40) / 100, 1.0],
            "bbox": [u + shift - 2, v - 2, 4, 4],
        }

    aligned = [ann(10, 10), ann(60, 30), ann(80, 70)]
    assert check_projection(aligned, K, 1) == []
    shifted = [ann(10, 10, 40), ann(60, 30, 40), ann(80, 70, 40)]
    assert "offset" in check_projection(shifted, K, 1)[0]
    assert check_projection(shifted[:2], K, 1) == []  # too few trusses to judge
