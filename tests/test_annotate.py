import numpy as np
import pytest

from annotate import (
    MAX_ID,
    RIPE_THRESHOLD,
    bbox_xywh,
    build_annotations,
    decode_ids,
    depth_to_mm16,
    id_to_srgb8,
    parse_args,
    ripeness,
    rle_decode,
    rle_encode,
    srgb8_to_linear,
    write_png16,
)


def test_id_colours_roundtrip_all_ids():
    ids = np.arange(MAX_ID + 1)
    rgb = np.array([id_to_srgb8(int(i)) for i in ids], dtype=np.uint8).reshape(1, -1, 3)
    assert (decode_ids(rgb)[0] == ids).all()
    assert id_to_srgb8(0) == (0, 0, 0)


def test_decode_tolerates_small_rounding():
    rgb = np.array([[id_to_srgb8(77)]], dtype=np.int32) + np.array([[[1, -1, 2]]])
    assert decode_ids(rgb.clip(0, 255).astype(np.uint8))[0, 0] == 77


@pytest.mark.parametrize("bad", [-1, MAX_ID + 1])
def test_id_out_of_range(bad):
    with pytest.raises(ValueError):
        id_to_srgb8(bad)


def test_srgb8_to_linear():
    assert srgb8_to_linear(0) == 0
    assert srgb8_to_linear(255) == pytest.approx(1.0)
    assert srgb8_to_linear(128) == pytest.approx(0.2158, abs=1e-4)


def test_rle_roundtrip_and_coco_convention():
    m = np.zeros((4, 5), dtype=bool)
    m[1:3, 2:4] = True
    rle = rle_encode(m)
    assert rle["size"] == [4, 5]
    assert sum(rle["counts"]) == 20
    assert rle["counts"][:2] == [9, 2]  # column-major: 2 empty columns + 1 row, then 2 set
    assert (rle_decode(rle) == m).all()

    first = np.zeros((2, 2), dtype=bool)
    first[0, 0] = True
    assert rle_encode(first)["counts"][0] == 0  # counts always start with background
    assert (rle_decode(rle_encode(first)) == first).all()


def test_bbox():
    m = np.zeros((10, 10), dtype=bool)
    m[2:5, 3:9] = True
    assert bbox_xywh(m) == [3, 2, 6, 3]
    assert bbox_xywh(np.zeros((3, 3), dtype=bool)) == [0, 0, 0, 0]


def test_ripeness_clamped():
    assert ripeness(0.95, -0.5, 0.05) == pytest.approx(0.5)
    assert ripeness(0.99, 0.05, 0.1) == 1.0
    assert ripeness(0.1, -0.6, 0.0) == 0.0


def test_build_annotations():
    ids = np.zeros((6, 8), dtype=np.int32)
    ids[1:3, 1:4] = 1
    ids[4, 6] = 2
    base = {"cut_point_px": [1, 1], "cut_point_cam": [0, 0, 1], "center_cam": [0, 0, 1]}
    trusses = [
        {"id": 1, "ripeness": 0.95, "cut_point_tcp": [0, 0, 0], **base},
        {"id": 2, "ripeness": 0.3, "cut_point_tcp": [0, 0, 0], **base},
        {"id": 3, "ripeness": 0.99, "cut_point_tcp": [0, 0, 0], **base},  # fully occluded
    ]
    anns = build_annotations(ids, trusses, image_id=7)
    assert [a["id"] for a in anns] == [7001, 7002]
    a1, a2 = anns
    assert (a1["area"], a1["bbox"], a1["ripe"]) == (6, [1, 1, 3, 2], True)
    assert (a2["area"], a2["bbox"], a2["ripe"]) == (1, [6, 4, 1, 1], False)
    assert (rle_decode(a1["segmentation"]) == (ids == 1)).all()
    assert [t["visible_px"] for t in trusses] == [6, 1, 0]
    assert RIPE_THRESHOLD == 0.8


def test_parse_args():
    args = parse_args(["--out", "x"])
    assert (args.seed, args.side, args.radius, args.samples, args.no_rgb) == (
        0,
        "left",
        2.0,
        32,
        False,
    )


def test_depth_to_mm16():
    d = np.array([[0.5817, 3.9448], [1e10, -1.0]])
    assert depth_to_mm16(d).tolist() == [[582, 3945], [0, 0]]
    assert depth_to_mm16(np.array([[65.5355]])).tolist() == [[0]]
    assert depth_to_mm16(np.array([[np.nan]])).tolist() == [[0]]


def test_write_png16_roundtrip(tmp_path):
    import struct
    import zlib

    img = np.array([[0, 1, 65535], [1000, 2, 3]], dtype=np.uint16)
    path = tmp_path / "d.png"
    write_png16(path, img)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
    assert (w, h, depth, ctype) == (3, 2, 16, 0)
    idat_len = struct.unpack(">I", data[33:37])[0]
    raw = zlib.decompress(data[41 : 41 + idat_len])
    rows = [raw[y * 7 + 1 : (y + 1) * 7] for y in range(2)]
    back = np.array([np.frombuffer(r, dtype=">u2") for r in rows])
    assert (back == img).all()
