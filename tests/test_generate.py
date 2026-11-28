import json
import subprocess

import pytest

import generate
from generate import (
    collect_sample,
    done_seeds,
    merge_coco,
    parse_args,
    parse_seeds,
    side_for,
    write_merged,
)


def test_parse_seeds():
    assert parse_seeds("0:3") == [0, 1, 2]
    assert parse_seeds("3,7,9") == [3, 7, 9]
    assert parse_seeds("12") == [12]
    for bad in ("5:5", "", "a"):
        with pytest.raises(ValueError):
            parse_seeds(bad)


def test_side_for():
    assert [side_for(s, "alternate") for s in range(3)] == ["left", "right", "left"]
    assert side_for(3, "right") == "right"


def _coco(seed, ann_ids, depth=True):
    return {
        "images": [
            {
                "id": seed,
                "file_name": "rgb.png",
                "mask_file": "mask.png",
                "depth_file": "depth.png" if depth else None,
                "width": 640,
                "height": 480,
            }
        ],
        "annotations": [{"id": seed * 1000 + i, "image_id": seed} for i in ann_ids],
        "categories": [{"id": 1, "name": "truss"}],
    }


def test_merge_coco():
    merged = merge_coco([_coco(7, [1, 2]), _coco(3, [1], depth=False)])
    assert [i["id"] for i in merged["images"]] == [3, 7]
    assert merged["images"][1]["file_name"] == "images/000007.png"
    assert merged["images"][1]["mask_file"] == "masks/000007.png"
    assert merged["images"][1]["depth_file"] == "depth/000007.png"
    assert merged["images"][0]["depth_file"] is None
    assert [a["id"] for a in merged["annotations"]] == [3001, 7001, 7002]
    with pytest.raises(ValueError, match="Duplicate"):
        merge_coco([_coco(7, [1]), _coco(7, [1])])


def test_collect_and_resume(tmp_path):
    out = tmp_path / "ds"
    for seed in (4, 5):
        sample = out / "_work" / f"{seed:06d}"
        sample.mkdir(parents=True)
        for name in ("rgb.png", "mask.png", "depth.png"):
            (sample / name).write_bytes(b"x")
        (sample / "coco.json").write_text(json.dumps(_coco(seed, [1])))
        (sample / "metadata.json").write_text(json.dumps({"seed": seed}))
        collect_sample(out, sample, seed)
        assert not sample.exists()

    assert (out / "images" / "000004.png").exists()
    assert (out / "depth" / "000005.png").exists()
    assert (out / "labels" / "000005.json").exists()
    assert done_seeds(out / "metadata.jsonl") == {4, 5}
    assert done_seeds(out / "missing.jsonl") == set()
    assert write_merged(out) == 2
    merged = json.loads((out / "annotations.json").read_text())
    assert [i["file_name"] for i in merged["images"]] == ["images/000004.png", "images/000005.png"]


def test_parse_args(tmp_path):
    args = parse_args(["--seeds", "0:4", "--out", str(tmp_path)])
    assert (args.seeds, args.side, args.radius, args.samples) == (
        [0, 1, 2, 3],
        "alternate",
        2.0,
        32,
    )


def test_timeout_marks_seed_failed(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(kw["timeout"])
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(generate.subprocess, "run", fake_run)
    code = generate.main(["--seeds", "0:2", "--out", str(tmp_path), "--timeout", "5"])
    assert code == 1
    assert calls == [5, 5]  # both seeds tried, the first timeout did not stop the batch
