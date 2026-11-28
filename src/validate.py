"""Validate a generated dataset and summarize it (step 8). Plain Python, no Blender.

    python src/validate.py out/dataset --overlays 8

Checks every sample for internal consistency, writes report.md with statistics into the dataset
folder and draws label overlays for a visual check. Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from annotate import RIPE_THRESHOLD, bbox_xywh, decode_ids, rle_decode  # noqa: E402
from robot_control import load_spec, validate_lane, validate_pose  # noqa: E402

MIN_DEPTH_COVER = 0.5  # share of a truss mask that must have a depth value


def load_dataset(ds: Path) -> tuple[dict, dict[int, dict]]:
    coco = json.loads((ds / "annotations.json").read_text(encoding="utf-8"))
    lines = (ds / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    meta = {m["seed"]: m for m in (json.loads(line) for line in lines if line.strip())}
    return coco, meta


def check_annotation(ann: dict, img: dict, mask_ids: np.ndarray | None) -> list[str]:
    """Problems with one annotation; `mask_ids` is the decoded mask.png, if available."""
    where = f"image {img['id']} ann {ann['id']}"
    problems = []
    m = rle_decode(ann["segmentation"])
    if list(m.shape) != [img["height"], img["width"]]:
        return [f"{where}: RLE size {list(m.shape)} != image size"]
    if int(m.sum()) != ann["area"]:
        problems.append(f"{where}: area {ann['area']} != RLE area {int(m.sum())}")
    if bbox_xywh(m) != ann["bbox"]:
        problems.append(f"{where}: bbox {ann['bbox']} != RLE bbox {bbox_xywh(m)}")
    if mask_ids is not None and not (m == (mask_ids == ann["id"] % 1000)).all():
        problems.append(f"{where}: RLE differs from mask.png")
    if not 0.0 <= ann["ripeness"] <= 1.0:
        problems.append(f"{where}: ripeness {ann['ripeness']} outside [0, 1]")
    if ann["ripe"] != (ann["ripeness"] >= RIPE_THRESHOLD):
        problems.append(f"{where}: ripe flag disagrees with threshold {RIPE_THRESHOLD}")
    if ann["image_id"] != img["id"]:
        problems.append(f"{where}: image_id {ann['image_id']}")
    return problems


def check_depth(ann: dict, depth_mm: np.ndarray) -> list[str]:
    m = rle_decode(ann["segmentation"])
    cover = float((depth_mm[m] > 0).mean()) if m.any() else 0.0
    if cover < MIN_DEPTH_COVER:
        return [f"image {ann['image_id']} ann {ann['id']}: depth covers {cover:.0%} of the mask"]
    return []


def check_meta(seed: int, meta: dict | None, spec: dict) -> list[str]:
    if meta is None:
        return [f"image {seed}: no metadata line"]
    try:
        validate_pose(meta["pose"], spec)
        validate_lane(meta["robot"]["lane_x"])
    except ValueError as e:
        return [f"image {seed}: {e}"]
    return []


def summarize(coco: dict, meta: dict[int, dict]) -> dict:
    anns = coco["annotations"]
    per_image = Counter(a["image_id"] for a in anns)
    counts = [per_image.get(img["id"], 0) for img in coco["images"]]
    areas = np.array([a["area"] for a in anns]) if anns else np.zeros(1)
    realized = sum(len(m.get("trusses", [])) for m in meta.values())

    def rng(values):
        values = list(values)
        return [round(min(values), 2), round(max(values), 2)] if values else None

    return {
        "images": len(coco["images"]),
        "annotations": len(anns),
        "trusses_per_image": {
            "min": min(counts, default=0),
            "mean": round(float(np.mean(counts or [0])), 1),
            "max": max(counts, default=0),
        },
        "ripe_share": round(sum(a["ripe"] for a in anns) / max(len(anns), 1), 3),
        "ripeness_quartiles": [
            round(float(q), 2)
            for q in np.quantile([a["ripeness"] for a in anns] or [0], [0.25, 0.5, 0.75])
        ],
        "area_px_quartiles": [int(q) for q in np.quantile(areas, [0.25, 0.5, 0.75])],
        "visible_of_realized": round(len(anns) / max(realized, 1), 3),
        "sides": dict(Counter(m["camera"]["side"] for m in meta.values())),
        "lanes_used": len({m["robot"]["lane_x"] for m in meta.values()}),
        "y_range_m": rng(m["robot"]["y"] for m in meta.values()),
        "sun_elevation_deg": rng(m["light"]["sun_elevation_deg"] for m in meta.values()),
        "ripe_shift": rng(m["ripe_shift"] for m in meta.values()),
    }


def report_markdown(stats: dict, problems: list[str]) -> str:
    lines = [
        "# Dataset validation",
        "",
        f"Checks: {'all passed' if not problems else f'{len(problems)} problems'}",
        "",
    ]
    lines += ["| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stats.items()]
    if problems:
        lines += ["", "## Problems", ""] + [f"- {p}" for p in problems[:200]]
    return "\n".join(lines) + "\n"


def mask_outline(m: np.ndarray) -> np.ndarray:
    """Mask pixels with at least one 4-neighbour outside the mask."""
    p = np.pad(m, 1)
    inner = p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return m & ~inner


def draw_overlay(ds: Path, img: dict, anns: list[dict], out: Path) -> None:
    from PIL import Image, ImageDraw

    # Mask outlines only, so the rendered fruit colour stays visible next to the ripeness label.
    px = np.array(Image.open(ds / img["file_name"]).convert("RGB"))
    for a in anns:
        px[mask_outline(rle_decode(a["segmentation"]))] = (
            (40, 220, 40) if a["ripe"] else (230, 60, 60)
        )
    im = Image.fromarray(px)
    draw = ImageDraw.Draw(im)
    for a in anns:
        x, y, w, h = a["bbox"]
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(255, 255, 255), width=1)
        if a["cut_point_px"]:
            u, v = a["cut_point_px"]
            draw.ellipse([u - 3, v - 3, u + 3, v + 3], fill=(0, 200, 255))
        draw.text((x + 2, y + 1), f"{a['ripeness']:.2f}", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def validate(ds: Path, overlays: int = 0) -> tuple[dict, list[str]]:
    from PIL import Image

    coco, meta = load_dataset(ds)
    spec = load_spec()
    problems = []
    ids = [a["id"] for a in coco["annotations"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate annotation ids")
    by_image = {img["id"]: [] for img in coco["images"]}
    for a in coco["annotations"]:
        if a["image_id"] not in by_image:
            problems.append(f"ann {a['id']}: unknown image_id {a['image_id']}")
            continue
        by_image[a["image_id"]].append(a)

    for k, img in enumerate(coco["images"]):
        files = {
            key: ds / img[key] for key in ("file_name", "mask_file", "depth_file") if img.get(key)
        }
        missing = [str(p.relative_to(ds)) for p in files.values() if not p.exists()]
        if missing:
            problems.append(f"image {img['id']}: missing {missing}")
            continue
        mask_ids = decode_ids(np.array(Image.open(files["mask_file"]).convert("RGB")))
        depth = np.array(Image.open(files["depth_file"])) if "depth_file" in files else None
        for a in by_image[img["id"]]:
            problems += check_annotation(a, img, mask_ids)
            if depth is not None:
                problems += check_depth(a, depth)
        problems += check_meta(img["id"], meta.get(img["id"]), spec)
        if k < overlays:
            draw_overlay(
                ds, img, by_image[img["id"]], ds / "overlays" / Path(img["file_name"]).name
            )

    stats = summarize(coco, meta)
    (ds / "report.md").write_text(report_markdown(stats, problems), encoding="utf-8")
    return stats, problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate a generated Harvix dataset")
    p.add_argument("dataset", type=Path)
    p.add_argument(
        "--overlays", type=int, default=8, help="Draw label overlays for the first N images"
    )
    args = p.parse_args(argv)
    stats, problems = validate(args.dataset.resolve(), args.overlays)
    print(json.dumps(stats, indent=1))
    for line in problems[:20]:
        print("[validate]", line)
    print(f"[validate] {'OK' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
