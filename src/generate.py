"""Batch generator: one Blender process per seed, merged into one COCO dataset.

Runs with plain Python (not inside Blender). Each sample gets a fresh scene, a failed seed does not
stop the batch, and a re-run skips seeds that are already done.

    python src/generate.py --seeds 0:20 --out out/dataset

Output layout:
    images/000007.png   RGB
    masks/000007.png    instance ids as colours (decode with annotate.decode_ids)
    depth/000007.png    uint16 millimetres, 0 = no hit
    labels/000007.json  per-sample COCO
    annotations.json    merged COCO
    metadata.jsonl      one line per sample: all random draws, camera, joints, 3D labels
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"rgb.png": "images", "mask.png": "masks", "depth.png": "depth", "coco.json": "labels"}


def parse_seeds(text: str) -> list[int]:
    """'0:5' -> 0..4, '3,7,9' -> [3, 7, 9], '12' -> [12]."""
    if ":" in text:
        start, stop = (int(v) for v in text.split(":", 1))
        if stop <= start:
            raise ValueError(f"Empty seed range {text!r}")
        return list(range(start, stop))
    seeds = [int(v) for v in text.split(",") if v.strip()]
    if not seeds:
        raise ValueError("No seeds given")
    return seeds


def side_for(seed: int, side: str) -> str:
    return ("left", "right")[seed % 2] if side == "alternate" else side


def stem(seed: int) -> str:
    return f"{seed:06d}"


def done_seeds(metadata_path: Path) -> set[int]:
    if not metadata_path.exists():
        return set()
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["seed"] for line in lines if line.strip()}


def merge_coco(samples: list[dict]) -> dict:
    """Merge per-sample COCO dicts; file names point into images/, masks/ and depth/."""
    images, annotations = [], []
    for coco in sorted(samples, key=lambda c: c["images"][0]["id"]):
        img = dict(coco["images"][0])
        name = stem(img["id"]) + ".png"
        img["file_name"] = f"images/{name}"
        img["mask_file"] = f"masks/{name}"
        img["depth_file"] = f"depth/{name}" if img.get("depth_file") else None
        images.append(img)
        annotations.extend(coco["annotations"])
    ids = [a["id"] for a in annotations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation ids across samples")
    return {
        "info": {"description": "Harvix synthetic tomato trusses"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "truss"}],
    }


def blender_cmd(args: argparse.Namespace, seed: int, sample_dir: Path) -> list[str]:
    cmd = [args.blender, "-b", str(args.blend), "--python", str(ROOT / "src" / "annotate.py")]
    cmd += ["--", "--seed", str(seed), "--side", side_for(seed, args.side)]
    cmd += ["--radius", str(args.radius), "--samples", str(args.samples), "--out", str(sample_dir)]
    return cmd


def collect_sample(out: Path, sample_dir: Path, seed: int) -> None:
    """Move one sample's files into the dataset layout and append its metadata line."""
    for src_name, kind in KINDS.items():
        src = sample_dir / src_name
        if src.exists():
            ext = src.suffix
            (out / kind).mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), out / kind / f"{stem(seed)}{ext}")
    meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    with (out / "metadata.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    shutil.rmtree(sample_dir)


def write_merged(out: Path) -> int:
    samples = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "labels").glob("*.json")]
    coco = merge_coco(samples)
    (out / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    return len(coco["images"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render and annotate a batch of seeds")
    p.add_argument("--seeds", type=parse_seeds, required=True, help="'0:100', '3,7' or '12'")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--side", default="alternate", choices=["alternate", "left", "right"])
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--blender", default="blender")
    p.add_argument("--blend", type=Path, default=ROOT / "assets" / "Harvix_Greenhouse.blend")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    done = done_seeds(out / "metadata.jsonl")
    todo = [s for s in args.seeds if s not in done]
    print(
        f"[generate] {len(todo)} to render, {len(args.seeds) - len(todo)} already done", flush=True
    )

    failed = []
    for i, seed in enumerate(todo, 1):
        sample_dir = out / "_work" / stem(seed)
        shutil.rmtree(sample_dir, ignore_errors=True)
        t0 = time.perf_counter()
        proc = subprocess.run(
            blender_cmd(args, seed, sample_dir), capture_output=True, text=True, errors="replace"
        )
        if proc.returncode != 0 or not (sample_dir / "metadata.json").exists():
            failed.append(seed)
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
            print(f"[generate] seed {seed} FAILED (exit {proc.returncode}):\n{tail}", flush=True)
            continue
        collect_sample(out, sample_dir, seed)
        print(
            f"[generate] {i}/{len(todo)} seed {seed}: {time.perf_counter() - t0:.0f} s", flush=True
        )

    shutil.rmtree(out / "_work", ignore_errors=True)
    n = write_merged(out) if (out / "labels").exists() else 0
    print(f"[generate] dataset has {n} images; failed seeds: {failed or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
