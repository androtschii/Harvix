"""Render a few frames of an existing scene camera and log timings.

Usage:
    blender -b assets/Harvix_Greenhouse.blend --python src/bench_render.py -- \
        --device CPU --frames 3 --out-dir out/bench_cpu

Writes PNG frames and bench.json (per-frame seconds, devices, settings). Never saves the .blend.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    p.add_argument("--device", default="CPU", choices=["CPU", "CUDA", "OPTIX"])
    p.add_argument("--camera", default="CAM_In_Row")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def setup_cycles_device(device):
    """Enable every device of the requested backend; return their names."""
    scene = bpy.context.scene
    if device == "CPU":
        scene.cycles.device = "CPU"
        return ["CPU"]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device
    prefs.refresh_devices()
    used = []
    for d in prefs.devices:
        d.use = d.type == device
        if d.use:
            used.append(d.name)
    if not used:
        raise SystemExit(
            f"[bench] no {device} device found: {[(d.name, d.type) for d in prefs.devices]}"
        )
    scene.cycles.device = "GPU"
    return used


def peak_rss_gb():
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)
    except ImportError:
        return None  # Windows: measured outside the process


def main():
    args = parse_args()
    scene = bpy.context.scene
    scene.camera = bpy.data.objects[args.camera]
    scene.render.engine = args.engine
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"

    devices = ["GPU (EEVEE)"]
    if args.engine == "CYCLES":
        devices = setup_cycles_device(args.device)
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPTIX" if args.device == "OPTIX" else "OPENIMAGEDENOISE"

    out = Path(args.out_dir).resolve()  # bpy resolves relative render paths against the drive root
    out.mkdir(parents=True, exist_ok=True)
    times = []
    for i in range(args.frames):
        scene.render.filepath = str(out / f"frame_{i:03d}.png")
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        times.append(round(time.perf_counter() - t0, 2))
        print(f"[bench] frame {i}: {times[-1]} s")

    report = {
        "blender": bpy.app.version_string,
        "host": platform.node(),
        "engine": args.engine,
        "device": args.device,
        "devices_used": devices,
        "camera": args.camera,
        "resolution": [args.width, args.height],
        "samples": args.samples if args.engine == "CYCLES" else None,
        "frame_seconds": times,
        "first_frame_s": times[0],
        "steady_frame_s": round(sum(times[1:]) / max(len(times) - 1, 1), 2)
        if len(times) > 1
        else None,
        "peak_rss_gb": peak_rss_gb(),
    }
    (out / "bench.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("[bench] " + json.dumps(report))


if __name__ == "__main__":
    main()
