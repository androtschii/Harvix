"""Truss annotations for one randomized sample (decisions D1, D3, D6).

Near the camera, visible truss instances are realized as linked-duplicate objects (shared mesh, no
geometry copy) and their plants are hidden in the fruit instancer. An RGB frame is rendered with
Cycles, then an instance-ID frame with Workbench (flat colour per truss, no AA), which gives exact,
occlusion-aware masks in seconds.

    blender -b assets/Harvix_Greenhouse.blend --python src/annotate.py -- --seed 1 --out out/s1
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_camera import (  # noqa: E402
    CameraSpec,
    add_camera,
    blender_to_opencv,
    intrinsics,
    point_instancers_at,
    project,
)

ID_STEP = 32  # mask colour grid in 8-bit sRGB: 8 levels per channel
MAX_ID = 8**3 - 1  # id 0 is background
# CROP_V3_fruit's colour ramp turns red at 0.8 (0.58 orange, 1.0 deep red). The float is stored too.
RIPE_THRESHOLD = 0.8
RIPE_OFFSET = 0.1  # per-truss spread around the sample-wide shift
FRUIT_MAT, STALK_MAT = 0, 1
PLANT_MATCH_M = 0.05
CUT_VISIBLE_TOL_M = 0.03  # depth vs cut-point z; peduncles are ~5 mm thick


# ---------- pure helpers ----------


def id_to_srgb8(i: int) -> tuple[int, int, int]:
    if not 0 <= i <= MAX_ID:
        raise ValueError(f"id {i} outside 0..{MAX_ID}")
    return tuple(((i >> s) & 7) * ID_STEP for s in (0, 3, 6))


def srgb8_to_linear(c: int) -> float:
    v = c / 255
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def decode_ids(rgb8: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 mask image -> HxW int id map (0 = background)."""
    q = np.rint(rgb8[..., :3].astype(np.int32) / ID_STEP).astype(np.int32).clip(0, 7)
    return q[..., 0] | (q[..., 1] << 3) | (q[..., 2] << 6)


def depth_to_mm16(depth_m: np.ndarray) -> np.ndarray:
    """Metres -> uint16 millimetres; 0 marks no hit or beyond 65.535 m (sky is 1e10)."""
    mm = np.rint(np.asarray(depth_m, dtype=np.float64) * 1000.0)
    mm[(mm <= 0) | (mm > 65535) | ~np.isfinite(mm)] = 0
    return mm.astype(np.uint16)


def write_png16(path: Path, img: np.ndarray) -> None:
    """Minimal 16-bit greyscale PNG writer (no PIL in Blender's Python)."""
    img = np.asarray(img, dtype=">u2")
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6))
    Path(path).write_bytes(png + chunk(b"IEND", b""))


def rle_encode(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE (column-major, counts start with background)."""
    flat = np.asarray(mask, dtype=bool).flatten(order="F")
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts = [0, *counts]
    return {"size": list(mask.shape), "counts": counts}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for n in rle["counts"]:
        flat[pos : pos + n] = val
        pos += n
        val = not val
    return flat.reshape((h, w), order="F")


def bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def ripeness(template_mean: float, shift: float, offset: float) -> float:
    """Matches the shader: clamp(ripe + offset + shift), averaged over the truss's fruit."""
    return min(1.0, max(0.0, template_mean + offset + shift))


def cut_point_visible(depth_m: np.ndarray, uv, z: float, tol: float = CUT_VISIBLE_TOL_M):
    """True if the rendered surface at the cut pixel (3x3 window) is the cut point itself.

    None when the point projects outside the image or behind the camera.
    """
    if uv is None or z <= 0:
        return None
    h, w = depth_m.shape
    u, v = int(uv[0]), int(uv[1])
    if not (0 <= u < w and 0 <= v < h):
        return None
    win = depth_m[max(v - 1, 0) : v + 2, max(u - 1, 0) : u + 2]
    win = win[(win > 0) & np.isfinite(win)]
    return bool(win.size) and bool(np.abs(win - z).min() <= tol)


def build_annotations(ids: np.ndarray, trusses: list[dict], image_id: int) -> list[dict]:
    """COCO annotations for trusses with at least one visible pixel."""
    anns = []
    for t in trusses:
        ripeness_r = round(t["ripeness"], 4)  # the flag must agree with the stored value
        mask = ids == t["id"]
        area = int(mask.sum())
        t["visible_px"] = area
        if area == 0:
            continue
        anns.append(
            {
                "id": image_id * 1000 + t["id"],
                "truss_id": t["id"],
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": area,
                "bbox": bbox_xywh(mask),
                "segmentation": rle_encode(mask),
                "ripeness": ripeness_r,
                "ripe": ripeness_r >= RIPE_THRESHOLD,
                "cut_point_px": t["cut_point_px"],
                "cut_point_visible": t.get("cut_point_visible"),
                "cut_point_cam": t["cut_point_cam"],
                "center_cam": t["center_cam"],
                "cut_point_tcp": t["cut_point_tcp"],
            }
        )
    return anns


# ---------- bpy layer ----------


def _template_stats(mesh) -> dict:
    """Mean `ripe` of the fruit and the stalk vertices of a truss template mesh (local coords)."""
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mats = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("material_index", mats)
    fruit, stalk = set(), set()
    for poly, m in zip(mesh.polygons, mats, strict=True):
        (fruit if m == FRUIT_MAT else stalk if m == STALK_MAT else set()).update(poly.vertices)
    ripe = np.zeros(len(mesh.vertices))
    if "ripe" in mesh.attributes:
        mesh.attributes["ripe"].data.foreach_get("value", ripe)
    fruit_idx = sorted(fruit)
    return {
        "ripe_mean": float(ripe[fruit_idx].mean()) if fruit_idx else 0.0,
        "stalk": co[sorted(stalk)] if stalk else co,
        "center": (co.min(0) + co.max(0)) / 2,
    }


def collect_plants(cam, radius: float) -> dict:
    """Plants with at least one truss within `radius` of the camera and in front of it."""
    import bpy
    from mathutils import Vector
    from mathutils.kdtree import KDTree

    # persistent_id is per Instance-on-Points node, not the plant point, so match by position:
    # a truss instance sits within ~1 cm of its plant point, plants are tens of cm apart.
    fi = bpy.data.objects["GH_fruit_instancer"]
    points = fi.data.vertices
    tree = KDTree(len(points))
    for i, v in enumerate(points):
        tree.insert(fi.matrix_world @ v.co, i)
    tree.balance()

    cam_pos = cam.matrix_world.translation
    to_cam = cam.matrix_world.inverted()
    by_plant = defaultdict(list)
    hit = set()
    dg = bpy.context.evaluated_depsgraph_get()
    for inst in dg.object_instances:
        if not (inst.is_instance and inst.parent and inst.parent.name == "GH_fruit_instancer"):
            continue
        ob = inst.instance_object
        mw = inst.matrix_world.copy()
        _, plant, dist = tree.find(mw.translation)
        if dist > PLANT_MATCH_M:
            raise RuntimeError(
                f"No plant point within {PLANT_MATCH_M} m of {ob.name} ({dist:.3f} m)"
            )
        by_plant[plant].append((ob.name, mw))
        if "_near_" in ob.name:
            c = mw @ (sum((Vector(b) for b in ob.bound_box), Vector()) / 8)
            if (c - cam_pos).length <= radius and (to_cam @ c).z < 0:
                hit.add(plant)
    return {p: by_plant[p] for p in sorted(hit)}


def hide_plants(plants) -> None:
    import bpy

    from robot_camera import _set_input

    fi = bpy.data.objects["GH_fruit_instancer"]
    hidden = fi.data.attributes["hidden"]
    for p in plants:
        hidden.data[p].value = True
    fi.data.update()
    for mod in fi.modifiers:
        if mod.type == "NODES":
            _set_input(mod, "Hide Plants", True)  # the node group ignores `hidden` otherwise
    fi.update_tag()


def add_ripe_offset_hook(material: str = "CROP_V3_fruit") -> None:
    """ripe + object property `hx_ripe_offset` (0 on instanced trusses) feeds the shift node."""
    import bpy

    tree = bpy.data.materials[material].node_tree
    if tree.nodes.get("HX_ripe_offset_add"):
        return
    shift = tree.nodes["HX_ripe_shift"]  # created by randomize.apply_ripe_shift
    link = shift.inputs[0].links[0]
    src = link.from_socket
    tree.links.remove(link)
    attr = tree.nodes.new("ShaderNodeAttribute")
    attr.name = "HX_ripe_offset"
    attr.attribute_type = "OBJECT"
    attr.attribute_name = "hx_ripe_offset"
    add = tree.nodes.new("ShaderNodeMath")
    add.name = "HX_ripe_offset_add"
    add.operation = "ADD"
    tree.links.new(src, add.inputs[0])
    tree.links.new(attr.outputs["Fac"], add.inputs[1])
    tree.links.new(add.outputs[0], shift.inputs[0])


def realize(plants: dict, seed: int, ripe_shift: float, cam, tcp, K) -> list[dict]:
    import bpy
    from mathutils import Matrix

    coll = bpy.data.collections.new("HX_TRUSSES")
    bpy.context.scene.collection.children.link(coll)
    rng = random.Random(f"{seed}-ripe")
    to_cam = cam.matrix_world.inverted()
    to_tcp = tcp.matrix_world.inverted()
    stats = {}
    trusses = []
    for plant, items in plants.items():
        for name, mw in items:
            tid = len(trusses) + 1
            if tid > MAX_ID:
                raise RuntimeError(f"More than {MAX_ID} trusses; lower --radius")
            template = bpy.data.objects[name]
            st = stats.setdefault(template.data.name, _template_stats(template.data))
            ob = bpy.data.objects.new(f"HX_truss_{tid:03d}", template.data)
            ob.matrix_world = mw
            ob.pass_index = tid
            offset = rng.uniform(-RIPE_OFFSET, RIPE_OFFSET)
            ob["hx_ripe_offset"] = offset
            coll.objects.link(ob)

            m = np.array(Matrix(mw))
            stalk_w = st["stalk"] @ m[:3, :3].T + m[:3, 3]
            cut_w = stalk_w[np.argmax(stalk_w[:, 2])]  # top of the peduncle, where it is cut
            center_w = m[:3, :3] @ st["center"] + m[:3, 3]
            cut_cam = blender_to_opencv(to_cam @ _vec(cut_w))
            uv = project(K, cut_cam)
            trusses.append(
                {
                    "id": tid,
                    "plant": int(plant),
                    "template": name,
                    "ripeness": ripeness(st["ripe_mean"], ripe_shift, offset),
                    "cut_point_px": [round(v, 2) for v in uv] if uv else None,
                    "cut_point_cam": [round(v, 4) for v in cut_cam],
                    "center_cam": [round(v, 4) for v in blender_to_opencv(to_cam @ _vec(center_w))],
                    "cut_point_tcp": [round(v, 4) for v in to_tcp @ _vec(cut_w)],
                }
            )
    bpy.context.view_layer.update()
    return trusses


def _vec(a):
    from mathutils import Vector

    return Vector((float(a[0]), float(a[1]), float(a[2])))


def setup_depth_output(directory: Path) -> None:
    """Compositor: Render Layers Depth -> single-layer float EXR next to the RGB frame."""
    import bpy

    sc = bpy.context.scene
    sc.view_layers[0].use_pass_z = True
    ng = bpy.data.node_groups.new("HX_depth", "CompositorNodeTree")
    rl = ng.nodes.new("CompositorNodeRLayers")
    rl.scene = sc
    fo = ng.nodes.new("CompositorNodeOutputFile")
    fo.directory = str(directory)
    fo.file_name = ""
    fo.format.media_type = "IMAGE"  # Blender 5.x: defaults to multilayer
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.format.color_mode = "BW"
    fo.file_output_items.clear()
    item = fo.file_output_items.new("FLOAT", "depth")
    ng.links.new(rl.outputs["Depth"], fo.inputs[item.name])
    sc.compositing_node_group = ng


def read_depth_exr(directory: Path) -> np.ndarray:
    """Read the depth EXR written by the compositor (top-down rows, metres) and delete it."""
    import bpy

    exr = next(directory.glob("depth*.exr"))
    img = bpy.data.images.load(str(exr))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)[::-1, :, 0]
    bpy.data.images.remove(img)
    exr.unlink()
    return px


def render_id_pass(trusses: list[dict], path: Path) -> np.ndarray:
    """Workbench render: flat object colour = truss id, everything else black."""
    import bpy

    sc = bpy.context.scene
    sc.compositing_node_group = None  # the depth compositor would overwrite depth with Workbench Z
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "FLAT"
    sh.color_type = "OBJECT"
    sc.display.render_aa = "OFF"
    sc.render.dither_intensity = 0
    vs = sc.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = 0
    vs.gamma = 1
    sc.world.color = (0, 0, 0)
    sc.render.film_transparent = False
    for ob in bpy.data.objects:
        ob.color = (0, 0, 0, 1)
    for t in trusses:
        rgb = id_to_srgb8(t["id"])
        bpy.data.objects[f"HX_truss_{t['id']:03d}"].color = (*map(srgb8_to_linear, rgb), 1)
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(path))
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1]  # bottom-up -> top-down
    return decode_ids(np.rint(px * 255).astype(np.uint8))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Render and annotate one randomized sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--radius", type=float, default=2.0, help="Label trusses within this range (m)")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--no-rgb", action="store_true", help="Skip the Cycles frame (labels only)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import bpy

    from randomize import apply_light, apply_ripe_shift, apply_robot, sample
    from robot_control import load_spec

    args = parse_args(argv)
    out = args.out.resolve()  # bpy resolves bare relative paths against the drive root
    out.mkdir(parents=True, exist_ok=True)

    spec = load_spec()
    params = sample(args.seed, spec)
    apply_robot(params, spec)
    apply_light(params["light"])
    apply_ripe_shift(params["ripe_shift"])
    add_ripe_offset_hook()
    cam_spec = CameraSpec()
    cam = add_camera("H01", args.side, cam_spec)
    point_instancers_at(cam)
    K = intrinsics(cam_spec)
    tcp = bpy.data.objects[f"H01_CUT_TCP_{args.side.capitalize()}"]

    plants = collect_plants(cam, args.radius)
    hide_plants(plants)
    trusses = realize(plants, args.seed, params["ripe_shift"], cam, tcp, K)

    sc = bpy.context.scene
    sc.render.image_settings.file_format = "PNG"
    if not args.no_rgb:
        sc.render.engine = "CYCLES"
        sc.cycles.samples = args.samples
        sc.cycles.use_denoising = True
        setup_depth_output(out)
        sc.render.filepath = str(out / "rgb.png")
        bpy.ops.render.render(write_still=True)
        depth_m = read_depth_exr(out)
        write_png16(out / "depth.png", depth_to_mm16(depth_m))
        for t in trusses:
            t["cut_point_visible"] = cut_point_visible(
                depth_m, t["cut_point_px"], t["cut_point_cam"][2]
            )
    ids = render_id_pass(trusses, out / "mask.png")

    anns = build_annotations(ids, trusses, image_id=args.seed)
    image = {
        "id": args.seed,
        "file_name": "rgb.png",
        "mask_file": "mask.png",
        "depth_file": None if args.no_rgb else "depth.png",
        "width": cam_spec.width,
        "height": cam_spec.height,
    }
    coco = {
        "images": [image],
        "annotations": anns,
        "categories": [{"id": 1, "name": "truss"}],
    }
    (out / "coco.json").write_text(json.dumps(coco), encoding="utf-8")
    meta = {
        **params,
        "camera": {
            "side": args.side,
            "K": K,
            "world_from_cam": [list(r) for r in cam.matrix_world],
        },
        "label_radius_m": args.radius,
        "render": {
            "samples": None if args.no_rgb else args.samples,
            "blender": bpy.app.version_string,
        },
        "trusses": trusses,
    }
    (out / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    visible = sum(1 for t in trusses if t["visible_px"])
    print(f"[annotate] {len(trusses)} realized, {visible} visible -> {out}", flush=True)


if __name__ == "__main__":
    main()
