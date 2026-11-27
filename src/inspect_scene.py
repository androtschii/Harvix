"""Read-only inventory of a .blend scene.

Usage:
    blender -b assets/Harvix_Greenhouse.blend --python src/inspect_scene.py -- \
        --root H01 --out docs/inventory_greenhouse.json

Never saves the .blend file.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import bpy

KEYWORDS = re.compile(r"tomat|fruit|truss|peduncle|calyx|stem|plant|leaf|crop|row|rail|pipe", re.I)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="H01")
    p.add_argument("--out", required=True)
    p.add_argument("--max-names", type=int, default=40)
    return p.parse_args(argv)


def collection_tree(coll, depth=0, max_depth=4):
    node = {
        "name": coll.name,
        "objects": len(coll.objects),
        "all_objects": len(coll.all_objects),
        "types": dict(Counter(o.type for o in coll.objects)),
    }
    if depth < max_depth and coll.children:
        node["children"] = [collection_tree(c, depth + 1, max_depth) for c in coll.children]
    return node


def name_patterns(names, max_names):
    """Collapse names like Tomato.0001 into Tomato.* with counts."""
    stems = Counter(re.sub(r"[._]\d+$", "", n) for n in names)
    return dict(stems.most_common(max_names))


def object_summary(o):
    info = {
        "type": o.type,
        "parent": o.parent.name if o.parent else None,
        "location": [round(v, 4) for v in o.location],
        "dimensions": [round(v, 4) for v in o.dimensions],
    }
    if o.instance_type != "NONE":
        info["instance_type"] = o.instance_type
        if o.instance_collection:
            info["instance_collection"] = o.instance_collection.name
    mods = [(m.type, m.name) for m in getattr(o, "modifiers", [])]
    if mods:
        info["modifiers"] = mods
    if o.type == "MESH":
        info["materials"] = [s.material.name for s in o.material_slots if s.material]
        info["verts"] = len(o.data.vertices)
    return info


def root_summary(root, max_names):
    props = {k: root[k] for k in root.keys() if not k.startswith("_")}
    props = {k: (v if isinstance(v, (int, float, str)) else str(list(v))) for k, v in props.items()}
    children = list(root.children_recursive)
    drivers = 0
    for o in children:
        ad = o.animation_data
        drivers += len(ad.drivers) if ad else 0
    return {
        "location": [round(v, 4) for v in root.location],
        "rotation_euler": [round(v, 4) for v in root.rotation_euler],
        "custom_props": props,
        "descendants": len(children),
        "descendant_types": dict(Counter(o.type for o in children)),
        "descendants_with_drivers": drivers,
        "descendant_cameras": [o.name for o in children if o.type == "CAMERA"],
        "descendant_name_patterns": name_patterns([o.name for o in children], max_names),
    }


def main():
    args = parse_args()
    t0 = time.perf_counter()
    scene = bpy.context.scene
    objs = list(bpy.data.objects)

    keyword_objs = [o.name for o in objs if KEYWORDS.search(o.name)]
    instancers = [o for o in objs if o.instance_type != "NONE"]
    geonodes = [o for o in objs if any(m.type == "NODES" for m in getattr(o, "modifiers", []))]

    report = {
        "file": bpy.data.filepath,
        "blender": bpy.app.version_string,
        "scene": scene.name,
        "render": {
            "engine": scene.render.engine,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "percentage": scene.render.resolution_percentage,
            "camera": scene.camera.name if scene.camera else None,
            "frame_range": [scene.frame_start, scene.frame_end],
        },
        "counts": {
            "objects": len(objs),
            "by_type": dict(Counter(o.type for o in objs)),
            "meshes": len(bpy.data.meshes),
            "materials": len(bpy.data.materials),
            "collections": len(bpy.data.collections),
            "node_groups": len(bpy.data.node_groups),
            "images": len(bpy.data.images),
        },
        "collections": collection_tree(scene.collection),
        "object_name_patterns": name_patterns([o.name for o in objs], args.max_names),
        "keyword_object_patterns": name_patterns(keyword_objs, args.max_names),
        "cameras": {o.name: object_summary(o) for o in objs if o.type == "CAMERA"},
        "lights": {
            o.name: {**object_summary(o), "light": o.data.type, "energy": o.data.energy}
            for o in objs
            if o.type == "LIGHT"
        },
        "world": scene.world.name if scene.world else None,
        "instancers": {
            "count": len(instancers),
            "collections": dict(
                Counter(o.instance_collection.name for o in instancers if o.instance_collection)
            ),
        },
        "geometry_nodes_objects": {o.name: object_summary(o) for o in geonodes[: args.max_names]},
        "node_groups": [g.name for g in bpy.data.node_groups][: args.max_names * 2],
        "materials": name_patterns([m.name for m in bpy.data.materials], args.max_names * 2),
        "keyword_samples": {n: object_summary(bpy.data.objects[n]) for n in keyword_objs[:15]},
    }

    root = bpy.data.objects.get(args.root)
    report["root"] = root_summary(root, args.max_names) if root else f"MISSING: {args.root}"

    report["inspect_seconds"] = round(time.perf_counter() - t0, 2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"[inspect] wrote {out} in {report['inspect_seconds']} s")


if __name__ == "__main__":
    main()
