# Scene notes (step 1: inspection)

Produced by `src/inspect_scene.py` (read-only). Raw inventories:
[`inventory_greenhouse.json`](inventory_greenhouse.json), [`inventory_robot.json`](inventory_robot.json).

Reproduce:

```bash
blender -b assets/Harvix_Greenhouse.blend --python src/inspect_scene.py -- --root H01 --out docs/inventory_greenhouse.json
blender -b assets/Harvix_Robot.blend --python src/inspect_scene.py -- --root HV_ROOT --out docs/inventory_robot.json
```

## Cost on this machine

Ryzen 7 3700U, Vega 10 iGPU (no CUDA/OptiX), 22 GB RAM, Blender 5.3.0.

| File | Open + inspect (wall) | Peak RAM |
|---|---|---|
| `Harvix_Greenhouse.blend` | 12 s | 0.6 GB |

Opening the scene is cheap; rendering will dominate.

## Greenhouse

- 828 objects, 118 collections, one sun (`HV_SUN`), world `HV_WORLD_greenhouse`.
- Default render: Cycles, 3840x2160 at 50 %, camera `CAM_Row_End_Hero`. Six film cameras, none on the robot.
- **Crop is procedural.** Plants and fruit are Geometry Nodes instances, not objects:
  - `GH_plant_instancer` and `GH_fruit_instancer` (collection `FIN_TOMATO`) each hold 24,024 points, one per plant.
    Point attributes: `plant_matrix`, `variant_kit` (0-15), `seed`, `plant_index`, `hidden`, `src_full`.
  - Node groups `GH_PlantInstancer` / `GH_FruitInstancer` pick per point one of 16 plant variants
    (`PLANT_V01`..`PLANT_V16`) at near / mid / far LOD and instance its parts.
  - Modifier inputs: `Camera`, `Use Camera`, `Near LOD Distance`, `Mid LOD Distance`, `Cull Distance`,
    `Cull Outside View`, `View Cone Cos`, `Keep Within`, `Hide Plants`. LOD and culling follow a camera,
    so the generator must point them at the robot camera.
  - Grow slabs are a third instancer (`GROWSLAB_instancer`).
- **Truss templates** `PLANT_Vxx_{near,mid,far}_Tnn` (4 to 9 trusses per variant). One mesh per truss with two
  materials: `CROP_V3_fruit` (fruit, material index 0) and `CROP_V3_stalk` (peduncle and pedicels, index 1).
  Near LOD: ~1,300-1,800 verts, ~50 loose parts. Geometry is offset from the plant origin, so the truss position
  is the instance matrix times the mesh bounds.
- **Ripeness is data.** Point attribute `ripe` (float) drives the fruit colour through a ColorRamp in
  `CROP_V3_fruit`. `PLANT_V01_near_T01` has values 0.92-1.0.
- Instances: all fruit instances of one plant share the plant transform. Inside 4 m of `H01` there are
  several hundred near-LOD truss instances.

## Robot `H01` (greenhouse) / `HV_ROOT` (robot file)

- `H01` at (13.0, 55.5, 0.08), rotated 90 deg about Z, i.e. driving along +Y on lane x = 13.
- 121 descendants (42 meshes, 79 empties), 170 drivers. All 21 `q_<joint>` properties are present at 0.
- No Blender camera on the robot, but each arm has a camera mount link:
  `H01_DESIGN_J_{left,right}_camera` with visual `H01_DESIGN_VIS_{left,right}_camera`.
- Tool frames: `H01_CUT_TCP_{Left,Right}` (cut tool centre point), `H01_COORD_Actual_Tip`,
  pins `H01_PIN_{side}_grab_front` / `_cut_front`.

## Consequences for the pipeline

1. Annotate **trusses**, not single fruit: the tool cuts the peduncle and grips it, so a truss is the unit
   the robot acts on. Fruit-level masks stay possible later (loose parts per truss).
2. Instance masks cannot come from object IDs, because instances share the template object. Plan: near the
   camera, realize the visible truss instances as linked-duplicate objects with unique `pass_index` and hide
   those plants in the instancer via the `hidden` point attribute.
3. Mount the render camera on `H01_DESIGN_J_{side}_camera` (new object, parented, mount link untouched).
4. Cut-point label: top of the stalk geometry of each truss, transformed to camera and TCP frames.
5. Ripeness label and randomization through the `ripe` attribute.
