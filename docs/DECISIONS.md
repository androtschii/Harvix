# Decisions and assumptions

Living log. Each entry: what, why, what it costs. The write-up summarizes this file.

## D1. Target task: truss perception from the arm camera

**Built for:** the perception team training the model that tells the arm where to cut.

The robot cuts the peduncle and grips it (tool: top blade pair cuts, bottom pair grips), so the unit it acts on
is the **truss**, not a single fruit. The dataset therefore answers three questions per image:

1. Where are the trusses? Instance mask and 2D box per truss.
2. Which ones are ready? Ripeness score per truss (from the scene's `ripe` attribute) and a ripe/unripe flag.
3. Where exactly do I cut? 3D cut point on the peduncle and 3D truss centre, in the camera frame and the
   cut-TCP frame, plus visibility (how much of the truss is occluded).

Also stored, cheap to add: depth map, camera intrinsics and extrinsics, all `q_<joint>` values and the `H01` pose.

**Left out on purpose:** single-fruit labels, grasp/trajectory labels, video sequences, physics (leaf
deformation by the arm), pest and disease classes. Each needs either assets we don't have or a different model
than the one asked for first.

## D2. Camera: robot-mounted, on the arm camera link

The robot has camera mount links `H01_DESIGN_J_{left,right}_camera` but no Blender camera. We add a camera
object parented to that link, so it follows the arm through the drivers without touching any robot part.
Images then come from the viewpoint the deployed model will actually see.

**Assumption:** sensor ~ RGB-D close-range camera, 640x480, horizontal FOV ~ 70 deg (typical of
RealSense-class sensors used on harvesters). Recorded per sample, easy to change by flag.

## D3. Annotation unit and masks

Fruit is Geometry Nodes instanced, so object IDs do not separate trusses. Near the camera we realize the
visible truss instances as linked-duplicate objects (shared mesh data, no copy of geometry) with a unique
`pass_index`, and hide the same plants' fruit in the instancer through its `hidden` point attribute (needs the
modifier input `Hide Plants` on). Instance masks come from a second, Workbench pass: flat object colour per
truss, everything else black, no anti-aliasing. It takes ~3 s, respects occlusion and needs no compositor.
Far trusses stay instanced and are treated as background.

**Cost:** a radius cut-off: trusses beyond it are rendered but unlabelled. They are small and mostly occluded
at harvest distance; the radius is a flag and is recorded.

## D4. Domain randomization, kept realistic

Randomized per sample, all from one `--seed`:

- robot lane (x = 9, 11, ..., 91) and Y position along the row, root only;
- arm pose within joint limits, biased to the working envelope in front of the crop wall;
- sun direction and strength, sky/exposure (time of day, cloud);
- truss ripeness via the `ripe` attribute (shift and spread per plant).

Not randomized: robot geometry and mechanisms (fixed by the brief), greenhouse structure, plant species.

## D5. Engine and defaults for a weak machine

Machine: Ryzen 7 3700U, integrated Vega 10, no CUDA, 22 GB RAM. Defaults favour a quick check by anyone:
640x480, low sample count with denoising. Engine choice is made from the step-4 benchmark and recorded in
`docs/benchmark.md`. Quality is a CLI flag away.

## D6. Output format

COCO-style JSON for 2D labels (masks as RLE), plus one `metadata.jsonl` line per sample with the 3D labels,
camera matrices, joint values, robot pose and all random draws. PNG for RGB and masks, 16-bit PNG depth in
millimetres (no EXR, so any tool can read it).

## D7. Prior art: reuse ideas, not dependencies

- **BlenderProc** (DLR): runs its own bundled Blender (3.x/4.x); no 5.x support found, and our `.blend` files
  need 5.2+. Its strength (assembling random scenes) is not needed, the scene is given. Not used.
- **bpycv**: instance masks via `pass_index` plus depth, the same approach as D3. Supports Blender up to 4.x and
  installs into Blender's Python, so it is not a dependency; the idea is reused with plain `bpy`.
- **Barth et al., synthetic sweet pepper in Blender** (WUR): closest precedent, with peduncle segmentation and a
  real harvesting robot. Shows the plant-part labelling is worth it.
- **Tomato-sim** (Isaac Sim Replicator, amodal tomato segmentation) and a UE5 procedural tomato greenhouse:
  same crop, other engines.
- **Real tomato peduncle segmentation set** (Roboflow Universe): candidate real-image test set for the
  sim-to-real check in the validation plan.

## Open questions (would ask the stakeholder; assumed for now)

- Real camera model and mount. Assumed as in D2.
- Ripeness threshold for "harvest now". Assumed `ripe >= 0.8`, where the fruit colour ramp turns red; stored
  as a float so it can be re-thresholded.
- Do arms work one row side each, or both sides? Checked in step 4 from the rig.
