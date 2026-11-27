# Assets

| File | What it is |
|---|---|
| `Harvix_Greenhouse.blend` | A Venlo glasshouse, 96 × 120 m, with about 24,000 tomato plants in rows and robot `H01` on the heating-pipe rails at the end of a row. Self-contained, about 140 MB. |
| `Harvix_Robot.blend` | The Harvix robot on its own. Self-contained, about 90 MB. |
| `Harvix_Robot.glb` | The robot in glTF format, for tools other than Blender. It is a single static pose with no rig, and its materials are simplified. |
| `robot_spec.json` | Robot dimensions, joint names and limits, and preset poses. |

## Facts you need

- **Blender 5.2 or newer** is required to open the `.blend` files. Older versions may not open them correctly. Allow Python auto-run (headless `blender -b` runs it by default), because the arm linkages are driven by expressions.
- **Units and axes:** metres and radians, with +Z up. In the greenhouse, crop rows run along Y. Rail lanes are at x = 9, 11, …, 91, with a 0.55 m gauge.
- **The robot** is a dual-arm machine about 2.43 × 0.85 × 2.55 m. Two arms lift independently on a central mast. Each arm ends in a tool with two blade pairs: the top pair cuts the stem and the bottom pair grips it.
- **Posing the robot:** set the custom properties `q_<joint>` on the root empty (`HV_ROOT` in the robot file, `H01` in the greenhouse), keeping values within the limits in `robot_spec.json`. Don't transform individual robot parts, because that breaks the linkages. To move the whole robot, transform the root. In the greenhouse, move it along a row by changing only its Y location so it stays on the rails.
- **Keep the robot's geometry, proportions and mechanisms unchanged.** Everything else in the scenes can be modified.
