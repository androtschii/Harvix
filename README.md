# Harvix synthetic dataset

Synthetic training data for the Harvix dual-arm tomato-harvesting robot, rendered headless
from the provided Venlo glasshouse scene with Blender.

> Work in progress. Sections below are filled in as the pipeline lands.

## Requirements

- Blender 5.2 or newer (tested with 5.3.0), available as `blender` or via `BLENDER` env var
- Python 3.11+ with `pytest` and `ruff` for the non-Blender checks

## Setup

1. Clone this repository.
2. Copy the files from the take-home package `assets/` folder into `assets/`
   (`Harvix_Greenhouse.blend`, `Harvix_Robot.blend`, `robot_spec.json`).
   The `.blend` files are 90-145 MB and are not stored in git.

## Layout

| Path | Purpose |
|---|---|
| `assets/` | Scene files and `robot_spec.json` (blends git-ignored) |
| `src/` | Pipeline scripts, run inside `blender -b` |
| `tests/` | Unit tests for the pure-Python parts |
| `docs/` | Assignment, decisions, benchmarks, write-up |
| `ai_logs/` | Raw, unedited AI session exports (required by the assignment) |
