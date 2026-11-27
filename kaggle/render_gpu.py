"""Kaggle GPU runner: fetch Blender, clone the repo, render with the pipeline scripts.

Edit RUNS to choose what to render. Outputs land in /kaggle/working/<run name>/.
Push with:  kaggle kernels push -p kaggle
"""

import shutil
import subprocess
from pathlib import Path

BLENDER_URL = "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"
REPO = "https://github.com/androtschii/Harvix.git"
BRANCH = "feat/gpu-bench"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TMP = Path("/tmp/harvix")

RUNS = {
    "bench_optix": ["src/bench_render.py", "--device", "OPTIX", "--frames", "3"],
    "bench_cuda": ["src/bench_render.py", "--device", "CUDA", "--frames", "3"],
}


def sh(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"])

    blender_dir = TMP / "blender"
    if not blender_dir.exists():
        sh(["wget", "-q", "-O", str(TMP / "blender.tar.xz"), BLENDER_URL])
        sh(["tar", "-xf", str(TMP / "blender.tar.xz"), "-C", str(TMP)])
        next(TMP.glob("blender-*-linux-x64")).rename(blender_dir)
    blender = blender_dir / "blender"
    sh([str(blender), "--version"])

    repo = TMP / "repo"
    shutil.rmtree(repo, ignore_errors=True)
    sh(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, str(repo)])
    # The mount path of attached datasets differs between Kaggle versions; locate by file name.
    assets = next(INPUT.rglob("Harvix_Greenhouse.blend")).parent
    print(f"[runner] assets at {assets}", flush=True)
    for f in assets.iterdir():
        target = repo / "assets" / f.name
        if not target.exists():
            target.symlink_to(f)

    for name, (script, *args) in RUNS.items():
        out = WORK / name
        cmd = [str(blender), "-b", "assets/Harvix_Greenhouse.blend", "--python", script, "--"]
        cmd += [*args, "--out-dir", str(out)]
        log = (WORK / f"{name}.log").open("w")
        try:
            sh(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"[runner] {name} failed: {e}", flush=True)
        finally:
            log.close()
        for line in (WORK / f"{name}.log").read_text(errors="replace").splitlines():
            if line.startswith("[bench]") or "Error" in line:
                print(line, flush=True)


if __name__ == "__main__":
    main()
