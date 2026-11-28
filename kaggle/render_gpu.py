"""Kaggle GPU runner: fetch Blender, clone the repo, render with the pipeline scripts.

Edit RUNS to choose what to render. Outputs land in /kaggle/working/<run name>/.
Push with:  kaggle kernels push -p kaggle
"""

import shutil
import subprocess
import threading
from pathlib import Path

BLENDER_URL = "https://download.blender.org/release/Blender5.3/blender-5.3.0-linux-x64.tar.xz"
REPO = "https://github.com/androtschii/Harvix.git"
BRANCH = "feat/gpu-bench"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
TMP = Path("/tmp/harvix")

RUNS = {
    "bench_cuda": ["src/bench_render.py", "--device", "CUDA", "--frames", "3"],
    "bench_optix": ["src/bench_render.py", "--device", "OPTIX", "--frames", "3"],
}
# A hung render otherwise burns the whole 12 h GPU session (seen three times).
RUN_TIMEOUT_S = 20 * 60


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
        print("$", " ".join(cmd), flush=True)
        # Stream Blender's output into the kernel log: a cancelled kernel keeps no output files.
        with (WORK / f"{name}.log").open("w") as log:
            proc = subprocess.Popen(
                cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            timer = threading.Timer(RUN_TIMEOUT_S, proc.kill)
            timer.start()
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)
            timer.cancel()
            code = proc.wait()
        if code != 0:
            print(
                f"[runner] {name} exited with {code} (killed after {RUN_TIMEOUT_S} s if -9)",
                flush=True,
            )


if __name__ == "__main__":
    main()
