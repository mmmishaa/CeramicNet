"""Run with Python 3.12 to install the isolated RTMDet CPU notebook environment."""
import os
from pathlib import Path
import subprocess
import sys
import venv


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[2]


def run(*args, cwd=PROJECT, env=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def main():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use Python 3.12 for the pinned torch/torchvision environment")
    target = PROJECT / ".venv-rtmdet"
    if not (target / "bin/python").exists():
        venv.EnvBuilder(with_pip=True).create(target)
    python = target / "bin/python"
    run(python, "-m", "pip", "install", "-r", FOLDER / "requirements-local.txt")
    sources = PROJECT / ".local/rtmdet"
    sources.mkdir(parents=True, exist_ok=True)
    for name, tag, commit in (
        ("mmcv", "v2.1.0", "57c4e25e06e2d4f8a9357c84bcd24089a284dc88"),
        ("mmdetection", "v3.3.0", "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"),
    ):
        destination = sources / name
        if not destination.exists():
            run("git", "clone", "--depth", "1", "--branch", tag,
                f"https://github.com/open-mmlab/{name}.git", destination)
        actual = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Unexpected source version in {destination}: {actual}")
    probe = subprocess.run([str(python), "-c", "from mmcv.ops import nms"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode:
        env = {**os.environ, "MMCV_WITH_OPS": "1", "MAX_JOBS": "4",
               "PATH": str(target / "bin") + os.pathsep + os.environ.get("PATH", "")}
        # New Apple Clang diagnoses a specialization in torch 2.5's bundled
        # strong_type.h. Keep it a warning for this third-party build only.
        if sys.platform == "darwin":
            for key in ("CFLAGS", "CXXFLAGS"):
                env[key] = env.get(key, "") + " -Wno-error=invalid-specialization"
        # MMCV 2.1 auto-detects MPS while building. Request a CPU-only extension
        # in this build process without changing PyTorch or upstream source.
        run(python, "-c", "import torch, runpy, sys; "
            "torch.backends.mps.is_available = lambda: False; "
            "sys.argv = ['setup.py', 'bdist_wheel']; "
            "runpy.run_path('setup.py', run_name='__main__')",
            cwd=sources / "mmcv", env=env)
        wheels = list((sources / "mmcv/dist").glob("mmcv-2.1.0-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one built MMCV wheel, found {wheels}")
        run(python, "-m", "pip", "install", wheels[0])
    run(python, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e",
        sources / "mmdetection")
    run(python, "-m", "ipykernel", "install", "--user", "--name", "ceramicnet-rtmdet",
        "--display-name", "CeramicNet RTMDet (CPU)")
    run(python, "-c", "import torch, mmcv, mmdet; from mmcv.ops import nms; "
        "print('Ready:', torch.__version__, mmcv.__version__, mmdet.__version__)")


if __name__ == "__main__":
    main()
