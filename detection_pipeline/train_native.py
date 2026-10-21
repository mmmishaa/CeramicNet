"""Launch an official trainer while keeping CeramicNet adapters importable."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def launch(framework, repo, config, extra):
    project = Path(__file__).resolve().parents[1]
    repo, config = Path(repo).resolve(), Path(config).resolve()
    native_root = repo / "rtdetrv2_pytorch" if framework == "rtdetr" else repo
    script = native_root / "tools/train.py"
    if not script.is_file() or not config.is_file():
        raise FileNotFoundError(f"Missing trainer/config: {script}, {config}")
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(native_root))
    # Also covers dataloader workers launched with spawn.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(project), str(native_root), os.environ.get("PYTHONPATH", "")])
    if framework == "rtdetr":
        from .integrations import rtdetr  # noqa: F401; registers TeamCocoDetection
        if not any(arg == "--seed" or arg.startswith("--seed=") for arg in extra):
            from src.core import YAMLConfig
            seed = YAMLConfig(str(config)).yaml_cfg.get("seed", 42)
            extra = [*extra, "--seed", str(seed)]
    sys.argv = [str(script), "-c", str(config), *extra] if framework == "rtdetr" else [str(script), str(config), *extra]
    os.chdir(native_root)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=["rtdetr", "mmdet"])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    args, extra = parser.parse_known_args()
    launch(args.framework, args.repo, args.config, extra[1:] if extra[:1] == ["--"] else extra)
