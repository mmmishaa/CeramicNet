"""Launch an official trainer while keeping CeramicNet adapters importable."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

from experiment_tracking import ExperimentRun
from experiment_tracking.native_logs import NativeLogForwarder
from .data import read_json


def tracking_configuration(framework, config, extra, native_root):
    """Resolve the data/output overrides used by the actual native trainer."""
    parser = argparse.ArgumentParser(add_help=False)
    if framework == "rtdetr":
        from src.core import YAMLConfig, yaml_utils
        parser.add_argument("-u", "--update", nargs="+")
        parser.add_argument("--output-dir")
        parser.add_argument("--seed", type=int)
        args, _ = parser.parse_known_args(extra)
        resolved = YAMLConfig(str(config)).yaml_cfg
        resolved = yaml_utils.merge_dict(resolved, yaml_utils.parse_cli(args.update))
        if args.output_dir:
            resolved["output_dir"] = args.output_dir
        if args.seed is not None:
            resolved["seed"] = args.seed
        output = resolved["output_dir"]
        datasets = {name: resolved[f"{name}_dataloader"]["dataset"] for name in ("train", "val")}
    else:
        from mmengine.config import Config, DictAction
        parser.add_argument("--work-dir")
        parser.add_argument("--cfg-options", nargs="+", action=DictAction)
        args, _ = parser.parse_known_args(extra)
        cfg = Config.fromfile(str(config))
        if args.cfg_options:
            cfg.merge_from_dict(args.cfg_options)
        if args.work_dir:
            cfg.work_dir = args.work_dir
        resolved = cfg.to_dict()
        output = resolved["work_dir"]
        datasets = {name: resolved[f"{name}_dataloader"]["dataset"] for name in ("train", "val")}

    def absolute(value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (native_root / path).resolve()

    splits = {}
    for name, dataset in datasets.items():
        while "ann_file" not in dataset and "dataset" in dataset:
            dataset = dataset["dataset"]
        if "ann_file" not in dataset:
            raise ValueError("Tracking requires an explicit annotation file for each split")
        path = Path(dataset["ann_file"])
        if not path.is_absolute() and dataset.get("data_root"):
            path = Path(dataset["data_root"]) / path
        splits[name] = absolute(path)
    metadata_path = config.with_suffix(".metadata.json")
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    parameters = {**metadata, "framework": framework, "native_config": resolved, "native_arguments": list(extra)}
    parameters["native_val_axis"] = (
        "iteration" if framework == "mmdet" and not resolved.get("default_hooks", {}).get("logger", {}).get("log_metric_by_epoch", True)
        else "epoch")
    try:
        parameters["native_repo_commit"] = subprocess.check_output(
            ["git", "-C", str(native_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        parameters["native_repo_commit"] = "unknown"
    return absolute(output), splits, parameters


def launch(framework, repo, config, extra, tracking="local", tracking_project="CeramicNet", run_name=None):
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
    output, splits, parameters = tracking_configuration(framework, config, extra, native_root)
    with ExperimentRun("detection", run_name or config.stem, output, parameters, splits,
                       mode=tracking, project=tracking_project, tags=[framework]) as run:
        run.artifact("native_config", config)
        previous_cwd, previous_argv = Path.cwd(), sys.argv
        try:
            sys.argv = [str(script), "-c", str(config), *extra] if framework == "rtdetr" else [str(script), str(config), *extra]
            os.chdir(native_root)
            with NativeLogForwarder(run, output, framework, val_axis=parameters["native_val_axis"]):
                runpy.run_path(str(script), run_name="__main__")
        finally:
            os.chdir(previous_cwd)
            sys.argv = previous_argv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=["rtdetr", "mmdet"])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracking", choices=["local", "offline", "clearml"], default="local")
    parser.add_argument("--tracking-project", default="CeramicNet")
    parser.add_argument("--run-name")
    args, extra = parser.parse_known_args()
    launch(args.framework, args.repo, args.config, extra[1:] if extra[:1] == ["--"] else extra,
           args.tracking, args.tracking_project, args.run_name)
