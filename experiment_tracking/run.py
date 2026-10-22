"""Local JSONL metrics with optional ClearML; no dependency on a model or dataset."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import random
import sys
import threading


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_value,
                                    allow_nan=False) + "\n", encoding="utf-8")


def split_record(split_files):
    """Hash supplied files rather than assuming the detection split for every task."""
    if not split_files:
        raise ValueError("Provide this task's actual split files")
    files = {}
    for name, path in sorted(split_files.items()):
        path = Path(path).resolve()
        files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    identity = {name: info["sha256"] for name, info in files.items()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": fingerprint, "files": files}


def main_process():
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if name in os.environ:
            return int(os.environ[name]) == 0
    return True


@contextmanager
def preserve_random_state():
    """Task.init seeds global RNGs; logging must not change the trainer's RNG state."""
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    torch = sys.modules.get("torch")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch_state = torch.random.get_rng_state() if torch is not None else None
    cuda_state = torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_initialized() else None
    cudnn_state = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) if torch is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_state
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class ExperimentRun:
    """One run, one task kind and one explicit split identity.

    mode=local requires no ClearML SDK or credentials. offline saves a genuine
    ClearML offline session; clearml sends the explicitly reported data to the server.
    Create this context before seeding/building the model. Only rank zero writes.
    """
    def __init__(self, task_kind, run_name, output_dir, config, split_files,
                 mode="local", project="CeramicNet", tags=None):
        if mode not in {"local", "offline", "clearml"}:
            raise ValueError(f"Unknown tracking mode: {mode}")
        self.kind, self.name, self.mode = task_kind, run_name, mode
        self.root = Path(output_dir).resolve() / "tracking"
        self.config = json.loads(json.dumps(config, default=json_value, allow_nan=False))
        self.split_files = split_files
        self.project, self.tags = f"{project.rstrip('/')}/{task_kind}", list(tags or [])
        self.task = self.logger = None
        self.enabled = main_process()
        self.lock = threading.RLock()
        self.summary_values = {}

    def __enter__(self):
        if not self.enabled:
            return self
        self.split = split_record(self.split_files)
        self.root.mkdir(parents=True, exist_ok=False)
        self.metadata = {"task_kind": self.kind, "run_name": self.name, "project": self.project,
                         "mode": self.mode, "status": "running", "split_fingerprint": self.split["fingerprint"],
                         "started_at": datetime.now(timezone.utc).isoformat()}
        dump(self.root / "config.json", self.config)
        dump(self.root / "split.json", self.split)
        dump(self.root / "run.json", self.metadata)
        if self.mode != "local":
            try:
                with preserve_random_state():
                    from clearml import Task
                    if Task.current_task() is not None:
                        raise RuntimeError("Close the previous ClearML task before starting another run")
                    Task.set_offline(self.mode == "offline")
                    self.task = Task.init(
                        project_name=self.project, task_name=self.name,
                        tags=[f"task:{self.kind}", f"split:{self.split['fingerprint'][:12]}", *self.tags],
                        reuse_last_task_id=False, auto_connect_frameworks=False,
                        auto_connect_arg_parser=False, auto_connect_streams=False,
                        auto_resource_monitoring=False, output_uri=False)
                self.logger = self.task.get_logger()
                parameters = {key: value for key, value in self.config.items()
                              if isinstance(value, (str, int, float, bool))}
                parameters.update(task_kind=self.kind, split_fingerprint=self.split["fingerprint"])
                self.task.connect(parameters, name="Run", ignore_remote_overrides=True)
                self.task.connect_configuration(self.config, name="Resolved configuration", ignore_remote_overrides=True)
                self.task.upload_artifact(name="split_identity", artifact_object=self.split)
                for name, path in self.split_files.items():
                    self.task.upload_artifact(name=f"split_{name}", artifact_object=str(Path(path).resolve()))
                self.metadata["clearml_task_id"] = self.task.id
                if self.mode == "offline":
                    self.metadata["clearml_offline_folder"] = str(self.task.get_offline_mode_folder())
                dump(self.root / "run.json", self.metadata)
            except Exception as error:
                self.metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
                dump(self.root / "run.json", self.metadata)
                if self.task is not None:
                    try:
                        self.task.mark_failed(status_reason=self.metadata["error"])
                    finally:
                        self.task.close()
                raise
        return self

    def log_metrics(self, metrics, step, namespace="train", axis="epoch"):
        if not self.enabled:
            return
        with self.lock:
            for name, value in metrics.items():
                if hasattr(value, "item"):
                    value = value.item()
                if not isinstance(value, Real) or not math.isfinite(value):
                    continue
                event = {"namespace": namespace, "axis": axis, "step": int(step),
                         "metric": name, "value": float(value)}
                with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
                if self.logger is not None:
                    self.logger.report_scalar(title=f"{namespace}/{axis}", series=name,
                                              value=float(value), iteration=int(step))

    def summary(self, **values):
        if not self.enabled:
            return
        self.summary_values.update(values)
        dump(self.root / "summary.json", self.summary_values)
        if self.logger is not None:
            for name, value in values.items():
                if isinstance(value, Real) and math.isfinite(value):
                    self.logger.report_single_value(name, value)

    def artifact(self, name, path):
        if not self.enabled:
            return
        path = Path(path).resolve()
        record = {"name": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with (self.root / "artifacts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.task is not None:
            self.task.upload_artifact(name=name, artifact_object=str(path))

    def __exit__(self, exc_type, exc, traceback):
        if not self.enabled:
            return False
        self.metadata.update(status="failed" if exc_type else "completed",
                             finished_at=datetime.now(timezone.utc).isoformat())
        if exc_type:
            self.metadata["error"] = f"{exc_type.__name__}: {exc}"
        dump(self.root / "run.json", self.metadata)
        if self.task is not None:
            try:
                if exc_type:
                    self.task.mark_failed(status_reason=self.metadata["error"])
            finally:
                self.task.close()
        return False
