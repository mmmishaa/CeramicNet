"""Live forwarding of native JSONL logs without modifying native model trainers."""
from contextlib import AbstractContextManager
import json
from numbers import Real
from pathlib import Path
import threading


COCO_NAMES = ("map_50_95", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
              "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large")


def report_native_record(run, record, framework, val_axis="epoch"):
    if framework == "rtdetr":
        epoch = int(record["epoch"]) + 1
        run.log_metrics({key[6:]: value for key, value in record.items() if key.startswith("train_")},
                        epoch, "train")
        for key, value in record.items():
            if key == "test_coco_eval_bbox":
                run.log_metrics({name: v for name, v in zip(COCO_NAMES, value) if v >= 0}, epoch, "val_native")
            elif key.startswith("test_"):
                run.log_metrics({key[5:]: value}, epoch, "val_native")
    elif framework == "mmdet":
        is_val = any(key.startswith(("val/", "coco/")) or "bbox_mAP" in key for key in record)
        metrics = {key.removeprefix("val/").removeprefix("train/"): value for key, value in record.items()
                   if key not in {"epoch", "iter", "step", "mode"}
                   and not ("bbox_mAP" in key and isinstance(value, Real) and value < 0)}
        run.log_metrics(metrics, record.get("step", record.get("iter", 0)),
                        "val_native" if is_val else "train", val_axis if is_val else "iteration")
    else:
        raise ValueError(framework)


class NativeLogForwarder(AbstractContextManager):
    """Tail complete lines, ignore historical files, and flush final metrics on exit."""
    def __init__(self, run, output_dir, framework, interval=1.0, val_axis="epoch"):
        self.run, self.root, self.framework = run, Path(output_dir), framework
        self.interval = interval
        self.val_axis = val_axis
        self.offsets, self.error = {}, None
        self.stop = threading.Event()

    def paths(self):
        return [self.root / "log.txt"] if self.framework == "rtdetr" else list(self.root.glob("**/vis_data/scalars.json"))

    def __enter__(self):
        if not self.run.enabled:
            return self
        self.offsets = {str(path): path.stat().st_size for path in self.paths() if path.is_file()}
        self.thread = threading.Thread(target=self.follow, name="native-metrics", daemon=True)
        self.thread.start()
        return self

    def poll(self):
        for path in self.paths():
            if not path.is_file():
                continue
            key = str(path)
            offset = self.offsets.get(key, 0)
            if path.stat().st_size < offset:
                offset = 0
            with path.open(encoding="utf-8") as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line.endswith("\n"):
                        break
                    record = json.loads(line)
                    report_native_record(self.run, record, self.framework, self.val_axis)
                    self.offsets[key] = stream.tell()

    def follow(self):
        while not self.stop.wait(self.interval):
            try:
                self.poll()
            except Exception as error:
                self.error = error
                break

    def __exit__(self, exc_type, exc, traceback):
        if self.run.enabled:
            self.stop.set()
            self.thread.join()
            if exc_type is None:
                if self.error:
                    raise RuntimeError("Failed to forward native metrics") from self.error
                self.poll()
            else:
                # Preserve the original training failure even if log forwarding also failed.
                try:
                    self.poll()
                except Exception:
                    pass
        return False
