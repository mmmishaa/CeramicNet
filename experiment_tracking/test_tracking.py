import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from . import ExperimentRun
from .native_logs import NativeLogForwarder
from .run import split_record


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train, self.val = self.root / "train.json", self.root / "val.json"
        self.train.write_text('["object-A"]')
        self.val.write_text('["object-B"]')
        self.splits = {"train": self.train, "val": self.val}

    def run_context(self, name="run", **kwargs):
        return ExperimentRun("detection", name, self.root / name, {"seed": 42, "model": "test"}, self.splits, **kwargs)

    def events(self, name="run"):
        return [json.loads(line) for line in (self.root / name / "tracking/metrics.jsonl").read_text().splitlines()]

    def test_split_identity_is_portable_and_changes_with_data(self):
        original = split_record(self.splits)["fingerprint"]
        other = self.root / "elsewhere"
        other.mkdir()
        for name, path in self.splits.items():
            (other / path.name).write_bytes(path.read_bytes())
        copied = {name: other / path.name for name, path in self.splits.items()}
        self.assertEqual(original, split_record(copied)["fingerprint"])
        self.val.write_text('["different-object"]')
        self.assertNotEqual(original, split_record(self.splits)["fingerprint"])

    def test_detection_and_matching_use_separate_projects_and_actual_splits(self):
        with self.run_context() as run:
            run.log_metrics({"loss": 0.5, "missing": None, "nan": float("nan")}, 1)
            run.summary(best_map=0.75)
        matching_val = self.root / "matching_val.csv"
        matching_val.write_text("photo,mark_id\nC.png,mark_12\n")
        with ExperimentRun("matching", "encoder", self.root / "matching", {"seed": 42},
                           {"train": self.train, "val": matching_val}) as run:
            run.log_metrics({"recall@1": 0.8, "mrr": 0.85}, 1, "val")
        detection = json.loads((self.root / "run/tracking/run.json").read_text())
        matching = json.loads((self.root / "matching/tracking/run.json").read_text())
        self.assertEqual(detection["project"], "CeramicNet/detection")
        self.assertEqual(matching["project"], "CeramicNet/matching")
        self.assertNotEqual(detection["split_fingerprint"], matching["split_fingerprint"])
        self.assertEqual(detection["status"], "completed")
        self.assertEqual(len(self.events()), 1)

    def test_failures_mark_run_failed_and_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with self.run_context():
                raise RuntimeError("training failed")
        metadata = json.loads((self.root / "run/tracking/run.json").read_text())
        self.assertEqual(metadata["status"], "failed")

    def test_other_ranks_do_not_create_duplicate_runs(self):
        with patch.dict("os.environ", {"RANK": "1"}):
            with self.run_context() as run:
                run.log_metrics({"loss": 1.0}, 1)
        self.assertFalse((self.root / "run").exists())

    def test_rtdetr_forwarding_ignores_history_and_waits_for_complete_lines(self):
        native = self.root / "native"
        native.mkdir()
        log = native / "log.txt"
        log.write_text(json.dumps({"epoch": 0, "train_loss": 999}) + "\n")
        with self.run_context() as run, NativeLogForwarder(run, native, "rtdetr", interval=60) as watcher:
            with log.open("a") as stream:
                stream.write('{"epoch": 1, "train_loss": 0.5, "test_coco_eval_bbox": [0.7,0.8,0.6,-1]}')
            watcher.poll()
            self.assertFalse((run.root / "metrics.jsonl").exists())
            with log.open("a") as stream:
                stream.write("\n")
            watcher.poll()
            watcher.poll()
        events = self.events()
        self.assertEqual(len(events), 4)
        self.assertEqual({event["step"] for event in events}, {2})
        self.assertEqual(events[0]["value"], 0.5)

    def test_mmdet_training_iteration_and_validation_epoch_are_separate(self):
        native = self.root / "native"
        with self.run_context() as run, NativeLogForwarder(run, native, "mmdet", interval=60) as watcher:
            path = native / "20261022/vis_data/scalars.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"loss": 0.2, "lr": 0.001, "step": 20}) + "\n" +
                            json.dumps({"coco/bbox_mAP": 0.7, "step": 1}) + "\n")
            watcher.poll()
        events = self.events()
        self.assertEqual(events[0]["axis"], "iteration")
        self.assertEqual(events[-1]["axis"], "epoch")
        self.assertEqual(events[-1]["namespace"], "val_native")

    def test_corrupt_native_logs_surface_without_hiding_training_failure(self):
        native = self.root / "native"
        native.mkdir()
        with self.assertRaisesRegex(ValueError, "model exploded"):
            with self.run_context() as run, NativeLogForwarder(run, native, "rtdetr", interval=60):
                (native / "log.txt").write_text("bad json\n")
                raise ValueError("model exploded")

    @unittest.skipUnless(importlib.util.find_spec("clearml"), "optional ClearML SDK not installed")
    def test_real_clearml_offline_session_and_random_state(self):
        import numpy as np
        import torch

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        before_python, before_numpy, before_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.run_context(mode="offline") as run:
            self.assertEqual(random.getstate(), before_python)
            self.assertTrue(np.array_equal(np.random.get_state()[1], before_numpy[1]))
            self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
            run.log_metrics({"loss": 0.25}, 1)
            run.summary(best_map=0.8)
            run.artifact("val_split", self.val)
            self.assertTrue(run.task.id.startswith("offline-"))
        metadata = json.loads((self.root / "run/tracking/run.json").read_text())
        self.assertEqual(metadata["status"], "completed")
        self.assertIn("clearml_offline_folder", metadata)


if __name__ == "__main__":
    unittest.main()
