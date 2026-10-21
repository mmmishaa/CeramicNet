"""Regression tests for the real class from the datasets branch."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dataset import StampDataset, Rotate90, boxes_to_original
from dataset.transforms.factory import build_augmentations
from .data import write_json
from .datasets import CocoStampDataset, make_datasets
from .experiments import resolve_augmentation
from .integrations.joint import augment_mmdet_results
from .test_pipeline import fixture_data


class TeamIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = fixture_data()
        for im in self.data["images"]:
            Image.new("RGB", (64, 32), (70, 80, 90)).save(self.root / im["file_name"])
        self.ann = self.root / "annotations.json"
        write_json(self.ann, self.data)

    def dataset(self, model, ann_file=None, **kwargs):
        return StampDataset(self.root, ann_file=ann_file or self.ann, image_root=self.root, model=model, **kwargs)

    def test_zero_based_coco_labels_for_every_model(self):
        for model, label in [("default", 0), ("faster_rcnn", 1), ("rtdetr", 0), ("rtmdet", 0)]:
            ds = self.dataset(model)
            image, target = ds[0]
            self.assertEqual(target["labels"].tolist(), [label])
            self.assertEqual(ds.label_to_category[label], 0)
            self.assertEqual(ds[1][1]["boxes"].shape, (0, 4))
            if model == "rtdetr":
                torch.testing.assert_close(target["boxes"], torch.tensor([[20/64, 10/32, 20/64, 10/32]]))

    def test_noncontiguous_category_ids_and_original_default_labels(self):
        data = deepcopy(self.data)
        data["categories"] = [{"id": 7, "name": "a"}, {"id": 42, "name": "b"}]
        data["annotations"][0]["category_id"] = 42
        write_json(self.ann, data)
        self.assertEqual(self.dataset("default")[0][1]["labels"].tolist(), [42])
        self.assertEqual(self.dataset("faster_rcnn")[0][1]["labels"].tolist(), [2])
        self.assertEqual(self.dataset("rtmdet")[0][1]["labels"].tolist(), [1])

    def test_original_directory_api_and_letterbox_recovery(self):
        old_images = self.root / "images/default"
        old_images.mkdir(parents=True)
        for im in self.data["images"]:
            (old_images / im["file_name"]).write_bytes((self.root / im["file_name"]).read_bytes())
        (self.root / "annotations").mkdir()
        write_json(self.root / "annotations/instances_default.json", self.data)
        ds = StampDataset(self.root, model="default", standard_size=(80, 80))
        image, target = ds[0]
        self.assertEqual(image.shape, (3, 80, 80))
        torch.testing.assert_close(boxes_to_original(target["boxes"], target), torch.tensor([[10., 5., 30., 15.]]))

    def test_pipeline_reuses_team_dataset_and_keeps_val_unaugmented(self):
        for split in ("train", "val"):
            write_json(self.root / f"{split}.json", self.data)
        transforms = build_augmentations(operations=[{"type": "Rotate90", "p": 1, "angles": [90]}])
        train, val = make_datasets(self.root, self.root, train_transforms=transforms)
        self.assertIsInstance(train, StampDataset)
        self.assertIsInstance(val, CocoStampDataset)
        self.assertEqual(train[0][0].shape, (3, 64, 32))
        self.assertEqual(val[0][0].shape, (3, 32, 64))
        torch.testing.assert_close(train[0][1]["boxes"], torch.tensor([[5., 34., 15., 54.]]))
        torch.testing.assert_close(val[0][1]["boxes"], torch.tensor([[10., 5., 30., 15.]]))

    def test_mmdet_bridge_preserves_bgr_order_and_rotates_boxes(self):
        rgb = np.zeros((32, 64, 3), dtype=np.uint8)
        rgb[:] = (90, 60, 30)
        results = {"img": rgb[..., ::-1].copy(), "img_shape": (32, 64),
                   "gt_bboxes": np.array([[10., 5., 30., 15.]], dtype=np.float32),
                   "gt_bboxes_labels": np.array([0]), "gt_ignore_flags": np.array([False])}
        output = augment_mmdet_results(results, [{"type": "Rotate90", "p": 1, "angles": [90]},
                                               {"type": "Brightness", "p": 1, "factor": [2, 2]}])
        self.assertEqual(output["img_shape"], (64, 32))
        self.assertEqual(output["img"][0, 0].tolist(), [60, 120, 180])
        np.testing.assert_allclose(output["gt_bboxes"], [[5, 34, 15, 54]])
        self.assertEqual(output["gt_bboxes_labels"].tolist(), [0])

    def test_per_model_choices_and_invalid_transforms(self):
        settings = {"models": {"rtmdet_tiny": {"augmentation": "rotation"},
                                "rtdetrv2_s": {"augmentation": "none"}}}
        path = self.root / "experiments.json"
        write_json(path, settings)
        self.assertEqual(resolve_augmentation("rtmdet_tiny", experiments=path)[1][0]["type"], "Rotate90")
        self.assertEqual(resolve_augmentation("rtdetrv2_s", experiments=path)[1], [])
        self.assertEqual(resolve_augmentation("rtmdet_tiny", profile="none", experiments=path)[1], [])
        with self.assertRaises(ValueError):
            build_augmentations(operations=[{"type": "Unknown"}])
        with self.assertRaises(ValueError):
            build_augmentations(operations=[{"type": "Brightness", "p": 2}])


if __name__ == "__main__":
    unittest.main()
