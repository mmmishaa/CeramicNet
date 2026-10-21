import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .adapters import xyxy_to_coco
from .data import prepare_split, read_json, write_json
from .evaluate import (annotations_digest, coco_metrics, operating_metrics,
                       select_threshold, validate_predictions)


def fixture_data():
    return {"images": [{"id": 1, "width": 64, "height": 32, "file_name": "a_1.png"},
                       {"id": 2, "width": 64, "height": 32, "file_name": "b_1.png"}],
            "categories": [{"id": 0, "name": "stamp"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 5, 20, 10],
                             "area": 200, "iscrowd": 0}]}


def artifact(data, detections):
    return {"annotations_sha256": annotations_digest(data),
            "records": [{"image_id": 1, "detections": detections}, {"image_id": 2, "detections": []}]}


class ProtocolTests(unittest.TestCase):
    def test_group_split_reproducible_and_no_object_or_hash_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            records = []
            names = ["objectA_1", "objectA_2", "objectB_1", "objectC_1", "objectD_1", "objectE_1"]
            for i, name in enumerate(names, 1):
                Image.new("RGB", (64, 32), (i * 30, 20, 30)).save(images / f"{name}.png")
                records.append({"id": i, "file_name": f"{name}.png", "width": 64, "height": 32})
            # Identical file with another object ID must join the original group.
            (images / "objectC_1.png").write_bytes((images / "objectB_1.png").read_bytes())
            data = {"images": records, "categories": [{"id": 1, "name": "stamp"}], "annotations": []}
            write_json(root / "input.json", data)
            a = prepare_split(root / "input.json", images, root / "split1")
            b = prepare_split(root / "input.json", images, root / "split2")
            self.assertEqual(a["fingerprint"], b["fingerprint"])
            self.assertEqual(a["image_ids"], b["image_ids"])
            membership = {image_id: split for split, ids in a["image_ids"].items() for image_id in ids}
            self.assertEqual(membership[1], membership[2])
            self.assertEqual(membership[3], membership[4])
            train, val = read_json(root / "split1/train.json"), read_json(root / "split1/val.json")
            self.assertFalse({im["group_id"] for im in train["images"]} & {im["group_id"] for im in val["images"]})
            self.assertEqual(sum(a["stats"][s]["negative_images"] for s in ("train", "val")), 6)
            with self.assertRaises(FileExistsError):
                prepare_split(root / "input.json", images, root / "split1")

    def test_duplicate_detections_match_ground_truth_once(self):
        data = fixture_data()
        d = {"bbox": [10, 5, 20, 10], "category_id": 0, "score": 0.99}
        result = operating_metrics(data, artifact(data, [d, d]))
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))

    def test_high_confidence_wrong_box_is_false_positive(self):
        data = fixture_data()
        wrong = {"bbox": [40, 20, 5, 5], "category_id": 0, "score": 0.999}
        result = operating_metrics(data, artifact(data, [wrong]))
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_negatives_and_empty_outputs(self):
        data = fixture_data()
        outputs = artifact(data, [])
        outputs["records"][1]["detections"] = [{"bbox": [1, 1, 5, 5], "category_id": 0, "score": 0.9}]
        result = operating_metrics(data, outputs)
        self.assertEqual(result["negative_image_fpr"], 1)
        self.assertEqual(result["recall"], 0)
        self.assertIsNone(select_threshold(data, artifact(data, []))["selected"])

    def test_missing_image_or_wrong_annotations_rejected(self):
        data = fixture_data()
        output = artifact(data, [])
        validate_predictions(data, output)
        output["records"].pop()
        with self.assertRaises(ValueError):
            validate_predictions(data, output)
        output = artifact(data, [])
        output["annotations_sha256"] = "other"
        with self.assertRaises(ValueError):
            validate_predictions(data, output)

    def test_label_mapping_and_original_coordinates(self):
        boxes = xyxy_to_coco([[10, 5, 30, 15]], [1], [0.9], {1: 0}, (64, 32))
        self.assertEqual(boxes[0]["bbox"], [10, 5, 20, 10])
        self.assertEqual(boxes[0]["category_id"], 0)
        with self.assertRaises(ValueError):
            xyxy_to_coco([[10, 5, 30, 15]], [8], [0.9], {1: 0}, (64, 32))

    @unittest.skipUnless(importlib.util.find_spec("pycocotools"), "pycocotools not installed")
    def test_perfect_coco_predictions_and_empty_predictions(self):
        data = fixture_data()
        perfect = {"bbox": [10, 5, 20, 10], "category_id": 0, "score": 0.9}
        self.assertAlmostEqual(coco_metrics(data, artifact(data, [perfect]))["map_50_95"], 1)
        self.assertEqual(coco_metrics(data, artifact(data, []))["map_50_95"], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_actual_dataset_negative_shape_and_joint_bbox_transform(self):
        import torch
        from dataset.transforms import Compose, HorizontalFlip
        from .datasets import make_datasets
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_data()
            for im in data["images"]:
                Image.new("RGB", (64, 32), (50, 60, 70)).save(root / im["file_name"])
            for split in ("train", "val"):
                write_json(root / f"{split}.json", data)
            transforms = Compose([HorizontalFlip(p=1)])
            train, val = make_datasets(root, root, train_transforms=transforms)
            _, target = train[0]
            self.assertTrue(torch.equal(target["boxes"], torch.tensor([[34., 5., 54., 15.]])))
            self.assertEqual(target["labels"].tolist(), [1])
            image_a, target = val[0]
            image_b, _ = val[0]
            self.assertTrue(torch.equal(image_a, image_b))
            self.assertEqual(val[1][1]["boxes"].shape, (0, 4))
            self.assertEqual(target["boxes"].tolist(), [[10., 5., 30., 15.]])


if __name__ == "__main__":
    unittest.main()
