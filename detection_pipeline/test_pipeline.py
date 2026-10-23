import importlib.util
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from .adapters import xyxy_to_coco
from .data import add_negative_split, prepare_split, read_json, write_json
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
    def test_25_per_source_then_random_split_without_source_or_group_quotas(self):
        from collections import Counter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_data()
            data["annotations"].append(dict(data["annotations"][0], id=2, image_id=2))
            for im in data["images"]:
                Image.new("RGB", (64, 32), (im["id"], 0, 0)).save(root / im["file_name"])
            write_json(root / "input.json", data)
            prepare_split(root / "input.json", root, root / "base")
            folders = [root / "empty", root / "ornaments_without_copies"]
            for n, folder in enumerate(folders):
                folder.mkdir()
                for i in range(30):
                    # All negatives share a positive's object ID. The requested
                    # image split must succeed and report the group overlap.
                    Image.new("RGB", (64, 32), (i, 100 + n, 0)).save(folder / f"a_{i + 10:02}.png")
            a = add_negative_split(root / "base", root, folders, root / "new", per_dir_count=25)
            b = add_negative_split(root / "base", root, folders[::-1], root / "repeat", per_dir_count=25)
            self.assertEqual(a, b)
            chosen = a["negatives"]["selected"]
            self.assertEqual({s: len(ims) for s, ims in chosen.items()}, {"train": 40, "val": 10})
            self.assertEqual(Counter(Path(im["file_name"]).parts[0]
                                     for ims in chosen.values() for im in ims),
                             {"empty": 25, "ornaments_without_copies": 25})
            # The combined sample is split without stratifying by directory.
            self.assertEqual(sum(im["file_name"].startswith("empty/") for im in chosen["val"]), 3)
            self.assertEqual(a["negatives"]["split_by"], "image")
            self.assertEqual(a["negatives"]["group_overlap"], ["a"])
            self.assertFalse({im["sha256"] for im in chosen["train"]} &
                             {im["sha256"] for im in chosen["val"]})
            for s in ("train", "val"):
                self.assertEqual(read_json(root / "new" / f"{s}.json")["annotations"],
                                 read_json(root / "base" / f"{s}.json")["annotations"])

    def test_fixed_50_negatives_reproducible_and_positive_split_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_data()
            data["annotations"].append(dict(data["annotations"][0], id=2, image_id=2))
            for im in data["images"]:
                Image.new("RGB", (64, 32), (im["id"], 0, 0)).save(root / im["file_name"])
            write_json(root / "input.json", data)
            prepare_split(root / "input.json", root, root / "base")
            negatives = root / "empty"
            ornaments = root / "ornaments_without_copies"
            negatives.mkdir()
            ornaments.mkdir()
            # Photos of one object in different source folders must stay together.
            for i in range(80):
                folder = negatives if i % 2 == 0 else ornaments
                Image.new("RGB", (64, 32), (i, 100, 0)).save(folder / f"negative{i // 2:02}_{i % 2 + 1}.png")
            # Same object as a positive must stay in that positive's split.
            for i, name in enumerate(("a_2.png", "b_2.png"), 1):
                Image.new("RGB", (64, 32), (i, 150, 0)).save(negatives / name)
            (negatives / "copy_1.png").write_bytes((root / "a_1.png").read_bytes())
            (negatives / "copy_2.png").write_bytes((negatives / "negative00_1.png").read_bytes())
            a = add_negative_split(root / "base", root, [negatives, ornaments], root / "new")
            b = add_negative_split(root / "base", root, [ornaments, negatives], root / "repeat")
            self.assertEqual(a, b)
            self.assertEqual({Path(im["file_name"]).parts[0]
                              for chosen in a["negatives"]["selected"].values() for im in chosen},
                             {"empty", "ornaments_without_copies"})
            self.assertEqual(a["negatives"]["random_state"], 42)
            self.assertEqual(a["negatives"]["candidate_count"], 82)
            self.assertEqual(a["negatives"]["skipped"]["base_image_or_exact_copy"], 2)
            splits = {}
            for split, count in (("train", 40), ("val", 10)):
                original = read_json(root / "base" / f"{split}.json")
                updated = read_json(root / "new" / f"{split}.json")
                splits[split] = updated
                self.assertEqual(updated["images"][:len(original["images"])], original["images"])
                self.assertEqual(updated["annotations"], original["annotations"])
                self.assertEqual(a["stats"][split]["negative_images"], count)
                self.assertEqual(len(updated["images"]), len(original["images"]) + count)
            for key in ("group_id", "sha256", "id"):
                self.assertFalse({im[key] for im in splits["train"]["images"]} &
                                 {im[key] for im in splits["val"]["images"]})
            c = add_negative_split(root / "base", root, [negatives, ornaments], root / "other", seed=43)
            self.assertNotEqual(a["fingerprint"], c["fingerprint"])
            with self.assertRaises(FileExistsError):
                add_negative_split(root / "base", root, [negatives, ornaments], root / "new")

    def test_insufficient_negatives_do_not_write_partial_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture_data()
            data["annotations"].append(dict(data["annotations"][0], id=2, image_id=2))
            for im in data["images"]:
                Image.new("RGB", (64, 32), (im["id"], 0, 0)).save(root / im["file_name"])
            write_json(root / "input.json", data)
            prepare_split(root / "input.json", root, root / "base")
            (root / "empty").mkdir()
            with self.assertRaisesRegex(ValueError, "Not enough eligible negatives"):
                add_negative_split(root / "base", root, [root / "empty"], root / "new")
            self.assertFalse((root / "new").exists())

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
