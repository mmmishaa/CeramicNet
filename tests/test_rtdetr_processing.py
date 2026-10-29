import tempfile
import unittest
import argparse
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.detection_test import load_processor
from scripts.detection_test import main as benchmark_main
from src.detection.detector import build_detector
from src.detection.rtdetr_processing import RTDetrProcessing


class RTDetrProcessingTests(unittest.TestCase):
    def test_common_processor_interface(self):
        processor = load_processor(
            "src.detection.rtdetr_processing.RTDetrProcessing",
            imgsz=640, conf_threshold=0.7, iou_threshold=0.5,
        )
        self.assertEqual(processor.imgsz, (640, 640))
        self.assertEqual(processor.conf_threshold, 0.7)

    def test_white_letterbox_rgb_and_normalization(self):
        image = np.full((160, 320, 3), [0, 64, 255], dtype=np.uint8)
        blob, metadata = RTDetrProcessing().preprocess(image)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)
        self.assertTrue(blob.flags.c_contiguous)
        np.testing.assert_allclose(blob[0, :, 320, 320], [1, 64 / 255, 0])
        np.testing.assert_array_equal(blob[0, :, 0, 0], [1, 1, 1])
        self.assertEqual(metadata["pad_top"], 160)

    def test_undo_letterbox_and_sigmoid(self):
        processor = RTDetrProcessing(conf_threshold=0.4)
        _, metadata = processor.preprocess(np.zeros((160, 320, 3), np.uint8))
        logits = np.array([[[0.0]]], np.float32)
        boxes = np.array([[[0.5, 0.5, 0.5, 0.25]]], np.float32)
        original = boxes.copy()
        actual, scores = processor.postprocess([logits, boxes], metadata)
        np.testing.assert_allclose(actual, [[80, 40, 240, 120]])
        np.testing.assert_allclose(scores, [0.5])
        np.testing.assert_array_equal(boxes, original)
        reverse, _ = processor.postprocess([boxes, logits], metadata)
        np.testing.assert_array_equal(reverse, actual)

    def test_nms_and_configured_confidence(self):
        processor = RTDetrProcessing(conf_threshold=0.7, iou_threshold=0.5)
        _, metadata = processor.preprocess(np.zeros((640, 640, 3), np.uint8))
        boxes = np.array([[[.5, .5, .5, .5], [.5, .5, .5, .5], [.1, .1, .1, .1]]], np.float32)
        actual, scores = processor.postprocess(
            [np.array([[[4.], [3.], [0.]]], np.float32), boxes], metadata
        )
        self.assertEqual(len(actual), 1)
        np.testing.assert_allclose(scores, [1 / (1 + np.exp(-4))])

    def test_empty_and_padding_only_predictions(self):
        processor = RTDetrProcessing(conf_threshold=0.7)
        _, metadata = processor.preprocess(np.zeros((160, 320, 3), np.uint8))
        for logit, box in [(-10., [.5, .5, .5, .25]), (10., [.5, .05, .1, .05])]:
            boxes, scores = processor.postprocess(
                [np.array([[[logit]]], np.float32), np.array([[box]], np.float32)], metadata
            )
            self.assertEqual(boxes.shape, (0, 4))
            self.assertEqual(scores.shape, (0,))
            self.assertEqual(boxes.dtype, np.float32)

    def test_rejects_invalid_outputs(self):
        processor = RTDetrProcessing()
        _, metadata = processor.preprocess(np.zeros((640, 640, 3), np.uint8))
        for outputs in [
            [np.zeros((1, 300, 1))],
            [np.zeros((2, 300, 1)), np.zeros((2, 300, 4))],
            [np.zeros((1, 300, 80)), np.zeros((1, 300, 4))],
            [np.zeros((1, 299, 1)), np.zeros((1, 300, 4))],
            [np.full((1, 300, 1), np.nan), np.zeros((1, 300, 4))],
        ]:
            with self.subTest(shapes=[a.shape for a in outputs]), self.assertRaises(ValueError):
                processor.postprocess(outputs, metadata)

    def test_onnx_detector_integration_uses_both_outputs(self):
        # A real two-output graph catches regressions in the ONNX call and factory.
        logits = np.array([[[4.]]], np.float32)
        boxes = np.array([[[.5, .5, .5, .25]]], np.float32)
        graph = helper.make_graph(
            [helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value))
             for name, value in [("logits", logits), ("pred_boxes", boxes)]],
            "rtdetr_contract",
            [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 640, 640])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1, 1]),
             helper.make_tensor_value_info("pred_boxes", TensorProto.FLOAT, [1, 1, 4])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "contract.onnx"
            onnx.save(model, path)
            config = {
                "detector": {"weights": str(path), "imgsz": 640,
                             "processor": "src.detection.rtdetr_processing.RTDetrProcessing"},
                "confidence_config": {"conf_threshold": .7, "iou_threshold": .5},
            }
            detector = build_detector(config)
            actual, _ = detector.predict(np.zeros((160, 320, 3), np.uint8))
            np.testing.assert_allclose(actual, [[80, 40, 240, 120]])

            # Exercise the benchmark entry point as well as the detector factory.
            import cv2
            import yaml
            cv2.imwrite(str(Path(folder) / "sample_1.png"), np.zeros((160, 320, 3), np.uint8))
            annotations = Path(folder) / "annotations.json"
            annotations.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "sample_1.png", "width": 320, "height": 160}],
                "categories": [{"id": 1, "name": "kleimo"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                 "bbox": [80, 40, 160, 80]}],
            }))
            benchmark_config = Path(folder) / "config.yaml"
            benchmark_config.write_text(yaml.safe_dump({
                "dataset": {"images_dir": folder, "coco_json": str(annotations)},
                "confidence_config": config["confidence_config"],
                "detectors": {"rtdetr_contract_test": config["detector"]},
            }))
            report = Path(folder) / "results.json"
            args = argparse.Namespace(config=str(benchmark_config), model="rtdetr_contract_test",
                                      conf=None, iou=None, output_csv=str(Path(folder) / "results.csv"),
                                      output_json=str(report))
            with patch("scripts.detection_test.parse_args", return_value=args), contextlib.redirect_stdout(io.StringIO()):
                benchmark_main()
                result = json.loads(report.read_text())["results"][0]
                self.assertEqual(result["metrics"]["true_positives"], 1)
                self.assertEqual(result["metrics"]["f1"], 1.)
                # Existing reports must be preserved, including their bytes.
                saved = report.read_bytes()
                with self.assertRaises(FileExistsError):
                    benchmark_main()
                self.assertEqual(report.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
