"""Compute standard COCO AP from a saved detection_test JSON (low conf recommended)."""
import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(report):
    images = report["results"][0]["images"]
    annotations, predictions = [], []
    for image_id, item in enumerate(images, 1):
        for box in item["gt_boxes"]:
            x1, y1, x2, y2 = box
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        for box, score in zip(item["pred_boxes"], item["pred_scores"]):
            x1, y1, x2, y2 = box
            predictions.append({"image_id": image_id, "category_id": 1,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {}, "images": [{"id": i, "file_name": item["file_name"]}
                                 for i, item in enumerate(images, 1)],
        "categories": [{"id": 1, "name": "kleimo"}], "annotations": annotations,
    }
    ground_truth.createIndex()
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {**ground_truth.dataset, "annotations": []}
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ["map50_95", "map50", "ap75", "ap_small", "ap_medium", "ap_large",
            "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"]
    return {key: float(value) for key, value in zip(keys, evaluator.stats)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    metrics = evaluate(report)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"evaluator": "pycocotools COCOeval bbox", "max_detections": [1, 10, 100],
                   "confidence_cutoff": report["results"][0]["conf_threshold"],
                   "metrics": metrics}, stream, indent=2)
        stream.write("\n")
