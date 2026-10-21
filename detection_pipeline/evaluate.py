"""Evaluate any detector's predictions in original-image COCO coordinates."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .data import read_json, write_json, validate_coco


def annotations_digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_predictions(path, ann_file, model_name, predictions_by_image, config=None):
    """Every evaluated image must have an entry, including [] for no detections.

    Each detection: category_id (canonical COCO ID), bbox=[x,y,w,h] in original
    pixels, score. Keep scores >=0.001; do not apply the deployment threshold.
    """
    data = read_json(ann_file)
    artifact = {"model": model_name, "annotations_sha256": annotations_digest(data),
                "config": config or {},
                "records": [{"image_id": image_id, "detections": boxes}
                            for image_id, boxes in sorted(predictions_by_image.items())]}
    validate_predictions(data, artifact)
    write_json(path, artifact)


def validate_predictions(data, artifact):
    if artifact["annotations_sha256"] != annotations_digest(data):
        raise ValueError("Predictions were generated for a different annotation file")
    ids = [r["image_id"] for r in artifact["records"]]
    if len(ids) != len(set(ids)) or set(ids) != {im["id"] for im in data["images"]}:
        raise ValueError("Predictions must cover every image exactly once, including empty outputs")
    categories = {c["id"] for c in data["categories"]}
    images = {im["id"]: im for im in data["images"]}
    for record in artifact["records"]:
        im = images[record["image_id"]]
        for d in record["detections"]:
            x, y, w, h = d["bbox"]
            if not all(math.isfinite(v) for v in (x, y, w, h, d["score"])) or not (
                0 <= d["score"] <= 1 and w > 0 and h > 0 and x >= 0 and y >= 0
                and x + w <= im["width"] + 1e-3 and y + h <= im["height"] + 1e-3
                and d["category_id"] in categories
            ):
                raise ValueError(f"Invalid prediction for image {record['image_id']}: {d}")


def box_iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    intersection = max(0, right - x) * max(0, bottom - y)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def operating_metrics(data, artifact, threshold=0.5, iou_threshold=0.5, category_id=None):
    """One-to-one, class-aware matching, all images including negatives.

    category_id=None gives micro metrics over all classes; pass the stamp ID
    when the dataset also contains ornaments or other classes.
    """
    truth = defaultdict(list)
    for a in data["annotations"]:
        if category_id is None or a["category_id"] == category_id:
            truth[a["image_id"]].append(a)
    tp = fp = 0
    total_gt = sum(map(len, truth.values()))
    negative_count = sum(not truth[im["id"]] for im in data["images"])
    false_positive_negatives = 0
    for record in artifact["records"]:
        gt, used = truth[record["image_id"]], set()
        detections = [d for d in record["detections"] if d["score"] >= threshold
                      and (category_id is None or d["category_id"] == category_id)]
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)[:100]
        if not gt and detections:
            false_positive_negatives += 1
        for d in detections:
            candidates = [(box_iou(d["bbox"], a["bbox"]), i) for i, a in enumerate(gt)
                          if i not in used and a["category_id"] == d["category_id"]]
            iou, match = max(candidates, default=(0, -1))
            if match >= 0 and iou >= iou_threshold:
                tp += 1
                used.add(match)
            else:
                fp += 1
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / total_gt if total_gt else None
    return {"threshold": threshold, "iou_threshold": iou_threshold, "category_id": category_id,
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "negative_images": negative_count,
            "negative_image_fpr": false_positive_negatives / negative_count if negative_count else None}


def select_threshold(data, artifact, min_precision=0.9, category_id=None):
    rows = [operating_metrics(data, artifact, threshold=i / 100, category_id=category_id)
            for i in range(1, 100)]
    feasible = [r for r in rows if r["precision"] is not None and r["precision"] >= min_precision
                and r["recall"] is not None and r["tp"] > 0]
    selected = max(feasible, key=lambda r: (r["recall"], r["precision"], -r["threshold"])) if feasible else None
    return {"min_precision": min_precision, "selected": selected, "curve": rows}


def coco_metrics(data, artifact):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO()
    gt.dataset = dict(data, info=data.get("info", {}))
    gt.createIndex()
    detections = [dict(d, image_id=r["image_id"]) for r in artifact["records"] for d in r["detections"]]
    if detections:
        dt = gt.loadRes(detections)
    else:
        # loadRes([]) fails in some pycocotools versions.
        dt = COCO()
        dt.dataset = dict(gt.dataset, annotations=[])
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = sorted(im["id"] for im in data["images"])
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["map_50_95", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
             "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"]
    return {name: float(value) if value >= 0 else None for name, value in zip(names, evaluator.stats)}


def evaluate(ann_file, predictions_file, output=None, min_precision=0.9, threshold=0.5, tune_threshold=False):
    data, artifact = read_json(ann_file), read_json(predictions_file)
    validate_coco(data)
    validate_predictions(data, artifact)
    result = {"model": artifact["model"], "annotations_sha256": artifact["annotations_sha256"],
              "config": artifact["config"], "coco": coco_metrics(data, artifact),
              "operating_point": operating_metrics(data, artifact, threshold)}
    if tune_threshold:
        result["threshold_search_on_validation"] = select_threshold(data, artifact, min_precision)
    if output:
        write_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true", help="Use ONLY on validation, never on test")
    args = parser.parse_args()
    result = evaluate(args.annotations, args.predictions, args.output, threshold=args.threshold,
                      tune_threshold=args.tune_threshold)
    print(json.dumps(result["coco"], indent=2))
