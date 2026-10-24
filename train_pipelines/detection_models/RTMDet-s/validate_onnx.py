"""Validate an ONNX export on every canonical val image against native predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from detection_pipeline.adapters import xyxy_to_coco
from detection_pipeline.data import read_json, write_json
from detection_pipeline.evaluate import save_predictions, evaluate, annotations_digest
from detection_pipeline.rtmdet_onnx_processing import RTMDetProcessing


def validate(model, annotations, images, reference, output):
    data, native = read_json(annotations), read_json(reference)
    if native["annotations_sha256"] != annotations_digest(data):
        raise ValueError("Reference predictions belong to another split")
    references = {r["image_id"]: r["detections"] for r in native["records"]}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    predictions, box_errors, score_errors, count_mismatches = {}, [], [], []
    operating_checks = {str(t): {"count_mismatch_image_ids": [], "max_box_error_pixels": 0.0,
                                "max_score_error": 0.0} for t in (0.5, 0.84)}
    label_map = {0: data["categories"][0]["id"]}
    for i, record in enumerate(data["images"], 1):
        with Image.open(Path(images) / record["file_name"]) as source:
            image = np.asarray(source.convert("RGB"))[:, :, ::-1].copy()
        blob, meta = RTMDetProcessing.preprocess(image)
        raw = session.run(None, {"images": blob})
        boxes, scores = RTMDetProcessing.postprocess(raw, meta, 0.001, 0.65)
        detections = xyxy_to_coco(boxes, np.zeros(len(scores)), scores, label_map,
                                  (record["width"], record["height"]))
        predictions[record["id"]] = detections
        expected = references[record["id"]]
        for threshold, check in operating_checks.items():
            left = [d for d in expected if d["score"] >= float(threshold)]
            right = [d for d in detections if d["score"] >= float(threshold)]
            if len(left) != len(right):
                check["count_mismatch_image_ids"].append(record["id"])
            elif left:
                a = np.array([d["bbox"] for d in left])
                b = np.array([d["bbox"] for d in right])
                cost = np.max(np.abs(a[:, None] - b[None, :]), axis=2)
                ii, jj = linear_sum_assignment(cost)
                check["max_box_error_pixels"] = max(check["max_box_error_pixels"], float(cost[ii, jj].max()))
                check["max_score_error"] = max(check["max_score_error"],
                    max(abs(left[i]["score"] - right[j]["score"]) for i, j in zip(ii, jj)))
        if len(detections) != len(expected):
            count_mismatches.append(record["id"])
        elif detections:
            box_errors.extend(np.abs(np.array([d["bbox"] for d in detections]) -
                                     np.array([d["bbox"] for d in expected])).ravel().tolist())
            score_errors.extend(abs(a["score"] - b["score"]) for a, b in zip(detections, expected))
        if i % 25 == 0 or i == len(data["images"]):
            print(f"ONNX val: {i}/{len(data['images'])}", flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(output / "predictions.json", annotations, "rtmdet_s_onnx", predictions,
                     {"onnx": str(model), "native_reference": str(reference), "nms_iou": 0.65})
    metrics = evaluate(annotations, output / "predictions.json", output / "metrics.json", threshold=0.5)
    native_metrics = evaluate(annotations, reference)
    delta = {k: metrics["coco"][k] - v for k, v in native_metrics["coco"].items()
             if v is not None and metrics["coco"][k] is not None}
    report = {"images": len(data["images"]), "count_mismatch_image_ids": count_mismatches,
              "all_scores_rankwise_max_box_error_pixels": max(box_errors, default=0),
              "all_scores_rankwise_max_score_error": max(score_errors, default=0),
              "note": "Very low confidence candidates may reorder or switch NMS survivor under floating point differences; compare operating detections by spatial assignment.",
              "operating_checks": operating_checks, "coco_delta_onnx_minus_native": delta,
              "metrics": metrics}
    write_json(output / "parity.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "metrics"}, indent=2))
    if (count_mismatches or abs(delta["map_50_95"]) > 1e-4 or abs(delta["ap50"]) > 1e-4
        or any(c["count_mismatch_image_ids"] or c["max_box_error_pixels"] > 0.1
               or c["max_score_error"] > 0.001 for c in operating_checks.values())):
        raise AssertionError("ONNX/native parity check failed; inspect parity.json")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "annotations", "images", "reference", "output"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    validate(a.model, a.annotations, a.images, a.reference, a.output)
