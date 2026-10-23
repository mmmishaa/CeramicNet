"""Evaluate RTMDet-s weights and save a gallery, using the shared COCO evaluator.

Run from the repository root. This native MMDetection runner accepts .pth
checkpoints; it does not require an ONNX export or change the YOLO benchmark.
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

from detection_pipeline.adapters import mmdet_prediction
from detection_pipeline.data import read_json, validate_coco, write_json
from detection_pipeline.evaluate import evaluate, save_predictions


def evaluation_data(splits, scope):
    """Val measures held-out photos; all stamps explicitly includes train."""
    train, val = [read_json(Path(splits) / f"{part}.json") for part in ("train", "val")]
    train_ids = {im["id"] for im in train["images"]}
    val_ids = {im["id"] for im in val["images"]}
    if train_ids & val_ids:
        raise ValueError("The saved train/val split has overlapping image IDs")
    if train["categories"] != val["categories"]:
        raise ValueError("Train/val categories disagree")
    if scope == "val":
        data = deepcopy(val)
    elif scope == "stamps":
        images = [im for part in (train, val) for im in part["images"]
                  if Path(im["file_name"]).parts[0] == "stamps"]
        chosen = {im["id"] for im in images}
        data = dict(deepcopy(train), images=sorted(images, key=lambda im: im["id"]),
                    annotations=[ann for part in (train, val) for ann in part["annotations"]
                                 if ann["image_id"] in chosen])
    else:
        raise ValueError(f"Unknown scope: {scope}")
    validate_coco(data)
    selected_ids = {im["id"] for im in data["images"]}
    positive_ids = {ann["image_id"] for ann in data["annotations"]}
    description = {
        "scope": scope, "images": len(selected_ids),
        "training_images": len(selected_ids & train_ids),
        "validation_images": len(selected_ids & val_ids),
        "negative_images": len(selected_ids - positive_ids),
        "independent_test": False,
        "purpose": "validation" if scope == "val" else "visual inspection; includes training photos",
    }
    return data, description


def save_gallery(data, predictions, images_root, output, threshold, description):
    image_dir = output / "images"
    image_dir.mkdir()
    truth = {}
    for ann in data["annotations"]:
        truth.setdefault(ann["image_id"], []).append(ann["bbox"])
    cards = []
    for record in data["images"]:
        with Image.open(images_root / record["file_name"]) as source:
            image = source.convert("RGB")
        original_width = image.width
        image.thumbnail((1100, 1100))
        scale = image.width / original_width
        draw = ImageDraw.Draw(image)
        for x, y, w, h in truth.get(record["id"], []):
            draw.rectangle([x*scale, y*scale, (x+w)*scale, (y+h)*scale], outline="#16b93c", width=3)
        kept = [d for d in predictions[record["id"]] if d["score"] >= threshold]
        for detection in kept:
            x, y, w, h = detection["bbox"]
            draw.rectangle([x*scale, y*scale, (x+w)*scale, (y+h)*scale], outline="#e13434", width=3)
            draw.text((x*scale, max(0, y*scale-13)), f"{detection['score']:.2f}", fill="#e13434")
        name = f"{record['id']:06d}.jpg"
        image.save(image_dir / name, quality=88)
        cards.append(f'<article><a href="images/{name}"><img loading="lazy" src="images/{name}" alt=""></a>'
                     f'<p>{html.escape(record["file_name"])}</p><small>Рамок модели: {len(kept)}</small></article>')
    note = ("Val: изображения не использовались при обучении."
            if description["scope"] == "val" else
            "Все stamps: включает 203 тренировочных кадра. Метрики не являются независимой оценкой качества.")
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>RTMDet-s</title>'
            '<style>body{font:16px system-ui;margin:30px;background:#f5f5f3;color:#202522}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}'
            'article{background:white;padding:12px;border-radius:12px}img{width:100%;height:330px;object-fit:contain}'
            'p{overflow-wrap:anywhere}small{color:#555}</style>'
            f'<h1>RTMDet-s · {description["images"]} фото</h1><p>{note}</p>'
            f'<p>Зелёные рамки — разметка; красные — модель. Порог: {threshold:.3f}.</p>'
            '<main>' + "".join(cards) + '</main></html>')
    (output / "gallery.html").write_text(page, encoding="utf-8")


def run(config, checkpoint, images_root, splits, repo, output, scope="val",
        validation_metrics=None, threshold=0.5, device="cpu", threads=4):
    config, checkpoint, images_root, splits, repo, output = [Path(p).resolve() for p in
                                                           (config, checkpoint, images_root, splits, repo, output)]
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if not config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Supply an existing config and trained checkpoint")
    if validation_metrics:
        previous = read_json(validation_metrics)
        selected = previous.get("threshold_search_on_validation", {}).get("selected")
        if selected:
            threshold = selected["threshold"]
        if Path(previous["config"]["checkpoint"]).resolve() != checkpoint:
            raise ValueError("Validation threshold belongs to a different checkpoint")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    torch.set_num_threads(threads)
    sys.path.insert(0, str(repo))
    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector

    data, description = evaluation_data(splits, scope)
    for record in data["images"]:
        path = images_root / record["file_name"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Missing or modified photo: {path}")
    cfg = Config.fromfile(str(config))
    cfg.model.backbone.init_cfg = None
    cfg.model.test_cfg.score_thr = 0.001
    cfg.model.test_cfg.max_per_img = 100
    model = init_detector(cfg, str(checkpoint), device=device)
    predictions, latencies = {}, []

    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    with torch.inference_mode():
        # Exclude model loading and one warm-up call from latency.
        inference_detector(model, str(images_root / data["images"][0]["file_name"]))
        for index, record in enumerate(data["images"], 1):
            with Image.open(images_root / record["file_name"]) as source:
                bgr = np.asarray(source.convert("RGB"))[:, :, ::-1].copy()
            synchronize()
            start = time.perf_counter()
            sample = inference_detector(model, bgr)
            synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            predictions[record["id"]] = mmdet_prediction(sample, data["categories"],
                                                        (record["width"], record["height"]))
            if index % 25 == 0 or index == len(data["images"]):
                print(f"{scope}: {index}/{len(data['images'])}", flush=True)
    output.mkdir(parents=True)
    annotations = output / "annotations.json"
    write_json(annotations, data)
    metadata = read_json(config.with_suffix(".metadata.json")) if config.with_suffix(".metadata.json").is_file() else {}
    metadata.update(description, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    config_path=str(config), device=device, threshold=threshold,
                    threshold_source=str(validation_metrics) if validation_metrics else "fixed parameter",
                    latency_scope="preprocess + native inference + NMS; excludes image decoding, warm-up and model loading")
    save_predictions(output / "predictions.json", annotations, "rtmdet_s", predictions, metadata)
    metrics = evaluate(annotations, output / "predictions.json", output / "metrics.json", threshold=threshold)
    timing = {"mean_ms": float(np.mean(latencies)), "median_ms": float(np.median(latencies)),
              "p95_ms": float(np.percentile(latencies, 95)), "fps": float(1000 / np.mean(latencies))}
    write_json(output / "timing.json", dict(timing, device=device, scope=metadata["latency_scope"]))
    save_gallery(data, predictions, images_root, output, threshold, description)
    row = {"Model": "rtmdet_s", "Dataset": scope, "Timestamp_UTC": datetime.now(timezone.utc).isoformat(),
           "Images": description["images"], "Training_Images": description["training_images"],
           "Validation_Images": description["validation_images"], "Negative_Images": description["negative_images"],
           "Annotations_SHA256": metrics["annotations_sha256"], "Threshold": threshold,
           "mAP50": metrics["coco"]["ap50"], "mAP50-95": metrics["coco"]["map_50_95"],
           "Precision": metrics["operating_point"]["precision"], "Recall": metrics["operating_point"]["recall"],
           "F1": metrics["operating_point"]["f1"], "Negative_Image_FPR": metrics["operating_point"]["negative_image_fpr"],
           "Latency_ms": timing["mean_ms"], "FPS": timing["fps"], "Device": device,
           "Checkpoint_SHA256": metadata["checkpoint_sha256"]}
    with (output / "benchmark_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    write_json(output / "run.json", {**description, **metadata, "status": "completed"})
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print("Gallery:", output / "gallery.html")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", dest="images_root", required=True)
    parser.add_argument("--repo", default=str(PROJECT / ".local/rtmdet/mmdetection"))
    parser.add_argument("--splits", default=str(PROJECT / "detection_pipeline/splits/v2_seed42_neg50"))
    parser.add_argument("--scope", choices=["val", "stamps"], default="val")
    parser.add_argument("--validation-metrics")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))
