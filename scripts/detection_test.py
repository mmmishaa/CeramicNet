import argparse
import csv
from datetime import datetime
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import time
import unicodedata
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import yaml

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from src.paths import ROOT_DIR, REPORTS_DIR, CONFIGS_DIR
from src.detection.dataset import TestDetectionDataset

def load_processor(processor_path: str, **kwargs):
    module_path, class_name = processor_path.rsplit('.', 1)
    
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(f"В модуле '{module_path}' не найден класс '{class_name}'")
    
    processor_cls = getattr(module, class_name)

    return processor_cls(**kwargs)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area

    return inter_area / np.maximum(union, 1e-6)


def compute_ap_101_point(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)

    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(y, x))


def evaluate_detections(records: list[dict], iou_thresh: float = 0.5) -> dict:
    total_gt = 0
    all_detections = []

    for img_idx, item in enumerate(records):
        pred_boxes = item["pred_boxes"]
        pred_scores = item["pred_scores"]
        gt_boxes = item["gt_boxes"]

        total_gt += len(gt_boxes)

        for p_idx in range(len(pred_boxes)):
            all_detections.append(
                {
                    "img_idx": img_idx,
                    "box": pred_boxes[p_idx],
                    "score": float(pred_scores[p_idx]),
                }
            )

    total_pred = len(all_detections)

    if total_gt == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "true_positives": 0,
            "total_pred": total_pred,
            "total_gt": total_gt,
        }

    if total_pred == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "true_positives": 0,
            "total_pred": 0,
            "total_gt": total_gt,
        }

    all_detections.sort(key=lambda x: x["score"], reverse=True)

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    point_tp = 0

    for iou_idx, cur_iou in enumerate(iou_thresholds):
        tp = np.zeros(total_pred)
        fp = np.zeros(total_pred)

        matched_gt = {i: set() for i in range(len(records))}

        for d_idx, det in enumerate(all_detections):
            img_idx = det["img_idx"]
            gt_boxes = records[img_idx]["gt_boxes"]

            if len(gt_boxes) == 0:
                fp[d_idx] = 1.0
                continue

            ious = compute_iou(det["box"][None, :], gt_boxes)[0]
            best_gt_idx = int(np.argmax(ious))
            best_iou = ious[best_gt_idx]

            if (
                best_iou >= cur_iou
                and best_gt_idx not in matched_gt[img_idx]
            ):
                tp[d_idx] = 1.0
                matched_gt[img_idx].add(best_gt_idx)
            else:
                fp[d_idx] = 1.0

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap_101_point(recalls, precisions)
        aps.append(ap)

        if np.isclose(cur_iou, iou_thresh):
            point_tp = int(tp_cumsum[-1])

    map50 = aps[0]
    map50_95 = float(np.mean(aps))

    precision = point_tp / total_pred if total_pred > 0 else 0.0
    recall = point_tp / total_gt if total_gt > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "true_positives": point_tp,
        "total_pred": total_pred,
        "total_gt": total_gt,
    }


def update_results_csv(output_path: Path, new_rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Model",
        "Timestamp",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "F1",
        "Latency_ms",
        "FPS",
        "TP",
        "Total_Pred",
        "Total_GT",
    ]
    existing_data = {}

    if output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "Model" in row:
                    existing_data[row["Model"]] = row

    for row in new_rows:
        existing_data[row["Model"]] = row

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in existing_data.values():
            writer.writerow(row)

    col_widths = {h: len(h) for h in headers}
    for r in existing_data.values():
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(r.get(h, ""))))

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    print("\nСводная таблица результатов:")
    print(header_line)
    print(separator)
    for r in existing_data.values():
        print(" | ".join(str(r.get(h, "")).ljust(col_widths[h]) for h in headers))


def parse_args():
    default_config = CONFIGS_DIR / "detection" / "detection_test_config.yaml"
    default_csv = REPORTS_DIR / "detection_test_results.csv"

    parser = argparse.ArgumentParser(
        description="Бенчмарк моделей детекции из YAML-конфига"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Путь к конфигурационному YAML файлу",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Имя конкретной модели для запуска. Если не передано — запускаются все.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Переопределить conf_threshold из конфига",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="Переопределить iou_threshold из конфига",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(default_csv),
        help="Путь к файлу сохранения результатов",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Новый JSON с метриками, параметрами и предсказаниями; существующий файл не заменяется",
    )
    parser.add_argument(
        "--split_manifest", type=str, default=None,
        help="JSON со списком files и SHA256 изображений для сохранённого тестового набора",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    json_path = ROOT_DIR / args.output_json if args.output_json else None
    if json_path is not None and json_path.exists():
        raise FileExistsError(f"Отчёт уже существует: {json_path}")
    config_path = ROOT_DIR / args.config

    if not config_path.is_file():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    conf_cfg = config["confidence_config"]
    conf_threshold = (
        args.conf
        if args.conf is not None
        else float(conf_cfg["conf_threshold"])
    )
    iou_threshold = (
        args.iou if args.iou is not None else float(conf_cfg["iou_threshold"])
    )

    dataset_cfg = config["dataset"]
    coco_json = ROOT_DIR / dataset_cfg["coco_json"]
    images_dir = ROOT_DIR / dataset_cfg["images_dir"]
    val_ratio = float(dataset_cfg.get("val", 0.2))
    random_state = int(dataset_cfg.get("random_state", 42))

    negative_dirs = []
    if dataset_cfg.get("negatives_dir"):
        neg_base = ROOT_DIR / dataset_cfg["negatives_dir"]
        for sub_name in ["empty", "ornaments"]:
            target_sub = neg_base / sub_name
            if target_sub.is_dir():
                negative_dirs.append(target_sub)
            else:
                raise FileNotFoundError(f"Не найдена папка с негативами: {target_sub}")

    dataset = TestDetectionDataset(
        images_dir=images_dir,
        ann_file=coco_json,
        negative_dirs=negative_dirs,
        negative_ratio=0.2,
        as_bgr=True,
        split=None if args.split_manifest else "val",
        val_ratio=val_ratio,
        random_state=random_state,
    )
    manifest = None
    if args.split_manifest:
        manifest_path = ROOT_DIR / args.split_manifest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if hashlib.sha256(coco_json.read_bytes()).hexdigest() != manifest["annotations_sha256"]:
            raise ValueError("SHA256 аннотаций не совпадает с тестовым манифестом")
        by_name = {unicodedata.normalize("NFC", item["file_name"]): item for item in dataset.items}
        names = [unicodedata.normalize("NFC", name) for name in manifest["files"]]
        if not names or len(set(names)) != len(names):
            raise ValueError("Манифест должен содержать непустой список уникальных файлов")
        selected = []
        for name in names:
            if name not in by_name:
                raise FileNotFoundError(f"Файл из манифеста не найден в датасете: {name}")
            item = by_name[name]
            if hashlib.sha256(item["path"].read_bytes()).hexdigest() != manifest["image_sha256"][name]:
                raise ValueError(f"SHA256 изображения не совпадает: {name}")
            selected.append(item)
        dataset.items = selected

    all_detectors = config.get("detectors", {})
    target_model_name = args.model or config.get("run_only")

    if target_model_name:
        if target_model_name not in all_detectors:
            available = list(all_detectors.keys())
            raise ValueError(f"Модель '{target_model_name}' не найдена. Доступны: {available}")
        models_to_test = [(target_model_name, all_detectors[target_model_name])]
    else:
        models_to_test = list(all_detectors.items())

    num_positive = sum(1 for item in dataset.items if len(item["boxes"]) > 0)
    num_negative = len(dataset) - num_positive

    print(f"Всего изображений в датасете: {len(dataset)} (с клеймом: {num_positive}, негативов: {num_negative})")
    print(f"Пороги: conf={conf_threshold}, iou={iou_threshold}")

    new_results = []
    json_results = []

    for name, model_cfg in models_to_test:
        raw_weights = model_cfg.get("weights")
        weights_path = ROOT_DIR / raw_weights if raw_weights else None
        
        processor_path = model_cfg.get("processor")
        imgsz = int(model_cfg.get("imgsz", 640))

        if not weights_path or not weights_path.is_file():
            if target_model_name:
                raise FileNotFoundError(f"Файл весов для {name} не найден: {weights_path}")
            print(f"\n[ОШИБКА] Файл весов для {name} не найден: {weights_path}")
            continue

        if not processor_path:
            print(f"\n[ОШИБКА] Для модели {name} не указан путь к процессору")
            continue

        proc_params = {k: v for k, v in model_cfg.items() if k not in ['processor', 'weights']}
        proc_params.setdefault("conf_threshold", conf_threshold)
        proc_params.setdefault("iou_threshold", iou_threshold)
        proc_params.setdefault("imgsz", imgsz)
        processor = load_processor(processor_path, **proc_params)

        print(f"\nЗапуск модели: {name}")
        print(f"Веса: {weights_path}")
        print(f"Путь процессора: {processor_path}")

        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        session = ort.InferenceSession(str(weights_path), providers=providers)
        
        input_name = session.get_inputs()[0].name

        latencies = []
        records = []

        progress_bar = tqdm(
            range(len(dataset)), desc=f"Тестирование {name}", unit="img"
        )

        for idx in progress_bar:
            image_np, target = dataset[idx]

            t0 = time.perf_counter()
            blob, metadata = processor.preprocess(image_np)

            raw_outputs = session.run(None, {input_name: blob})

            pred_boxes, pred_scores = processor.postprocess(
                output=raw_outputs,
                metadata=metadata,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

            records.append(
                {
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "gt_boxes": target["boxes"],
                }
            )

        metrics = evaluate_detections(records, iou_thresh=iou_threshold)
        mean_ms = np.mean(latencies)
        fps = 1000.0 / mean_ms
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\nРезультаты для {name}:")
        print(f"  mAP50:     {metrics['map50']:.4f}")
        print(f"  mAP50-95:  {metrics['map50_95']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  Latency:   {mean_ms:.2f} ms ({fps:.1f} FPS)")
        print(f"  TP: {metrics['true_positives']} | Total Pred: {metrics['total_pred']} | Total GT: {metrics['total_gt']}")

        new_results.append(
            {
                "Model": name,
                "Timestamp": current_time,
                "mAP50": f"{metrics['map50']:.4f}",
                "mAP50-95": f"{metrics['map50_95']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "Latency_ms": f"{mean_ms:.2f}",
                "FPS": f"{fps:.1f}",
                "TP": str(metrics["true_positives"]),
                "Total_Pred": str(metrics["total_pred"]),
                "Total_GT": str(metrics["total_gt"]),
            }
        )
        if json_path is not None:
            json_results.append({
                "model": name,
                "timestamp": current_time,
                "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                "processor": processor_path,
                "imgsz": imgsz,
                "conf_threshold": processor.conf_threshold,
                "nms_iou_threshold": processor.iou_threshold,
                "matching_iou_threshold": iou_threshold,
                "providers": session.get_providers(),
                "metrics": metrics,
                "latency_ms": float(mean_ms),
                "fps": float(fps),
                "images": [
                    {"file_name": item["file_name"],
                     "pred_boxes": record["pred_boxes"].tolist(),
                     "pred_scores": record["pred_scores"].tolist(),
                     "gt_boxes": record["gt_boxes"].tolist()}
                    for item, record in zip(dataset.items, records)
                ],
            })

    if new_results:
        out_csv = ROOT_DIR / args.output_csv
        update_results_csv(out_csv, new_results)
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("x", encoding="utf-8") as stream:
                json.dump({
                    "annotations_sha256": hashlib.sha256(coco_json.read_bytes()).hexdigest(),
                    "dataset_config": dataset_cfg,
                    "split_manifest": manifest,
                    "positive_images": num_positive,
                    "negative_images": num_negative,
                    "evaluation_note": "AP is computed after the configured confidence cutoff, using the repository evaluator.",
                    "results": json_results,
                }, stream, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
