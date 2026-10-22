import argparse
import csv
from datetime import datetime
import importlib.util
import os
from pathlib import Path
import time
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import yaml

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

from detection_configs.test_detection_dataset import TestDetectionDataset


def load_processor(file_path_str: str, class_name: str):
    file_path = Path(file_path_str)
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл процессора не найден: {file_path}")

    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль из {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"В {file_path} не найден класс '{class_name}'")

    processor_cls = getattr(module, class_name)
    return processor_cls()


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

    return float(np.trapezoid(y, x))


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
    default_config = "test_pipelines/detection_models/detection_test_config.yaml"

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
        default="test_pipelines/detection_models/benchmark_results.csv",
        help="Путь к файлу сохранения результатов",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Конфигурационный файл не найден: {config_path}"
        )

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
    coco_json = Path(dataset_cfg["coco_json"])
    images_dir = Path(dataset_cfg["images_dir"])
    val_ratio = float(dataset_cfg.get("val", 0.2))
    random_state = int(dataset_cfg.get("random_state", 42))

    negative_dirs = []
    if dataset_cfg.get("negatives_dir"):
        neg_path = Path(dataset_cfg["negatives_dir"])
        if neg_path.is_dir():
            negative_dirs.append(neg_path)

    dataset = TestDetectionDataset(
        images_dir=images_dir,
        ann_file=coco_json,
        negative_dirs=negative_dirs,
        negative_ratio=0.2,
        as_bgr=True,
        split="val",
        val_ratio=val_ratio,
        random_state=random_state,
    )

    models_to_test = config["models"]
    target_model_name = args.model or config.get("run_only")
    if target_model_name:
        models_to_test = [
            m for m in models_to_test if m["name"] == target_model_name
        ]
        if not models_to_test:
            available = [m["name"] for m in config["models"]]
            raise ValueError(
                f"Модель '{target_model_name}' не найдена в конфиге. Доступны: {available}"
            )

    num_positive = sum(1 for item in dataset.items if len(item["boxes"]) > 0)
    num_negative = len(dataset) - num_positive

    print(f"Всего изображений в датасете: {len(dataset)} (с клеймом: {num_positive}, негативов: {num_negative})")
    print(f"Пороги: conf={conf_threshold}, iou={iou_threshold}")

    new_results = []

    for model_cfg in models_to_test:
        name = model_cfg["name"]
        weights_path = (
            model_cfg.get("weights_path")
            or model_cfg.get("weithts_path")
            or model_cfg.get("weitgts_path")
        )
        process_file = model_cfg.get("process_file")
        process_class = model_cfg.get("process_class")
        imgsz = int(model_cfg.get("imgsz", 640))

        if not weights_path or not Path(weights_path).is_file():
            print(f"\n[ОШИБКА] Файл весов для {name} не найден: {weights_path}")
            continue

        if not process_file or not process_class:
            print(
                f"\n[ОШИБКА] Для модели {name} не указаны 'process_file' или 'process_class'"
            )
            continue

        processor = load_processor(process_file, process_class)

        print(f"\nЗапуск модели: {name}")
        print(f"Веса: {weights_path}")
        print(f"Класс процессора: {process_class} ({process_file})")

        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available]
        session = ort.InferenceSession(str(weights_path), providers=providers)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        latencies = []
        records = []

        progress_bar = tqdm(
            range(len(dataset)), desc=f"Тестирование {name}", unit="img"
        )

        for idx in progress_bar:
            image_np, target = dataset[idx]

            t0 = time.perf_counter()
            blob, metadata = processor.preprocess(
                image_np, target_size=(imgsz, imgsz)
            )

            raw_outputs = session.run([output_name], {input_name: blob})

            pred_boxes, pred_scores = processor.postprocess(
                output=raw_outputs,
                metadata=metadata,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
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
        print(
            f"  TP: {metrics['true_positives']} | Total Pred: {metrics['total_pred']} | Total GT: {metrics['total_gt']}"
        )

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

    if new_results:
        update_results_csv(Path(args.output_csv), new_results)

if __name__ == "__main__":
    main()