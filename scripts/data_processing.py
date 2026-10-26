import importlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import cv2

from src.utils import load_yaml
from src.paths import CONFIGS_DIR, DATA_DIR, BOOKS_DIR
from src.detection.detector import StampDetector
from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import read_bgr

CONFIG_DIR = CONFIGS_DIR / "data_processing" / "data_processing.yaml"
config = load_yaml(CONFIG_DIR)

low_conf = float(config["confidence"]["low_confidence"])
high_conf = float(config["confidence"]["high_confidence"])

INPUT_DIR = BOOKS_DIR
OUTPUT_DIR = DATA_DIR / "processed_books"

module_path, class_name = config["detection_model"]["processor"].rsplit(".", 1)
module = importlib.import_module(module_path)

processor = getattr(module, class_name)(
    imgsz=640,
    conf_threshold=low_conf,
    iou_threshold=config["detection_model"]["iou_threshold"]
)

detector = StampDetector(
    model_path=config["detection_model"]["weights"],
    processor=processor
)

records = []

image_paths = sorted(INPUT_DIR.rglob("*.png"))

for img_path in tqdm(image_paths, desc="Обработка эталонов"):
    image = read_bgr(path=img_path)
    if image is None:
        continue

    boxes, scores = detector.predict(image=image)

    if len(scores) == 0:
        records.append({
            "image_path": img_path.relative_to(INPUT_DIR).as_posix(),
            "status": "no_detection",
            "score": 0.0,
            "box": None
        })
        continue

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_box = boxes[best_idx].tolist()

    if best_score >= high_conf:
        status = "high_confidence"
    elif best_score >= low_conf:
        status = "low_confidence"
    else:
        status = "no_detection"

    records.append({
        "image_path": img_path.relative_to(INPUT_DIR).as_posix(),
        "status": status,
        "score": best_score if status != "no_detection" else 0.0,
        "box": best_box if status != "no_detection" else None,
    })

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

json_path = OUTPUT_DIR / "processed_books_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

for item in tqdm(records, desc="Сохранение файлов"):
    rel_path = Path(item["image_path"])
    full_src = INPUT_DIR / rel_path
    status = item["status"]

    target_folder = OUTPUT_DIR / rel_path.parent / status
    target_folder.mkdir(parents=True, exist_ok=True)

    dest_file = target_folder / rel_path.name
    img = read_bgr(path=full_src)
    if img is None:
        continue

    target_img = crop_stamp(image=img, box=item["box"]) if status in ("high_confidence", "low_confidence") else img
    success, buffer = cv2.imencode(dest_file.suffix, target_img)
    if success:
        buffer.tofile(str(dest_file))