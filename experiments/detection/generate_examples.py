"""Run the three saved ONNX detectors on the user-selected images.

This builds illustrations, not a new test score. Split labels are verified for
RTMDet only; historical YOLO membership is deliberately left unverified.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unicodedata

from matplotlib import get_data_path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from src.detection.yolo_processing import YOLOProcessing


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_image(root, record):
    path = root / record["file_name"]
    if path.is_file():
        return path
    basename = unicodedata.normalize("NFC", Path(record["file_name"]).name)
    matches = [p for p in root.rglob("*") if p.is_file()
               and unicodedata.normalize("NFC", p.name) == basename]
    if len(matches) != 1:
        raise ValueError(f"Expected one image: {record['file_name']}; found {len(matches)}")
    return matches[0]


def picture(record, path, boxes, scores, title, threshold):
    source = Image.open(path).convert("RGB")
    source.thumbnail((590, 495))
    scale = source.width / record["width"]
    draw = ImageDraw.Draw(source)
    font = ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), 17)
    for x, y, width, height in record["ground_truth_xywh"]:
        draw.rectangle([x*scale, y*scale, (x+width)*scale, (y+height)*scale], outline="#19a45b", width=3)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [float(value)*scale for value in box]
        draw.rectangle([x1, y1, x2, y2], outline="#df3434", width=3)
        draw.text((x1, max(0, y1-20)), f"{score:.3f}", font=font, fill="#df3434", stroke_width=1, stroke_fill="white")
    canvas = Image.new("RGB", (640, 620), "white")
    canvas.paste(source, ((640-source.width)//2, 105+(495-source.height)//2))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), title, font=font, fill="#172735")
    draw.text((20, 36), record["name"], font=font, fill="#172735")
    split_note = f"RTMDet: {record['rtmdet_split']}"
    if title.startswith("YOLO"):
        split_note += "; YOLO split: unverified"
    draw.text((20, 62), f"{split_note}; confidence {threshold:g}", font=font, fill="#526474")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--yolo-v1", type=Path, required=True)
    parser.add_argument("--yolo-v2", type=Path, required=True)
    parser.add_argument("--rtmdet", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads((ROOT / "selected_images.json").read_text())
    records = selection["images"]
    paths = {}
    for record in records:
        path = resolve_image(args.images_root, record)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Image hash differs from saved RTMDet protocol: {record['name']}")
        paths[record["name"]] = path
    spec = importlib.util.spec_from_file_location("rtmdet_processing", ROOT / "RTMDet/onnx_processing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, title, model_path, folder in [
        ("yolo_v1", "YOLO26s v1", args.yolo_v1, ROOT / "yolo/v1"),
        ("yolo_v2", "YOLO26s v2", args.yolo_v2, ROOT / "yolo/v2"),
        ("rtmdet_s", "RTMDet-s", args.rtmdet, ROOT / "RTMDet"),
    ]:
        model_meta = selection["models"][key]
        if sha256(model_path) != model_meta["sha256"]:
            raise ValueError(f"Wrong weights: {key}")
        threshold = model_meta["confidence_threshold"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        processor = (module.RTMDetProcessing if key == "rtmdet_s" else
                     YOLOProcessing(conf_threshold=threshold, iou_threshold=model_meta["nms_iou"]))
        examples = folder / "examples"
        examples.mkdir(exist_ok=True)
        results, covers, links, lower_threshold_covers, lower_threshold_results = [], [], [], [], []
        for index, record in enumerate(records):
            path = paths[record["name"]]
            with Image.open(path) as image:
                bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            blob, metadata = processor.preprocess(bgr)
            raw = session.run(None, {input_name: blob})
            if key == "rtmdet_s":
                boxes, scores = processor.postprocess(raw, metadata, threshold, model_meta["nms_iou"])
            else:
                boxes, scores = processor.postprocess(raw, metadata)
            image = picture(record, path, boxes, scores, title, threshold)
            name = f"{record['name']}.jpg"
            image.save(examples / name, quality=88)
            if record["use_in_readme"]:
                covers.append(image)
                if key == "rtmdet_s":
                    lower_boxes, lower_scores = processor.postprocess(raw, metadata, 0.5, model_meta["nms_iou"])
                    lower_threshold_covers.append(picture(record, path, lower_boxes, lower_scores, title, 0.5))
                    lower_threshold_results.append({"name": record["name"], "boxes_xyxy": lower_boxes.tolist(),
                                                    "scores": lower_scores.tolist()})
            results.append({"name": record["name"], "image_sha256": record["sha256"],
                            "rtmdet_split": record["rtmdet_split"],
                            "model_training_membership": record["rtmdet_split"] if key == "rtmdet_s" else "unverified",
                            "boxes_xyxy": boxes.tolist(), "scores": scores.tolist()})
            links.append(f"### {record['name']}\n\nRTMDet split: **{record['rtmdet_split']}**. "
                         + ("" if key == "rtmdet_s" else "Исторический YOLO split не подтверждён. ")
                         + f"Детекций: {len(scores)}.\n\n![{record['name']}]({name})\n")
            print(f"{key}: {index+1}/{len(records)}", flush=True)
        cover = Image.new("RGB", (1280, 1240), "#eef1f3")
        assert len(covers) == 4
        for index, image in enumerate(covers):
            cover.paste(image, ((index % 2)*640, (index // 2)*620))
        cover.save(folder / "predictions_examples.png")
        payload = {"model": key, **model_meta, "purpose": "qualitative illustrations; not a test evaluation",
                   "ground_truth": "RTMDet v3 annotations", "records": results}
        if lower_threshold_covers:
            lower_cover = Image.new("RGB", (1280, 1240), "#eef1f3")
            for index, image in enumerate(lower_threshold_covers):
                lower_cover.paste(image, ((index % 2)*640, (index // 2)*620))
            lower_cover.save(folder / "predictions_examples_0.5.png")
            payload["additional_operating_point"] = {"confidence_threshold": 0.5, "records": lower_threshold_results,
                "note": "Pre-existing validation operating point; threshold was not tuned on these four images"}
        (folder / "selected_predictions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        (examples / "README.md").write_text(f"# {title}: все 23 выбранных фото\n\n"
            f"Confidence **{threshold:g}**. Зелёные рамки — разметка RTMDet v3; красные — детектор.\n\n"
            "В списке 4 val и 19 train по сохранённому split RTMDet. Это иллюстрации; по ним не считается итоговое качество.\n\n"
            + "\n".join(links))


if __name__ == "__main__":
    main()
