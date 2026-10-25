import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path

from src.detection.detector import StampDetector
from src.paths import REPORTS_DIR
from src.pipeline import CeramicPipeline
from src.retrieval.books import load_retrieval_config, project_path
from src.retrieval.encoder import StampEncoder
from src.retrieval.index import StampIndex, require_top_k
from src.retrieval.preview import open_preview, render_preview


def parse_args():
    parser = argparse.ArgumentParser(
        description="Найти top-k ближайших клейм эталона для фотографии"
    )
    parser.add_argument("--image", type=str, required=True, help="Фото фрагмента или готовый кроп")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Сколько клейм эталона вернуть. Перекрывает top_k из конфига.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--index", type=str, default=None, help="Файл индекса .npz")
    parser.add_argument("--encoder", type=str, default=None, help="ONNX энкодера")
    parser.add_argument("--books", type=str, default=None, help="Папка books, если индекс перенесли")
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Картинка уже кроп клейма: детектор не запускать",
    )
    parser.add_argument("--conf", type=float, default=None, help="Порог детектора")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Напечатать результат как JSON",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Не открывать картинку с детекцией, кропом и совпадениями",
    )
    return parser.parse_args()


def build_detector(config: dict, conf_threshold: float):
    detector_cfg = config["detector"]
    weights = project_path(detector_cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"ONNX детектора не найден: {weights}")

    module_path, class_name = detector_cfg["processor"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    processor = getattr(module, class_name)()
    image_size = int(detector_cfg.get("imgsz", 640))
    iou_threshold = float(config["confidence_config"]["iou_threshold"])
    sized = _SizedProcessor(processor, image_size, conf_threshold, iou_threshold)
    return StampDetector(weights, sized)


class _SizedProcessor:

    def __init__(self, inner, image_size: int, conf_threshold: float, iou_threshold: float):
        self.inner = inner
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, image):
        return self.inner.preprocess(
            image,
            target_size=(self.image_size, self.image_size),
        )

    def postprocess(self, output, metadata):
        return self.inner.postprocess(
            output,
            metadata,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )


def format_result(result: dict) -> str:
    lines = [
        f"top_k: {result['top_k']}",
        f"детекций: {len(result['detections'])}",
    ]
    if not result["detections"]:
        lines.append("Клеймо на фото не найдено.")
        return "\n".join(lines)

    for index, detection in enumerate(result["detections"], start=1):
        box = ", ".join(f"{value:.1f}" for value in detection["box"])
        if detection["score"] is None:
            lines.append(f"[{index}] кроп без детектора  box=({box})")
        else:
            lines.append(
                f"[{index}] детект score={detection['score']:.3f}  box=({box})"
            )
        if not detection["matches"]:
            lines.append("  совпадений нет")
            continue
        for hit in detection["matches"]:
            catalog = hit["catalog_no"] or "—"
            factory = hit["factory"] or "—"
            lines.append(
                f"  {hit['rank']}. {hit['score']:.4f}  {hit['mark_id']}  "
                f"{hit['book']}  №{catalog}  {factory}"
            )
            lines.append(f"     {hit['image_path']}")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_retrieval_config(args.config)

    if args.top_k is None:
        top_k = require_top_k(int(config.get("top_k", 3)))
    else:
        top_k = require_top_k(args.top_k)

    conf_threshold = (
        float(args.conf)
        if args.conf is not None
        else float(config["confidence_config"]["conf_threshold"])
    )

    encoder_cfg = dict(config["encoder"])
    if args.encoder:
        encoder_cfg["weights"] = str(project_path(args.encoder))
    else:
        encoder_cfg["weights"] = str(project_path(encoder_cfg["weights"]))
    encoder = StampEncoder.from_mapping(encoder_cfg)

    books_dir = project_path(args.books) if args.books else None
    if books_dir is not None and not books_dir.is_dir():
        raise FileNotFoundError(f"Папка эталона не найдена: {books_dir}")
    index_path = project_path(args.index or config["index"]["path"])
    index = StampIndex.load(index_path, books_dir=books_dir)

    detector = None if args.no_detect else build_detector(config, conf_threshold)
    pipeline = CeramicPipeline(
        detector=detector,
        encoder=encoder,
        index=index,
        conf_threshold=conf_threshold,
        top_k=top_k,
    )
    result = pipeline.predict(
        Path(args.image),
        top_k=top_k,
        detect=not args.no_detect,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_result(result))

    if not args.no_show:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_path = render_preview(
            result,
            encoder.input_size,
            REPORTS_DIR / f"retrieval_{stamp}.png",
        )
        open_preview(preview_path)
        print(f"просмотр: {preview_path}")


if __name__ == "__main__":
    main()
