import argparse
import importlib
from pathlib import Path
import random
import yaml

from src.detection.detector import StampDetector
from src.paths import CONFIGS_DIR, DATA_DIR, REPORTS_DIR
from src.pipeline import CeramicPipeline
from src.retrieval.books import project_path, resolve_reference
from src.retrieval.encoder import StampEncoder
from src.retrieval.index import StampIndex
from src.retrieval.preview import open_preview, render_preview


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Empty config: {path}")
    return data


def get_random_image(images_dir: Path) -> Path:
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in valid_extensions
    ]
    if not candidates:
        raise FileNotFoundError(f"No valid images found in: {images_dir}")
    return random.choice(candidates)


def build_detector(det_cfg: dict) -> StampDetector:
    weights = project_path(det_cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights}")

    module_path, class_name = det_cfg["processor"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    processor = getattr(module, class_name)()

    image_size = int(det_cfg.get("imgsz", 640))
    conf_threshold = float(det_cfg.get("conf_threshold", 0.75))
    iou_threshold = float(det_cfg.get("iou_threshold", 0.5))

    class SizedProcessor:
        def __init__(self, inner):
            self.inner = inner

        def preprocess(self, image):
            return self.inner.preprocess(image, target_size=(image_size, image_size))

        def postprocess(self, output, metadata):
            return self.inner.postprocess(
                output,
                metadata,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )

    return StampDetector(weights, SizedProcessor(processor))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CeramicNet Pipeline")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to image for inference",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random image from the dataset",
    )
    parser.add_argument(
        "--retrieval-config",
        type=Path,
        default=CONFIGS_DIR / "retrieval" / "retrieval_config.yaml",
    )
    parser.add_argument(
        "--detection-config",
        type=Path,
        default=CONFIGS_DIR / "detection" / "detection_config.yaml",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-detect", action="store_true", help="Run without detector")
    parser.add_argument("--show", action="store_true", help="Display result preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_images_dir = DATA_DIR / "ceramic_dataset" / "images"

    if args.random or args.image is None:
        image_path = get_random_image(dataset_images_dir)
        print(f"Selected random image: {image_path.name}")
    else:
        image_path = project_path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

    ret_cfg = load_yaml(args.retrieval_config)
    det_cfg = load_yaml(args.detection_config)

    books_path, _ = resolve_reference(
        books_dir=ret_cfg.get("books", {}).get("dir"),
        meta_csv=ret_cfg.get("books", {}).get("meta_csv"),
    )

    encoder = StampEncoder.from_mapping(
        ret_cfg["encoder"],
        weights=project_path(ret_cfg["encoder"]["weights"]),
    )

    index_path = project_path(ret_cfg["index"]["path"])
    index = StampIndex.load(index_path, books_dir=books_path)

    top_k = args.top_k if args.top_k is not None else int(ret_cfg.get("top_k", 3))

    detector = None
    if not args.no_detect:
        detector = build_detector(det_cfg)

    pipeline = CeramicPipeline(
        encoder=encoder,
        index=index,
        detector=detector,
        conf_threshold=float(det_cfg.get("conf_threshold", 0.75)),
        top_k=top_k,
    )

    result = pipeline.predict(image_path, top_k=top_k, detect=not args.no_detect)

    print(f"Processed: {result['image']}")
    print(f"Found detections: {len(result['detections'])}")

    for i, detection in enumerate(result["detections"], start=1):
        score_val = detection["score"]
        score_str = f"{score_val:.3f}" if score_val is not None else "no-detector"
        print(f"\nDetection #{i} | Conf: {score_str} | Box: {detection['box']}")
        for match in detection["matches"]:
            print(
                f"  Rank {match['rank']}: ID={match['mark_id']} "
                f"(Score: {match['score']:.4f}) | "
                f"Book: {match['book']} | Factory: {match['factory']}"
            )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = REPORTS_DIR / f"result_{image_path.stem}.png"
    render_preview(result, target_size=pipeline.target_size, output_path=preview_path)
    print(f"\nPreview saved to: {preview_path}")

    if args.show:
        open_preview(preview_path)

if __name__ == "__main__":
    main()