import argparse
import importlib
from pathlib import Path
import random
from typing import List
import yaml

from src.paths import CONFIGS_DIR, DATA_DIR, REPORTS_DIR
from src.pipeline import CeramicPipeline
from src.retrieval.books import project_path, resolve_reference
from src.retrieval.encoder import StampEncoder
from src.retrieval.index import StampIndex
from src.detection.detector import StampDetector
from src.retrieval.preview import open_preview, render_preview

from src.utils import load_yaml

VALID_EXTENSIONS = {".jpg", ".png"}

def get_random_images(images_dir: Path, count: int = 1) -> List[Path]:
    candidates = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No valid images found in: {images_dir}")
    num_to_take = min(count, len(candidates))
    return random.sample(candidates, num_to_take)

def resolve_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {input_path}")
        return [input_path]

    if input_path.is_dir():
        images = sorted(
            [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        )
        if not images:
            raise FileNotFoundError(f"No valid images found in directory: {input_path}")
        return images

    raise FileNotFoundError(f"Path not found: {input_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CeramicNet Pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to an image file or directory with images for inference",
    )
    parser.add_argument(
        "--random",
        nargs="?",
        const=1,
        type=int,
        default=None,
        help="Pick N random images from the dataset (default: 1 if flag is passed without value)",
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

    if args.random is not None:
        target_images = get_random_images(dataset_images_dir, count=args.random)
        print(f"Selected {len(target_images)} random image(s): {[img.name for img in target_images]}")
    elif args.input is None:
        target_images = get_random_images(dataset_images_dir, count=1)
        print(f"Selected random image: {target_images[0].name}")
    else:
        resolved_path = project_path(args.input)
        target_images = resolve_image_paths(resolved_path)
        print(f"Total images to process: {len(target_images)}")

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

    module_path, class_name = det_cfg["processor"].rsplit(".", 1)
    module = importlib.import_module(module_path)

    processor = getattr(module, class_name)(
        imgsz=det_cfg.get("imgsz", 640),
        conf_threshold=det_cfg.get("conf_threshold", 0.7),
        iou_threshold=det_cfg.get("iou_threshold", 0.5),
    )

    detector = StampDetector(model_path=det_cfg["weights"], processor=processor)

    pipeline = CeramicPipeline(
        encoder=encoder,
        index=index,
        detector=detector,
        conf_threshold=float(det_cfg.get("conf_threshold", 0.75)),
        top_k=top_k,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(target_images, start=1):
        print(f"\n[{idx}/{len(target_images)}] Processing: {image_path.name}")
        result = pipeline.predict(image_path, top_k=top_k, detect=not args.no_detect)

        detections = result.get("detections", [])
        print(f"Found detections: {len(detections)}")

        for i, detection in enumerate(detections, start=1):
            score_val = detection["score"]
            score_str = f"{score_val:.3f}" if score_val is not None else "no-detector"
            print(f"  Detection #{i} | Conf: {score_str} | Box: {detection['box']}")
            for match in detection.get("matches", []):
                print(
                    f"    Rank {match['rank']}: ID={match['mark_id']} "
                    f"(Score: {match['score']:.4f}) | "
                    f"Book: {match['book']} | Factory: {match['factory']}"
                )

        preview_path = REPORTS_DIR / f"result_{image_path.stem}.png"
        saved_previews = render_preview(result, target_size=pipeline.target_size, output_path=preview_path)

        for p in saved_previews:
            print(f"Preview saved to: {p}")
            if args.show:
                open_preview(p)

if __name__ == "__main__":
    main()