"""Export an official RT-DETR/MMDetection checkpoint to the common evaluator."""
import argparse
from pathlib import Path
import sys

import torch

from .adapters import mmdet_prediction, rtdetr_prediction
from .data import read_json
from .evaluate import save_predictions, evaluate


def prepare_imports(framework, repo):
    repo = Path(repo).resolve()
    native_root = repo / "rtdetrv2_pytorch" if framework == "rtdetr" else repo
    if not native_root.is_dir():
        raise FileNotFoundError(native_root)
    sys.path.insert(0, str(native_root))
    if framework == "rtdetr":
        from .integrations import rtdetr  # noqa: F401


@torch.inference_mode()
def predict_rtdetr(config, checkpoint, images_root, annotations, device):
    from src.core import YAMLConfig

    cfg = YAMLConfig(str(config))
    # Inference restores every detector weight from checkpoint; no backbone download.
    if "PResNet" in cfg.yaml_cfg:
        cfg.yaml_cfg["PResNet"]["pretrained"] = False
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = 0
    dataset_cfg = cfg.yaml_cfg["val_dataloader"]["dataset"]
    dataset_cfg.update(img_folder=str(images_root), ann_file=str(annotations), augmentation_ops=[])
    model = cfg.model.to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state["ema"]["module"] if state.get("ema") else state["model"]
    model.load_state_dict(weights)
    categories = read_json(annotations)["categories"]
    results = {}
    for images, targets in cfg.val_dataloader:
        outputs = model(images.to(device))
        original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
        decoded = cfg.postprocessor(outputs, original_sizes)
        for target, output in zip(targets, decoded):
            image_id = int(target["image_id"].item())
            width, height = map(int, target["orig_size"].tolist())
            results[image_id] = rtdetr_prediction(output, categories, (width, height))
    return results


@torch.inference_mode()
def predict_mmdet(config, checkpoint, images_root, annotations, device):
    from mmdet.apis import init_detector, inference_detector

    model = init_detector(str(config), str(checkpoint), device=str(device))
    data = read_json(annotations)
    predictions = {}
    for record in data["images"]:
        result = inference_detector(model, str(images_root / record["file_name"]))
        predictions[record["id"]] = mmdet_prediction(result, data["categories"],
                                                    (record["width"], record["height"]))
    return predictions


def export(framework, repo, config, checkpoint, images_root, annotations, output, tune_threshold=False,
           threshold=0.5, device=None):
    prepare_imports(framework, repo)
    config, checkpoint, images_root, annotations, output = map(
        lambda value: Path(value).resolve(), (config, checkpoint, images_root, annotations, output))
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    predictor = predict_rtdetr if framework == "rtdetr" else predict_mmdet
    predictions = predictor(config, checkpoint, images_root, annotations, device)
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = config.with_suffix(".metadata.json")
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    metadata.update(config_path=str(config), checkpoint=str(checkpoint), device=str(device),
                    score_floor=0.001, max_detections=100, torch=str(torch.__version__))
    model_name = metadata.get("model", config.stem)
    save_predictions(output / "predictions.json", annotations, model_name, predictions, metadata)
    return evaluate(annotations, output / "predictions.json", output / "metrics.json",
                    tune_threshold=tune_threshold, threshold=threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("framework", choices=["rtdetr", "mmdet"])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tune-threshold", action="store_true", help="Validation only; never use on test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Operating threshold selected on validation")
    parser.add_argument("--device")
    args = parser.parse_args()
    export(args.framework, args.repo, args.config, args.checkpoint, args.images, args.annotations,
           args.output, args.tune_threshold, args.threshold, args.device)
