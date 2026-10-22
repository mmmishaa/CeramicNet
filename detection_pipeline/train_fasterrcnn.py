"""Working torchvision baseline. Run as a module from the project root."""
import argparse
from copy import deepcopy
from pathlib import Path
import platform
import time

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .adapters import torchvision_prediction
from .data import read_json, write_json
from .datasets import make_datasets, make_loader, seed_everything
from .evaluate import evaluate, save_predictions
from .experiments import DEFAULT_EXPERIMENTS, resolve_augmentation
from experiment_tracking import ExperimentRun


def build_model(num_classes, pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None,
                                     min_size=640, max_size=640,
                                     box_score_thresh=0.001, box_detections_per_img=100)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    predictions = {}
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for image, target, output in zip(images, targets, outputs):
            image_id = int(target["image_id"].item())
            predictions[image_id] = torchvision_prediction(output, loader.dataset,
                                                           (image.shape[-1], image.shape[-2]))
    return predictions


def train(images_root, splits, output, epochs=None, batch_size=None, profile=None, seed=None,
          workers=2, device=None, experiments=None, tracking="local", tracking_project="CeramicNet",
          run_name=None):
    settings = read_json(experiments or DEFAULT_EXPERIMENTS)
    epochs = settings["epochs"] if epochs is None else epochs
    batch_size = settings["batch_size"] if batch_size is None else batch_size
    seed = settings["seed"] if seed is None else seed
    seed_everything(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    splits, output = Path(splits), Path(output)
    # A fresh directory keeps checkpoint/config history unambiguous.
    output.mkdir(parents=True, exist_ok=False)
    profile, operations = resolve_augmentation("fasterrcnn_resnet50_fpn_v2", profile, experiments)
    train_ds, val_ds = make_datasets(images_root, splits, profile, operations=operations)
    train_loader = make_loader(train_ds, True, seed, batch_size, workers)
    val_loader = make_loader(val_ds, False, seed, batch_size, workers)
    model = build_model(len(train_ds.categories)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    config = {"model": "fasterrcnn_resnet50_fpn_v2", "augmentation": profile,
              "augmentation_operations": operations, "dataset": "dataset.StampDataset",
              "seed": seed, "epochs": epochs, "batch_size": batch_size, "workers": workers,
              "optimizer": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "scheduler": "cosine",
              "pretrained": "FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT",
              "size": {"min_size": 640, "max_size": 640}, "score_floor": 0.001,
              "max_detections": 100, "device": str(device),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "python": platform.python_version(), "images_root": str(Path(images_root).resolve()),
              "split_manifest": read_json(splits / "manifest.json")}
    write_json(output / "config.json", config)
    with ExperimentRun("detection", run_name or output.name, output, config,
                       {"train": splits / "train.json", "val": splits / "val.json"},
                       mode=tracking, project=tracking_project, tags=[config["model"], profile]) as run:
        best_map, history = -1, []
        for epoch in range(epochs):
            started = time.perf_counter()
            model.train()
            loss_sum = 0.0
            loss_components = {}
            learning_rate = optimizer.param_groups[0]["lr"]
            for images, targets in train_loader:
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                components = model([im.to(device) for im in images], targets)
                loss = sum(components.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss in epoch {epoch + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(images)
                for name, value in components.items():
                    loss_components[name] = loss_components.get(name, 0.0) + float(value.detach()) * len(images)
            scheduler.step()
            predictions = predict(model, val_loader, device)
            pred_path = output / "last_val_predictions.json"
            save_predictions(pred_path, splits / "val.json", config["model"], predictions, config)
            metrics = evaluate(splits / "val.json", pred_path, tune_threshold=True)
            map_value = metrics["coco"]["map_50_95"]
            history.append({"epoch": epoch + 1, "train_loss": loss_sum / len(train_ds),
                            "val_map_50_95": map_value, "seconds_train_and_val": time.perf_counter() - started})
            write_json(output / "history.json", history)
            print(history[-1], flush=True)
            run.log_metrics({"loss": loss_sum / len(train_ds), "lr": learning_rate,
                             **{name: total / len(train_ds) for name, total in loss_components.items()}}, epoch + 1, "train")
            run.log_metrics(metrics["coco"], epoch + 1, "val")
            run.log_metrics({key: metrics["operating_point"][key] for key in ("precision", "recall", "f1")},
                            epoch + 1, "val_at_0.5")
            run.log_metrics({"seconds_train_and_val": history[-1]["seconds_train_and_val"]}, epoch + 1, "timing")
            if map_value is not None and map_value > best_map:
                best_map = map_value
                torch.save({"model": model.state_dict(), "epoch": epoch + 1, "config": config,
                            "val_map_50_95": best_map, "label_to_category": train_ds.label_to_category},
                           output / "best.pt")
                save_predictions(output / "best_val_predictions.json", splits / "val.json",
                                 config["model"], predictions, config)
                write_json(output / "best_val_metrics.json", deepcopy(metrics))
                run.summary(best_val_map_50_95=best_map, best_epoch=epoch + 1)
        for filename in ("config.json", "history.json", "best_val_metrics.json", "best_val_predictions.json"):
            if (output / filename).is_file():
                run.artifact(filename, output / filename)
        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--profile", choices=["none", "light", "rotation"], help="Override configs/experiments.json")
    parser.add_argument("--experiments", help="Path to per-model experiment configuration")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--tracking", choices=["local", "offline", "clearml"], default="local")
    parser.add_argument("--tracking-project", default="CeramicNet")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    train(args.images, args.splits, args.output, epochs=args.epochs, batch_size=args.batch_size,
          profile=args.profile, workers=args.workers, device=args.device, experiments=args.experiments,
          tracking=args.tracking, tracking_project=args.tracking_project, run_name=args.run_name)
