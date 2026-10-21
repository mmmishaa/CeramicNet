"""Resolve a model's augmentation choice from one shared experiment file."""
from pathlib import Path

from dataset.transforms.factory import augmentation_ops
from .data import read_json


DEFAULT_EXPERIMENTS = Path(__file__).resolve().parents[1] / "configs/experiments.json"


def resolve_augmentation(model, profile=None, experiments=None):
    config = read_json(experiments or DEFAULT_EXPERIMENTS)
    specification = config["models"][model]
    selected = profile if profile is not None else specification["augmentation"]
    operations = specification.get("operations") if profile is None else None
    if selected == "native":
        if model == "fasterrcnn_resnet50_fpn_v2":
            raise ValueError("Faster R-CNN has no native augmentations; choose none/light/rotation or custom operations")
        if operations is not None:
            raise ValueError("Use either native augmentations or team operations to avoid applying both")
        return selected, []
    return selected, augmentation_ops(selected, operations)
