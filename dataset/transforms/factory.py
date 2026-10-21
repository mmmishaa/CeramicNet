"""Build the same team augmentations for torchvision, RT-DETR and MMDetection."""
from copy import deepcopy

from .augment import (Brightness, Defocus, Grayscale, HighContrast, HorizontalFlip,
                      LowContrast, Rotate90, VerticalFlip)
from .base import Compose


TRANSFORMS = {cls.__name__: cls for cls in (
    Brightness, Defocus, Grayscale, HighContrast, HorizontalFlip, LowContrast, Rotate90, VerticalFlip
)}
PROFILES = {
    "none": [],
    "light": [
        {"type": "Brightness", "p": 0.5, "factor": [0.85, 1.15]},
        {"type": "LowContrast", "p": 0.15, "factor": [0.8, 1.0]},
        {"type": "HighContrast", "p": 0.15, "factor": [1.0, 1.2]},
        {"type": "Grayscale", "p": 0.1},
    ],
    "rotation": [
        {"type": "Rotate90", "p": 0.5, "angles": [90, 180, 270]},
        {"type": "Brightness", "p": 0.5, "factor": [0.85, 1.15]},
    ],
}


def augmentation_ops(profile="light", operations=None):
    """Return a fresh, serializable list for run metadata and native configs."""
    if operations is None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown team augmentation profile {profile!r}; choose {tuple(PROFILES)}")
        operations = PROFILES[profile]
    operations = deepcopy(operations)
    # Validate names, probabilities and constructor parameters before training.
    for operation in operations:
        if not isinstance(operation, dict) or operation.get("type") not in TRANSFORMS:
            raise ValueError(f"Unknown augmentation: {operation!r}")
        if not 0 <= operation.get("p", 0.5) <= 1:
            raise ValueError(f"Augmentation probability must be in [0, 1]: {operation}")
        TRANSFORMS[operation["type"]](**{k: v for k, v in operation.items() if k != "type"})
    return operations


def build_augmentations(profile="light", operations=None):
    return Compose([TRANSFORMS[op["type"]](**{k: v for k, v in op.items() if k != "type"})
                    for op in augmentation_ops(profile, operations)])
