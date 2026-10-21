"""Thin torchvision adapter around the team's dataset.StampDataset."""
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import StampDataset, collate_stamps
from dataset.transforms.factory import build_augmentations


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


collate_detection = collate_stamps


def build_transforms(profile="light", train=False, operations=None):
    """Team transforms operate on numpy uint8 RGB and numpy XYXY boxes."""
    return build_augmentations(profile, operations) if train else build_augmentations("none")


class CocoStampDataset(StampDataset):
    """Keep the pipeline API; image/bbox reading lives in StampDataset."""
    def __init__(self, images_root, ann_file, transforms=None):
        super().__init__(root_dir=images_root, ann_file=ann_file, image_root=images_root,
                         transform=transforms, standard_size=None, model="faster_rcnn")
        self.categories = [{"id": category_id, "name": name} for category_id, name in self.categories.items()]

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        boxes = target["boxes"]
        target = {"boxes": boxes, "labels": target["labels"],
                  "image_id": torch.tensor(target["image_id"], dtype=torch.int64),
                  "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}
        return image, target


def make_loader(dataset, train, seed=42, batch_size=8, workers=2):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=False,
                      num_workers=workers, collate_fn=collate_detection,
                      worker_init_fn=seed_worker, generator=generator)


def make_datasets(images_root, split_dir, profile="light", dataset_cls=CocoStampDataset,
                  train_transforms=None, eval_transforms=None, operations=None):
    """Two datasets share fixed COCO splits, never training randomness."""
    split_dir = Path(split_dir)
    train = dataset_cls(images_root=images_root, ann_file=split_dir / "train.json",
                        transforms=train_transforms if train_transforms is not None else build_transforms(profile, True, operations))
    val = dataset_cls(images_root=images_root, ann_file=split_dir / "val.json",
                      transforms=eval_transforms if eval_transforms is not None else build_transforms("none", False))
    return train, val
