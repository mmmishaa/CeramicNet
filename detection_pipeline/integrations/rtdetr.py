"""Register StampDataset in the official lyuwenyu/RT-DETR repository.

Imported by train_native after adding rtdetrv2_pytorch to sys.path. This adapter
keeps CocoDetection's COCO evaluator and native transforms/collate interface.
"""
from PIL import Image
import torch

from src.core import register
from src.data.dataset.coco_dataset import CocoDetection
from src.data._misc import convert_to_tv_tensor

from dataset import StampDataset
from dataset.transforms.factory import build_augmentations


@register()
class TeamCocoDetection(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False,
                 remap_mscoco_category=False, augmentation_ops=None):
        if return_masks or remap_mscoco_category:
            raise ValueError("TeamCocoDetection expects bbox-only data with contiguous zero-based categories")
        super().__init__(img_folder, ann_file, transforms, return_masks, remap_mscoco_category)
        self.stamps = StampDataset(img_folder, ann_file=ann_file, image_root=img_folder,
                                   model="default", standard_size=None,
                                   transform=build_augmentations(operations=augmentation_ops or []))
        if [item["image_id"] for item in self.stamps.items] != self.ids:
            raise ValueError("COCO evaluator IDs and StampDataset IDs disagree")

    def load_item(self, index):
        array, raw = self.stamps.load_raw(index)
        image = Image.fromarray(array)
        boxes = torch.as_tensor(raw["boxes"], dtype=torch.float32).reshape(-1, 4)
        target = {"boxes": convert_to_tv_tensor(boxes, key="boxes", spatial_size=array.shape[:2]),
                  "labels": torch.as_tensor(raw["labels"], dtype=torch.int64),
                  "image_id": torch.tensor([raw["image_id"]]), "idx": torch.tensor([index]),
                  # This official repository uses [width, height], including postprocessing.
                  "orig_size": torch.tensor([raw["orig_size"][1], raw["orig_size"][0]]),
                  "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}
        return image, target
