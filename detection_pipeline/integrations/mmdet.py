"""Register team augmentations in MMDetection's normal CocoDataset pipeline."""
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

from .joint import augment_mmdet_results


@TRANSFORMS.register_module()
class TeamAugment(BaseTransform):
    def __init__(self, operations):
        self.operations = operations

    def transform(self, results):
        return augment_mmdet_results(results, self.operations)
