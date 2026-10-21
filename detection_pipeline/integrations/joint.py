"""Framework-independent bridge for the team's RGB/XYXY augmentation contract."""
import numpy as np

from dataset.transforms.factory import build_augmentations


def augment_mmdet_results(results, operations):
    """MMDetection uses BGR images and optional BaseBoxes; the team uses RGB/numpy."""
    boxes = results["gt_bboxes"]
    tensor = getattr(boxes, "tensor", None)
    numpy_boxes = tensor.detach().cpu().numpy() if tensor is not None else np.asarray(boxes)
    target = {"boxes": numpy_boxes.copy(), "labels": results["gt_bboxes_labels"].copy()}
    image, target = build_augmentations(operations=operations)(
        np.ascontiguousarray(results["img"][..., ::-1]), target)
    if len(target["boxes"]) != len(results["gt_ignore_flags"]):
        raise ValueError("Augmentations that drop boxes must also update MMDetection ignore flags")
    results["img"] = np.ascontiguousarray(image[..., ::-1])
    results["img_shape"] = image.shape[:2]
    results["gt_bboxes"] = type(boxes)(target["boxes"]) if tensor is not None else target["boxes"]
    results["gt_bboxes_labels"] = target["labels"]
    return results
