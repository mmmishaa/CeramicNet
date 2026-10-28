import cv2
import numpy as np
from src.retrieval.books import load_retrieval_config

try:
    _CFG = load_retrieval_config().get("preprocessing", {}).get("crop", {})
except Exception:
    _CFG = {}

DEFAULT_TARGET_SIZE = int(_CFG.get("target_size", 224))
DEFAULT_PADDING_RATIO = float(_CFG.get("padding_ratio", 0.1))
DEFAULT_PAD_VALUE = int(_CFG.get("pad_value", 255))

def crop_stamp(
    image: np.ndarray,
    box: tuple[float, float, float, float] | list[float],
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    target_size: int = DEFAULT_TARGET_SIZE,
    pad_value: int = DEFAULT_PAD_VALUE,
) -> np.ndarray:
    if image is None or image.size == 0 or box is None:
        return np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)

    orig_h, orig_w = image.shape[:2]
    x1, y1, x2, y2 = box

    box_h = float(y2 - y1)
    box_w = float(x2 - x1)

    if box_w <= 0 or box_h <= 0:
        return np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)

    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0

    side = max(box_w, box_h) * (1.0 + padding_ratio)

    new_x1 = int(round(cx - side / 2.0))
    new_y1 = int(round(cy - side / 2.0))
    new_x2 = int(round(cx + side / 2.0))
    new_y2 = int(round(cy + side / 2.0))

    crop_x1 = max(0, new_x1)
    crop_y1 = max(0, new_y1)
    crop_x2 = min(orig_w, new_x2)
    crop_y2 = min(orig_h, new_y2)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)

    pad_left = max(0, crop_x1 - new_x1)
    pad_top = max(0, crop_y1 - new_y1)
    pad_right = max(0, new_x2 - crop_x2)
    pad_bottom = max(0, new_y2 - crop_y2)

    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )

    if target_size is not None and crop.shape[:2] != (target_size, target_size):
        crop = cv2.resize(
            crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )

    return crop