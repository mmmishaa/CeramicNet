import cv2
import numpy as np

def crop_stamp(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    padding_ratio: float = 0.1,
    target_size: int = 224
) -> np.ndarray:
    
    orig_h, orig_w = image.shape[:2]
    x1, y1, x2, y2 = box

    box_h, box_w = y2 - y1, x2 - x1
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

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    pad_left = crop_x1 - new_x1
    pad_top = crop_y1 - new_y1
    pad_right = new_x2 - crop_x2
    pad_bottom = new_y2 - crop_y2

    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

    if target_size is not None and crop.shape[:2] != (target_size, target_size):
        crop = cv2.resize(
            crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )

    return crop