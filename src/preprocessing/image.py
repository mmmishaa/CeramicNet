from pathlib import Path

import cv2
import numpy as np


def read_bgr(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def fit_square(
    image: np.ndarray,
    target_size: int,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)

    if height == side and width == side:
        square = image
    else:
        square = np.full((side, side, 3), fill, dtype=image.dtype)
        top = (side - height) // 2
        left = (side - width) // 2
        square[top : top + height, left : left + width] = image

    if square.shape[0] != target_size:
        square = cv2.resize(
            square,
            (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )
    return square
