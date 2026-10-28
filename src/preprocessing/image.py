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


def enhance_stamp_contrast(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE по яркости и лёгкая резкость. Стоит после кропа и до ч/б энкодера."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Ожидался цветной кадр HWC, получено: {image_bgr.shape}")
    source = image_bgr if image_bgr.dtype == np.uint8 else np.clip(image_bgr, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
