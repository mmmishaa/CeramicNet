from pathlib import Path
import cv2
import numpy as np
from src.retrieval.books import load_retrieval_config

try:
    _CONTRAST_CFG = load_retrieval_config().get("preprocessing", {}).get("contrast", {})
except Exception:
    _CONTRAST_CFG = {}


def read_bgr(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image

def enhance_stamp_contrast(
    image_bgr: np.ndarray,
    clip_limit: float | None = None,
    tile_grid_size: tuple[int, int] | None = None,
    invert_dark: bool | None = None,
    sharpen: bool | None = None,
) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Ожидался цветной кадр HWC, получено: {image_bgr.shape}")

    source = (
        image_bgr
        if image_bgr.dtype == np.uint8
        else np.clip(image_bgr, 0, 255).astype(np.uint8)
    )

    c_limit = (
        clip_limit
        if clip_limit is not None
        else float(_CONTRAST_CFG.get("clip_limit", 2.0))
    )
    grid = (
        tile_grid_size
        if tile_grid_size is not None
        else tuple(_CONTRAST_CFG.get("tile_grid_size", [8, 8]))
    )
    do_invert = (
        invert_dark
        if invert_dark is not None
        else bool(_CONTRAST_CFG.get("invert_dark_background", True))
    )
    do_sharpen = (
        sharpen
        if sharpen is not None
        else bool(_CONTRAST_CFG.get("sharpen", False))
    )

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    if do_invert and np.mean(gray) < 120:
        gray = 255 - gray

    clahe = cv2.createCLAHE(clipLimit=c_limit, tileGridSize=grid)
    enhanced_gray = clahe.apply(gray)

    enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    if do_sharpen:
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    return np.clip(enhanced, 0, 255).astype(np.uint8)