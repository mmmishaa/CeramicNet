"""Превращает книжный эталон (чёрный рисунок на сером) в подобие фото клейма на фарфоре.

На реальных снимках: цветная краска на белой глазури, поворот на любой угол, кривая поверхность,
стёртые и обрезанные куски, кракелюр, блики, низкое разрешение, рельефные (бескрасочные) клейма.
"""

from __future__ import annotations

import cv2
import numpy as np

# Цвета краски в BGR: кобальт, голубой, зелёный, бирюза, красно-коричневый, оранжевый, коричневый, чёрный, золото.
INKS = np.array(
    [
        (160, 50, 20), (210, 140, 70), (60, 130, 50), (130, 120, 30), (40, 50, 170),
        (40, 110, 220), (40, 60, 100), (40, 40, 40), (60, 150, 190),
    ],
    dtype=np.float32,
)


def _smooth_field(rng: np.random.Generator, shape, cells: int) -> np.ndarray:
    small = rng.random((cells, cells)).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def ink_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Доля краски 0..1: фон страницы (светлый процентиль) -> 0, самые тёмные штрихи -> 1."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    paper, ink = np.percentile(gray, 90), np.percentile(gray, 1)
    return ((paper - gray) / max(paper - ink, 20.0)).clip(0, 1)


def _pad(mask: np.ndarray, ratio: float) -> np.ndarray:
    height, width = mask.shape
    pad = int(max(height, width) * ratio)
    return cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)


def _geometry(rng, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    # Перспектива: снимок не строго сверху.
    jitter = 0.12 * min(height, width)
    src = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
    mask = cv2.warpPerspective(mask, cv2.getPerspectiveTransform(src, dst), (width, height))
    # Изгиб: донце чашки или тарелки.
    if rng.random() < 0.5:
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        amp = rng.uniform(0, 0.06) * min(height, width)
        freq = rng.uniform(0.5, 1.5) * np.pi / max(height, width)
        map_x = xs + amp * np.sin(ys * freq + rng.uniform(0, 6.28))
        map_y = ys + amp * np.sin(xs * freq + rng.uniform(0, 6.28))
        mask = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    # Произвольный поворот: клеймо на донце лежит как угодно.
    angle = rng.uniform(0, 360)
    rot = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cos, sin = abs(rot[0, 0]), abs(rot[0, 1])
    new_w, new_h = int(height * sin + width * cos), int(height * cos + width * sin)
    rot[0, 2] += new_w / 2 - width / 2
    rot[1, 2] += new_h / 2 - height / 2
    mask = cv2.warpAffine(mask, rot, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    if rng.random() < 0.15:
        mask = mask[:, ::-1]
    return np.ascontiguousarray(mask)


def _wear(rng, mask: np.ndarray) -> np.ndarray:
    # Толщина штриха: печать краской растекается или истончается.
    if rng.random() < 0.5:
        size = int(rng.integers(2, 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel) if rng.random() < 0.6 else cv2.erode(mask, kernel)
    # Неравномерный оттиск и стёртость.
    mask = mask * (0.25 + 0.75 * _smooth_field(rng, mask.shape, int(rng.integers(3, 8))))
    if rng.random() < 0.6:
        speckle = _smooth_field(rng, mask.shape, int(rng.integers(20, 60)))
        mask = mask * (speckle > rng.uniform(0.05, 0.35))
    # Скол или обрезанное фото: часть клейма отрезана прямой.
    if rng.random() < 0.3:
        height, width = mask.shape
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        theta = rng.uniform(0, 6.28)
        side = (xs - width / 2) * np.cos(theta) + (ys - height / 2) * np.sin(theta)
        mask = mask * (side < rng.uniform(0.1, 0.45) * max(height, width))
    return mask


def _compose(rng, mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    glaze = rng.uniform(205, 250) + rng.uniform(-12, 12, 3)
    shade = 1.0 - rng.uniform(0, 0.12) * _smooth_field(rng, (height, width), 3)
    background = glaze[None, None, :] * shade[..., None]
    if rng.random() < 0.3:
        # Пожелтевшая или грязная глазурь.
        stain = _smooth_field(rng, (height, width), 6)[..., None]
        background = background * (1 - 0.25 * stain * np.array([0.9, 0.5, 0.2]))
    if rng.random() < 0.15:
        # Рельефное клеймо без краски: только светотень по краям.
        relief = cv2.GaussianBlur(mask, (0, 0), 1.2)
        dx = cv2.Sobel(relief, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(relief, cv2.CV_32F, 0, 1, ksize=3)
        light = rng.uniform(15, 45) * (dx * np.cos(1.0) + dy * np.sin(1.0))
        image = background + light[..., None]
    else:
        ink = INKS[rng.integers(len(INKS))] + rng.uniform(-25, 25, 3)
        alpha = mask * rng.uniform(0.45, 1.0)
        if rng.random() < 0.4:
            # Подглазурная краска расплывается.
            alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(0.8, 2.0))
        image = background * (1 - alpha[..., None]) + ink[None, None, :] * alpha[..., None]
    if rng.random() < 0.2:
        # Кракелюр: сетка тонких тёмных трещин.
        cracks = np.zeros((height, width), np.uint8)
        for _ in range(int(rng.integers(2, 8))):
            points = np.cumsum(rng.normal(0, max(height, width) / 12, (5, 2)), axis=0)
            points += rng.uniform(0, [width, height])
            cv2.polylines(cracks, [points.astype(np.int32)], False, 255, 1, cv2.LINE_AA)
        image = image - rng.uniform(20, 70) * (cracks[..., None] / 255.0) * np.array([0.8, 0.9, 1.0])
    if rng.random() < 0.3:
        # Блик глазури.
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        radius = rng.uniform(0.1, 0.4) * max(height, width)
        glare = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * radius**2))
        image = image + rng.uniform(20, 60) * glare[..., None]
    return image


def _camera(rng, image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    # Реальные кропы мелкие (часто 60–150 px): сначала уменьшаем, потом пайплайн растянет обратно.
    side = rng.uniform(50, 260)
    scale = min(1.0, side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (max(8, int(width * scale)), max(8, int(height * scale))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.6:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.6))
    image = image + rng.normal(0, rng.uniform(1, 7), image.shape)
    image = image.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        quality = int(rng.integers(35, 90))
        image = cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    return image


def photo_like(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Эталон -> «фото» того же клейма. Возвращает BGR uint8 произвольного размера; дальше обычный prepare()."""
    mask = ink_mask(image_bgr)
    longest = max(mask.shape)
    if longest > 320:
        scale = 320 / longest
        mask = cv2.resize(mask, (max(1, int(mask.shape[1] * scale)), max(1, int(mask.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    mask = _pad(mask, 0.15)
    mask = _geometry(rng, mask)
    mask = _wear(rng, mask)
    mask = _tight(rng, mask)
    return _camera(rng, _compose(rng, mask))


def _tight(rng, mask: np.ndarray) -> np.ndarray:
    """Детектор режет по видимой части клейма: обрезаем по ней с небольшим и неровным запасом."""
    ys, xs = np.nonzero(mask > 0.15)
    if len(xs) < 10:
        return mask
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    margin = rng.uniform(-0.03, 0.12, 4) * max(x1 - x0, y1 - y0)
    height, width = mask.shape
    x0, y0 = int(max(0, x0 - margin[0])), int(max(0, y0 - margin[1]))
    x1, y1 = int(min(width, x1 + margin[2])), int(min(height, y1 + margin[3]))
    return mask[y0:y1, x0:x1] if x1 - x0 > 4 and y1 - y0 > 4 else mask


def light_aug(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Слабая аугментация чистого вида: эталон в галерее остаётся эталоном."""
    height, width = image_bgr.shape[:2]
    if rng.random() < 0.5:
        scale = rng.uniform(0.6, 1.0)
        image_bgr = cv2.resize(image_bgr, (max(8, int(width * scale)), max(8, int(height * scale))), interpolation=cv2.INTER_AREA)
    return image_bgr
