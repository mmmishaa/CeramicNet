import random

import numpy as np


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        width = image.shape[1]
        image = np.ascontiguousarray(np.flip(image, axis=1))
        boxes = np.array(target["boxes"], dtype=np.float32, copy=True)
        x1 = boxes[:, 0].copy()
        boxes[:, 0] = width - boxes[:, 2]
        boxes[:, 2] = width - x1
        return image, {**target, "boxes": boxes}


class VerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        height = image.shape[0]
        image = np.ascontiguousarray(np.flip(image, axis=0))
        boxes = np.array(target["boxes"], dtype=np.float32, copy=True)
        y1 = boxes[:, 1].copy()
        boxes[:, 1] = height - boxes[:, 3]
        boxes[:, 3] = height - y1
        return image, {**target, "boxes": boxes}


class LowContrast:
    """Снижает контраст. factor берётся случайно из [low, high], оба от 0 до 1.

    factor = 1 — кадр без изменений, factor = 0 — сплошной серый.
    """

    def __init__(self, p=0.5, factor=(0.5, 0.8)):
        low, high = (float(value) for value in factor)
        if not 0 <= low <= high <= 1:
            raise ValueError(f"LowContrast: нужно 0 <= low <= high <= 1, получено {factor}")
        self.p = p
        self.factor = (low, high)

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        return _contrast(image, random.uniform(*self.factor)), target


class HighContrast:
    """Повышает контраст. factor берётся случайно из [low, high], оба не меньше 1.

    factor = 1 — кадр без изменений. Чем больше factor, тем больше пикселей
    упирается в 0 или 255.
    """

    def __init__(self, p=0.5, factor=(1.2, 1.5)):
        low, high = (float(value) for value in factor)
        if not 1 <= low <= high:
            raise ValueError(f"HighContrast: нужно 1 <= low <= high, получено {factor}")
        self.p = p
        self.factor = (low, high)

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        return _contrast(image, random.uniform(*self.factor)), target


def _contrast(image, factor):
    """pixel' = mean + factor * (pixel - mean), mean — средняя яркость кадра в сером."""
    gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    # standart weights to transform coloured picture to gray from ITU-R BT.601 standart
    mean = float(gray.mean())
    out = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
