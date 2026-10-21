import random

import numpy as np
import torch
import torch.nn.functional as F


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


class Rotate90:
    # Поворот против часовой стрелки на угол из angles. При 90 и 270 ширина и высота меняются местами.
    def __init__(self, p=0.5, angles=(90, 180, 270)):
        angles = tuple(int(angle) for angle in angles)
        if not angles or any(angle not in (90, 180, 270) for angle in angles):
            raise ValueError(f"Rotate90: углы только из 90, 180, 270, получено {angles}")
        self.p = p
        self.angles = angles

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        angle = random.choice(self.angles)
        height, width = image.shape[:2]
        image = np.ascontiguousarray(np.rot90(image, k=angle // 90))
        boxes = np.array(target["boxes"], dtype=np.float32, copy=True)
        x1, y1, x2, y2 = (boxes[:, i].copy() for i in range(4))
        if angle == 90:
            boxes[:, 0] = y1
            boxes[:, 1] = width - x2
            boxes[:, 2] = y2
            boxes[:, 3] = width - x1
        elif angle == 180:
            boxes[:, 0] = width - x2
            boxes[:, 1] = height - y2
            boxes[:, 2] = width - x1
            boxes[:, 3] = height - y1
        else:
            boxes[:, 0] = height - y2
            boxes[:, 1] = x1
            boxes[:, 2] = height - y1
            boxes[:, 3] = x2
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


class Defocus:
    def __init__(self, p=0.5, radius=(1.0, 3.0)):
        if isinstance(radius, (int, float)):
            radius = (radius, radius)
        low, high = (float(value) for value in radius)
        if not 0 <= low <= high:
            raise ValueError(f"Defocus: нужно 0 <= low <= high, получено {radius}")
        self.p = p
        self.radius = (low, high)

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        return _defocus(image, random.uniform(*self.radius)), target


def _contrast(image, factor):
    """pixel' = mean + factor * (pixel - mean), mean — средняя яркость кадра в сером."""
    gray = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114
    # standart weights to transform coloured picture to gray from ITU-R BT.601 standart
    mean = float(gray.mean())
    out = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _defocus(image, radius):
    if radius <= 0:
        return image
    kernel = _disk_kernel(radius)
    size = kernel.shape[0]
    pad = size // 2
    channels = image.shape[2]
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    weight = torch.from_numpy(kernel)[None, None].repeat(channels, 1, 1, 1)
    out = F.conv2d(x, weight, groups=channels)
    return out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()


def _disk_kernel(radius, supersample=8):
    half = int(np.ceil(radius))
    n = (2 * half + 1) * supersample
    coords = (np.arange(n) + 0.5) / supersample - (half + 0.5)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    inside = (xx ** 2 + yy ** 2 <= radius ** 2).astype(np.float32)
    kernel = inside.reshape(2 * half + 1, supersample, 2 * half + 1, supersample).mean(axis=(1, 3))
    total = kernel.sum()
    if total == 0:
        kernel[half, half] = 1.0
        return kernel
    return kernel / total
