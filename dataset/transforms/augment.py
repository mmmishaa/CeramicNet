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


class Grayscale:
    """Переводит кадр в чёрно-белое. Три канала остаются, в каждом одна яркость."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        gray = np.clip(np.rint(_luma(image)), 0, 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2), target


class Brightness:
    """Меняет яркость. factor берётся случайно из [low, high].
    factor = 1 — кадр без изменений. Меньше 1 — темнее, больше 1 — светлее.
    """

    def __init__(self, p=0.5, factor=(0.6, 1.4)):
        low, high = (float(value) for value in factor)
        if not 0 <= low <= high:
            raise ValueError(f"Brightness: нужно 0 <= low <= high, получено {factor}")
        self.p = p
        self.factor = (low, high)

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        out = image.astype(np.float32) * random.uniform(*self.factor)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8), target


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


def _luma(image):
    """Яркость по ITU-R BT.601: 0.299 R + 0.587 G + 0.114 B."""
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def _contrast(image, factor):
    """pixel' = mean + factor * (pixel - mean), mean — средняя яркость кадра в сером."""
    mean = float(_luma(image).mean())
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
