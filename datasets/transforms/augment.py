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
