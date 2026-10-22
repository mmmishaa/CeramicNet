import json
import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR
_MODELS = ("default", "faster_rcnn", "rtdetr", "rtmdet")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class StampDataset(Dataset):
    """Кадры с боксами клейм из COCO.

    transform(image, target) -> (image, target) вызывается на uint8 HWC.
    target — словарь с numpy-полями boxes (N, 4) xyxy и labels (N,).
    standard_size=(H, W) вписывает кадр в этот размер с белыми полями
    и так же сдвигает боксы. None оставляет исходный размер.
    model выбирает запись боксов на выходе: default, faster_rcnn, rtdetr, rtmdet.
    negative_dirs — папки кадров без клейма. У такого кадра боксов нет.
    В COCO боксов тоже может не быть: кадр остаётся, boxes пустой.
    negative_ratio ограничивает внешние негативы долей от числа кадров COCO;
    None берёт все доступные файлы, 0 отключает добавление. negative_seed задаёт
    воспроизводимый отбор (42 по умолчанию, 0 — прежний отбор ветки datasets).
    ann_file и image_root позволяют читать общий train.json/val.json без
    копирования изображений. При None используется прежняя структура root_dir.
    model="default" сохраняет COCO ID; у Faster R-CNN метки начинаются с 1,
    у RT-DETR/RTMDet — с 0, независимо от исходных ID категорий.
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        standard_size=None,
        model="default",
        negative_dirs=None,
        negative_ratio=0.2,
        ann_file=None,
        image_root=None,
        negative_seed=42,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if model not in _MODELS:
            names = ", ".join(_MODELS)
            raise ValueError(f"model должен быть одним из: {names}. Получено {model!r}")
        self.model = model
        self.negative_dirs = tuple(Path(path) for path in (negative_dirs or ()))
        self.negative_seed = int(negative_seed)
        if negative_ratio is None:
            self.negative_ratio = None
        else:
            negative_ratio = float(negative_ratio)
            if not math.isfinite(negative_ratio) or negative_ratio < 0:
                raise ValueError(
                    f"negative_ratio — доля от числа кадров в root, получено {negative_ratio}"
                )
            self.negative_ratio = negative_ratio
        if standard_size is None:
            self.standard_size = None
        else:
            self.standard_size = (int(standard_size[0]), int(standard_size[1]))
            if self.standard_size[0] < 1 or self.standard_size[1] < 1:
                raise ValueError(f"standard_size должен быть (H, W) из положительных чисел, получено {standard_size}")

        ann_path = Path(ann_file) if ann_file is not None else self.root_dir / "annotations" / "instances_default.json"
        image_root = Path(image_root) if image_root is not None else self.root_dir / "images" / "default"
        self.ann_file, self.image_root = ann_path, image_root
        raw = json.loads(ann_path.read_text(encoding="utf-8"))
        self.categories = {int(cat["id"]): str(cat["name"])
                           for cat in sorted(raw["categories"], key=lambda c: int(c["id"]))}
        category_ids = list(self.categories)
        if self.model == "default":
            self.category_to_label = {category_id: category_id for category_id in category_ids}
        else:
            offset = 1 if self.model == "faster_rcnn" else 0
            self.category_to_label = {category_id: i + offset for i, category_id in enumerate(category_ids)}
        self.label_to_category = {label: category_id for category_id, label in self.category_to_label.items()}
        paths = _index_images(image_root)

        grouped = defaultdict(list)
        for ann in raw["annotations"]:
            grouped[int(ann["image_id"])].append(ann)

        self.items = []
        missing = []
        for image in sorted(raw["images"], key=lambda item: int(item["id"])):
            file_name = unicodedata.normalize("NFC", str(image["file_name"]).replace("\\", "/"))
            cipher, shot = _cipher_and_shot(file_name)
            path = paths.get(file_name)
            if path is None:
                missing.append(file_name)
                continue
            boxes, labels = _boxes_xyxy(
                grouped.get(int(image["id"]), []),
                int(image["width"]),
                int(image["height"]),
            )
            labels = np.asarray([self.category_to_label[int(label)] for label in labels], dtype=np.int64)
            self.items.append({
                "path": path,
                "file_name": file_name,
                "image_id": int(image["id"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "cipher": cipher,
                "shot": shot,
                "boxes": boxes,
                "labels": labels,
            })

        if missing:
            raise FileNotFoundError(f"в {image_root} нет кадров из COCO, например {missing[0]}")
        self._add_negatives()

    def __len__(self):
        return len(self.items)

    def load_raw(self, idx):
        """Shared RGB uint8 HWC + numpy XYXY target, before model-specific formatting.

        Training augmentations and optional letterbox are applied once here.
        Native integrations use this method to avoid tensor/PIL round trips.
        """
        item = self.items[idx]
        with Image.open(item["path"]) as image:
            array = np.array(image.convert("RGB"), copy=True)
        if array.shape[0] != item["height"] or array.shape[1] != item["width"]:
            raise ValueError(
                f"{item['file_name']}: на диске {array.shape[1]}x{array.shape[0]}, "
                f"в индексе {item['width']}x{item['height']}"
            )

        target = {"boxes": item["boxes"].copy(), "labels": item["labels"].copy()}
        if self.transform is not None:
            array, target = self.transform(array, target)

        boxes = np.asarray(target["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(target["labels"], dtype=np.int64).reshape(-1)
        if len(boxes) != len(labels):
            raise ValueError(f"{item['file_name']}: boxes и labels разной длины")

        scale, pad = (1.0, 1.0), (0, 0)
        if self.standard_size is not None:
            array, boxes, scale, pad = _letterbox(array, boxes, self.standard_size)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": item["image_id"],
            "cipher": item["cipher"],
            "file_name": item["file_name"],
            "orig_size": (item["height"], item["width"]),
            "scale": scale,
            "pad": pad,
        }
        return array, target

    def __getitem__(self, idx):
        array, target = self.load_raw(idx)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).float().div_(255)
        target = {**target,
                  "boxes": torch.from_numpy(np.ascontiguousarray(target["boxes"])).float(),
                  "labels": torch.from_numpy(np.ascontiguousarray(target["labels"])).long()}
        return tensor, self._formatter(tensor, target)

    def _formatter(self, image, target):
        """Форматирует данные в соответствии с выбранной моделью."""
        if self.model == "default":
            return target
        back = {key: target[key] for key in ("orig_size", "scale", "pad")}
        if self.model == "faster_rcnn":
            return {
                "boxes": target["boxes"],
                "labels": target["labels"],
                "image_id": target["image_id"],
                **back,
            }
        if self.model == "rtmdet":
            return {
                "boxes": target["boxes"],
                "labels": target["labels"],
                "image_id": target["image_id"],
                **back,
            }
        height, width = image.shape[-2:]
        return {
            "boxes": _xyxy_to_cxcywh(target["boxes"], height, width),
            "labels": target["labels"],
            "image_id": target["image_id"],
            "size": (int(height), int(width)),
            **back,
        }

    def _add_negatives(self):
        n_root = len(self.items)
        known = {Path(item["file_name"]).name for item in self.items}
        next_id = max((item["image_id"] for item in self.items), default=0) + 1
        paths = []
        for folder in self.negative_dirs:
            if not folder.is_dir():
                raise FileNotFoundError(f"нет папки без клейм: {folder}")
            paths.extend(
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            )
        paths.sort(key=lambda path: unicodedata.normalize("NFC", str(path)))
        chosen = []
        for path in paths:
            name = unicodedata.normalize("NFC", path.name)
            if name in known:
                continue
            known.add(name)
            chosen.append(path)
        if self.negative_ratio is not None:
            limit = round(n_root * self.negative_ratio)
            random.Random(self.negative_seed).shuffle(chosen)
            chosen = chosen[:limit]
        for path in chosen:
            name = unicodedata.normalize("NFC", path.name)
            cipher, shot = _cipher_and_shot(name)
            with Image.open(path) as image:
                width, height = image.size
            self.items.append({
                "path": path,
                "file_name": f"{unicodedata.normalize('NFC', path.parent.name)}/{name}",
                "image_id": next_id,
                "width": width,
                "height": height,
                "cipher": cipher,
                "shot": shot,
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
            })
            next_id += 1


def get_collate_fn(model="default"):
    """Функция для DataLoader(collate_fn=...). Собирает уже отформатированные кадры в батч."""
    if model not in _MODELS:
        names = ", ".join(_MODELS)
        raise ValueError(f"model должен быть одним из: {names}. Получено {model!r}")
    if model == "rtdetr":
        return _collate_rtdetr
    return _collate_list

def _collate_list(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _collate_rtdetr(batch):
    images, targets = zip(*batch)
    if len({tuple(image.shape) for image in images}) != 1:
        raise ValueError("для model='rtdetr' нужен standard_size: кадры разной высоты нельзя сложить в один тензор")
    labels = []
    for target in targets:
        boxes = target["boxes"].clamp(0, 1)
        if len(boxes):
            keep = (boxes[:, 2] > 1e-4) & (boxes[:, 3] > 1e-4)
            boxes = boxes[keep]
            class_labels = target["labels"][keep]
        else:
            class_labels = target["labels"]
        labels.append({"class_labels": class_labels.long(), "boxes": boxes})
    return torch.stack(list(images)), labels


def boxes_to_original(boxes, target):
    """Выполняет обратное _letterbox: возвращает боксы в координаты исходного кадра."""
    scale_h, scale_w = target["scale"]
    top, left = target["pad"]
    height, width = target["orig_size"]
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4).clone()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / scale_w).clamp(0, width)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / scale_h).clamp(0, height)
    return boxes


def _index_images(image_root):
    found = {}
    for path in Path(image_root).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        key = unicodedata.normalize("NFC", path.relative_to(image_root).as_posix())
        if key in found:
            raise ValueError(f"два файла дают одно NFC-имя: {key}")
        found[key] = path
    return found


def _cipher_and_shot(file_name):
    stem = Path(unicodedata.normalize("NFC", Path(file_name).name)).stem
    cipher, separator, shot = stem.rpartition("_")
    if separator == "" or cipher == "" or not shot.isdigit():
        raise ValueError(f"имя кадра не в формате ШИФР_N.png: {file_name}")
    return cipher, int(shot)


def _boxes_xyxy(anns, width, height):
    boxes, labels = [], []
    for ann in sorted(anns, key=lambda item: int(item["id"])):
        x, y, w, h = (float(value) for value in ann["bbox"])
        x1 = min(max(0.0, x), float(width))
        y1 = min(max(0.0, y), float(height))
        x2 = min(max(0.0, x + w), float(width))
        y2 = min(max(0.0, y + h), float(height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"пустой бокс ann id {ann['id']}")
        boxes.append((x1, y1, x2, y2))
        labels.append(int(ann["category_id"]))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _xyxy_to_cxcywh(boxes, height, width):
    if boxes.numel() == 0:
        return boxes
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack((
        (x1 + x2) / 2 / width,
        (y1 + y2) / 2 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ), dim=1)


def _letterbox(image, boxes, size):
    target_h, target_w = size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w = min(target_w, max(1, int(round(width * scale))))
    new_h = min(target_h, max(1, int(round(height * scale))))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), _RESAMPLE), dtype=np.uint8)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    scale_h, scale_w = new_h / height, new_w / width
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h + top
    return canvas, boxes, (scale_h, scale_w), (top, left)
