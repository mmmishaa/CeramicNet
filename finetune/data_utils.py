"""Чтение книг и реальных кропов, ч/б по BT.601 и аугментации."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from src.paths import ROOT_DIR
from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import fit_square, read_bgr
from src.retrieval.books import resolve_reference

INPUT_SIZE = 224


class GrayscaleBT601(nn.Module):
    """Перевод RGB в яркость ITU-R BT.601 и обратно в 3 одинаковых канала.

    Формула применяется до энкодера и в обучении, и в индексе, и в запросе:
    Y = 0.299 R + 0.587 G + 0.114 B.
    """

    def __init__(self) -> None:
        super().__init__()
        weight = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("weight", weight)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Ожидался тензор BCHW с 3 каналами, получено {tuple(images.shape)}")
        luminance = (images * self.weight).sum(dim=1, keepdim=True)
        return luminance.expand(-1, 3, -1, -1).contiguous()


@dataclass(frozen=True)
class BookSample:
    mark_id: str
    path: Path
    factory_norm: str
    factory_id: int
    book: str
    page: str
    catalog_no: str


@dataclass(frozen=True)
class RealSample:
    path: Path
    box: tuple[float, float, float, float]
    file_name: str


def load_books(meta_csv: Path | None = None, books_dir: Path | None = None) -> tuple[list[BookSample], list[str]]:
    """Книжные кропы и словарь фабрик. Пустой factory_norm получает id -1 и не учит учителя."""
    if meta_csv is None or books_dir is None:
        discovered_books, discovered_meta = resolve_reference(None, None)
        books_dir = books_dir or discovered_books
        meta_csv = meta_csv or discovered_meta

    names: list[str] = []
    name_to_id: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    with meta_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "factory_norm" not in reader.fieldnames:
            raise ValueError(f"В {meta_csv} нет колонки factory_norm")
        rows = list(reader)

    for row in rows:
        factory = (row.get("factory_norm") or "").strip()
        if factory and factory not in name_to_id:
            name_to_id[factory] = len(names)
            names.append(factory)

    samples: list[BookSample] = []
    for row in rows:
        relative = Path((row.get("image_path") or "").replace("\\", "/").strip())
        if relative.as_posix() in {"", "."}:
            continue
        factory = (row.get("factory_norm") or "").strip()
        samples.append(
            BookSample(
                mark_id=(row.get("mark_id") or "").strip(),
                path=books_dir / relative,
                factory_norm=factory,
                factory_id=name_to_id[factory] if factory else -1,
                book=(row.get("book") or "").strip(),
                page=(row.get("page") or "").strip(),
                catalog_no=(row.get("catalog_no") or "").strip(),
            )
        )
    if not samples:
        raise ValueError(f"В {meta_csv} нет клейм")
    return samples, names


def default_coco_dir() -> Path:
    candidate = ROOT_DIR.parent / "stamps_coco_merged"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Не найдена папка stamps_coco_merged рядом с репозиторием")


def load_real_crops(coco_dir: Path | None = None) -> list[RealSample]:
    """Реальные кропы по COCO-боксам. file_name лежит в images/default/."""
    coco_dir = coco_dir or default_coco_dir()
    annotation = coco_dir / "annotations" / "instances_default.json"
    images_root = coco_dir / "images" / "default"
    raw = json.loads(annotation.read_text(encoding="utf-8"))
    by_id = {int(image["id"]): image for image in raw["images"]}
    samples: list[RealSample] = []
    for ann in raw["annotations"]:
        image = by_id.get(int(ann["image_id"]))
        if image is None:
            continue
        file_name = str(image["file_name"]).replace("\\", "/")
        path = images_root / file_name
        if not path.is_file():
            continue
        x, y, width, height = (float(value) for value in ann["bbox"])
        if width <= 1 or height <= 1:
            continue
        samples.append(
            RealSample(
                path=path,
                box=(x, y, x + width, y + height),
                file_name=file_name,
            )
        )
    if not samples:
        raise ValueError(f"В {annotation} нет пригодных кропов")
    return samples


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_book_pil(path: Path) -> Image.Image:
    return _to_pil(read_bgr(path))


def load_real_pil(sample: RealSample) -> Image.Image:
    image = read_bgr(sample.path)
    crop = crop_stamp(image, sample.box, target_size=INPUT_SIZE)
    return _to_pil(crop)


def square_tensor(path: Path) -> torch.Tensor:
    """Книжный кроп без аугментации: квадрат 224, RGB в диапазоне 0..1. Ч/б делает модель."""
    square = fit_square(read_bgr(path), INPUT_SIZE)
    rgb = square[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


class GaussianNoise:
    """Лёгкий шум для реальных кропов. Сильное размытие и random crop сюда не входят."""

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


def book_transform() -> transforms.Compose:
    """Поворот 0..180° и обрезка вплоть до половины кадра. Это позитивная пара одного клейма."""
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=(0, 180), fill=114),
            transforms.RandomResizedCrop(
                INPUT_SIZE,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))],
                p=0.3,
            ),
            transforms.ToTensor(),
        ]
    )


def real_transform() -> transforms.Compose:
    """Мягкие аугментации реала. RandomResizedCrop и сильное размытие запрещены."""
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=15, fill=114),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            GaussianNoise(std=0.02),
        ]
    )


class BookPairDataset(Dataset):
    def __init__(self, samples: list[BookSample]) -> None:
        self.samples = samples
        self.transform = book_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = load_book_pil(self.samples[index].path)
        return self.transform(image), self.transform(image), index


class RealPairDataset(Dataset):
    def __init__(self, samples: list[RealSample]) -> None:
        self.samples = samples
        self.transform = real_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image = load_real_pil(self.samples[index])
        return self.transform(image), self.transform(image), index
