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
from src.preprocessing.image import enhance_stamp_contrast, fit_square, read_bgr
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
    factory: str
    factory_id: int
    book: str
    page: str
    catalog_no: str


@dataclass(frozen=True)
class RealSample:
    path: Path
    box: tuple[float, float, float, float]
    file_name: str


def load_books(meta_csv: Path | None = None, books_dir: Path | None = None) -> list[BookSample]:
    """Книжные кропы. Пустая фабрика получает id -1 и не входит в метрику разнообразия."""
    if meta_csv is None or books_dir is None:
        discovered_books, discovered_meta = resolve_reference(None, None)
        books_dir = books_dir or discovered_books
        meta_csv = meta_csv or discovered_meta

    names: list[str] = []
    name_to_id: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    with meta_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_path" not in reader.fieldnames:
            raise ValueError(f"В {meta_csv} нет колонки image_path")
        rows = list(reader)

    for row in rows:
        factory = (row.get("factory") or "").strip()
        if factory and factory not in name_to_id:
            name_to_id[factory] = len(names)
            names.append(factory)

    samples: list[BookSample] = []
    for row in rows:
        relative = Path((row.get("image_path") or "").replace("\\", "/").strip())
        if relative.as_posix() in {"", "."}:
            continue
        factory = (row.get("factory") or "").strip()
        samples.append(
            BookSample(
                mark_id=(row.get("mark_id") or "").strip(),
                path=books_dir / relative,
                factory=factory,
                factory_id=name_to_id[factory] if factory else -1,
                book=(row.get("book") or "").strip(),
                page=(row.get("page") or "").strip(),
                catalog_no=(row.get("catalog_no") or "").strip(),
            )
        )
    if not samples:
        raise ValueError(f"В {meta_csv} нет клейм")
    return samples


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


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def load_detected_crops(photos_dir: Path, weights: Path, conf: float = 0.75) -> list[RealSample]:
    """Кропы полевых фото: боксы ставит YOLO, в обучение идёт каждый бокс выше порога."""
    from src.detection.detector import StampDetector
    from src.detection.yolo_processing import YOLOProcessing

    if not photos_dir.is_dir():
        raise FileNotFoundError(f"Папка фото не найдена: {photos_dir}")
    if not weights.is_file():
        raise FileNotFoundError(f"Веса детектора не найдены: {weights}")
    if not 0.0 < conf < 1.0:
        raise ValueError(f"Порог детектора должен быть в (0, 1), получено {conf}")

    images = sorted(
        path
        for path in photos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"В {photos_dir} нет изображений")

    detector = StampDetector(weights, YOLOProcessing(conf_threshold=conf))
    samples: list[RealSample] = []
    empty = 0
    for path in images:
        boxes, _scores = detector.predict(read_bgr(path))
        if len(boxes) == 0:
            empty += 1
            continue
        relative = path.relative_to(photos_dir).as_posix()
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (float(value) for value in box)
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue
            name = relative if index == 0 else f"{relative}#{index}"
            samples.append(RealSample(path=path, box=(x1, y1, x2, y2), file_name=name))
    print(f"YOLO: {len(samples)} кропов из {len(images)} фото, без клейма {empty}")
    if len(samples) < 2:
        raise ValueError(f"Детектор нашёл меньше двух кропов в {photos_dir}")
    return samples


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_book_pil(path: Path) -> Image.Image:
    square = enhance_stamp_contrast(fit_square(read_bgr(path), INPUT_SIZE))
    return _to_pil(square)


def load_real_pil(sample: RealSample) -> Image.Image:
    image = read_bgr(sample.path)
    crop = crop_stamp(image, sample.box, target_size=INPUT_SIZE)
    return _to_pil(enhance_stamp_contrast(crop))


def square_tensor(path: Path) -> torch.Tensor:
    """Книжный кроп: квадрат, CLAHE и резкость, RGB 0..1. Ч/б делает модель."""
    square = fit_square(read_bgr(path), INPUT_SIZE)
    enhanced = enhance_stamp_contrast(square)
    rgb = enhanced[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1)))


class GaussianNoise:
    """Лёгкий шум. Вторая обрезка кропа сюда не входит."""

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)


def book_transform() -> transforms.Compose:
    """Поворот, лёгкая гомография, цвет, квадратный кроп, размытие и шум.

    Окно кропа квадратное: после ресайза к 224 пропорции клейма не тянутся.
    Гомография слабая, только чтобы страница была снята не строго сверху.
    """
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=(0, 180), fill=114),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=114),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4)],
                p=0.8,
            ),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomResizedCrop(
                INPUT_SIZE,
                scale=(0.75, 1.0),
                ratio=(1.0, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))],
                p=0.4,
            ),
            transforms.ToTensor(),
            transforms.RandomApply([GaussianNoise(std=0.02)], p=0.5),
        ]
    )


def real_transform() -> transforms.Compose:
    """Поворот тарелки, лёгкая гомография, цвет, шум и слабое размытие.

    Случайного кропа нет: полевой бокс и так часто срезает клеймо.
    Гомография не меняет пропорции сильно: distortion_scale 0.15.
    """
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=(0, 360), fill=114),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4, fill=114),
            transforms.ColorJitter(brightness=0.45, contrast=0.45),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                p=0.3,
            ),
            transforms.ToTensor(),
            GaussianNoise(std=0.02),
        ]
    )


def hard_negative_batches(
    embeddings: torch.Tensor,
    batch_size: int,
    neighbors: int,
    generator: torch.Generator,
) -> list[list[int]]:
    """Батчи, где клеймо лежит рядом со своими ближайшими чужими.

    Берём случайное ещё не использованное клеймо и добавляем к нему
    neighbors ближайших, тоже не использованных. ArcFace в одном батче
    видит именно те пары, которые путает поиск.
    """
    count = embeddings.shape[0]
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    similarity = normalized @ normalized.T
    similarity.fill_diagonal_(-2.0)
    ranked = torch.argsort(similarity, dim=1, descending=True)

    unused = torch.ones(count, dtype=torch.bool)
    order = torch.randperm(count, generator=generator).tolist()
    batches: list[list[int]] = []
    current: list[int] = []
    for seed in order:
        if not unused[seed]:
            continue
        group = [seed]
        unused[seed] = False
        for neighbor in ranked[seed].tolist():
            if len(group) > neighbors:
                break
            if unused[neighbor]:
                group.append(neighbor)
                unused[neighbor] = False
        current.extend(group)
        if len(current) >= batch_size:
            batches.append(current[:batch_size])
            current = current[batch_size:]
    if len(current) >= 2:
        batches.append(current)
    return batches


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
