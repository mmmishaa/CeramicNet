"""Эталоны и реальные кропы, предобработка как в CeramicPipeline, оценка на реальных фото."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import DATA_DIR
from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import enhance_stamp_contrast, read_bgr
from src.retrieval.books import load_book_records, resolve_reference

SIZE = 224
DEFAULT_REAL_COCO = DATA_DIR / "stamps_coco_merged"
LABELS_CSV = Path(__file__).with_name("real_factory_labels.csv")

# Метка реального кропа -> правило на поле factory эталона.
FACTORY_RULES = {
    "kuz": lambda f: "кузнец" in f,
    "kuz_ms": lambda f: "кузнец" in f and bool(re.search(r"м\.\s?с\.|м\. кузнец|товарищ", f)),
    "kuz_ie": lambda f: "кузнец" in f and bool(re.search(r"и\.\s?е\.|волхов", f)),
    "kornilov": lambda f: "корнилов" in f,
    "fomin": lambda f: "фомин" in f,
    "gardner": lambda f: "гарднер" in f,
    "zinoviev": lambda f: "зиновьев" in f,
    "khrapunov": lambda f: "храпун" in f,
    "emelyanov": lambda f: "емельян" in f,
    "kalinin": lambda f: "калинин" in f,
    "pervomay": lambda f: "первомай" in f,
    "dmitrov": lambda f: "дмитров" in f,
    "dulevo": lambda f: "дулев" in f,
    "baranovka": lambda f: "барановк" in f,
}


@dataclass(frozen=True)
class Mark:
    mark_id: str
    path: Path
    factory: str
    book: str


@dataclass(frozen=True)
class RealCrop:
    ann_id: int
    path: Path
    box: tuple[float, float, float, float]
    item: str
    label: str


def load_marks(books_dir=None, meta_csv=None) -> list[Mark]:
    """По умолчанию — эталоны из retrieval_config.yaml, те же, что попадут в индекс."""
    books_dir, meta_csv = resolve_reference(books_dir, meta_csv)
    return [
        Mark(record["mark_id"], books_dir / record["image_path"], record["factory"], record["book"])
        for record in load_book_records(meta_csv, books_dir)
        if (books_dir / record["image_path"]).is_file()
    ]


def load_real(coco_dir=None) -> list[RealCrop]:
    """Размеченные по фабрике кропы реальных фото из COCO. Нет папки — пустой список, оценка пропускается."""
    coco_dir = Path(coco_dir or DEFAULT_REAL_COCO)
    annotation = coco_dir / "annotations" / "instances_default.json"
    if not annotation.is_file():
        return []
    raw = json.loads(annotation.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in raw["images"]}
    with LABELS_CSV.open(encoding="utf-8") as handle:
        labels = {int(row["ann_id"]): row["label"] for row in csv.DictReader(handle)}
    crops = []
    for ann in raw["annotations"]:
        label = labels.get(int(ann["id"]))
        if not label:
            continue
        name = images[ann["image_id"]]["file_name"]
        x, y, w, h = ann["bbox"]
        # Шифр предмета без номера снимка: по нему делим, чтобы снимки одного предмета не разошлись.
        item = Path(name).stem.rsplit("_", 1)[0]
        crops.append(RealCrop(int(ann["id"]), coco_dir / "images" / "default" / name, (x, y, x + w, y + h), item, label))
    return crops


def prepare(image_bgr: np.ndarray, box=None) -> np.ndarray:
    """Как в пайплайне и scripts.build_index: квадратный кроп с полями 10%, 224, CLAHE и резкость. Выход BGR uint8."""
    if box is None:
        height, width = image_bgr.shape[:2]
        box = (0, 0, width, height)
    return enhance_stamp_contrast(crop_stamp(image_bgr, box, target_size=SIZE))


def to_tensor(prepared_bgr: np.ndarray) -> torch.Tensor:
    """uint8 CHW: галерея целиком в float32 заняла бы почти 2 ГБ."""
    rgb = np.ascontiguousarray(prepared_bgr[:, :, ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


@torch.no_grad()
def embed(model, tensors: torch.Tensor, device, batch: int = 64) -> torch.Tensor:
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(tensors), batch):
        chunk = tensors[start : start + batch].to(device).float().div(255.0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            vectors = model(chunk)
        out.append(F.normalize(vectors.float(), dim=-1).cpu())
    model.train(was_training)
    return torch.cat(out)


class GalleryTensors:
    def __init__(self, marks: list[Mark]):
        self.marks = marks
        self.tensors = torch.stack([to_tensor(prepare(read_bgr(mark.path))) for mark in marks])


class RealTensors:
    def __init__(self, crops: list[RealCrop]):
        self.crops = crops
        self.tensors = torch.stack([to_tensor(prepare(read_bgr(crop.path), crop.box)) for crop in crops]) if crops else None


def eval_real(model, gallery: GalleryTensors, real: RealTensors, device, ks=(1, 3, 5, 10), gallery_vectors=None) -> dict:
    """hit@k: в top-k есть эталон той же фабрики. macro — каждая фабрика с одним весом, иначе Кузнецов М.С. забивает всё."""
    if not real.crops:
        return {}
    if gallery_vectors is None:
        gallery_vectors = embed(model, gallery.tensors, device)
    factories = [mark.factory.lower() for mark in gallery.marks]
    match = {label: torch.tensor([rule(f) for f in factories]) for label, rule in FACTORY_RULES.items()}
    labels = [crop.label for crop in real.crops]
    top = (embed(model, real.tensors, device) @ gallery_vectors.T).topk(max(ks), dim=1).indices
    result = {}
    for k in ks:
        hits = [bool(match[label][top[i, :k]].any()) for i, label in enumerate(labels)]
        per_label: dict[str, list[bool]] = {}
        for label, hit in zip(labels, hits):
            per_label.setdefault(label, []).append(hit)
        result[f"real_hit@{k}"] = float(np.mean(hits))
        result[f"real_macro@{k}"] = float(np.mean([np.mean(v) for v in per_label.values()]))
    return result


def chance_level(gallery: GalleryTensors, real: RealTensors, k: int) -> float:
    """Вероятность попасть случайной выдачей top-k: ориентир для hit@k."""
    factories = [mark.factory.lower() for mark in gallery.marks]
    probs = []
    for crop in real.crops:
        p = np.mean([FACTORY_RULES[crop.label](f) for f in factories])
        probs.append(1 - (1 - p) ** k)
    return float(np.mean(probs))
