import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TestCeramicDataset(Dataset):
    """Универсальный тестовый датасет для бенчмаркинга моделей детекции.

    Возвращает сырой numpy-кадр и Ground Truth боксы [x1, y1, x2, y2]
    в оригинальных пикселях снимка.
    """

    def __init__(
        self,
        root_dir: str | Path,
        negative_dirs: list[str | Path] | tuple[str | Path, ...] | None = None,
        negative_ratio: float | None = 0.2,
        as_bgr: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.as_bgr = as_bgr
        self.negative_dirs = tuple(Path(path) for path in (negative_dirs or ()))
        self.negative_ratio = float(negative_ratio) if negative_ratio is not None else None

        ann_path = self.root_dir / "annotations" / "instances_default.json"
        image_root = self.root_dir / "images" / "default"

        if not ann_path.is_file():
            raise FileNotFoundError(f"Файл аннотаций не найден: {ann_path}")

        raw = json.loads(ann_path.read_text(encoding="utf-8"))
        self.categories = {int(cat["id"]): str(cat["name"]) for cat in raw["categories"]}
        paths = self._index_pngs(image_root)

        grouped = defaultdict(list)
        for ann in raw["annotations"]:
            grouped[int(ann["image_id"])].append(ann)

        self.items = []
        missing = []
        for image_info in sorted(raw["images"], key=lambda item: int(item["id"])):
            file_name = unicodedata.normalize("NFC", str(image_info["file_name"]).replace("\\", "/"))
            cipher, shot = self._cipher_and_shot(file_name)
            path = paths.get(file_name)
            if path is None:
                missing.append(file_name)
                continue

            boxes, labels = self._boxes_xyxy(
                grouped.get(int(image_info["id"]), []),
                int(image_info["width"]),
                int(image_info["height"]),
            )

            self.items.append(
                {
                    "path": path,
                    "file_name": file_name,
                    "image_id": int(image_info["id"]),
                    "width": int(image_info["width"]),
                    "height": int(image_info["height"]),
                    "cipher": cipher,
                    "shot": shot,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        if missing:
            raise FileNotFoundError(f"В {image_root} отсутствуют кадры, например: {missing[0]}")

        self._add_negatives()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        item = self.items[idx]

        # Чтение кадра в исходном разрешении
        image = cv2.imread(str(item["path"]))
        if image is None:
            with Image.open(item["path"]) as pil_img:
                image = np.array(pil_img.convert("RGB"), dtype=np.uint8)
                if self.as_bgr:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif not self.as_bgr:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target = {
            "boxes": item["boxes"].copy(),
            "labels": item["labels"].copy(),
            "image_id": item["image_id"],
            "file_name": item["file_name"],
            "cipher": item["cipher"],
            "shot": item["shot"],
            "orig_shape": (item["height"], item["width"]),
        }

        return image, target

    def _boxes_xyxy(self, anns: list[dict], width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        boxes, labels = [], []
        for ann in sorted(anns, key=lambda item: int(item["id"])):
            x, y, w, h = (float(v) for v in ann["bbox"])
            x1 = max(0.0, min(x, float(width)))
            y1 = max(0.0, min(y, float(height)))
            x2 = max(0.0, min(x + w, float(width)))
            y2 = max(0.0, min(y + h, float(height)))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))

        if not boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

        return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def _add_negatives(self) -> None:
        n_root = len(self.items)
        known = {Path(item["file_name"]).name for item in self.items}
        next_id = max((item["image_id"] for item in self.items), default=0) + 1

        paths = []
        for folder in self.negative_dirs:
            if not folder.is_dir():
                raise FileNotFoundError(f"Папка с негативами не найдена: {folder}")
            paths.extend(
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() == ".png"
            )

        paths.sort(key=lambda p: unicodedata.normalize("NFC", str(p)))
        chosen = []
        for path in paths:
            name = unicodedata.normalize("NFC", path.name)
            if name in known:
                continue
            known.add(name)
            chosen.append(path)

        if self.negative_ratio is not None and chosen:
            limit = round(n_root * self.negative_ratio)
            random.Random(0).shuffle(chosen)
            chosen = chosen[:limit]

        for path in chosen:
            name = unicodedata.normalize("NFC", path.name)
            cipher, shot = self._cipher_and_shot(name)
            with Image.open(path) as img:
                w, h = img.size

            self.items.append(
                {
                    "path": path,
                    "file_name": f"{unicodedata.normalize('NFC', path.parent.name)}/{name}",
                    "image_id": next_id,
                    "width": w,
                    "height": h,
                    "cipher": cipher,
                    "shot": shot,
                    "boxes": np.empty((0, 4), dtype=np.float32),
                    "labels": np.empty((0,), dtype=np.int64),
                }
            )
            next_id += 1

    @staticmethod
    def _index_pngs(image_root: Path) -> dict[str, Path]:
        found = {}
        for path in Path(image_root).rglob("*"):
            if not path.is_file() or path.suffix.lower() != ".png":
                continue
            key = unicodedata.normalize("NFC", path.relative_to(image_root).as_posix())
            if key in found:
                raise ValueError(f"Дублирующееся NFC имя: {key}")
            found[key] = path
        return found

    @staticmethod
    def _cipher_and_shot(file_name: str) -> tuple[str, int]:
        stem = Path(unicodedata.normalize("NFC", Path(file_name).name)).stem
        cipher, separator, shot = stem.rpartition("_")
        if separator == "" or cipher == "" or not shot.isdigit():
            raise ValueError(f"Имя кадра не в формате ШИФР_N.png: {file_name}")
        return cipher, int(shot)