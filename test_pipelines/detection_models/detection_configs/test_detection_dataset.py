from collections import defaultdict
import json
from pathlib import Path
import random
import unicodedata

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class TestDetectionDataset(Dataset):

    def __init__(
        self,
        images_dir: str | Path,
        ann_file: str | Path,
        negative_dirs: list[str | Path] | tuple[str | Path, ...] | None = None,
        negative_ratio: float | None = 0.2,
        as_bgr: bool = True,
        split: str | None = "val",
        val_ratio: float = 0.2,
        random_state: int = 42,
    ):
        self.images_root = Path(images_dir)
        self.ann_path = Path(ann_file)
        self.as_bgr = as_bgr
        self.negative_dirs = tuple(Path(p) for p in (negative_dirs or ()))
        self.negative_ratio = (
            float(negative_ratio) if negative_ratio is not None else None
        )

        if not self.ann_path.is_file():
            raise FileNotFoundError(f"Файл аннотаций не найден: {self.ann_path}")

        if not self.images_root.is_dir():
            raise FileNotFoundError(
                f"Папка с изображениями не найдена: {self.images_root}"
            )

        raw = json.loads(self.ann_path.read_text(encoding="utf-8"))
        self.categories = {
            int(cat["id"]): str(cat["name"]) for cat in raw["categories"]
        }
        paths = self._index_pngs(self.images_root)

        grouped = defaultdict(list)
        for ann in raw["annotations"]:
            grouped[int(ann["image_id"])].append(ann)

        all_items = []
        missing = []

        for img_info in sorted(raw["images"], key=lambda item: int(item["id"])):
            raw_file_name = str(img_info["file_name"]).replace("\\", "/")
            norm_file_name = unicodedata.normalize("NFC", raw_file_name)
            base_name = unicodedata.normalize(
                "NFC", Path(norm_file_name).name
            )

            path = (
                paths.get(norm_file_name)
                or paths.get(base_name)
                or paths.get(f"stamps/{base_name}")
            )

            if path is None:
                missing.append(norm_file_name)
                continue

            cipher, shot = self._cipher_and_shot(base_name)

            boxes, labels = self._boxes_xyxy(
                grouped.get(int(img_info["id"]), []),
                int(img_info["width"]),
                int(img_info["height"]),
            )

            all_items.append(
                {
                    "path": path,
                    "file_name": norm_file_name,
                    "image_id": int(img_info["id"]),
                    "width": int(img_info["width"]),
                    "height": int(img_info["height"]),
                    "cipher": cipher,
                    "shot": shot,
                    "boxes": boxes,
                    "labels": labels,
                }
            )

        if missing:
            raise FileNotFoundError(
                f"В директории {self.images_root} не найдены изображения из COCO JSON (всего {len(missing)} шт).\n"
                f"Пример отсутствующего файла: '{missing[0]}'"
            )

        if split in ("train", "val") and 0.0 < val_ratio < 1.0:
            indices = list(range(len(all_items)))
            random.Random(random_state).shuffle(indices)

            n_val = max(1, round(len(all_items) * val_ratio))
            val_indices = set(indices[:n_val])

            if split == "val":
                self.items = [all_items[i] for i in sorted(val_indices)]
            else:
                train_indices = set(indices[n_val:])
                self.items = [all_items[i] for i in sorted(train_indices)]
        else:
            self.items = all_items

        self._add_negatives(split=split, val_ratio=val_ratio, random_state=random_state)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, dict]:
        item = self.items[idx]

        file_bytes = np.fromfile(str(item["path"]), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

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

    def _boxes_xyxy(
        self, anns: list[dict], width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
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
            return np.empty((0, 4), dtype=np.float32), np.empty(
                (0,), dtype=np.int64
            )

        return np.asarray(boxes, dtype=np.float32), np.asarray(
            labels, dtype=np.int64
        )

    def _add_negatives(
        self, split: str | None, val_ratio: float, random_state: int
    ) -> None:
        n_pos = len(self.items)
        known = {Path(item["file_name"]).name for item in self.items}
        next_id = max((item["image_id"] for item in self.items), default=0) + 1

        target_folders = []
        for folder in self.negative_dirs:
            if not folder.is_dir():
                raise FileNotFoundError(f"Папка с негативами не найдена: {folder}")
            subdirs = sorted([d for d in folder.iterdir() if d.is_dir()])
            if subdirs:
                target_folders.extend(subdirs)
            else:
                target_folders.append(folder)

        if not target_folders:
            return

        all_chosen = []

        val_negative_ratio = 0.50
        quota_per_val_folder = max(
            1, round((n_pos * val_negative_ratio) / len(target_folders))
        )

        for dir_idx, folder in enumerate(target_folders):
            folder_seed = random_state + dir_idx * 1000

            paths = [
                p
                for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() == ".png"
            ]
            paths.sort(key=lambda p: unicodedata.normalize("NFC", str(p)))

            unique_paths = []
            for path in paths:
                name = unicodedata.normalize("NFC", path.name)
                if name in known:
                    continue
                known.add(name)
                unique_paths.append(path)

            rng = random.Random(folder_seed)
            indices = list(range(len(unique_paths)))
            rng.shuffle(indices)

            n_val = max(1, round(len(unique_paths) * val_ratio))
            val_indices = set(indices[:n_val])

            if split == "val":
                val_pool = [unique_paths[i] for i in sorted(val_indices)]
                
                # Отбираем детерминированно квоту (50% от позитивов / количество папок)
                sample_rng = random.Random(folder_seed + 2)
                sample_rng.shuffle(val_pool)
                selected = val_pool[:quota_per_val_folder]

            elif split == "train":
                train_indices = set(indices[n_val:])
                train_pool = [unique_paths[i] for i in sorted(train_indices)]
                
                if self.negative_ratio is not None and train_pool:
                    n_take = max(1, round(len(train_pool) * self.negative_ratio))
                    sample_rng = random.Random(folder_seed + 1)
                    sample_rng.shuffle(train_pool)
                    selected = train_pool[:n_take]
                else:
                    selected = train_pool
            else:
                selected = unique_paths

            all_chosen.extend(selected)

        for path in all_chosen:
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

            rel_key = unicodedata.normalize(
                "NFC", path.relative_to(image_root).as_posix()
            )
            found[rel_key] = path

            base_key = unicodedata.normalize("NFC", path.name)
            found[base_key] = path

        return found

    @staticmethod
    def _cipher_and_shot(file_name: str) -> tuple[str, int]:
        base_name = Path(unicodedata.normalize("NFC", file_name)).name
        stem = Path(base_name).stem
        cipher, separator, shot = stem.rpartition("_")
        if separator == "" or cipher == "" or not shot.isdigit():
            raise ValueError(f"Имя кадра не в формате ШИФР_N.png: {file_name}")
        return cipher, int(shot)