"""Поиск top-k: фото, кроп YOLO, ч/б внутри энкодера, FAISS IndexFlatIP."""

from __future__ import annotations

import os
from pathlib import Path

# faiss и PyTorch на macOS оба тянут libomp. Без этого поиск падает при первом вызове FAISS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
import torch

from finetune.data_utils import BookSample, square_tensor
from finetune.model import DinoEncoder
from src.detection.detector import StampDetector
from src.detection.yolo_processing import YOLOProcessing
from src.paths import WEIGHTS_DIR
from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import fit_square, read_bgr


class StampRetriever:
    def __init__(
        self,
        baseline_encoder: DinoEncoder,
        finetuned_encoder: DinoEncoder,
        samples: list[BookSample],
        device: torch.device,
        detector: StampDetector | None = None,
        conf_threshold: float = 0.75,
    ) -> None:
        self.baseline_encoder = baseline_encoder.eval()
        self.finetuned_encoder = finetuned_encoder.eval()
        self.samples = samples
        self.device = device
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.index: faiss.IndexFlatIP | None = None
        self._active = "finetuned"

    @classmethod
    def with_default_detector(
        cls,
        baseline_encoder: DinoEncoder,
        finetuned_encoder: DinoEncoder,
        samples: list[BookSample],
        device: torch.device,
    ) -> "StampRetriever":
        weights = WEIGHTS_DIR / "detection_weights" / "yolo26s_v2.onnx"
        detector = None
        if weights.is_file():
            detector = StampDetector(weights, YOLOProcessing())
        return cls(baseline_encoder, finetuned_encoder, samples, device, detector)

    def build_baseline_index(self) -> faiss.IndexFlatIP:
        self._active = "baseline"
        self.index = self._build(self.baseline_encoder)
        return self.index

    def build_finetuned_index(self) -> faiss.IndexFlatIP:
        self._active = "finetuned"
        self.index = self._build(self.finetuned_encoder)
        return self.index

    def search(self, query_image: np.ndarray | str | Path, top_k: int = 20) -> list[dict]:
        """Фото BGR или путь -> лучший кроп YOLO -> энкодер (уже с ч/б) -> FAISS."""
        if self.index is None:
            raise RuntimeError("Сначала вызовите build_finetuned_index или build_baseline_index")
        if top_k < 1:
            raise ValueError("top_k должен быть >= 1")

        image = read_bgr(query_image) if not isinstance(query_image, np.ndarray) else query_image
        crop = self._best_crop(image)
        if crop is None:
            return []
        crop = fit_square(crop, 224)

        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
        encoder = self.finetuned_encoder if self._active == "finetuned" else self.baseline_encoder
        vector = np.ascontiguousarray(_encode(encoder, tensor.to(self.device)), dtype=np.float32)
        faiss.normalize_L2(vector)
        scores, ids = self.index.search(vector, min(top_k, len(self.samples)))

        hits = []
        for rank, (score, row) in enumerate(zip(scores[0], ids[0]), start=1):
            if row < 0:
                continue
            sample = self.samples[int(row)]
            hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "mark_id": sample.mark_id,
                    "book": sample.book,
                    "factory_norm": sample.factory_norm,
                    "catalog_no": sample.catalog_no,
                    "page": sample.page,
                    "image_path": str(sample.path),
                }
            )
        return hits

    def _build(self, encoder: DinoEncoder) -> faiss.IndexFlatIP:
        vectors = []
        batch: list[torch.Tensor] = []

        def flush() -> None:
            if not batch:
                return
            blob = torch.stack(batch, dim=0).to(self.device)
            vectors.append(_encode(encoder, blob))
            batch.clear()

        for sample in self.samples:
            batch.append(square_tensor(sample.path))
            if len(batch) == 64:
                flush()
        flush()
        matrix = np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    def _best_crop(self, image: np.ndarray) -> np.ndarray | None:
        if self.detector is None:
            return image
        boxes, scores = self.detector.predict(image)
        scores = np.asarray(scores).reshape(-1)
        if len(scores) == 0:
            return None
        best = int(np.argmax(scores))
        if float(scores[best]) < self.conf_threshold:
            return None
        return crop_stamp(image, boxes[best], target_size=224)


def _encode(encoder: DinoEncoder, images: torch.Tensor) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(images)
    return embeddings.float().cpu().numpy().astype(np.float32)
