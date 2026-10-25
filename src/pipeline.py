from pathlib import Path
import numpy as np

from src.preprocessing.crop import crop_stamp
from src.preprocessing.image import read_bgr
from src.retrieval.index import require_top_k

class CeramicPipeline:
    def __init__(
        self,
        encoder,
        index,
        detector=None,
        conf_threshold: float = 0.75,
        top_k: int = 3,
    ):
        self.encoder = encoder
        self.index = index
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.top_k = require_top_k(top_k)
        self.target_size = getattr(self.encoder, "input_size", 224)

    def _resolve_top_k(self, top_k: int | None) -> int:
        if top_k is None:
            return self.top_k
        return require_top_k(top_k)

    def _detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes, scores = self.detector.predict(image)
        return boxes, scores

    def _preprocess(self, image: np.ndarray, boxes: np.ndarray) -> list[np.ndarray]:
        return [
            crop_stamp(image=image, box=box, target_size=self.target_size)
            for box in boxes
        ]

    def _retrieve(self, crop: np.ndarray, top_k: int | None = None) -> list[dict]:
        k = self._resolve_top_k(top_k)
        vector = self.encoder.embed(crop)
        return self.index.search(vector, top_k=k)

    def predict(
        self,
        image_path: Path,
        top_k: int | None = None,
        detect: bool = True,
    ) -> dict:
        k = self._resolve_top_k(top_k)
        image = read_bgr(image_path)

        if not detect or self.detector is None:
            height, width = image.shape[:2]
            return {
                "image": str(image_path),
                "top_k": k,
                "detections": [
                    {
                        "box": [0.0, 0.0, float(width), float(height)],
                        "score": None,
                        "matches": self._retrieve(image, top_k=k),
                    }
                ],
            }

        boxes, scores = self._detect(image)
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(boxes) != len(scores):
            raise RuntimeError(f"Detector returned {len(boxes)} boxes and {len(scores)} scores")

        keep = scores >= self.conf_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        order = np.argsort(scores)[::-1]
        boxes = boxes[order]
        scores = scores[order]

        crops = self._preprocess(image, boxes) if len(boxes) else []
        detections = []
        for box, score, crop in zip(boxes, scores, crops):
            detections.append(
                {
                    "box": [float(val) for val in box],
                    "score": float(score),
                    "matches": self._retrieve(crop, top_k=k),
                }
            )

        return {
            "image": str(image_path),
            "top_k": k,
            "detections": detections,
        }