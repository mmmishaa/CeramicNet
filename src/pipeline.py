import numpy as np
from pathlib import Path

from src.preprocessing.crop import crop_stamp

class CeramicPipeline:

    def __init__(
        self,
        detector,
        encoder,
        index,
        conf_threshold: float = 0.75,
        top_k: int = 3
    ):

        self.detector = detector
        self.encoder = encoder
        self.index = index
        self.conf_threshold = conf_threshold
        self.top_k = top_k
        self.target_size = getattr(self.encoder, 'input_size', 224)

    def _detect(self, image: np.ndarray):
        boxes, scores = self.detector.predict(image)
        return boxes, scores

    def _preprocess(self, image: np.ndarray, boxes: np.ndarray):
        crops = []

        for box in boxes:
            crop = crop_stamp(
                image=image,
                box=box,
                target_size=self.target_size
            )

            crops.append(crop)

        return crops

    def _retrieve(self, crop: np.ndarray, top_k: int = 3):
        ...

    def predict(self, image_path: Path) -> dict:
        ...