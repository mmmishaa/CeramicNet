import importlib
from pathlib import Path
import numpy as np
import onnxruntime as ort

from src.paths import ROOT_DIR


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


class StampDetector:

    def __init__(
        self,
        model_path: Path,
        processor,
        num_threads: int = 4,
    ):
        self.model_path = model_path
        self.processor = processor

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def _forward(self, tensor: np.ndarray) -> list[np.ndarray]:
        return self.session.run(None, {self.input_name: tensor})

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, metadata = self.processor.preprocess(image)
        raw_outputs = self._forward(tensor)
        boxes, scores = self.processor.postprocess(raw_outputs, metadata)
        return boxes, scores


class _SizedProcessor:

    def __init__(self, inner, image_size: int, conf_threshold: float, iou_threshold: float):
        self.inner = inner
        self.image_size = image_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, image):
        return self.inner.preprocess(
            image,
            target_size=(self.image_size, self.image_size),
        )

    def postprocess(self, output, metadata):
        return self.inner.postprocess(
            output,
            metadata,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )


def build_detector(config: dict, conf_threshold: float | None = None) -> StampDetector:
    detector_cfg = config.get("detector", config)
    weights = project_path(detector_cfg["weights"])
    if not weights.is_file():
        raise FileNotFoundError(f"ONNX детектора не найден: {weights}")

    module_path, class_name = detector_cfg["processor"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    processor = getattr(module, class_name)()

    image_size = int(detector_cfg.get("imgsz", 640))
    conf = (
        float(conf_threshold)
        if conf_threshold is not None
        else float(config.get("confidence_config", {}).get("conf_threshold", 0.75))
    )
    iou = float(config.get("confidence_config", {}).get("iou_threshold", 0.5))

    sized = _SizedProcessor(processor, image_size, conf, iou)
    return StampDetector(weights, sized)
