from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.preprocessing.image import fit_square

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StampEncoder:
    def __init__(
        self,
        model_path: Path,
        input_size: int = 224,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        num_threads: int = 4,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ONNX энкодера не найден: {self.model_path}. "
                "Первый выход сети должен быть вектором изображения."
            )
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean и std энкодера должны содержать 3 канала")

        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.channels_first, height, width, batch_limit = _input_layout(model_input.shape)
        self.uint8_input = "uint8" in model_input.type

        if height is not None and width is not None and height != width:
            raise ValueError(
                f"Энкодер ждёт прямоугольный вход {height}x{width}, нужен квадрат"
            )
        if height is not None and height != input_size:
            raise ValueError(
                f"Энкодер ждёт вход {height}, в конфиге input_size={input_size}"
            )

        self.input_size = input_size
        self.batch_limit = batch_limit

    @classmethod
    def from_mapping(cls, mapping: dict, weights: str | Path | None = None) -> "StampEncoder":
        model_path = Path(weights) if weights is not None else Path(mapping["weights"])
        return cls(
            model_path=model_path,
            input_size=int(mapping.get("input_size", 224)),
            mean=tuple(mapping.get("mean", IMAGENET_MEAN)),
            std=tuple(mapping.get("std", IMAGENET_STD)),
            num_threads=int(mapping.get("num_threads", 4)),
        )

    def embed(self, image: np.ndarray) -> np.ndarray:
        return self.embed_many([image])[0]

    def embed_many(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("Пустой список изображений")

        limit = self.batch_limit or len(images)
        parts = []
        for start in range(0, len(images), limit):
            chunk = images[start : start + limit]
            blob = np.stack([self._prepare(image) for image in chunk], axis=0)
            raw = self.session.run(None, {self.input_name: blob})[0]
            parts.append(_to_vectors(raw))
        return np.concatenate(parts, axis=0)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError(f"Ожидался цветной кадр HWC, получено: {image.shape}")

        square = fit_square(image, self.input_size)
        rgb = square[:, :, ::-1]
        if self.uint8_input:
            tensor = rgb if not self.channels_first else rgb.transpose(2, 0, 1)
            return np.ascontiguousarray(tensor)

        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - self.mean) / self.std
        if self.channels_first:
            tensor = tensor.transpose(2, 0, 1)
        return np.ascontiguousarray(tensor, dtype=np.float32)


def _dim(value) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


def _input_layout(shape) -> tuple[bool, int | None, int | None, int | None]:
    if len(shape) != 4:
        raise ValueError(f"Энкодер должен принимать тензор NCHW или NHWC, получено: {shape}")

    batch, first, second, last = shape
    first_dim = _dim(first)
    last_dim = _dim(last)
    channels_first = not (last_dim == 3 and first_dim != 3)

    if channels_first:
        height, width = _dim(second), last_dim
    else:
        height, width = first_dim, _dim(second)

    batch_limit = _dim(batch)
    return channels_first, height, width, batch_limit


def _to_vectors(raw: np.ndarray) -> np.ndarray:
    if raw.ndim == 4:
        raw = raw.mean(axis=(2, 3))
    elif raw.ndim == 3:
        raw = raw[:, 0]
    elif raw.ndim == 1:
        raw = raw.reshape(1, -1)
    elif raw.ndim != 2:
        raise ValueError(f"Неожиданная форма выхода энкодера: {raw.shape}")

    vectors = raw.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms
