import cv2
import numpy as np
from PIL import Image

_RESAMPLE = getattr(Image, "Resampling", Image).BILINEAR


class RTDetrProcessing:

    @staticmethod
    def preprocess(image: np.ndarray, target_size: tuple[int, int] = (640, 640)):
        # Кадр вписывается в квадрат с белым полем, пиксели делятся на 255.
        # Так же StampDataset готовит картинку для model="rtdetr".
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_size

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = min(target_w, max(1, int(round(orig_w * scale))))
        new_h = min(target_h, max(1, int(round(orig_h * scale))))
        resized = np.asarray(
            Image.fromarray(rgb).resize((new_w, new_h), _RESAMPLE),
            dtype=np.uint8,
        )

        canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        chw = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
        blob = np.expand_dims(chw / 255.0, axis=0)
        metadata = {
            "orig_shape": (orig_h, orig_w),
            "input_size": (target_h, target_w),
            "scale": (new_h / orig_h, new_w / orig_w),
            "pad_left": pad_left,
            "pad_top": pad_top,
        }
        return blob, metadata

    @staticmethod
    def postprocess(
        output: list[np.ndarray],
        metadata: dict,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        logits, pred_boxes = _split_outputs(output)
        if logits.ndim == 1:
            logits = logits[:, None]

        scores_map = _sigmoid(logits)
        num_queries, num_classes = scores_map.shape
        flat = scores_map.reshape(-1)
        k = min(num_queries, flat.size)
        if k < flat.size:
            chosen = np.argpartition(-flat, k - 1)[:k]
        else:
            chosen = np.arange(flat.size)
        chosen = chosen[np.argsort(-flat[chosen])]

        scores = flat[chosen]
        query_index = chosen // num_classes
        keep = scores > conf_threshold
        scores = scores[keep]
        query_index = query_index[keep]
        if len(scores) == 0:
            return _empty()

        boxes = pred_boxes[query_index]
        input_h, input_w = metadata["input_size"]
        cx, cy, width, height = (boxes[:, i] for i in range(4))
        xyxy = np.stack(
            [
                (cx - width / 2.0) * input_w,
                (cy - height / 2.0) * input_h,
                (cx + width / 2.0) * input_w,
                (cy + height / 2.0) * input_h,
            ],
            axis=1,
        )

        keep_nms = _nms(xyxy, scores, iou_threshold)
        xyxy = xyxy[keep_nms]
        scores = scores[keep_nms]

        scale_h, scale_w = metadata["scale"]
        orig_h, orig_w = metadata["orig_shape"]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - metadata["pad_left"]) / scale_w
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - metadata["pad_top"]) / scale_h
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h)

        alive = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        return xyxy[alive].astype(np.float32), scores[alive].astype(np.float32)


def _split_outputs(output: list[np.ndarray]):
    logits = None
    boxes = None
    for item in output:
        array = np.asarray(item)
        if array.ndim == 3:
            array = array[0]
        if array.ndim != 2:
            continue
        if array.shape[-1] == 4:
            boxes = array
        else:
            logits = array
    if logits is None or boxes is None:
        shapes = [tuple(np.asarray(item).shape) for item in output]
        raise ValueError(
            "RT-DETR ждёт два выхода ONNX, logits и pred_boxes. "
            f"Пришли тензоры с формами {shapes}"
        )
    return logits, boxes


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[_iou(boxes[current], boxes[rest]) <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return inter / np.maximum(area_box + area_boxes - inter, 1e-6)


def _empty():
    return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
