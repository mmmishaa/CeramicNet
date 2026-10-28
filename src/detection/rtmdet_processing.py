import cv2
import numpy as np


class RTMDetProcessing:

    def __init__(self, imgsz: int = 640, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.65):
        if imgsz != 640:
            raise ValueError("This RTMDet ONNX export requires imgsz=640")
        if not 0 <= conf_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("Confidence and NMS IoU thresholds must be between 0 and 1")
        self.imgsz = (640, 640)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

    def preprocess(self, image: np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("Expected a non-empty HxWx3 BGR image")
        height, width = image.shape[:2]
        target_h, target_w = self.imgsz
        scale = min(target_w / width, target_h / height)
        # MMCV imrescale rounds half-pixel ties upward, unlike round().
        new_w, new_h = int(width * scale + 0.5), int(height * scale + 0.5)
        if min(new_w, new_h) < 1:
            raise ValueError("Image aspect ratio is too extreme for the 640x640 input")
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # RTMDet places the image at the top left, padding bottom/right only.
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        return blob, {"orig_shape": (height, width), "img_shape": self.imgsz,
                      "scale_factor": (new_w / width, new_h / height)}

    def postprocess(self, output: list[np.ndarray], metadata: dict):
        if len(output) != 1:
            raise ValueError("Expected one RTMDet output tensor [1, N, 5]")
        tensor = np.asarray(output[0], dtype=np.float32)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != 5:
            raise ValueError(f"Expected RTMDet output [1, N, 5], got {tensor.shape}")
        raw = tensor[0]
        keep = np.isfinite(raw).all(axis=1) & (raw[:, 4] > self.conf_threshold)
        boxes, scores = raw[keep, :4].copy(), raw[keep, 4].copy()
        if not len(scores):
            return boxes, scores

        # Match native MMDetection: clip against the padded input before NMS.
        padded_h, padded_w = metadata["img_shape"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, padded_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, padded_h)
        sx, sy = metadata["scale_factor"]
        boxes /= np.array([sx, sy, sx, sy], dtype=np.float32)

        # Single-class continuous-coordinate NMS; the graph has decoded xyxy,
        # so neither cxcywh conversion nor an additional sigmoid is required.
        order = np.argsort(-scores, kind="stable")
        selected = []
        area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        while order.size and len(selected) < 100:
            index, rest = order[0], order[1:]
            selected.append(index)
            lt = np.maximum(boxes[index, :2], boxes[rest, :2])
            rb = np.minimum(boxes[index, 2:], boxes[rest, 2:])
            wh = np.maximum(rb - lt, 0)
            intersection = wh[:, 0] * wh[:, 1]
            iou = intersection / np.maximum(area[index] + area[rest] - intersection, 1e-12)
            order = rest[iou <= self.iou_threshold]
        boxes, scores = boxes[selected], scores[selected]

        # The common native adapter clips to original-image bounds after NMS.
        height, width = metadata["orig_shape"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], scores[valid]
