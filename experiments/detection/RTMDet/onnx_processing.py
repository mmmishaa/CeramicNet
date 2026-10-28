"""RTMDet-s ONNX processing; implements the shared detection benchmark API.

Input: BGR float32 [1,3,640,640], values 0..255 (normalization in graph).
Output: [1,8400,5], decoded xyxy in input pixels followed by probability.
Only NumPy/OpenCV are needed; no torch, MMCV or MMDetection at inference.
"""
import cv2
import numpy as np


class RTMDetProcessing:
    @staticmethod
    def preprocess(image, target_size=(640, 640)):
        if tuple(target_size) != (640, 640):
            raise ValueError("This export requires target_size=(640, 640)")
        h, w = image.shape[:2]
        scale = min(640 / w, 640 / h)
        # MMCV imrescale uses int(value + 0.5), including half-pixel ties.
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        return blob, {"orig_shape": (h, w), "img_shape": (640, 640),
                      "scale_factor": (nw / w, nh / h)}

    @staticmethod
    def postprocess(output, metadata, conf_threshold=0.5, iou_threshold=0.65):
        raw = output[0][0]
        keep = raw[:, 4] > conf_threshold
        boxes, scores = raw[keep, :4].copy(), raw[keep, 4].copy()
        if not len(scores):
            return boxes, scores
        nh, nw = metadata["img_shape"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, nw)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, nh)
        sx, sy = metadata["scale_factor"]
        boxes /= np.array([sx, sy, sx, sy], dtype=np.float32)
        # Single-class greedy NMS, continuous xyxy (no +1 area convention).
        order = np.argsort(-scores, kind="stable")
        selected = []
        area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        while order.size and len(selected) < 100:
            i, rest = order[0], order[1:]
            selected.append(i)
            lt = np.maximum(boxes[i, :2], boxes[rest, :2])
            rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
            wh = np.maximum(rb - lt, 0)
            inter = wh[:, 0] * wh[:, 1]
            iou = inter / np.maximum(area[i] + area[rest] - inter, 1e-12)
            order = rest[iou <= iou_threshold]
        boxes, scores = boxes[selected], scores[selected]
        # Native MMDetection clips against the padded image before NMS;
        # the common adapter clips to the original image afterwards.
        h, w = metadata["orig_shape"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], scores[valid]
