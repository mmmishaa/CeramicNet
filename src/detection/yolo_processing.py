import cv2
import numpy as np

class YOLOProcessing:

    def __init__(self, imgsz: int = 640, conf_threshold: float = 0.7, iou_threshold: float = 0.5):
        self.imgsz = (imgsz, imgsz)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def preprocess(self, image: np.ndarray):
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.imgsz

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(
            image, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(chw, axis=0)

        metadata = {
            "orig_shape": (orig_h, orig_w),
            "scale": scale,
            "pad": (pad_left, pad_top),
        }

        return blob, metadata

    def postprocess(
        self,
        output: list[np.ndarray],
        metadata: dict
    ):
        raw = output[0]
        if raw.ndim == 3:
            raw = raw.squeeze(0)
        preds = raw.T if raw.shape[0] < raw.shape[1] else raw

        scores = preds[:, 4]
        mask = scores > self.conf_threshold

        valid_boxes = preds[mask, :4]
        valid_scores = scores[mask]

        if len(valid_boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        cx = valid_boxes[:, 0]
        cy = valid_boxes[:, 1]
        w = valid_boxes[:, 2]
        h = valid_boxes[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes_for_nms = np.stack([x1, y1, w, h], axis=1)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_for_nms.tolist(),
            scores=valid_scores.tolist(),
            score_threshold=0.0,
            nms_threshold=self.iou_threshold,
        )

        if len(indices) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        indices = np.array(indices).flatten()

        xyxy_boxes = np.stack([x1, y1, x2, y2], axis=1)
        final_boxes = xyxy_boxes[indices]
        final_scores = valid_scores[indices]

        pad_left, pad_top = metadata["pad"]
        scale = metadata["scale"]
        orig_h, orig_w = metadata["orig_shape"]

        final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - pad_left) / scale
        final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - pad_top) / scale

        final_boxes[:, [0, 2]] = np.clip(final_boxes[:, [0, 2]], 0, orig_w)
        final_boxes[:, [1, 3]] = np.clip(final_boxes[:, [1, 3]], 0, orig_h)

        return final_boxes.astype(np.float32), final_scores.astype(np.float32)