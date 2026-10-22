import numpy as np
import cv2

class YOLOProcessing:

    def preprocess(self, image: np.ndarray, target_size: tuple[int, int] = (640, 640)):
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_size

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
    
    def postprocess(self):
        raise NotImplementedError