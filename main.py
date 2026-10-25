import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import CeramicPipeline
from src.detection.detector import StampDetector
from src.detection.yolo_processing import YOLOProcessing
from src.paths import WEIGHTS_DIR

processor = YOLOProcessing()

detector = StampDetector(
    model_path=WEIGHTS_DIR / "detection_weights" / "yolo26s_v1.onnx",
    processor=processor
)

ceramic_pipeline = CeramicPipeline(
    detector=detector,
    encoder=None,
    index=None
)

image_path = r"C:\Users\Misha_Savinov\Desktop\CeramicNet\ceramic_dataset\images\КБ-2014-Р5-0422_5.png"

file_bytes = np.fromfile(image_path, dtype=np.uint8)
image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

boxes, scores = ceramic_pipeline._detect(image)

image_np_list = ceramic_pipeline._preprocess(image, boxes)

for i, crop in enumerate(image_np_list):
    image_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(4,4))
    plt.title(f"Crop {i}")
    plt.imshow(image_rgb)
    plt.show()