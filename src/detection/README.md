# Обработчики детекции

`YOLOProcessing` и `RTMDetProcessing` принимают BGR-изображение и реализуют один
интерфейс: `preprocess(image) -> (blob, metadata)`,
`postprocess(outputs, metadata) -> (boxes, scores)`. Рамки возвращаются как
`float32 [N, 4]` в формате xyxy в пикселях исходного фото; scores — `float32 [N]`.
Оба обработчика подключаются к существующему `StampDetector`.

RTMDet-обработчик рассчитан на наш одноклассовый `rtmdet_s_best.onnx`:
BGR 0–255, фиксированный вход 640×640, нормализация внутри ONNX, padding справа
и снизу, готовые xyxy и вероятности на выходе. Внешний NMS использует IoU 0.65
и оставляет максимум 100 рамок. Это другой экспортный контракт, чем у YOLO.

Для подключения в существующем конфиге приложения указать processor
`src.detection.rtmdet_processing.RTMDetProcessing`, путь к ONNX, `imgsz: 640`
и `iou_threshold: 0.65`. Confidence **0.84** выбран на val для precision ≥ 90%;
другой сохранённый рабочий порог — **0.5**. Параметр `iou_threshold` обработчика
относится к NMS, а не к сопоставлению рамок с разметкой. Отдельно детектор можно
использовать так:

```python
from pathlib import Path
import cv2
from src.detection.detector import StampDetector
from src.detection.rtmdet_processing import RTMDetProcessing

detector = StampDetector(
    model_path=Path("weights/detection_weights/rtmdet_s_best.onnx"),
    processor=RTMDetProcessing(conf_threshold=0.5, iou_threshold=0.65),
)
boxes, scores = detector.predict(cv2.imread("/path/to/photo.png"))
```

Для ONNX-инференса нужны NumPy, OpenCV и ONNX Runtime; torch/MMCV/MMDetection
не требуются. Источники весов и качество: [отчёт RTMDet](../../experiments/detection/RTMDet/README.md).
