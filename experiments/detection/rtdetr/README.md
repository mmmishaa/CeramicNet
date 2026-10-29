# RT-DETRv2-S: проверка ONNX-процессора

Процессор: [`src/detection/rtdetr_processing.py`](../../../src/detection/rtdetr_processing.py).
Обучение: [`rtdetr_train_pipeline.ipynb`](../../../notebooks/detection/rtdetr/rtdetr_train_pipeline.ipynb).

Интерфейс совпадает с `YOLOProcessing`: параметры `imgsz`, `conf_threshold`,
`iou_threshold` задаются в конструкторе; `preprocess(image)` возвращает тензор и
метаданные, `postprocess(output, metadata)` — рамки `xyxy` в исходном изображении
и оценки уверенности.

Предобработка соответствует обучению RT-DETR: BGR → RGB, Pillow bilinear,
сохранение пропорций, белые поля по центру, float32 NCHW и деление на 255.
Экспорт выдаёт **два** тензора: `logits [1, Q, 1]` и
`pred_boxes [1, Q, 4]`. Применяются sigmoid, перевод нормализованных `cxcywh`
в `xyxy`, NMS и обратное преобразование letterbox с отдельными фактическими
масштабами по осям. Этот адаптер предназначен для одноклассового экспорта из
ноутбука, а не для других форматов RT-DETR.

NMS добавлен для общего протокола проекта. Визуализация в ноутбуке использует
Hugging Face postprocessing без NMS; при необходимости отключить подавление
в адаптере можно установить `iou_threshold: 1.0`.

## Проверено 29 октября 2026

[Сохранённый результат проверки](processor_validation.json).

- 7 автоматических проверок пройдены, включая реальный ONNX Runtime на
  искусственном графе с двумя выходами, общий загрузчик, фабрику детектора и
  точку входа бенчмарка. Это проверка интеграции, не измерение качества модели.
- Предобработка всех 77 изображений текущего разбиения `TestDetectionDataset`
  побайтно совпала с `_letterbox` из обучающего `dataset/datasets.py` ветки
  `rtdetr`: максимальное отличие float32-тензоров равно 0.
- Текущий локальный набор: 51 изображение с клеймом, 26 негативов, 51 рамка;
  `val=0.2`, `random_state=42`. В выводе обучающего ноутбука тест содержит
  61 изображение из 305. Эти разбиения нельзя считать одинаковыми или
  независимыми от обучения без проверки исходных списков файлов.

## Метрики модели

29 октября 2026 выполнен прогон обученного `rtdetr_v2_s.onnx`, предоставленного
пользователем из Downloads. Размер — 80 703 254 байта (76.96 MiB), SHA256:
`e4a16ac419f39f278ef5561a74dbc46550092abdc51c28aa4bced4c8f7746cae`.
Выходы реального графа: `logits [1,300,1]`, `pred_boxes [1,300,4]`.

Исходная валидация восстановлена по алгоритму ноутбука: 254 размеченных фото,
51 выбранный негатив (`random.Random(0)`), shuffle 305 элементов с seed 42,
первые `int(305*0.2)=61` элементов. Все 47 названий положительных фото в точном
порядке и первый негатив совпали с сохранённым выводом ноутбука. Остальные
негативы восстановлены из исходной локальной коллекции; байты всех 61 фото
сопоставлены с ней. Полного исторического манифеста в ноутбуке нет, поэтому
реконструкция негативов остаётся ограничением воспроизводимости.
Папка `ornaments_without_copies` в общем архиве называется `ornaments`.

[Манифест файлов с SHA256](notebook_holdout_manifest.json): 61 фото,
47 с клеймами, 14 негативов, 47 рамок. Аннотации совпадают с
`stamps_coco_merged.zip` и общим `ceramic_dataset` по SHA256
`cdd3005bbc5045accc4a965c4e244c74fd0b25a403535047c043e6df080a225c`.

| Метрика | Значение | Протокол |
| :--- | ---: | :--- |
| mAP@50 | 0.960981 | COCOeval bbox, `conf>0.001`, maxDets=100 |
| mAP@50–95 | 0.735770 | COCOeval bbox, `conf>0.001`, maxDets=100 |
| AP@75 | 0.849635 | COCOeval bbox |
| Precision | 1.000000 | `conf>0.75`, matching IoU=0.5 |
| Recall | 0.872340 | `conf>0.75`, matching IoU=0.5 |
| F1 | 0.931818 | `conf>0.75`, matching IoU=0.5 |
| TP / FP / FN | 41 / 0 / 6 | NMS IoU=0.5 |
| Негативы с ложными детекциями | 0 / 14 | `conf>0.75` |
| Задержка / FPS | 215.43 мс / 4.64 | CPUExecutionProvider, Apple M5, `conf>0.75` |

Эта выборка использовалась для выбора лучшей эпохи по loss и является
валидацией, а не независимым тестом. Она отличается от исторических наборов
YOLO/RTMDet; напрямую ранжировать модели по разным выборкам нельзя.
Скорость измеряется
для предобработки, ONNX-инференса и постобработки без чтения фотографии с диска.
Отдельного прогрева не было; число потоков — по умолчанию ONNX Runtime 1.20.1.

[Полные метрики и параметры](metrics.json) · [COCO JSON](coco_metrics.json) ·
[предсказания при conf 0.001](low_confidence_predictions.json) ·
[предсказания при conf 0.75](benchmark_notebook_holdout.json) ·
[CSV](benchmark_notebook_holdout.csv).

Общий оценщик репозитория на тех же 61 фото при `conf>0.75` даёт mAP@50
0.936100 и mAP@50–95 0.767739. Он использует собственную интерполяцию AP после
confidence cutoff; это отдельные значения, они не заменяют COCO AP в таблице.

Первый прогон на текущем разбиении `TestDetectionDataset` также сохранён:
77 фото, 51 с клеймом, 26 негативов; mAP@50 0.9594, mAP@50–95 0.8605,
Precision 0.9423, Recall 0.9608, F1 0.9515 по общему оценщику при `conf>0.75`.
Из этих 51 положительных фото только 8 входят в валидацию ноутбука,
**43 входят в его обучение**. Этот прогон проверяет работоспособность,
его нельзя использовать для оценки качества на отложенных данных.
[Предсказания первого прогона](benchmark_current_split.json).

Первоначальный запуск был заблокирован отсутствием весов; этот этап сохранён
в `processor_validation.json`, к нему добавлена ссылка на завершённую оценку.

## Воспроизведение

Разместить предоставленный ONNX в `weights/detection_weights/rtdetr_v2_s.onnx`,
а общий датасет — в `data/ceramic_dataset`. Бенчмарк проверяет SHA256 аннотаций
и каждого выбранного фото по сохранённому манифесту.

```bash
python -m scripts.detection_test --model rtdetr_v2_s \
  --split_manifest experiments/detection/rtdetr/notebook_holdout_manifest.json \
  --output_csv reports/rtdetr_new/notebook_holdout.csv \
  --output_json reports/rtdetr_new/notebook_holdout.json

python -m scripts.detection_test --model rtdetr_v2_s --conf 0.001 \
  --split_manifest experiments/detection/rtdetr/notebook_holdout_manifest.json \
  --output_csv reports/rtdetr_new/low_conf.csv \
  --output_json reports/rtdetr_new/low_conf.json

python experiments/detection/rtdetr/coco_metrics.py reports/rtdetr_new/low_conf.json \
  --output reports/rtdetr_new/coco.json
```

При повторном запуске выбирать новый каталог отчёта. JSON не перезаписывает
существующий файл и сохраняет метрики, параметры, SHA256 весов и аннотаций,
имена изображений, предсказания и разметку. CSV направлен в отдельный файл,
чтобы сохранить предыдущие результаты. Путь к данным можно задать отдельным
YAML через `--config`, не меняя общий конфиг. Предсказания двух прогонов
согласованы: отсечение низкопорогового результата по score > 0.75 воспроизводит
отдельный запуск при conf 0.75 на всех 61 фото.

Зависимости для проверок: `numpy`, `opencv-python`, `pillow`, `onnxruntime`,
`onnx`, `torch`, `PyYAML`, `tqdm`; для COCO AP — `pycocotools`.

```bash
python -m unittest discover -s tests -v
```
