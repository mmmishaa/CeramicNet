# RTMDet-s

[rtmdet_s_train.ipynb](rtmdet_s_train.ipynb) обучает официальный RTMDet-s
через общий пайплайн CeramicNet: подготовленные split, аугментации команды,
общий логгер и оценщик. Основа — ветка `pipeline`.

Данные: `detection_pipeline/splits/v3_seed42_reserve80_active15`.
Train: **203 stamps + 56 негативов = 259 фото**; val: **51 stamps + 94 негатива = 145 фото**.
Из каждого источника сначала резервируются 80% для потенциального обучения,
остальные 20% остаются в val. Из объединённого резерва 374 фото выбираются 15%
(56) для фактического train. Остальные **318 фото не используются**.
Списки уже записаны в JSON с seed 42; повторного отбора через `negative_dirs` нет.
Ноутбук проверяет резерв и его хеши; загрузчики читают только `train.json` и `val.json`.
В логгер включаются пять файлов протокола, включая manifest и оба файла резерва.
При переходе с v2 обучение начинается заново с COCO-весов, `resume=False`.

## Локальный Mac

Из корня CeramicNet с Python 3.12:

```bash
python3.12 train_pipelines/detection_models/RTMDet-s/setup_local.py
```

Выберите в ноутбуке ядро **CeramicNet RTMDet (CPU)**. Установка создаёт
`.venv-rtmdet`, клонирует MMDetection 3.3.0 и MMCV 2.1.0 в `.local/rtmdet`,
собирает CPU-операторы MMCV. Нужны Apple Command Line Tools; первая сборка
занимает несколько минут. Для нового Apple Clang при сборке сторонних заголовков
torch используется `-Wno-error=invalid-specialization`.

По умолчанию фото ищутся в соседней с CeramicNet папке `Клеймы`, содержащей
`stamps`, `empty`, `ornaments_without_copies`. Другой путь можно задать в
`IMAGES` или через переменную окружения `CERAMIC_IMAGES` перед запуском ядра.
Файлы и SHA256 проверяются по сохранённым JSON. Нужны фотографии, перечисленные в текущих train/val JSON; списки резерва служат для проверки протокола.

Полный бюджет из `configs/experiments.json`: 50 эпох, размер 640, seed 42,
аугментации `light`. На Mac batch по умолчанию 2; на CUDA — 8 из общей конфигурации.
Для первого опыта поменяйте `EPOCHS` на 1–3 и выполните все ячейки.
Альтернатива при автоматическом исполнении: `CERAMIC_RTMDET_EPOCHS=1`.
Фактические параметры записываются в metadata и общий логгер.

На Mac ноутбук явно выбирает CPU через `CPURunner`; остальные части нативного
тренера не меняются. Эта сборка MMCV предоставляет CPU-операторы, а не MPS.
`mmcv-lite` для запуска модели не подходит. Скрипт установки предназначен
для локального CPU-окружения; не используйте его для установки CUDA.

## Linux / CUDA

Установите совместимую пару torch/torchvision, MMEngine, полноценный MMCV 2.1.0
и MMDetection 3.3.0 согласно [официальной инструкции](https://mmdetection.readthedocs.io/en/latest/get_started.html).
Клонируйте MMDetection v3.3.0 и укажите путь в `MMDET_REPO`; установите
`nbformat`, `nbclient`, `ipykernel`, `pycocotools`, `matplotlib` и зависимости
общего пайплайна. Ноутбук выберет CUDA автоматически.

## Что сохраняется

Каждый запуск создаёт новую папку `runs/rtmdet_s_*` с фактическим конфигом,
источником и SHA256 COCO-весов, журналами, лучшим checkpoint по val mAP,
общими `predictions.json` и `metrics.json`, графиками и примерами рамок.
По умолчанию используется локальный логгер; ClearML включается параметром
`TRACKING="clearml"` после установки `requirements-tracking.txt` и `clearml-init`.

### ONNX

После обучения ноутбук автоматически экспортирует лучший checkpoint в
`onnx/rtmdet_s_best.onnx` (opset 17, стандартные ONNX-операторы, batch 1, 640×640).
`.pth` сохраняется для возобновления обучения. Для передачи модели используйте
`.onnx` вместе с обработчиком
[`RTMDetProcessing`](../../../detection_pipeline/rtmdet_onnx_processing.py).
Он реализует `preprocess`/`postprocess` общего тестового пайплайна:
`processor_module: detection_pipeline.rtmdet_onnx_processing`,
`process_class: RTMDetProcessing`, `imgsz: 640`.
Обработчик YOLO не подходит: RTMDet использует BGR, другой resize/pad,
нормализацию и формат рамок. Для инференса достаточно `onnxruntime`, NumPy и OpenCV;
MMDetection и MMCV нужны только для обучения/экспорта.

Повторный экспорт существующего опыта (без обучения):

```bash
.venv-rtmdet/bin/python train_pipelines/detection_models/RTMDet-s/export_onnx.py \
  --config runs/<run>/configs/rtmdet_s.py \
  --checkpoint runs/<run>/train/best_coco_bbox_mAP_epoch_49.pth \
  --output runs/<run>/onnx/rtmdet_s_best.onnx
```

Вход ONNX: BGR float32 `[1,3,640,640]`, значения 0–255; нормализация внутри графа.
Выход: `[1,8400,5]`, xyxy в пикселях входа + вероятность клейма.
NMS и восстановление исходных координат выполняет обработчик. Параметры исходной
оценки: NMS IoU 0.65, confidence 0.001 для COCO AP; рабочий confidence 0.84
выбран на val. Порог NMS и IoU сопоставления с разметкой — разные параметры.
Текущий общий бенчмарк может передавать один `iou_threshold` в оба места;
для воспроизведения COCO-метрик используйте сохранённый v3 val и общий оценщик.

`validate_onnx.py` проверяет все val-фото и сравнивает экспорт с сохранёнными
нативными предсказаниями (`--model`, `--annotations`, `--images`, `--reference`,
`--output`). Отчёт содержит метрики, расхождения рамок и вероятностей.

Есть 41 пересечение предметов между фактическими train/val, включая 8 между негативами; manifest фиксирует их. Общих файлов и SHA256 нет. Независимого test нет. Порог
подбирается только на val; метрики короткого прогона не являются итоговым
качеством модели.

Официальные [конфиг и веса RTMDet-s](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/metafile.yml),
[сборка MMCV](https://github.com/open-mmlab/mmcv/blob/v2.1.0/docs/en/get_started/build.md).

## Историческая локальная проверка v2, 23 октября 2026

Следующие числа относятся к старому split v2, а не к текущему v3.

На MacBook Air M5 выполнены все ячейки ноутбука: одна эпоха, batch 2,
640×640, все 243 train и 61 val фото. Настоящие загрузчики MMDetection сохранили
40/10 негативов. Созданы лучшие веса, общий логгер со статусом `completed`,
предсказания для всех 61 фото, метрики и три иллюстрации. Все 22 теста
`detection_pipeline.test_pipeline` и `detection_pipeline.test_integration` прошли.

Общий оценщик после пробной эпохи: mAP@[.50:.95] = 0.0432, AP50 = 0.0673.
При пороге 0.5 детекций нет; порог с precision ≥ 90% не найден. Это проверка
полного цикла запуска с новой головой на один класс, не итоговое качество модели.
Полное обучение на 50 эпох в этой проверке не выполнялось.


## Полное обучение v3, 24 октября 2026

Выполнены 50 эпох на `v3_seed42_reserve80_active15`: 259 train / 145 val,
CPU, batch 2, размер 640, seed 42, аугментации light. После остановок
восстановлены оптимизатор, EMA и расписание обучения. Лучший checkpoint —
49-я эпоха; финальный checkpoint 50-й эпохи сохраняет состояние для продолжения.

Общий COCO-оценщик на val: mAP50–95 **0.6536**, AP50 **0.8481**, AP75 **0.6932**.
Сохранены предсказания и галереи всех 145 val-фото и всех 254 stamps.
[Подробные метрики и ограничения](../../../test_pipelines/detection_models/RTMDet-s/README.md#результат-полного-обучения-24-октября-2026).
