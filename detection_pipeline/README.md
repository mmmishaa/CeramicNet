# Универсальный пайплайн CeramicNet

Ветка `pipeline` построена от `datasets` (исходный commit `c3b1e54`).
Пайплайн использует настоящий `dataset.StampDataset` и аугментации коллег.
Ноутбук для работы: `train_pipelines/universal_detection.ipynb`.

## Как части связаны

- `dataset/datasets.py`: чтение COCO, проверка размеров, преобразования рамок и меток.
- `dataset/transforms/`: аугментации команды; фабрика собирает их из параметров.
- `configs/experiments.json`: отдельный выбор аугментаций для каждой модели и общий бюджет.
- `detection_pipeline/data.py`: фиксированное групповое разбиение.
- `detection_pipeline/datasets.py`: небольшой адаптер StampDataset для torchvision.
- `detection_pipeline/integrations/rtdetr.py`: StampDataset внутри штатного RT-DETR Dataset/evaluator.
- `detection_pipeline/integrations/mmdet.py`: те же аугментации в штатном COCO-пайплайне MMDetection.
- `train_fasterrcnn.py`, `train_native.py`: запуск обучения.
- `predict_native.py`, `evaluate.py`, `compare.py`: общий формат предсказаний и сравнение.

У MMDetection остаётся его штатный CocoDataset, поскольку trainer ожидает
DetDataSample, метаданные и собственный collate. Он читает те же JSON и использует
те же преобразования команды через TeamAugment. Для RT-DETR сохраняются его
COCO evaluator, set_epoch и collate, а чтение изображений/рамок передано StampDataset.

## Данные уже разделены

| Часть | Фото | Рамки | Предметы |
|---|---:|---:|---:|
| train | 203 | 205 | 201 |
| val | 51 | 52 | 51 |

Файлы: `detection_pipeline/splits/v1_seed42/{train,val,manifest}.json`.
Seed 42, примерно 80/20 по предметам. Последний суффикс `_номер_снимка` отбрасывается,
снимки одного предмета остаются вместе. Точные копии объединяются по SHA256.
Две пары фотографий одного предмета не разделены. Проверка SHA256 не находит
пережатые копии или другие ракурсы с неправильно записанным шифром: правило
группировки стоит подтвердить, при необходимости задайте явный group_id.

Исходная разметка — COCO `annotations/instances_default.json` из Downloads,
254 фото, 257 bbox, один класс kleimo. Четыре пути исправлены во вновь созданных
JSON, потому что фото оказались во вложенных папках. Журнал есть в manifest.
Фотографии и исходный JSON не изменены; фотографии в Git не включены.

В сохранённых COCO JSON `kleimo=0`. StampDataset сам отображает произвольные COCO
ID в непрерывные метки: Faster R-CNN — с 1 (0 занят фоном), RT-DETR/RTMDet — с 0.
В `model="default"` сохраняются исходные COCO ID. `label_to_category` возвращает
метки модели обратно в COCO. Поэтому разметка с ID 1 тоже поддерживается.

Укажите корень изображений, содержащий `stamps/`. Например, в текущем окружении:

```bash
export CERAMIC_IMAGES='/Users/karinaevm/Documents/kleima/Клеймы'
```

На Kaggle/Colab замените путь. Пути file_name в COCO относительны к этому корню.
Не делайте новое random_split в каждой модели. Для изменённой разметки создайте
новую версию разбиения, сохраняя старую:

```bash
python -m detection_pipeline.data --annotations /path/instances_default.json \
  --images "$CERAMIC_IMAGES" --output detection_pipeline/splits/v2_seed42
```

Все 254 фото положительные. Подтверждённые негативы нужно включать в общий COCO
как images без annotations ДО разбиения. Не передавайте одни и те же negative_dirs
в уже разделённые train и val. Отсутствие аннотаций на непроверенном фото не означает,
что клейма нет. В текущем split доля срабатываний на отрицательных кадрах — null.

Независимого test нет. На val выбираем checkpoint/аугментации/порог, затем фиксируем
их и оцениваем на отдельно собранном test без предметов из train/val.

## Выбор аугментаций

В `configs/experiments.json` у каждой модели своё поле augmentation:

```json
"models": {
  "fasterrcnn_resnet50_fpn_v2": {"augmentation": "light"},
  "rtdetrv2_s": {"augmentation": "none"},
  "rtmdet_tiny": {"augmentation": "rotation"},
  "rtmdet_s": {"augmentation": "native"}
}
```

Это пример разных выборов; в поставляемом конфиге у всех стоит одинаковый light.
Поддерживаются none, light, rotation; у RT-DETR и RTMDet также native.

- none: без случайных преобразований.
- light: Brightness, LowContrast, HighContrast, Grayscale из кода коллег.
- rotation: Rotate90 и Brightness.
- native: собственный тренировочный рецепт библиотеки, без повторных аугментаций команды.

Свой список можно записать в той же записи модели:

```json
{"augmentation": "custom", "operations": [
  {"type": "Rotate90", "p": 0.5, "angles": [90, 180, 270]},
  {"type": "Defocus", "p": 0.2, "radius": [0.5, 1.5]}
]}
```

В val всегда нет случайных аугментаций; остаются детерминированные resize/format.
Одинаковый light использует одинаковые преобразования и параметры во всех трёх
стеках, до resize; независимые потоки генератора не гарантируют одни и те же
случайные числа на соответствующем батче. Поиск лучшего профиля автоматически
не выполняется. Сравнивайте согласованное число профилей на модель и записывайте
бюджет. Разные профили сравнивают решения «архитектура + рецепт обучения».

Исходный бюджет: seed 42, 50 эпох, batch 8, размер 640. Эпохи, batch и seed берутся
из experiments.json, если не переопределены. У Faster R-CNN resize сохраняет
пропорции, RTMDet дополняет до квадрата, штатный RT-DETR делает 640×640. Поэтому
даже одинаковые аугментации не делают весь входной рецепт абсолютно одинаковым.

## Запуск

Все команды выполнять из корня CeramicNet, Python >=3.10.
Используйте совместимую пару torch/torchvision для вашей CUDA; для официальных
RT-DETR и MMDetection устанавливайте их зависимости в отдельных окружениях.

```bash
python -m pip install -r detection_pipeline/requirements.txt
python -m detection_pipeline.train_fasterrcnn \
  --images "$CERAMIC_IMAGES" --splits detection_pipeline/splits/v1_seed42 \
  --output runs/fasterrcnn_light_seed42
```

Faster R-CNN загружает COCO weights, использует AdamW и сохраняет лучший checkpoint
по val mAP@[.50:.95], историю и общий файл метрик. Рабочий порог не участвует
в выборе checkpoint.

Для RT-DETRv2-S (ResNet-18), RTMDet-tiny и RTMDet-s сначала создайте конфиги:

```bash
python -m detection_pipeline.native_configs \
  --images "$CERAMIC_IMAGES" --splits detection_pipeline/splits/v1_seed42 \
  --output runs/configs \
  --rtdetr-repo /absolute/path/RT-DETR --mmdet-repo /absolute/path/mmdetection
```

Можно передать только один репозиторий. Выбор для каждой модели читается из
experiments.json; `--profile none` переопределяет его для контрольного опыта.
Вместе с конфигами сохраняется metadata JSON с разрешёнными операциями и бюджетом.

Запускайте через обёртку, которая регистрирует адаптеры и настраивает импорты:

```bash
python -m detection_pipeline.train_native rtdetr --repo /absolute/path/RT-DETR \
  --config runs/configs/rtdetrv2_s_light.yml \
  --seed 42 -t /absolute/path/rtdetrv2_s_coco_checkpoint.pth

python -m detection_pipeline.train_native mmdet --repo /absolute/path/mmdetection \
  --config runs/configs/rtmdet_tiny_light.py
```

У RTMDet базовый конфиг предобучает backbone на ImageNet. У Faster R-CNN здесь
предобучен COCO detector. Для сравнения с одинаковым происхождением предобучения
подключите соответствующие COCO detection weights RTMDet штатным load_from;
обязательно сохраните, какие веса использовались. Native checkpoint loading
и установка CUDA-зависимостей остаются частью окружения владельца модели.
В MMDetection технический test_dataloader указывает на val; независимым test он
не является.

## Общая оценка

Экспорт обученного checkpoint и оценка на val:

```bash
python -m detection_pipeline.predict_native rtdetr --repo /absolute/path/RT-DETR \
  --config runs/configs/rtdetrv2_s_light.yml --checkpoint /path/best.pth \
  --images "$CERAMIC_IMAGES" --annotations detection_pipeline/splits/v1_seed42/val.json \
  --output runs/rtdetr_val --tune-threshold

python -m detection_pipeline.predict_native mmdet --repo /absolute/path/mmdetection \
  --config runs/configs/rtmdet_tiny_light.py --checkpoint /path/best.pth \
  --images "$CERAMIC_IMAGES" --annotations detection_pipeline/splits/v1_seed42/val.json \
  --output runs/rtmdet_val --tune-threshold
```

Данные проходят только штатный val pipeline без случайных преобразований.
Сохраняются predictions.json и metrics.json. Для независимого test передайте его
JSON и не используйте --tune-threshold; рабочий порог переносите из val
через --threshold у predict_native или evaluate. Значение по умолчанию 0.5
нужно заменить выбранным на val.

Оценщик проверяет хеш разметки и присутствие КАЖДОГО image_id, даже если модель
не нашла рамок. Общий bbox — XYWH в исходных пикселях, score floor 0.001,
максимум 100 предсказаний. Адаптеры также поддерживают torchvision и YOLO.
Для YOLO используйте predict(conf=0.001, max_det=100), затем yolo_prediction и
save_predictions. Старый YOLO нужно переобучить/оценить на общем split.

Основная метрика — mAP@[.50:.95], дополнительно AP75. Рабочие precision/recall
считаются при IoU>=0.5 с однозначным сопоставлением bbox. Можно подобрать на val
максимальный recall при precision>=90% (пример требования). Если условие не
достигается, selected=null. Confidence не является гарантированной вероятностью
правильной локализации; для такой интерпретации нужна отдельная калибровка.
При 51 фото в val небольшая разница mAP будет шумной.

```bash
python -m detection_pipeline.compare runs/fasterrcnn_light_seed42/best_val_metrics.json \
  runs/rtdetr_val/metrics.json runs/rtmdet_val/metrics.json --output runs/comparison.csv
python -m unittest detection_pipeline.test_pipeline detection_pipeline.test_integration -v
```

Техническая проверка и ограничения описаны в VALIDATION.md.
Официальные интерфейсы: [torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html),
[RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch),
[MMDetection](https://mmdetection.readthedocs.io/en/latest/advanced_guides/customize_dataset.html).
