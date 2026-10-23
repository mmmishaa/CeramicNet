# RTMDet-s

[rtmdet_s_train.ipynb](rtmdet_s_train.ipynb) обучает официальный RTMDet-s
через общий пайплайн CeramicNet: подготовленные split, аугментации команды,
общий логгер и оценщик. Основа — ветка `pipeline`.

Данные: `detection_pipeline/splits/v2_seed42_neg50`. Train: 203 stamps + 40
негативов; val: 51 stamps + 10 негативов. По 25 негативов из `empty` и
`ornaments_without_copies` уже выбраны с `random_state=42` и записаны в JSON.
Ноутбук не выбирает их повторно и не добавляет через `negative_dirs`.

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
Файлы и SHA256 проверяются по сохранённым JSON. Нужны именно выбранные 50 негативов.

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

Есть 8 пересечений предметов между позитивами и негативами разных частей
сохранённого split; manifest фиксирует их. Независимого test нет. Порог
подбирается только на val; метрики короткого прогона не являются итоговым
качеством модели.

Официальные [конфиг и веса RTMDet-s](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/metafile.yml),
[сборка MMCV](https://github.com/open-mmlab/mmcv/blob/v2.1.0/docs/en/get_started/build.md).

## Локальная проверка 23 октября 2026

На MacBook Air M5 выполнены все ячейки ноутбука: одна эпоха, batch 2,
640×640, все 243 train и 61 val фото. Настоящие загрузчики MMDetection сохранили
40/10 негативов. Созданы лучшие веса, общий логгер со статусом `completed`,
предсказания для всех 61 фото, метрики и три иллюстрации. Все 22 теста
`detection_pipeline.test_pipeline` и `detection_pipeline.test_integration` прошли.

Общий оценщик после пробной эпохи: mAP@[.50:.95] = 0.0432, AP50 = 0.0673.
При пороге 0.5 детекций нет; порог с precision ≥ 90% не найден. Это проверка
полного цикла запуска с новой головой на один класс, не итоговое качество модели.
Полное обучение на 50 эпох в этой проверке не выполнялось.
