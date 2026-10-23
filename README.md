# CeramicNet

Общий пайплайн детекции клейм для Faster R-CNN ResNet-50 FPN v2,
RT-DETRv2-S и RTMDet-tiny/s. Построен поверх класса и аугментаций из ветки `datasets`.

Начать с [ноутбука](train_pipelines/universal_detection.ipynb).
Запуск и соединение частей описаны в [инструкции](detection_pipeline/README.md).
Обучение RTMDet-s: [отдельный ноутбук](train_pipelines/detection_models/RTMDet-s/rtmdet_s_train.ipynb)
и [локальная установка](train_pipelines/detection_models/RTMDet-s/README.md).

- Общий split: **train 243 фото / val 61 фото (40/10 негативов)**, seed **42**.
- Каждая модель выбирает аугментации в [configs/experiments.json](configs/experiments.json).
- Метрики считаются общим оценщиком; val не получает случайные аугментации.
- [Общий логгер ClearML](experiment_tracking/README.md) подходит для detection и будущего matching с отдельными split.
- Фотографии, веса и ключи доступа в новые файлы не включены.

В split выбраны 25 негативов из empty и 25 из ornaments_without_copies;
общие 50 фото разделены случайно: 40 train и 10 val, random_state=42.
Есть 8 пересечений предметов между негативами и положительными фото в разных частях;
подробности и список — в manifest разбиения.
Независимого test пока нет.
Реальное качество моделей измеряется после обучения; проверка кода описана
в [VALIDATION.md](detection_pipeline/VALIDATION.md).
