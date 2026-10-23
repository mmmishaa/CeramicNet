# CeramicNet

Общий пайплайн детекции клейм для Faster R-CNN ResNet-50 FPN v2,
RT-DETRv2-S и RTMDet-tiny/s. Построен поверх класса и аугментаций из ветки `datasets`.

Начать с [ноутбука](train_pipelines/universal_detection.ipynb).
Запуск и соединение частей описаны в [инструкции](detection_pipeline/README.md).
Обучение RTMDet-s: [отдельный ноутбук](train_pipelines/detection_models/RTMDet-s/rtmdet_s_train.ipynb)
и [локальная установка](train_pipelines/detection_models/RTMDet-s/README.md).
Проверка RTMDet-s и галерея всех stamps: [тестовый ноутбук](test_pipelines/detection_models/RTMDet-s/rtmdet_s_test.ipynb).

- Общий split: **train 259 фото / val 145 фото (56/94 негативов)**, seed **42**.
- Каждая модель выбирает аугментации в [configs/experiments.json](configs/experiments.json).
- Метрики считаются общим оценщиком; val не получает случайные аугментации.
- [Общий логгер ClearML](experiment_tracking/README.md) подходит для detection и будущего matching с отдельными split.
- Фотографии, веса и ключи доступа в новые файлы не включены.

Негативы из empty и ornaments_without_copies отдельно разделены на 80% резерва
для обучения и 20% val. Из общего резерва берётся 15% для фактического обучения:
374 в резерве, 56 в train, 318 не используются; 94 в val. Seed 42.
Списки резерва и неиспользованных кадров сохранены отдельно и не читаются
обычными загрузчиками. Файлы резерва не попадают в val.
Разные снимки одного предмета могут оказаться в разных частях; список пересечений
и подробности — в manifest разбиения.
Независимого test пока нет.
Реальное качество моделей измеряется после обучения; проверка кода описана
в [VALIDATION.md](detection_pipeline/VALIDATION.md).
