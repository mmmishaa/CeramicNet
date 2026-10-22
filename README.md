# CeramicNet

Общий пайплайн детекции клейм для Faster R-CNN ResNet-50 FPN v2,
RT-DETRv2-S и RTMDet-tiny/s. Построен поверх класса и аугментаций из ветки `datasets`.

Начать с [ноутбука](train_pipelines/universal_detection.ipynb).
Запуск и соединение частей описаны в [инструкции](detection_pipeline/README.md).

- Общий split: **train 203 фото / val 51 фото**, seed **42**, предметы не пересекаются.
- Каждая модель выбирает аугментации в [configs/experiments.json](configs/experiments.json).
- Метрики считаются общим оценщиком; val не получает случайные аугментации.
- [Общий логгер ClearML](experiment_tracking/README.md) подходит для detection и будущего matching с отдельными split.
- Фотографии, веса и ключи доступа в новые файлы не включены.

В текущем split нет независимого test и отрицательных изображений.
Реальное качество моделей измеряется после обучения; проверка кода описана
в [VALIDATION.md](detection_pipeline/VALIDATION.md).
