# Инструкция по работе со скриптами

## 1. Скачивание артефактов (веса и датасет)

Скрипт `scripts.download_artifacts` загружает ONNX-модели и ZIP-архив с датасетом из Google Drive, распределяет веса по папкам и распаковывает данные.

Пример запуска:
```bash
python -m scripts.download_artifacts --folder_id <ID папки на гугл диске>
```

### Аргументы scripts.download_artifacts
| Аргумент | Тип | Обязательный | Описание |
| :--- | :---: | :---: | :--- |
| `--folder_id` | `str` | Да | ID папки на Google Drive с весами и датасетом. |

## 2. Запуск тестирования моделей деткции

Скрипт scripts.detection_test выполняет инференс моделей через ONNX Runtime, рассчитывает метрики точности (mAP, Precision, Recall, F1) и замеряет скорость работы (Latency, FPS).

Пример запуска:
```bash
python -m scripts.detection_test
```

### Аргументы scripts.detection_test
| Аргумент | Тип | По умолчанию | Описание |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `configs/detection/detection_test_config.yaml` | Путь к конфигурационному YAML файлу. |
| `--model` | `str` | `None` | Имя конкретной модели из конфига для запуска. |
| `--conf` | `float` | `None` | Переопределить порог уверенности (`conf_threshold`). |
| `--iou` | `float` | `None` | Переопределить порог IOU (`iou_threshold`). |
| `--output_csv` | `str` | `reports/detection_test_results.csv` | Путь к файлу сохранения результатов (CSV). |