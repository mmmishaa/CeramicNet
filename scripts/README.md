# Инструкция по работе со скриптами

## 1. Скачивание артефактов (веса и датасет)

Скрипт `scripts.download_artifacts` загружает ONNX-модели, индекс книг, `meta.csv` и ZIP-архивы с книгами и датасетом из Google Drive, распределяет веса по папкам и распаковывает данные.

По умолчанию (без флагов) скачивается всё. Чтобы скачать только часть артефактов, укажите нужные флаги, их можно комбинировать.

Примеры запуска:
```bash
# скачать всё
python -m scripts.download_artifacts --folder_id <ID папки на гугл диске>

# посмотреть, что лежит в папке, ничего не скачивая
python -m scripts.download_artifacts --folder_id <ID папки на гугл диске> --list

# скачать только веса детекции
python -m scripts.download_artifacts --folder_id <ID папки на гугл диске> --detection

# скачать только датасет
python -m scripts.download_artifacts --folder_id <ID папки на гугл диске> --dataset
```

### Аргументы scripts.download_artifacts
| Аргумент | Тип | Обязательный | Описание |
| :--- | :---: | :---: | :--- |
| `--folder_id` | `str` | Да | ID папки на Google Drive с весами и датасетом. |
| `--list` | флаг | Нет | Только показать файлы в папке и их категории, затем выйти. Ничего не скачивается. |
| `--detection` | флаг | Нет | Скачать веса детекции (`*.onnx`) в `weights/detection_weights`. |
| `--retrieval` | флаг | Нет | Скачать веса retrieval (`*.onnx`) в `weights/retrieval_weights`. |
| `--index` | флаг | Нет | Скачать индекс книг (`*.npz`) в `data/books_index`. |
| `--books` | флаг | Нет | Скачать и распаковать `books.zip` в `data/books`. |
| `--dataset` | флаг | Нет | Скачать и распаковать `ceramic_dataset.zip` в `data/ceramic_dataset`. |

**Примечания:**
- Если не указан ни один из флагов выбора (`--detection`, `--retrieval`, `--index`, `--meta`, `--books`, `--dataset`), скачиваются все категории.
- Если `data/books` или `data/ceramic_dataset` уже существуют и не пусты, соответствующий архив не скачивается и не распаковывается повторно.
- Уже имеющиеся файлы весов не перезаписываются.

## 2. Запуск тестирования моделей детекции

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