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

Скрипт `scripts.detection_test` выполняет инференс моделей через ONNX Runtime, рассчитывает метрики точности (mAP, Precision, Recall, F1) и замеряет скорость работы (Latency, FPS).

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

## 3. Запуск тестирования моделей поиска (Retrieval)

Скрипт `scripts.retrieval_test` выполняет бенчмарк моделей поиска клейм по экспертной разметке (`test_retrieval_annotations.json`), рассчитывает метрики ранжирования (Hit@1, Hit@3, Hit@5, Hit@10, MRR), фиксирует статистику работы детектора и выводит сравнительную сводную таблицу в консоль.

По умолчанию скрипт тестирует все модели, описанные в секции `models` конфигурационного файла.

Примеры запуска:
```bash
# протестировать все модели из конфига
python -m scripts.retrieval_test

# протестировать конкретную модель
python -m scripts.retrieval_test --model stamp_encoder_v2

# протестировать чистый ретривал без детектора (поиск по кадру целиком)
python -m scripts.retrieval_test --no-detect

# сохранить результаты в кастомные файлы
python -m scripts.retrieval_test --output-csv reports/custom_results.csv --summary-csv reports/custom_summary.csv
```

### Аргументы scripts.retrieval_test
| Аргумент | Тип | По умолчанию | Описание |
| :--- | :--- | :--- | :--- |
| `--annotations` | `Path` | `data/ceramic_dataset/annotations/test_retrieval_annotations.json` | Путь к JSON-файлу с тестовыми аннотациями. |
| `--config` | `Path` | `configs/retrieval/retrieval_config.yaml` | Путь к конфигурационному YAML файлу. |
| `--model` | `str` | `None` | Имя модели из реестра `models`. Если не указано, тестируются все доступные модели. |
| `--top-k` | `int` | `10` | Глубина поиска (количество возвращаемых эталонов). |
| `--no-detect` | флаг | `False` | Отключить YOLO-детектор (оценка чистого энкодера). |
| `--output-csv` | `Path` | `reports/retrieval_test_results.csv` | Путь к файлу сохранения подробных результатов по каждому запросу (CSV). |
| `--summary-csv` | `Path` | `reports/retrieval_summary.csv` | Путь к файлу сохранения итоговой сводной таблицы метрик (CSV). |

## 4. Визуализация экспертной разметки (Ground Truth)

Скрипт `scripts.visualize_retrieval_test` визуализирует разметку человека: формирует карточки сравнения исходной тестовой фотографии с эталонами из книг, указанными экспертом. Модели и инференс не используются — скрипт служит для визуального аудита тестовой выборки.

Примеры запуска:
```bash
# сформировать карточки для всех 20 тестовых запросов
python -m scripts.visualize_retrieval_test

# сформировать карточку для конкретного изображения и открыть её
python -m scripts.visualize_retrieval_test --query БП12-Г02-3659_5.png --show

# сформировать первые 5 карточек
python -m scripts.visualize_retrieval_test --limit 5
```

### Аргументы scripts.visualize_retrieval_test
| Аргумент | Тип | По умолчанию | Описание |
| :--- | :--- | :--- | :--- |
| `--annotations` | `Path` | `data/ceramic_dataset/annotations/test_retrieval_annotations.json` | Путь к JSON-файлу с разметкой человека. |
| `--output-dir` | `Path` | `reports/ground_truth_previews` | Папка для сохранения сформированных изображений карточек. |
| `--query` | `str` | `None` | Имя конкретного файла запроса для визуализации. |
| `--limit` | `int` | `None` | Ограничить количество формируемых карточек. |
| `--show` | флаг | `False` | Автоматически открыть первую созданную карточку в системе. |

## 5. Построение индекса эталонов

Скрипт `scripts.build_index` извлекает векторные представления эталонов из книг через ONNX-энкодер и сохраняет сжатый индекс (`*.npz`), содержащий L2-нормализованные эмбеддинги и метаданные каталогов.

Примеры запуска:
```bash
# построить индекс по текущему конфигу
python -m scripts.build_index --config configs/retrieval/retrieval_config.yaml

# переопределить путь к весам энкодера и выходному индексу
python -m scripts.build_index --encoder weights/retrieval_weights/stamp_encoder_v2.onnx --output data/books_index/stamp_books_index_v2.npz
```

### Аргументы scripts.build_index
| Аргумент | Тип | По умолчанию | Описание |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `None` | Путь к конфигурационному YAML файлу (если `None`, берется дефолтный). |
| `--books` | `str` | `None` | Переопределить путь к директории с книгами. |
| `--meta` | `str` | `None` | Переопределить путь к `meta.csv`. |
| `--encoder` | `str` | `None` | Переопределить путь к ONNX-модели энкодера. |
| `--output` | `str` | `None` | Путь для сохранения итогового `.npz` файла индекса. |
| `--batch-size` | `int` | `16` | Размер батча при кодировании эталонов. |
| `--limit` | `int` | `None` | Ограничить количество обрабатываемых эталонов (для отладки). |
