# Общее логирование: детекция и сопоставление клейм

ExperimentRun работает независимо от Dataset, модели и формата разметки.
Один запуск соответствует одной модели, задаче и конкретному набору split-файлов.

В ClearML создаются проекты CeramicNet/detection и CeramicNet/matching.
В каждом запуске сохраняются параметры, конфигурация, идентификатор разбиения,
графики метрик и явно переданные JSON/CSV артефакты. Идентификатор — SHA256
содержимого фактических train/val файлов; абсолютный путь на него не влияет.
Детекция и matching могут использовать разные файлы и пропорции train/val.
Одинаковый seed сам по себе не подтверждает одинаковое разбиение.
Fingerprint проверяет содержимое переданных split/manifest файлов. Для matching
с кропами сохраняйте в отдельном dataset_manifest SHA256 изображений и версию
детектора/кропов и передавайте этот manifest в split_files: одни пути в CSV
не обнаружат замену содержимого изображения.

## Установка и режимы

```bash
python -m pip install -r requirements-tracking.txt
clearml-init
```

clearml-init один раз настраивает сервер и ключи SDK. Используйте свой ClearML
аккаунт/сервер; ключи сохраняются вне отслеживаемого кода. Поддерживаются штатные
переменные окружения CLEARML_API_HOST, CLEARML_WEB_HOST, CLEARML_FILES_HOST,
CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY. В ноутбуках Kaggle/Colab берите
значения из Secrets, а не из коммитов. Этот модуль не читает ключи из старых ноутбуков.

- local (по умолчанию): локальные JSON/JSONL, ClearML не требуется.
- offline: настоящий offline-запуск ClearML без доступа к серверу, плюс локальные файлы.
- clearml: отправка явных метрик, конфигурации и артефактов в настроенный ClearML.

Локально всё лежит в OUTPUT/tracking/: run.json, config.json, split.json,
metrics.jsonl, summary.json, artifacts.jsonl. Для каждого опыта выбирайте новый
OUTPUT. Завершение/ошибка фиксируются в run.json; в распределённом запуске пишет
только rank 0. Инициализация SDK сохраняет уже созданные состояния RNG.
Автоматический перехват framework/консоли отключён; графики отправляются явно,
с осью epoch или iteration. Веса не загружаются автоматически.

ClearML offline-folder записывается в tracking/run.json. После настройки сервера
его можно импортировать штатной командой clearml-task --import-offline-session PATH.

## Детекция

```bash
python -m detection_pipeline.train_fasterrcnn \
  --images "$CERAMIC_IMAGES" --splits detection_pipeline/splits/v2_seed42_neg50 \
  --output runs/fasterrcnn_light_clearml --tracking clearml \
  --run-name fasterrcnn-light-seed42

python -m detection_pipeline.train_native rtdetr --repo /path/RT-DETR \
  --config runs/configs/rtdetrv2_s_light.yml --tracking clearml \
  --run-name rtdetr-light-seed42 -t /path/coco_checkpoint.pth

python -m detection_pipeline.train_native mmdet --repo /path/mmdetection \
  --config runs/configs/rtmdet_tiny_light.py --tracking clearml \
  --run-name rtmdet-tiny-light-seed42
```

Для проверки без сервера замените clearml на offline.
Faster R-CNN логирует общий loss и компоненты, lr, mAP/AP50/AP75/AR,
precision/recall/F1 при пороге 0.5, длительность эпохи и лучший checkpoint по mAP.
Сохраняются history и метрики/предсказания лучшей эпохи.

У RT-DETR графики читаются из штатного log.txt после каждой эпохи; у MMDetection
из новых vis_data/scalars.json по мере записи. Требуется стандартный LocalVisBackend,
который сохраняется в сгенерированных конфигурациях. Старые строки логов пропускаются,
неполная строка дочитывается после завершения записи. Поддерживаются стандартные
форматы официального lyuwenyu RT-DETRv2 и MMEngine 0.10.x. При изменении native
формата логов потребуется адаптация парсера. --work-dir/--output-dir и основные
--cfg-options/-u overrides учитываются в сохранённой конфигурации и хешах split.

Метрики native validation идут в val_native: это мониторинг родной библиотеки.
Итоговое сравнение моделей выполняйте общим evaluate/compare на одном val/test,
как раньше. Онлайн-логирование не меняет разбиение или критерий выбора checkpoint.

## Следующая задача: изображение → название/ID клейма

Подготовьте свои train/val, например CSV с путями изображений и mark_id.
Передайте именно эти файлы; COCO и split детектора здесь не требуются.
Подключение к будущему trainer выглядит так:

```python
from experiment_tracking import ExperimentRun

with ExperimentRun(
    task_kind="matching",
    run_name="encoder-baseline-seed42",
    output_dir="runs/matching_baseline",
    config={"model": "your_encoder", "seed": 42, "protocol": "closed_set_retrieval"},
    split_files={"train": "matching/splits/train.csv", "val": "matching/splits/val.csv"},
    mode="clearml",
) as run:
    # Внутри вашего реального цикла обучения:
    # run.log_metrics(train_metrics, epoch + 1, namespace="train")
    # run.log_metrics(val_metrics, epoch + 1, namespace="val")
    # run.summary(best_recall_at_1=best_recall)
    # run.artifact("best_metrics", "runs/matching_baseline/metrics.json")
    pass
```

Сами matching trainer/модель ещё не реализованы. Логгер принимает произвольные
численные метрики: для retrieval обычно Recall@1/@3/@5 и MRR; для классификации
можно передать accuracy/macro-F1. Протокол нужно выбрать под задачу, а не менять
его вместе с моделью. Все сопоставляемые модели внутри одной задачи должны
использовать одинаковые train/val и одинаковую галерею кандидатов; при необходимости
добавьте gallery.csv в split_files, чтобы его версия также вошла в fingerprint.

Другой split для отдельной задачи допустим. Для финальной оценки каскада
«детекция → сопоставление» тестовые предметы/изображения должны быть исключены
из обучения и настройки обоих этапов. Укажите, проверяете известные клейма на
новых фотографиях или обобщение на новые ID; эти протоколы дают разные оценки.
Разные фото/кропы одного предмета сохраняйте в одной группе.

Проверка: python -m unittest experiment_tracking.test_tracking -v.
Источники: [ClearML Tasks](https://clear.ml/docs/latest/docs/fundamentals/task/),
[SDK Task](https://clear.ml/docs/latest/docs/references/sdk/task/),
[offline](https://www.clear.ml/docs/latest/docs/faq/).
