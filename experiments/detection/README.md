# Эксперименты по детекции

- **YOLO26s:** [v1](yolo/v1/README.md) · [v2](yolo/v2/README.md).
- **RTMDet-s:** [v3](RTMDet/README.md).
- **RT-DETRv2-S:** [проверка процессора и метрики ONNX](rtdetr/README.md) на восстановленной валидации ноутбука: 61 фото, COCO mAP@50 0.9610, mAP@50–95 0.7358.

В каждом отчёте — метрики, встроенные картинки, сохранённые численные данные, источник и ссылка на ноутбук. Примеры обновлены на выбранных пользователем фотографиях: в README — четыре val-фото по RTMDet, в отдельных галереях — все 23 с пометками train/val. Исторический split YOLO пока не подтверждён. [Полный список и проверка разбиения](selected_images.md).

**Это архив отдельных опытов, не рейтинг моделей.** YOLO v1 проверялся на 61 фото, v2 и RTMDet — на 145. Состав и разметка YOLO и RTMDet не подтверждены как одинаковые: например, в журналах YOLO v2 указано 51 bbox на val, у RTMDet — 52. Оценщики также различаются. Для сравнения нужен единый сохранённый набор и один протокол оценки.

Перестроить графики обучения и порогов из сохранённых CSV (без обучения и ClearML):

```bash
python -m pip install matplotlib
python experiments/detection/render_figures.py
```

Новый опыт сохранять отдельно, не заменяя старые метрики. В Git добавлять небольшие JSON/CSV/PNG и код/ссылку обучения; веса, фотографии датасета, ключи и полные логи оставлять во внешнем хранилище.

Источник общего корпуса — [папка команды на Drive](https://drive.google.com/drive/folders/1MiKPCtREOFUfp6PxKqZUrviIB-TUI1Un). Снимок `ceramic_dataset.zip` содержит 722 фото, 254 размеченных фото и 257 рамок; в нём нет готовых списков train/val. [SHA256 и проверка выбранных фото](dataset_source.json).

Пересоздать новые иллюстрации (пути к ONNX и корню изображений задать локально):

```bash
python -m pip install onnxruntime opencv-python numpy pillow matplotlib
python experiments/detection/generate_examples.py --images-root /path/to/images \
  --yolo-v1 /path/to/yolo26s_v1.onnx --yolo-v2 /path/to/yolo26s_v2.onnx \
  --rtmdet /path/to/rtmdet_s_best.onnx
```

Веса должны совпадать с SHA256 из `selected_images.json`; байты всех фото также проверяются. Предсказания и параметры иллюстраций сохраняются в `selected_predictions.json` каждого опыта. Исторические метрики обучения этими иллюстрациями не заменяются.
