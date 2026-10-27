# YOLO26s · v2

50 эпох · 640 px · batch 16 · seed 0 · AdamW · Tesla T4. Train **248**, val **145** фото (94 негативов).

| mAP50–95 | AP50 | Precision | Recall |
| --- | --- | --- | --- |
| 0.7022 | 0.9349 | 0.8793 | 0.8571 |

Метрики `best.pt` из ClearML; независимого test нет.

![Обучение YOLO v2](learning_curves.png)

![Выбранные примеры детекции YOLO v2](predictions_examples.png)

Четыре выбранных фото из **RTMDet val**, confidence **0.75** из конфигурации общего приложения. Их split для исторического YOLO не подтверждён; картинки не являются новой оценкой качества. P/R в таблице — исторические значения валидатора.

[PR-кривая](precision_recall.png) · [Метрики](metrics.json) · [История обучения](training_history.csv) · [ClearML](https://app.clear.ml/projects/d6209850b8fb417bb10decb4dfaccfd8/tasks/f1e9824d5941477989f9641f42e62622/output/scalars)

[Ноутбук обучения](../../../../notebooks/detection/yolo/yolo_v2_train_pipeline.ipynb) · [Как воспроизвести](reproduce.md). Код не дублируется; [источники и ограничения](provenance.json).

[Все 23 выбранных фото](examples/README.md) · [Проверка разбиения](../../selected_images.md) · [Исходные примеры ноутбука](notebook_examples.png).
