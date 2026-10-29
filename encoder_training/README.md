# Обучение энкодера клейм

DINOv2 ViT-S/14, дообученный искать книжный эталон по фото клейма. Готовый ONNX кладётся в
`weights/retrieval_weights/student_encoder.onnx` и работает с текущим `retrieval_config.yaml`
(mean `[0,0,0]`, std `[1,1,1]`, `enhance_contrast: true`).

## Как устроено

- Пара для обучения: «как на фото» (`photo_aug.py`: цветная краска на глазури, поворот на любой угол,
  изгиб, стёртость, сколы, кракелюр, блики, низкое разрешение) ↔ чистый эталон.
- Функция потерь InfoNCE: остальные эталоны батча служат негативами. Температура 0.05.
- Разморожены последние 6 блоков из 12 и финальная нормировка. AdamW, lr 3e-5, 20 эпох, batch 64.
- Предобработка та же, что в `CeramicPipeline` и `scripts.build_index`: `crop_stamp` (квадрат, поля 10%, 224),
  затем `enhance_stamp_contrast`. Перевод в ч/б и нормировка ImageNet — внутри модели.

## Данные

- Эталоны берутся из `retrieval_config.yaml` (`data/books`, `data/books/meta.csv`), то есть те же, что попадут в индекс.
- Оценка идёт на реальных фото: COCO `stamps_coco_merged` (по умолчанию `data/stamps_coco_merged`, иначе `--real-coco`)
  и `real_factory_labels.csv` — 98 кропов, размеченных по фабрике по надписям (разметка не экспертная).
  `hit@k` — в top-k есть эталон той же фабрики.

## Запуск (из корня репозитория)

```bash
python -m encoder_training.train --name s_u6_e20 --real-coco <путь>/stamps_coco_merged
python -m encoder_training.evaluate --checkpoint runs/encoder/s_u6_e20/last.pt --real-coco <путь>/stamps_coco_merged
python -m encoder_training.show_topk runs/encoder/s_u6_e20/last.pt --out-dir reports/topk --real-coco <путь>/stamps_coco_merged
python -m encoder_training.export_onnx runs/encoder/s_u6_e20/last.pt --out weights/retrieval_weights/student_encoder.onnx
python -m scripts.build_index
```

Чекпоинт, `history.json` (метрики по эпохам) и `args.json` пишутся в `runs/encoder/<name>/`.
После смены весов индекс обязательно пересобрать.

## Результат (98 реальных фото, без поворотов на запросе)

| | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| Исходный DINOv2 ViT-S | 0.19 | 0.36 | 0.46 |
| `s_u6_e20` (10% эталонов отложено) | 0.66 | 0.78 | 0.84 |

Случайная выдача: hit@3 ≈ 0.09. Повороты запроса (максимум, среднее, голосование) после дообучения ничего не дают.
