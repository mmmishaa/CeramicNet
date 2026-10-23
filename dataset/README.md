# StampDataset

`StampDataset` читает COCO и отдаёт картинку и боксы клейм.

```python
StampDataset(
    root_dir,
    transform=None,
    standard_size=None,
    model="default",
    negative_dirs=None,
    negative_ratio=0.2,
    ann_file=None,
    image_root=None,
    negative_seed=42,
)
```

| Параметр | Описание |
|---|---|
| `root_dir` | Папка `stamps_coco_merged`: в ней `annotations/instances_default.json` и `images/default/`. |
| `transform` | Аугментации. Вызываются при каждом чтении кадра. `None` — кадр как в файле. |
| `standard_size` | `(H, W)`. Кадр вписывается в этот размер, пустые поля белые, боксы сдвигаются вместе с пикселями. `None` — размер файла. |
| `model` | `"default"`, `"faster_rcnn"`, `"rtdetr"` или `"rtmdet"`. От этого зависит формат боксов на выходе. |
| `ann_file` | Явный COCO JSON для train или val. None — прежний путь внутри root_dir. |
| `image_root` | Корень относительно file_name из COCO. None — images/default внутри root_dir. |
| `negative_dirs` | Папки снимков, на которых клейма нет. `None` — только кадры из json. |
| `negative_ratio` | Лимит внешних негативов относительно числа кадров COCO. По умолчанию 0.2; 0 — не добавлять, 1.0 — столько же, None — все доступные файлы. |
| `negative_seed` | Seed отбора внешних негативов: 42 по умолчанию; 0 воспроизводит прежний отбор datasets. |

## Метки классов

COCO category_id может начинаться с 0 или быть разреженным (например, 7 и 42).
model="default" возвращает исходные ID. faster_rcnn возвращает непрерывные метки
с 1 (0 — фон), rtdetr и rtmdet — с 0. label_to_category хранит обратное отображение.
У rtdetr bbox нормализуется в CXCYWH; у остальных вариантов остаётся XYXY.

## negative_dirs

Это папки с вручную проверенными фотографиями без клейма. Одно название папки `ornaments` или `empty` не подтверждает отсутствие клейма. Такие кадры попадают в датасет с пустыми `boxes` и `labels`. Файл, чьё имя уже есть в COCO, второй раз не добавляется. Кадр из json без боксов тоже остаётся в выборке.

`negative_ratio` считает лимит от числа кадров COCO, не от размера папок.
При 254 кадрах и negative_ratio=1.0 берётся до 254 проверенных негативов;
при значении по умолчанию 0.2 — до 51 (round(254 * 0.2)). None берёт все
доступные файлы. Аннотированные имена и повторяющиеся имена кандидатов исключаются
до отбора. negative_seed=42 задаёт воспроизводимую выборку. Для общего benchmark
отбор выполняется при подготовке данных до разбиения, а не в каждом Dataset.

## Трансформации

Шаг принимает uint8 RGB `(H, W, 3)` и словарь с `boxes` xyxy и возвращает их же. `p` — вероятность применить шаг при чтении кадра. Отражения двигают боксы, цветовые шаги их не трогают. Несколько шагов собирает `Compose`.

| Шаг | Что делает |
|---|---|
| `HorizontalFlip(p=0.5)` | Отражает слева направо и переносит бокс. |
| `Rotate90(p=0.5, angles=(90, 180, 270))` | Поворачивает против часовой стрелки и преобразует bbox. |
| `VerticalFlip(p=0.5)` | Отражает сверху вниз и переносит бокс. |
| `Grayscale(p=0.5)` | Делает кадр чёрно-белым. Три канала остаются, в каждом одна яркость. |
| `Brightness(p=0.5, factor=(0.6, 1.4))` | Умножает пиксели на случайный множитель. `1` — без изменений, меньше темнее, больше светлее. |
| `LowContrast(p=0.5, factor=(0.5, 0.8))` | Снижает контраст. |
| `HighContrast(p=0.5, factor=(1.2, 1.5))` | Повышает контраст. |
| `Defocus(p=0.5, radius=(1.0, 3.0))` | Размывает кадр. `radius` — число или диапазон в пикселях. |

```python
from dataset import StampDataset
from dataset.transforms import Brightness, Compose, Grayscale, HorizontalFlip, VerticalFlip

dataset = StampDataset(
    "stamps_coco_merged",
    transform=Compose([
        HorizontalFlip(0.5),
        VerticalFlip(0.5),
        Grayscale(0.3),
        Brightness(0.5),
    ]),
    standard_size=(640, 640),
    negative_dirs=["/path/verified_negatives"],
    negative_ratio=1.0,
    negative_seed=42,
)
```

## Коллаторы

get_collate_fn("default"), get_collate_fn("faster_rcnn") и get_collate_fn("rtmdet")
возвращают список изображений и список targets, сохраняя image_id и метаданные.
get_collate_fn("rtdetr") складывает одинаковые по размеру изображения в один tensor
и возвращает targets с boxes/class_labels. При разном размере кадров нужен
standard_size. Этот вариант предназначен для кода, ожидающего именно такой контракт.

Официальный lyuwenyu RT-DETR в нашем benchmark использует свой штатный
BatchImageCollateFunction: ему нужны labels, image_id и orig_size. Переходник
TeamCocoDetection сохраняет этот формат; generic rtdetr-коллатор туда не подставляется.

## Общие train и val

В ветке pipeline положительные фото разделяются один раз по предметам с seed 42,
а train и val читают разные сохранённые JSON. Внутри класса разбиения нет.
Полная инструкция: [detection_pipeline/README.md](../detection_pipeline/README.md).

```python
from detection_pipeline.datasets import make_datasets, make_loader

train_ds, val_ds = make_datasets(
    images_root="/path/to/images",  # здесь находится stamps/
    split_dir="detection_pipeline/splits/v3_seed42_reserve80_active15",
    profile="light",              # настоящие аугментации команды
)
train_loader = make_loader(train_ds, train=True, seed=42, batch_size=8)
val_loader = make_loader(val_ds, train=False, seed=42, batch_size=8)
```

CocoStampDataset здесь — адаптер над StampDataset, а не второй читатель COCO.
Он добавляет tensor image_id/area/iscrowd для общего цикла torchvision.
Профили none/light/rotation и произвольный список операций собираются через
`dataset.transforms.factory.build_augmentations`.
В RT-DETR используются тот же StampDataset и штатный postprocess; в MMDetection
те же преобразования команды встроены в штатный CocoDataset.

Текущий split v3_seed42_reserve80_active15 использует резервирование негативов:
из каждой папки empty/ornaments_without_copies 80% закреплены за обучающим
резервом, 20% — за val. Из общего резерва 374 кадров в фактический train выбраны
15% (56), остальные 318 не участвуют в обучении и оценке. В val 94 негатива.
negative_train_reserve.json хранит весь резерв, negative_unused.json — неиспользованную
часть. Dataset читает только train.json/val.json: резерв автоматически не добавляется.
При увеличении доли обучения внутри этого резерва val остаётся неизменной.
Всё воспроизводится с seed 42; файлы резерва и val не пересекаются по путям и SHA256.
Разные фото одного предмета могут пересекаться между частями: manifest.negatives
содержит group_overlap, negative_group_overlap и reserve_val_group_overlap.
Негативы записаны в COCO как images без annotations. Для нового разбиения:
detection_pipeline.data --base-splits с --negative-reserve-fraction 0.8
и --negative-active-fraction 0.15; источники передаются через --negative-dirs.
Передача одинаковых negative_dirs в уже разделённые train/val создаст пересечение. В val/test не
должно быть случайных аугментаций. Независимого test в текущем 80/20 split нет.
