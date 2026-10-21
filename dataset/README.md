# StampDataset

`StampDataset` читает COCO и отдаёт картинку и боксы клейм.

```python
StampDataset(
    root_dir,
    transform=None,
    standard_size=None,
    model="default",
    negative_dirs=None,
    ann_file=None,
    image_root=None,
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

## Метки классов

COCO category_id может начинаться с 0 или быть разреженным (например, 7 и 42).
model="default" возвращает исходные ID. faster_rcnn возвращает непрерывные метки
с 1 (0 — фон), rtdetr и rtmdet — с 0. label_to_category хранит обратное отображение.
У rtdetr bbox нормализуется в CXCYWH; у остальных вариантов остаётся XYXY.

## negative_dirs

Это папки с вручную проверенными фотографиями без клейма. Одно название папки `ornaments` или `empty` не подтверждает отсутствие клейма. Такие кадры попадают в датасет с пустыми `boxes` и `labels`. Файл, чьё имя уже есть в COCO, второй раз не добавляется. Кадр из json без боксов тоже остаётся в выборке.

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
)
```

## Общие train и val

В ветке pipeline разделение выполняется один раз по предметам с seed 42,
а train и val читают разные сохранённые JSON. Внутри класса разбиения нет.
Полная инструкция: [detection_pipeline/README.md](../detection_pipeline/README.md).

```python
from detection_pipeline.datasets import make_datasets, make_loader

train_ds, val_ds = make_datasets(
    images_root="/path/to/images",  # здесь находится stamps/
    split_dir="detection_pipeline/splits/v1_seed42",
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

Проверенные негативы включайте в общий COCO до split. Передача одинаковых
negative_dirs в уже разделённые train/val создаст пересечение. В val/test не
должно быть случайных аугментаций. Независимого test в текущем 80/20 split нет.
