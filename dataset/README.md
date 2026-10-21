# StampDataset

`StampDataset` читает COCO и отдаёт картинку и боксы клейм.

```python
StampDataset(
    root_dir,
    transform=None,
    standard_size=None,
    model="default",
    negative_dirs=None,
)
```

| Параметр | Описание |
|---|---|
| `root_dir` | Папка `stamps_coco_merged`: в ней `annotations/instances_default.json` и `images/default/`. |
| `transform` | Аугментации. Вызываются при каждом чтении кадра. `None` — кадр как в файле. |
| `standard_size` | `(H, W)`. Кадр вписывается в этот размер, пустые поля белые, боксы сдвигаются вместе с пикселями. `None` — размер файла. |
| `model` | `"default"`, `"faster_rcnn"`, `"rtdetr"` или `"rtmdet"`. От этого зависит формат боксов на выходе. |
| `negative_dirs` | Папки снимков, на которых клейма нет. `None` — только кадры из json. |

## negative_dirs

Это папки PNG без клейма, например `ornaments` и `empty`. Такие кадры попадают в датасет с пустыми `boxes` и `labels`. Файл, чьё имя уже есть в COCO, второй раз не добавляется. Кадр из json без боксов тоже остаётся в выборке.

## Трансформации

Шаг принимает uint8 RGB `(H, W, 3)` и словарь с `boxes` xyxy и возвращает их же. `p` — вероятность применить шаг при чтении кадра. Отражения двигают боксы, цветовые шаги их не трогают. Несколько шагов собирает `Compose`.

| Шаг | Что делает |
|---|---|
| `HorizontalFlip(p=0.5)` | Отражает слева направо и переносит бокс. |
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
    negative_dirs=["Клеймы /ornaments", "Клеймы /empty"],
)
```

## Train и test

Датасет сам выборку не делит. Индексы уже созданного `dataset` перемешиваются, часть уходит в test, остальные остаются в train. Аугментации нужны только train, поэтому это два объекта `StampDataset` с одним `root_dir` и одними индексами: `transform` задан только у train. `Subset` хранит индексы.

`collate_stamps` собирает батч списком картинок и списком словарей: число боксов на кадре разное, в один тензор их не склеить.

```python
import random

from torch.utils.data import DataLoader, Subset

from dataset import StampDataset, collate_stamps
from dataset.transforms import Brightness, Compose, Grayscale, HorizontalFlip, VerticalFlip

order = list(range(len(dataset)))
random.Random(0).shuffle(order)
n_test = int(len(order) * 0.2)
test_index, train_index = order[:n_test], order[n_test:]

common = dict(
    root_dir="stamps_coco_merged",
    standard_size=(640, 640),
    model="default",
    negative_dirs=["Клеймы /ornaments", "Клеймы /empty"],
)
train_data = StampDataset(
    **common,
    transform=Compose([
        HorizontalFlip(0.5),
        VerticalFlip(0.5),
        Grayscale(0.3),
        Brightness(0.5),
    ]),
)
test_data = StampDataset(**common)
train_set = Subset(train_data, train_index)
test_set = Subset(test_data, test_index)

loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=collate_stamps)
```
