# StampDataset

`StampDataset` — класс `torch.utils.data.Dataset` для кадров с клеймами.
Он читает COCO и отдаёт картинку и боксы. Параметр `model` выбирает, в каком
виде эти боксы попадут в голову: `default` для YOLO, отдельные варианты для
Faster R-CNN, RT-DETR и RTMDet.

## Структура данных на диске

```
stamps_coco_merged/
  annotations/instances_default.json
  images/default/stamps/<ШИФР_N>.png
```

В этой выгрузке категория одна: `kleimo`, id = 1. Кадров 254, все RGB, ширина 480,
высота от 189 до 984. Боксов 257: у трёх кадров их два
(`КБ-2014-Р2-0597_2`, `КБ-2014-Р3-0049_2`, `Т-12-Р4-040_3`).
Поле `segmentation` пустое, класс читает только `bbox`. В COCO бокс записан как
`x, y, w, h`. В `self.items` он уже `x1, y1, x2, y2`.

Имя файла — `ШИФР_N.png`. Шифр — всё до последнего `_`. Уникальных шифров 252.
У `БП12-Г01-5844` и `Т-12-Р4-1262` по два кадра: это разные стороны одного
предмета, не повтор одного ракурса. В `Нейш` буква «й» на диске хранится как
«и» плюс комбинируемый знак. Датасет приводит шифр к NFC, иначе один предмет
считается двумя.

Кадр из json остаётся в датасете и тогда, когда у него нет ни одного бокса.
Папки `ornaments` и `empty` в json не входят. Их передают в `negative_dirs`:
у этих снимков клейма нет, список боксов пустой.

## Конструктор

```python
StampDataset(
    root_dir,
    transform=None,
    ciphers=None,
    standard_size=None,
    model="default",
    negative_dirs=None,
)
```

| Параметр | Описание |
|---|---|
| `root_dir` | Папка `stamps_coco_merged`. |
| `transform` | `transform(image, target) -> (image, target)`. `image` — uint8 RGB `(H, W, 3)`. `target["boxes"]` — float32 `(N, 4)` xyxy, `target["labels"]` — int64 `(N,)`. Формат ещё общий: `model` применяется после аугментации. |
| `ciphers` | Какие шифры оставить. `None` — все. Так устроены train/val/test. |
| `standard_size` | `(H, W)`. Кадр вписывается в этот прямоугольник, поля белые, боксы сдвигаются вместе с пикселями. `None` — размер файла. |
| `model` | `"default"`, `"faster_rcnn"`, `"rtdetr"` или `"rtmdet"`. Другое значение — `ValueError`. |
| `negative_dirs` | Папки PNG, на которых клейма нет. `None` — только json. Кадр получает пустые `boxes` и `labels`. |

### Что происходит при инициализации

1. Читается `annotations/instances_default.json`.
2. PNG из `images/default/` складываются в словарь по NFC-имени.
3. К каждому кадру приклеиваются его боксы. COCO `xywh` становится `xyxy` и обрезается по границе кадра.
4. Если у кадра в json нет аннотаций, он всё равно попадает в индекс с пустыми `boxes` и `labels`.
5. Если задан `ciphers`, остаются только эти шифры.
6. Из каждой папки `negative_dirs` читаются PNG. Имя уже есть среди кадров json — файл пропускается, второй раз он не добавляется. У остальных боксов нет. Шифр берётся из имени так же, как у размеченных.
7. Словарь кадра попадает в `self.items`. Пиксели на этом шаге не читаются. Для кадра без json читается только размер файла.

Поля элемента `self.items`: `path`, `file_name`, `image_id`, `width`, `height`, `cipher`, `shot`, `boxes`, `labels`. `boxes` здесь всегда xyxy исходного файла, без `transform` и без `standard_size`. Если клейма нет, `boxes` имеет форму `(0, 4)`, а `labels` — форму `(0,)`. У размеченного kleimo `labels` равен `1`. Незнакомый шифр, из-за которого не осталось ни одного кадра, даёт `ValueError`.

## `__len__`

```python
def __len__(self):
    return len(self.items)
```

## `__getitem__`

```python
def __getitem__(self, idx) -> tuple[torch.Tensor, dict]
```

Порядок такой:

1. PNG читается как RGB uint8. Размер сверяется с COCO.
2. Если задан `transform`, он получает картинку и словарь `{boxes, labels}` и возвращает их же. Боксы всё ещё в пикселях этой картинки. Число `boxes` и `labels` должно совпасть.
3. Если задан `standard_size`, кадр вписывается в `(H, W)` с сохранением пропорций. Свободное место заливается 255. Боксы получают тот же масштаб и сдвиг.
4. Картинка становится float32 `(3, H, W)` в диапазоне `[0, 1]`.
5. Словарь приводится к виду, который просит `model`.

До этого шага ключи одни и те же: `boxes`, `labels`, `image_id`, `cipher`, `file_name`, `orig_size`. `orig_size` — `(H, W)` файла, не картинки после `standard_size`.

Высота файлов разная, и боксов на кадре бывает 0, 1 или 2. Стандартный collate DataLoader такие батчи не собирает. `collate_stamps` возвращает список картинок и список словарей.

## Параметр `model`

`split_by_cipher` копирует `model` в каждую часть.

| `model` | Куда | Боксы | Метка kleimo | Классов у головы | Ключи `target` |
|---|---|---|---|---|---|
| `"default"` | YOLO26s | xyxy, пиксели | `1` | перевод делает тренер YOLO | `boxes`, `labels`, `image_id`, `cipher`, `file_name`, `orig_size` |
| `"faster_rcnn"` | Faster R-CNN ResNet-50 FPN v2 | xyxy, пиксели | `1`, ноль занят фоном | 2 | `boxes`, `labels`, `image_id` |
| `"rtdetr"` | RT-DETRv2-S | cxcywh, доли кадра от 0 до 1 | `0` | 1 | `boxes`, `labels`, `image_id`, `orig_size`, `size` |
| `"rtmdet"` | RTMDet-tiny / RTMDet-s | xyxy, пиксели | `0` | 1 | `boxes`, `labels`, `image_id` |

У `"rtdetr"` поле `size` — `(H, W)` картинки, которую видит модель. `orig_size` — размер файла.

```python
detr = StampDataset("stamps_coco_merged", standard_size=(640, 640), model="rtdetr")
image, target = detr[0]
# target["boxes"]  — cx, cy, w, h
# target["labels"] — нули
# target["size"]   — (640, 640)
```

Официальный тренер Ultralytics читает txt, а не этот класс. Строка файла: `0 cx cy w h`, числа от 0 до 1, имя txt совпадает с именем png. Координаты для txt считаются из `dataset.items`: там боксы исходного файла, без аугментации и без `standard_size`. Для своего цикла на `DataLoader` YOLO берёт `model="default"`.

`standard_size` нужен YOLO, RT-DETR и RTMDet, потому что их батч — один тензор. Faster R-CNN принимает кадры разной высоты, ему этот аргумент не обязателен.

## Аугментации

Шаг — функция или класс с `__call__(image, target) -> (image, target)`.
Геометрия картинки и боксов меняется вместе. `labels` при флипе те же.
Цветовой шаг боксы не трогает и возвращает `target` как есть.

Готовые шаги лежат в `datasets.transforms`: `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)`, `Compose([...])`. Свой шаг добавляется в `Compose` рядом с ними. Формула горизонтального отражения: `x1' = width - x2`, `x2' = width - x1`. Вертикального: `y1' = height - y2`, `y2' = height - y1`. `width` и `height` берутся у картинки, которую шаг получил.

## Train / val / test

Резать нужно по шифру. Иначе два кадра одного донца разъедутся по train и test. Это касается и тех двух предметов, у которых в `stamps` по два ракурса.

```python
parts = split_by_cipher(dataset, val_ratio=0.1, test_ratio=0.1, seed=0)
parts["train"].transform = Compose([HorizontalFlip(0.5), VerticalFlip(0.5)])
```

`split_by_cipher` перемешивает шифры, не файлы, и сохраняет `model`, `standard_size`, `transform` и `negative_dirs`. Кадр без клейма уходит в ту же часть, что и остальные снимки его шифра. Внутри части порядок кадров остаётся порядком индекса. Доли округляются вниз, остаток остаётся в train. `test_ratio=0` даёт пустой test.

Обе доли должны быть неотрицательными, их сумма меньше 1. Иначе `ValueError`.

## Пример

```python
from torch.utils.data import DataLoader

from datasets import StampDataset, collate_stamps, split_by_cipher
from datasets.transforms import Compose, HorizontalFlip, VerticalFlip

source = StampDataset(
    "stamps_coco_merged",
    standard_size=(640, 640),
    model="default",
    negative_dirs=["Клеймы /ornaments", "Клеймы /empty"],
)
parts = split_by_cipher(source, val_ratio=0.1, test_ratio=0.1, seed=0)
parts["train"].transform = Compose([HorizontalFlip(0.5), VerticalFlip(0.5)])

loader = DataLoader(
    parts["train"],
    batch_size=8,
    shuffle=True,
    collate_fn=collate_stamps,
)

image, target = parts["train"][0]
print(image.shape, target["boxes"].shape, target["labels"])
```

Запускать из корня репозитория, чтобы пакет `datasets` был этим каталогом.

При `num_workers > 0` воркеры наследуют один и тот же `random` и повторяют одни и те же флипы. В `worker_init_fn` достаточно:

```python
def seed_worker(worker_id):
    random.seed(torch.initial_seed() % 2**32)
```

## Ограничения

- В json нет сегментации, маска из этого класса не собирается.
- Кадр из `negative_dirs` с тем же именем файла, что уже есть в json, не добавляется. В текущих данных так устроены 39 общих файлов `stamps` и `ornaments`: они остаются одним размеченным кадром.
- При `model="default"` и `"faster_rcnn"` метка kleimo равна 1. При `"rtdetr"` и `"rtmdet"` она равна 0.
- `standard_size` не растягивает кадр, а вписывает его и дополняет белым.
- Нормализации ImageNet здесь нет. Деление на 255 — единственное преобразование цвета. Среднее и std задаёт тренер своей модели.
