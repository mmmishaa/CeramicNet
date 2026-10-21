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
    standard_size=None,
    model="default",
    negative_dirs=None,
)
```

| Параметр | Описание |
|---|---|
| `root_dir` | Папка `stamps_coco_merged`. |
| `transform` | `transform(image, target) -> (image, target)`. `image` — uint8 RGB `(H, W, 3)`. `target["boxes"]` — float32 `(N, 4)` xyxy, `target["labels"]` — int64 `(N,)`. Формат ещё общий: `model` применяется после аугментации. |
| `standard_size` | `(H, W)`. Кадр вписывается в этот прямоугольник, поля белые, боксы сдвигаются вместе с пикселями. `None` — размер файла. |
| `model` | `"default"`, `"faster_rcnn"`, `"rtdetr"` или `"rtmdet"`. Другое значение — `ValueError`. |
| `negative_dirs` | Папки PNG, на которых клейма нет. `None` — только json. Кадр получает пустые `boxes` и `labels`. |

### Что происходит при инициализации

1. Читается `annotations/instances_default.json`.
2. PNG из `images/default/` складываются в словарь по NFC-имени.
3. К каждому кадру приклеиваются его боксы. COCO `xywh` становится `xyxy` и обрезается по границе кадра.
4. Если у кадра в json нет аннотаций, он всё равно попадает в индекс с пустыми `boxes` и `labels`.
5. Из каждой папки `negative_dirs` читаются PNG. Имя уже есть среди кадров json — файл пропускается, второй раз он не добавляется. У остальных боксов нет. Шифр берётся из имени так же, как у размеченных.
6. Словарь кадра попадает в `self.items`. Пиксели на этом шаге не читаются. Для кадра без json читается только размер файла.

Поля элемента `self.items`: `path`, `file_name`, `image_id`, `width`, `height`, `cipher`, `shot`, `boxes`, `labels`. `boxes` здесь всегда xyxy исходного файла, без `transform` и без `standard_size`. Если клейма нет, `boxes` имеет форму `(0, 4)`, а `labels` — форму `(0,)`. У размеченного kleimo `labels` равен `1`.

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
3. Если задан `standard_size`, кадр вписывается в `(H, W)` с сохранением пропорций. Свободное место заливается 255. Боксы получают тот же масштаб и сдвиг. Масштаб и ширина полей запоминаются в `scale` и `pad`.
4. Картинка становится float32 `(3, H, W)` в диапазоне `[0, 1]`.
5. Словарь приводится к виду, который просит `model`.

До этого шага ключи одни и те же: `boxes`, `labels`, `image_id`, `cipher`, `file_name`, `orig_size`, `scale`, `pad`. `orig_size` — `(H, W)` файла, не картинки после `standard_size`. `scale` — `(по высоте, по ширине)`, во сколько раз кадр уменьшили или увеличили. `pad` — `(сверху, слева)`, ширина белого поля в пикселях. Без `standard_size` `scale = (1.0, 1.0)`, `pad = (0, 0)`.

Высота файлов разная, и боксов на кадре бывает 0, 1 или 2. Стандартный collate DataLoader такие батчи не собирает. `collate_stamps` возвращает список картинок и список словарей.

## Параметр `model`

| `model` | Куда | Боксы | Метка kleimo | Классов у головы | Ключи `target` |
|---|---|---|---|---|---|
| `"default"` | YOLO26s | xyxy, пиксели | `1` | перевод делает тренер YOLO | `boxes`, `labels`, `image_id`, `cipher`, `file_name`, `orig_size`, `scale`, `pad` |
| `"faster_rcnn"` | Faster R-CNN ResNet-50 FPN v2 | xyxy, пиксели | `1`, ноль занят фоном | 2 | `boxes`, `labels`, `image_id`, `orig_size`, `scale`, `pad` |
| `"rtdetr"` | RT-DETRv2-S | cxcywh, доли кадра от 0 до 1 | `0` | 1 | `boxes`, `labels`, `image_id`, `size`, `orig_size`, `scale`, `pad` |
| `"rtmdet"` | RTMDet-tiny / RTMDet-s | xyxy, пиксели | `0` | 1 | `boxes`, `labels`, `image_id`, `orig_size`, `scale`, `pad` |

У `"rtdetr"` поле `size` — `(H, W)` картинки, которую видит модель. `orig_size` — размер файла. Доли в `boxes` считаются от `size`, вместе с белыми полями.

```python
detr = StampDataset("stamps_coco_merged", standard_size=(640, 640), model="rtdetr")
image, target = detr[0]
# target["boxes"]  — cx, cy, w, h
# target["labels"] — нули
# target["size"]   — (640, 640)
```

Официальный тренер Ultralytics читает txt, а не этот класс. Строка файла: `0 cx cy w h`, числа от 0 до 1, имя txt совпадает с именем png. Координаты для txt считаются из `dataset.items`: там боксы исходного файла, без аугментации и без `standard_size`. Для своего цикла на `DataLoader` YOLO берёт `model="default"`.

`standard_size` нужен YOLO, RT-DETR и RTMDet, потому что их батч — один тензор. Faster R-CNN принимает кадры разной высоты, ему этот аргумент не обязателен.

## Как вернуть предсказание на исходный снимок

С `standard_size` модель видит кадр с белыми полями, и её боксы записаны в координатах этого кадра, а не файла. Обратный перевод: `x = (x' - pad_left) / scale_w`, `y = (y' - pad_top) / scale_h`. Его делает `boxes_to_original(boxes, target)`: принимает xyxy в пикселях входа модели и возвращает xyxy в пикселях файла, обрезанные по `orig_size`.

Пример. Файл 480×984, `standard_size=(640, 640)`. Кадр уменьшается до 312×640 и встаёт по центру, поля слева и справа по 164. В `target` лежит `scale ≈ (0.6504, 0.65)`, `pad = (0, 164)`. Модель выдала `x1 = 190`, `x2 = 242`. На файле это `(190 - 164) / 0.65 = 40` и `(242 - 164) / 0.65 = 120`.

```python
from dataset import boxes_to_original

image, target = dataset[i]
pred = model([image])[0]                          # Faster R-CNN: xyxy в пикселях image
boxes = boxes_to_original(pred["boxes"], target)  # xyxy в пикселях файла
```

У `"rtdetr"` модель выдаёт доли `cx, cy, w, h`. Сначала их переводят в xyxy-пиксели картинки `size`, потом вызывают `boxes_to_original`:

```python
from torchvision.ops import box_convert

h, w = target["size"]
# pred_boxes — выход модели для этого кадра: доли cx, cy, w, h, форма (N, 4)
xyxy = box_convert(pred_boxes, "cxcywh", "xyxy") * torch.tensor([w, h, w, h])
boxes = boxes_to_original(xyxy, target)
```

Умножать доли сразу на `orig_size` нельзя: поля не учитываются, и бокс съезжает. В примерах Hugging Face так и делают через `post_process_object_detection(..., target_sizes=...)`. Если пользоваться ей, в `target_sizes` передают `target["size"]`, а потом вызывают `boxes_to_original`.

## Аугментации

Шаг — функция или класс с `__call__(image, target) -> (image, target)`.
Геометрия картинки и боксов меняется вместе. `labels` при флипе те же.
Цветовой шаг боксы не трогает и возвращает `target` как есть.

Готовые шаги лежат в `dataset.transforms`: `HorizontalFlip(p=0.5)`, `VerticalFlip(p=0.5)`, `LowContrast(p=0.5, factor=(0.5, 0.8))`, `HighContrast(p=0.5, factor=(1.2, 1.5))`, `Defocus(p=0.5, radius=(1.0, 3.0))`, `Compose([...])`. Свой шаг добавляется в `Compose` рядом с ними. Формула горизонтального отражения: `x1' = width - x2`, `x2' = width - x1`. Вертикального: `y1' = height - y2`, `y2' = height - y1`. `width` и `height` берутся у картинки, которую шаг получил.

`LowContrast` и `HighContrast` — цветовые шаги, боксы они не трогают. Формула: `pixel' = mean + factor * (pixel - mean)`, где `mean` — средняя яркость кадра в оттенках серого, результат обрезается до `0..255`. `factor` выбирается случайно из диапазона при каждом вызове: у `LowContrast` он от 0 до 1, у `HighContrast` не меньше 1. При `factor = 1` кадр не меняется.

`Defocus` имитирует снимок не в фокусе. Каждый пиксель усредняется с соседями внутри круга радиуса `radius`. Боксы шаг не трогает. `radius` задаёт степень расфокуса в пикселях исходного кадра, до `standard_size`. Это либо число, либо диапазон `(low, high)`, из которого радиус берётся случайно при каждом вызове. `0` — без размытия, дробный радиус работает плавно. По умолчанию `(1.0, 3.0)`: на кадре шириной 480 клеймо ещё видно, а мелкие детали оттиска уже плывут. Радиус 5 и больше размывает уже сильно.

```python
transform = Compose([
    HorizontalFlip(0.5),
    VerticalFlip(0.5),
    LowContrast(0.3),
    HighContrast(0.3),
    Defocus(0.3),               # радиус 1–3
    # Defocus(0.3, radius=2),   # всегда радиус 2
])
```

## Пример

```python
from torch.utils.data import DataLoader

from dataset import StampDataset, collate_stamps
from dataset.transforms import Compose, HorizontalFlip, VerticalFlip

dataset = StampDataset(
    "stamps_coco_merged",
    transform=Compose([HorizontalFlip(0.5), VerticalFlip(0.5)]),
    standard_size=(640, 640),
    model="default",
    negative_dirs=["Клеймы /ornaments", "Клеймы /empty"],
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_stamps,
)

image, target = dataset[0]
print(image.shape, target["boxes"].shape, target["labels"])
```

Запускать из корня репозитория, чтобы импортировался пакет `dataset`.

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
- `scale` и `pad` описывают только вписывание в `standard_size`. Если `transform` когда-нибудь начнёт менять размер кадра (кроп, масштаб), `boxes_to_original` вернёт боксы на кадр после `transform`, а не на файл. Флипы размер не меняют.
- Нормализации ImageNet здесь нет. Деление на 255 — единственное преобразование цвета. Среднее и std задаёт тренер своей модели.
