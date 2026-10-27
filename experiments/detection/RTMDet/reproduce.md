# Воспроизведение RTMDet-s

Исходный опыт обучался кодом `61199ac4ebc09baa9ce15f6cd9e7e6de2c1947ac` из ветки `pipeline`. Пакеты обучения ещё не входят в `main`; для запуска нужен отдельный checkout:

```bash
git clone https://github.com/mmmishaa/CeramicNet.git CeramicNet-rtmdet-source
cd CeramicNet-rtmdet-source
git checkout --detach 61199ac4ebc09baa9ce15f6cd9e7e6de2c1947ac
python3.12 train_pipelines/detection_models/RTMDet-s/setup_local.py
```

Далее выбрать ядро **CeramicNet RTMDet (CPU)**, подготовить фотографии из сохранённого split `detection_pipeline/splits/v3_seed42_reserve80_active15` и открыть [train.ipynb](train.ipynb) из этого отчёта. Перед первой ячейкой кода задать в ядре:

```python
import os
os.environ["CERAMIC_SOURCE_ROOT"] = "/absolute/path/to/CeramicNet-rtmdet-source"
os.environ["CERAMIC_IMAGES"] = "/absolute/path/to/Клеймы"
```

На CUDA использовать установку из README закреплённого checkout вместо Mac-скрипта. Фактические версии и параметры исходного CPU-опыта записаны в [training_config.json](training_config.json); значения по умолчанию — 50 эпох, batch 2 на CPU, 640 px. Ноутбук проверяет SHA256 фото и split до обучения. Каждый полный запуск создаёт новый опыт; существующий результат не перезаписывать.

Лучший нативный checkpoint и проверенный ONNX лежат в [ClearML](https://app.clear.ml/projects/667be44d3e6345ff8c818f4a5261b3f7/tasks/7e70ead7461e4d93888f7d17e169b5ec/output/log); ID моделей — в [provenance.json](provenance.json). Для скачивания нативных весов после настройки SDK:

```python
from clearml import InputModel
checkpoint = InputModel(model_id="bf0fe86f32d44493837ce11e14034b21").get_local_copy()
```

`training_history.csv` объединяет все сегменты обучения с учётом откатов при продолжении. `precision_recall.png` строится по 99 порогам confidence при IoU 0.5 и не является усреднённой COCO PR-кривой. Картинка примеров взята из сохранённого опыта, а не из нового запуска. В [metrics.json](metrics.json) сохранены оба рабочих порога: 0.5 и 0.84.
