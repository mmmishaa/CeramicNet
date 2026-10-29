"""Оценка энкодера на размеченных реальных фото (попадание фабрики в top-k).

Запуск из корня репозитория:
    python -m encoder_training.evaluate                                  # исходный DINOv2
    python -m encoder_training.evaluate --checkpoint runs/encoder/s_u6_e20/last.pt
"""

import argparse
import json

import torch

from encoder_training.data import GalleryTensors, RealTensors, chance_level, eval_real, load_marks, load_real
from encoder_training.model import load_checkpoint
from src.retrieval.books import project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="last.pt из train.py. Без него — исходный DINOv2 ViT-S")
    parser.add_argument("--books", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--real-coco", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery = GalleryTensors(load_marks(args.books, args.meta))
    real = RealTensors(load_real(args.real_coco))
    if not real.crops:
        raise SystemExit("Нет размеченных реальных кропов: укажите --real-coco")
    print(f"Эталонов: {len(gallery.marks)}, размеченных реальных кропов: {len(real.crops)}")
    print("Случайная выдача:", {f"hit@{k}": round(chance_level(gallery, real, k), 3) for k in (1, 3, 5, 10)})
    model = load_checkpoint(project_path(args.checkpoint) if args.checkpoint else None, device)
    print(json.dumps({k: round(v, 3) for k, v in eval_real(model, gallery, real, device).items()}, indent=1))
