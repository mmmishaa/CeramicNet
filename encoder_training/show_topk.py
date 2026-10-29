"""Наглядная выдача: реальный кроп -> top-5 эталонов. Зелёная рамка — фабрика совпала с разметкой, красная — нет.

Сохраняет два листа с одними и теми же запросами: исходный DINOv2 и дообученный.

Запуск из корня репозитория:
    python -m encoder_training.show_topk runs/encoder/s_u6_e20/last.pt --out-dir reports/topk
"""

import argparse
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from encoder_training.data import FACTORY_RULES, GalleryTensors, RealTensors, embed, load_marks, load_real
from encoder_training.model import load_checkpoint
from src.retrieval.books import project_path

GOOD, BAD = "#1a9850", "#d73027"


def rgb(tensor):
    return tensor.permute(1, 2, 0).numpy()[:, :, ::-1]


def frame(axis, color):
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(4 if color != "none" else 0)


def draw(path, title, pick, real, gallery, top, scores):
    factories = [mark.factory for mark in gallery.marks]
    rows = len(pick)
    fig, axes = plt.subplots(rows, 6, figsize=(13, 2.45 * rows))
    fig.suptitle(title, fontsize=14, y=0.999)
    for row, index in enumerate(pick):
        crop = real.crops[index]
        rule = FACTORY_RULES[crop.label]
        axis = axes[row, 0]
        axis.imshow(rgb(real.tensors[index]))
        axis.set_title(f"запрос #{crop.ann_id}\nразметка: {crop.label}", fontsize=8, fontweight="bold")
        frame(axis, "#333333")
        for col in range(5):
            j = top[row, col].item()
            ok = rule(factories[j].lower())
            axis = axes[row, col + 1]
            axis.imshow(rgb(gallery.tensors[j]))
            name = textwrap.shorten(factories[j] or "фабрика не указана", 34, placeholder="…")
            axis.set_title(f"{col + 1}. {gallery.marks[j].mark_id}  cos {scores[row, col]:.2f}\n{name}",
                           fontsize=7, color=GOOD if ok else BAD)
            frame(axis, GOOD if ok else BAD)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--books", default=None)
    parser.add_argument("--meta", default=None)
    parser.add_argument("--real-coco", default=None)
    args = parser.parse_args()
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    gallery = GalleryTensors(load_marks(args.books, args.meta))
    real = RealTensors(load_real(args.real_coco))
    if not real.crops:
        raise SystemExit("Нет размеченных реальных кропов: укажите --real-coco")
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(real.crops), size=min(args.count, len(real.crops)), replace=False))
    for checkpoint, name, title in (
        (None, "topk_baseline.png", "Исходный DINOv2: top-5 эталонов для реальных фото"),
        (project_path(args.checkpoint), "topk_finetuned.png", "Дообученный энкодер: top-5 эталонов для реальных фото"),
    ):
        model = load_checkpoint(checkpoint, device)
        vectors = embed(model, gallery.tensors, device)
        scores, top = (embed(model, real.tensors[pick], device) @ vectors.T).topk(5, dim=1)
        draw(out_dir / name, title, pick, real, gallery, top, scores)
        print(out_dir / name)
        del model
