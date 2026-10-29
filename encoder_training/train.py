"""Дообучение DINOv2: фото-подобный вид эталона должен находить свой чистый эталон среди остальных.

Запуск из корня репозитория:
    python -m encoder_training.train --name s_u6_e20

Пара — «как на фото ↔ как в книге», чужие эталоны батча работают негативами (InfoNCE).
Поворот на любой угол — внутри аугментации, поэтому на запросе повороты не нужны.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from encoder_training.data import GalleryTensors, RealTensors, embed, eval_real, load_marks, load_real, prepare, to_tensor
from encoder_training.model import DinoEncoder
from encoder_training.photo_aug import light_aug, photo_like
from src.paths import ROOT_DIR
from src.preprocessing.image import read_bgr

RUNS_DIR = ROOT_DIR / "runs" / "encoder"


def parse_args():
    parser = argparse.ArgumentParser(description="Дообучение энкодера клейм: фото-подобный эталон ↔ чистый эталон")
    parser.add_argument("--name", required=True, help="Папка запуска в runs/encoder/")
    parser.add_argument("--size", default="s", choices=["s", "b"])
    parser.add_argument("--unfreeze", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--holdout", type=float, default=0.1, help="Доля эталонов вне обучения для синтетической проверки. 0 — учить на всех.")
    parser.add_argument("--books", default=None, help="Папка эталонов. По умолчанию из retrieval_config.yaml")
    parser.add_argument("--meta", default=None, help="meta.csv эталонов. По умолчанию из retrieval_config.yaml")
    parser.add_argument("--real-coco", default=None, help="COCO реальных фото для оценки. По умолчанию data/stamps_coco_merged")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class PairDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rng = np.random.default_rng(torch.randint(0, 2**31, (1,)).item())
        image = read_bgr(self.paths[index])
        photo = to_tensor(prepare(photo_like(image, rng)))
        clean = to_tensor(prepare(light_aug(image, rng)))
        return photo, clean


def synthetic_queries(paths: list[Path], seed: int) -> torch.Tensor:
    """Фиксированные фото-подобные виды отложенных эталонов: одинаковые для всех эпох и запусков."""
    rng = np.random.default_rng(seed)
    if not paths:
        return torch.empty(0, 3, 224, 224, dtype=torch.uint8)
    return torch.stack([to_tensor(prepare(photo_like(read_bgr(path), rng))) for path in paths])


def evaluate(model, gallery, real, holdout_idx, holdout_queries, device):
    vectors = embed(model, gallery.tensors, device)
    metrics = eval_real(model, gallery, real, device, gallery_vectors=vectors)
    if len(holdout_idx):
        scores = embed(model, holdout_queries, device) @ vectors.T
        ranks = (scores > scores[torch.arange(len(holdout_idx)), holdout_idx][:, None]).sum(1)
        for k in (1, 5, 20):
            metrics[f"synth_R@{k}"] = float((ranks < k).float().mean())
    return metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out_dir = RUNS_DIR / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=1))

    marks = load_marks(args.books, args.meta)
    order = np.random.default_rng(12345).permutation(len(marks))
    n_hold = int(len(marks) * args.holdout)
    holdout_idx = torch.tensor(np.sort(order[:n_hold]), dtype=torch.long)
    train_paths = [marks[i].path for i in np.sort(order[n_hold:])]
    print(f"Эталонов: {len(marks)}, в обучении {len(train_paths)}, отложено {n_hold}")

    gallery = GalleryTensors(marks)
    real = RealTensors(load_real(args.real_coco))
    if not real.crops:
        print("Реальных размеченных кропов нет: оценка только синтетическая")
    holdout_queries = synthetic_queries([marks[i].path for i in holdout_idx.tolist()], seed=777)

    model = DinoEncoder(size=args.size, unfreeze_blocks=args.unfreeze).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    loader = DataLoader(
        PairDataset(train_paths), batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True,
    )
    total_steps = args.epochs * len(loader)
    warmup = len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )

    history = [{"epoch": 0, **evaluate(model, gallery, real, holdout_idx, holdout_queries, device)}]
    print(json.dumps(history[-1]))
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, losses = time.time(), []
        for photo, clean in loader:
            photo = photo.to(device, non_blocking=True).float().div(255.0)
            clean = clean.to(device, non_blocking=True).float().div(255.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                vectors = model(torch.cat([photo, clean]))
            q, g = F.normalize(vectors.float(), dim=-1).chunk(2)
            logits = q @ g.T / args.temperature
            target = torch.arange(len(q), device=device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "sec": round(time.time() - started),
               **evaluate(model, gallery, real, holdout_idx, holdout_queries, device)}
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()
                          if k in ("epoch", "loss", "sec") or k.startswith("synth") or "@1" in k or "@3" in k}))
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    torch.save({"state": model.state_dict(), "size": args.size}, out_dir / "last.pt")
    print(f"Веса: {out_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
