"""Метрики эпохи: свой кроп находится, мягкие соседи не вымываются, фабрика не схлопывается."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from finetune.data_utils import BookSample, RealSample, load_real_pil, real_transform, square_tensor
from finetune.model import Student


@torch.no_grad()
def evaluate_epoch(
    student: Student,
    baseline_embeddings: torch.Tensor,
    books: list[BookSample],
    reals: list[RealSample],
    device: torch.device,
    logger=None,
    iteration: int = 0,
) -> dict[str, float]:
    student.eval()
    book_clean = _embed_books(student, books, device)
    book_aug = _embed_book_augs(student, books, device)
    book_recall = _recall_at_1(book_aug, book_clean)

    baseline = F.normalize(baseline_embeddings.to(device), dim=-1)
    preservation = _soft_neighbor_preservation(baseline, book_clean, neighbors=5, kept=10)
    diversity = _intra_factory_diversity(book_clean, books)

    real_clean = _embed_reals(student, reals, device, augment=False)
    real_aug = _embed_reals(student, reals, device, augment=True)
    real_recall = _recall_at_1(real_aug, real_clean) if len(reals) > 1 else 0.0

    metrics = {
        "book_recall_at_1": book_recall,
        "soft_neighbor_preservation": preservation,
        "real_recall_at_1": real_recall,
        "intra_factory_diversity": diversity,
    }
    if logger is not None:
        for name, value in metrics.items():
            logger.report_scalar("validation", name, value, iteration)
    return metrics


def _embed_books(student: Student, books: list[BookSample], device: torch.device) -> torch.Tensor:
    parts = []
    batch: list[torch.Tensor] = []
    for sample in books:
        batch.append(square_tensor(sample.path))
        if len(batch) == 64:
            parts.append(_forward(student, torch.stack(batch).to(device)))
            batch = []
    if batch:
        parts.append(_forward(student, torch.stack(batch).to(device)))
    return torch.cat(parts, dim=0)


def _embed_book_augs(student: Student, books: list[BookSample], device: torch.device) -> torch.Tensor:
    from finetune.data_utils import book_transform, load_book_pil

    transform = book_transform()
    parts = []
    batch: list[torch.Tensor] = []
    for sample in books:
        batch.append(transform(load_book_pil(sample.path)))
        if len(batch) == 64:
            parts.append(_forward(student, torch.stack(batch).to(device)))
            batch = []
    if batch:
        parts.append(_forward(student, torch.stack(batch).to(device)))
    return torch.cat(parts, dim=0)


def _embed_reals(
    student: Student,
    reals: list[RealSample],
    device: torch.device,
    augment: bool,
) -> torch.Tensor:
    transform = real_transform() if augment else None
    parts = []
    batch: list[torch.Tensor] = []
    for sample in reals:
        image = load_real_pil(sample)
        tensor = transform(image) if transform is not None else _pil_tensor(image)
        batch.append(tensor)
        if len(batch) == 64:
            parts.append(_forward(student, torch.stack(batch).to(device)))
            batch = []
    if batch:
        parts.append(_forward(student, torch.stack(batch).to(device)))
    if not parts:
        return torch.empty(0, 384, device=device)
    return torch.cat(parts, dim=0)


def _pil_tensor(image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _forward(student: Student, images: torch.Tensor) -> torch.Tensor:
    embeddings = student.embed(images)
    return F.normalize(embeddings, dim=-1)


def _recall_at_1(queries: torch.Tensor, gallery: torch.Tensor) -> float:
    if len(queries) == 0:
        return 0.0
    neighbors = torch.argmax(queries @ gallery.T, dim=1)
    expected = torch.arange(queries.shape[0], device=queries.device)
    return float((neighbors == expected).float().mean())


def _soft_neighbor_preservation(
    baseline: torch.Tensor,
    finetuned: torch.Tensor,
    neighbors: int,
    kept: int,
) -> float:
    """Доля топ-5 соседей baseline, которые дообученная модель всё ещё держит в топ-10."""
    count = baseline.shape[0]
    take_neighbors = min(neighbors, count - 1)
    take_kept = min(kept, count - 1)
    baseline_sim = baseline @ baseline.T
    finetuned_sim = finetuned @ finetuned.T
    baseline_sim.fill_diagonal_(-1e9)
    finetuned_sim.fill_diagonal_(-1e9)
    soft = torch.topk(baseline_sim, k=take_neighbors, dim=1).indices
    current = torch.topk(finetuned_sim, k=take_kept, dim=1).indices
    preserved = []
    for row in range(count):
        current_ids = set(current[row].tolist())
        hits = sum(int(index) in current_ids for index in soft[row].tolist())
        preserved.append(hits / take_neighbors)
    return float(np.mean(preserved))


def _intra_factory_diversity(embeddings: torch.Tensor, books: list[BookSample]) -> float:
    """Среднее (1 - косинус) между разными клеймами одной фабрики. Ноль значило бы, что они слиплись."""
    groups: dict[int, list[int]] = {}
    for index, sample in enumerate(books):
        if sample.factory_id < 0:
            continue
        groups.setdefault(sample.factory_id, []).append(index)

    distances: list[float] = []
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        vectors = embeddings[indexes]
        similarity = vectors @ vectors.T
        # Нулевое сходство тоже пара, поэтому маска, а не отбор ненулевых значений.
        mask = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
        pair_sim = similarity[mask]
        distances.extend((1.0 - pair_sim).tolist())
    if not distances:
        return 0.0
    return float(np.mean(distances))
