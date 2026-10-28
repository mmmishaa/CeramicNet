"""Barlow Twins на двух видах одного кропа и ArcFace по метке клейма."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class BarlowTwinsLoss(nn.Module):
    """Кросс-корреляция двух видов: диагональ к 1, остальные элементы к 0.

    Похожие чужие клейма здесь не расталкиваются. Штраф только за то,
    что компоненты вектора копируют друг друга внутри пары.
    """

    def __init__(self, lambda_off: float = 5e-3) -> None:
        super().__init__()
        self.lambda_off = lambda_off

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError(f"Ожидались две матрицы B×D, получено {tuple(first.shape)} и {tuple(second.shape)}")
        batch = first.shape[0]
        if batch < 2:
            raise ValueError("Barlow Twins требует батч из хотя бы двух кропов")

        first = _batch_standardize(first)
        second = _batch_standardize(second)
        correlation = first.T @ second / batch

        diagonal = torch.diagonal(correlation)
        on_diag = (diagonal - 1.0).pow(2).sum()
        off_diag = _off_diagonal(correlation).pow(2).sum()
        return on_diag + self.lambda_off * off_diag


def _batch_standardize(embeddings: torch.Tensor) -> torch.Tensor:
    centered = embeddings - embeddings.mean(dim=0)
    # unbiased=False: диагональ корреляции одинаковых видов тогда равна 1, а не N/(N-1).
    deviation = embeddings.std(dim=0, unbiased=False).clamp_min(1e-6)
    return centered / deviation


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


class ArcFaceHead(nn.Module):
    """Центр на каждое клеймо и угловой отступ между своим центром и чужими.

    Кроп тянется к центру своего mark_id и отталкивается от остальных центров.
    После обучения голова не нужна: в индекс идёт вектор энкодера.
    """

    def __init__(self, dim: int, classes: int, margin: float = 0.3, scale: float = 30.0) -> None:
        super().__init__()
        if not 0.0 <= margin < math.pi / 2:
            raise ValueError(f"margin должен быть в [0, π/2), получено {margin}")
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(classes, dim))
        nn.init.xavier_uniform_(self.weight)

    @torch.no_grad()
    def set_centers(self, centers: torch.Tensor) -> None:
        """Начальные центры из текущих эмбеддингов книг."""
        if centers.shape != self.weight.shape:
            raise ValueError(f"Ожидались центры {tuple(self.weight.shape)}, получено {tuple(centers.shape)}")
        self.weight.copy_(F.normalize(centers.to(self.weight.dtype), dim=-1))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings, dim=-1), F.normalize(self.weight, dim=-1))
        cosine = cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        with_margin = torch.cos(torch.acos(cosine) + self.margin)
        target = F.one_hot(labels, cosine.shape[1]).bool()
        logits = torch.where(target, with_margin, cosine) * self.scale
        return F.cross_entropy(logits, labels)
