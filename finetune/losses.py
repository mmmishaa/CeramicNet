"""KD по мягким фабрикам и Barlow Twins по двум видам одного кропа."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def kd_loss(student_logits: torch.Tensor, soft_targets: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL между студентом и уже посчитанными soft targets учителя.

    soft_targets = softmax(teacher / T). Деление логитов студента на T и множитель T^2
    сохраняют масштаб градиента, как в дистилляции Хинтона.
    """
    log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(log_prob, soft_targets, reduction="batchmean") * (temperature ** 2)


class BarlowTwinsLoss(nn.Module):
    """Кросс-корреляция двух видов: диагональ к 1, остальные элементы к 0.

    Так один клейм становится устойчивым к повороту и обрезке, но компоненты вектора
    не копируют друг друга. Похожие чужие клейма здесь не объявляются жёсткими негативами,
    в отличие от InfoNCE: штраф только за лишнюю корреляцию признаков внутри пары.
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
