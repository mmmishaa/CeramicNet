"""Слабый учитель: MLP на замороженных эмбеддингах DINOv2, метка — factory_norm."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from finetune.model import FactoryHead


class WeakTeacher(nn.Module):
    def __init__(self, num_factories: int) -> None:
        super().__init__()
        self.head = FactoryHead(num_factories)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head(embeddings)

    @torch.no_grad()
    def soft_targets(self, embeddings: torch.Tensor, temperature: float = 3.0) -> torch.Tensor:
        """Мягкое распределение по фабрикам. temperature=3 сглаживает редкие заводы."""
        logits = self.forward(embeddings)
        return F.softmax(logits / temperature, dim=-1)


def train_weak_teacher(
    teacher: WeakTeacher,
    embeddings: torch.Tensor,
    factory_ids: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """CrossEntropy только по клеймам, у которых factory_norm не пустой. Потом учитель замораживается."""
    device = device or embeddings.device
    labeled = factory_ids >= 0
    if int(labeled.sum()) < 2:
        raise ValueError("Слишком мало клейм с factory_norm для учителя")

    dataset = TensorDataset(embeddings[labeled].detach().cpu(), factory_ids[labeled].detach().cpu())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    teacher.to(device)
    teacher.train()
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []

    for _ in range(epochs):
        total = 0.0
        seen = 0
        for batch_embeddings, batch_labels in loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.to(device)
            logits = teacher(batch_embeddings)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_embeddings.shape[0]
            seen += batch_embeddings.shape[0]
        losses.append(total / max(seen, 1))

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return losses
