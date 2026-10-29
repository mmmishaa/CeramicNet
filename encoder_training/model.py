"""DINOv2 с контрактом StampEncoder: RGB 0..1 на входе, ч/б BT.601 и нормировка ImageNet внутри."""

from __future__ import annotations

import timm
import torch
from torch import nn

from src.retrieval.encoder import IMAGENET_MEAN, IMAGENET_STD

MODEL_NAMES = {
    "s": "vit_small_patch14_dinov2.lvd142m",
    "b": "vit_base_patch14_dinov2.lvd142m",
}


class DinoEncoder(nn.Module):
    def __init__(self, size: str = "s", unfreeze_blocks: int = 0, pretrained: bool = True) -> None:
        super().__init__()
        self.size = size
        self.backbone = timm.create_model(MODEL_NAMES[size], pretrained=pretrained, num_classes=0, img_size=224)
        self.register_buffer("gray", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.unfreeze(unfreeze_blocks)

    def unfreeze(self, count: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if count <= 0:
            return
        for block in list(self.backbone.blocks)[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        luminance = (images * self.gray).sum(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        return self.backbone((luminance - self.mean) / self.std)


def load_checkpoint(path, device="cpu") -> DinoEncoder:
    """Чекпоинт train.py; без пути — исходный DINOv2 ViT-S для сравнения."""
    if path is None:
        return DinoEncoder(size="s").to(device)
    payload = torch.load(path, map_location="cpu")
    model = DinoEncoder(size=payload["size"], pretrained=False)
    model.load_state_dict(payload["state"])
    return model.to(device)
