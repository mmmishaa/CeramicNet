"""DINOv2: ч/б до энкодера, учим только последние блоки и norm."""

from __future__ import annotations

import torch
from torch import nn

from finetune.data_utils import GrayscaleBT601

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"


class DinoEncoder(nn.Module):
    """ViT-S/14. На входе RGB 0..1, внутри ч/б BT.601 и нормировка ImageNet."""

    def __init__(self, unfreeze_blocks: int = 0, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            img_size=224,
        )
        self.gray = GrayscaleBT601()
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.unfreeze_last_blocks(unfreeze_blocks)

    def unfreeze_last_blocks(self, count: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if count <= 0:
            return
        blocks = list(self.backbone.blocks)
        if count > len(blocks):
            raise ValueError(f"В DINOv2 {len(blocks)} блоков, запрошено {count}")
        for block in blocks[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = self.gray(images)
        normalized = (gray - self.mean) / self.std
        return self.backbone(normalized)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


class Student(nn.Module):
    """Энкодер без головы: на поиске нужен только вектор 384."""

    def __init__(self, unfreeze_blocks: int = 2) -> None:
        super().__init__()
        self.encoder = DinoEncoder(unfreeze_blocks=unfreeze_blocks, pretrained=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)
