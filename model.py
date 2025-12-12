"""
Transfer Learning Model for Drowsy Driving Detection

- Backbone: ResNet (18 / 34 / 50)
- Pretrained on ImageNet
- Binary classification (Normal / Drowsy)
"""

import torch
import torch.nn as nn
from torchvision import models


class DrowsyDetectionModel(nn.Module):
    """
    Transfer Learning Strategy:
    1. ImageNet pretrained backbone 사용
    2. Backbone 동결 여부 선택 가능
    3. Fully Connected layer를 2-class classifier로 교체
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone_name = backbone

       
        # Backbone selection
       
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = 512

        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = 512

        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = 2048

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

 
    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cuda",
) -> nn.Module:
    """
    Model factory function
    """
    model = DrowsyDetectionModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    model = model.to(device)

    print("=" * 50)
    print(f"Backbone        : {backbone}")
    print(f"Pretrained      : {pretrained}")
    print(f"Freeze backbone : {freeze_backbone}")
    print(f"Total params    : {model.count_total_params():,}")
    print(f"Trainable params: {model.count_trainable_params():,}")
    print("=" * 50)

    return model


if __name__ == "__main__":
    device = "cpu"
    model = create_model(
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=False,
        device=device,
    )

    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model forward test passed ✔")

