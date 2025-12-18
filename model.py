
import torch
import torch.nn as nn
from torchvision import models


class DrowsyDetectionModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

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
            raise ValueError(f"Unsupported backbone: {backbone}. "
                           f"Choose from: resnet18, resnet34, resnet50")

        # Backbone 동결 (선택적)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone {backbone} frozen")

        # Custom Classifier Head
        # 논문 Section 3.2
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
        """학습 가능한 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        """전체 파라미터 수 반환"""
        return sum(p.numel() for p in self.parameters())
    
    def get_feature_extractor(self):
        # fc 레이어 전까지의 모듈 반환
        return nn.Sequential(*list(self.backbone.children())[:-1])


def create_model(
    backbone: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    device: str = "cuda",
) -> nn.Module:
    
    model = DrowsyDetectionModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )

    model = model.to(device)

    print("=" * 50)
    print(f"Model: {backbone}")
    print(f"Pretrained: {pretrained}")
    print(f"Freeze backbone: {freeze_backbone}")
    print(f"Total params: {model.count_total_params():,}")
    print(f"Trainable params: {model.count_trainable_params():,}")
    print("=" * 50)

    return model


def load_checkpoint(checkpoint_path, backbone="resnet18", device="cuda"):
    model = create_model(
        backbone=backbone,
        pretrained=False,  # weights will be loaded from checkpoint
        device=device,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    info = {
        'best_val_acc': checkpoint.get('best_val_acc', None),
        'config': checkpoint.get('config', None)
    }
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    if info['best_val_acc']:
        print(f"Best validation accuracy: {info['best_val_acc']*100:.2f}%")
    
    return model, info


# 테스트
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Testing model creation...")
    
    # ResNet18 테스트
    model = create_model(
        backbone="resnet18",
        pretrained=True,
        freeze_backbone=False,
        device=device,
    )

    # Forward 테스트
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Softmax 확률
    probs = torch.softmax(output, dim=1)
    print(f"Output probabilities (sample): {probs[0].detach().cpu().numpy()}")
    
    print("\n✓ Model test passed!")