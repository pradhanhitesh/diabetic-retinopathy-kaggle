import torch
import torch.nn as nn
import torchvision.models as models


class OCTViT_B16(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3, pretrained: bool = True):
        """
        Custom Vision Transformer (ViT-B/16) for RGB OCT classification.

        Args:
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate for dense layers.
            pretrained (bool): Use pretrained ImageNet weights.
        """
        super(OCTViT_B16, self).__init__()

        # Load pretrained ViT-B/16
        self.backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Extract feature dimension
        in_features = self.backbone.heads.head.in_features

        # Replace the classification head with a custom dense layer
        self.backbone.heads = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
