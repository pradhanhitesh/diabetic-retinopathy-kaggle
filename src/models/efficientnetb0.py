import torch
import torch.nn as nn
import torchvision.models as models


class OCTEfficientNetB0(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3, pretrained: bool = True):
        """
        Custom EfficientNet-B0 model for RGB OCT image classification.

        Args:
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate for dense layers.
            pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        super(OCTEfficientNetB0, self).__init__()

        # Load pretrained EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Number of features from the last layer
        in_features = self.backbone.classifier[1].in_features

        # Replace the classifier with a custom dense head
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
