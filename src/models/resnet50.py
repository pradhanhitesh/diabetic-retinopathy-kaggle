import torch
import torch.nn as nn
import torchvision.models as models

class OCTResNet50_V0(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3, pretrained: bool = True):
        """
        Custom ResNet50 model for RGB OCT image classification.

        Args:
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate for dense layers.
            pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        super(OCTResNet50_V0, self).__init__()

        # Load pretrained ResNet50 backbone
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Extract number of features from last FC layer
        in_features = self.backbone.fc.in_features

        # Replace the final FC layer with a custom dense head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
