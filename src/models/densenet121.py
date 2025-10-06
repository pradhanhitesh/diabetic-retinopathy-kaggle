import torch
import torch.nn as nn
import torchvision.models as models


class OCTDenseNet121(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3, pretrained: bool = True):
        """
        Custom DenseNet121 model for RGB OCT image classification.

        Args:
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate for dense layers.
            pretrained (bool): Whether to use pretrained ImageNet weights.
        """
        super(OCTDenseNet121, self).__init__()

        # Load pretrained DenseNet121 backbone
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Number of features from the final pooling layer
        in_features = self.backbone.classifier.in_features

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
