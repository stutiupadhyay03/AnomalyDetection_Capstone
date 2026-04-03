"""
Model architecture definitions.

These classes must match the architecture used at training time so that
saved state-dicts can be loaded without key mismatches.
"""

import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    """
    Crowd Scene Recognition Network for density-map estimation.

    Frontend: VGG-like encoder (no final MaxPool from VGG).
    Backend:  Dilated convolution stack → single-channel density map.
    """

    def __init__(self) -> None:
        super().__init__()
        self.frontend = self.make_layers(
            [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,  64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,    1, kernel_size=1),
        )

    def forward(self, x):
        x = self.frontend(x)
        return self.backend(x)

    # Keep original name so legacy checkpoints load without remapping.
    def make_layers(
        self,
        cfg,
        in_channels: int = 3,
        dilation: bool = False,
    ) -> nn.Sequential:
        layers = []
        d_rate = 2 if dilation else 1
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(
                    in_channels, v,
                    kernel_size=3, padding=d_rate, dilation=d_rate,
                )
                layers += [conv, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class AnomalyClassifier(nn.Module):
    """
    ResNet-34 backbone with a custom dropout + linear head for
    3-class anomaly classification (normal / unusual action / abnormal object).
    """

    NUM_CLASSES = 3

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        # Attribute name must match training checkpoint ("base.*").
        self.base = models.resnet34(weights="IMAGENET1K_V1")
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base.fc.in_features, num_classes),
        )

    def forward(self, x):
        return self.base(x)
