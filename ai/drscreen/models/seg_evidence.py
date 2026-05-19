from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class LesionSegEvidence(nn.Module):
    """Standalone lesion evidence segmenter.

    This model is deliberately classifier-independent. It predicts MA/HE/EX/SE
    probability maps and can be used as a separate evidence path while keeping
    the deployed v31 classifier unchanged.
    """

    def __init__(
        self,
        encoder: str = "resnet50",
        out_channels: int = 4,
        pretrained: bool = True,
        decoder_channels: tuple[int, int, int, int] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        if encoder != "resnet50":
            raise ValueError("LesionSegEvidence currently supports encoder='resnet50'")

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.encoder_name = encoder
        self.out_channels = int(out_channels)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        d4, d3, d2, d1 = decoder_channels
        self.center = _ConvBlock(2048, d4)
        self.up3 = _UpBlock(d4, 1024, d3)
        self.up2 = _UpBlock(d3, 512, d2)
        self.up1 = _UpBlock(d2, 256, d1)
        self.up0 = _UpBlock(d1, 64, d1)
        self.head = nn.Conv2d(d1, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x0 = self.stem(x)       # 1/2
        x1 = self.layer1(self.pool(x0))  # 1/4
        x2 = self.layer2(x1)    # 1/8
        x3 = self.layer3(x2)    # 1/16
        x4 = self.layer4(x3)    # 1/32

        y = self.center(x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        logits = self.head(y)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def predict_seg(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def predict_seg_union(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self.predict_seg(x)
        if probabilities.shape[1] == 1:
            return probabilities
        return probabilities.amax(dim=1, keepdim=True)
