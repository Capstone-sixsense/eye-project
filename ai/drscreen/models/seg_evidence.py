"""분류기와 독립적인 병변 근거(evidence) 분할기.

v8b 계열 세그멘터의 본체다. ResNet50 인코더 + U-Net 스타일 디코더(또는
DeepLabV3)로 안저 이미지에서 4채널(MA/HE/EX/SE) 병변 확률맵을 만든다.
'분류기와 독립적'인 점이 핵심: 배포된 v31 분류기를 건드리지 않고 별도 경로로
병변 후보 영역을 제시한다(인과적 근거가 아니라 별도 탐지 결과).

- _ConvBlock: (conv-BN-ReLU) x2 기본 블록.
- _UpBlock:   디코더 업샘플 + skip connection 결합 블록.
- LesionSegEvidence: 인코더(layer1~4)에서 받은 다중 스케일 특징을 디코더가
  점진적으로 복원해 입력 해상도 logit으로 되돌린다.
"""

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
        self.encoder_name = encoder
        self.out_channels = int(out_channels)
        self._is_deeplab = encoder == "deeplabv3_resnet50"

        if self._is_deeplab:
            backbone_weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.deeplab = models.segmentation.deeplabv3_resnet50(
                weights=None,
                weights_backbone=backbone_weights,
                num_classes=self.out_channels,
                aux_loss=False,
            )
            return

        if encoder != "resnet50":
            raise ValueError(
                "LesionSegEvidence supports encoder='resnet50' or 'deeplabv3_resnet50'"
            )

        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
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
        if self._is_deeplab:
            out = self.deeplab(x)
            logits = out["out"] if isinstance(out, dict) else out
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return logits

        # 인코더: 해상도를 단계적으로 줄이며 특징을 추출(주석의 1/N은 입력 대비 배율).
        input_size = x.shape[-2:]
        x0 = self.stem(x)       # 1/2
        x1 = self.layer1(self.pool(x0))  # 1/4
        x2 = self.layer2(x1)    # 1/8
        x3 = self.layer3(x2)    # 1/16
        x4 = self.layer4(x3)    # 1/32

        # 디코더: 가장 깊은 x4에서 시작해 각 단계의 인코더 특징(skip)을 결합하며 복원.
        y = self.center(x4)
        y = self.up3(y, x3)
        y = self.up2(y, x2)
        y = self.up1(y, x1)
        y = self.up0(y, x0)
        logits = self.head(y)
        # 디코더 출력(1/2 해상도)을 입력 원본 해상도로 되돌려 픽셀 단위 logit 반환.
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def predict_seg(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self(x))

    def predict_seg_union(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self.predict_seg(x)
        if probabilities.shape[1] == 1:
            return probabilities
        return probabilities.amax(dim=1, keepdim=True)
