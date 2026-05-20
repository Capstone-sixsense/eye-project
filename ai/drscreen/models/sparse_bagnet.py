from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PointwiseResidualBlock(nn.Module):
    """Residual 1x1 block that preserves the patch receptive field."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SparseBagNet(nn.Module):
    """Local-evidence binary classifier with an explicit patch-logit map.

    The first convolution consumes one local patch. Every following operation is
    1x1, so each spatial output logit is still constrained to that original
    patch. The image logit is the mean of all patch logits, making the evidence
    map part of the forward decision path rather than a post-hoc CAM.
    """

    def __init__(
        self,
        *,
        num_outputs: int = 1,
        in_channels: int = 3,
        patch_size: int = 33,
        patch_stride: int = 8,
        hidden_channels: int = 128,
        depth: int = 4,
        dropout: float = 0.15,
        aggregation: str = "mean",
    ) -> None:
        super().__init__()
        if num_outputs != 1:
            raise ValueError("SparseBagNet currently supports binary num_outputs=1 only.")
        if patch_size < 3 or patch_stride < 1:
            raise ValueError("patch_size must be >=3 and patch_stride must be >=1.")
        if hidden_channels < 8:
            raise ValueError("hidden_channels must be >=8.")
        aggregation = aggregation.strip().lower()
        if aggregation not in {"mean", "topk_mean"}:
            raise ValueError("aggregation must be 'mean' or 'topk_mean'.")

        self.num_outputs = int(num_outputs)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.hidden_channels = int(hidden_channels)
        self.depth = int(depth)
        self.aggregation = aggregation

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=patch_size,
                stride=patch_stride,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[_PointwiseResidualBlock(hidden_channels, dropout) for _ in range(depth)]
        )
        self.logit_head = nn.Conv2d(hidden_channels, num_outputs, kernel_size=1)
        self._last_patch_logits: torch.Tensor | None = None

    @property
    def classifier(self) -> nn.Module:
        return self.logit_head

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))

    def patch_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit_head(self.forward_features(x))

    def aggregate_patch_logits(self, patch_logits: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return patch_logits.mean(dim=(2, 3))

        # Keep this option available for later ablation, but default configs use
        # mean aggregation to match Sparse BagNet's inherent-evidence premise.
        flat = patch_logits.flatten(2)
        k = max(1, int(round(flat.shape[-1] * 0.10)))
        return flat.topk(k, dim=-1).values.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_logits = self.patch_logits(x)
        self._last_patch_logits = patch_logits
        return self.aggregate_patch_logits(patch_logits)

    def latest_patch_logits(self) -> torch.Tensor | None:
        return self._last_patch_logits

    def get_evidence_map(
        self,
        x: torch.Tensor,
        *,
        output_size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Return sigmoid patch-logit evidence upsampled to image size."""
        patch_logits = self.patch_logits(x)
        evidence = torch.sigmoid(patch_logits)
        if output_size is None:
            output_size = x.shape[-2:]
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        return F.interpolate(
            evidence,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
