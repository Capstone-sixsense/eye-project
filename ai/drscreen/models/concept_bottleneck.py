from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConceptHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        concept_channels: int,
        *,
        hidden_channels: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        hidden = int(hidden_channels or max(64, in_channels // 2))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, concept_channels, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class ConceptBottleneckModel(nn.Module):
    """Spatial concept bottleneck classifier.

    The abnormal logit is computed only from lesion concept logits:

        feature map -> concept spatial logits -> GAP -> linear -> abnormal logit

    This makes the evidence map part of the forward classification path rather
    than a post-hoc attribution. Channel order is MA / HE / EX / SE.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        block_index: int = 4,
        concept_channels: int = 4,
        output_size: int = 512,
        head_hidden_channels: int | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if concept_channels < 1:
            raise ValueError("concept_channels must be >= 1")
        self.backbone = backbone
        self.block_index = int(block_index)
        self.concept_channels = int(concept_channels)
        self.output_size = int(output_size)

        blocks = getattr(backbone, "blocks", getattr(backbone, "features", None))
        if blocks is None:
            raise ValueError("Backbone has neither .blocks nor .features")

        probe: dict[str, torch.Tensor] = {}
        handle = blocks[self.block_index].register_forward_hook(
            lambda _m, _i, o: probe.__setitem__("x", o)
        )
        was_training = backbone.training
        backbone.eval()
        probe_device = next(backbone.parameters()).device
        with torch.no_grad():
            backbone(torch.zeros(1, 3, self.output_size, self.output_size, device=probe_device))
        if was_training:
            backbone.train()
        handle.remove()
        if "x" not in probe:
            raise RuntimeError(f"Failed to probe block{self.block_index} activation")

        self._feat: dict[str, torch.Tensor] = {}
        self._hook = blocks[self.block_index].register_forward_hook(
            lambda _m, _i, o: self._feat.__setitem__("x", o)
        )

        in_channels = int(probe["x"].shape[1])
        self.seg_head = _ConceptHead(
            in_channels,
            self.concept_channels,
            hidden_channels=head_hidden_channels,
            dropout=dropout,
        )
        self.classifier = nn.Linear(self.concept_channels, 1)
        self._latest_concept_logits: torch.Tensor | None = None
        self._latest_concept_map_logits: torch.Tensor | None = None

    def _forward_impl(
        self,
        x: torch.Tensor,
        *,
        output_size: int | tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.backbone.forward_features(x)
        feat = self._feat["x"]
        concept_map_logits = self.seg_head(feat)
        concept_logits = F.adaptive_avg_pool2d(concept_map_logits, 1).flatten(1)
        abnormal_logits = self.classifier(concept_logits)

        self._latest_concept_logits = concept_logits
        self._latest_concept_map_logits = concept_map_logits

        target_size = output_size if output_size is not None else self.output_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        concept_map_up = F.interpolate(
            concept_map_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return abnormal_logits, concept_map_up

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits, concept_map_logits = self._forward_impl(x)
        if self.training:
            return logits, concept_map_logits
        return logits

    def latest_concept_logits(self) -> torch.Tensor | None:
        return self._latest_concept_logits

    def predict_seg(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-concept probability maps [B,4,H,W]."""
        with torch.no_grad():
            _logits, concept_map_logits = self._forward_impl(x, output_size=x.shape[-2:])
            return torch.sigmoid(concept_map_logits)

    def predict_seg_union(self, x: torch.Tensor) -> torch.Tensor:
        probabilities = self.predict_seg(x)
        return probabilities.amax(dim=1, keepdim=True)

    def get_evidence_map(
        self,
        x: torch.Tensor,
        *,
        output_size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            _logits, concept_map_logits = self._forward_impl(
                x,
                output_size=output_size or x.shape[-2:],
            )
            return torch.sigmoid(concept_map_logits).amax(dim=1, keepdim=True)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["backbone"], name)
