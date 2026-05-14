from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SegAuxHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_size: int = 512,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.output_size = output_size
        mid = max(16, in_channels // 2)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(mid, out_channels, 1),
        )

    def forward_logits(
        self,
        feat: torch.Tensor,
        *,
        output_size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        x = self.head(feat)
        target_size = output_size if output_size is not None else self.output_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        return F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.forward_logits(feat)


class MultiTaskModel(nn.Module):
    """Backbone + thin auxiliary segmentation head.

    In training mode forward() returns (cls_logits, seg_logits).
    In eval mode forward() returns cls_logits only so that inference
    code (service.py, gradcam.py) works without modification.

    The seg head reads from an intermediate block via a forward hook,
    leaving the backbone's own forward() unmodified.
    """

    def __init__(
        self,
        backbone: nn.Module,
        block_index: int = 2,
        output_size: int = 512,
        seg_channels: int = 1,
        use_gated_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_gated_pooling = use_gated_pooling

        blocks = getattr(backbone, "blocks", getattr(backbone, "features", None))
        if blocks is None:
            raise ValueError("Backbone has neither .blocks nor .features")

        # Probe to get intermediate channel count
        _tmp: dict[str, torch.Tensor] = {}
        _probe_handle = blocks[block_index].register_forward_hook(
            lambda m, i, o: _tmp.update({"x": o})
        )
        was_training = backbone.training
        backbone.eval()
        probe_device = next(backbone.parameters()).device
        with torch.no_grad():
            backbone(torch.zeros(1, 3, output_size, output_size, device=probe_device))
        if was_training:
            backbone.train()
        _probe_handle.remove()

        in_channels = _tmp["x"].shape[1]

        # Permanent hook used during forward()
        self._feat: dict[str, torch.Tensor] = {}
        self._hook = blocks[block_index].register_forward_hook(
            lambda m, i, o: self._feat.update({"x": o})
        )

        self.seg_channels = int(seg_channels)
        self.seg_head = _SegAuxHead(
            in_channels,
            output_size,
            out_channels=self.seg_channels,
        )

    @staticmethod
    def _seg_logits_to_gate(seg_logits: torch.Tensor) -> torch.Tensor:
        if seg_logits.shape[1] == 1:
            return seg_logits
        return seg_logits.amax(dim=1, keepdim=True)

    def _forward_gated_classifier(self, x: torch.Tensor) -> torch.Tensor:
        if not all(
            hasattr(self.backbone, attr)
            for attr in ("forward_features", "forward_head", "classifier")
        ):
            raise ValueError("Gated pooling requires a timm-style backbone.")

        feat_map = self.backbone.forward_features(x)
        gate_logits = self.seg_head.forward_logits(
            self._feat["x"],
            output_size=feat_map.shape[-2:],
        )
        gate_logits = self._seg_logits_to_gate(gate_logits)
        gate = torch.sigmoid(gate_logits)
        gate = gate / gate.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled = self.backbone.forward_head(feat_map * gate, pre_logits=True)
        return self.backbone.classifier(pooled)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_gated_pooling:
            logits = self._forward_gated_classifier(x)
        else:
            logits = self.backbone(x)
        if self.training:
            return logits, self.seg_head(self._feat["x"])
        return logits

    def predict_seg(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid seg probability map [B,C,H,W] in eval mode.

        Runs backbone to populate the intermediate feature hook, then passes
        those features through the seg head. Dropout2d is disabled because the
        model must be in eval mode when called (enforced by caller).
        """
        with torch.no_grad():
            self.backbone(x)
            return torch.sigmoid(self.seg_head(self._feat["x"]))

    def predict_seg_union(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single-channel union probability map [B,1,H,W]."""
        probabilities = self.predict_seg(x)
        if probabilities.shape[1] == 1:
            return probabilities
        return probabilities.amax(dim=1, keepdim=True)

    # Delegate unknown attribute lookups to backbone so that external code
    # accessing model.classifier, model.blocks, etc. keeps working.
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["backbone"], name)
