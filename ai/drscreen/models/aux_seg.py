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


class _UNetAuxHead(nn.Module):
    """Top-down decoder consuming multiple block features.

    Inputs are an ordered list of backbone block outputs (coarsest-last by
    convention: e.g. [block2, block3, block4]). Each block goes through a
    lateral 1x1 conv to unify channels, then features are merged top-down
    (coarse to fine) with bilinear upsampling and 3x3 smoothing.

    The intent of B1 in the XAI Phase 4 plan: propagate lesion mask
    supervision to higher-resolution scales so seg/CAM granularity matches
    GT region annotations, while leaving the classifier path untouched.
    """

    def __init__(
        self,
        in_channels_per_block: list[int],
        output_size: int = 512,
        out_channels: int = 1,
        decoder_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if not in_channels_per_block:
            raise ValueError("in_channels_per_block must be non-empty")
        self.output_size = output_size
        self.decoder_channels = decoder_channels
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, decoder_channels, 1) for c in in_channels_per_block]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
                    nn.BatchNorm2d(decoder_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels_per_block
            ]
        )
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(decoder_channels, out_channels, 1)

    def forward_logits(
        self,
        feats: list[torch.Tensor],
        *,
        output_size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(feats) != len(self.laterals):
            raise ValueError(
                f"expected {len(self.laterals)} feature maps, got {len(feats)}"
            )
        target_size = output_size if output_size is not None else self.output_size
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        laterals = [lat(f) for lat, f in zip(self.laterals, feats)]
        x = self.smooth[-1](laterals[-1])
        for i in range(len(laterals) - 2, -1, -1):
            x = F.interpolate(
                x, size=laterals[i].shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + laterals[i]
            x = self.smooth[i](x)
        x = self.dropout(x)
        x = self.head(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        return self.forward_logits(feats)


class MultiTaskModel(nn.Module):
    """Backbone + auxiliary segmentation head.

    Two decoder variants:
        - ``single_block``: thin 1x1 conv head over one intermediate block
          (original v24~v35 behavior).
        - ``unet``: top-down decoder over several blocks (B1 of Phase 4 XAI
          plan) for higher-resolution lesion supervision.

    In training mode forward() returns (cls_logits, seg_logits). In eval mode
    forward() returns cls_logits only so that inference code (service.py,
    gradcam.py) works without modification.

    The decoder reads from intermediate blocks via forward hooks, leaving the
    backbone's own forward() unmodified.
    """

    def __init__(
        self,
        backbone: nn.Module,
        block_index: int = 2,
        output_size: int = 512,
        seg_channels: int = 1,
        use_gated_pooling: bool = False,
        decoder_type: str = "single_block",
        decoder_blocks: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_gated_pooling = use_gated_pooling
        self.decoder_type = decoder_type

        blocks = getattr(backbone, "blocks", getattr(backbone, "features", None))
        if blocks is None:
            raise ValueError("Backbone has neither .blocks nor .features")

        if decoder_type not in {"single_block", "unet"}:
            raise ValueError(
                f"decoder_type must be 'single_block' or 'unet', got {decoder_type!r}"
            )

        if decoder_type == "unet":
            user_blocks = list(decoder_blocks) if decoder_blocks else [block_index]
            self._decoder_block_indices: list[int] = sorted(set(user_blocks))
        else:
            self._decoder_block_indices = [block_index]
        self._primary_block_index = int(block_index)
        if self._primary_block_index not in self._decoder_block_indices:
            # Always keep the primary block hooked so gated pooling / CAM align
            # use the same activation we annotate.
            self._decoder_block_indices = sorted(
                set(self._decoder_block_indices) | {self._primary_block_index}
            )

        # Probe channel counts
        _probe: dict[int, torch.Tensor] = {}
        probe_handles = []
        for idx in self._decoder_block_indices:
            probe_handles.append(
                blocks[idx].register_forward_hook(
                    lambda m, i, o, _k=idx: _probe.__setitem__(_k, o)
                )
            )
        was_training = backbone.training
        backbone.eval()
        probe_device = next(backbone.parameters()).device
        with torch.no_grad():
            backbone(torch.zeros(1, 3, output_size, output_size, device=probe_device))
        if was_training:
            backbone.train()
        for h in probe_handles:
            h.remove()

        # Permanent hooks. Each hook fires twice per forward only if its block
        # happens to be both primary and a decoder block (still cheap; same
        # tensor reference stored in both dicts).
        self._feat: dict[str, torch.Tensor] = {}
        self._decoder_feats: dict[int, torch.Tensor] = {}
        self._hooks: list = []
        for idx in self._decoder_block_indices:
            def _make_hook(k: int):
                def hook(m, i, o):
                    self._decoder_feats[k] = o
                    if k == self._primary_block_index:
                        self._feat["x"] = o
                return hook
            self._hooks.append(blocks[idx].register_forward_hook(_make_hook(idx)))

        in_channels_per_block = [_probe[i].shape[1] for i in self._decoder_block_indices]
        primary_in_channels = _probe[self._primary_block_index].shape[1]

        self.seg_channels = int(seg_channels)
        if decoder_type == "unet":
            self.seg_head: nn.Module = _UNetAuxHead(
                in_channels_per_block,
                output_size,
                out_channels=self.seg_channels,
            )
        else:
            self.seg_head = _SegAuxHead(
                primary_in_channels,
                output_size,
                out_channels=self.seg_channels,
            )
        if self.seg_channels > 1:
            self.lesion_weights = nn.Parameter(torch.zeros(self.seg_channels))
        else:
            self.register_parameter("lesion_weights", None)

    # ------------------------------------------------------------------
    # Seg head dispatch helpers (handle single-block vs unet input shapes)
    # ------------------------------------------------------------------
    def _seg_forward(
        self,
        *,
        output_size: int | tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if self.decoder_type == "unet":
            feats = [self._decoder_feats[i] for i in self._decoder_block_indices]
            if output_size is not None:
                return self.seg_head.forward_logits(feats, output_size=output_size)
            return self.seg_head(feats)
        if output_size is not None:
            return self.seg_head.forward_logits(
                self._feat["x"], output_size=output_size
            )
        return self.seg_head(self._feat["x"])

    @staticmethod
    def _seg_logits_to_gate(seg_logits: torch.Tensor) -> torch.Tensor:
        if seg_logits.shape[1] == 1:
            return seg_logits
        return seg_logits.amax(dim=1, keepdim=True)

    def _gated_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        if not all(
            hasattr(self.backbone, attr)
            for attr in ("forward_features", "forward_head", "classifier")
        ):
            raise ValueError("Gated pooling requires a timm-style backbone.")

        feat_map = self.backbone.forward_features(x)
        seg_logits = self._seg_forward(output_size=feat_map.shape[-2:])
        if self.seg_channels > 1 and self.lesion_weights is not None:
            weights = torch.softmax(self.lesion_weights, dim=0).view(1, -1, 1, 1)
            gate = (torch.sigmoid(seg_logits) * weights).sum(dim=1, keepdim=True)
        else:
            gate = torch.sigmoid(seg_logits)
        gate = gate / gate.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return self.backbone.forward_head(feat_map * gate, pre_logits=True)

    def classify_pooled_features(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.backbone.classifier(pooled)

    def forward_with_gated_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        pooled = self._gated_pooled_features(x)
        logits = self.classify_pooled_features(pooled)
        seg_logits = self._seg_forward() if self.training else None
        return logits, seg_logits, pooled

    def _forward_gated_classifier(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self._gated_pooled_features(x)
        return self.classify_pooled_features(pooled)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.use_gated_pooling:
            logits = self._forward_gated_classifier(x)
        else:
            logits = self.backbone(x)
        if self.training:
            return logits, self._seg_forward()
        return logits

    def predict_seg(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid seg probability map [B,C,H,W] in eval mode."""
        with torch.no_grad():
            self.backbone(x)
            return torch.sigmoid(self._seg_forward())

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
