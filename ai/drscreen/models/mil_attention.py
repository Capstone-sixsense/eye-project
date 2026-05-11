from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al., NeurIPS 2018).

    Each spatial position in the feature map is treated as an independent
    "instance". Attention weights are learned via a two-branch gating
    mechanism and used for weighted pooling. The resulting spatial weight
    map is directly interpretable as an XAI heatmap.

    a_k = softmax( w^T * (tanh(V h_k) ⊙ sigmoid(U h_k)) )
    z   = Σ_k a_k h_k
    """

    def __init__(self, in_channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.V = nn.Linear(in_channels, hidden, bias=False)
        self.U = nn.Linear(in_channels, hidden, bias=False)
        self.w = nn.Linear(hidden, 1, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] feature map from backbone.
        Returns:
            context:  [B, C] attention-pooled representation.
            attn_map: [B, H, W] normalised attention weights (sum to 1 over H*W).
        """
        B, C, H, W = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]

        gate = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # [B, N, hidden]
        logits = self.w(gate)  # [B, N, 1]
        attn = torch.softmax(logits, dim=1)  # [B, N, 1]

        context = (attn * h).sum(dim=1)  # [B, C]
        attn_map = attn.reshape(B, H, W)  # [B, H, W]
        return context, attn_map


class MILAttentionModel(nn.Module):
    """EfficientNet-B5 backbone + gated attention pooling head.

    Replaces global average pooling with gated MIL attention so that the
    attention weight map is directly available as an interpretable XAI
    heatmap without requiring gradient computation.

    forward() → logits [B, num_outputs]  (compatible with existing training loop)
    get_attention_map(x) → [B, H_in, W_in] attention map upsampled to input size.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_outputs: int = 1,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = backbone

        # Probe feature channels via forward_features
        was_training = backbone.training
        backbone.eval()
        device = next(backbone.parameters()).device
        with torch.no_grad():
            probe = torch.zeros(1, 3, 512, 512, device=device)
            feat = backbone.forward_features(probe)
        if was_training:
            backbone.train()

        in_channels = feat.shape[1]
        self.attn_pool = GatedAttentionPool(in_channels, hidden=hidden)
        self.classifier = nn.Linear(in_channels, num_outputs)

        # Cache for XAI retrieval (populated by forward())
        self._last_attn_map: torch.Tensor | None = None
        self._last_input_size: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._last_input_size = (x.shape[-2], x.shape[-1])
        features = self.backbone.forward_features(x)        # [B, C, H, W]
        context, attn_map = self.attn_pool(features)        # [B, C], [B, H, W]
        self._last_attn_map = attn_map.detach()
        return self.classifier(context)                     # [B, num_outputs]

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return attention map upsampled to input size.

        Returns:
            Tensor [B, H_in, W_in] with values in [0, 1] (min-max normalised).
        """
        with torch.no_grad():
            self.forward(x)
        attn = self._last_attn_map                         # [B, H_feat, W_feat]
        H, W = self._last_input_size
        upsampled = F.interpolate(
            attn.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)                                       # [B, H_in, W_in]
        mn = upsampled.amin(dim=(1, 2), keepdim=True)
        mx = upsampled.amax(dim=(1, 2), keepdim=True)
        return (upsampled - mn) / (mx - mn + 1e-8)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.__dict__["_modules"]["backbone"], name)
