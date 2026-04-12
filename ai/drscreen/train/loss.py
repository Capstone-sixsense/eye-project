from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017, arXiv:1708.02002).

    Down-weights easy, well-classified examples and concentrates training
    on hard, misclassified ones. Useful when the model is over-confident
    on in-domain examples while under-confident on shifted domains.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. gamma=0 reduces to weighted BCE.
               Typical range: 0.5 -- 5.0. Default: 2.0.
        alpha: Weight for the positive (abnormal) class, in [0, 1].
               The negative class receives weight (1 - alpha).
               None disables per-class weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None and not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * bce).mean()
