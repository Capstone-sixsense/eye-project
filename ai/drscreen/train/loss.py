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


class CoralLoss(nn.Module):
    """Deep CORAL: Correlation Alignment (Sun & Saenko, 2016, arXiv:1607.01719).

    Minimises the squared Frobenius-norm difference between the unbiased sample
    covariance matrices of two domain feature sets.  Applied on pooled
    pre-classifier features so it acts directly on the representation space.

    L_CORAL = (1 / 4d²) * ||C_source − C_target||²_F

    Args:
        n < 2 per domain: returns zero loss (no covariance defined).
    """

    @staticmethod
    def _covariance(x: torch.Tensor) -> torch.Tensor:
        n, d = x.shape
        if n < 2:
            return x.new_zeros(d, d)
        xc = x - x.mean(0, keepdim=True)
        return (xc.T @ xc) / (n - 1)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = source.size(1)
        cs = self._covariance(source.float())
        ct = self._covariance(target.float())
        return torch.norm(cs - ct, p="fro") ** 2 / (4 * d * d)
